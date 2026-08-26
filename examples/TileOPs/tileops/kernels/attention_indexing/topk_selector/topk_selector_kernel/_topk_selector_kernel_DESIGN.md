# _topk_selector_kernel 算子设计文档

> 迁移任务：源算子来自 TileOPs-fork（GPU/CUDA 风格 TileLang 实现），迁移到 TileLang-NPUIR（华为昇腾 NPU，target="npuir"，Developer 模式）。
> 源文件：`/home/tilelang/zuochuanuong/TileOPs-fork/tileops/kernels/topk_selector.py`（函数 `_topk_selector_kernel`，TopkSelectorKernel / TopkSelectorOp，family=attention_indexing）。
> NPU 侧提取件（Stage 0）：`examples/TileOPs/tileops/kernels/attention_indexing/topk_selector/_topk_selector_kernels.py`。

## 0. 源算子解读与迁移分析（迁移类任务必填）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论见 tilelang-op-design skill 的 references/migration-analysis.md）：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§10 的所有设计决策。

### 0.1 源算子语义（做什么）

**数学语义**：对每个 `(b, s, g)`（batch、seq 行、kv 组）三元组，在窗口 `W(b,s) = [starts[b,s], ends[b,s]) ∩ [0, seq_len_kv)` 内，从 `index_score[b, s, :, g]` 中选出 **topk 个最大值的 kv 位置索引**（绝对索引，∈ [0, seq_len_kv)），写入 `index[b, s, g, 0:topk]`。即 torch.topk 的窗口化变体：

$$
\text{index}[b,s,g,\cdot] = \operatorname{TopKIdx}\Big(\; \text{index\_score}[b,s,i,g] \;\Big|\; i \in [\,\text{starts}[b,s],\; \text{ends}[b,s]\,) \cap [0, \text{seq\_len\_kv}),\; k=\text{topk} \;\Big)
$$

**比较语义**：fp32 精确值比较（源实现最终按 fp32 全部位（sign/exponent/mantissa）逐字节细化，等价于 fp32 全序比较；第一阶段的 fp16 截断分箱是单调的，只用于快速剪枝、不影响最终序）。

**规约/选择语义**：非数值规约，是**选择规约**；并列值（tie）时在并列边界组内**任意**取 k 个（源实现由原子操作到达顺序决定，torch.topk 取最低索引；精度用例 `_set_compare` 用集合比较，两者均兼容）。

**输出顺序语义**：源实现的输出位置由「值降序 + 同 bin 原子到达序」构成，非严格保证的降序；测试按**集合**比较（顺序无关）。本设计输出为严格值降序（vsort 结果），是源语义的合法实例化。

**dtype 语义**：输入 `index_score` 为 fp32（spec 全部 workload 为 float32）；输出索引 int32；`starts`/`ends` int32（源以 out_dtype=int32 声明）。无中间精度提升/截断（radix 位序即 fp32 位序）。

**边界语义**：
- 元素同时受 `starts ≤ i < ends` 与 `i < seq_len_kv` 双重约束（源码 L93-94 / L124 的三条件判断）。
- 窗口有效元素数 ≥ topk 时输出恰好 topk 个索引（manifest shape_rules 保证 `0 < topk ≤ seq_len_kv`，测试 harness 恒用全窗口 `starts=0, ends=seq_len_kv`）。
- 窗口有效元素数 < topk：**源实现行为未定义**（suffix 计数全部 ≤ topk 时阈值 bin 搜索失败，读未初始化 SMEM）。本设计显式定义该行为（见 §0.6 R1）。
- NaN/Inf：源的 radix 位型映射将 NaN 归入大值区间（语义未定义契约）；测试输入为 randn（无 NaN）。本设计同样将 NaN 视为契约外（vsort 的 NaN 序为硬件定义）。
- 源实现存在 **SMEM_INPUT_SIZE=4096 候选截断隐患**：窗口内与阈值 bin 同 bin 的候选超过 4096 时（如大 kv + 值高度集中），任意丢弃候选可能导致漏选真 top-k。测试的展平集合比较（见 §8.3）在大 workload 上几乎恒真，未暴露此隐患。本设计消除该隐患（候选恒 ≤ topk + K2，见 §0.6 R1）。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `index_score` | 输入 | `(batch, seq_len, seq_len_kv, kv_group)` | float32 | 评分张量；动态轴：batch、seq_len_kv（源声明 `T.dynamic`） |
| `starts` | 输入 | `(batch, seq_len)` | int32 | 每行窗口起点（含） |
| `ends` | 输入 | `(batch, seq_len)` | int32 | 每行窗口终点（不含） |
| `index` | **输出** | `(batch, seq_len, kv_group, topk)` | int32 | 选中 kv 位置的**绝对索引**；注意输出把 kv_group 维提前、topk 作尾维（与输入 (S_kv, G) 尾两维顺序相反），golden 需 `permute(0,1,3,2)` 对齐 |
| `topk` | 参数 | 标量 | int | 编译期常量，`0 < topk ≤ seq_len_kv` |

输出 shape（迁移必需字段）：`(batch, seq_len, kv_group, topk)`，int32，**非转置布局、直接以该顺序连续写入**。

### 0.3 实现算法解读（怎么算）

源码结构化阅读（按 skill 阅读协议：入口签名 → Kernel 结构 → 原语 → 内存 → 循环 → host → 编译配置）：

- **入口签名**：工厂 `_topk_selector_kernel(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype)`（`lru_cache` 按全部 shape 参数缓存）→ 返回 `topk_selector_fwd_func(RADIX=256, BLOCK_SIZE=1024, SMEM_INPUT_SIZE=4096, block_m=32)` → 返回 prim_func `(index_score, index, starts, ends)`，`out_idx=[1]`。
- **Kernel 结构**：`T.Kernel(batch, seq_len, kv_group, threads=BLOCK_SIZE=1024)` — **三维 grid**，每 block 负责一行 `(bx, by, g)`，block 内 1024 线程协作。
- **核心数据结构**（全部 SMEM）：`s_histogram[257]`（int32 直方图，RADIX=256 bins + 1 哨兵）、`s_threshold_bin_id[1]`、`s_num_input[2]`（候选计数，双缓冲）、`s_input_idx[2][4096]`（候选索引双缓冲）；线程局部变量（寄存器）若干。

**计算步骤分解**（源码每个计算语句均归入某步骤，行号对照 NPU 侧提取件）：

| 步骤 | 源码行 | 计算 | 输入 | 输出 | 语义对应 |
|------|--------|------|------|------|----------|
| 1. 行参数装载 | L78-80 | `l_start_idx=starts[bx,by]`, `l_end_idx=ends[bx,by]`, `l_new_topk=topk` | GM starts/ends | 寄存器标量 | 窗口边界 |
| 2. 直方图清零 | L86-89 | `s_histogram[j]=0 (j≤256)`, `s_num_input[0]=0` + `sync_threads` | — | SMEM | 初始化 |
| 3. Stage-1 直方图 | L91-97 | 1024 线程跨窗口 stride 扫描（每线程处理 `s*1024+tx`）；`convert_to_uint16(x)`（fp32→fp16 位型→单调无符号 16bit→取高 8bit）得 bin；`atomic_add(s_histogram[bin],1)` | GM index_score（窗口内） | SMEM 直方图 | 粗粒度（fp16 级）值分布统计 |
| 4. 并行后缀和 | L100-109 | Hillis-Steele 倍增扫描（8 轮，offset=1..128），仅前 RADIX 线程参与，`sync_threads(3, RADIX)` 部分屏障；结果 `s_histogram[tx] = count(bin ≥ tx)` | SMEM 直方图 | SMEM（原地） | 高位到低位的累计计数 |
| 5. 阈值 bin 搜索 | L111-115 | 找 tx 使 `count(bin≥tx) > topk ≥ count(bin≥tx+1)` → `s_threshold_bin_id` | SMEM 后缀和 | SMEM 标量 | topk 阈值分界 bin |
| 6. 剩余配额计算 | L116-117 | `l_new_topk = topk − count(bin > threshold)` | SMEM | 寄存器 | 需从阈值 bin 内补选的数量 |
| 7. Stage-1 scatter | L122-140 | 再扫一遍窗口：bin > 阈值 → `pos = atomic_add(s_histogram[bin+1],1,return_prev)`，`pos < topk` 时直接写 `index[bx,by,g,pos]=input_idx`（位置 = 更大 bin 元素数 + 同 bin 内到达序）；bin == 阈值且配额 > 0 → append 到 `s_input_idx[0]`（≤4096 截断） | GM index_score | GM index / SMEM 候选 | 确定选中超阈值元素、收集阈值 bin 候选 |
| 8. Stage-2 尾部细化（≤4 轮） | L142-214 | `for round in 0..3`（fp32 单调位型的第 `24−8r` 字节）：`l_new_topk≤0` 时 `loop_break` 早退；候选（上轮 `s_input_idx[r_idx]`，双缓冲 `r_idx=round%2`）重建直方图（L150-165）→ 同款后缀和+阈值（L167-185）→ scatter（L187-214）：bin > 阈值 → 以 `l_start_pos=topk−l_new_topk` 为基的原子定位直接写输出；bin == 阈值：round<3 → 转入下一候选缓冲（≤4096 截断），round==3 → 直接写输出（`pos≥topk` 丢弃） | SMEM 候选 + GM index_score（按候选索引重读） | GM index / SMEM 候选 | 候选集内按 fp32 位序逐字节精化 |
| 9. 输出语义 | L133/198/206 | 写入的是**绝对 kv 索引** `input_idx`（stage-2 为 `s_input_idx` 保存的绝对索引） | — | GM index | 语义契约 |

**数据流与内存访问模式**（源硬件视角，覆盖全部 buffer）：

```
GM[index_score] --线程分散读(stride=kv_group 的行内 kv 扫描)--> reg(convert_to_uint16/32)
  --atomic_add--> SMEM[s_histogram] --Hillis-Steele--> SMEM[s_histogram](后缀和)
  --atomic scatter--> GM[index]（超阈值元素直接输出）
                  \-> SMEM[s_input_idx 2×4096 双缓冲]（阈值 bin 候选，按候选绝对索引回读 GM[index_score] 再细化）
GM[starts/ends] --标量读--> reg
```

**循环与并行结构**：三维 grid（batch × seq_len × kv_group）× 1024 线程；行内两个动态边界串行循环 `T.ceildiv(seq_len_kv, BLOCK_SIZE)`（直方图遍历、scatter 遍历）+ 4 轮尾部循环（候选数动态边界）；部分屏障 `sync_threads(3, RADIX)` 用于扫描同步。

**host 侧逻辑**：`lru_cache(32)` 按完整 shape 元组缓存 JIT 工厂；`torch.library.custom_op` 包装（mutates_args=()，fake 返回 `torch.empty([batch, seq_len, kv_group, topk], int32)`）；`Kernel` 类默认 config `{RADIX:256, BLOCK_SIZE:1024, SMEM_INPUT_SIZE:4096, block_m:32}`（block_m 未被 kernel 使用，属遗留参数）；autotune supply_prog 将 starts 置 0、ends 置 seq_len_kv。NPU 侧 op 封装（`examples/TileOPs/tileops/ops/attention_indexing/topk_selector.py` 与 `.../kernels/attention_indexing/topk_selector/topk_selector.py`）已完成适配（npub:: 命名、去 autotune、supported_archs=None），**接口契约由该封装固定**（见 §3.5.2）。

**编译配置**：`pass_configs={TL_DISABLE_THREAD_STORAGE_SYNC: True}`（GPU codegen 关闭线程存储自动同步，因 kernel 手工管理 sync_threads）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | radix-select 两阶段算法（8bit 粗筛 + 4 轮字节细化） | 以 ~2–5 遍数据扫描代替 O(N·logN) 全排序 / O(k·N) 迭代 argmax | 直方图阈值定位 + 候选集逐轮收缩 | 无（纯算法层，实现见 #4–#6） | 算法思想可移植，实现强相关 |
| 2 | fp16 截断粗分箱（`convert_to_uint16` 取高 8bit） | 第一轮用低精度位型快速缩小候选集 | fp32→fp16 单调位型截断 | 无（纯算法层） | 可移植 |
| 3 | 浮点→无符号整数单调位技巧（`convert_to_uint16/32`：负数取反、正数置符号位） | 使 radix 整数比较等价于浮点比较 | IEEE 位型的单调重映射 | 位运算 + reinterpret | 算法层，NPU 有等价指令 |
| 4 | 1024 线程 SMEM 原子直方图 | 单遍并行统计值分布 | `atomic_add(s_histogram[bin],1)` | CUDA SMEM 原子操作 + 大线程块 | **硬件强相关** |
| 5 | Hillis-Steele 跨线程并行扫描（部分屏障 `sync_threads(3, RADIX)`） | O(log R) 并行后缀和 | 256 lane 倍增扫描 | warp/block 屏障原语 | **硬件强相关** |
| 6 | 原子 scatter 定位 + 候选紧凑化（`atomic_add(..., return_prev=True)`） | 免前缀和的排名/紧凑写入；超阈值元素一遍定位输出 | SMEM 原子取旧值作写位置 | CUDA SMEM 原子 | **硬件强相关** |
| 7 | 三维 grid + 每 block 1024 线程（一行一 block） | 行间完全并行 + 行内线程协作 | `T.Kernel(batch, seq_len, kv_group, threads=1024)` | CUDA block×thread 两级并行模型 | **硬件强相关** |
| 8 | SMEM 候选双缓冲（`s_input_idx[2][4096]` ping-pong） | 尾部细化免拷贝轮转；候选驻留片上 | 2×16KB SMEM | SMEM 容量（48KB+） | **硬件强相关**（且有 4096 截断正确性隐患） |
| 9 | coalesced 窗口扫描（线程沿 kv 连续 stride） | GM 合并访存带宽 | 相邻线程读相邻地址 | CUDA memory coalescing | 硬件强相关（NPU 用整段 copy 替代） |
| 10 | 早退（`l_new_topk ≤ 0` 时 `T.loop_break`） | topk 已集满时跳过剩余细化轮 | 动态循环中断 | GPU codegen loop_break | 算法层，模式可移植 |
| 11 | `pass_configs TL_DISABLE_THREAD_STORAGE_SYNC` | 关闭线程存储自动同步插桩（手工同步已完备） | 编译 pass 配置 | GPU TileLang codegen | **编译器强相关** |

> 识别不出来 ≠ 不存在。未列出的优化会在迁移中被静默丢弃，导致性能回退无法追溯。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？（判定依据：migration-analysis.md §5.3 映射表条目 / examples/ 佐证 / docs/ 条目）

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 计算语义：窗口内 topk 绝对索引选择（int32、集合比较、tie 任意） | 语义 | 无 | — | **保留** | 语义层无条件保留（方法论 §5.4 规则 1） |
| 窗口掩码思想（starts/ends 裁剪） | 算法 | 无 | 有 | **保留**（等价实现：越窗元素置 −inf 掩码） | `examples/indexer/indexer_fwd.py` 用 `-T.infinity` 掩码 ks/ke 窗口，同构模式已验证 |
| radix-select 两阶段字节细化算法（直方图→阈值→细化） | 算法（实现依赖线程原子） | SMEM 原子 + 线程协作 | 无高效等价（Vector 核无 warp 级 SMEM 原子直方图形态；逐 bin 计数需 256 遍扫描） | **重新设计 → R1** | 用硬件原生 `T.vsort` 分块排序 + 流式归并替代；依据 docs/Tilelang.language/排序操作/T.vsort.md（fp32 尾轴排序 + int32 索引输出）+ testing/npuir/sort_ops/test_vsort_dev.py 实测；方法论 §5.3「warp shuffle/跨线程通信类 → 重新设计」同族判定 |
| 浮点→无符号单调位技巧（convert_to_uint16/32） | 算法 | 位运算/reinterpret | 有（T.vbitcast/vand/vor/vnot/vshr，docs 有条目） | **舍弃** | R1 改用 vsort 原生 fp32 比较，位技巧无存在必要；舍弃理由：重设计后冗余（若未来走 radix 路线可用 vbitcast+vnot/vor+vshr 等价实现，特此记录） |
| 三维 grid + 1024 线程 block（一行一 block） | 优化 | CUDA block×thread 模型 | 无（本项目 `T.Kernel` 仅一维 block 数，无线程绑定） | **重新设计 → R2** | 行集 R=batch×seq_len×kv_group 静态展开；一维 persistent Kernel + 核内静态串行；依据 ascend-constraints.md「不支持三维 Kernel」+ 开发指南.md §3.3；`examples/elementwise/vec_add_2d_dynamic_shape.py` 静态核数+守卫范式 |
| SMEM 原子直方图 + 原子 scatter 定位/紧凑化 | 优化 | CUDA SMEM 原子 | 无直接等价 | **重新设计 → R1**（并入） | vsort 全排序天然给出全序排名，输出位置 = 排序位置 0..topk−1，无需原子紧凑化 |
| Hillis-Steele 部分屏障并行扫描 | 优化 | warp/block 屏障原语 | 有近似（T.vcumsum）但 R1 不需要 | **重新设计 → R1**（并入） | R1 中阈值/排名由 vsort 直接给出，扫描步骤整体消失 |
| SMEM 候选双缓冲（2×4096 截断） | 优化 | SMEM 容量 | 有（UB 192KB） | **重新设计 → R1**（并入） | UB 流式归并缓冲（topk + K2），候选恒 ≤ topk+K2 **无截断**，同时修复源 4096 截断正确性隐患 |
| coalesced 窗口扫描 | 优化 | memory coalescing | 有（整段连续搬运） | **等价替换** | `T.copy` 连续段 GM→UB（尾轴 32B 对齐）；依据 docs/Tilelang.language/内存操作/T.copy.md + examples/vectorize/vectorization_in_parallel.py |
| 早退 loop_break | 算法 | 无 | 可实现 | **舍弃**（v1 不迁移） | 流式归并下逐 chunk 早退需额外 chunk 级窗口重叠判断（本设计已有 chunk 级跳过守卫承接其主要收益）；不影响正确性；列为 §9 后续调优点 |
| pass_configs TL_DISABLE_THREAD_STORAGE_SYNC | 编译配置 | GPU TileLang codegen | 无此 pass 语义 | **舍弃** | NPU Developer 模式自动同步（§7），无线程存储同步概念 |
| host lru_cache / custom_op / op 封装 | host | 无 | 有 | **保留** | NPU 侧 op 封装已存在（examples/TileOPs/tileops/.../topk_selector.py），接口不变 |

**判定统计**：保留 3 项、等价替换 1 项、重新设计 5 项（归并为 2 个重设计项 R1/R2）、舍弃 3 项（均写明理由）。

### 0.6 NPU 算法重设计

**重设计项 R1：radix-select 选 topk → 分块 vsort + 流式归并 topk**

- **源方案**：8bit 直方图快速定位阈值 bin + 最多 4 轮 fp32 位型逐字节细化，SMEM 原子直方图/scatter/扫描实现，约 2–5 遍数据访问；性能意图是避免 O(N·logN) 全排序与 O(k·N) 迭代 argmax。
- **NPU 新算法**（每行 `(b,s,g)` 独立执行，行内流程）：
  1. **初始化**：运行候补缓冲 `mergeA_val[0:topk] = −inf`（`T.vbrc(-T.infinity("float32"), ...)`）、`mergeA_idx[0:topk] = 0`（元素级并行置零）。
  2. **chunk 循环**（`for c in T.serial(NUM_CHUNKS)`，静态边界）：对 kv 轴按静态块长 C 切块，`c0 = c*C`；
     - **chunk 级跳过守卫**（运行时）：`c0 < ends && c0+C > starts && c0 < seq_len_kv` 不成立则跳过本 chunk（承接源早退的主要收益：窄窗口行自动少做功，缓解窗口长度差异导致的负载不均）；
     - **搬入**：`T.copy(index_score[b, s, c0 : c0+n_valid, g], chunk_src_ub[0:n_valid])`（`n_valid = min(C, seq_len_kv − c0)`；G=1 时行内 kv 段连续；G>1 回退逐元素读，见 §5.4）；
     - **窗口掩码**：`for i in T.Parallel(C)`：`abs = c0+i`，若 `abs < starts || abs ≥ ends || abs ≥ seq_len_kv` → `chunk_src_ub[i] = −inf`（未拷贝的尾部垃圾同时被此掩码覆盖为 −inf）；
     - **块内排序**：`T.vsort(chunk_src_ub, chunk_val_ub, chunk_idx_ub, descending=True, sort_axis=-1)`（降序；−inf 掩码项自然沉底）；
     - **绝对索引化**：`for i in T.Parallel(K2)`：`chunk_idx_ub[i] += c0`（K2 = min(C, topk)，静态；int32 标量加法，vadd 不支持 int32 故用元素级循环）；
     - **合并**：拼装 `mergeA = [候补(topk) | chunk 头部(K2)]`（UB→UB `T.copy` 两段），`T.vsort(mergeA, mergeB_val, mergeB_idx, descending=True)`，取 `mergeB[0:topk]` 回填 `mergeA[0:topk]`。
  3. **输出**：`T.copy(mergeA_idx[0:topk], index[b, s, g, 0:topk])`（连续段，值降序）。
- **语义保持论证**：
  1. **数学等价**：流式 topk 恒等式 `TopK(A∪B) = TopK(TopK(A) ∪ TopK(B))` 对值集合严格成立（归纳于 chunk 数：每轮 `mergeA[0:topk]` 恰为已扫描前缀的精确 topk 值集合与一组合法索引）；−inf 掩码项在降序序中恒排于任何有限值之后，当窗口有效元素数 ≥ topk 时（manifest 全部 workload 满足），输出 = 窗口内 fp32 精确 topk。fp32 位序比较（源）与 fp32 值比较（vsort）在非 NaN 域上为同一全序。并列值：集合比较语义下并列边界组内任取 k 个均合法（源为原子序任意取、torch.topk 取最低索引、测试集合比较，三者兼容；vsort 对重复值的索引顺序与 torch.sort 可能不同，docs 已明示，集合语义覆盖）。
  2. **边界语义核对**：窗口 < topk（源未定义）→ 本设计定义：前 W 槽为窗口元素索引、余槽为 −inf 掩码项的填充索引（初始化为 0，因 `mergeA_idx` 置零且掩码项索引为其绝对位置）；窗口为空（starts ≥ ends 或越界）→ 输出全 0。NaN：vsort 硬件序（契约外，与源一致地未定义）。`topk ≤ seq_len_kv` 且全窗口测试下有效元素 = seq_len_kv ≥ topk，恒走精确路径。
  3. **正确性增强**：候选规模恒 ≤ topk + K2 ≤ 2·topk（fp32 值 8KB×2 级别），**消除源 SMEM_INPUT_SIZE=4096 截断导致的漏选隐患**。
  4. **验证方式**：§8.1 golden 独立实现（掩码 + torch.topk），L0 逐行集合比较。
- **性能意图承接**：源 radix 的意图是"少读数据"（2–5 遍）；R1 同样单遍读 GM（每元素恰读一次），排序/归并在 UB 内完成；GM 流量与源同量级（源 stage-2 候选重读为小量），比较次数多于 radix（约 O(C·logC) 块内 + O(2·topk·log) 归并）但由 Vector 硬件排序指令承担——此为性能风险项，见 §9.1，Stage 4 用 msprof 数据验证并可回退调 C。

**重设计项 R2：三维 grid × 1024 线程 block → 一维 persistent Kernel + 核内静态串行**

- **源方案**：`T.Kernel(batch, seq_len, kv_group, threads=1024)`，一行一 block，block 内 1024 线程 warp 协作（原子/扫描）；性能意图是行间大规模并行 + 行内线程级并行。
- **NPU 新算法**：行集 `R = batch × seq_len × kv_group`（编译期常量，见 §4.6 特化决策）；`T.Kernel(NUM_KERNELS, is_npu=True)` 一维固定核数（=40，依据见 §5.5）；核内 `for t in T.serial(MAX_TASKS)`（`MAX_TASKS = ceildiv(R, NUM_KERNELS)`，**静态边界**）+ 运行时守卫 `row = t*NUM_KERNELS + kernel_id; if row < R`；由 row 解码 `(b, s, g)`；行间完全独立、无核间通信。行 stride 交错分配（而非连续段分配）使窗口长度差异在核间统计均衡。
- **语义保持论证**：行级任务划分不改变任何一行的计算（行独立、无跨行数值组合，不存在累加顺序问题）；输出地址由 (b,s,g) 唯一解码确定；并行度变化仅影响调度不影响结果。守卫保证 R 非 NUM_KERNELS 整数倍时无越界访问。
- **性能意图承接**：源"每行一个 block 充分并行"的意图由"每核串行处理多行 + 40 核满波"承接，同时消除超发逻辑核的串行调度开销与启动风暴（开发指南.md §3.3）。

**其余条目**：§0.5 中无其他「重新设计」条目；保留/等价替换/舍弃条目已在 §0.5 表中给出方案与依据。

### 0.7 标杆实现

- 源算子路径（golden 依据）：`/home/tilelang/zuochuanuong/TileOPs-fork/tileops/kernels/topk_selector.py`；NPU 侧提取件：`examples/TileOPs/tileops/kernels/attention_indexing/topk_selector/_topk_selector_kernels.py`。
- 源仓参考实现：manifest `ref_api: torch.topk`；测试基准 `examples/TileOPs/tests/ops/test_topk_selector.py` 的 `ref_program`（`torch.topk(index_score, topk, dim=2)[1].permute(0,1,3,2)`，全窗口语义）与 `_set_compare`（集合比较）。
- golden 以 §0.1 语义为依据，优先移植源仓参考实现并扩展窗口掩码语义（见 §8.1），不复刻 NPU 算法。

---

## 1. 概述

### 1.1 算子名称

`_topk_selector_kernel`（属 TopkSelectorOp，family=attention_indexing）

### 1.2 功能描述

对 `[batch, seq_len, seq_len_kv, kv_group]` 评分张量，在每行 `[starts, ends)` 窗口内沿 seq_len_kv 维选出 topk 个最大值的绝对索引，输出 `[batch, seq_len, kv_group, topk]` int32（值降序，集合比较语义）。

### 1.3 数学公式

$$
\text{index}[b,s,g,0:\text{topk}] = \operatorname{TopKIdx}_{k=\text{topk}}\Big(\; \text{index\_score}[b,s,i,g] \;\Big|\; i \in [\,\text{starts}[b,s],\; \text{ends}[b,s]\,) \cap [0, \text{seq\_len\_kv}) \;\Big)
$$

### 1.4 算法描述（迁移决策后的 NPU 侧算法，基于 §0.5/§0.6）

**分块 vsort + 流式归并 topk**（重设计项 R1）：

1. 行集 `R = batch × seq_len × kv_group`，40 个一维内核分摊（重设计项 R2），每核串行处理 `ceildiv(R,40)` 行（静态边界 + 守卫）。
2. 每行：候补缓冲初始化（−inf / 0）→ 按 C 分块循环：{chunk 越窗跳过 → GM 拷入 UB → 窗口 −inf 掩码 → `T.vsort` 降序 → 前 K2 项绝对索引化 → 与候补拼接 2·topk 再 vsort → 回填 topk}。
3. 行末：候补索引段 `T.copy` 写 GM 输出。

**与源算法的差异**（逐条注明来源）：
- radix 直方图/阈值/细化 → vsort 排序归并（§0.6 R1）；
- 三维 grid × 1024 线程 → 一维 40 核 persistent + 核内静态串行（§0.6 R2）；
- SMEM 原子/scatter/扫描 → 全部消失（排序天然给出排名）（§0.6 R1）；
- 候选 4096 截断 → 无截断（正确性增强）（§0.6 R1）；
- coalesced 扫描 → 整段 `T.copy`（§0.5 等价替换）；
- 输出顺序：源半任意序 → 严格值降序（语义合法实例化，torch.topk 同序）。

### 1.5 数据流图

```
GM[index_score] --T.copy(行内 kv 连续段)--> UB[chunk_src_ub]
  --T.Parallel(-inf 窗口掩码)--> UB[chunk_src_ub]
  --T.vsort(descending)--> UB[chunk_val_ub | chunk_idx_ub]
  --T.Parallel(+c0 绝对索引化)--> UB[chunk_idx_ub]
  --T.copy(UB→UB 拼装)--> UB[mergeA_val | mergeA_idx]  (=[候补 topk | chunk 头 K2])
  --T.vsort(descending)--> UB[mergeB_val | mergeB_idx]
  --T.copy(UB→UB 回填)--> UB[mergeA[0:topk]]            (运行候补)
  [chunk 循环结束]
  --T.copy--> GM[index[b,s,g,0:topk]]
GM[starts/ends] --标量读--> 寄存器（每行 2 次标量 GM 读，同 examples/indexer 的 CuSeqLenKS 读法）
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer（用户指定 developer，与算子特征一致）

### 2.2 选型理由

- 纯 Vector 算子（排序/掩码/搬运，无 matmul、无 Cube/L1/L0 需求），不需要 Expert 的手动内存层级与流水编排。
- 全部计算可由 v 前缀高层 API 表达（T.vsort / T.vbrc / T.copy / T.Parallel 元素级），无手动同步点需求（行间独立、核内串行、无核间通信）。
- Developer 模式下 `T.alloc_shared` 自动映射 UB（docs/Tilelang.language/排序操作/T.vsort.md Developer 示例同款用法），编译器自动插同步，降低 Stage 2 实现风险。
- 参照同类实现：`examples/indexer/indexer_fwd.py`（Developer 模式、`os.environ["TILELANG_ASCEND_MODE"]="Developer"`）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（Developer 模式自动映射 UB；7 个 UB buffer 见 §4.3） |
| 计算方式 | `T.vsort`（块内排序/归并排序）+ `T.vbrc`（−inf 填充）+ `T.Parallel` 元素级（掩码、索引换算、置零）+ `T.copy`（搬运） |
| 同步 | 编译器自动同步；无手动 sync_block_set/pipe_barrier（§7） |
| 环境变量 | `TILELANG_ASCEND_MODE=Developer` |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `valid(i) = starts ≤ i < ends ∧ i < seq_len_kv` | 窗口有效判定 |
| 2 | `masked(i) = valid(i) ? x[i] : −inf` | 越窗/越界元素掩码 |
| 3 | `(vals↓, idxs) = sort_desc(masked)` | 块内降序排序 + 索引跟踪 |
| 4 | `cand⁽ᵗ⁾ = TopK(cand⁽ᵗ⁻¹⁾ ∪ head_K2(vals↓, idxs+c0))` | 流式归并 topk |
| 5 | `index[b,s,g,·] = idx(cand⁽ᵀ⁾)` | 输出绝对索引 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 搬入 | `chunk_src ← index_score[b,s,c0:c0+n,g]` | `T.copy(src_slice, dst_slice)` | GM 行内连续段 → UB，运行时长度切片（`n_valid = min(C, seq_len_kv−c0)`） | Developer |
| 掩码 | 越窗元素置 −inf | `for i in T.Parallel(C): buf[i] = T.if_then_else(cond, −inf, buf[i])`（元素级，运行时条件） | cond 为 `(c0+i) < starts ∨ (c0+i) ≥ ends ∨ (c0+i) ≥ seq_len_kv` | Developer |
| −inf 填充 | 候补值初始化 | `T.vbrc(-T.infinity("float32"), mergeA_val_ub)` | 标量 → fp32 buffer 广播 | Developer |
| 索引置零 | 候补索引初始化 | `for i in T.Parallel(topk): mergeA_idx_ub[i] = 0` | int32 元素级（T.clear 仅支持 fp16/fp32，故用元素级循环） | Developer |
| 块内排序 | 降序排序 + 索引 | `T.vsort(src, dst_value, dst_index, descending=True, sort_axis=-1)` | src=chunk_src_ub(C)；dst_value fp32 同 shape；dst_index int32 同 shape | Developer |
| 绝对索引化 | `idx[i] += c0` | `for i in T.Parallel(K2): chunk_idx_ub[i] = chunk_idx_ub[i] + c0` | int32 标量加（T.vadd 仅支持 fp16/fp32，int32 用元素级循环） | Developer |
| 拼装 | 候补 ∪ chunk 头 | `T.copy(mergeA_val[0:topk] ← …)` 等 4 段 UB→UB copy | [0:topk]=候补、[topk:topk+K2]=chunk 头部 | Developer |
| 归并排序 | 再排序取 topk | `T.vsort(mergeA, mergeB_val, mergeB_idx, descending=True, sort_axis=-1)` | shape=(topk+K2,) | Developer |
| 回填 | `mergeA[0:topk] ← mergeB[0:topk]` | `T.copy(mergeB_val[0:topk], mergeA_val[0:topk])` 等 2 段 | UB→UB | Developer |
| 输出 | `index[b,s,g,0:topk] ← mergeA_idx[0:topk]` | `T.copy(dst_slice, src_slice)` | UB → GM 连续段（topk 静态长度） | Developer |
| 窗口参数 | `starts[b,s]` / `ends[b,s]` | 标量 GM 读（同 indexer 的 `CuSeqLenKS[s_idx]` 用法） | int32 标量 | Developer |

### 3.3 计算伪代码

接口契约（对外不变，配置参数属后端实现相关、允许适配）：`_topk_selector_kernel(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype)` 返回可两段调用对象 `f(...)(index_score, starts, ends)`（配置位由 GPU 的 `RADIX/BLOCK_SIZE/SMEM_INPUT_SIZE/block_m` 适配为 NPU 的 `(chunk_size, num_kernels)`，缺省自动推导）。`T.Tensor` 的 dtype 以第二位置参数传入（非 `dtype=` 关键字）。

```python
R = batch * seq_len * kv_group                 # 静态常量
NUM_KERNELS = 40                               # 分核常量，见 §5.5
MAX_TASKS = ceildiv(R, NUM_KERNELS)            # 静态
C = align_up(min(seq_len_kv, C_MAX=8192), 128) # 静态 chunk 长度
K2 = min(C, topk)                              # 静态 chunk 贡献上限
NUM_CHUNKS = ceildiv(seq_len_kv, C)            # 静态

@T.prim_func
def _topk_selector_kernel_main(
    index_score: T.Tensor((batch, seq_len, seq_len_kv, kv_group), in_dtype),
    index:       T.Tensor((batch, seq_len, kv_group, topk), out_dtype),
    starts:      T.Tensor((batch, seq_len), "int32"),
    ends:        T.Tensor((batch, seq_len), "int32"),
):
    with T.Kernel(NUM_KERNELS, is_npu=True) as (kid, _):
        # UB buffer（Developer 模式 alloc_shared 自动映射 UB）
        chunk_src = T.alloc_shared((C,), in_dtype)          # fp32
        chunk_val = T.alloc_shared((C,), in_dtype)
        chunk_idx = T.alloc_shared((C,), "int32")
        mrgA_val  = T.alloc_shared((topk + K2,), in_dtype)
        mrgA_idx  = T.alloc_shared((topk + K2,), "int32")
        mrgB_val  = T.alloc_shared((topk + K2,), in_dtype)
        mrgB_idx  = T.alloc_shared((topk + K2,), "int32")

        for t in T.serial(MAX_TASKS):                       # 静态边界
            row = t * NUM_KERNELS + kid
            if row < R:                                     # 运行时守卫
                b = row // (seq_len * kv_group)
                rem = row % (seq_len * kv_group)
                s = rem // kv_group
                g = rem % kv_group
                sv = starts[b, s]                           # 标量 GM 读
                ev = ends[b, s]

                # 初始化运行候补
                T.vbrc(-T.infinity("float32"), mrgA_val)    # 全缓冲 -inf
                for i in T.Parallel(topk + K2):
                    mrgA_idx[i] = 0

                for c in T.serial(NUM_CHUNKS):              # 静态边界
                    c0 = c * C
                    if c0 < ev and c0 + C > sv and c0 < seq_len_kv:  # chunk 级跳过
                        n_valid = T.min(C, seq_len_kv - c0)
                        T.copy(index_score[b, s, c0 : c0 + n_valid, g],
                               chunk_src[0 : n_valid])
                        for i in T.Parallel(C):             # 窗口掩码（含尾部垃圾覆盖）
                            pos = c0 + i
                            chunk_src[i] = T.if_then_else(
                                (pos < sv) or (pos >= ev) or (pos >= seq_len_kv),
                                -T.infinity(in_dtype), chunk_src[i])
                        T.vsort(chunk_src, chunk_val, chunk_idx,
                                descending=True, sort_axis=-1)
                        for i in T.Parallel(K2):            # 绝对索引化（int32 标量加）
                            chunk_idx[i] = chunk_idx[i] + c0
                        # 拼装 mergeA = [候补 | chunk 头部]：
                        # [0:topk) 段候补已在位（vsort 输入即 mergeA 本体），仅追加头部段
                        T.copy(chunk_val[0 : K2], mrgA_val[topk : topk + K2])
                        T.copy(chunk_idx[0 : K2], mrgA_idx[topk : topk + K2])
                        T.vsort(mrgA_val, mrgB_val, mrgB_idx,
                                descending=True, sort_axis=-1)
                        T.copy(mrgB_val[0 : topk], mrgA_val[0 : topk])  # 回填候补
                        T.copy(mrgB_idx[0 : topk], mrgA_idx[0 : topk])

                T.copy(mrgA_idx[0 : topk], index[b, s, g, 0 : topk])    # 输出
```

> 实现注记：伪代码中「候补已在位」表示拼装时 [0:topk) 段无需搬运（vsort 输入即 mergeA 本体）；Stage 2 落地时若 vsort 要求 src 与 dst_value 不得同 buffer，则按上式使用独立 mergeA/mergeB 双缓冲（UB 预算 §4.5 已按双缓冲核算）。

### 3.4 API 可行性确认

| API | 用途 | 来源确认 | 验证状态 |
|-----|------|----------|----------|
| `T.vsort(src, dst_value, dst_index, descending, sort_axis)` | 块内/归并排序 | docs/Tilelang.language/排序操作/T.vsort.md（fp32 √、尾轴、int32 索引、降序） | testing/npuir/sort_ops/test_vsort_dev.py 实测（N=1024）；本设计最大用量 8192 —— **L0 前置验证项**（§8.3 用例 6） |
| `T.copy(src_slice, dst_slice)` | GM↔UB / UB→UB 搬运 | docs/Tilelang.language/内存操作/T.copy.md（ub-ub、gm-ub；运行时长度切片） | examples/vectorize/vectorization_in_parallel.py、examples/elementwise/vec_add_2d_dynamic_shape.py 佐证 |
| `T.vbrc(scalar, dst)` | −inf 填充 | docs/Tilelang.language/shape操作/T.vbrc.md（标量→fp32 buffer，fp32 √） | docs/Tilelang.language/创建操作/T.infinity.md 示例同款（`-T.infinity("float32")`） |
| `T.Parallel` 元素级 + `T.if_then_else` 运行时条件 | 窗口掩码 | examples/vectorize/vectorization_in_parallel.py（T.Parallel + if_then_else）；examples/indexer/indexer_fwd.py（运行时窗口条件逐元素写 −inf） | 已验证模式 |
| `T.Parallel` 元素级 int32 标量算术 | 绝对索引化 / 索引置零 | examples/vectorize（`C_VEC[i] = A_VEC[i] + B_VEC[i]`） | 已验证模式（T.vadd/T.clear 不支持 int32，故用此路径） |
| 标量 GM 读（`starts[b,s]`） | 窗口参数装载 | examples/indexer/indexer_fwd.py（`CuSeqLenKS[s_idx]`） | 已验证模式 |
| `T.Kernel(n, is_npu=True)` 一维静态核数 + `T.serial` 静态边界 + 运行时守卫 | persistent 分核 | docs/开发指南.md §3.3 核内串行示例；examples/elementwise/vec_add_2d_dynamic_shape.py | 已验证模式 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **Yes**（源 `T.Kernel(batch, seq_len, kv_group)`） | 一维展开：R = batch×seq_len×kv_group 静态行集 + persistent 40 核（§0.6 R2 / §5.5） |
| GPU 专用 API（threads/get_thread_binding/sync_threads/atomic_add SMEM/loop_break/alloc_shared GPU 语义） | **Yes**（源大量使用） | 全部移除/重设计：线程模型→T.Parallel 向量化；SMEM 原子→vsort 排序排名（§0.6 R1） |
| GEMM 要求 M,N 为 block 整数倍 | No（纯 Vector，无 GEMM） | 不涉及 |
| L0C 容量上限 | No（无 Cube 计算） | 不涉及 |
| 物理核数限制 | **Yes**（R=1024~32768 远超物理核） | 分核策略三要素见 §5.5（极大规模路径：固定 40 核 + 核内静态串行） |
| 尾轴 32B 对齐 | **Yes** | C/K2/(topk+K2) 取 8 的倍数（fp32：尾维 8 元素 = 32B）；topk 非 8 倍数时编译器自动填充（性能项，非正确性项），manifest 全部 topk（32/64/1024/2048）为 8 倍数 |
| Host 侧输入操作约束 | Yes（边界） | host 仅做 shape 校验与 JIT 特化（沿用既有 op 封装），不触碰输入张量内容；无 im2col 类预处理 |

#### 3.5.2 参考实现差异说明（影响 API 选型的关键差异；完整差异分析见 §0.5/§0.6）

| 差异项 | 参考实现（GPU） | 本项目（Ascend NPU） | 转换方案 |
|--------|----------------|---------------------|----------|
| Kernel 维度 | 三维 `T.Kernel(batch, seq_len, kv_group, threads=1024)` | 一维 `T.Kernel(40)` persistent + 核内静态串行 | §0.6 R2；参照 docs/开发指南.md §3.3 示例 |
| 线程模型 | `T.get_thread_binding()` 1024 线程协作 | 无线程绑定；元素级 `T.Parallel` 向量化 | §0.6 R1 |
| 核心选 topk 机制 | SMEM 原子 radix-select（直方图+扫描+scatter） | 硬件 `T.vsort` 分块排序 + 流式归并 | §0.6 R1 |
| 内存分配 | `T.alloc_shared`（GPU SMEM，threads 可见） | `T.alloc_shared`（Developer 模式自动映射 UB，192KB 预算） | §4.5 |
| 同步 | `T.sync_threads` / 部分屏障 `sync_threads(3,RADIX)` | Developer 自动同步，无手动点 | §7 |
| 动态 shape | `T.dynamic("batch")` / `T.dynamic("seq_len_kv")` | 静态特化（per-shape 编译，与源 lru_cache 粒度一致） | §4.6 |
| pass_configs | `TL_DISABLE_THREAD_STORAGE_SYNC: True` | 无（GPU codegen 专属，舍弃） | §0.5 |
| 早退 | `T.loop_break` | chunk 级窗口重叠跳过守卫 | §0.5 |

**接口契约（Stage 2 必须遵守）**：NPU 侧 op 封装 `examples/TileOPs/tileops/kernels/attention_indexing/topk_selector/topk_selector.py` 以 `_topk_selector_kernel(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype)(config...)(index_score, starts, ends)` 两段式调用；本设计保持工厂签名与 `(index_score, starts, ends)` 调用序不变，配置参数位（GPU 的 RADIX/BLOCK_SIZE/SMEM_INPUT_SIZE/block_m）属后端实现相关，适配为 `(chunk_size=0 自动, num_kernels=40)` 并对旧键名容错。

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/indexer/indexer_fwd.py` | 高度相似（同为窗口化 attention_indexing 类算子） | ks/ke 窗口 `-T.infinity` 掩码、标量 GM 读 CuSeqLenKS、一维 Kernel + 动态循环 + 运行时守卫、Developer 模式 |
| `testing/npuir/sort_ops/test_vsort_dev.py` | 高度相似（排序原语） | `T.vsort` Developer 模式用法（alloc_shared + descending + int32 索引）、重复值索引校验方式 |
| `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_tiled/DESIGN.md` | 相似（TileOPs 迁移设计范式） | 大 N 分块 + UB 预算约束 + DESIGN 结构范式 |
| `examples/vectorize/vectorization_in_parallel.py` | 相似（Vector 元素级） | T.Parallel 元素级算术、if_then_else 运行时条件、运行时长度 T.copy |
| `examples/elementwise/vec_add_2d_dynamic_shape.py` | 相似（分核范式） | 固定核数 + T.serial 串行多任务 + `if block_id < total` 守卫 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `index_score` | `(batch, seq_len, seq_len_kv, kv_group)` | float32 | 尾维 kv_group 为 stride-1 维；G=1 时行内 kv 段连续（全部 workload G=1） |
| `starts` | `(batch, seq_len)` | int32 | 窗口起点（含） |
| `ends` | `(batch, seq_len)` | int32 | 窗口终点（不含） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `index` | `(batch, seq_len, kv_group, topk)` | int32 | topk 为尾维、连续写入；值降序 |

### 4.3 中间缓冲区（每核私有，全部 UB；Developer 模式 `T.alloc_shared` 自动映射）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `chunk_src` | `(C,)` | float32 | UB | chunk 搬入 + 掩码后的排序输入 |
| `chunk_val` | `(C,)` | float32 | UB | 块内 vsort 排序值输出 |
| `chunk_idx` | `(C,)` | int32 | UB | 块内 vsort 排序索引输出（chunk 相对索引 → 绝对索引化） |
| `mrgA_val` | `(topk+K2,)` | float32 | UB | 归并输入（[0:topk) 为运行候补值） |
| `mrgA_idx` | `(topk+K2,)` | int32 | UB | 归并输入索引 |
| `mrgB_val` | `(topk+K2,)` | float32 | UB | 归并 vsort 排序值输出 |
| `mrgB_idx` | `(topk+K2,)` | int32 | UB | 归并 vsort 排序索引输出 |

### 4.4 内存搬运路径

```
GM[index_score] --T.copy(行内连续段, 运行时长度)--> UB[chunk_src]
UB[chunk_src] --T.vsort--> UB[chunk_val | chunk_idx]
UB[chunk_val | chunk_idx] --T.copy(UB→UB, K2 段)--> UB[mrgA_val | mrgA_idx]
UB[mrgA] --T.vsort--> UB[mrgB] --T.copy(UB→UB, topk 段)--> UB[mrgA[0:topk]]
UB[mrgA_idx[0:topk]] --T.copy(连续段)--> GM[index[b,s,g,0:topk]]
GM[starts/ends] --标量读--> 寄存器（每行 2 次）
```

无 L1/L0A/L0B/L0C 参与纯 Vector 计算（GM↔UB 两级路径完整覆盖全部数据流）。

### 4.5 UB 内存预算（按 workload，dtype 均 4B）

C 取 `align_up(min(kv, 8192), 128)`，K2 = min(C, topk)，merge 长 = topk+K2：

| workload | C | K2 | topk+K2 | chunk 三缓冲 | merge 四缓冲 | 合计 | 预算 |
|----------|-----|-----|---------|--------------|--------------|------|------|
| smoke-topk32-s256-kv1k（kv=1024） | 1024 | 32 | 64 | 12 KB | 512 B | ≈12.5 KB | 192 KB ✓ |
| topk64-s512-kv2k（kv=2048） | 2048 | 64 | 128 | 24 KB | 2 KB | ≈26 KB | 192 KB ✓ |
| topk1024-s32k-kv64k（kv=65536） | 8192 | 1024 | 2048 | 96 KB | 32 KB | **128 KB** | 192 KB ✓（余 64 KB） |
| topk2048-s32k-kv64k（kv=65536） | 8192 | 2048 | 4096 | 96 KB | 64 KB | **160 KB** | 192 KB ✓（余 32 KB） |

最大占用 160 KB < 192 KB（A2/A3 UB 容量，docs/开发指南.md §3），留 ≥32 KB 余量吸收编译器对齐/临时开销。若实际编译报 UB 不足，降压方案：C 降为 4096（chunk 三缓冲 48 KB，合计 112 KB，chunk 数翻倍）。

### 4.6 动态轴定义

源声明 `batch`、`seq_len_kv` 为动态轴。**本设计将二者静态特化**（per-shape 编译）：

| 轴 | 声明方式 | 运行时范围 | 特化依据 |
|----|----------|-----------|----------|
| batch | JIT 工厂参数（静态特化，不声明 T.dynamic） | 1 ~ 8（manifest） | ① 源工厂 `lru_cache` 已按完整 shape 元组缓存，实际编译粒度本就是 per-shape，行为等价；② 分核要求核内串行循环边界为静态值（MAX_TASKS 依赖 R=batch×seq_len×kv_group），静态特化是满足该约束的直接手段；③ workload 形态有限（4 个），重编译成本可控 |
| seq_len_kv | JIT 工厂参数（静态特化） | 1024 ~ 65536（manifest） | 同上；同时使 NUM_CHUNKS、C 静态化 |
| starts/ends | 运行时张量数据（非 shape 轴） | 每元素任意 int32 | 真正的运行时变量：窗口掩码/chunk 跳过均以运行时条件处理，不影响循环边界静态性 |

### 4.7 JIT 配置

```python
@tilelang.jit(target="npuir", out_idx=[1])   # 输出为参数位 1（index），与源一致
```

不携带 `pass_configs`（源的 `TL_DISABLE_THREAD_STORAGE_SYNC` 为 GPU codegen 专属，已舍弃，见 §0.5）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 全部计算为排序（T.vsort）、掩码/元素级算术（T.Parallel）、数据搬运（T.copy），无 matmul；无 L1/L0 需求，仅 GM↔UB 两级。

### 5.2 Block 划分

两级划分：

```python
# 任务级：行（逻辑工作单元）
R = batch * seq_len * kv_group            # 逻辑任务总数（一行 = 一个 (b,s,g) 的窗口 topk）
NUM_KERNELS = 40                          # 物理核适配常量（§5.5）
MAX_TASKS = ceildiv(R, NUM_KERNELS)       # 每核串行任务数（静态）

# 数据级：行内 kv 轴 chunk
C = align_up(min(seq_len_kv, 8192), 128)  # chunk 长度：UB 预算约束下的最大值（§4.5）
K2 = min(C, topk)                         # chunk 对候补的贡献上限
NUM_CHUNKS = ceildiv(seq_len_kv, C)
```

选择理由：一行是不可分割的最小独立工作单元（行内归约状态（候补缓冲）跨 chunk 保持，跨核拆分一行需核间同步，得不偿失）；C 在 UB 预算（§4.5）内取最大值以最小化 chunk 循环次数与归并排序次数（开发指南 §3.2「缓存不超限前提下最大化分块」）。

### 5.3 约束分析

- **对齐约束**: C=1024/2048/8192、K2、topk+K2=64/128/2048/4096 均为 8 的倍数（fp32 尾轴 8 元素 = 32B 对齐 ✓）；topk=32/64/1024/2048 均为 8 倍数 ✓。一般 topk 非 8 倍数时：topk+K2 尾轴由编译器自动填充，多占 ≤7 元素（性能项非正确性项）。
- **UB 容量**: 最大 160 KB < 192 KB ✓（§4.5 详表，含降压方案）。
- **L0 容量**: 不适用（无 Cube 计算）。
- **vsort 规格约束**: 仅 fp16/fp32（本设计 fp32 ✓）、仅尾轴（一维 buffer 尾轴即唯一轴 ✓）、dst_index int32 ✓；单次排序最大长度 8192（L0 前置验证项）。

### 5.4 注意事项

- **非整除 chunk**：`seq_len_kv % C ≠ 0` 时尾 chunk `n_valid < C`：拷贝只搬 `n_valid` 段，`[n_valid, C)` 的未初始化数据由窗口掩码 pass 统一覆盖为 −inf（掩码条件 `pos ≥ seq_len_kv` 恒真），排序后沉底，不污染结果——**无需 host 侧 padding**（符合 host 输入操作约束）。
- **kv_group > 1 的访存**：`index_score[b,s,:,g]` 在 GM 中 stride = kv_group。G=1（全部 workload）走连续段 `T.copy` 快路径；G>1 时回退为 `for i in T.Parallel(C): chunk_src[i] = index_score[b, s, c0+i, g]` 逐元素读（正确性优先，性能次之），列入 §9 风险与 L0 验证项。
- **窗口为空 / 越界窗口**：chunk 跳过守卫使全部 chunk 被跳过，输出 = 初始化的 0 索引（定义行为，§0.6 R1）。
- **R < NUM_KERNELS**（极小 batch×seq_len×kv_group）：守卫使多余核空转，正确性不受影响（此时等效"逻辑核数 ≤ 物理核数：无需适配"路径）。
- **窗口负载均衡**：各行窗口长度差异巨大（[starts, ends) 可变）；行 stride 交错分配（row = t×40 + kid）+ chunk 级跳过守卫共同缓解：窄窗口行自动跳过无重叠 chunk，跨核统计均衡。不引入跨核窃取（需核间同步，v1 不做，列 §9 调优项）。

### 5.5 分核策略（物理核数适配）⭐

> 依据 docs/开发指南.md §3.3：AI Core 物理核数有限（A2 系列 Cube 核约 20~24 个，Vector 核数量翻倍），超发逻辑内核会被运行时串行调度并引入额外核启动开销；内核总数非物理核数整数倍会造成负载不均。

- **① 逻辑核数计算**：本算子输出网格为 `(batch, seq_len, kv_group)`，一行 `(b,s,g)` 为最小独立逻辑任务，逻辑核数（每逻辑核一行口径）`num_logical_kernels = batch × seq_len × kv_group = R`。按 workload：smoke 4×256×1 = **1024**；topk64 8×512×1 = **4096**；topk1024/2048 1×32768×1 = **32768**。
- **② 物理核数依据**：本算子为纯 Vector 计算 → 目标 Vector 物理核数。取 **40~48**（A2 系列 Cube 约 20~24、Vector 数量翻倍；来源：docs/开发指南.md §3.3 与 skill ascend-constraints.md 的**文档假设，显式标注**；Stage 2/4 优先通过设备接口查询实际核数覆写该假设）。
- **③ 规模判定与分核方案**：三个规模的 R（1024 / 4096 / 32768）均 **远大于物理核数 40~48 → 极大规模路径**：
  - 固定启动内核数 `NUM_KERNELS = 40`（= 物理核数下限的整数倍对齐：40 = 2×20，且 ≤ 40~48 区间下限——既保证 40 核设备满载，也避免 48 核设备上 41+ 核超发串行；40 与 20/40 物理核数整除对齐，杜绝"21 核执行两倍任务"型负载不均）。
  - 核内串行：`for t in T.serial(MAX_TASKS)`，`MAX_TASKS = ceildiv(R, 40)`（**静态值**：R 因 batch/seq_len_kv 静态特化为编译期常量，smoke=26、topk64=103、topk1024=820、topk2048=820）+ 运行时守卫 `if t*40 + kid < R` 处理非整除尾差（每核任务数差 ≤1）。
  - 行间 stride 交错分配（row = t×40 + kid）：窗口长度差异在 40 核间统计均衡（对比连续段分配需工作量大小的前缀和划分，v1 不引入）。
  - 未选"中等规模对齐物理核整数倍"的依据：一行是最小独立工作单元，"增大 block 减少核数"在本算子只能表现为"每核串行多行"——即本 persistent 方案本身；且 smoke 的 R=1024 若按 26 行/核发 40 核与按对齐发 20/40 核相比，满波一次完成、无第二轮调度，persistent 方案对所有 manifest 规模统一最优。
  - 极小规模兜底（R ≤ 40，manifest 外）：守卫自动退化为"逻辑核数 ≤ 物理核数：无需适配"，每核 ≤1 行，单波完成。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 边界 | 理由 |
|------|----------|-----|------|------|
| 核间（40 核） | block 级并行 | `T.Kernel(40, is_npu=True)` | 静态（常量 40） | persistent 分核（§5.5，承接源三维 grid 的行并行意图） |
| 核内行循环 | 串行 | `T.serial(MAX_TASKS)` | **静态**（编译期常量）+ 运行时守卫 `row < R` | 摊薄核启动开销（开发指南 §3.3 极大规模模式）；静态边界满足约束 |
| 行内 chunk 循环 | 串行 | `T.serial(NUM_CHUNKS)` | **静态**（kv 静态特化）+ 运行时 chunk 跳过守卫 | 逐块流式归约；静态边界满足约束 |
| chunk 内掩码/索引化 | 元素级并行 | `T.Parallel(C)` / `T.Parallel(K2)` | 静态 | Vector 向量化元素操作 |
| 候补初始化/回填 | 元素级并行 / 段拷贝 | `T.Parallel(topk)` / `T.copy` | 静态 | 向量化 |

无 `T.Pipelined`（v1）；无 `T.Persistent` 显式算子（以固定核数 + 核内 serial 实现 persistent 语义，与开发指南 §3.3 示例一致）。

### 6.2 循环伪代码

见 §3.3 完整伪代码（四层结构：Kernel(40) → serial(MAX_TASKS) → serial(NUM_CHUNKS) → Parallel(C)）。

### 6.3 流水线优化

v1 不使用 `T.Pipelined`：chunk 循环体内 vsort 为长延迟算子，双缓冲（chunk 拷入与上一块 vsort/归并重叠）为 **Stage 4 调优项**——需 UB 再增一份 chunk 缓冲（+32 KB，预算内可行：160→192 KB 临界，需 C=4096 配合）。设计预留：chunk 循环结构天然适配 `T.Pipelined(NUM_CHUNKS, num_stages=2)`。

### 6.4 尾块处理

- 尾 chunk（`n_valid < C`）：仅拷贝有效段 + 掩码 pass 覆盖尾部（§5.4），无需 host padding。
- 尾行（`R % 40 ≠ 0`）：`row < R` 守卫空转多余迭代。
- 尾任务核（`MAX_TASKS` 不均）：每核任务数差 ≤1（ceildiv 分配）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（Developer 模式编译器自动插入）

### 7.2 同步点说明

无手动同步点。理由：
- 行间完全独立（无跨行读写共享数据）→ 无核间同步需求（不使用 sync_block_set/wait_flag）。
- 核内为纯串行控制流（serial 循环），循环体内 UB 读写由编译器按依赖自动排序（Developer 模式职责）。
- 源的 `T.sync_threads`/`sync_threads(3, RADIX)` 属线程协作原语，随线程模型一并消失（§0.5）。

### 7.3 pass_configs 配置

```python
# 无（源 TL_DISABLE_THREAD_STORAGE_SYNC 为 GPU codegen 专属，舍弃）
@tilelang.jit(target="npuir", out_idx=[1])
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据实现（移植源仓 `test_topk_selector.py` 的 `ref_program` 并扩展窗口掩码语义），**不复刻 §0.6 的 NPU 算法**——保证验证独立性。

```python
import torch

def golden_topk_selector(index_score: torch.Tensor, starts: torch.Tensor,
                         ends: torch.Tensor, topk: int) -> torch.Tensor:
    """窗口化 topk 索引选择参考实现（§0.1 语义）。

    index_score: (B, S, S_kv, G) fp32；starts/ends: (B, S) int32。
    返回 (B, S, G, topk) int32，值降序（torch.topk 序）。
    """
    B, S, S_kv, G = index_score.shape
    kv_idx = torch.arange(S_kv, device=index_score.device).view(1, 1, S_kv, 1)
    # 窗口掩码：有效 = starts <= i < ends 且 i < S_kv（与源三条件一致）
    mask = (kv_idx >= starts.view(B, S, 1, 1)) & \
           (kv_idx < ends.view(B, S, 1, 1)) & \
           (kv_idx < S_kv)
    masked = torch.where(mask, index_score,
                         torch.full_like(index_score, float("-inf")))
    idx = torch.topk(masked, topk, dim=2)[1]        # (B, S, topk, G)
    return idx.permute(0, 1, 3, 2).contiguous()      # (B, S, G, topk)
```

全窗口（starts=0, ends=S_kv）时与源仓 `ref_program`（`torch.topk(index_score, topk, dim=2)[1].permute(0,1,3,2)`）逐位一致；窗口 < topk 的退化行为：torch.topk 会把 −inf 掩码项选入尾部，与 §0.6 R1 定义行为一致（索引值可能不同——契约外场景，仅校验窗口元素集合，见 8.3 用例 3）。

### 8.2 精度标准

输出为 int32 索引，**无浮点容差**（atol/rtol 不适用）；比较语义为**集合比较**（对齐 `examples/TileOPs/tests/ops/test_topk_selector.py` 的 `_set_compare`：topk 并列值时索引集合比较而非逐位比较）：

| 输出 dtype | 比较方式 | 标准 |
|-----------|----------|------|
| int32 索引 | 逐行集合比较（L0 门槛，强于 harness） | 每行 `set(out[b,s,g,:]) ⊇ set(golden_row)` 且基数 = topk（有效窗口 ≥ topk 时双向相等） |
| int32 索引 | 展平集合比较（harness `_set_compare`，正式契约） | `len(set_ref & set_trt) / len(set_ref) == 1.0` |

并列值补充校验（金标准独立性）：逐行验证 `out` 索引指向的原值降序排列且前 topk 值集合与 golden 值集合一致（防"索引集合对但值错"）。

### 8.3 L0 门槛测试计划（L0 代表 shape = smoke workload：batch=4, seq_len=256, seq_len_kv=1024, kv_group=1, topk=32, fp32/int32）

| # | 用例 | 输入构造 | 通过标准 |
|---|------|----------|----------|
| 1 | 全窗口随机（harness 同款） | `randn` 分数、starts=0、ends=1024 | 逐行集合比较 100% + 值降序校验 |
| 2 | 随机部分窗口 | 每行 `starts ∈ [0, 1024−topk]`、`ends ∈ [starts+topk, 1024]` 随机 | 逐行集合比较 100%（golden 按窗口掩码计算） |
| 3 | 窗口 < topk 退化 | starts/ends 使有效元素 < 32 | 窗口元素索引集合完整出现在输出前 W 槽；余槽为填充（定义行为，§0.6 R1） |
| 4 | 并列值/重复 | 分数含大段常数（如全 0.5、重复最大值） | 值集合相等 + 索引指向值一致（集合语义） |
| 5 | 多 chunk 路径 | kv=1024 > C 的人为缩参（C=256 编译实例）或 mid shape（kv=2048） | 同用例 1 标准（验证 chunk 归并跨块正确性） |
| 6 | vsort 规模前置冒烟 | 独立小 kernel：fp32 降序 vsort 8192 元素（含 −inf 与重复值） | 与 torch.sort 值一致 + 索引为排列 |
| 7 | kv_group > 1 回退路径 | G=2 小 shape（如 2×8×64×2, topk=16） | 逐行集合比较 100%（逐元素读路径正确性） |

L0 通过线：用例 1–6 全过（用例 7 为回退路径验证，G>1 不在 manifest 内，不过不阻塞但必须记录）。完整 L1（全 workload 精度）/L2（性能基线）/Boundary 套件由 Stage 3（tilelang-op-develop）按本计划展开。

---

## 9. 风险点与注意事项

### 9.1 已知约束与技术风险

1. **vsort 单次排序规模上限未文档化**（docs 未给尺寸上限，实测最大 1024；本设计最大 8192）。缓解：L0 用例 6 前置冒烟；若受限，C 自适应降档（4096/2048/1024），算法结构不变（NUM_CHUNKS 相应增大）。
2. **vsort 重复值索引顺序与 torch.sort 不一致**（docs 明示）。集合比较语义覆盖；golden 值校验兜底（§8.2）。
3. **物理核数 40~48 为文档假设**（非设备接口实测）。缓解：NUM_KERNELS 为编译期常量可调；Stage 4 以 msprof 核利用率数据校准（40→实测值整数倍对齐）。
4. **vsort 路径 vs 源 radix 路径的性能差**：比较运算量约 O(C·logC)×NUM_CHUNKS + O(2·topk·log(topk))×NUM_CHUNKS，高于 radix 的 ~N 常数倍扫描；GM 流量两者同量级（单遍读）。属性能风险非正确性风险；Stage 4 验证，备选优化：chunk 级剪枝（`reduce_max(chunk) ≤ 候补第 topk 值` 则跳过该 chunk 排序）、T.Pipelined 双缓冲（§6.3）。
5. **kv_group > 1 逐元素读性能**（G>1 时行内 kv 段 stride=G 非连续，回退逐元素 T.Parallel 读）。manifest 全部 G=1 不受影响；正确性由 L0 用例 7 保证。
6. **UB 预算余量**：最大 workload 160 KB / 192 KB，余 32 KB。若编译器额外占用超预期报 UB 不足 → C 降 4096（§4.5 降压方案）。
7. **NaN 排序未定义**（与源一致，契约外；测试 randn 无 NaN）。
8. **窗口 < topk 行为差异**：源未定义、本设计已定义（§0.6 R1）；harness 全窗口测试不触发。
9. **harness `_set_compare` 展平集合比较在大 workload 近乎恒真**（67M 索引取值域仅 64K，集合几乎必为全集）——正式契约如此、照常执行；本设计 L0 逐行集合比较（严格更强）作为设计阶段门槛，避免以弱测试自证。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| vsort 编译报 dtype/轴不支持 | src 用了 int32 或 sort_axis≠-1 | 编译失败 | 严格按规格：src/dst_value fp32、sort_axis=-1、dst_index int32 |
| UB 溢出 | C 过大或 buffer 数过多 | 编译失败 | §4.5 预算表 + C 降档 |
| 尾 chunk 垃圾数据污染 | 未做窗口掩码即排序 | 结果错误 | 掩码 pass 条件含 `pos ≥ seq_len_kv`，覆盖未拷贝尾部 |
| int32 运算误用 vadd/T.clear | vadd/T.clear 仅 fp16/fp32 | 编译失败 | int32 用 T.Parallel 元素级标量算术（§3.2） |
| 动态循环边界 | MAX_TASKS/NUM_CHUNKS 依赖运行时值 | 违反静态边界约束 / 不可预期的核行为 | batch/seq_len_kv 静态特化（§4.6） |
| 输出布局写反 | 沿用输入 (S_kv, G) 尾序 | shape/语义错误 | 输出固定 (…, kv_group, topk)，golden permute(0,1,3,2) 对齐 |

### 9.3 特殊场景处理

见 §5.4（非整除 chunk、G>1、空窗口、R<40、负载均衡）与 §8.3（退化窗口、并列值）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/topk_selector/_topk_selector_kernel/
├── DESIGN.md                      # 本设计文档
├── _topk_selector_kernel.py       # 算子实现（kernel + golden + L0 冒烟测试）
└── README.md                      # 使用说明（可选）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 本设计文档 |
| `_topk_selector_kernel.py` | 待实现（Stage 2） | NPU kernel：`_topk_selector_kernel(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype)` 两段式调用工厂，接口契约见 §3.5.2；内含 §8.1 golden 与 §8.3 L0 冒烟 |
| `test__topk_selector_kernel.py` | 待实现（Stage 3） | L1/L2/Boundary 完整套件（对齐 examples/TileOPs/tests/ops/test_topk_selector.py 契约） |

### 10.3 命名规范

- 项目目录名: `topk_selector`（snake_case，project_name）
- 算子目录名: `_topk_selector_kernel`（snake_case，op_name）
- 实现文件: `_topk_selector_kernel.py`
- 测试文件: `test__topk_selector_kernel.py`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）
2. ⬜ Golden 函数（§8.1，验证基准）
3. ⬜ L0 前置冒烟：vsort@8192 规模验证（§8.3 用例 6）
4. ⬜ 算子实现（`_topk_selector_kernel.py`）+ L0 逐行集合比较（§8.3 用例 1–5、7）
5. ⬜ Stage 3 全量精度套件 / Stage 4 性能调优（chunk 剪枝、Pipelined、核数校准）
