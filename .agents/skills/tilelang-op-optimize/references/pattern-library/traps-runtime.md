# 已知运行时/数值/语义陷阱（工具链版本绑定 ⚠️）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。条目带 front-matter（schema 见 INDEX.md §3）；`repro: repro-missing` 表示待回填最小复现代码。编译器/解析器类陷阱见 [traps-compiler.md](traps-compiler.md)。
>
> **证伪协议（强制，canonical 见 INDEX.md §2，两者同步演进）**：
>
> 1. **否定任何 API/模式前，必须用文档合法形态测试**——先查 `docs/Tilelang.language/` 确认 API 的合法参数/形式，穷举代表性写法后再下结论。
> 2. **一切运行时/数值结论必须盖工具链版本戳**（tilelang commit/build 时间 + 来源任务），工具链变更后**自动视为待重验**，不得直接引用旧结论。
> 3. 证伪更正时须在 opt_log 写明"误判根因 + 合法形态 + 新数据"。

---
id: TRAP-C9-taskqueue-async
kind: trap
family: [general]
apis: []
dtype: [fp32]
device: 910B2C
status: verified
origin_task: mixed（2026-08 溯源）
toolchain: 截至 2026-08-28 build
repro: repro-missing
---

### C9 `TILELANG_ENABLE_TASKQUEUE=false` 异步 launch 损坏 fp32 数据

有效（截至 2026-08-28 build）。

---
id: TRAP-UB-dst-align
kind: trap
family: [general]
apis: [T.copy]
dtype: []
device: 910B2C
status: verified
origin_task: mixed（2026-08 溯源）
toolchain: 截至 2026-08-28 build
repro: repro-missing
---

### UB dst 非零起点切片 + 32B 倍宽度触发 VEC 对齐错误

有效（截至 2026-08-28 build；host pad 或 0 起点拷贝绕开）。

---
id: TRAP-fp16-opmath-golden
kind: trap
family: [elementwise, migration]
apis: [vcast, vadd, vsub, vmul]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z
toolchain: torch 2.9.0+cpu + tilelang-mlir-dev dev root build 2026-09-07
repro: repro-missing
---

### torch CPU golden 的 fp16 opmath 域分歧

torch.lerp（torch 2.9.0+cpu）对 fp16 输入经**fp32 opmath + 单次舍回**计算，NPU fp16 原生域三步链（逐步 fp16 舍入）与之差 ~2–3 ulp fp16，N=2^24、atol=rtol=1e-3 下违反率 ~0.125% 且与 shape/block_size/核数无关（纯舍入路径统计性质，非 tiling/同步缺陷）；bf16 golden 与 fp32 中转 kernel 逐位一致。对齐通解：`vcast(rint)` 升 fp32 → fp32 域 v-prefix 链 → `vcast(rint)` 单次舍回，差 ≤1 ulp fp16（rtol≥5e-4 即覆盖）；NaN/Inf 角点 IEEE 传播不受中转影响。复现（provenance，允许失效）：`python examples/lerp_tensor/_make_lerp_tensor_kernel/history_version/_make_lerp_tensor_kernel_impl_s3_attempt1.py --level L0`（fp16 违反）vs `python examples/lerp_tensor/_make_lerp_tensor_kernel/_make_lerp_tensor_kernel.py --level all`（全过）。

---
id: TRAP-load-nd2nz-strided
kind: trap
family: [attention, expert, cube]
apis: [T.load_nd2nz, T.copy]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: repro/TRAP-load-nd2nz-strided.py
---

### `T.load_nd2nz` 对跨步（非尾二维连续）src 区域静默平坦误读

BSHD [B,S,H,D] 按固定头取 (S,D) tile（区域 [1,real_m,1,dim]，dim1 步长 H·D）被按「基址起平坦连续内存」读取——无告警无报错，数值表现为整块乱值（ws_s 与任何 head 的 QK^T 均不匹配；flat-read 假设逐点复现 got[i,j]==flat_q[i]·flat_k[j]）；`T.copy` **base+size 形态**对相同跨步区域同样错误（diff 3.4），**slice 形态** `T.copy(q[bz, s_lo:s_lo+real_m, by, 0:dim], l1[0:real_m, 0:dim])` bit-exact 正确且同为 PIPE_MTE2 单次搬运——跨步 GM 区域装载的实测正确形态为 slice 形态 T.copy（文档「src 支持 2-4D tensor」未标注连续性约束；能力缺口登记 CG-2026-0004）。复现：`repro/TRAP-load-nd2nz-strided.py`（知识域——slice 绕法 bit-exact 断言）；原 session 探针（provenance，session-local 允许失效）：probe_stridecopy.py（slice vs base+size 对照）、probe_l1slot.py——关键发现镜像于任务 debug_log D1/D2。

---
id: TRAP-T-copy-region-semantics
kind: trap
family: [attention, expert, general]
apis: [T.copy, T.transpose]
dtype: [fp16, f16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（HEAD 21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: repro/TRAP-T-copy-region-semantics.py
---

### `T.copy` 区域语义三规则

① 标量基址 + size 的 extents **前向补 1**（size=[1,N] 落末维、[N,1] 落倒数第二维；`tilelang/language/copy.py` L42-95 源码语义）——lse 写出方向由它决定；② **src/dst 区域元素数不匹配时引擎按 src 搬运并越界写坏相邻 GM**（`[0:half]` 源写 `[0:real_m_half]` 目的区砸坏相邻 tensor，症状酷似跨 pipe 竞态——仅部分核数据坏、set_flag/wait_flag 无效；判据 = 溢出值恰为源 buffer 内容；只在尾块位形 Sq%bm≠0 出现，整除 shape 全量测试发现不了；正确形态 = src 切片 `[0:real_rows]`）；③ [N,1] UB 源 + size=[1,N] 按步长越界读（MTE DDR fault）——行向量写 GM 尾维连续区先 `T.transpose` 到 [1,N]（Developer shared 源不需要，UB 源必须）。**推荐通用形态：src/dst 双显式 slice**。复现：`repro/TRAP-T-copy-region-semantics.py`（知识域——合法 slice 形态断言）；原 session 探针（provenance，session-local 允许失效）：probe_lsemulti2.py、probe_copyforms.py——关键发现镜像于任务 debug_log D2/D3。**〔2026-09-10 重验注记〕**repro 转正重跑（repro/TRAP-T-copy-region-semantics.py）：slice 形态 PASS 存活；历史 base+size=[HALF,D] 标量基址形态（原会话通过）在当前工具链复现 MTE DDR fault——按版本戳规则该形态**待重验**，条目结论以 slice 形态为准。

---
id: TRAP-zero-input-crash
kind: trap
family: [expert, general]
apis: [T.prim_func]
dtype: [bf16, fp16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: repro/TRAP-zero-input-crash.py
---

### 零输入 kernel（out_idx-only）运行期必崩 "MTE DDR address out of range"

启动参数错乱——崩溃形态与真实越界写完全同款，曾误诊为拷贝形态 bug 消耗 3 轮探针；探针 kernel 保留 ≥1 个输入 tensor 后同形态全过。诊断「部分核/部分行数据坏」先做元素级 fp64 对照判定哪侧偏离真值（本例 golden 才是偏离方：|kernel−true|=4.3e-4 vs |golden−true|=5.8e-3，bf16 P 量化噪声放大），避免在错误层面（跨 pipe 竞态假设）空转。复现：`repro/TRAP-zero-input-crash.py`（知识域——dummy input 绕法断言）；原 session 探针（provenance，session-local 允许失效）：probe_copymin.py 加 dummy input 前后对照——关键发现镜像于任务 debug_log D2/D6。

---
id: TRAP-vrsqrt-plain-precision
kind: trap
family: [norm, reduction, elementwise]
apis: [T.vrsqrt, T.vsqrt, T.vdiv]
dtype: [fp32, fp16, bf16]
device: 910B2C
status: verified
origin_task: ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z
toolchain: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 + torch 2.9.0+cpu / 2026-09-10
repro: repro/TRAP-vrsqrt-plain-precision.py
---

### `T.vrsqrt` plain 模式为近似指令（max rel err ~2.9e-3）

npuir 上 `T.vrsqrt` lower 为 plain 近似 Vector 指令：输入几何扫描 [1e-6, 1e6] 实测 max rel err 2.87e-3（返回值量化到 ~10–11 位有效数字，如 1010/1024）。传播效应（ada_layer_norm Stage 3 实测）：fp32 1e-5 门禁下 87% 元素违反、fp16 1e-3 门禁下 6%、bf16 1.6e-2 宽容差掩盖——「fp32 大面积违反而 bf16 全过」是常数相对误差（近似指令）的指纹，属舍入路径性质而非数据依赖/同步缺陷。`examples/norm/layer_norm.py` 通过的 1e-2 容差会掩盖该问题，不构成 vrsqrt 精度佐证。

**绕法**（ada_layer_norm 交付形态）：`T.vsqrt` + `T.vdiv` 组合（sqrt(v)/v ≡ 1/sqrt(v) 实数恒等，两 op 全精度）实测 max rel err 1.07e-7；修正形式 Newton×2 迭代亦可达 6.3e-8（代价 12 个微型 op）。含 rsqrt 且容差 <1e-2 的算子（layernorm/rmsnorm/softmax 归一化族）在本工具链上以组合形态达标。定位手法：中间量分层导出（`out_idx` 多输出落 GM + fp64 精确值对照）单次运行把误差定位到具体指令（ada 案例：mean/d 精确 1.1e-7 而 rstd 偏 1.57e-3 ⇒ 误差独占于 vrsqrt 段）。配套事实：文档文件名与 API 导出名存在偏差是常态（T.rsqrt.md → `T.vrsqrt`；T.vLn.md → `T.vln`），`examples/` 实调代码是 API 名核对入口。

复现：`python repro/TRAP-vrsqrt-plain-precision.py`（断言量级：raw > 1e-3 现象存在 + bypass < 1e-6 绕法通过；工具链修复后首断言翻转即条目推翻信号，届时刷新版本戳）。溯源（provenance，允许失效）：`examples/ada_layer_norm/_ada_layer_norm_kernel/`（Stage 3 分层探针 + Implementation Notes attempt-1 precision fix 段；repro 原件 `examples/ada_layer_norm/_ada_layer_norm_kernel/repro/TRAP-vrsqrt-plain-precision.py`）。
