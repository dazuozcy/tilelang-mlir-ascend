# Attention / Expert persistent 模式与硬上限（attention 族实测）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。条目带 front-matter（schema 见 INDEX.md §3）；`repro: repro-missing` 表示待回填最小复现代码。

---
id: PL-1.7-expert-persistent-boundary
kind: pattern
family: [attention, expert]
apis: [T.gemm, sync_block_set, sync_block_wait]
dtype: [fp16, bf16]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: expert 侧 tilelang dev build 21586b5（2026-09-07）+ CANN 8.5.0；developer 基线侧 tilelang 67db6f3（2026-09-04）+ CANN 26.0.rc1
repro: repro-missing
---

### 1.7 Expert 双 Scope persistent 与简单 tiling 的收益分界（attention 族同门实测）✅ 已验证

- **模式**：同一算子（GQA prefill flash-attention 前向）两代实现同门对比——Expert persistent 24 核双 Scope（Cube/Vector 跨引擎块级流水 + GM workspace 多槽 + per-slot flag）vs Developer 简单 tiling（fragment + 自动 cv_split + T.Pipelined）；同 wrapper config（block_m=64, block_n=64, num_stages=1）、同 msprof op kernel-only 口径。
- **实测收益分界**（10 workload bench）：S=2048 长 KV **1.57–1.63× 提速**（8b-long 13271 vs 20883 µs fp16；70b-long 13307 vs 20838 µs）；S=512 短 KV **~1.31× 回退**（3764/3774 vs 2872/2873 µs）；smoke d64 ~1.27× 回退（282 vs 222 µs）。吞吐画像随 S 反转：short 2.70 vs 3.53 TOps/s、long 5.58 vs 3.55 TOps/s——工作假设（本轮未单独验证）：persistent + 跨引擎流水的固定入口成本只在长 KV 循环摊销。集成期 bench 见短 workload 回退属该形态的预期画像（Stage 5 只记录不修复）。
- **口径注记**：expert 侧 tilelang dev build 21586b5（2026-09-07）+ CANN 8.5.0，developer 基线侧 tilelang 67db6f3（2026-09-04）+ CANN 26.0.rc1——跨工具链代对比，倍率含版本间差异成分。
- 溯源：`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/integration_log.md` §Bench；`examples/TileOPs/profile_run_msprof_20260907_173440.log`（expert）vs `examples/TileOPs/profile_run_msprof_20260904_110930.log`（developer）；复现：wrapper baseline 激活态下 TileOPs 根 `python -m pytest benchmarks/ops/bench_multi_head_attention.py -v -s`；origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z。

---
id: PL-1.8-bf16-cube-direct
kind: pattern
family: [attention, expert, cube]
apis: [T.copy, T.gemm, T.vcast]
dtype: [bf16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: repro-missing
---

### 1.8 bf16 Cube 直连链（Expert 形态）实测可用且无延迟税 ✅ 已验证

- **形态**：`T.copy` bf16 GM→L1（slice 形态，见 traps-runtime.md TRAP-load-nd2nz-strided 行）+ Scope("Cube") `T.gemm` bf16×bf16→fp32（L0C 累加）+ `T.vcast` f32→bf16 回写。文档依据：`docs/Tilelang.language/线性代数操作/T.gemm.md` §2.2.1（bf16 √）+ §2.3（bf16×bf16 dst fp32 实测可用）；`testing/npuir/bf16_support_ops/test_gemm_bf16.py`（Developer 形态先例）。
- **实测（Expert 端到端，L0+契约/L1/L2/Boundary 29 用例全过）**：bf16 out tier-2 max_diff 1.56e-2 = 2 bf16 ulp、lse 1.9e-6、tier-1 翻转率 77–96/262144 ≈ 0.03%（与上轮 developer 谱系 D8 校准一致）；集成 bench 10 workload bf16/fp16 差全部 **≤0.2%**（如 8b-long 13274.85 vs 13270.57 µs）——上一 developer 轮载体链 bf16 普遍慢 2–4%，直连形态无延迟税。
- **证伪更正**：两轮迁移设计曾断言「bf16 Cube 输入 ×」并据此驱动 E5 载体链全链设计——实为负向断言未亲核文档表格（引用了正确路径、转述了相反内容）；本轮文档+测试复核推翻该断言（举证规则见 `_shared/standards/negative-claim-evidence.md`，提案见 queue VP-2026-0025）。
- 溯源：`examples/multi_head_attention/_gqa_prefill_fwd_kernel/debug_log.md` Final results + 同目录 `RETROSPECTIVE.md` Stage 3/Stage 5；复现：`python examples/multi_head_attention/_gqa_prefill_fwd_kernel/_gqa_prefill_fwd_kernel.py --level L0`；tilelang dev root build 2026-09-07（21586b5）+ CANN 8.5.0 + Ascend910B2C；origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z。

---
id: PL-1.9-blockwidth
kind: pattern
family: [attention, expert]
apis: [T.gemm, sync_block_set, sync_block_wait, T.vcast, store_fixpipe]
dtype: [fp16, bf16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z
toolchain: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68（三轮同 commit）/ CANN 8.5.0 / Ascend910B2C
repro: repro-missing
---

### 1.9 Expert 跨引擎块级流水的「块宽摊减」调优序与硬上限（attention 族实测）✅ 已验证

- **块宽摊减律**：persistent 双 Scope（Cube/Vector per-slot flag 块级流水）结构下，每块跨引擎串行链 V1(softmax)→FLAG_P→C2(gemm2)→FLAG_O→V2(accumulate) 是周期地板（busy 核 aic_scalar 38–47% = flag 等待自旋，cube gemm 仅 9–12% 已近峰值）。**加大 bn 同时削减向量 op 数/块、flag 往返数/列与 gemm 碎片化**，比加深流水更根本——实测 bn 64→256→512 使 ns=1 与 ns=2 打平（宽块上双槽 ws 流水被单槽 L1 的 v-load↔gemm2 WAR 串行抵消）。G=2 批量重基准（批末共享 max + L0C initC=True/False 对累加 + 每对 1 次 fixpipe/FLAG_O/V2）在 bn=256 上 -5%~-10%，但被 bn=512 宽块全面反超（273 vs 327µs @fa4096），且 pair@bn=512 需双 V tile 驻留 L1 而结构性不可行。
- **cbuf(L1)=512KB 实测硬上限**（Ascend910B2C，BishengIR `cbuf overflow, requires N bits while 4194304 bits available` 报错文本反推，2026-09-08）。双槽 k/v/p 的 bn 上限：`2·slots·bn·dim·2 + slots·bm·bn·2 + bm·dim·2 ≤ 512KB`（bm=44 时 bn≤427）。**wide 单槽模式**：L1 缓冲单槽（KS=1，C2-first issue 保序消 WAR）+ GM workspace/flag 保持双槽——L1 减半换块宽，bn=512 可行（k+v+p+q=312KB）；UB 侧 clean trace 特化（剔除 mask/softcap/guard 死缓冲 ~90KB/AIV@bn=256）后 bm=44/bn=512 ≈97KB ✓。
- **store_fixpipe f32(L0C)→f16(GM) 为正确数值转换**（probe 实测 max_rel 4.7e-4 = 1 f16 ulp，非位重解释；vcast f16→f32 合法）——S-f16 跨引擎传输可行：S 字节减半，宽块上 -3%（fa2048/4096 达 DESIGN 备选 B 采纳阈值）；单块 M7 链上 vcast +1 op 抵消收益（平区不采纳）。**探针教训**：探针 kernel 必须带 Cube→Vector flag 同步，缺同步的数值崩坏是探针自身竞态而非 API 缺陷。
- **工厂级多 @T.prim_func 变体分派**：trace 级特化（full/clean/G2/...）用工厂闭包层 Python if 选择不同 builder（每个 builder 独立 @tilelang.jit），绕开「parser 不折叠 if」限制；full 变体逐字节保留使非目标 dispatch 零回归（causal 实测 +0.12%/-0.26%）。
- **工厂内 tuned 分派表（config 级，S4-5 模式；VP-2026-0035，attention 首证 + ada_layer_norm 第二证 2026-09-10）**：wrapper 契约保持的分派语义——caller 传 wrapper 默认 config 且 shape 命中 TUNED_DEFAULT_CONFIGS/TUNED_DEFAULT_BLOCK_M 时替换为调优 config；caller 传任何显式非默认 config 一律尊重原值（不静默覆盖用户意图）。tuned 分派覆盖的目标 case 纳入内嵌测试套件作 blocking 精度层（fa-tuned 层：dispatch 路径 tier-1 翻转数对照基线），防止分派表与全量套件漂移。第二证要点（norm 族 TUNED_DEFAULT_BLOCK_M）：① bm 甜点非单调且逐 shape（dit 1024×1152 bm=7 局部最优 {5:9.75, 6:9.66, 7:8.76, 8:9.55}µs，同 N 下 4096×1152 则 bm=8 胜——按 shape 实测显式记录，不从单点泛化「UB 松弛律」类规则）；② 表条目须做邻域闭合验证（bm±1 未测点全档 +1.0~7.0% 劣化即条目确认）；③ TileOPs bench/pytest 走 wrapper `default_config`（GPU 结构性值，如 block_m=1）而非 kernel 工厂 UB 表默认——集成 bench 的低 Ratio 常是 wrapper 默认未覆盖所致，Stage 4 第一杠杆即 wrapper config 覆盖（ada：bm=1→调优表贡献全组 2.14x 几何平均的最大份额）。
- **H=1 任务平衡**：persistent 24 核下 logical tasks = ceildiv(S,bm)·H·B，wall-rows = ceil(ceildiv(S,bm)/24)·bm——bm=44 使 S=1024/2048/4096 得 44/88/176 行（理想 42.7/85.3/170.7），bm=22 使 S=512 得 24 任务 1:1。
- **终局数据（dispatch 路径，msprof op Task Duration，median of 20）**：fa512 18.20 / fa1024 29.58 / fa2048 77.53 / fa4096 263.77 µs（vs 基线 56.62/101.81/375.22/1101.88 = -67.9%/-71.0%/-79.3%/-76.1%；有效吞吐 7.4/18.2/27.7/32.6 TFLOPS）。跨谱系对照（developer 简单 tiling 谱系调优终值 16.1/33.6/105.2/300.3µs，2026-09-07，`examples/multi_head_attention/_prev_task_20260907_developer_optimize/`——任务工作区，未上库，provenance 允许失效；对照结论已自包含于本条与 PL-1.7）：fa1024/2048/4096 超 12%/26%/12%，fa512 落后 13%——单块链 + 固定入口开销地板（17.9–18.2µs 平区，M7-clean〔clean 覆盖 nk==1 退化链〕-5.6% 后仍贴地板）。clean 特化同时使 non-causal 整除 dispatch 的 vcmp 标量化警告因果贡献归零（traps-compiler.md Expert v 算子操作数行的「optimize 入口线索」在该 dispatch 类闭环；causal/full trace 仍在）。

---
id: PL-1.9-hardlimits
kind: constant
family: [attention, expert]
apis: [T.vexp, T.vsub, T.vmul, T.reduce, T.vcast, sync_block_wait]
dtype: [fp16, fp32]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z
toolchain: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 / Ascend910B2C / 2026-09-09
repro: repro-missing
---

### 第二轮硬目标续调实测补充（2026-09-09，同 origin_task 谱系续）

- **f16 softmax 链（有效，四 case -3.5%~-14.1%）**：clean trace 的 [half,bn] 大 pass 全 f16（vexp/vsub/vmul/reduce 文档均支持 fp16），vcast-in/out 两 pass 消失；**running max 以 f16 表示**（f16 值上的 max 精确、f16→f32 vcast 无损）经 f32 无损镜像供 alpha/ell/lse 算术——直接 f32→f16 镜像 m 会引入系统性 lse 平移（经 ell 的 e^-δ 缩放，实测 1.4e-3 量级）；f16 ell 求和的绝对间距是残余 lse 误差主导项（ell~200 时 ~1e-3，「有效容差 atol 1e-3+rtol 1e-3×|lse|」内有 10x 裕量）。终值 17.55/27.24/69.13/226.72µs。
- **L0C(cc)=128KB 实测上限**（BishengIR `cc overflow, requires 1310720 bits while 1048576 bits available` 反推）：l0c_s[bm,bn]f32 + l0c_o[bm,dim]f32 ≤ 128KB ⇒ bn=512 时 bm≤51——与 L1 512KB、UB 192KB 并列为 expert 迁移三硬上限，共同封死「大 bm 减 KV 重读」再平衡路线。
- **L1 端口 r+w ≈ 148–154 GB/s/核**（探针复刻 KV+gemm1 数据流 + 真实 kernel 反推双证）：ping-pong 双槽**不能**解除 MTE2（GM→L1 写）与 MTE1（L1→L0 读）的端口读写串行——Cube 操作数流的吞吐地板；persistent 结构的 KV 重读流量（tasks/核 × 2MB × 2）÷ 此率 = 该结构的算子级下界（fa4096 ≈ 125–131µs）。
- **fabric 聚合带宽 ≥1.73TB/s**（pp 探针 24 核 × 72GB/s，shared=distinct=pp 无争用惩罚）：L2/HBM fabric 在 fa4096 现聚合 1.41TB/s 下未触顶——该 workload 的地板因子排序为 L1 端口 / 向量发射率先于 fabric。
- **向量算子发射开销定律**：逐元素速率 f16≈f32（无 f16 打包加速），每 op ~0.5µs 固定发射成本主导——同元素总量下**宽块 op（bn=512）恒优于窄块 op 翻倍（bn=256）**：深流水重构（defer-2 + KV 预取先于 P-wait 入队）虽解锁 Cube（aic_scalar 50.7→40.5%）但 bn=256 使 vec-active 90→180µs，净回退 +12%；bn=512 深流水需 K/V 双槽 567KB>L1 封死。「块宽摊减」上轮经验律的机制解释。
- **`sync_block_wait` 阻塞后续发射流的实测行为**（ns=1≈ns=2、wide≈serial、深流水在槽受限时无效的综合解释）：flag wait 不是窄管道门控——流水化只有在 wait 到达时必然已满足（深 deferral）才有效，而深 deferral 与 L1 槽容量冲突；online 递推的 alpha/ellcur 跨 deferral 需 ping-pong（v-op codegen 拒绝多维 UB 切片操作数，平铺多 buffer + 运行时三分支是合法形态）。
- **[DESIGN_LIMIT] 完整档案**：`perf_opt/perf_feedback.md` + opt_log 第二轮 round 8–10（硬上限测绘→f16 链→深流水回退的证据链）；attention 族 H=1 大 S 目标的复合地板 = store_fixpipe 仅 GM（S/P 往返强制）+ L1 端口 + 向量发射率，非 tiling 可解。**〔第三轮修正〕**该地板系冻结设计族（per-block 链 + bm=44）局部地板——两相位重构实测四目标全部达成（见 PL-1.9-twophase），perf_feedback.md 修正附录已回填。

---
id: PL-1.9-twophase
kind: pattern
family: [attention, expert]
apis: [T.gemm, T.reduce, T.copy, T.transpose]
dtype: [fp16, f16]
device: 910B2C
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z
toolchain: tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68 / CANN 8.5.0 / Ascend910B2C / 2026-09-09
repro: repro/PATT-twophase-restructure.py
---

### 第三轮（2026-09-09）：两相位重构达标终值

- 参考结构（`examples/flash_attention/flash_attn_npuir.py`）迁移：非持久 grid ceildiv(S,bm=96)、Cube pass-1(全 S)/pass-2(全 O_partial) 独立串行大循环（flag=n-block 下标，nk≤16）、单 L0C [96,256] 由 gemm1/gemm2 顺序复用、K/V/P/Q L1 生命周期复用（112KB）、Vector f32 softmax 链 + UB `scales[]` 延迟 rescale 回放、O_partial f16、**Q hoist**（免参考的每 n-block 重读）、**ell 正确递推**（`T.reduce` 必带 `clear=True`——参考无 clear 的 sum 系文档化静默错误形态，docs/Tilelang.language/规约操作/T.reduce.md §2.3）、**transpose-free lse**（traps-compiler.md 毒化条目的绕法，[B,H,S,1] 视图 + wrapped reshape）。
- 终值 **12.94/18.11/30.88/98.05µs**（四硬目标 14/19/33/100 全达成，fa1024/2048/4096 优于参考 18.20/32.41/99.09）；每核 L1 r+w 18.70→11.17MB（-40%）、每核 cube 管道 busy 171→66µs；管道 busy/wall 0.66x 与参考（0.68x）等价——加速来自流量削减而非重叠度。
- **morph 阶梯**（参考实现上逐级叠加本方契约增量的 14 点二分）是本轮定位上下文级差异（transpose 毒化）的决定性方法。bm 上探死点：(128,256) 105.16 / (112,256) 101.84 > (96,256) 98.38（双波装载失衡 + L0C 满配）。
- 溯源：`examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/opt_log.md`（rounds 1–11）+ 同目录 `_gqa_prefill_fwd_kernel.py` + `perf_records.jsonl`（round 8–10 = 第二轮，round 11 = 第三轮两相位达标）+ `perf_feedback.md`（[DESIGN_LIMIT] 五段档案）；tilelang 0.1.2+3a214cde7aa4f54fc4a103f0f324a43341122d68（三轮同 commit）/ CANN 8.5.0 / Ascend910B2C / 2026-09-08～09；origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z（第一轮 Stage 4）+ multi_head_attention-_gqa_prefill_fwd_kernel-20260909T033622Z（第二轮硬目标续调，产出「第二轮硬目标续调实测补充」小节）+ multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z（第三轮两相位重构，产出本小节，另产出 traps-compiler.md 活跃源 transpose 毒化行——2026-09-09 蒸馏溯源归位，原缺第三轮 task_id）；复现：结构关键更改与机制归因见 `repro/PATT-twophase-restructure.py`（知识域 delta 形态）；端到端测量（provenance，任务工作区允许失效）：dispatch 路径 `msprof op --kernel-name=_gqa_prefill_fwd_main_mix_aic --launch-count=20 --warm-up=5`（median of 20）跑 bench `--use-default-config`（工厂 TUNED_DEFAULT_CONFIGS 生效），raw 对账 `profiles/round8|9|10|11|final2|final3/`。
