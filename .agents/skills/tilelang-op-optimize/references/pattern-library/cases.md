# 案例索引与参考实现集（同类问题的正/反例参考）

> 本文件是 pattern-library 主题文件之一（入口与预算见 [INDEX.md](INDEX.md)）。条目带 front-matter（schema 见 INDEX.md §3）。
>
> **用途**：按"哪类问题该参考哪个算子目录"组织 `examples/` 的检索入口。**目标未上库的条目标注「任务工作区，未上库」并给出 durable 载体**（该经验的持久形态 = pattern-library 自包含条目 / repro）——工作区目录随时可能被清理，条目知识不随之失效（ED-A）。Stage 1 设计（强制步骤 0.5）与 optimize skill Phase 0 均会读取本文件——命中时优先精读对应算子目录，优先于盲目 Glob。**维护**：由 `tilelang-skill-evolver` 在任务终态追加（C 类，Tier 0）；条目 = 路径 + 一句话 + 适用触发条件，不复制内容；路径失效或长期零引用 → deprecate。
>
> **反例档案**（BLOCKED 终态任务的根因链 + 能力缺口互链）见本文件末节——capability-gaps（`.agents/evolution/capability-gaps.md`）的 open 条目实质是"该形态当前做不了"的反例，以指针行互链（只加指针不复制内容，边界按该簿 §与 pattern-library 的边界）。

---
id: CASE-pool-maxpool3d-tileops
kind: case
family: [pool]
mode: [developer]
dtype: []
status: verified
origin_task: max_pool3d-harness-migration-2026-08
toolchain: 2026-08
repro: none
---

### `examples/TileOPs/tileops/kernels/pool/max_pool3d/max_pool3d_kernel/`（任务工作区，未上库）

**durable 载体**：本文件无自包含池化条目时见 layout.md PL-1.1/1.2/1.4（模式与实测数字自包含）；集成包结构形态见已上库的 mish/logsumexp 同构目录。

TileOPs 集成包形态：kernel + `{func}_DESIGN.md` 快照 + 聚合 `__init__.py` + integration_log。适用触发条件：migration-harness 集成结构参考；Stage 5 集成问题排查。

---
id: CASE-elementwise-mish-opt
kind: case
family: [elementwise]
mode: [developer]
dtype: [fp16, fp32]
status: verified
origin_task: mish-optimize-2026-08
toolchain: 2026-08（详见其 opt_log 版本戳）
repro: none
---

### `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/perf_opt/`

elementwise cast 类完整调优档案（persistent 甜点、multi-buffer 教训、Expert 流水平台约束、run 双态协议）。适用触发条件：elementwise 调优前必读先例；<3% 结论多 run 协议。

---
id: CASE-reduction-logsumexp
kind: case
family: [reduction]
mode: [developer]
dtype: []
status: verified
origin_task: logsumexp-migration-2026-08
toolchain: 2026-08
repro: none
---

### `examples/TileOPs/tileops/kernels/reduction/logsumexp/_logsumexp_kernel_single/`

规约类（logsumexp）独立算子目录：DESIGN.md + kernel + 分层测试。适用触发条件：规约类（水平归约/online 单遍）设计与 L0 测试计划参考。

---
id: CASE-elementwise-lerp-opt
kind: case
family: [elementwise]
mode: [developer]
dtype: [fp16, bf16, fp32]
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T025419Z
toolchain: tilelang 0.1.2+ed787bb（2026-09-07 build）/ Ascend910B2C / CANN 8.5.0
repro: none
---

### `examples/TileOPs/tileops/kernels/elementwise/lerp_tensor/lerp_tensor_kernel/perf_opt/`

elementwise 3 输入 persistent 调优：copy-floor 探针标定法 + MTE2 带宽退化曲线 + 交错 A/B 采纳复核 + grid-stride vs 连续分块反例。适用触发条件：判定"已贴内存地板"时的方法学参考；多输入 elementwise 调优。

---
id: CASE-elementwise-lerp-precision
kind: case
family: [elementwise]
mode: [developer]
dtype: [fp16, bf16]
status: verified
origin_task: lerp_tensor-_make_lerp_tensor_kernel-20260907T010433Z
toolchain: tilelang-mlir-dev dev root build（2026-09-07）；golden torch 2.9.0+cpu
repro: none
---

### `examples/lerp_tensor/_make_lerp_tensor_kernel/`（任务工作区，未上库）

**durable 载体**：traps-runtime.md TRAP-fp16-opmath-golden（现象/对齐通解/数字自包含）。

fp16 精度失败定征与修复完整档案：attempt-1 原生域失败基线（`history_version/`）+ R2 fp32 中转修复（≤1 ulp，全量 62 PASS）+ kernel 头部 Implementation Notes 等价性论证。适用触发条件：fp16/bf16 逐元素迁移精度失败（golden opmath 域分歧）定征与修复参考；vcast 中转路径 UB 混合字节预算复算案例。

---
id: CASE-pool-maxpool3d-standalone
kind: case
family: [pool]
mode: [developer]
dtype: []
status: verified
origin_task: maxpool3d-2026-08
toolchain: 2026-08
repro: none
---

### `examples/maxpool3d/_max_pool3d_kernel/`（任务工作区，未上库）

**durable 载体**：layout.md PL-1.1/1.2/1.4（池化类模式与实测数字自包含）。

standalone 形态的池化算子目录（DESIGN.md + kernel）。适用触发条件：窗口/池化类设计参考（分核策略、边界处理）。

---
id: CASE-attention-gqa-expert-full
kind: case
family: [attention]
mode: [expert]
dtype: [fp16, bf16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: tilelang dev root build 2026-09-07（21586b5）+ CANN 8.5.0 + Ascend910B2C
repro: none
---

### `examples/multi_head_attention/_gqa_prefill_fwd_kernel/`（任务工作区，未上库）

**durable 载体**：attention.md PL-1.8（bf16 直连）、traps-compiler/runtime.md attention 族陷阱条目（正文自包含）。

Expert 模式 GQA/MHA prefill attention 完整档案：v0→v1→v2 设计修订链（K_A ceildiv→floordiv 差 1 反例 + 检视建议自我纠正全程 + per-lane 阈值符号错误）+ 3 轮 REVIEW + debug_log D1–D6（load_nd2nz 误读 / T.copy 语义 / 解析器规则 / tier-3 精度分类）+ bf16 直连 + E7 守卫不变式。适用触发条件：attention 族 Expert 迁移/重设计；跨引擎 flag 流水协议设计；bf16 Cube 直连决策；分界公式检视反例。

---
id: CASE-attention-expert-stage4
kind: case
family: [attention]
mode: [expert]
dtype: [fp16, f16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z + 20260909T033622Z + 20260909T071018Z
toolchain: tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-08～09（三轮）
repro: none
---

### `examples/TileOPs/tileops/kernels/attention/multi_head_attention/multi_head_attention_kernel/perf_opt/`（任务工作区，未上库）

**durable 载体**：attention.md PL-1.9-blockwidth/hardlimits/twophase（三轮结论自包含）+ `repro/PATT-twophase-restructure.py` + constants.md 硬件常数表。

attention 族 expert Stage 4 完整调优档案：块宽摊减调优序（bn 64→512）、cbuf=512KB 实测、wide 单槽 L1 模式、clean trace 特化、S-f16 传输裁决、G=2 备选 A 裁决、H=1 任务平衡公式、工厂内 tuned 分派表（S4-5 模式）；第二轮硬上限测绘（L0C 128KB / L1 端口 ~148–154GB/s/核 / 向量发射 ~0.5µs/op / fabric ≥1.73TB/s）+ f16 softmax 链（终值 17.55/27.24/69.13/226.72µs）+ [DESIGN_LIMIT] 五段式反馈档案（perf_feedback.md，硬目标 2.09x/2.27x 缺口归因）；**第三轮两相位重构达标**（参考实现 flash_attn_npuir.py 结构迁移 + transpose-free lse，终值 12.94/18.11/30.88/98.05µs 四硬目标全达成、fa1024/2048/4096 优于参考、三轮累计 vs Stage-3 基线 -77.2%/-82.2%/-91.8%/-91.1%）+ [DESIGN_LIMIT] 负结论被同 API 结构推翻的完整闭环档案（perf_feedback.md 修正附录：局部地板 vs 绝对地板界定 + 两处根因 + morph 阶梯 14 点矩阵）。适用触发条件：attention/persistent 双 Scope 类调优前必读；跨引擎流水结构选型（块宽 vs 流水深度）；备选 A/B 裁决执行范例；设计层天花板判定与 [DESIGN_LIMIT] 归因/证据组织参考；**结构不可达类负结论的修正闭环参考**。

---
id: CASE-attention-developer-stage4
kind: case
family: [attention]
mode: [developer]
dtype: [fp16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260908T005751Z
toolchain: tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-08
repro: none
---

### `examples/multi_head_attention/_prev_task_20260907_developer_optimize/perf_opt/`（任务工作区，未上库）

**durable 载体**：attention.md PL-1.7（expert vs developer 同门对照结论自包含）。

developer 谱系（简单 tiling）同 4 case 调优档案：S4-1 mask 消除 + S4-2/3/4 死缓冲/in-place/退化 trace-time 特化 + S4-5 工厂内 shape 分派（终值 16.1/33.6/105.2/300.3µs）。适用触发条件：与 expert 谱系（attention.md PL-1.9 系列）做同门调优杠杆对照（mask 消除/config 扫描 vs 块宽摊减/L1 上限突破）；编程模式/谱系选型参考。

---
id: CASE-deepseek-v4-highperf
kind: case
family: [attention, expert]
mode: [expert]
dtype: [fp16, bf16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260907T115424Z
toolchain: 2026-09-07
repro: none
---

### `examples/deepseek_v4/example_sparse_attn_kernel_highperf.py`

Expert 模式 attention/跨引擎结构先例：双 Scope + GM workspace 多槽 + per-slot flag + staggered stream + 运行时 if + int16 索引矩阵 vcmp mask 链（L345–354）+ PIPE_V 内 T.arange strides（L611）。适用触发条件：Expert 模式 kernel 结构设计/先例检索（Stage 1 设计起步）；vcmp 标量规则与 flag 协议先例核对。

---
id: CASE-ref-flash-attn-npuir
kind: case
family: [attention]
mode: [developer]
dtype: [fp16]
status: verified
origin_task: multi_head_attention-_gqa_prefill_fwd_kernel-20260909T071018Z（第三轮实证）
toolchain: tilelang 0.1.2+3a214cde / CANN 8.5.0 / Ascend910B2C / 2026-09-09
repro: none
---

### 参考实现集：`examples/flash_attention/flash_attn_npuir.py`

**参考实现集首条**（T-1 参照锚定）。仓库内同 API 可达结构的参照锚点：两相位（pass-1 全 S / pass-2 全 O_partial）+ 非持久 grid + 单 L0C 顺序复用 + Q hoist + transpose-free lse——第三轮 [DESIGN_LIMIT] 负结论正是被该结构推翻（四硬目标全达成 12.94/18.11/30.88/98.05µs）。适用触发条件：attention 族调优 plateau / 拟产出 `[DESIGN_LIMIT]` 时的**强制参照对照**（perf-feedback.md §1 参照锚定门槛）；两相位结构迁移的母本。

---
id: CASE-norm-adalayern-migration
kind: case
family: [norm]
mode: [developer]
dtype: [fp32, fp16, bf16]
status: verified
origin_task: ada_layer_norm-_ada_layer_norm_kernel-20260910T132324Z
toolchain: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
repro: none
---

### `examples/ada_layer_norm/_ada_layer_norm_kernel/`（任务工作区，未上库）

**durable 载体**：vrsqrt 陷阱与绕法自包含于 traps-runtime.md TRAP-vrsqrt-plain-precision（含知识域 repro）；迁移决策与双判据验证见该目录 DESIGN.md/verify_equiv.py（溯源，允许失效）。

norm 族首个迁移档案（Developer 模式单函数，零设计修订零重试一次过）：vrsqrt 近似指令精度链定位（TRAP-vrsqrt-plain-precision，分层探针单次运行定位）+ GPU 256-pad 方差校正整体舍弃（恒等式机器验证 0 违反，精确 N 式在小 N 大均值时数值更优）+ verify_equiv 45 案例等价性验证设计（矩式方差 rejection evidence）。适用触发条件：layernorm/rmsnorm/AdaLN-Zero gated 族迁移设计参考（wrapper 已保留 has_gate 分支，gated 变体复用同一集成路径）；Stage 3 精度失败含 rsqrt 段时优先对照。

---
id: CASE-norm-adalayern-stage4
kind: case
family: [norm, reduction]
mode: [developer]
dtype: [fp16, bf16, fp32]
status: verified
origin_task: ada_layer_norm-_ada_layer_norm_kernel-20260910T145715Z
toolchain: tilelang 0.1.2+a83118285a + Ascend910B2C + CANN 8.5.0 / 2026-09-10
repro: none
---

### `examples/TileOPs/tileops/kernels/norm/ada_layer_norm/ada_layer_norm_kernel/perf_opt/`（任务工作区，未上库）

**durable 载体**：staging 解耦模式自包含于 elementwise.md PL-1.10 + repro/PL-1.10-loads-first-decoupling.py（知识域）；调优数字溯源 opt_log.md（允许失效）。

norm/row-reduction 族完整调优档案（6 轮，全组几何平均 2.14x）：wrapper 部署态 bm=1 → per-shape 调优表（第一杠杆）+ per-input staging 解耦（PL-1.10-loads-first-decoupling）+ bm 任务几何精搜（bm=7 非单调局部最优）+ 调优表条目 U 曲线闭合验证 + A/B 交错协议两次防误判 + 保护性泛化验证（「UB 85% 松弛律」单点否定）。适用触发条件：norm/规约族 Stage 4 调优；多输入 staging 复用链串行化诊断（pipe 忙时和 ≈ wall 信号）；调优表条目鲁棒性验证方法参考。

---
id: CASE-CG-INDEX
kind: case
family: [general]
mode: [developer, expert]
dtype: []
status: verified
origin_task: conductor-improvement-2026-09-10
toolchain: 登记于 2026-09-10；各条目版本戳见 .agents/evolution/capability-gaps.md
repro: none
---

### 反例档案（能力缺口互链）

capability-gaps 登记簿（`.agents/evolution/capability-gaps.md`）open 条目指针——"该形态当前做不了"的反例，调优/设计遇同族负向结论时先查：

| gap_id | 层 | 一句话触发条件 |
|--------|-----|---------------|
| CG-2026-0001 | codegen (Developer mode) | Developer 形态 codegen 能力缺口 |
| CG-2026-0002 | BishengIR (auto-multi-buffer) | auto-multi-buffer 相关缺口 |
| CG-2026-0003 | TileLangIR pass (Expert fixpipe lowering) | Expert fixpipe lowering 与 BishengIR 版本配对缺口 |
| CG-2026-0004 | tladapter/codegen (load_nd2nz 区域描述符下推) | load_nd2nz 跨步区域静默误读（绕法见 traps-runtime.md TRAP-load-nd2nz-strided） |
| CG-2026-0005 | Frontend API / codegen | 前端 API 能力缺口 |
| CG-2026-0006 | TileLangIR pass / codegen（向量算子融合与 f16 打包发射） | 向量算子融合缺口（f16≈f32 发射速率的机制根源，见 attention.md PL-1.9-hardlimits） |
| CG-2026-0007 | runtime / codegen（跨引擎同步原语粒度） | 跨引擎同步原语粒度缺口 |

> BLOCKED 终态任务的反例根因链条目同入本节（由 evolver 追加，标注「反例」）。
