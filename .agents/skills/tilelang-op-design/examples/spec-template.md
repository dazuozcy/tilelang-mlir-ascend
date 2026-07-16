# {算子名称} 算子规格说明书 (SPEC)

> **Spec-Driven 声明**：本文件是算子开发的唯一事实源。`IC/FC/AC/CC` 为权威契约（检视通过后冻结，实现必须符合，测试必须覆盖且可追溯）；`ID` 为推荐实现方案（可在契约边界内偏离）；`RR/DC` 为参考。每条契约带唯一条款 ID（如 `IC-1`），下游实现与测试通过 ID 回溯。
>
> **冻结状态**：`未冻结` / `已冻结（hash: ..., at: ...）`（由 conductor 在 Stage 2 通过后填写）

## 0. 迁移上下文（如果不是迁移类任务，则不用填写本章节）

### 0.1 功能概述

### 0.2 输入输出

### 0.3 详细解读

### 0.4 标杆实现

---

## 1. 接口契约 (Interface Contract, IC)

> 权威契约。实现必须严格符合；Stage 3/4 的 kernel 入口签名以此为准。

### IC-1 算子名称

{算子名称}

### IC-2 输入张量规格

| 参数名 | Shape | dtype | 动态轴 | 说明 |
|--------|-------|-------|--------|------|
| {A} | {(M, N)} | {float16} | {如 B} | {输入矩阵} |
| ... | ... | ... | ... | ... |

### IC-3 输出张量规格

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| {C} | {(M, N)} | {float16} | {输出矩阵} |
| ... | ... | ... | ... |

（迁移类任务：标注原算子输出 shape，说明是否需要 transpose）

### IC-4 JIT 配置

```python
@tilelang.jit(
    out_idx=[{输出索引}],
    target="npuir",
)
```

### IC-5 Kernel 入口签名

```python
@T.prim_func
def {op}_kernel(
    A: T.Tensor((M, N), in_dtype),
    C: T.Tensor((M, N), out_dtype),
):
    ...
```

> 若迁移前 TIR 原语函数签名中 `T.Tensor` 通过关键字参数 `dtype=` 传递数据类型，改为将数据类型直接作为第二个位置参数传入（避免 `name 'in_dtype' is not defined` 的 false alarm）。此修改不违反"迁移执行规则"第 3 条。

---

## 2. 功能契约 (Functional Contract, FC)

> 权威契约。定义算子**做什么**（数学语义），不限定**怎么做**。

### FC-1 数学公式

$$
{数学公式}
$$

### FC-2 算法分解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | {子表达式} | {说明} |
| 2 | {子表达式} | {说明} |
| ... | ... | ... |

（多步算子描述计算步骤分解逻辑；单步算子可省略。Host 预处理（如 im2col）在此显式声明。）

### FC-3 数据流图

```
输入张量 → [计算步骤1] → [计算步骤2] → ... → 输出张量
```

---

## 3. 验收契约 (Acceptance Contract, AC)

> 权威契约。Stage 3 测试必须覆盖每条 AC 条款且逐条可追溯（测试用例标注验证的 AC ID）；Stage 4 调优不得使 AC 退化。

### AC-1 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float16 | 1e-2 | 1e-2 |
| float32 | 1e-4 | 1e-4 |

### AC-2 测试契约 — L0（规则 shape，block 整除）

- 代表性规则 shape（block 整除），如 `(128,128)`、`(256,256)`
- dtype 覆盖：{float16 / float32}
- 通过标准：`assert_close` 满足 `AC-1`

### AC-3 测试契约 — L1（非整除 / 中等规模）

- 非整除 shape（如 `(130,130)`、`(64,200)`），验证 `ID-3` 尾块处理
- 通过标准：`assert_close` 满足 `AC-1`

### AC-4 测试契约 — L2（异常 / 极小，告警不阻塞）

- 极小 shape（如 `(1,1)`）
- 失败仅 `WARN` 记录到 `debug_log.md`，不阻塞 `[PRECISION_PASS]`

### AC-5 测试契约 — Boundary（特殊值）

- 特殊值（zeros / 极大 / 极小 / NaN 输入处理）
- 失败仅 `WARN` 记录，不阻塞

### AC-6 Golden 参考实现（PyTorch CPU）

```python
def golden_{算子名}({参数}):
    """PyTorch CPU 参考实现，可在 CPU 独立运行（不依赖 torch_npu）"""
    {参考实现代码}
```

> 完整分层测试套件（`run_L0/run_L1/run_L2/run_boundary` + main 入口）由 `tilelang-op-develop` skill 生成，每个用例标注其验证的 AC 条款 ID。

---

## 4. 约束契约 (Constraint Contract, CC)

> 权威契约。定义算子必须遵守的平台与设计约束。

### CC-1 计算类型

**类型**: {纯 Vector / 纯 Cube / 混合}

**判定依据**: {如: 算子仅包含 element-wise 运算，无 matmul，判定为纯 Vector}

### CC-2 编程模式

**模式**: {Developer / Expert / 混合}

**选型理由**: {基于算子特征（计算类型、是否含 matmul、是否含归约、是否需要流水线）的分析}

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | {如: T.alloc_ub 显式指定 UB} |
| 计算方式 | {如: T.Parallel + 运算符} |

### CC-3 NPU 硬件约束

**⚠️ 必查**（违反会导致编译错误或运行时错误）：

1. **分形限制**：L0A M≥16,K≥32；L0B K≥32,N≥16；L0C M≥16,N≥16
2. **对齐要求**：UB/L1 32B；L0A/L0B 512B；L0C 64B
3. **存储上限**：L0A/L0B 64KB；L0C 128KB；L1 512KB；UB 192KB（A2/A3）

本算子约束核算：{说明本算子涉及的约束}

### CC-4 技术约束检测结论

| 检测项 | 本算子是否涉及 | 处理方案 |
|--------|---------------|----------|
| 三维 Kernel | {Yes/No} | {改成一维 block_metadata 方案 / 不涉及} |
| GPU 专用 API | {Yes/No} | {查阅本项目 Ascend API / 不涉及} |
| GEMM 非整除 | {Yes/No} | {padding+crop / 动态 block / 不涉及} |
| L0C 溢出风险 | {Yes/No} | {block_M×block_N 核算 / 不涉及} |

> 详细限制清单与强制检测规则见 [ascend-constraints.md](../references/ascend-constraints.md)。

### CC-5 内存预算

| Buffer | Shape | dtype | 存储层级 | 大小 (Bytes) |
|--------|-------|-------|----------|-------------|
| {a_ub} | {(128, 128)} | {float16} | {UB} | {32768} |
| ... | ... | ... | ... | ... |
| **总计** | | | | {总字节数} / {目标平台容量，如 196608 (192KB, A2/A3)} |

（Cube 类须额外核算 L0C：`block_M × block_N × sizeof(accum) ≤ 128KB`）

---

## 5. 实现设计 (Implementation Design, ID)

> 推荐实现方案。Stage 3 可在 `IC/FC/AC/CC` 契约边界内偏离 `ID` 而不触发规格修订；仅当偏离将违反契约或契约本身不可行时返回 `[SPEC_ERROR]`。

### ID-1 API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | {子表达式} | {如: T.vexp(dst, src)} | {参数说明} | {Developer/Expert} |
| 2 | {子表达式} | {如: T.reduce_sum(buf, out, dim=-1)} | {参数说明} | {Developer/Expert} |
| ... | ... | ... | ... | ... |

**API 可行性确认**：{列出使用的 API 及其来源确认（examples / docs / 源码），标注是否经过验证}

**v-prefix 优先**：新设计优先 v-prefix API（vadd/vmul/vexp/vcast/vbrc），npuir_xxx 仅作兼容说明。

### ID-2 内存层级规划与搬运路径

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| {a_ub} | {(block_M, block_N)} | {float16} | {UB} | {输入 tile 缓冲} |
| ... | ... | ... | ... | ... |

```
{完整的数据搬运路径图}

示例（纯 Vector）:
GM[A] --T.copy--> UB[a_ub] --计算--> UB[c_ub] --T.copy--> GM[C]

示例（Cube + Vector）:
GM[A] --T.copy--> L1[a_l1] --T.copy--> L0A[a_l0a]
GM[B] --T.copy--> L1[b_l1] --T.copy--> L0B[b_l0b]
L0A + L0B --T.gemm--> L0C[c_l0c] --T.copy--> UB[c_ub]
UB[c_ub] --后处理--> UB[c_ub] --T.copy--> GM[C]
```

### ID-3 Tiling 策略

```python
block_M = {值}  # {选择理由}
block_N = {值}  # {选择理由}
block_num = (M // block_M) * (N // block_N)
```

**约束分析**：
- **对齐约束**: {如: block_N=128, fp16 尾轴 128 > 16 ✓}
- **UB 容量**: {如: 总 buffer = 64KB < 192KB ✓}
- **L0 容量**: {如: 无 Cube 计算，不适用}

**非整除处理**（GEMM 类必填，纯 Vector 若不整除也需说明）：
{host 侧 padding+crop 或 Kernel 内动态 block 方案，说明溢出 / 下溢处理}

### ID-4 循环与调度结构

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| {M 方向} | {block 级并行} | {T.Kernel} | {每个 block 处理一个 M 分块} |
| {K 方向} | {迭代} | {T.serial(K // block_K)} | {K 维分块迭代累加} |
| {元素级} | {向量化} | {T.Parallel(block_M, block_N)} | {block 内逐元素并行} |

```python
# Block 级并行（隐式，由 T.Kernel 管理）
with T.Kernel(block_num, is_npu=True) as (cid, _):
    {block 内循环结构}
```

**流水线优化**：{是否使用 T.Pipelined？如使用，说明 num_stages 设计和 buffer 管理策略}

**尾块处理**：{当输入 shape 不能被 block size 整除时的处理策略}

### ID-5 同步策略

**模式**: {自动同步 / 手动同步 / 混合}

| 位置 | 同步 API | 理由 |
|------|----------|------|
| {搬入后} | {T.barrier_all()} | {等待 DMA 搬运完成} |
| ... | ... | ... |

**pass_configs 配置**：
```python
pass_configs = {
    {与同步相关的 pass 配置}
}
```

> 同步策略须与 `CC-2` 编程模式匹配（Developer 用自动同步；Expert 标明手动同步点）。

### ID-6 CV 融合设计（混合算子填写）

{Cube + Vector 融合的核间流水线、sync_block_set/wait、跨核 workspace 协同。详见 mixcv skill。纯 Vector / 纯 Cube 算子填"不适用"。}

---

## 6. 风险登记册 (Risk Register, RR)

### RR-1 已知约束

{列出本算子在 TileLang-NPUIR 上的已知限制}

### RR-2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| {UB 溢出} | {block 过大} | {编译失败} | {减小 block size} |
| ... | ... | ... | ... |

### RR-3 特殊场景处理

{如: 非整除分块、极小 shape、混合精度等}

---

## 7. 交付清单 (Delivery Checklist, DC)

### DC-1 目录结构

```
examples/{项目名}/{算子名}/
├── {算子名}.py             # 算子实现 + AC 追溯分层测试
├── SPEC.md                 # 本规格说明书（冻结后为唯一事实源）
└── README.md               # 使用说明（可选）
```

> 项目名由 conductor 从用户提示词中解析；解析不出时与算子名相同。同一项目目录下可含多个算子子目录。

### DC-2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `SPEC.md` | {已完成} | 契约化规格 |
| `{算子名}.py` | {待实现} | 算子实现（Stage 3） |
| `test_{算子名}.py` | {待实现} | 测试文件（可选，放入 testing/） |

### DC-3 命名规范

- 项目目录名: `{项目名}`（snake_case，无明确项目名时与算子名相同）
- 算子目录名: `{算子名}`（snake_case）
- 实现文件: `{算子名}.py`
- 测试文件: `test_{算子名}.py`

### DC-4 实现顺序

1. ✅ 规格说明书（SPEC.md，含 IC/FC/AC/CC/ID/RR/DC 契约）
2. ⬜ Golden 函数（`AC-6`，验证基准）
3. ⬜ 算子实现（`{算子名}.py`，符合 `IC-*`、覆盖 `AC-*`） + 与 Golden 函数精度比对
