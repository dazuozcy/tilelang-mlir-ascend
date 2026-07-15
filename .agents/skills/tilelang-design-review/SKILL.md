---
name: tilelang-design-review
description: "对 Stage 1 产出的算子设计文档（DESIGN.md）进行风险优先的检视，生成 REVIEW.md，必须给出明确结论（通过/不通过）与不通过时的具体修改建议。触发：review 设计文档、检视设计、生成 review.md。"
---

# TileLang-NPUIR 算子设计文档检视

## 1. 目标

对 Stage 1 产出的 `DESIGN.md` 进行**风险优先**的检视，生成一份 `REVIEW.md`。`REVIEW.md` 必须包含明确的 `结论: 通过` 或 `结论: 不通过`，以及不通过时的具体修改建议，供 conductor 决定是否进入 Stage 3 或回退 Stage 1 重新设计。

> 本 skill 为纯文档分析，**不涉及 NPU 执行**，无需打桩。

---

## 2. 输入

| 字段 | 说明 |
|------|------|
| `design_md_path` | 待检视的 `DESIGN.md` 路径（由 conductor 传入） |
| 算子目录 | `examples/{op}/`，用于核对同类实现引用是否真实存在 |

检视前必须用 Read 完整读取 `DESIGN.md`，并用 Glob 核对其中引用的 `examples/` 文件是否真实存在。

---

## 3. 检视维度（风险优先，按顺序）

> 优先识别会直接导致 Stage 3 编译/运行/精度失败的**阻塞级**问题，其次识别**建议级**问题。

### 维度 1：API 可行性（阻塞级）

| 检查项 | 通过标准 | 不通过示例 |
|--------|----------|-----------|
| API 存在性 | DESIGN.md §3.2 列出的每条 TileLang DSL API 能在 `examples/` 或 `tilelang/language/` 中找到使用佐证 | 设计用了 `T.flash_attention` 但项目无此 API |
| API 与模式匹配 | Developer 模式用自动同步 API；Expert 模式才用手动 set_flag/wait_flag | Developer 模式却写手动同步 |
| v-prefix 优先 | 新设计优先 v-prefix API（vadd/vmul/vexp/vcast/vbrc），npuir_xxx 仅作兼容说明 | — |

### 维度 2：内存层级规划（阻塞级）

| 检查项 | 通过标准 |
|--------|----------|
| 搬运路径完整 | §4.4 给出完整 GM → L1/UB → L0 路径，无 GM→L0 直连等越级路径 |
| UB 预算 | §4.5 中间缓冲区总和不超过目标平台 UB 容量（A2/A3 = 192KB） |
| L0C 容量 | Cube 类算子 `block_M × block_N × sizeof(accum)` ≤ 128KB |
| Buffer 层级标注 | 每个 buffer 标注存储层级（UB/L1/L0A/L0B/L0C） |

### 维度 3：Tiling 策略（阻塞级）

| 检查项 | 通过标准 |
|--------|----------|
| Block 划分明确 | §5.2 给出 block_M/block_N 具体值及理由 |
| 对齐约束 | 尾轴对齐满足（fp16 ≥ 16B 等） |
| 非整除处理 | GEMM 类必须含 padding+crop 或动态 block 策略；纯 Vector 若 shape 不被 block 整除也需说明 |

### 维度 4：技术约束检测（阻塞级）

| 检查项 | 通过标准 |
|--------|----------|
| 三维 Kernel | 若涉及，必须给出 block_metadata 方案 |
| GEMM 非整除 | 已在 Tiling 维度处理 |
| L0C 溢出 | 已在内存层级维度处理 |

### 维度 5：循环与同步（建议级）

| 检查项 | 通过标准 |
|--------|----------|
| 循环结构 | §6 明确 T.Parallel / T.serial / T.Pipelined / T.Persistent 选择 |
| 同步与模式匹配 | §7 同步策略与编程模式一致 |
| 尾块处理 | §6.4 说明 shape 不整除时的尾块逻辑 |

### 维度 6：验证方案（阻塞级）

| 检查项 | 通过标准 |
|--------|----------|
| Golden 函数 | §8.1 含 PyTorch 参考实现草案 |
| L0 测试计划 | §8 含 L0 门槛测试计划：具体规则 shape（block 整除）、dtype、按算子类别的精度标准（atol/rtol） |
| 精度标准 | 按 dtype 给出 atol/rtol（fp16 默认 1e-2，fp32 默认 1e-4） |

### 维度 7：完整性与一致性（建议级）

| 检查项 | 通过标准 |
|--------|----------|
| 章节齐全 | 模板 11 章节齐全（迁移类含 §0） |
| 无占位符 | 不含 `{placeholder}`、`TODO`、`待补充` |
| 同类引用真实 | §3.5.3 引用的 examples/ 路径真实存在 |
| 内部一致 | API 映射、内存规划、Tiling 三处描述相互一致，无矛盾 |

---

## 4. 工作流程

### Phase 1：读取与核对
1. Read `DESIGN.md` 全文。
2. Glob 核对 §3.5.3 引用的 `examples/` 路径是否存在。
3. 必要时 Grep `tilelang/language/` 确认 API 是否有导出佐证（仅静态文本核对，不执行）。

### Phase 2：逐维度检视
按 §3 的 7 个维度逐项检查，每项标记 `pass / warn / fail` 并记录证据（DESIGN.md 章节号 + 引用文件）。

### Phase 3：结论判定

| 条件 | 结论 |
|------|------|
| 所有**阻塞级**维度均为 pass（warn 不阻塞） | `结论: 通过` |
| 任一**阻塞级**维度为 fail | `结论: 不通过`，必须给出具体修改建议 |

### Phase 4：生成 REVIEW.md
按 §5 模板写入 `examples/{op}/REVIEW.md`。

---

## 5. REVIEW.md 模板

```markdown
# {op} 设计文档检视报告

## 检视结论

结论: 通过
<!-- 或：结论: 不通过 -->

## 检视元信息
- 文档: examples/{op}/DESIGN.md
- 检视维度: 7 项（API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性）
- 阻塞级问题数: {N}
- 建议级问题数: {N}

## 检视详情

### 1. API 可行性 — pass / warn / fail
{说明与证据}

### 2. 内存层级规划 — pass / warn / fail
{说明与证据}

### 3. Tiling 策略 — pass / warn / fail
{说明与证据}

### 4. 技术约束检测 — pass / warn / fail
{说明与证据}

### 5. 循环与同步 — pass / warn / fail
{说明与证据}

### 6. 验证方案 — pass / warn / fail
{说明与证据}

### 7. 完整性与一致性 — pass / warn / fail
{说明与证据}

## 检视问题列表
<!-- 仅当「结论: 不通过」时出现本章节；通过时删除本章节 -->

### 问题 1: {问题标题}
- 严重程度: 阻塞 / 建议
- 位置: DESIGN.md §{X.Y}
- 问题描述: {具体描述}
- 修改建议: {可执行的修改建议，供 Stage 1 revision 使用}

### 问题 2: ...
```

---

## 6. 输出规则

1. `REVIEW.md` 必须写入 `examples/{op}/REVIEW.md`。
2. **结论行必须是字面量** `结论: 通过` 或 `结论: 不通过`（conductor 据此判定路由），不得用"基本通过"、"建议通过"等模糊表述。
3. 不通过时，每个阻塞级问题必须给出**可执行的修改建议**，作为 Stage 1 `revision` 模式的 `design_error_summary` 输入。
4. 通过时不得出现"检视问题列表"章节。
5. 不得修改 `DESIGN.md`。

---

## 7. 错误处理

| 场景 | 处理 |
|------|------|
| `DESIGN.md` 不存在或为空 | 结论: 不通过，问题：文档缺失 |
| 章节大面积缺失 | 结论: 不通过，逐项列出缺失章节 |
| 引用的 examples 路径不存在 | 标记 fail（维度 7），建议 Stage 1 修正引用或换真实参考 |
| API 无法在源码中找到佐证 | 标记 warn（若 examples 有用法）或 fail（若全无佐证） |
