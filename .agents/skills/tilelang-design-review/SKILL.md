---
name: tilelang-design-review
description: "对 Stage 1 产出的契约化规格说明书（SPEC.md）进行风险优先的契约检视，生成 REVIEW.md。校验 IC/FC/AC/CC/ID 契约的完备性、一致性、可验证性，必须给出明确结论（通过/不通过）与不通过时针对条款 ID 的具体修改建议。触发：review 规格、检视 SPEC、生成 review.md。"
---

# TileLang-NPUIR 算子规格检视（Spec-Driven）

## 1. 目标

对 Stage 1 产出的 `SPEC.md` 进行**风险优先的契约检视**，生成一份 `REVIEW.md`。检视聚焦契约的**完备性、一致性、可验证性**，`REVIEW.md` 必须包含明确的 `结论: 通过` 或 `结论: 不通过`，以及不通过时**针对条款 ID** 的具体修改建议，供 conductor 决定是否冻结 SPEC 并进入 Stage 3，或回退 Stage 1 修订规格。

> 检视通过是 conductor 冻结 `SPEC.md` 的前置条件。

---

## 2. 输入

| 字段 | 说明 |
|------|------|
| `spec_md_path` | 待检视的 `SPEC.md` 路径（由 conductor 传入） |
| 算子目录 | `examples/{project}/{op}/`，用于核对同类实现引用是否真实存在 |

检视前必须用 Read 完整读取 `SPEC.md`，并用 Glob 核对其中引用的 `examples/` 文件是否真实存在。

---

## 3. 检视维度（风险优先，按顺序）

> 优先识别会直接导致 Stage 3 编译/运行/精度失败的**阻塞级**问题，其次识别**建议级**问题。每个问题必须定位到具体契约条款 ID（`IC/FC/AC/CC/ID-*`）。

### 维度 1：接口契约 IC 可行性（阻塞级）

| 检查项 | 通过标准 | 不通过示例 | 定位条款 |
|--------|----------|-----------|----------|
| I/O 规格完整 | `IC-2/IC-3` shape+dtype+动态轴完整 | 输入缺 dtype | IC-2/IC-3 |
| Kernel 入口签名 | `IC-5` 签名与 I/O 规格一致 | 签名参数与 IC-2 不符 | IC-5 |
| JIT 配置 | `IC-4` out_idx/target 正确 | target 非 npuir | IC-4 |

### 维度 2：API 可行性（阻塞级）

| 检查项 | 通过标准 | 定位条款 |
|--------|----------|----------|
| API 存在性 | `ID-1` 列出的每条 TileLang DSL API 能在 `examples/` 或 `tilelang/language/` 中找到使用佐证 | ID-1 |
| API 与模式匹配 | `ID-1` API 与 `CC-2` 编程模式匹配（Developer 自动同步；Expert 手动同步） | ID-1 / CC-2 |
| v-prefix 优先 | 新设计优先 v-prefix API（vadd/vmul/vexp/vcast/vbrc），npuir_xxx 仅作兼容说明 | ID-1 |

### 维度 3：内存层级规划（阻塞级）

| 检查项 | 通过标准 | 定位条款 |
|--------|----------|----------|
| 搬运路径完整 | `ID-2` 给出完整 GM → L1/UB → L0 路径，无 GM→L0 直连等越级路径 | ID-2 |
| UB 预算 | `CC-5` 中间缓冲区总和不超过目标平台 UB 容量（A2/A3 = 192KB） | CC-5 |
| L0C 容量 | Cube 类 `block_M × block_N × sizeof(accum)` ≤ 128KB | CC-5 |
| Buffer 层级标注 | `ID-2` 每个 buffer 标注存储层级（UB/L1/L0A/L0B/L0C） | ID-2 |

### 维度 4：Tiling 策略（阻塞级）

| 检查项 | 通过标准 | 定位条款 |
|--------|----------|----------|
| Block 划分明确 | `ID-3` 给出 block_M/block_N 具体值及理由 | ID-3 |
| 对齐约束 | `CC-3` 尾轴对齐满足（fp16 ≥ 16B 等） | CC-3 |
| 非整除处理 | GEMM 类 `ID-3` 必须含 padding+crop 或动态 block 策略；纯 Vector 若 shape 不被 block 整除也需说明 | ID-3 |

### 维度 5：技术约束检测（阻塞级）

| 检查项 | 通过标准 | 定位条款 |
|--------|----------|----------|
| 三维 Kernel | `CC-4` 若涉及，必须给出改成一维 block_metadata 方案 | CC-4 |
| GEMM 非整除 | 已在维度 4 处理 | CC-4 / ID-3 |
| L0C 溢出 | 已在维度 3 处理 | CC-5 |

### 维度 6：循环与同步（建议级）

| 检查项 | 通过标准 | 定位条款 |
|--------|----------|----------|
| 循环结构 | `ID-4` 明确 T.Parallel / T.serial / T.Pipelined / T.Persistent 选择 | ID-4 |
| 同步与模式匹配 | `ID-5` 同步策略与 `CC-2` 编程模式一致 | ID-5 / CC-2 |
| 尾块处理 | `ID-4` 说明 shape 不整除时的尾块逻辑 | ID-4 |

### 维度 7：验收契约可验证性（阻塞级）

| 检查项 | 通过标准 | 定位条款 |
|--------|----------|----------|
| Golden 函数 | `AC-6` 含 PyTorch 参考实现草案，可独立运行 | AC-6 |
| L0 测试契约 | `AC-2` 含具体规则 shape（block 整除）、dtype、通过标准 | AC-2 |
| 精度标准 | `AC-1` 按 dtype 给出 atol/rtol（fp16 默认 1e-2，fp32 默认 1e-4） | AC-1 |
| AC 可追溯 | 每条 `AC-*` 可被 Stage 3 测试逐条执行与追溯（语义清晰、无歧义） | AC-2..AC-5 |

### 维度 8：完整性与一致性（建议级）

| 检查项 | 通过标准 | 定位条款 |
|--------|----------|----------|
| 契约章节齐全 | `IC/FC/AC/CC/ID/RR/DC` 齐全（迁移类含 §0） | 全部 |
| 条款 ID 完整 | 每条契约带唯一条款 ID，无遗漏 | 全部 |
| 无占位符 | 不含 `{placeholder}`、`TODO`、`待补充` | 全部 |
| 同类引用真实 | `ID-1`/`CC` 引用的 examples/ 路径真实存在 | ID-1 |
| 内部一致 | IC/FC/AC/CC/ID 间无矛盾（API 映射、内存规划、Tiling 三处描述相互一致） | 全部 |

---

## 4. 工作流程

### Phase 1：读取与核对
1. Read `SPEC.md` 全文。
2. Glob 核对 `ID-1`/`CC` 引用的 `examples/` 路径是否存在。
3. 必要时 Grep `tilelang/language/` 确认 API 是否有导出佐证（仅静态文本核对，不执行）。

### Phase 2：逐维度契约检视
按 §3 的 8 个维度逐项检查，每项标记 `pass / warn / fail` 并记录证据（契约条款 ID + 引用文件）。

### Phase 3：结论判定

| 条件 | 结论 |
|------|------|
| 所有**阻塞级**维度均为 pass（warn 不阻塞） | `结论: 通过` |
| 任一**阻塞级**维度为 fail | `结论: 不通过`，必须给出针对条款 ID 的具体修改建议 |

### Phase 4：生成 REVIEW.md
按 §5 模板写入 `examples/{project}/{op}/REVIEW.md`。

---

## 5. REVIEW.md 模板

```markdown
# {op} 规格检视报告

## 检视结论

结论: 通过
<!-- 或：结论: 不通过 -->

## 检视元信息
- 文档: examples/{project}/{op}/SPEC.md
- 检视维度: 8 项（IC 可行性 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验收可验证性 / 完整一致性）
- 阻塞级问题数: {N}
- 建议级问题数: {N}
- 涉及条款: {条款 ID 列表，如 IC-5, ID-1, CC-5}

## 检视详情

### 1. 接口契约 IC 可行性 — pass / warn / fail
{说明与证据（条款 ID）}

### 2. API 可行性 — pass / warn / fail
{说明与证据（条款 ID）}

### 3. 内存层级规划 — pass / warn / fail
{说明与证据（条款 ID）}

### 4. Tiling 策略 — pass / warn / fail
{说明与证据（条款 ID）}

### 5. 技术约束检测 — pass / warn / fail
{说明与证据（条款 ID）}

### 6. 循环与同步 — pass / warn / fail
{说明与证据（条款 ID）}

### 7. 验收契约可验证性 — pass / warn / fail
{说明与证据（条款 ID）}

### 8. 完整性与一致性 — pass / warn / fail
{说明与证据（条款 ID）}

## 检视问题列表
<!-- 仅当「结论: 不通过」时出现本章节；通过时删除本章节 -->

### 问题 1: {问题标题}
- 严重程度: 阻塞 / 建议
- 条款 ID: {如 ID-1 / CC-5}
- 位置: SPEC.md §{X.Y} / {条款 ID}
- 问题描述: {具体描述}
- 修改建议: {可执行的修改建议，指明条款 ID + 修改方向，供 Stage 1 revision 使用}

### 问题 2: ...
```

---

## 6. 输出规则

1. `REVIEW.md` 必须写入 `examples/{project}/{op}/REVIEW.md`。
2. **结论行必须是字面量** `结论: 通过` 或 `结论: 不通过`（conductor 据此判定冻结/修订路由），不得用"基本通过"、"建议通过"等模糊表述。
3. 不通过时，每个阻塞级问题必须给出**可执行的修改建议**（指明条款 ID + 修改方向），作为 Stage 1 `revision` 模式的 `spec_error_summary` 输入。
4. 通过时不得出现"检视问题列表"章节。
5. 不得修改 `SPEC.md`。

---

## 7. 错误处理

| 场景 | 处理 |
|------|------|
| `SPEC.md` 不存在或为空 | 结论: 不通过，问题：文档缺失 |
| 契约章节大面积缺失 | 结论: 不通过，逐项列出缺失契约（条款 ID） |
| 引用的 examples 路径不存在 | 标记 fail（维度 8），建议 Stage 1 修正引用或换真实参考（条款 ID-1） |
| API 无法在源码中找到佐证 | 标记 warn（若 examples 有用法）或 fail（若全无佐证）（条款 ID-1） |
| AC 条款不可追溯/不可验证 | 标记 fail（维度 7）（条款 AC-*） |
