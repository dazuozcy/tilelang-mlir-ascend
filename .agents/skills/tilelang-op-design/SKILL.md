---
name: tilelang-op-design
description: "根据算子需求生成 TileLang-NPUIR 契约化规格说明书（SPEC.md），以可验证契约组织：IC 接口契约 / FC 功能契约 / AC 验收契约 / CC 约束契约 / ID 实现设计 / RR 风险登记 / DC 交付清单，每条带条款 ID 供下游追溯。触发：制定算子规格、生成 SPEC.md、算子方案设计、新算子开发、算子实现方案。"
---

# TileLang-NPUIR 算子规格说明书生成（Spec-Driven）

## 1. 目标

根据算子需求信息，生成一份**契约化规格说明书** `SPEC.md`，作为下游所有阶段的**唯一事实源**。`SPEC.md` 以可验证契约组织，每条契约带唯一条款 ID，定义算子**做什么**（接口、行为、验收标准、约束），并附**实现设计**（推荐方案）：

- **接口契约 IC**：算子名、I/O 张量规格（shape/dtype/动态轴）、JIT 配置、Kernel 入口签名
- **功能契约 FC**：数学公式、算法分解、数据流
- **验收契约 AC**：精度标准（dtype→atol/rtol）、分层测试契约 L0/L1/L2/Boundary、Golden 参考实现
- **约束契约 CC**：计算类型、编程模式（Developer/Expert/混合）、NPU 硬件约束、内存预算、技术约束检测结论
- **实现设计 ID**：API 映射、内存层级规划、Tiling 策略、循环结构、同步策略、CV 融合
- **风险登记 RR**：已知约束、常见错误、特殊场景
- **交付清单 DC**：目录结构、文件清单、命名、实现顺序

> `IC/FC/AC/CC` 为**权威契约**（检视通过后冻结，实现必须符合，测试必须覆盖且可追溯）；`ID` 为**推荐实现方案**（Stage 3 可在契约边界内偏离）。`SPEC.md` 整体在 Stage 2 检视通过后由 conductor 冻结。

---

## 2. 输入要求

### 必需信息

| 字段 | 说明 | 归属契约 |
|------|------|---------|
| **项目名称** | 项目分组名，决定 `examples/{project}/` 项目目录；由 conductor 解析，无明确项目名时与算子名相同 | DC |
| 算子名称 | 如 `softmax`、`layer_norm`、`flash_attention`，决定 `examples/{project}/{op}/` 算子目录及 `{op}.py` 文件名 | IC-1 |
| 数学公式 | 算子的数学表达，如 $\text{softmax}(x_i) = e^{x_i} / \sum e^{x_j}$ | FC-1 |
| 输入张量规格 | shape、dtype | IC-2 |
| 输出张量规格 | shape、dtype | IC-3 |
| 编程模式偏好 | Developer / Expert / 混合 | CC-2 |
| **迁移算子路径** ⭐ | 原算子文件路径（迁移时必需），用于分析原始实现及实现 golden 函数 | FC/AC（迁移分析） |
| **输出形状** ⭐ | 原算子输出 shape（迁移时必需），如 `(N, M)` 或 `(M, N)` | IC-3（迁移校验） |

**迁移算子时必须提供原算子路径和输出形状**，否则无法证明迁移正确性。Golden 实现一致性要求详见 [tilelang-op-develop SKILL.md](../tilelang-op-develop/SKILL.md)。


**提问规则（必须严格遵守）**：
1. **优先使用调用方传入的字段**：若调用方（如 `@tilelang-op-conductor` 通过 designer 传入 `op_requirements` 结构）已经提供了字段值，**全部跳过提问**，直接进入约束检测和 SPEC.md 生成
2. **每次只询问一个字段**：使用 `question` 工具时，`questions` 数组中只包含一个元素
3. **按表格顺序依次询问**：算子名称 → 数学公式 → 输入张量规格 → 输出张量规格 → 编程模式偏好
4. **已提供的字段跳过**：如果用户在初始请求中已提供某个字段的值，跳过该字段继续下一个
5. **示例**：
   - 第 1 次询问：只问"数学公式"
   - 用户回答后，第 2 次询问：只问"输入张量规格"
   - 以此类推

**⚠️ 当被 conductor → designer Subagent 链路调度时**：
- designer 会把 conductor 在 Primary 上下文预检收集到的 `op_requirements` 完整传入
- 此时 5 个必需字段应当全部已 provided，跳过整个提问环节
- 若 skill 仍发现字段歧义或缺漏，**不要**在当前 Subagent 上下文调用 `AskUserQuestion`（透传不到真实用户），而是让 designer 返回 `partial_input` + 缺失字段名给 conductor，由 conductor 在 Primary 上下文追问

### 推荐信息

| 字段 | 说明 | 归属契约 |
|------|------|---------|
| 典型配置 | 常用的 shape 组合与优先级 | AC |
| 参考实现 | PyTorch / NumPy 参考代码 | FC/AC-6 |
| 性能目标 | 目标吞吐量或延迟 | AC（性能目标附录） |
| 动态轴说明 | 哪些维度在运行时变化 | IC-2 |

若用户未提供**必需信息**中的任一项，通过提问补全后再继续。

---

## 3. 技术约束（必须遵守，写入 CC-*）

本项目为 TileLang-NPUIR （后端为华为昇腾 NPU），与 GPU 版 TileLang 有显著差异。外部参考实现不可直接使用，必须转换为 Ascend 兼容方案。

**生成 SPEC.md 前必须执行强制检测**：三维 Kernel、GPU 专用 API、GEMM 非整除、L0C 溢出等。检测结果写入约束契约 `CC-4`（技术约束检测结论）。

详细已知限制清单、强制检测规则、警告输出模板见 [references/ascend-constraints.md](references/ascend-constraints.md)。

---

## 4. 工作流程

### Phase 1：输入解析与算子特征分析

1. 解析算子名称与数学公式
2. 验证必需字段是否完整
3. 分析算子特征（写入 `CC-1` 计算类型）：
   - **计算类型判定**：
     - 纯 Vector（element-wise / reduction）→ 仅需 UB
     - 纯 Cube（仅 matmul）→ 需要 L1 + L0A/L0B/L0C
     - 混合（matmul + element-wise 后处理）→ 核间流水线，需要 CV 融合
      - **Host 预处理**：如 im2col 等 Python 侧预处理步骤，标明在 `FC-2` 和 `ID-2` 中
   - **复杂度级别**：
     - 单步（如 element-wise add）→ 无循环、单次搬运
     - 多步（如 softmax = max + sub + exp + sum + div）→ 多次计算、可能需要中间缓冲
     - 融合（如 flash attention = GEMM + softmax + GEMM）→ 核间协作、流水线
   - **动态 shape 判定**：是否存在运行时才确定的维度
4. **非整除场景预判**（写入 `ID-3` / `CC-4`）：检查输入 shape 是否可能不被 block size 整除。GEMM 类算子的 `M // block_M` 和 `N // block_N` 在 `M < block_M` 或 `N < block_N` 时产生零 block 或不完整 tile，必须在 `ID-3` 中明确处理策略（host 侧 zero-padding + crop，或 Kernel 内动态 block size）

### Phase 2：信息收集

**必须执行强制步骤 0：搜索本项目同类实现**。详细工具调用、信息收集步骤、禁止行为见 [references/info-sources.md](references/info-sources.md)。

### Phase 3：生成 SPEC.md

基于 [examples/spec-template.md](examples/spec-template.md) 模板，填充全部契约章节并赋予条款 ID：

0. 迁移上下文（迁移类任务填写）
1. 接口契约 IC（IC-1..IC-5）
2. 功能契约 FC（FC-1..FC-3）
3. 验收契约 AC（AC-1..AC-6）
4. 约束契约 CC（CC-1..CC-5）
5. 实现设计 ID（ID-1..ID-6）
6. 风险登记 RR（RR-1..RR-3）
7. 交付清单 DC

### Phase 4：契约质量自检

按照 [references/quality-checklist.md](references/quality-checklist.md) 中的契约自检清单逐项检查，确保每条契约完备、一致、可验证。

### Phase 5：针对性修订

仅修正未通过自检的契约条款。信息确实不足的标注为「待确认」并说明原因（待确认项不得为权威契约 `IC/FC/AC/CC` 的必需条款）。

### Phase 6：输出

将 `SPEC.md` 输出到 `examples/{project}/{op}/` 算子目录（`{project}` 为项目名称、`{op}` 为算子名称，均由调用方传入；无明确项目名时与算子名相同）。若文件已存在，询问是否覆盖。

---

## 5. 算子特征分析决策树

详细决策树（Ascend 版）、平台识别、API 映射规则、NPU 硬件约束（分形限制 / 对齐要求 / 存储大小上限）见 [references/decision-tree.md](references/decision-tree.md)。决策结论分别落入 `CC-1`（计算类型）、`CC-2`（编程模式）、`ID-1`（API 映射）。

---

## 6. 信息源优先级

信息源优先级表与冲突处理原则见 [references/info-sources.md](references/info-sources.md)。

---

## 7. 错误处理

| 场景 | 处理方式 |
|------|----------|
| 用户未提供数学公式 | 提问补全，给出常见算子公式作为参考 |
| 必需字段缺失 | 列出缺失项，逐一提问 |
| API 查询无结果 | 在 `ID-1` 标注为「需扩展」，在 `RR-2` 风险点中说明 |
| 目标文件已存在 | 询问用户是否覆盖或另存 |
| 算子过于复杂 | 建议拆分为多个子算子分别制定规格 |
| revision 模式输入含不可行条款 ID | 针对该条款产出 delta，明确新方案如何规避 |

---

## 8. 完成报告

文档生成完成后，按 [examples/completion-report-template.md](examples/completion-report-template.md) 输出报告（含契约条款清单）。

---

## 9. 生成算子

完成报告后，询问用户是否根据此规格生成对应算子代码（由 Stage 3 执行）。

---

## 子目录索引

- [examples/spec-template.md](examples/spec-template.md) — SPEC.md 契约化完整模板
- [examples/completion-report-template.md](examples/completion-report-template.md) — 完成报告模板
- [references/ascend-constraints.md](references/ascend-constraints.md) — 技术约束清单（CC-3/CC-4/CC-5 依据）
- [references/decision-tree.md](references/decision-tree.md) — 算子特征决策树（CC-1/CC-2/ID-1 依据）
- [references/info-sources.md](references/info-sources.md) — 信息源优先级
- [references/quality-checklist.md](references/quality-checklist.md) — 契约质量自检清单
