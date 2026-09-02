# 进化统计（Evolution Stats）

> 由 `tilelang-skill-evolver` 在每次蒸馏（distill 模式）后更新。用于度量自进化机制的有效性：条目是否被读到、同类错误是否复发。

## 1. 任务蒸馏记录

> 每次蒸馏任务追加一行。`vp_count` 按 D/P/R/C 分列计数（候选数，非合入数）。

| date | task_id | scenario | final_phase | vp_count (D/P/R/C) | merged | enqueued | verdict |
|------|---------|----------|-------------|--------------------|--------|----------|---------|
| — | — | — | — | — | — | — | — |

## 2. 条目命中统计

> 统计 pattern-library / bottleneck-patterns 条目被后续任务引用的情况。
>
> **计数方式**（由 evolver 在蒸馏时增量更新，不要求全量重扫）：
> - **主动命中**：任务工件（DESIGN.md / opt_log.md / REVIEW.md / RETROSPECTIVE.md）中显式引用了 `pattern-library §x.x` / `BP-xxx` 条目——每个任务每条目计 1 次。
> - **被动命中**：条目位于某 skill 的强制读取路径（design 强制步骤 0.5 / optimize Phase 0 / develop Phase 1 / conductor 带记忆重试注入）且该任务执行过对应阶段——不计入（只统计主动引用）。
>
> **治理规则**：某条目累计主动命中为 0 且超过 2 个蒸馏周期 → consolidate 候选；同类错误在已有对应条目的情况下复发（复盘章节可识别）→ 检索注入点缺失，优先补「失败触发读取」而非新增条目。

| entry | hits | last_hit_task | note |
|-------|------|---------------|------|
| — | — | — | — |

## 3. 系统指标

> 北极星指标。原始数据来自各任务 `.stage_state.json` 与 RETROSPECTIVE.md；由 evolver 记录原始数据，趋势分析可人工或后续工具化。

| 指标 | 定义 | 当前值 |
|------|------|--------|
| first_pass_rate | Stage 3 `attempt=1` 即 `[PRECISION_PASS]` 的任务占比（长期应随模式库增长而上升） | 暂无数据 |
| 同类错误复发率 | 已有 D/P 条目对应的失败模式在后续任务中复发占比（复发=检索注入失效） | 暂无数据 |
| design_revision_avg | 任务平均设计修订次数（`retry_count` 均值） | 暂无数据 |

## 4. 变更日志

| date | by | change |
|------|-----|--------|
| 2026-09-01 | 初始化（conductor-self-evolution-design 落地） | 创建骨架 |
