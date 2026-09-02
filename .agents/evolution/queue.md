# 进化提案队列（Evolution Queue）

> TileLang-Op-Conductor 自进化机制的提案队列。由 `tilelang-skill-evolver` 维护（唯一写入者）。
>
> - **用途**：暂存未达合入阈值的 P 类（模式方法）提案与待人工审批的 R 类（流程规则）diff 提案；D/C 类（实测数据/案例索引）不进本队列，由 evolver 直接合入 pattern-library。
> - **字段规范**：见 `.agents/skills/tilelang-skill-evolution/references/queue-schema.md`（本文件头部仅列速查）。
> - **状态**：`pending`（待确认/待审批）→ `verified`（Tier 1 证据达阈值待合入）→ `merged`；或 `rejected` / `expired`（90 天未决）/ `conflict`（与现有条目矛盾且无法自动裁决，留人工）。
> - **审阅方式**：Tier 2（R 类）提案经用户批准后由 conductor 调度 evolver（`mode=apply`）执行写入；Tier 1 达阈值后由 evolver 在下次蒸馏时合入。

## 字段速查

```yaml
proposal_id: VP-{YYYY}-{NNNN}     # 唯一 ID，由 evolver 递增分配
type: D / P / R / C               # D 实测数据 / P 模式方法 / R 流程规则 / C 案例索引
title: <一句话标题>
evidence:                         # 证据链（Tier 1/2 必填；D 类直接合入时也须具备）
  - <工件路径#定位>
repro: <复现命令或 none>
toolchain_stamp: <tilelang build/commit + 设备 + CANN 版本，或 none>
target_doc: <目标文件路径>
delta: <五种合法动作之一：add / update / consolidate / negate / deprecate + 条目正文>
status: pending / verified / merged / rejected / expired / conflict
confirmations: {n}/{2}            # Tier 1 合入阈值：2 次独立证据（须来自不同任务）
created_by: <task_id + 日期>
decided_by: evolver / human / -   # 裁决者
```

---

## Pending

（暂无条目）

---

## Decided（merged / rejected / expired / conflict 归档）

（暂无条目）
