# 编译器能力缺口登记簿（Capability Gaps）

> 编译器层面自进化机制的登记簿。与 skill 层自进化（`queue.md` / pattern-library）平行：skill 层沉淀「如何在现有能力下绕」，本簿登记「npuir 缺什么能力」并驱动补齐。
>
> - **用途**：任务过程中识别出「当前 npuir 能力不足以实现所选算法」（无法实现 / 被迫换算法 / 触及性能上限）时，登记具体缺失能力、证据与绕法，跟踪复发次数与补齐状态。
> - **写入者**：任务各阶段 Agent（design 调研与 API 映射 / develop / optimize / review）在得出能力不足结论时**直接追加条目或为既有条目计数**；`tilelang-skill-evolver` 在任务终态蒸馏时查重、冲突消解与升级裁决（与 queue.md 的维护分工同口径）。
> - **证据规则**：与 AGENTS.md「Docs Auto Routing / Negative claims」同口径——须引用具体 `docs/` 路径与限制条款、`testing/`/`examples/` 佐证或复现命令；无法佐证时显式标注「未文档化假设 + 估算依据」。禁止凭记忆或 GPU 先验断言能力缺失。
> - **版本戳**：一切「能力缺失」结论绑定工具链版本（tilelang commit + CANN/设备版本）；工具链升级后相关条目自动待重验，不得沿用旧结论否决新设计。
> - **升级阈值**：同一缺口被 **≥2 个不同任务**独立识别 → `recurring`，**必须推动补齐**：产出能力补齐提案（`proposal` 字段），并在任务终态报告中向用户显式列出（与 Tier 1 的 2 次独立证据阈值同口径）。

## 字段速查

```yaml
gap_id: CG-{YYYY}-{NNNN}           # 唯一 ID，追加时按现有最大号递增
layer: Frontend API / TileLangIR pass / BishengIR / tladapter / runtime / codegen
capability: <缺失能力的具体描述：缺什么 API / 优化 / dtype / 布局 / 同步原语 / 性能上限，越具体越好>
blocked_algo: <受阻的算法/算法族与结论形态：无法实现 / 被迫换算法（换成什么）/ 性能上限（差多少）>
evidence:                          # 证据链，规则同负向断言
  - <docs 路径#限制条款 / 复现命令 / 源码位置 / pattern-library 条目 / 实测数据>
workaround: <当前任务的实际绕法及其代价量化（如"两遍算法替代单遍，访存 +1.8x"），或 none（被迫弃用）>
occurrences: {n}                   # 独立任务识别次数（同一任务反复命中只计 1）
tasks: [<task_id>, ...]
toolchain_stamp: <tilelang commit + CANN 版本 + 设备，或 latest-rebuild>
status: open / recurring / proposal / in-progress / fixed / closed / withdrawn
proposal: <达 recurring 后必填：目标层 + 建议改动（API 签名 / pass 行为）+ 收益量化（引用各任务实测）+ 受影响算子清单 + issue 草稿（[npuir] 前缀）>
created_by: <task_id + 日期>
last_seen: <最近一次识别的任务与日期>
```

## 生命周期

```
open ──(≥2 独立任务识别)──> recurring ──(提案产出)──> proposal ──> in-progress ──> fixed ──> closed
  │                                                                            │
  └── withdrawn（证据被证伪/复核不成立）            fixed 后曾绕行的任务回访评估去绕化 ┘
```

- **追加即计数**：新识别的缺口先查重（本簿 + pattern-library §2），命中既有条目则 `occurrences+1` 并更新 `last_seen`/`tasks`，不新建重复条目。
- **推动补齐的形式**：提案正文落在条目 `proposal` 字段；对外建 issue（标题带 `[npuir]` 前缀，流程见 tilelang-github-operations skill）**前须用户批准**。
- **fixed 判定**：新工具链实测确认能力可用（附版本戳与验证记录）；fixed 后记录过 workaround 的任务可回访评估去绕化收益。
- **与 pattern-library 的边界**：绕法经验（怎么绕、绕的代价、陷阱）按 D 类进 pattern-library §1/§2；本簿只登记缺口本体与补齐跟踪，两处以 `gap_id` / 条目路径互相引用，不复制内容。

---

## Open

（暂无条目）

---

## Recurring（达升级阈值，待推动补齐）

（暂无条目）

---

## Resolved（proposal / in-progress / fixed / closed / withdrawn 归档）

（暂无条目）
