# 调优复盘与模式沉淀

## 用途

在算子性能调优完成后读取本文件。目标是复盘本次调优过程是否暴露出 skill 流程问题，并判断是否需要提出新的 `BP_xxx` 瓶颈模式。

本文件只用于生成复盘记录和改进建议，不自动修改 `SKILL.md`、`iteration-diagnosis.md`、`profile-collection.md`、`bottleneck-patterns.md` 或 `autotune.md`。只有用户明确要求时，才把建议合入 skill 文档。

---

## 输入

复盘前必须具备：

- `perf_opt/opt_log.md`
- final `perf_opt/{op}.py`
- baseline 与 final 的 `msprof op` 数据，以及触发调度结构分析时的 NPU event 数据
- 所有实验分支的 `improved / config_no_gain / family_no_gain / invalid / blocked / defer` 记录
- 本次最终 `stop_reason`

如果 `perf_opt/opt_log.md` 缺少关键实验记录，先补齐日志，再做复盘。

---

## Step 1：复盘 Skill 流程

从 `perf_opt/opt_log.md` 回看本次调优过程，检查 skill 流程是否有问题。

重点检查：

- 性能采集是否覆盖了必测 dispatch。
- profile 口径是否清楚，是否误采到框架小算子。
- event 与 `msprof Task Duration` 是否明显背离，若背离是否进入诊断。
- event 是否存在平区、session 漂移或 anchor 异常；是否先解决测量分辨力再排序。
- 当前现象分析是否足够解释候选优化点。
- 候选优化点是否遗漏明显方向。
- 实验分支是否都从同一个 current best 派生。
- 是否有分支一次混入多个主要优化点，导致无法归因。
- 是否把单个 `config_no_gain` 错误扩大成 `family_no_gain`。
- `T.serial / num_cores / pipeline` 等结构候选是否完成足够配置覆盖；`num_cores` 是否分析了任务并发不足、甜点区、派发开销过高和整除性。
- `bottleneck-patterns.md` 是否缺少某类现象或动作。
- `autotune.md` 的触发时机是否正确。
- stop_reason 是否有 profile 和实验记录支撑。
- 日志模板是否足够复现 winner 选择。

复盘结论必须绑定本次 workload、profile 和实验记录，不要把单次偶然现象泛化成通用规则。

---

## Step 2：提取新的 BP_xxx 候选

只有满足以下条件，才提出新的 `BP_xxx`：

- 它解释了本次真实 profile 现象。
- 现有 [bottleneck-patterns.md](bottleneck-patterns.md) 不能很好覆盖。
- 它能对应明确的优化动作。
- 它有可验证指标。
- 它不是单个算子的偶然实现细节。

如果只是对现有模式的补充，优先提出“更新现有 BP_xxx”的建议，不要强行新增模式。

---

## Step 3：写入 opt_log.md

把复盘写入 `perf_opt/opt_log.md` 的 `Skill Retrospective` 章节。

最小模板：

```markdown
## Skill Retrospective

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change |
|---|---|---|---|
| none | none | none | none |

### BP Proposals

| proposal_id | type | source_case | summary | target_doc |
|---|---|---|---|---|
| none | none | none | none | none |
```

字段说明：

- `area`：`Profile-collection / Iteration-diagnosis / Bottleneck-patterns / Autotune / SKILL / logging / stop_condition`
- `issue`：本次暴露的问题，没有则写 `none`
- `evidence`：来自 `opt_log.md`、profile 或实验分支的证据
- `suggested_doc_change`：建议修改哪个文档，以及怎么改
- `proposal_id`：新的或待更新的 `BP_xxx`
- `type`：`new_bp / update_bp / none`
- `source_case`：触发该建议的 operator、dispatch、workload 或实验分支
- `summary`：模式或更新建议的一句话说明
- `target_doc`：通常为 `bottleneck-patterns.md`

---

## Step 4：最终报告

最终报告必须包含：

```markdown
- skill_retrospective: {none_or_summary}
- bp_proposals: {none_or_list}
```

如果提出了 `BP_xxx`，只作为 proposal 返回，不直接修改 skill 文档。
