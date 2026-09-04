# 性能采集流程

本文档服务 `tilelang-op-optimize` 的 Stage 4 性能采集。目标是用最小流程覆盖算子中的不同 dispatch path，并为每个 path 采集一个可诊断的 `msprof op` 基准数据。`msprof op` 是唯一 kernel 时延测量方式；不采集 NPU event 或端到端口径。

一句话规则：

> 从 `{op}.py` 中找出真实存在的不同 dispatch path；每个 dispatch path 选择一个能触发该分支的代表测试数据；逐个串行运行 `msprof op`，记录每个 path 的 `Task Duration(us)` 和 raw profile 目录。

---

## 1. 找 dispatch path

读取 Stage 3 产物：

- `examples/{project}/{op}/{op}.py`
- `examples/{project}/{op}/DESIGN.md`

从 `{op}.py` 中识别真实代码分支，例如：

- dtype 分支：fp16 / fp32。
- fast path / fallback path。
- axis、flag、layout 等参数触发的不同实现。
- config 分支
   block_m、tile_n、block_size、num_cores 等参数是否导致 kernel 内不同 if/else。

不要把普通 shape 大小差异当作 dispatch path，除非它确实触发了不同代码分支。

如果没有显式分支，记录一个默认分支：

```text
dispatch_path = default
```

每个 dispatch path 必须绑定一个目标 kernel 名：

```text
target_kernel_name = @T.prim_func 对应的函数名
```

如果同一个 `{op}.py` 中有多个 TileLang kernel，按 dispatch path 分别记录对应的 `target_kernel_name`。如果无法确定目标 kernel 名，停止采集并记录缺失原因。

---

## 2. 每个 dispatch 选一个测试数据

每个 dispatch path 默认只选择一个代表 workload。

选择顺序：

1. 优先用性能目标或 conductor 明确指定的 case。
2. 其次用 `{op}.py` 中已通过的 L0/L1/L2/Boundary case。
3. 再从 `{op}.py` 的 main/level/case 入口中选择。
4. 如果已有 case 不能触发某个 dispatch path，才基于 `DESIGN.md` 的合法范围和 `{op}.py` 已有 input generator 补一个最小 case。

要求：

- workload 必须能触发绑定的 dispatch path。
- 每个 dispatch path 不要默认扩展多个 shape/dtype。
- 只有用户明确要求多 shape、多 dtype 或性能目标本身包含多个 case 时，才额外采集。

---

## 3. 串行运行 msprof op

对每个 `(dispatch_path, workload_id)` 逐个运行 `msprof op`。禁止并发运行多个 profiling。

命令模板：

```bash
msprof op \
  --kernel-name={target_kernel_name} \
  --output={output_dir} \
  --launch-count=20 \
  --warm-up=5 \
  --dump=off \
  --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio \
  python {op}.py
```

Phase 2 复用本模板测试 current best 或实验分支时，`python {op}.py` 必须替换为本次实际要测的文件，例如 `python {op}_opt_v{iter}_{opt_id}.py`。日志中的 `command` 必须包含该分支路径，避免误测原始 Stage 3 文件。

实验迭代可用较小 `launch-count` 快速判断方向，但必须在日志中记录 launch 数；baseline、候选 winner 和 final 推荐使用更稳定的 launch 数复测。

如果当前 CANN 版本不兼容逗号形式的 `--aic-metrics`，退回：

```bash
msprof op --kernel-name={target_kernel_name} --output={output_dir} --launch-count=10 --warm-up=5 --aic-metrics=Default python {op}.py --case {workload_id}
```

输出目录只需保证唯一，推荐：

```text
perf_opt/profiles/{profile_stage}/{run_id}_{dispatch_path}/
```

新建 `msprof op` 输出目录后，先确保 group/other 不可写，例如 `chmod 700 {output_dir}`；否则某些环境会拒绝采集或写入失败。

如果 `msprof op` 在输出目录下生成 `OPPROF_xxx` 子目录，记录实际 `raw_profile_dir`。

注意：`msprof op` 多 launch 稳定性需要靠多次独立运行，不要用 CSV 记录条数推断实际 launch 次数。

实验 stdout/stderr 不写到 `perf_opt/` 顶层，统一写入：

```text
perf_opt/logs/{profile_stage}/{run_id}.log
```

---

## 4. 校验采集结果

每次采集后检查：

- 能读取目标记录`OpBasicInfo.csv`的 `Task Duration(us)`。
- `captured_op_name` 能通过命令、输出目录和运行日志追溯到本次 `target_kernel_name` 或目标 TileLang kernel。
- 采到的不是 Cast / Mul / OnesLike / Random 等框架小算子。
- kernel launch 次数足够覆盖 `warm-up + launch-count`。

注意：`captured_op_name` 不一定机械等于 Python 函数名。只要能证明它属于本次目标 TileLang kernel，即可标记为 valid。

无效 profile 记录 reason，并重新采集；无效数据不能进入诊断。

---

## 5. 记录格式

把采集结果写入 `perf_opt/opt_log.md` 的 `Performance Test Data` 章节。

最小格式：

```markdown
## Performance Test Data

| dispatch_path | workload_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir | command |
|---|---|---|---|---:|---|---|---|
| default | L0_main | {target_kernel} | {captured_op} | {v} | valid | {raw_dir} | {cmd} |
| fp32_path | fp32_main | {target_kernel} | {captured_op} | {v} | valid | {raw_dir} | {cmd} |
```

若某个 dispatch path 未覆盖，单独记录：

```markdown
### Uncovered Dispatch

| dispatch_path | reason |
|---|---|
| {path} | no workload triggers this branch |
```

返回：

```text
PERF_DATA_COLLECTED
```
