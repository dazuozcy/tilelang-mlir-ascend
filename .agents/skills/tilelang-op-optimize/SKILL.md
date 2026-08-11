---
name: tilelang-op-optimize
description: "对精度通过的算子实现进行性能调优，产出 {op}.py 与调优日志。触发：性能调优、optimize、性能优化、perf tuning。"
---

# TileLang-NPUIR 算子性能调优

## 1. 目标

对 Stage 3 精度通过的 `{op}.py` 进行性能调优，每轮只改一个优化点，改完单独验证，产出优化后的 `perf_opt/{op}.py` 与调优日志，直到满足中止条件。

> **环境前提**：本 skill 运行在已具备 NPU 设备的环境中，性能 profiling 在 NPU 上真实执行。调优分析（瓶颈识别、优化策略）与性能测量均为真实结果。

---

## 2. 参考资料

- cube kernel 优化参考：[tilelang-cube-skill](../tilelang-cube-skill/)
- vector kernel 优化参考：[tilelang-vector-skill](../tilelang-vector-skill/)
- mix kernel 优化参考：[tilelang-mixcv-skill](../tilelang-mixcv-skill/)

---

## 3. 工作流程

### Phase 1：基线分析
1. Read `{op}.py` 与 `DESIGN.md` 性能目标章节。
2. 识别性能瓶颈(基于设计：访存瓶颈、计算密度、流水线深度、block 大小)。
3. 测量基线性能(NPU 上真实 profiling).
> 注意，只能使用 msprof op 命令采集 kernel 本身的耗时, 命令参考下面所示：
```shell
msprof op --kernel-name=kernel_name --output=output --launch-count=10 --warm-up=5 python script.py
```

### Phase 2：优化迭代（每轮）

每轮迭代执行：

1. **选定优化策略**（按下表优先级）：
   | 策略 | 适用场景 | 预期收益 |
   |------|----------|----------|
   | 调整 block size | 访存/计算不均衡 | 10-30% |
   | 增加 T.Pipelined 流水深度 | 有多次循环迭代 | 10-20% |
   | double-buffer / 多缓冲 | 搬运与计算可重叠 | 10-15% |
   | v-prefix API 替换 npuir_xxx | Vector 类 | 5-15% |
   | 减少中间 buffer | UB 压力 | 5-10% |
   | data reuse / 寄存器化 | 重复访存 | 5-15% |
2. **生成优化版本** `perf_opt/{op}}_v{iter}.py`（基于上一版修改）。
3. **精度回归检查**：跑 L0 确保优化未破坏精度。
4. **性能测量**（NPU 上真实 profiling）。
5. **记录调优日志**到 `perf_opt/opt_log.md`：迭代号、策略、latency、相对基线提升、精度状态。
6. 判断中止条件。

### Phase 3：中止条件判定

满足任一即结束：
1. 迭代次数达到用户指定上限(默认 10).
2. 连续三次无性能提升.
3. 达到用户指定的性能目标(latency ≤ 目标 或 throughput ≥ 目标 或 优于 baseline).

### Phase 4：交付
1. 选最优版本作为 `perf_opt/{op}.py`。
2. 汇总调优日志。
3. 返回 `TUNING_COMPLETED`。

---

## 4. 产物结构

```text
examples/{project}/{op}/perf_opt/
├── {op}.py            # 最终最优版本
├── {op}_opt_v1.py     # 各迭代版本
├── {op}_opt_v2.py
├── opt_log.md         # 调优日志
└── baseline.py -> ../{op}.py  # 基线（软链或拷贝）
```

### opt_log.md 模板

```markdown
# {op} 性能调优日志

## 基线
- latency: {base} us
- 来源: {op}.py

## 迭代记录
| iter | 策略 | latency(us) | 提升 | 精度 |
|------|------|-------------|------|------|
| 1 | {策略} | {v} | {x}% | pass |
| 2 | {策略} | {v} | {x}% | pass |

## 结论
- 最优版本: {op}.py (iter {N})
- 最终 latency: {v} us
- 总提升: {x}%
- 中止原因: {达目标 / 迭代上限 / 连续无提升}
```

---

## 5. 完成报告

```markdown
## Stage Result
- stage: 4
- project: {project}
- operator: {op}
- output: examples/{project}/{op}/perf_opt/{op}.py
- verdict: TUNING_COMPLETED
- iterations: {N}
- baseline_latency: {v} us
- final_latency: {v} us
- improvement: {x}%
- stop_reason: {达目标 / 迭代上限 / 连续无提升}
- skills_consulted: <引用的 skill 路径>
- summary: <一句话>
- issues: <若无则 none>
```

---

## 6. 注意事项

- **调优不逆向反馈**：性能不足时由本 skill 自完成最优版本，不触发 Stage 3 或 Stage 1 修改（对齐 conductor 设计）。
- **精度回归必须检查**：每轮优化后跑 L0，精度退化则回滚该轮优化。
