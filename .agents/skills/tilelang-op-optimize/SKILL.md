---
name: tilelang-op-optimize
description: "对精度通过的算子实现进行性能调优，产出 kernel_opt.py 与调优日志。非 NPU 环境对性能测量打桩以验证端到端流程。触发：性能调优、optimize、性能优化、perf tuning。"
---

# TileLang-NPUIR 算子性能调优

## 1. 目标

对 Stage 3 精度通过的 `example_{op}.py` 进行性能调优，产出优化后的 `perf_tuning/kernel_opt.py` 与调优日志，直到满足中止条件。

> **关键约束**：当前环境无 NPU，性能 profiling 无法真实执行，需对**性能测量**打桩。调优分析（瓶颈识别、优化策略）不打桩，保持真实推理。

---

## 2. 输入

| 字段 | 说明 |
|------|------|
| `kernel_py_path` | Stage 3 精度通过的 `example_{op}.py` |
| `design_md_path` | 含性能目标章节的 `DESIGN.md` |
| 性能目标 | 类型（latency/throughput/baseline_compare/best_effort）、目标数值、测试 shape、噪声阈值、最大迭代数 |

---

## 3. 打桩策略（必须遵循）

### 打桩范围

| 过程 | 真实行为 | 打桩行为 | 标记 |
|------|----------|----------|------|
| 性能 profiling | NPU 上测延迟/吞吐/带宽 | 返回**模拟性能数据**（基于算子规模的估算值） | `# [STUB: PERF-MEASURE]` |
| kernel 执行（精度回归检查） | NPU 编译+执行 | 复用 Stage 3 的 `TILELANG_OP_STUB_NPU` 打桩 | `# [STUB: NPU-EXEC]` |

### 打桩开关

- 性能测量打桩：`TILELANG_OP_STUB_PERF=1`（或继承 `TILELANG_OP_STUB_NPU=1`）
- 生成 `perf_tuning/kernel_opt.py` 时同样嵌入 `TILELANG_OP_STUB_NPU` 打桩区（与 Stage 3 一致）

### 打桩标记规范

```python
# === [STUB: PERF-MEASURE] 打桩区开始 ===
... 模拟性能数据 ...
# === [STUB: PERF-MEASURE] 打桩区结束 ===
```

取消打桩：全局搜索 `[STUB: PERF-MEASURE]` 清理；真实 NPU 环境设 `TILELANG_OP_STUB_PERF=0`。

### 模拟性能数据规则（打桩时）

为保证端到端流程可验证且迭代有"改善信号"（避免恒定值导致连续无提升立即中止），打桩数据按以下规则生成：

```python
# === [STUB: PERF-MEASURE] 模拟延迟（随迭代递减，模拟优化收益） ===
import os
_STUB_ITER = int(os.environ.get("TILELANG_OP_STUB_ITER", "0"))
base_latency = 100.0  # us，基线
latency_us = max(10.0, base_latency * (0.85 ** _STUB_ITER))  # 每轮降 15%
print(f"[STUB] 模拟 latency={latency_us:.2f}us (iter={_STUB_ITER})")
# === [STUB: PERF-MEASURE] 打桩区结束 ===
```

> 该规则使每轮迭代有约 15% 改善，验证"迭代收敛 / 达目标 / 中止"逻辑均可触发。真实环境取消打桩后用真实 profiling 替换。

---

## 4. 工作流程

### Phase 1：基线分析
1. Read `example_{op}.py` 与 `DESIGN.md` 性能目标章节。
2. 识别性能瓶颈（基于设计：访存瓶颈、计算密度、流水线深度、block 大小）。
3. 测量基线性能（打桩：模拟基线 `latency_us=100.0`）。

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
2. **生成优化版本** `perf_tuning/kernel_opt_v{iter}.py`（基于上一版修改）。
3. **精度回归检查**：跑 L0 确保优化未破坏精度（打桩：golden 充当输出，恒通过）。
4. **性能测量**（打桩：模拟递减 latency）。
5. **记录调优日志**到 `perf_tuning/tuning_log.md`：迭代号、策略、latency、相对基线提升、精度状态。
6. 更新 `_STUB_ITER`，判断中止条件。

### Phase 3：中止条件判定

满足任一即结束：
1. 迭代次数达到用户指定上限（默认 10）。
2. 连续三次无性能提升（打桩模式下不触发，因模拟值恒递减）。
3. 达到用户指定的性能目标（latency ≤ 目标 / throughput ≥ 目标 / 优于 baseline）。

### Phase 4：交付
1. 选最优版本作为 `perf_tuning/kernel_opt.py`。
2. 汇总调优日志。
3. 返回 `TUNING_COMPLETED`。

---

## 5. 产物结构

```text
examples/{op}/perf_tuning/
├── kernel_opt.py            # 最终最优版本（含 Stage 3 同款 NPU-EXEC 打桩区）
├── kernel_opt_v1.py         # 各迭代版本
├── kernel_opt_v2.py
├── tuning_log.md            # 调优日志
└── baseline.py -> ../example_{op}.py  # 基线（软链或拷贝）
```

### tuning_log.md 模板

```markdown
# {op} 性能调优日志

## 基线
- latency: {base} us
- 来源: example_{op}.py

## 迭代记录
| iter | 策略 | latency(us) | 提升 | 精度 |
|------|------|-------------|------|------|
| 1 | {策略} | {v} | {x}% | pass |
| 2 | {策略} | {v} | {x}% | pass |

## 结论
- 最优版本: kernel_opt.py (iter {N})
- 最终 latency: {v} us
- 总提升: {x}%
- 中止原因: {达目标 / 迭代上限 / 连续无提升}
- stub_mode: true / false
```

---

## 6. 完成报告

```markdown
## Stage Result
- stage: 4
- operator: {op}
- output: examples/{op}/perf_tuning/kernel_opt.py
- verdict: TUNING_COMPLETED
- iterations: {N}
- baseline_latency: {v} us
- final_latency: {v} us
- improvement: {x}%
- stop_reason: {达目标 / 迭代上限 / 连续无提升}
- stub_mode: true / false
- skills_consulted: <引用的 skill 路径>
- summary: <一句话>
- issues: <若无则 none>
```

---

## 7. 注意事项

- **调优不逆向反馈**：性能不足时由本 skill 自完成最优版本，不触发 Stage 3 或 Stage 1 修改（对齐 Orchestrator 设计）。
- **精度回归必须检查**：每轮优化后跑 L0，精度退化则回滚该轮优化（打桩模式恒通过，真实环境需严格检查）。
- **打桩模式下性能数据为模拟值**，仅验证流程通路与迭代/中止逻辑，不反映真实性能。真实性能须在 NPU 环境取消打桩后测量。
