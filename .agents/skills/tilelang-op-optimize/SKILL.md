---
name: tilelang-op-optimize
description: "对精度通过的算子实现进行规格保形性能调优，依据冻结 SPEC.md 的 AC 验收契约确保调优不使精度退化，产出 kernel_opt.py 与调优日志。触发：性能调优、optimize、性能优化、perf tuning。"
---

# TileLang-NPUIR 规格保形性能调优（Spec-Driven）

## 1. 目标

对 Stage 3 精度通过的 `{op}.py` 进行性能调优，产出优化后的 `perf_tuning/kernel_opt.py` 与调优日志，直到满足中止条件。

> **Spec-Driven 约束（AC 保形）**：调优以冻结 `SPEC.md` 的验收契约 `AC-*` 为不可逾越的红线——每轮优化后须验证 `AC-2/AC-3`（L0/L1）仍通过；任何导致 `AC-*` 退化的优化必须回滚。接口契约 `IC-5` 须保持一致（kernel 入口签名不变）。

> **环境前提**：本 skill 运行在已具备 NPU 设备的环境中，性能 profiling 在 NPU 上真实执行。调优分析（瓶颈识别、优化策略）与性能测量均为真实结果。

---

## 2. 输入

| 字段 | 说明 |
|------|------|
| `kernel_py_path` | Stage 3 精度通过的 `{op}.py`（已符合 `IC-*`、通过 `AC-*`） |
| `spec_md_path` | 冻结的 `SPEC.md`（含 `AC-*` 验收契约、`IC-5` 入口签名、性能目标条款） |
| 性能目标 | 类型（latency/throughput/baseline_compare/best_effort）、目标数值、测试 shape、噪声阈值、最大迭代数 |

---

## 3. 工作流程

### Phase 1：基线分析
1. Read `{op}.py` 与 `SPEC.md`（`AC-*` 验收契约、`IC-5` 入口签名、性能目标条款）。
2. 识别性能瓶颈（基于规格：访存瓶颈、计算密度、流水线深度、block 大小）。
3. 测量基线性能（NPU 上真实 profiling）。

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
2. **生成优化版本** `perf_tuning/kernel_opt_v{iter}.py`（基于上一版修改；保持 `IC-5` 入口签名不变）。
3. **AC 保形校验**：跑 `AC-2`(L0) + `AC-3`(L1) 确保优化未使 `AC-*` 退化；精度阈值取自 `AC-1`。
   - 退化 → **回滚该轮优化**（丢弃 `kernel_opt_v{iter}.py`，记入日志"AC 退化回滚"），`consecutive_no_improvement += 1`。
4. **性能测量**（NPU 上真实 profiling）。
5. **记录调优日志**到 `perf_tuning/tuning_log.md`：迭代号、策略、latency、相对基线提升、**AC 保形状态**（标注遵守的 AC 条款）。
6. 判断中止条件。

### Phase 3：中止条件判定

满足任一即结束：
1. 迭代次数达到用户指定上限（默认 10）。
2. 连续三次无性能提升。
3. 达到用户指定的性能目标（latency ≤ 目标 / throughput ≥ 目标 / 优于 baseline）。
4. 任一轮 `AC-*` 退化且无法回滚恢复（记录为"AC 退化中止"）。

### Phase 4：交付
1. 选最优版本（AC 保形）作为 `perf_tuning/kernel_opt.py`。
2. 汇总调优日志（含 AC 保形结论）。
3. 返回 `TUNING_COMPLETED`。

---

## 4. 产物结构

```text
examples/{project}/{op}/perf_tuning/
├── kernel_opt.py            # 最终最优版本（AC 保形）
├── kernel_opt_v1.py         # 各迭代版本
├── kernel_opt_v2.py
├── tuning_log.md            # 调优日志（标注 AC 保形状态）
└── baseline.py -> ../{op}.py  # 基线（软链或拷贝）
```

### tuning_log.md 模板

```markdown
# {op} 性能调优日志

## 基线
- latency: {base} us
- 来源: {op}.py
- AC 状态: AC-2/AC-3 pass

## 迭代记录
| iter | 策略 | latency(us) | 提升 | AC 保形 | 精度 |
|------|------|-------------|------|---------|------|
| 1 | {策略} | {v} | {x}% | pass | pass |
| 2 | {策略} | {v} | {x}% | pass / 回滚(退化) | pass / fail |

## 瓶颈与限制
- {若性能瓶颈源于规格层不可行，记录于此；不主动触发规格修订}

## 结论
- 最优版本: kernel_opt.py (iter {N})
- 最终 latency: {v} us
- 总提升: {x}%
- AC 保形: pass / fail
- 中止原因: {达目标 / 迭代上限 / 连续无提升 / AC退化中止}
```

---

## 5. 完成报告

```markdown
## Stage Result
- stage: 4
- project: {project}
- operator: {op}
- output: examples/{project}/{op}/perf_tuning/kernel_opt.py
- verdict: TUNING_COMPLETED
- iterations: {N}
- baseline_latency: {v} us
- final_latency: {v} us
- improvement: {x}%
- stop_reason: {达目标 / 迭代上限 / 连续无提升 / AC退化中止}
- ac_preserved: pass / fail   # AC-* 验收是否仍通过
- ac_status:
  - AC-2 (L0): pass / fail
  - AC-3 (L1): pass / fail
- interface_preserved: pass / fail   # IC-* 保形
- skills_consulted: <引用的 skill 路径>
- summary: <一句话>
- issues: <若无则 none>
```

---

## 6. 注意事项

- **调优不逆向反馈**：性能不足时由本 skill 自完成最优版本，不触发 Stage 3 或 Stage 1 修改（对齐 conductor 设计）。性能瓶颈若源于规格层不可行，仅记入日志"瓶颈与限制"章节。
- **AC 保形是硬约束**：每轮优化后跑 `AC-2`(L0)（必要时 `AC-3` L1），精度退化则回滚该轮优化；退化且无法回滚时按"AC 退化中止"结束。
- **接口保形**：`kernel_opt.py` 入口签名须与 `IC-5` 一致。
