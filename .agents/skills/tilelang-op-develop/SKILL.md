---
name: tilelang-op-develop
description: "根据冻结的 DESIGN.md 生成算子实现（example_{op}.py：kernel + golden + 分层测试套件 L0/L1/L2/Boundary），执行测试并返回三态判定。非 NPU 环境通过打桩验证端到端流程。触发：实现算子、生成 kernel、算子开发、跑精度。"
---

# TileLang-NPUIR 算子开发与验证

## 1. 目标

根据 Stage 1 冻结的 `DESIGN.md` 与 Stage 2 通过的 `REVIEW.md`，生成算子实现文件 `example_{op}.py`（含 `@tilelang.jit` kernel + 内嵌 PyTorch golden + 分层测试套件 L0/L1/L2/Boundary + main 入口），执行测试，并返回三态判定供 Orchestrator 路由。

> **关键约束**：本 skill 的产物 `example_{op}.py` 必须能在**非 NPU 环境**通过打桩跑通端到端精度流程。打桩规则见 [references/stub-harness.md](references/stub-harness.md)。

---

## 2. 输入

| 字段 | 说明 |
|------|------|
| `design_md_path` | 冻结的 `DESIGN.md`（含 L0 测试计划） |
| `review_md_path` | Stage 2 通过的 `REVIEW.md`（设计已检视通过） |
| `mode` | `first_impl` / `retry_impl` / `precision_fix`（由 Orchestrator 传入） |
| `attempt_index` | 当前 Stage 3 attempt 序号 |
| `last_failure_summary` | 重试时传入的失败信息（stderr 摘要 / 精度失败详情） |
| `design_revision_count` | 设计修订次数（用于回退后清零判断） |

---

## 3. 打桩策略（必须遵循）

> **背景**：当前环境无 NPU 设备，`tilelang`（依赖 `tvm`）与 `torch_npu` 均不可导入，仅有 CPU `torch`。为验证端到端流程可行性，对 NPU 执行过程打桩。

### 打桩开关

通过环境变量 `TILELANG_OP_STUB_NPU=1` 启用打桩。生成的 `example_{op}.py` 必须在文件顶部检测该变量并自动分支。

### 打桩范围（仅这两类 NPU 执行过程打桩）

| 过程 | 真实行为 | 打桩行为 | 标记 |
|------|----------|----------|------|
| kernel 编译 + 执行 | `@tilelang.jit(target="npuir")` 编译并在 NPU 运行 | 跳过编译与执行，**用 CPU golden 输出充当 kernel 输出**，使精度校验通过 | `# [STUB: NPU-EXEC]` |
| 性能 profiling | NPU 上测延迟/吞吐 | 不在本 skill；由 tilelang-op-optimize 打桩 | — |

### 打桩标记规范

所有打桩代码必须用以下注释对包裹，便于后续全局搜索取消打桩：

```python
# === [STUB: NPU-EXEC] 打桩区开始 ===
... 打桩逻辑 ...
# === [STUB: NPU-EXEC] 打桩区结束 ===
```

取消打桩时：设置 `TILELANG_OP_STUB_NPU=0`（或不设置）即可走真实 NPU 路径；删除打桩时全局搜索 `[STUB: NPU-EXEC]` 清理。

### 不打桩的部分

- **kernel 源码本身**：`@tilelang.jit` kernel 函数体必须按 DESIGN.md 真实生成（打桩时仅不执行，源码仍完整保留在文件中，供真实 NPU 环境使用）。
- **golden 函数**：PyTorch CPU 参考实现，真实计算。
- **测试用例 shape/dtype/精度标准**：按 DESIGN.md L0 计划真实生成。

详细打桩模板见 [references/stub-harness.md](references/stub-harness.md)，可运行示例模板见 [examples/example_template.py](examples/example_template.py)。

---

## 4. 工作流程

### Phase 1：读取设计
1. Read `DESIGN.md` 全文，提取：算子名、I/O 规格、编程模式、API 映射、Tiling、内存层级、同步策略、L0 测试计划、精度标准。
2. Read `REVIEW.md`，确认检视已通过（如有 warn 项记录但不阻塞）。

### Phase 2：生成 kernel
1. 按 DESIGN.md §3 API 映射 + §6 循环结构生成 `@tilelang.jit(target="npuir")` kernel。
2. **优先 v-prefix API**（vadd/vmul/vexp/vcast/vbrc），npuir_xxx 仅作兼容。
3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。

### Phase 3：生成 golden
1. 按 DESIGN.md §8.1 生成 PyTorch CPU 参考实现 `golden_{op}(...)`。
2. Golden 必须在 CPU 上可独立运行（不依赖 tilelang / torch_npu）。

### Phase 4：生成测试套件（分层）
按 DESIGN.md L0 计划 + 本 skill 扩展规则：

| 层级 | 内容 | 生成时机 | 失败影响 |
|------|------|----------|----------|
| L0 | 门槛规则 shape（block 整除）、dtype、golden 对比 | 首次 `first_impl` 即生成 | 阻塞（→ precision_fix） |
| L1 | 功能覆盖（含不规则 shape、多 dtype） | L0 通过后扩展 | 阻塞（→ precision_fix） |
| L2 | 异常输入（空、超大、错误 dtype） | L0 通过后扩展 | 仅记录，不阻塞 |
| Boundary | 特殊值（0、inf、nan、极小/极大） | L0 通过后扩展 | 仅记录，不阻塞 |

### Phase 5：执行测试（含打桩分支）
1. 检测 `TILELANG_OP_STUB_NPU`。
2. 跑 L0：`python example_{op}.py --level L0`（打桩时 golden 充当输出，精度通过）。
3. L0 通过后扩展 L1/L2/Boundary 并跑全量 `--level all`。
4. 收集结果：max_diff、失败用例 shape、层级。

### Phase 6：三态判定与返回

| 条件 | 返回标记 |
|------|----------|
| L0 + L1 全过（L2/Boundary 告警仅记录） | `[PRECISION_PASS]` |
| L0 或 L1 未过 | `[PRECISION_FAIL]` |
| 发现设计层错误（API 不可用、L0C 溢出、内存层级冲突等实现层无法修复） | `[DESIGN_ERROR]` + 原因 |
| 无标记且 exit code ≠ 0 | 运行失败（Orchestrator 按 retry_impl 路由） |

---

## 5. `example_{op}.py` 结构规范

生成的文件必须包含以下组成部分（顺序）：

```python
# 1. 版权 + imports
# 2. [STUB: NPU-EXEC] 打桩开关区
# 3. golden_{op}(...) 函数
# 4. @tilelang.jit kernel（if not STUB_NPU: 守卫，源码完整保留）
# 5. 分层测试函数：run_L0() / run_L1() / run_L2() / run_boundary()
# 6. main()：argparse --level {L0|all}，按 level 调用对应测试
```

完整可运行模板见 [examples/example_template.py](examples/example_template.py)。

### main 块的打桩分支（核心）

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    if args.level == "L0":
        run_L0()
    else:
        run_L0(); run_L1(); run_L2(); run_boundary()
```

每个 `run_LX()` 内部按打桩开关分支：打桩时用 CPU 张量 + golden 充当输出；真实时用 NPU 张量 + kernel 执行。详见模板。

---

## 6. 失败处理

| 失败类型 | 识别 | 处理 |
|---------|------|------|
| 编译错误（实现层） | stderr 含 lowering/codegen 错误 | 返回运行失败 + stderr 摘要，Orchestrator 走 retry_impl |
| API 不存在 | `AttributeError` / 设计用 API 无导出 | 返回 `[DESIGN_ERROR]` + 原因 |
| L0C/UB 溢出 | 编译期或运行期报容量超限 | 返回 `[DESIGN_ERROR]` + 原因 |
| 精度不达标 | `assert_close` 失败 | 返回 `[PRECISION_FAIL]` + max_diff/失败 shape |
| 内存层级越级 | stderr 提示 GM/L1/UB/L0 访问违规 | 返回 `[DESIGN_ERROR]` + 原因 |
| 环境问题 | `ImportError` 且非打桩模式 | 返回运行失败，提示检查环境 |

> **打桩模式下不应出现 ImportError**：打桩时跳过 tilelang/torch_npu 导入。若打桩模式仍报 ImportError，说明打桩开关未正确生效，需修复文件顶部打桩区。

---

## 7. 备份规则

`precision_fix` 模式每次修改 `example_{op}.py` 前，必须先备份：
```bash
cp example_{op}.py history_version/{op}_impl_s3_attempt{N}.py
```
（`{N}` = 当前 attempt_index）

---

## 8. 完成报告

返回结构化摘要：

```markdown
## Stage Result
- stage: 3
- mode: first_impl / retry_impl / precision_fix
- operator: {op}
- output: examples/{op}/example_{op}.py
- verdict: [PRECISION_PASS] / [PRECISION_FAIL] / [DESIGN_ERROR] / RUNTIME_FAIL
- test_results:
  - L0: pass / fail (N cases)
  - L1: pass / fail (N cases)
  - L2: pass / warn (N cases, 不阻塞)
  - Boundary: pass / warn (N cases, 不阻塞)
- max_diff: <精度数值>
- stub_mode: true / false  # 是否在打桩模式
- design_error_summary: <仅 DESIGN_ERROR 时填>
- skills_consulted: <引用的 skill 路径>
- summary: <一句话>
- issues: <若无则 none>
```
