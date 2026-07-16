---
name: tilelang-op-develop
description: "依据冻结的 SPEC.md 规格驱动实现算子（{op}.py：kernel + golden），生成 AC 追溯分层测试套件，执行测试并返回三态判定。触发：实现算子、生成 kernel、算子开发、跑精度。"
---

# TileLang-NPUIR 规格驱动实现与验证（Spec-Driven）

## 1. 目标

依据 Stage 1 冻结的 `SPEC.md`（权威契约 `IC/FC/AC/CC` + 推荐方案 `ID`）与 Stage 2 通过的 `REVIEW.md`，生成算子实现文件 `{op}.py`（含 `@tilelang.jit` kernel + 内嵌 PyTorch golden + **AC 追溯分层测试套件** + main 入口），执行测试，并返回三态判定供 conductor 路由。

> **Spec-Driven 约束**：
> - kernel 入口签名必须符合 `IC-5`，I/O 规格符合 `IC-2/IC-3`。
> - 测试用例必须覆盖 `AC-2..AC-5`，每个用例标注其验证的 AC 条款 ID；精度阈值取自 `AC-1`；golden 取自 `AC-6`。
> - `ID-*` 为推荐方案，可在契约边界内偏离；仅当契约条款本身不可行时返回 `[SPEC_ERROR]`（须附不可行条款 ID）。

> **环境前提**：本 skill 运行在已具备 NPU 设备的环境中，`tilelang` 与 `torch_npu` 可正常导入。kernel 编译与执行在 NPU 上真实进行，精度校验为真实结果。

---

## 2. 输入

| 字段 | 说明 |
|------|------|
| `spec_md_path` | 冻结的 `SPEC.md`（`spec_frozen=true`，含 `IC/FC/AC/CC/ID` 条款） |
| `review_md_path` | Stage 2 通过的 `REVIEW.md`（规格已检视通过） |
| `mode` | `first_impl` / `retry_impl` / `precision_fix`（由 conductor 传入） |
| `attempt_index` | 当前 Stage 3 attempt 序号 |
| `last_failure_summary` | 重试时传入的失败信息（stderr 摘要 / 精度失败详情，含未过 AC 条款 ID） |
| `spec_revision_count` | 规格修订次数（用于回退后清零判断） |

---

## 3. 工作流程

### Phase 1：读取规格
1. Read `SPEC.md` 全文，提取契约条款：`IC-5` 入口签名、`IC-2/IC-3` I/O 规格、`CC-2` 编程模式、`ID-1` API 映射、`ID-3` Tiling、`ID-2` 内存层级、`ID-5` 同步策略、`AC-1` 精度标准、`AC-2..AC-6` 测试契约。
2. Read `REVIEW.md`，确认检视已通过（如有 warn 项记录但不阻塞）。

### Phase 2：生成 kernel
1. 按 `IC-5` 入口签名 + `ID-1` API 映射 + `ID-4` 循环结构生成 `@tilelang.jit(target="npuir")` kernel。
2. **优先 v-prefix API**（vadd/vmul/vexp/vcast/vbrc），npuir_xxx 仅作兼容。
3. 遵循项目根 AGENTS.md："不要凭记忆猜 API"、"从示例入手"——先 Glob `examples/` 同类实现参考。
4. 若发现 `ID-1` API 不可用 / `CC-5` 容量超限 / `ID-2` 内存层级冲突等**契约不可行** → 返回 `[SPEC_ERROR]` + 不可行条款 ID。

### Phase 3：生成 golden
1. 按 `AC-6` 生成 PyTorch CPU 参考实现 `golden_{op}(...)`。
2. Golden 必须在 CPU 上可独立运行（不依赖 torch_npu）。

### Phase 4：生成 AC 追溯分层测试
1. 按 `AC-2..AC-5` 生成 `run_L0()/run_L1()/run_L2()/run_boundary()`，每个用例标注其验证的 AC 条款 ID。
2. 精度阈值取自 `AC-1`（dtype→atol/rtol）。

### Phase 5：执行测试（验收契约执行）
1. 跑 L0（`AC-2`）：`python {op}.py --level L0`。
2. L0 通过后扩展 L1/L2/Boundary（`AC-3/AC-4/AC-5`）并跑全量 `--level all`。
3. 收集结果：max_diff、失败用例 shape、未过 AC 条款 ID、层级。

### Phase 6：三态判定与返回

| 条件 | 返回标记 |
|------|----------|
| `AC-2`(L0) + `AC-3`(L1) 全过（`AC-4/AC-5` 告警仅记录） | `[PRECISION_PASS]` |
| `AC-2` 或 `AC-3` 未过 | `[PRECISION_FAIL]` + 未过 AC 条款 + max_diff |
| 契约条款不可行（`IC/FC/AC/CC` 不可行，或偏离 `ID` 将违反契约） | `[SPEC_ERROR]` + 不可行条款 ID + 原因 |
| 无标记且 exit code ≠ 0 | 运行失败（conductor 按 retry_impl 路由） |

---

## 4. `{op}.py` 结构规范

生成的文件必须包含以下组成部分（顺序）：
注意：*.py 中的注释只能使用英文

```python
# 1. Copyright (c) Huawei Technologies Co., Ltd. 2026.
# 2. imports (tilelang or torch_npu)
# 2. golden_{op}(...) function            # AC-6
# 3. @tilelang.jit kernel                  # conforms to IC-5
# 4. precision comparing func：run_case()
# 5. main()
```

完整可运行模板见 [examples/example_template.py](examples/example_template.py)。

### main 块结构

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

每个 `run_LX()` 内部在 NPU 上创建张量、调用 kernel 执行、与 golden 对比精度。**每个用例须以注释标注其验证的 AC 条款 ID**（如 `# AC-2: L0 regular shape`）。详见模板。

---

## 5. 失败处理

| 失败类型 | 识别 | 处理 |
|---------|------|------|
| 编译错误（实现层） | stderr 含 lowering/codegen 错误 | 返回运行失败 + stderr 摘要，conductor 走 retry_impl |
| API 不存在（契约层） | `AttributeError` / `IC-5`/`ID-1` 用 API 无导出 | 返回 `[SPEC_ERROR]` + 不可行条款 ID（`IC-5`/`ID-1`） + 原因 |
| L0C/UB 溢出（契约层） | 编译期或运行期报容量超限，超出 `CC-5` | 返回 `[SPEC_ERROR]` + 不可行条款 ID（`CC-5`/`ID-3`） + 原因 |
| 精度不达标 | `assert_close` 失败（`AC-2/AC-3` 未过） | 返回 `[PRECISION_FAIL]` + 未过 AC 条款 + max_diff/失败 shape |
| 内存层级越级（契约层） | stderr 提示 GM/L1/UB/L0 访问违规，违反 `CC-3/ID-2` | 返回 `[SPEC_ERROR]` + 不可行条款 ID + 原因 |
| 环境问题 | `ImportError` 指向 tilelang/torch_npu 未安装或未 `source set_env.sh` | 返回运行失败，提示检查环境 |

---

## 6. 备份规则

`precision_fix` 模式每次修改 `{op}.py` 前，必须先备份：
```bash
cp {op}.py history_version/{op}_impl_s3_attempt{N}.py
```
（`{N}` = 当前 attempt_index）

---

## 7. 完成报告

返回结构化摘要：

```markdown
## Stage Result
- stage: 3
- mode: first_impl / retry_impl / precision_fix
- project: {project}
- operator: {op}
- output: examples/{project}/{op}/{op}.py
- verdict: [PRECISION_PASS] / [PRECISION_FAIL] / [SPEC_ERROR] / RUNTIME_FAIL
- interface_conformance: pass / fail   # IC-* 符合性
- ac_trace:
  - AC-1 (精度标准): applied / mismatched
  - AC-2 (L0): pass / fail (N cases)
  - AC-3 (L1): pass / fail (N cases)
  - AC-4 (L2): pass / warn (N cases, 不阻塞)
  - AC-5 (Boundary): pass / warn (N cases, 不阻塞)
  - AC-6 (Golden): present / missing
- test_results:
  - L0: pass / fail (N cases)
  - L1: pass / fail (N cases)
  - L2: pass / warn (N cases, 不阻塞)
  - Boundary: pass / warn (N cases, 不阻塞)
- max_diff: <精度数值>
- spec_error_summary: <仅 SPEC_ERROR 时填，含不可行条款 ID + 原因>
- skills_consulted: <引用的 skill 路径>
- summary: <一句话>
- issues: <若无则 none>
```
