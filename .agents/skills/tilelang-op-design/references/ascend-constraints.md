# 技术约束清单（必须遵守，写入 SPEC.md 约束契约 CC-*）

本项目为 TileLang-NPUIR（后端为华为昇腾 NPU），与 GPU 版 TileLang 有显著差异。
**外部参考实现不可直接使用，必须转换为 NPUIR 兼容方案。**

> 本清单的结论是 `SPEC.md` 中 `CC-3`（NPU 硬件约束）、`CC-4`（技术约束检测结论）、`CC-5`（内存预算）的依据。`[SPEC_ERROR]` 触发时，若根因属本清单约束，须指明对应 CC 条款 ID。

## 目录

- [技术约束清单（必须遵守）](#技术约束清单必须遵守)
  - [目录](#目录)
  - [1. 本项目已知限制](#1-本项目已知限制)
  - [2. 强制检测规则](#2-强制检测规则)
  - [3. 警告输出格式](#3-警告输出格式)

---

## 1. 本项目已知限制

| 约束 | 说明 | 影响 | 替代方案 |
|------|------|------|----------|
| **不支持三维 Kernel** | `T.Kernel` 只接受一维 block 数 | 三维并行设计无法实现 | 将三维相乘结果作为 一维 block 数 |
| **部分 GPU API 不可用** | CUDA 专用 API 在 Ascend 不存在 | 直接移植 GPU 代码失败 | 查阅本项目 `examples/` 确认 Ascend API |
| **GEMM 要求 M,N 为 block 整数倍** | `M // block_M` 整除依赖；`M < block_M` 时零 block 启动 | 输出全零或除零编译崩溃 | 设计文档 §4/§5 必须明确处理策略：host 侧 padding+crop 或 Kernel 动态 block |
| **L0C 容量上限** | A2/A3 设备 L0C = 128KB | `block_M × block_N × sizeof(accum) > 128KB` 导致 segfault | 设计 block 时满足 `block_M × block_N ≤ 16384`（float32 accum） |

## 2. 强制检测规则

在规格说明书生成前，**必须**执行以下检测，结论写入 `CC-4`：

| 检测项 | 触发条件 | 处理方式 | 归属条款 |
|--------|----------|----------|----------|
| 三维 Kernel | 参考实现包含 `T.Kernel(..., batch_count)` 或 3 个维度参数 | **立即警告**，提出 改成一维 方案 | CC-4 / ID-4 |
| GPU 专用 API | CUDA 相关 API（如 `T.gemm` 通用版） | **立即警告**，查阅本项目确认 Ascend API | CC-4 / ID-1 |
| GEMM 非整除风险 | `M` 或 `N` 不被 block size 整除（即 `M % block_M ≠ 0` 或 `N % block_N ≠ 0`） | **立即警告**，要求 `ID-3` 中明确 padding 策略 | CC-4 / ID-3 |
| L0C 溢出风险 | block_M × block_N × sizeof(accum_dtype) > 131072 (128KB) | **立即警告**，建议减小 block 或拆分 | CC-5 / ID-3 |

## 3. 警告输出格式

```
⚠️ 技术限制检测警告（写入 CC-4）

检测到参考实现包含本项目不支持的功能：

1. 三维 Kernel（本项目只支持一维 Kernel）
   - 参考实现：T.Kernel(m_num, n_num, batch_count)
   - 本项目方案：T.Kernel(total_blocks)
   - 参考：examples/gemm/matmul.py

建议：
- 先查阅本项目 examples/ 中的同类实现
- 确认 Ascend API 用法后再生成规格条款（CC-4 / ID-1 / ID-4）

是否继续生成规格说明书？
```
