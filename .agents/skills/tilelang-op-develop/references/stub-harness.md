# NPU 执行打桩参考（stub-harness）

> 本文档定义 `example_{op}.py` 中 NPU 执行过程的打桩规范。所有打桩点用 `[STUB: NPU-EXEC]` 标记，便于后续取消。

## 1. 为什么打桩

当前环境无 NPU 设备：
- `tilelang` 依赖 `tvm`，未安装 → `import tilelang` 失败
- `torch_npu` 未安装 → `.npu()` / `device="npu"` 失败
- 仅有 CPU `torch` 可用

为验证"设计 → 检视 → 开发 → 调优"端到端流程可行性，对 **kernel 编译 + NPU 执行** 打桩，用 CPU golden 充当 kernel 输出，使精度校验可跑通。

## 2. 打桩开关

```python
import os
STUB_NPU = os.environ.get("TILELANG_OP_STUB_NPU", "0") == "1"
```

- `TILELANG_OP_STUB_NPU=1` → 打桩（非 NPU 环境）
- `TILELANG_OP_STUB_NPU=0` 或未设置 → 真实 NPU 路径

## 3. 打桩区模板

### 3.1 顶部 imports + 开关

```python
import os
import argparse
import torch

# === [STUB: NPU-EXEC] 打桩区开始 ===
# 非 NPU 环境设 TILELANG_OP_STUB_NPU=1 跳过 NPU 编译与执行，
# 用 CPU golden 充当 kernel 输出以验证端到端流程。
# 真实 NPU 环境设 TILELANG_OP_STUB_NPU=0 或不设置即可走真实路径。
STUB_NPU = os.environ.get("TILELANG_OP_STUB_NPU", "0") == "1"
if STUB_NPU:
    print("[STUB] NPU 执行打桩已启用（TILELANG_OP_STUB_NPU=1），kernel 输出由 golden 充当")
    tilelang = None
    T = None
else:
    import tilelang
    import tilelang.language as T
# === [STUB: NPU-EXEC] 打桩区结束 ===
```

### 3.2 kernel 定义（源码完整保留，仅打桩时不定义/不执行）

```python
if not STUB_NPU:
    # === [STUB: NPU-EXEC] 以下 kernel 在打桩模式跳过定义 ===
    @tilelang.jit(out_idx=[-1], target="npuir")
    def {op}_kernel(M, N, block_M, block_N, in_dtype="float32", out_dtype="float32"):
        @T.prim_func
        def _main(
            A: T.Tensor((M, N), in_dtype),
            B: T.Tensor((M, N), in_dtype),
            C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
                # ... 按 DESIGN.md 真实实现 ...
                pass
        return _main
    # === [STUB: NPU-EXEC] kernel 定义结束 ===
```

> 打桩时 `tilelang`/`T` 为 None，kernel 函数体含 `T.prim_func` 等无法解析，因此用 `if not STUB_NPU:` 守卫跳过定义。**源码本身完整保留**，真实环境取消打桩即生效。

### 3.3 测试执行分支

```python
def run_L0():
    # L0 规则 shape（block 整除）
    cases = [(128, 128), (256, 256)]
    for M, N in cases:
        if STUB_NPU:
            # === [STUB: NPU-EXEC] CPU 张量 + golden 充当输出 ===
            a = torch.randn(M, N, dtype=torch.float32)
            b = torch.randn(M, N, dtype=torch.float32)
            golden = golden_{op}(a, b)
            c = golden.clone()
            # === [STUB: NPU-EXEC] 充当结束 ===
        else:
            a = torch.randn(M, N, dtype=torch.float32, device="npu")
            b = torch.randn(M, N, dtype=torch.float32, device="npu")
            kernel = {op}_kernel(M, N, block_M=32, block_N=32)
            c = kernel(a, b)
            golden = golden_{op}(a, b)
        torch.testing.assert_close(c, golden, rtol=1e-2, atol=1e-2)
        print(f"[L0] PASS: shape=({M},{N})")
```

## 4. 取消打桩步骤

1. 在真实 NPU 环境确保 `tilelang`、`torch_npu` 可导入。
2. 设置 `TILELANG_OP_STUB_NPU=0`（或不设置）。
3. 全局搜索 `[STUB: NPU-EXEC]`，确认所有打桩区在 `STUB_NPU=False` 时走真实分支。
4. 若要永久删除打桩代码：删除所有 `[STUB: NPU-EXEC]` 标记区与 `STUB_NPU` 开关，保留 `else` 分支为唯一路径。

## 5. 注意事项

- **精度校验在打桩模式下恒通过**（kernel 输出 = golden），因此打桩模式仅验证流程通路，不验证真实精度。真实精度须在 NPU 环境取消打桩后验证。
- **golden 函数必须真实计算**，不得打桩——它是真实 NPU 环境下的精度基准。
- **kernel 源码不得简化**——打桩只跳过执行，不跳过生成。
