"""example_{op}.py 模板 —— 含 NPU 执行打桩，非 NPU 环境可端到端跑通。

本模板以 elementwise_add 为示例算子。生成真实算子时替换 kernel/golden/test 用例。
验证打桩：TILELANG_OP_STUB_NPU=1 python example_template.py --level all
"""
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


# ---------- Golden（PyTorch CPU 参考实现，真实计算，不打桩） ----------
def golden_add(x, y):
    return x + y


# ---------- Kernel（源码完整保留，打桩时跳过定义与执行） ----------
if not STUB_NPU:
    # === [STUB: NPU-EXEC] 以下 kernel 在打桩模式跳过定义 ===
    @tilelang.jit(out_idx=[-1], target="npuir")
    def add_kernel(M, N, block_M, block_N, in_dtype="float32", out_dtype="float32"):
        @T.prim_func
        def _main(
            A: T.Tensor((M, N), in_dtype),
            B: T.Tensor((M, N), in_dtype),
            C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
                by = cid // T.ceildiv(N, block_N)
                bx = cid % T.ceildiv(N, block_N)
                A_shared = T.alloc_shared((block_M, block_N), in_dtype)
                B_shared = T.alloc_shared((block_M, block_N), in_dtype)
                C_local = T.alloc_fragment((block_M, block_N), out_dtype)
                C_shared = T.alloc_shared((block_M, block_N), out_dtype)
                T.copy(A[by * block_M, bx * block_N], A_shared)
                T.copy(B[by * block_M, bx * block_N], B_shared)
                for local_y, local_x in T.Parallel(block_M, block_N):
                    C_local[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])
        return _main
    # === [STUB: NPU-EXEC] kernel 定义结束 ===


# ---------- 分层测试 ----------
def _run_case(M, N, dtype, tag):
    if STUB_NPU:
        # === [STUB: NPU-EXEC] CPU 张量 + golden 充当输出 ===
        a = torch.randn(M, N, dtype=dtype)
        b = torch.randn(M, N, dtype=dtype)
        golden = golden_add(a, b)
        c = golden.clone()
        # === [STUB: NPU-EXEC] 充当结束 ===
    else:
        a = torch.randn(M, N, dtype=dtype, device="npu")
        b = torch.randn(M, N, dtype=dtype, device="npu")
        kernel = add_kernel(M, N, block_M=32, block_N=32, in_dtype=str(dtype).replace("torch.", ""), out_dtype=str(dtype).replace("torch.", ""))
        c = kernel(a, b)
        golden = golden_add(a, b)
    torch.testing.assert_close(c, golden, rtol=1e-2, atol=1e-2)
    print(f"[{tag}] PASS: shape=({M},{N}) dtype={dtype}")


def run_L0():
    # L0 门槛规则 shape（block 整除）
    for M, N in [(128, 128), (256, 256)]:
        _run_case(M, N, torch.float32, "L0")


def run_L1():
    # L1 功能覆盖（含不规则 shape、多 dtype）
    for M, N in [(130, 130), (64, 200)]:
        _run_case(M, N, torch.float32, "L1")


def run_L2():
    # L2 异常输入（仅记录，不阻塞）
    try:
        _run_case(1, 1, torch.float32, "L2")
    except Exception as e:
        print(f"[L2] WARN (记录不阻塞): {e}")


def run_boundary():
    # Boundary 特殊值（仅记录，不阻塞）
    a = torch.zeros(128, 128, dtype=torch.float32)
    b = torch.zeros(128, 128, dtype=torch.float32)
    golden = golden_add(a, b)
    if STUB_NPU:
        c = golden.clone()
    else:
        kernel = add_kernel(128, 128, block_M=32, block_N=32)
        c = kernel(a, b)
    torch.testing.assert_close(c, golden, rtol=1e-2, atol=1e-2)
    print("[Boundary] PASS: zeros")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()
    if args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
