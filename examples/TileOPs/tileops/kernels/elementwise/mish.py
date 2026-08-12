"""Mish forward kernel using TileLang.

Extracted from ``tileops/kernels/elementwise.py`` (``MishFwdKernel``) and
adapted for NPU (Ascend/TileLang) backend.

Mish is a unary elementwise activation: ``y = x * tanh(softplus(x))`` =
``x * tanh(log(1 + exp(x)))``.

Adaptation summary:
  1. **TileLang kernel factory function** (``_mish_kernel``) — interface-only
     stub.  The GPU version uses the shared ``UnaryKernel`` template factories
     (``_make_unary_direct`` / ``_make_unary_explicit`` / ``_make_unary_regcopy``)
     with ``threads`` + ``num_per_thread`` (npt) parameters.  On NPU, the
     ``threads * npt`` product collapses into a single ``block_size`` parameter
     (K11); the factory callable signature is ``_func(block_size)``.

  2. **MishFwdKernel class** — config selection and ``forward`` dispatch are
     preserved.  Autotune has been removed; the kernel uses heuristic config
     only.  GPU-specific parts adapted:
        - ``supported_archs``: ``[80, 86, 89, 90]`` (CUDA SM) → ``None``
          (all architectures).
        - ``threads`` + ``num_per_thread`` → ``block_size`` (K11).
        - ``autotune_configs`` / ``autotune()`` / ``tune`` param → removed (K8).
"""

import functools
import os
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["MishFwdKernel"]

os.environ.setdefault("TILELANG_ASCEND_MODE", "Dev")
# tilelang.cache.clear_cache()

# ---------------------------------------------------------------------------
# TileLang kernel factory functions — interfaces preserved, implementations
# emptied.  To be implemented by the NPU kernel component.
# ---------------------------------------------------------------------------


# @functools.lru_cache(maxsize=32)
# def _mish_kernel(N: int, dtype: str):
#     """Build a flat Mish kernel.

#     Uses ``@tilelang.jit(target="npuir")`` + ``@T.prim_func``.
#     The returned callable signature is ``_func(block_size)`` (no ``threads``
#     or ``npt`` — the GPU ``threads * npt`` product collapses into a single
#     ``block_size`` per K11).  The grid is ``ceildiv(N, block_size)`` blocks
#     (NPU ``is_npu=True`` kernels have no ``threads`` dimension).

#     Algorithm (per element, computed in fp32 for precision):
#       mish(x) = x * tanh(log(1 + exp(x)))

#     The GPU ``op_func`` casts to fp32 intermediates to avoid fp16/bf16
#     precision loss in ``exp`` / ``log`` / ``tanh``.  The NPU implementation
#     should do the same: cast input to fp32, compute mish, cast back to the
#     output dtype.
#     """
#     raise NotImplementedError("NPU kernel implementation pending")


@functools.lru_cache(maxsize=32)
def _mish_kernel(N, dtype, output_dtype=None, num_inner=4, num_stages=2):
    out_dtype = output_dtype or dtype
    if num_stages is None:
        num_stages = 2

    @tilelang.jit(out_idx=[1], target="npuir")
    def kernel(inner_tile):
        block_size = inner_tile * num_inner

        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            y: T.Tensor((N,), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), is_npu=True) as (cid, _):
                if dtype != "float32":
                    # --- float16 / bfloat16 path: 4 UB buffers at inner_tile ---
                    # xy_ub: f16 input load -> f16 output store (reused)
                    # x_f32: upcast input -> final vmul result (in-place dst=src1)
                    # work_a: exp -> +1 -> sqr -> +1 = den
                    # work_b: -1 = num -> /den -> *x = y_f32 (in-place)
                    xy_ub = T.alloc_ub((inner_tile,), dtype)
                    x_f32 = T.alloc_ub((inner_tile,), "float32")
                    work_a = T.alloc_ub((inner_tile,), "float32")
                    work_b = T.alloc_ub((inner_tile,), "float32")

                    for i in T.Pipelined(num_inner, num_stages=num_stages):
                        offset = cid * block_size + i * inner_tile
                        # Safe tail: T.max(0, ...) for out-of-bounds inner tiles
                        inner_tail = T.max(
                            0, T.min(inner_tile, N - offset))

                        # GM -> UB (copy only valid elements)
                        T.copy(x[offset : offset + inner_tail],
                               xy_ub[0:inner_tail])

                        # Upcast to float32 for numerical stability
                        T.vcast(xy_ub, x_f32, round_mode="rint")

                        # Core mish (algebraic identity, in-place ops).
                        # tanh(softplus(x)) = (t2²-1)/(t2²+1), t2 = 1+exp(x)
                        # avoids T.vtanh (Taylor divergence on 1-D UB, see
                        # Stage 3 DESIGN.md §3.3 fallback note).
                        T.vexp(x_f32, work_a)             # work_a = exp(x)
                        T.vadd(work_a, 1.0, work_a)       # work_a = 1 + exp(x) = t2
                        T.vmul(work_a, work_a, work_a)    # work_a = t2² (in-place)
                        T.vsub(work_a, 1.0, work_b)       # work_b = t2² - 1 = num
                        T.vadd(work_a, 1.0, work_a)       # work_a = t2² + 1 = den
                        T.vdiv(work_b, work_a, work_b)    # work_b = num/den
                        T.vmul(x_f32, work_b, x_f32)      # x_f32 = x * tanh(sp) (in-place)

                        # Downcast back to original dtype
                        T.vcast(x_f32, xy_ub, round_mode="round")

                        # UB -> GM (copy only valid elements)
                        T.copy(xy_ub[0:inner_tail],
                               y[offset : offset + inner_tail])
                else:
                    # --- float32 path: 3 UB buffers at inner_tile ---
                    # xy_ub: input load -> final vmul result (in-place dst=src1)
                    # work_a: exp -> +1 -> sqr -> +1 = den
                    # work_b: -1 = num -> /den = tanh_sp
                    xy_ub = T.alloc_ub((inner_tile,), "float32")
                    work_a = T.alloc_ub((inner_tile,), "float32")
                    work_b = T.alloc_ub((inner_tile,), "float32")

                    for i in T.Pipelined(num_inner, num_stages=num_stages):
                        offset = cid * block_size + i * inner_tile
                        inner_tail = T.max(
                            0, T.min(inner_tile, N - offset))

                        # GM -> UB
                        T.copy(x[offset : offset + inner_tail],
                               xy_ub[0:inner_tail])

                        # Core mish (algebraic identity, in-place ops)
                        T.vexp(xy_ub, work_a)             # work_a = exp(x)
                        T.vadd(work_a, 1.0, work_a)       # work_a = 1 + exp(x) = t2
                        T.vmul(work_a, work_a, work_a)    # work_a = t2² (in-place)
                        T.vsub(work_a, 1.0, work_b)       # work_b = t2² - 1 = num
                        T.vadd(work_a, 1.0, work_a)       # work_a = t2² + 1 = den
                        T.vdiv(work_b, work_a, work_b)    # work_b = num/den
                        T.vmul(xy_ub, work_b, xy_ub)      # xy_ub = x * tanh(sp) (in-place)

                        # UB -> GM
                        T.copy(xy_ub[0:inner_tail],
                               y[offset : offset + inner_tail])

        return main

    return kernel

# ---------------------------------------------------------------------------
# custom_op wrapper for torch.compile compatibility
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::mish_fwd", mutates_args=())
def _mish_fwd_wrapped(
    N: int,
    dtype_str: str,
    block_size: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _mish_kernel(N, dtype_str)(block_size)(x)


@_mish_fwd_wrapped.register_fake
def _(N: int, dtype_str: str, block_size: int, x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


# ---------------------------------------------------------------------------
# Kernel class — main flow preserved, GPU-specific parts adapted
# ---------------------------------------------------------------------------

# Default block_size values — the GPU ``threads * npt`` product (K11).
# GPU default: threads=256, npt=4 (fp32) / npt=8 (fp16/bf16, register_copy).
_BLOCK_SIZE_FP32 = 1024       # 256 * 4
_BLOCK_SIZE_NON_FP32 = 2048  # 256 * 8


class MishFwdKernel(Kernel):
    """Mish forward kernel: ``y = x * tanh(softplus(x))``.

    A flat 1-D elementwise kernel.  The input tensor is flattened to
    ``(N_total,)`` by the Op layer before dispatch.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Data type (float16, bfloat16, or float32).
        config: Optional kernel configuration dict (e.g. ``{"block_size": 2048}``).
        device_index: Device index (unused on NPU, kept for API consistency).
    """

    # NPU adaptation: [80, 86, 89, 90] (CUDA SM) → None (all architectures).
    supported_archs: Optional[list] = None
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        # Factory call — will raise NotImplementedError until the NPU kernel
        # component fills in ``_mish_kernel``.
        self.kernel = _mish_kernel(self.N_total, self.dtype_str)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config with ``block_size`` (K11: threads * npt).

        GPU default: ``threads=256, npt=4`` (fp32) / ``npt=8`` (fp16/bf16,
        register_copy strategy).  The product collapses to ``block_size``.
        """
        if self.dtype == torch.float32:
            return {"block_size": _BLOCK_SIZE_FP32}
        return {"block_size": _BLOCK_SIZE_NON_FP32}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the Mish kernel on a flat ``(N_total,)`` tensor."""
        return _mish_fwd_wrapped(
            self.N_total,
            self.dtype_str,
            self.config["block_size"],
            x,
        )
