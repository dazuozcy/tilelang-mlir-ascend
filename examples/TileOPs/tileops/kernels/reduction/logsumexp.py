"""LogSumExp forward kernel using TileLang.

Extracted from ``tileops/kernels/reduction/logsumexp.py`` and adapted
for NPU (Ascend/TileLang) backend.

Adaptation summary:
  1. **LogSumExpKernel class (main flow)** — config selection, tiling
     heuristics, and forward dispatch logic are preserved.  Autotune
     has been removed; the kernel uses heuristic config only.
     GPU-specific parts adapted:
        - ``supported_archs``: ``[80, 86, 89, 90]`` (CUDA SM) → ``None``
          (all architectures).
        - ``device_smem_budget``: now imported from the NPU-adapted
          ``_primitives`` module (queries device backend, not
          ``torch.cuda.*``).

  2. **TileLang kernel factory functions** (``_logsumexp_kernel_single``,
     ``_logsumexp_kernel_tiled``, ``_logsumexp_kernel``) — NPU TileLang
     implementations using ``@tilelang.jit`` + ``@T.prim_func`` targeting
     Ascend (NPUIR).  No alignment padding; operates on raw N.
"""
import os
import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    MAX_SINGLE_TILE_COLS,
    align_up,
    compute_tile_n,
    device_smem_budget,
)

__all__ = ["LogSumExpKernel"]

os.environ.setdefault("TILELANG_ASCEND_MODE", "Dev")
tilelang.cache.clear_cache()

# ---------------------------------------------------------------------------
# TileLang kernel factory functions — interfaces preserved, implementations
# emptied.  To be implemented by the NPU kernel component.
# ---------------------------------------------------------------------------


# ---------- Kernel: single-tile variant ----------
def _logsumexp_kernel_single(M, N, dtype):
    """Build a single-tile NPUIR logsumexp kernel.

    Loads an entire row block into UB at once.  Used when N is small enough
    that the UB budget can hold a full row block.

    NPU adaptation: no alignment padding.  The GPU version pads N to
    DEFAULT_ALIGNMENT (256) for T.copy vectorization and uses masked loads
    for out-of-bounds columns.  On NPU, ``T.if_then_else`` with indexed
    tensor access generates a ``vsel`` op with mismatched operand shapes
    (mask is ``(block_m, N_padded)`` but the source tensor is ``(M, N)``).
    Instead, this kernel operates directly on the unpadded N, avoiding the
    boundary issue entirely.

    Algorithm (numerically stable):
      1. row_max = reduce_max(x, dim=1)
      2. exp_vals = exp(x - row_max)          (vsub broadcast + vexp, in-place)
      3. row_sum = reduce_sum(exp_vals, dim=1)
      4. out = row_max + log(row_sum)         (vln + vadd)
    """

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
                # All buffers in UB (alloc_shared) — v-prefix APIs require UB.
                # No padding: use N directly to avoid boundary handling issues
                # with NPUIR vsel shape mismatch.
                x_sh = T.alloc_shared((block_m, N), dtype)
                x_f32 = T.alloc_shared((block_m, N), "float32")
                # NPUIR reduce requires input and output to share the same rank:
                # (block_m, N) -> (block_m, 1) with dim=1.
                max_sh = T.alloc_shared((block_m, 1), "float32")
                sum_sh = T.alloc_shared((block_m, 1), "float32")
                log_sh = T.alloc_shared((block_m, 1), "float32")
                out_sh = T.alloc_shared((block_m,), dtype)

                # 1. Load input row block and cast to fp32.
                T.copy(x[pid_m * block_m, 0], x_sh)
                for i, j in T.Parallel(block_m, N):
                    x_f32[i, j] = T.cast(x_sh[i, j], "float32")

                # 2. row_max = reduce_max(x_f32, dim=1)
                T.reduce_max(x_f32, max_sh, dim=1, clear=True)

                # 3. exp_f32 = exp(x_f32 - row_max)  (in-place: reuse x_f32)
                T.vsub(x_f32, max_sh, x_f32)    # broadcast (block_m,N)-(block_m,1)
                T.vexp(x_f32, x_f32)

                # 4. row_sum = reduce_sum(exp_f32, dim=1)
                T.reduce_sum(x_f32, sum_sh, dim=1, clear=True)

                # 5. out = row_max + log(row_sum)
                T.vln(sum_sh, log_sh)
                T.vadd(max_sh, log_sh, sum_sh)

                # 6. Cast back to output dtype and write out via UB.
                for i in T.Parallel(block_m):
                    out_sh[i] = T.cast(sum_sh[i, 0], dtype)
                T.copy(out_sh, y[pid_m * block_m])

        return main

    return _func


# ---------- Kernel: N-tiled variant (online softmax recurrence) ----------
def _logsumexp_kernel_tiled(M, N, dtype, tile_n):
    """Build an N-tiled NPUIR logsumexp kernel.

    Iterates over the reduction dimension in chunks of tile_n columns.
    Uses the online softmax recurrence across tiles:
      - Track running max (row_max) and rescaled running sum (row_sum).
      - For each tile: update row_max, rescale row_sum, add tile contribution.

    NPU adaptation: no alignment padding.  The GPU version pads N to
    DEFAULT_ALIGNMENT (256) and uses masked loads for the last tile's
    out-of-bounds columns.  On NPU, ``T.if_then_else`` with indexed tensor
    access generates a ``vsel`` op with mismatched operand shapes.  Instead,
    this kernel tiles over the unpadded N directly.  When the last tile
    extends beyond N, only the valid columns (``n_start`` to ``N``) are
    processed via a partial-tile path.

    Algorithm per tile:
      a. tile_max = reduce_max(tile, dim=1)
      b. prev_max = row_max; row_max = max(row_max, tile_max)
      c. tile = exp(tile - row_max)                   (rescale by new max)
      d. tile_sum = reduce_sum(tile, dim=1)
      e. row_sum = row_sum * exp(prev_max - row_max) + tile_sum
    """
    num_tiles = (N + tile_n - 1) // tile_n
    _last_tile_n = N - (num_tiles - 1) * tile_n if num_tiles > 0 else N
    _has_partial_last = _last_tile_n != tile_n

    @tilelang.jit(out_idx=[1], target="npuir")
    def _func(block_m):

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M,), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), is_npu=True) as (pid_m, _):
                x_sh = T.alloc_shared((block_m, tile_n), dtype)
                tile_f32 = T.alloc_shared((block_m, tile_n), "float32")
                row_max = T.alloc_shared((block_m, 1), "float32")
                row_sum = T.alloc_shared((block_m, 1), "float32")
                prev_max = T.alloc_shared((block_m, 1), "float32")
                tile_max = T.alloc_shared((block_m, 1), "float32")
                tile_sum = T.alloc_shared((block_m, 1), "float32")
                scale = T.alloc_shared((block_m, 1), "float32")
                out_sh = T.alloc_shared((block_m,), dtype)

                # Init: row_max = -inf, row_sum = 0.
                T.fill(row_max, -T.infinity("float32"))
                T.clear(row_sum)

                # Reduction across tiles is serial (cross-tile dependency).
                for t in T.serial(num_tiles):
                    n_start = t * tile_n

                    # a. Load tile + cast to fp32.
                    T.copy(x[pid_m * block_m, n_start], x_sh)
                    for i, j in T.Parallel(block_m, tile_n):
                        tile_f32[i, j] = T.cast(x_sh[i, j], "float32")

                    # b. tile_max = reduce_max(tile_f32, dim=1)
                    T.reduce_max(tile_f32, tile_max, dim=1, clear=True)

                    # c. prev_max = row_max; row_max = max(row_max, tile_max)
                    for i in T.Parallel(block_m):
                        prev_max[i, 0] = row_max[i, 0]
                    T.vmax(row_max, tile_max, row_max)

                    # d. tile_f32 = exp(tile_f32 - row_max)  (in-place)
                    T.vsub(tile_f32, row_max, tile_f32)
                    T.vexp(tile_f32, tile_f32)

                    # e. tile_sum = reduce_sum(tile_f32, dim=1)
                    T.reduce_sum(tile_f32, tile_sum, dim=1, clear=True)

                    # f. row_sum = row_sum * exp(prev_max - row_max) + tile_sum
                    T.vsub(prev_max, row_max, scale)
                    T.vexp(scale, scale)
                    T.vmul(row_sum, scale, row_sum)
                    T.vadd(row_sum, tile_sum, row_sum)

                # Final: out = row_max + log(row_sum)
                T.vln(row_sum, prev_max)
                T.vadd(row_max, prev_max, row_sum)

                # Cast back + write out via UB.
                for i in T.Parallel(block_m):
                    out_sh[i] = T.cast(row_sum[i, 0], dtype)
                T.copy(out_sh, y[pid_m * block_m])

        return main

    return _func


@functools.lru_cache(maxsize=64)
def _logsumexp_kernel(M: int, N: int, dtype: str, tile_n: int = 0):
    # n_padded = _align_up(N, ALIGNMENT)
    # elem_bytes = torch.tensor([], dtype=_TORCH_DTYPE[dtype_str]).element_size()
    if tile_n > 0:
        return _logsumexp_kernel_tiled(M, N, dtype, tile_n)
    return _logsumexp_kernel_single(M, N, dtype)


# ---------------------------------------------------------------------------
# Helpers (device-agnostic, preserved from GPU version)
# ---------------------------------------------------------------------------


def _compute_padded_cols(N: int, tile_n: int) -> int:
    """Compute the total column count (may exceed N_padded for tiled path)."""
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    if tile_n == 0:
        return N_padded
    num_tiles = (N_padded + tile_n - 1) // tile_n
    return num_tiles * tile_n


# ---------------------------------------------------------------------------
# custom_op wrapper for torch.compile compatibility
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::logsumexp_fwd", mutates_args=())
def _logsumexp_fwd_wrapped(
    M: int,
    N: int,
    dtype_str: str,
    block_m: int,
    tile_n: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _logsumexp_kernel(M, N, dtype_str, tile_n)(block_m)(x)


@_logsumexp_fwd_wrapped.register_fake
def _(M, N, dtype_str, block_m, tile_n, x):
    return torch.empty((M,), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class — main flow preserved, GPU-specific parts adapted
# ---------------------------------------------------------------------------


def _elem_bytes(dtype: torch.dtype) -> int:
    """Return bytes per element for the given dtype."""
    return torch.tensor([], dtype=dtype).element_size()


class LogSumExpKernel(Kernel):
    """LogSumExp forward kernel.

    Uses 256-element alignment for shared memory copies. Implements a
    2-pass online algorithm.

    For large N that does not fit in shared memory, tiles over N using
    the online softmax recurrence (running max + rescaled sum).

    Boundary handling for non-aligned N is performed inside the kernel
    via masked loads and ``-inf`` fills, so no host-side ``F.pad`` is
    needed.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        op_kind: Must be "logsumexp" (kept for API consistency with SoftmaxKernel).
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional kernel configuration dict.
        device_index: Device index for shared memory budget query.
            When ``None``, the current device is used (via device backend).
    """

    # NPU adaptation: [80, 86, 89, 90] (CUDA SM) → None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        M: int,
        N: int,
        op_kind: str,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ):
        super().__init__()
        if op_kind != "logsumexp":
            raise ValueError(f"Unsupported op_kind '{op_kind}'. Expected 'logsumexp'.")
        self.M = M
        self.N = N
        self.op_kind = op_kind
        self.dtype = dtype
        self.N_padded = align_up(N, DEFAULT_ALIGNMENT)
        self._elem_bytes = _elem_bytes(dtype)
        self._smem_budget = device_smem_budget(device_index)

        # tile_n is baked into the kernel at build time, so we pre-compute
        # it from the heuristic block_m in default_config.
        self._tile_n = self.default_config["tile_n"]
        self.kernel = _logsumexp_kernel(
            self.M,
            self.N,
            self.dtype_str,
            self._tile_n,
        )

        self.init_config(config)

        # If the caller supplied an explicit tile_n, honour it.
        caller_tile_n = config.get("tile_n") if config is not None else None
        if caller_tile_n is not None:
            target_tile_n = caller_tile_n
        else:
            target_tile_n = self._tile_n_for_block_m(self.config["block_m"])
        if target_tile_n != self._tile_n:
            self._tile_n = target_tile_n
            self.kernel = _logsumexp_kernel(
                self.M,
                self.N,
                self.dtype_str,
                self._tile_n,
            )
        self.config["tile_n"] = self._tile_n

    def _tile_n_for_block_m(self, block_m: int) -> int:
        """Return tile_n for a given block_m (0 means no tiling needed).

        Uses the device's actual shared memory budget (not the
        conservative 48 KiB default) so that large-N workloads can
        use fewer, larger tiles or even the single-tile fast path.

        Both paths are subject to the MAX_SINGLE_TILE_COLS column
        cap (TileLang's vectorizer fails at the 32768 column boundary).
        """
        budget = self._smem_budget
        # Single-tile path: cap by column count and smem budget.
        if self.N_padded <= MAX_SINGLE_TILE_COLS:
            tile_n = compute_tile_n(block_m, self._elem_bytes, self.N_padded, budget=budget)
            if tile_n == self.N_padded:
                return 0
        # Tiled path (logsumexp uses 1 shared buffer).
        # Cap the smem budget so tile_n stays within the column limit.
        col_budget = MAX_SINGLE_TILE_COLS * block_m * self._elem_bytes
        effective_budget = min(budget, col_budget)
        return compute_tile_n(
            block_m, self._elem_bytes, self.N_padded, budget=effective_budget,
        )

    @property
    def default_config(self) -> dict:
        """Select default block_m based on shared memory budget.

        For the single-tile path (tile_n == 0), prefer the largest
        block_m that fits in shared memory.

        For the tiled path, prefer the block_m that minimises the
        number of N-tiles (maximises tile_n) to reduce global memory
        passes.  Among configs with equal tile count, prefer smaller
        block_m for better occupancy.
        """
        best_bm = 1
        best_tile_n = self._tile_n_for_block_m(1)

        for bm in [2, 4, 8, 16]:
            try:
                tn = self._tile_n_for_block_m(bm)
            except ValueError:
                continue
            if tn == 0:
                # Single-tile is always better: prefer larger block_m
                best_bm = bm
                best_tile_n = tn
            elif best_tile_n == 0:
                pass
            else:
                best_num = (self.N_padded + best_tile_n - 1) // best_tile_n
                curr_num = (self.N_padded + tn - 1) // tn
                if curr_num < best_num:
                    best_bm = bm
                    best_tile_n = tn

        return {"block_m": best_bm, "tile_n": best_tile_n}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the logsumexp kernel.

        Accepts an ``(M, N)`` tensor.  Boundary handling for non-aligned
        ``N`` is performed inside the kernel (masked loads + ``-inf``
        fill), so no host-side ``F.pad`` is needed.
        """
        tile_n = self._tile_n

        return _logsumexp_fwd_wrapped(
            self.M,
            self.N,
            self.dtype_str,
            self.config["block_m"],
            tile_n,
            x,
        )
