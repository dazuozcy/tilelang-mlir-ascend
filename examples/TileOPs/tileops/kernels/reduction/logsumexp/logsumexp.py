"""LogSumExp forward kernel using TileLang (NPU-adapted).

Implements a 2-pass online algorithm for:
  - logsumexp: y[i] = max_i + log(sum_i(exp(x[i,j] - max_i)))

Supports arbitrarily large N dimensions by tiling over N when the full
N_padded does not fit in shared memory.  Uses the online softmax recurrence
(track running max and rescaled running sum) across N-tiles.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel functions (``_logsumexp_kernel_single``,
    ``_logsumexp_kernel_tiled``) are extracted from the GPU repo via
    ``extract_tl_kernel.py`` and imported as-is.  They serve as the
    reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 adaptations (decorator, grid/sync,
    ``threads`` removal, padding strategy) are handled by the NPU
    component during re-implementation.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K6: ``device_smem_budget`` imported from ``_primitives`` (backend-agnostic).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: ``autotune_configs`` / ``autotune()`` / ``_tile_n_candidates`` /
        ``_MAX_TILE_N_CANDIDATES`` / ``tune`` param -- all removed.
    K9: ``threads`` removed from ``default_config`` and ``forward`` call.
"""

import functools
from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.reduction._primitives import (
    DEFAULT_ALIGNMENT,
    MAX_SINGLE_TILE_COLS,
    align_up,
    compute_tile_n,
    device_smem_budget,
)

from . import _log_sum_exp_fwd_kernels as _extracted_mod
# from ._log_sum_exp_fwd_kernels import (
#     _logsumexp_kernel_single,
#     _logsumexp_kernel_tiled,
# )

from ._logsumexp_kernel_single._logsumexp_kernel_single import _logsumexp_kernel_single
from ._logsumexp_kernel_tiled._logsumexp_kernel_tiled import _logsumexp_kernel_tiled

__all__ = ["LogSumExpKernel"]

# Make align_up available to the extracted module's namespace so the GPU
# reference functions can resolve it at runtime.  The extraction script
# emits @tilelang.jit functions and module-level constants but not helper
# functions like align_up.
_extracted_mod.align_up = align_up


# ---------------------------------------------------------------------------
# Dispatcher (not extracted by the script -- defined here)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _logsumexp_kernel(M: int, N: int, dtype: str, tile_n: int = 0):
    """Build the appropriate logsumexp kernel.

    Dispatches between the single-tile path (tile_n == 0, N fits in
    shared memory) and the multi-tile path (tile_n > 0, N tiled over
    shared memory using online softmax recurrence).
    """
    if tile_n == 0:
        return _logsumexp_kernel_single(M, N, dtype)
    return _logsumexp_kernel_tiled(M, N, dtype, tile_n)


def _compute_padded_cols(N: int, tile_n: int) -> int:
    """Compute the total column count (may exceed N_padded for tiled path)."""
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    if tile_n == 0:
        return N_padded
    num_tiles = (N_padded + tile_n - 1) // tile_n
    return num_tiles * tile_n


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: threads removed)
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
def _(M: int, N: int, dtype_str: str, block_m: int, tile_n: int, x: torch.Tensor):
    return torch.empty((M,), dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Kernel class (K5-K9 adaptations)
# ---------------------------------------------------------------------------


def _elem_bytes(dtype: torch.dtype) -> int:
    """Return bytes per element for the given dtype."""
    return torch.tensor([], dtype=dtype).element_size()


class LogSumExpKernel(Kernel):
    """LogSumExp forward kernel.

    Supports all architectures (NPU adaptation K5: ``supported_archs = None``).
    Uses 256-element alignment for shared memory copies. Implements a 2-pass
    online algorithm.

    For large N that does not fit in shared memory, tiles over N using
    the online softmax recurrence (running max + rescaled sum).

    Boundary handling for non-aligned N is performed inside the kernel
    via masked loads and ``-inf`` fills, so no host-side ``F.pad`` is
    needed.

    NPU adaptation (K8): autotune has been removed; the kernel uses
    heuristic config selection only.  ``init_config(config)`` takes no
    ``tune`` argument.

    Args:
        M: Number of rows (product of all dims except last).
        N: Hidden dimension (last dim).
        op_kind: Must be "logsumexp" (kept for API consistency with SoftmaxKernel).
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional kernel configuration dict.
        device_index: Device index for shared memory budget query.
            When ``None``, the current device is used.
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
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

        # Build self.kernel BEFORE init_config: the config selection
        # logic references self._tile_n which is derived from the kernel.
        #
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

        # Apply post-init tile_n fixup for user-provided configs.
        # (K8: no tune branch -- always apply.)
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
            block_m,
            self._elem_bytes,
            self.N_padded,
            budget=effective_budget,
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

        K9: ``threads`` removed from the config dict.
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

        K9: ``threads`` removed from the call.
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
