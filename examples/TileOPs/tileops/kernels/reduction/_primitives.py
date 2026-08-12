"""Shared reduction primitives for reduction kernels.

Extracted from ``tileops/kernels/reduction/_primitives.py`` with one
GPU-to-NPU adaptation:

1. ``device_smem_budget`` — queries device properties via the device
   backend abstraction instead of hard-coded ``torch.cuda.*`` calls.
   On NPU, falls back to ``SHARED_MEMORY_BUDGET_BYTES`` when the device
   properties do not expose shared-memory attributes.

``tune_by_forward`` has been removed (autotune is not used on NPU).

All T.macro factories (``make_*``) are preserved verbatim — they are
TileLang DSL code, not GPU-specific.
"""

import tilelang.language as T
import torch

__all__ = [
    "DEFAULT_ALIGNMENT",
    "MAX_SINGLE_TILE_COLS",
    "SHARED_MEMORY_BUDGET_BYTES",
    "align_up",
    "compute_tile_n",
    "device_smem_budget",
    "make_cumulative_scan",
    "make_reduce_epilogue",
    "make_softmax_epilogue",
    "make_welford_update",
]

# 256-element alignment (512 bytes for fp16/bf16) required by T.copy()
# shared memory instructions.  Sub-categories may override this default.
DEFAULT_ALIGNMENT: int = 256

# Maximum column count for a single fragment/shared-memory tile.
# TileLang's vectorizer fails when the *column dimension* of a
# fragment or shared buffer reaches 32768 (a LLVM scalable-vector
# boundary).  Empirical testing on H200 (SM90) confirms that
# 32512 columns compile and execute correctly, while 32768 triggers
# the "scalable vector" error.  We use 32512 (= 32768 - 256) as the
# safe upper bound.
MAX_SINGLE_TILE_COLS: int = 32512

# Default shared memory budget per SM (48 KiB) used to compute the maximum
# block_m that fits within a single thread block's shared memory allocation.
SHARED_MEMORY_BUDGET_BYTES: int = 48 * 1024


def device_smem_budget(device_index: int | None = None) -> int:
    """Return the opt-in shared memory budget for the current device.

    NPU adaptation: queries device properties via the device backend
    abstraction (``tileops.device.get_device_backend``) instead of
    hard-coded ``torch.cuda.*`` calls. On NPU, Ascend device properties
    may not expose ``shared_memory_per_block_optin``; in that case the
    default ``SHARED_MEMORY_BUDGET_BYTES`` (48 KiB) is returned.

    Falls back to ``SHARED_MEMORY_BUDGET_BYTES`` only if the device is
    unavailable. Invalid explicit device indices are not silently masked
    — only the ``None`` (auto-detect) case falls back gracefully.
    """
    explicit = device_index is not None
    try:
        from tileops.device import get_device_backend
        backend = get_device_backend()
    except Exception:
        if explicit:
            raise
        return SHARED_MEMORY_BUDGET_BYTES

    try:
        if not backend.is_available():
            if explicit:
                raise RuntimeError(
                    f"Device is not available but explicit device_index={device_index} was requested"
                )
            return SHARED_MEMORY_BUDGET_BYTES

        if device_index is None:
            device_index = backend.current_device()

        props = backend.get_device_properties(device_index)
        smem_optin = getattr(props, "shared_memory_per_block_optin", 0)
        if smem_optin > 0:
            return smem_optin
        return getattr(props, "shared_memory_per_block", SHARED_MEMORY_BUDGET_BYTES)
    except (RuntimeError, AssertionError):
        if explicit:
            raise
        return SHARED_MEMORY_BUDGET_BYTES


def align_up(n: int, alignment: int) -> int:
    """Round *n* up to the nearest multiple of *alignment*."""
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((n + alignment - 1) // alignment) * alignment


def compute_tile_n(
    block_m: int,
    elem_bytes: int,
    N_padded: int,
    alignment: int = DEFAULT_ALIGNMENT,
    budget: int = SHARED_MEMORY_BUDGET_BYTES,
    num_buffers: int = 1,
) -> int:
    """Compute the tile_n (column chunk) for shared memory, preferring divisibility."""
    per_buffer = block_m * elem_bytes
    if num_buffers * per_buffer * N_padded <= budget:
        return N_padded

    max_cols = budget // (num_buffers * per_buffer)
    tile_n_max = (max_cols // alignment) * alignment
    if tile_n_max == 0:
        raise ValueError(
            f"Cannot fit even {alignment} columns in {budget} bytes "
            f"with block_m={block_m}, elem_bytes={elem_bytes}, "
            f"num_buffers={num_buffers}."
        )

    best_dividing = 0
    for candidate in range(tile_n_max, 0, -alignment):
        if N_padded % candidate == 0:
            best_dividing = candidate
            break

    if best_dividing > 0:
        div_tiles = N_padded // best_dividing
        max_tiles = (N_padded + tile_n_max - 1) // tile_n_max
        if div_tiles <= max_tiles:
            return best_dividing
    return tile_n_max


# Supported op_kind values for each macro factory
_REDUCE_KINDS = {"sum", "max", "min"}
_SOFTMAX_KINDS = {"softmax", "log_softmax"}
_SCAN_KINDS = {"sum", "prod"}


def make_reduce_epilogue(op_kind: str):
    """Create a post-reduce processing T.macro."""
    if op_kind not in _REDUCE_KINDS:
        raise ValueError(
            f"Unsupported op_kind '{op_kind}' for reduce epilogue. "
            f"Expected one of {sorted(_REDUCE_KINDS)}."
        )

    @T.macro
    def epilogue(result, output):
        T.copy(result, output)

    return epilogue


def make_welford_update(block_m: int, N_padded: int):
    """Create a single-pass Welford mean+variance update T.macro."""

    @T.macro
    def welford_update(x, mean, m2, count):
        row_sum = T.alloc_fragment((block_m,), "float32")
        sq_diff = T.alloc_fragment((block_m, N_padded), "float32")
        row_sq_sum = T.alloc_fragment((block_m,), "float32")

        T.reduce_sum(x, row_sum, dim=1)

        batch_mean = T.alloc_fragment((block_m,), "float32")
        new_count = T.alloc_fragment((block_m,), "float32")
        new_mean = T.alloc_fragment((block_m,), "float32")
        for i in T.Parallel(block_m):
            batch_mean[i] = row_sum[i] / float(N_padded)
            new_count[i] = count[i] + float(N_padded)
            new_mean[i] = (mean[i] * count[i] + row_sum[i]) / new_count[i]

        for i, j in T.Parallel(block_m, N_padded):
            sq_diff[i, j] = (x[i, j] - batch_mean[i]) * (x[i, j] - batch_mean[i])
        T.reduce_sum(sq_diff, row_sq_sum, dim=1)

        for i in T.Parallel(block_m):
            delta = batch_mean[i] - mean[i]
            m2[i] = (
                m2[i] + row_sq_sum[i] + delta * delta * (count[i] * float(N_padded) / new_count[i])
            )
            mean[i] = new_mean[i]
            count[i] = new_count[i]

    return welford_update


def make_softmax_epilogue(op_kind: str):
    """Create a softmax family post-processing T.macro."""
    if op_kind not in _SOFTMAX_KINDS:
        raise ValueError(
            f"Unsupported op_kind '{op_kind}' for softmax epilogue. "
            f"Expected one of {sorted(_SOFTMAX_KINDS)}."
        )

    if op_kind == "softmax":

        @T.macro
        def epilogue(row_exp, row_sum, block_rows, block_cols, output):
            for i, j in T.Parallel(block_rows, block_cols):
                output[i, j] = row_exp[i, j] / row_sum[i]

    else:  # log_softmax

        @T.macro
        def epilogue(row_exp, row_sum, block_rows, block_cols, output):
            for i, j in T.Parallel(block_rows, block_cols):
                output[i, j] = T.log(row_exp[i, j] / row_sum[i])

    return epilogue


def make_cumulative_scan(op_kind: str):
    """Create an inclusive prefix scan T.macro."""
    if op_kind not in _SCAN_KINDS:
        raise ValueError(
            f"Unsupported op_kind '{op_kind}' for cumulative scan. "
            f"Expected one of {sorted(_SCAN_KINDS)}."
        )

    if op_kind == "sum":

        @T.macro
        def scan(input_buf, block_rows, block_cols, output_buf):
            for i in T.Parallel(block_rows):
                output_buf[i, 0] = input_buf[i, 0]
            for j in T.Serial(1, block_cols):
                for i in T.Parallel(block_rows):
                    output_buf[i, j] = output_buf[i, j - 1] + input_buf[i, j]

    else:  # prod

        @T.macro
        def scan(input_buf, block_rows, block_cols, output_buf):
            for i in T.Parallel(block_rows):
                output_buf[i, 0] = input_buf[i, 0]
            for j in T.Serial(1, block_cols):
                for i in T.Parallel(block_rows):
                    output_buf[i, j] = output_buf[i, j - 1] * input_buf[i, j]

    return scan
