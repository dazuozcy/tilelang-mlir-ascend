"""Stage 5 integration package for topk_selector (NPU, target="npuir").

Aggregates the TileLang kernel functions integrated from the conductor
deliverable ``examples/topk_selector/_topk_selector_kernel/`` (kernel +
``_topk_selector_kernel_DESIGN.md`` snapshot of the Stage 1 design doc).

Kernel interface (DESIGN.md §3.5.2 interface contract, two-stage call):

    _topk_selector_kernel(batch, seq_len, seq_len_kv, kv_group, topk,
                          in_dtype, out_dtype)(chunk_size=0, num_kernels=40
                          )(index_score, starts, ends)

Legacy GPU config keys (RADIX / BLOCK_SIZE / SMEM_INPUT_SIZE / block_m) are
accepted positionally for call-site compatibility and ignored; the NPU knobs
are ``chunk_size`` (0 = auto-derive per DESIGN.md §5.2) and ``num_kernels``.
``out_dtype`` must be "int32".
"""

from ._topk_selector_kernel import (
    _topk_selector_kernel,
    golden_topk_selector,
)

__all__ = ["_topk_selector_kernel", "golden_topk_selector"]
