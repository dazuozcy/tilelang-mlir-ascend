"""Top-k selector kernel using TileLang (NPU-adapted).

Selects the ``topk`` largest-score indices along the ``seq_len_kv`` axis of
an ``[B, S, S_kv, G]`` score tensor.  The NPU migration (DESIGN.md §0.6)
replaces the GPU two-stage SMEM-atomic radix-select with a chunked
``T.vsort`` (descending, tail axis) + streaming merge top-k over a
one-dimensional persistent kernel (``num_kernels`` vector cores); window
masking uses -inf fill and chunk-level skip guards.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (integrated, Stage 5):
    The NPU TileLang kernel function (``_topk_selector_kernel``) is
    integrated from the conductor deliverable into the sibling package
    ``.topk_selector_kernel`` and imported from there.  The migration
    replaces the GPU SMEM-atomic radix-select pipeline with a chunked
    ``T.vsort`` + streaming-merge top-k over a one-dimensional persistent
    kernel (``target="npuir"``); see
    ``topk_selector_kernel/_topk_selector_kernel_DESIGN.md``.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[90]``).
    K7: ``custom_op("npub::...")`` (was ``"top::..."``).
    K8: ``autotune_configs`` / ``autotune_supply_prog`` /
        ``_make_supply_prog`` / ``tune`` param -- all removed;
        ``init_config(config)`` takes no ``tune`` argument.
    K9: the GPU config keys ``RADIX`` / ``BLOCK_SIZE`` /
        ``SMEM_INPUT_SIZE`` / ``block_m`` (radix-select specific) are
        mapped to the NPU config semantics ``chunk_size`` (0 = auto-derive)
        and ``num_kernels`` (persistent vector cores, default 40), per
        DESIGN.md §3.5.2 interface contract.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

from .topk_selector_kernel import _topk_selector_kernel

__all__ = ["TopkSelectorKernel"]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: top:: -> npub::, K9: NPU config semantics)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::topk_selector_wrapped_kernel", mutates_args=())
def _topk_selector_wrapped_kernel(
    batch: int,
    seq_len: int,
    seq_len_kv: int,
    kv_group: int,
    topk: int,
    in_dtype: str,
    out_dtype: str,
    chunk_size: int,
    num_kernels: int,
    index_score: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    # Two-stage call per DESIGN.md §3.5.2: factory -> config -> tensors.
    # The kernel accepts legacy GPU config keys positionally (ignored);
    # the NPU knobs chunk_size / num_kernels are passed by keyword.
    return _topk_selector_kernel(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype)(
        chunk_size=chunk_size, num_kernels=num_kernels
    )(index_score, starts, ends)


@_topk_selector_wrapped_kernel.register_fake
def _(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype, *inputs) -> None:
    return torch.empty([batch, seq_len, kv_group, topk], device=inputs[0].device, dtype=torch.int32)


# ---------------------------------------------------------------------------
# Kernel class (K5, K7, K8 adaptations)
# ---------------------------------------------------------------------------


class TopkSelectorKernel(Kernel):
    """Per-row top-k index selection over an ``[B, S, S_kv, G]`` score tensor.

    Args:
        batch: Batch size.
        seq_len: Query sequence length.
        seq_len_kv: Key/value sequence length.
        kv_group: Number of key/value groups.
        topk: Number of indices selected per row.
        dtype: Torch dtype of the input scores.
        out_dtype: Torch dtype of the emitted indices (must be int32).
        config: Optional dict with "chunk_size" (0 = auto-derive per
            DESIGN.md §5.2 UB budget) and "num_kernels" (persistent
            vector cores, default 40).

    NPU adaptations:
        - K5: ``supported_archs = None`` (was ``[90]``, CUDA SM90 only).
        - K8: autotune removed -- heuristic config selection only;
          ``init_config(config)`` takes no ``tune`` argument.
        - K9: GPU radix-select keys (RADIX / BLOCK_SIZE / SMEM_INPUT_SIZE /
          block_m) mapped to the NPU config semantics chunk_size /
          num_kernels (DESIGN.md §3.5.2).
    """

    # K5: [90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # NPU default: auto chunk derivation + 40 persistent vector cores
    # (matches the integrated kernel's NUM_KERNELS_DEFAULT).
    NUM_KERNELS_DEFAULT = 48

    def __init__(
        self,
        batch: int,
        seq_len: int,
        seq_len_kv: int,
        kv_group: int,
        topk: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.topk = topk
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.out_dtype_str = self.dtype_to_str(self.out_dtype)

        # Build the factory callable (lru_cached by shape; compilation is
        # deferred to forward() via the custom_op wrapper).
        self.kernel = _topk_selector_kernel(
            self.batch,
            self.seq_len,
            self.seq_len_kv,
            self.kv_group,
            self.topk,
            self.dtype_str,
            self.out_dtype_str,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {
            "chunk_size": 0,  # 0 = auto-derive per DESIGN.md §5.2
            "num_kernels": self.NUM_KERNELS_DEFAULT,
        }

    def forward(
        self, index_score: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
    ) -> torch.Tensor:
        return _topk_selector_wrapped_kernel(
            self.batch,
            self.seq_len,
            self.seq_len_kv,
            self.kv_group,
            self.topk,
            self.dtype_str,
            self.out_dtype_str,
            self.config["chunk_size"],
            self.config["num_kernels"],
            index_score,
            starts,
            ends,
        )
