"""Top-k selector operator (spec-conformant interface).

Selects the ``topk`` largest-score indices along ``seq_len_kv`` (dim=2) of
``index_score [B, S, S_kv, G]`` within each row's ``[starts, ends)`` range,
emitting ``[B, S, G, topk]`` int32 indices (reference: ``torch.topk``).

Adapted from the GPU ``tileops/ops/topk_selector.py``:

- O1/O2: ``x.is_cuda`` device checks -> ``backend.is_device_tensor(x)`` /
  backend name via :func:`get_device_backend`.
- O3: ``tune`` parameter removed; the kernel constructor takes no ``tune``.
- O4: no ``get_sm_version``; arch checks go through the device backend.
- O5: no ``compile_boundary`` registration.
- O6: op-specific flow (validation, dispatch, output shape) preserved.
- ``eval_roofline`` is implemented directly (the NPU project removed the
  GPU manifest codegen): the formula is ported verbatim from the GPU
  manifest roofline func ``tileops.perf.formulas.topk_selector_roofline``
  (roofline is device-agnostic).
"""

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.attention_indexing.topk_selector import TopkSelectorKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["TopkSelectorOp"]


class TopkSelectorOp(Op):
    def __init__(self, topk: int, kernel_map: Optional[Dict[str, Kernel]] = None) -> None:
        self.batch = None
        self.seq_len = None
        self.seq_len_kv = None
        self.kv_group = None
        self.topk = topk
        self.in_dtype = None
        self.out_dtype = torch.int32

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"topk_selector_kernel": TopkSelectorKernel}

    def _get_kernel(
        self,
        batch: int,
        seq_len: int,
        seq_len_kv: int,
        kv_group: int,
        in_dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, seq_len, seq_len_kv, kv_group, self.topk, in_dtype, device_index)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["topk_selector_kernel"](
                batch, seq_len, seq_len_kv, kv_group, self.topk, in_dtype, self.out_dtype
            )
        return self._kernel_cache[key]

    def forward(self, index_score, starts, ends) -> torch.Tensor:
        backend = get_device_backend()
        if not backend.is_device_tensor(index_score):
            raise ValueError(f"TopkSelectorOp expects {backend.name} inputs")
        if index_score.ndim != 4:
            raise ValueError("TopkSelectorOp expects index_score shape [B, S, S_kv, G]")
        if starts.ndim != 2 or ends.ndim != 2:
            raise ValueError("TopkSelectorOp expects starts/ends shape [B, S]")
        if not backend.is_device_tensor(starts) or not backend.is_device_tensor(ends):
            raise ValueError(f"starts and ends must be {backend.name} tensors")
        if starts.dtype != torch.int32 or ends.dtype != torch.int32:
            raise ValueError("TopkSelectorOp expects int32 starts/ends tensors")

        batch, seq_len, seq_len_kv, kv_group = index_score.shape
        if starts.shape != (batch, seq_len) or ends.shape != (batch, seq_len):
            raise ValueError("TopkSelectorOp starts/ends must match index_score batch/seq_len")
        if not 0 < self.topk <= seq_len_kv:
            raise ValueError(f"topk must satisfy 0 < topk <= seq_len_kv={seq_len_kv}")

        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.in_dtype = index_score.dtype
        self.kernel = self._get_kernel(
            batch, seq_len, seq_len_kv, kv_group, index_score.dtype, index_score.device.index
        )

        return self.kernel(index_score, starts, ends)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Ported verbatim from the GPU
        ``tileops.perf.formulas.topk_selector_roofline``: one comparison per
        (batch, seq_len, kv_group, seq_len_kv) score element; bytes = score
        read + starts/ends read + index write.

        Requires a prior ``forward()`` call to bind the dynamic shapes.
        """
        if self.batch is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        batch = int(self.batch)
        seq_len = int(self.seq_len)
        seq_len_kv = int(self.seq_len_kv)
        kv_group = int(self.kv_group)
        topk = int(self.topk)
        in_elem = self.in_dtype.itemsize
        out_elem = self.out_dtype.itemsize
        comparisons = batch * seq_len * kv_group * seq_len_kv
        nbytes = comparisons * in_elem + batch * seq_len * 2 * out_elem
        nbytes += batch * seq_len * kv_group * topk * out_elem
        return int(comparisons), int(nbytes)
