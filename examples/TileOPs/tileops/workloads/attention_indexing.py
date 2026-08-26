"""Workload definitions for the attention_indexing op family."""

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase

__all__ = ["TopkSelectorWorkload"]


class TopkSelectorWorkload(WorkloadBase):
    """Workload for TopkSelectorOp.

    Ported from the GPU ``workloads/topk_selector.py``; the generation logic
    (randn scores + all-rows start/end range) is preserved.  W1: the input
    device is resolved via :func:`get_device_backend` instead of the
    hard-coded ``"cuda"``.
    """

    def __init__(
        self,
        batch: int,
        seq_len: int,
        seq_len_kv: int,
        kv_group: int,
        topk: int,
        in_dtype: torch.dtype,
        out_dtype: torch.dtype,
    ):
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.topk = topk
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backend = get_device_backend()
        device = backend.name
        index_score = torch.randn(
            self.batch,
            self.seq_len,
            self.seq_len_kv,
            self.kv_group,
            dtype=self.in_dtype,
            device=device,
        )
        starts = torch.zeros(self.batch, self.seq_len, dtype=self.out_dtype, device=device)
        ends = (
            torch.ones(self.batch, self.seq_len, dtype=self.out_dtype, device=device)
            * self.seq_len_kv
        )
        return index_score, starts, ends
