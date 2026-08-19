"""Workload definitions for the elementwise op family."""

from math import prod

from tileops.workloads.workload_base import RandnWorkload


class ShapedRandnWorkload(RandnWorkload):
    """One ``randn`` tensor of arbitrary rank, with its element count.

    Adaptation point: ``device`` is resolved from :func:`get_device_backend`
    via ``RandnWorkload.gen_inputs`` instead of hard-coded ``"cuda"``.
    """

    def __init__(self, shape: tuple, dtype):
        super().__init__(tuple(shape), dtype)
        self.n_total = prod(self.shape)


class MishWorkload(ShapedRandnWorkload):
    """Workload definition for MishFwdOp (shape + dtype, randn input)."""
