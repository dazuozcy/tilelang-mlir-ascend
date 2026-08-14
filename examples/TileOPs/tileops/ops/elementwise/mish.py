"""Mish operator (spec-conformant interface).

Adaptation from GPU (TileOPs) to NPU:

- ``_validate``: ``x.is_cuda`` → ``backend.is_device_tensor(x)`` (O1).
- Kernel dispatch: ``MishFwdKernel`` is the NPU TileLang kernel (factory
  stubbed until the NPU kernel component fills it in).
- ``tune`` parameter removed (O3); kernel constructor takes
  ``(N_total, dtype, config=None, device_index=None)``.
- ``eval_roofline``: identical arithmetic (roofline is device-agnostic).
- ``_OP_REGISTRY`` / ``_wrapped`` custom_op at the Op layer removed (O5) —
  ``forward`` calls ``_eager_forward`` directly.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.elementwise.mish import MishFwdKernel
from tileops.kernels.kernel_base import Kernel
from tileops.ops.op_base import Op

__all__ = ["MishFwdOp"]


class MishFwdOp(Op):
    """Element-wise Mish: ``y = x * tanh(softplus(x))``.

    The input is flattened to a 1-D ``(N_total,)`` vector, the kernel is
    dispatched on the flat vector, and the output is reshaped back to the
    original input shape.

    Args:
        N_total: Total number of elements (flattened).
        dtype: Data type (float16, bfloat16, or float32).
        inplace: When True, copy the result back into ``input`` and return
            ``input`` (preserving tensor identity).  The kernel still
            computes into a fresh buffer; only the user-visible tensor is
            mutated, mirroring ``torch.nn.functional.mish``.
        kernel_map: Optional kernel dispatch override.
    """

    _op_name = "mish"
    _kernel_key = "mish"
    _kernel_class = MishFwdKernel
    # Manifest: flops = "4 * N". Per roofline.md §1.3:
    # mish(x) = x * tanh(softplus(x));
    # softplus = exp + log1p = 2; tanh(transcendental) + final mul = 4 per elem.
    FLOPS_PER_ELEM = 4

    def __init__(
        self,
        N_total: int,
        dtype: torch.dtype,
        inplace: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.N_total = N_total
        self.dtype = dtype
        self.inplace = inplace
        self.output_dtype = dtype  # same_as(input)
        self.dispatch_kernel(kernel_map)
        # Kernel construction will raise NotImplementedError until the NPU
        # kernel component fills in the factory stub.
        self.kernel = self.kernel_map[self._kernel_key](N_total, dtype)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._kernel_key: self._kernel_class}

    @property
    def total_memory(self) -> float:
        """Read x + write y."""
        return self.N_total * (self.dtype.itemsize + self.output_dtype.itemsize)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this unary elementwise op instance.

        Mirrors the manifest roofline:
        ``flops = FLOPS_PER_ELEM * N`` and
        ``bytes = N * input_elem_bytes + N * output_elem_bytes``.
        For Mish, output dtype == input dtype, so bytes collapse to
        ``2 * N * elem_bytes``.
        """
        return self.FLOPS_PER_ELEM * self.N_total, int(self.total_memory)

    def _validate(self, x: torch.Tensor) -> None:
        """Validate input tensor against the op's dtype / numel contract.

        NPU adaptation: device check uses ``backend.is_device_tensor(x)``
        instead of ``x.is_cuda`` (O1).
        """
        backend = get_device_backend()
        if not backend.is_device_tensor(x):
            raise ValueError(f"input must be a {backend.name} tensor, got device {x.device}")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {x.dtype}")
        if x.numel() != self.N_total:
            raise ValueError(f"Expected {self.N_total} elements, got {x.numel()}")

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        """Direct kernel call for use inside forward."""
        orig_shape = input.shape
        flat = input.contiguous().reshape(-1)
        return self.kernel(flat).reshape(orig_shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._validate(input)
        if self.inplace:
            result = self._eager_forward(input)
            input.copy_(result.reshape(input.shape))
            return input
        return self._eager_forward(input)
