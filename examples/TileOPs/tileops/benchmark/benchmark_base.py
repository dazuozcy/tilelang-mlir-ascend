"""Benchmark base class — device-agnostic timing and reporting.

Adaptation from GPU (TileOPs) to NPU:

**GPU original** (``benchmarks/benchmark_base.py``):
  - ``bench_kernel`` uses CUPTI (``torch.profiler`` with Kineto) for
    pure-kernel timing — no launch overhead.
  - L2 cache flush before every iteration (sized to actual L2).
  - Input tensor cloning per iteration (address diversity).
  - CUPTI projection validation with CUDA-events fallback.
  - ``_native_output_suppressor`` uses tilelang's ``suppress_stdout_stderr``.
  - ``_get_env_metadata`` queries ``nvidia-smi``.

**NPU version** (this file):
  - ``bench_kernel`` uses device Event timing (``torch.npu.Event`` /
    ``torch.cuda.Event``) — includes launch overhead (~50-60us) but is
    the standard NPU timing method.
  - Cache flush is a no-op on NPU (``backend.cache_flush()``); CUDA
    retains the L2 flush.
  - Input cloning preserved (address diversity is still relevant).
  - No CUPTI projection logic — event timing is the primary path.
  - No tilelang dependency for output suppression.
  - ``_get_env_metadata`` queries the device backend (``npu-smi`` for NPU,
    ``nvidia-smi`` for CUDA).
  - ``workloads_to_params`` and ``ManifestBenchmark`` are preserved
    verbatim (they read from the standalone manifest).
"""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Generic, Optional, TypeVar

import pytest
import torch

from tileops.device import get_device_backend
from tileops.manifest import (
    WORKLOAD_RESERVED_KEYS,
    load_manifest,
    load_workloads,
    single_input_workload_contract,
)

try:
    from tileops.benchmark.msprof import bench_kernel_msprof
except ImportError:
    bench_kernel_msprof = None  # type: ignore[assignment]

W = TypeVar("W")

_logger = logging.getLogger("tileops.benchmark")

_bench_results = threading.local()


def _workload_contract(op_name: str) -> tuple[str, frozenset[str]]:
    sig = load_manifest()[op_name].get("signature") or {}
    contract = single_input_workload_contract(sig)
    if contract is None:
        raise KeyError(
            f"workloads_to_params({op_name!r}) needs exactly one manifest "
            "tensor input; multi-input ops use their own bench files."
        )
    return contract


def bench_kernel(
    fn: Callable,
    args: tuple[Any, ...] = (),
    n_warmup: int = 10,
    n_repeat: int = 50,
    n_trials: int = 3,
) -> float:
    """Benchmark a kernel with device-event timing.

    Protocol:
      1. Run *n_warmup* un-timed iterations with cache flush.
      2. For each of *n_trials* trials, time *n_repeat* iterations using
         device Events.  Cache is flushed before every iteration.
         Input tensors are cloned each iteration so the kernel sees fresh
         addresses.
      3. Report the median trial mean.

    NPU adaptation: uses ``backend.Event(enable_timing=True)`` instead of
    CUPTI/Kineto.  Latency includes ~50-60us launch overhead per call —
    acceptable for NPU benchmarks where kernel time typically dominates.

    Returns:
        Kernel latency in **milliseconds**.
    """
    if not isinstance(args, tuple):
        raise TypeError(f"bench_kernel expects a tuple of args, got {type(args).__name__}.")

    backend = get_device_backend()
    has_args = len(args) > 0

    _N_CLONES = 3
    _MAX_CLONE_BYTES = 1 << 30

    if has_args:
        tensor_mask = tuple(isinstance(a, torch.Tensor) for a in args)
        total_bytes = sum(
            a.nelement() * a.element_size() for a, m in zip(args, tensor_mask, strict=True) if m
        )
        if total_bytes * _N_CLONES <= _MAX_CLONE_BYTES:
            arg_pool = [
                tuple(a.clone() if m else a for a, m in zip(args, tensor_mask, strict=True))
                for _ in range(_N_CLONES)
            ]

            def _run(i):
                return fn(*arg_pool[i % _N_CLONES])
        else:
            _logger.warning(
                "bench_kernel: inputs total %.2f GiB; skipping per-iteration cloning.",
                total_bytes / (1 << 30),
            )
            arg_pool = None

            def _run(i):
                return fn(*args)
    else:
        arg_pool = None

        def _run(i):
            return fn()

    for i in range(n_warmup):
        backend.cache_flush()
        _run(i)
    backend.synchronize()

    trial_means: list[float] = []
    for _ in range(n_trials):
        start_events = [backend.Event(enable_timing=True) for _ in range(n_repeat)]
        end_events = [backend.Event(enable_timing=True) for _ in range(n_repeat)]

        for i in range(n_repeat):
            backend.cache_flush()
            backend.synchronize()
            start_events[i].record()
            _run(i)
            end_events[i].record()
        backend.synchronize()

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events, strict=True)]
        trial_means.append(sum(times) / len(times))

    if arg_pool is not None:
        del arg_pool
    backend.empty_cache()

    trial_means.sort()
    return trial_means[len(trial_means) // 2]


def _bench_with_mode(
    functor: Callable,
    args: tuple = (),
    mode: Optional[str] = None,
) -> tuple[float, str]:
    """Dispatch to events or msprof profiling based on *mode* / env var.

    Returns ``(latency_ms, prof_mode)`` where *prof_mode* is either
    ``"events"`` or ``"msprof"``.

    When ``TILEOPS_PROF_MODE=msprof`` (or *mode*="msprof"), attempts
    msprof profiling.  If the functor is not supported by msprof (e.g.
    closures/lambdas) or msprof is unavailable, falls back to events
    with a warning.
    """
    mode = mode or os.environ.get("TILEOPS_PROF_MODE", "events")
    mode = mode.lower().strip()

    if mode == "msprof":
        if bench_kernel_msprof is None:
            _logger.warning(
                "TILEOPS_PROF_MODE=msprof but msprof module is unavailable; falling back to events."
            )
            return bench_kernel(functor, args=args), "events"
        try:
            # Call functor once in-process so that Op state (e.g. roofline
            # spec, kernel cache) is initialized before msprof runs it in
            # a subprocess.  This mirrors the events-mode warm-up path.
            _ = functor(*args)
            backend = get_device_backend()
            backend.synchronize()

            latency = bench_kernel_msprof(functor, args=args)
            return latency, "msprof"
        except (TypeError, RuntimeError, FileNotFoundError) as e:
            _logger.warning(
                "msprof profiling failed (%s: %s); falling back to events.",
                type(e).__name__,
                e,
            )
            return bench_kernel(functor, args=args), "events"

    return bench_kernel(functor, args=args), "events"


class BenchmarkBase(Generic[W], ABC):
    """Abstract base class for op benchmarking."""

    def __init__(self, workload: W):
        self.workload = workload

    @abstractmethod
    def calculate_flops(self) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def calculate_memory(self) -> Optional[float]:
        raise NotImplementedError

    def profile(self, functor: Any, *inputs: Any, mode: Optional[str] = None) -> dict:
        with torch.no_grad():
            latency, prof_mode = _bench_with_mode(functor, args=inputs, mode=mode)
        result = self._build_result(latency)
        result["prof_mode"] = prof_mode
        return result

    def _build_result(self, latency: float) -> dict:
        latency_us = latency * 1000.0
        result: dict[str, Any] = {"latency_us": latency_us}
        flops = self.calculate_flops()
        if flops is not None:
            result["tflops"] = flops / latency * 1e-9
        memory = self.calculate_memory()
        if memory is not None:
            result["bandwidth_tbs"] = memory / latency * 1e-9
        return result


def _workload_extra_params(w: dict, shape_key: str) -> dict[str, Any]:
    reserved = WORKLOAD_RESERVED_KEYS | {shape_key}
    return {
        k: v
        for k, v in w.items()
        if isinstance(k, str) and k not in reserved and not k.startswith("__")
    }


def workloads_to_params(op_name: str, include_extra: bool = False) -> list:
    """Convert manifest workload dicts for *op_name* to pytest params."""
    workloads = load_workloads(op_name)
    shape_key, allowed = _workload_contract(op_name)
    params = []
    for w in workloads:
        if shape_key not in w:
            raise KeyError(
                f"workload {w.get('label', w)!r} of {op_name!r} is missing {shape_key!r}."
            )
        unknown = sorted(
            repr(k)
            for k in w
            if not isinstance(k, str) or (k not in allowed and not k.startswith("__"))
        )
        if unknown:
            raise KeyError(
                f"workload {w.get('label', w)!r} of {op_name!r} has unknown keys {unknown}."
            )
        shape = tuple(w[shape_key])
        label = w.get("label", "x".join(str(s) for s in shape))
        extra = _workload_extra_params(w, shape_key) if include_extra else {}
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            param_args = (shape, dtype, dict(extra)) if include_extra else (shape, dtype)
            params.append(pytest.param(*param_args, id=f"{label}-{dtype_str}"))
    return params


class ManifestBenchmark(BenchmarkBase[Any]):
    """Generic benchmark that reads FLOP/memory from an Op instance.

    Usage::

        op = LogSumExpFwdOp(dtype=dtype, dim=0)
        bm = ManifestBenchmark("LogSumExpFwdOp", op, workload)
        result = bm.profile(op, *inputs)
    """

    def __init__(self, op_name: str, op: Any, workload: Any):
        super().__init__(workload)
        self._op_name = op_name
        self._op = op
        self._roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            flops, mem_bytes = self._op.eval_roofline()
            self._roofline_cache = (float(flops), float(mem_bytes))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


def _extract_op_config(op: object) -> Optional[dict]:
    op_config = getattr(op, "config", None)
    if op_config:
        return op_config
    kernel = getattr(op, "kernel", None)
    op_config = getattr(kernel, "config", None) if kernel is not None else None
    if op_config:
        return op_config
    cache = getattr(op, "_kernel_cache", None)
    if cache:
        try:
            first_kernel = next(iter(cache.values()))
        except StopIteration:
            first_kernel = None
        if first_kernel is not None:
            op_config = getattr(first_kernel, "config", None)
            if op_config:
                return op_config
    return None


class BenchmarkReport:
    """Collects benchmark results and dumps a markdown report."""

    _records: dict = {}
    _prof_mode: str = "events"

    @staticmethod
    def set_prof_mode(mode: str) -> None:
        BenchmarkReport._prof_mode = mode

    @staticmethod
    def record(op_or_name, params: dict, result: dict, tag: str = "tileops") -> None:
        if isinstance(op_or_name, str):
            name = op_or_name
            op_module = None
            op_config = None
        else:
            name = op_or_name.__class__.__name__
            op_module = op_or_name.__class__.__module__
            op_config = _extract_op_config(op_or_name)

        def _is_serializable(v: Any) -> bool:
            if isinstance(v, (int, float, bool, str, torch.dtype)):
                return True
            if isinstance(v, tuple):
                return all(_is_serializable(x) for x in v)
            return False

        filtered_params = {
            k: v
            for k, v in params.items()
            if k not in ("test", "bm", "op", "inputs", "result", "result_bl", "baseline_fn")
            and not k.startswith("_")
            and _is_serializable(v)
        }
        record_entry = {
            "params": filtered_params,
            "result": result,
            "tag": tag,
        }
        if op_config:
            record_entry["config"] = op_config
        BenchmarkReport._records.setdefault(name, []).append(record_entry)

        if not hasattr(_bench_results, "entries"):
            _bench_results.entries = []
        entry = {"tag": tag, "op": name, **result}
        if op_module:
            entry["op_module"] = op_module
        _bench_results.entries.append(entry)

        _logger.info(
            "op=%s module=%s tag=%s latency_us=%.4f tflops=%.2f",
            name,
            op_module or "N/A",
            tag,
            result.get("latency_us", 0),
            result.get("tflops", 0),
        )

    @staticmethod
    def dump(path: str) -> None:
        if not BenchmarkReport._records:
            return

        backend = get_device_backend()
        lines = [
            "# NPU Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Environment",
            "",
        ]
        lines.extend(backend.env_metadata())
        lines.append(f"- **Profiling mode**: {BenchmarkReport._prof_mode}")
        lines.append("")

        default_result_keys = ["latency_us", "tflops", "bandwidth_tbs"]

        for name, entries in BenchmarkReport._records.items():
            if not entries:
                continue

            lines.append(f"## {name}")
            lines.append("")

            tag_entries: dict[str, list] = {}
            for entry in entries:
                tag_entries.setdefault(entry["tag"], []).append(entry)
            result_keys = list(default_result_keys)
            for entry in entries:
                for key in entry["result"]:
                    if key not in result_keys:
                        result_keys.append(key)

            for tag, tag_group in tag_entries.items():
                lines.append(f"### {tag}")
                lines.append("")

                param_keys = list(tag_group[0]["params"].keys())
                trailing_keys = []
                for k in ("dtype", "shape"):
                    if k in param_keys:
                        param_keys.remove(k)
                        trailing_keys.append(k)
                has_config = any("config" in e for e in tag_group)
                header_parts = param_keys + result_keys
                if has_config:
                    header_parts.append("config")
                header_parts.extend(trailing_keys)
                lines.append("| " + " | ".join(header_parts) + " |")
                lines.append("| " + " | ".join(["---"] * len(header_parts)) + " |")

                for entry in tag_group:
                    row = [str(entry["params"].get(k, "")) for k in param_keys]
                    for rk in result_keys:
                        val = entry["result"].get(rk)
                        if val is None:
                            row.append("N/A")
                        elif isinstance(val, (int, float)) and not isinstance(val, bool):
                            row.append(f"{val:.4f}")
                        else:
                            row.append(str(val))
                    if has_config:
                        cfg = entry.get("config")
                        row.append(str(cfg) if cfg else "")
                    for tk in trailing_keys:
                        row.append(str(entry["params"].get(tk, "")))
                    lines.append("| " + " | ".join(row) + " |")

                lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        print(f"\nBenchmark report saved to {path}")

    @staticmethod
    def clear() -> None:
        BenchmarkReport._records.clear()
        BenchmarkReport._prof_mode = "events"
