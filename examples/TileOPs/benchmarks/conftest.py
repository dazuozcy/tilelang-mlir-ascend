import gc
import os

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, _bench_results
from tileops.device import get_device_backend


def _release_device_cache() -> None:
    gc.collect()
    backend = get_device_backend()
    if backend.is_available():
        backend.empty_cache()


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(1235)
    backend = get_device_backend()
    backend.manual_seed_all(1235)


def pytest_sessionstart(session):
    BenchmarkReport.clear()


def pytest_sessionfinish(session, exitstatus):
    prof_mode = os.environ.get("TILEOPS_PROF_MODE", "events")
    BenchmarkReport.set_prof_mode(prof_mode)
    BenchmarkReport.dump("profile_run.log")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _bench_results.entries = []
    try:
        yield
        entries = getattr(_bench_results, "entries", [])
        if not entries:
            return

        tileops_entry = None
        baseline_entries = []
        for e in entries:
            if e["tag"].startswith("tileops"):
                if tileops_entry is None:
                    tileops_entry = e
            else:
                baseline_entries.append(e)

        if tileops_entry:
            item.user_properties.append(("op", tileops_entry["op"]))
            if "op_module" in tileops_entry:
                item.user_properties.append(("op_module", tileops_entry["op_module"]))
            item.user_properties.append(
                ("tileops_latency_us", f"{tileops_entry.get('latency_us', 0):.4f}")
            )
            tflops = tileops_entry.get("tflops")
            if tflops is not None:
                item.user_properties.append(("tileops_tflops", f"{tflops:.2f}"))
            bw = tileops_entry.get("bandwidth_tbs")
            if bw is not None:
                item.user_properties.append(("tileops_bandwidth_tbs", f"{bw:.2f}"))

        for _idx, be in enumerate(baseline_entries):
            tag = be["tag"]
            bl_latency = be.get("latency_us", 0)
            bl_tflops = be.get("tflops")

            item.user_properties.append((f"{tag}_latency_us", f"{bl_latency:.4f}"))
            if bl_tflops is not None:
                item.user_properties.append((f"{tag}_tflops", f"{bl_tflops:.2f}"))
            if tileops_entry:
                tl = tileops_entry.get("latency_us", 0)
                if tl > 0 and bl_latency > 0:
                    item.user_properties.append((f"{tag}_ratio", f"{bl_latency / tl:.4f}"))
    finally:
        _bench_results.entries = []
        _release_device_cache()
