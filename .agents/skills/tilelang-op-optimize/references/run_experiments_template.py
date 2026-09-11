"""run_experiments.py -- Stage 4 experiment batch runner template (T-4).

Copied into perf_opt/ by the optimizer on its FIRST Phase 2 round and adapted
(this file is a TEMPLATE -- the adaptation points are the CONFIG constants
and the `build_bench_cmd` function below). After the first round, each round
just updates EXPERIMENTS and re-runs this script; the agent only reads the
summary table it prints. This turns "agent round-trips drive experiments"
into "script executes, agent reads the table" (>=60% fewer agent interaction
rounds; also immune to the large-artifact session-overflow failure class,
see queue VP-2026-0021).

Per experiment branch this runner, in order:
  1. runs the L0 precision regression (branch file `--level L0`);
  2. wraps the msprof op measurement (kernel-only Task Duration, the single
     latency metric);
  3. appends ONE line to perf_records.jsonl (append-only, contract:
     _shared/standards/signal-registry.md #5);
  4. prints a candidate-vs-current-best summary table at the end.

Adaptation points (marked "ADAPT"):
  - CONFIG: op name, kernel name, bench entry, dispatch/workload labels;
  - build_bench_cmd(): how a branch file is launched for msprof profiling;
  - EXPERIMENTS: the round's branch list (file + opt_id + parent_id).

Usage (inside perf_opt/):
  python run_experiments.py                     # run all pending experiments
  python run_experiments.py --round 3           # label records with round 3
  python run_experiments.py --only v3_op1       # run a single branch
"""

import argparse
import csv
import datetime
import glob
import json
import os
import statistics
import subprocess
import sys

# --------------------------------------------------------------- CONFIG (ADAPT)
OP = "my_op"  # ADAPT: operator name (file stem)
KERNEL_NAME = "main"  # ADAPT: target kernel name for msprof
BENCH_CMD = "python bench.py --impl {impl} --dtype float16 --N 16777216 --check"  # ADAPT: bench entry; {impl} is replaced with the branch file path
DISPATCH_PATH = "default"  # ADAPT: dispatch path label
WORKLOAD = "N16M_fp16"  # ADAPT: representative workload label
LAUNCH_COUNT = 15
WARM_UP = 5
TIMEOUT_S = 900
CURRENT_BEST = f"{OP}.py"  # ADAPT: round base (current best file)
# --------------------------------------------------------------- /CONFIG

PERF_RECORDS = "perf_records.jsonl"
MSPROF_METRICS = (
    "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,"
    "MemoryL0,L2Cache,ResourceConflictRatio"
)

# ----------------------------------------------------------- EXPERIMENTS (ADAPT)
# Each entry: {file, opt_id, parent_id}
EXPERIMENTS = [
    # {"file": f"{OP}_opt_v2_op1.py", "opt_id": "v2_op1",
    #  "parent_id": "baseline"},
]
# ------------------------------------------------------------------ /EXPERIMENTS


def build_bench_cmd(impl_path: str) -> str:  # ADAPT if bench needs more args
    return BENCH_CMD.format(impl=impl_path)


def run_l0(branch_file: str) -> bool:
    proc = subprocess.run(
        [sys.executable, branch_file, "--level", "L0"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    return proc.returncode == 0


def run_msprof(branch_file: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cmd = (
        f"msprof op --kernel-name={KERNEL_NAME} --output={out_dir} "
        f"--launch-count={LAUNCH_COUNT} --warm-up={WARM_UP} --dump=off "
        f"--aic-metrics={MSPROF_METRICS} {build_bench_cmd(branch_file)}"
    )
    log = os.path.join("logs", os.path.basename(out_dir) + ".log")
    os.makedirs("logs", exist_ok=True)
    with open(log, "w") as f:
        f.write(f"# {cmd}\n")
        f.flush()
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=TIMEOUT_S
        )
        f.write(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        return {"error": f"msprof rc={proc.returncode}", "log": log}
    durs = []
    for path in sorted(
        glob.glob(os.path.join(out_dir, "OPPROF_*", "*", "*", "OpBasicInfo_*.csv"))
    ):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("Task Duration(us)"):
                    durs.append(float(row["Task Duration(us)"]))
    if not durs:
        return {"error": "no OpBasicInfo rows", "log": log}
    return {
        "duration_us": round(statistics.median(durs), 3),
        "n": len(durs),
        "msprof_raw_path": out_dir,
    }


def append_record(record: dict) -> None:
    with open(PERF_RECORDS, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--only", help="run a single opt_id")
    args = ap.parse_args()

    todo = [e for e in EXPERIMENTS if not args.only or e["opt_id"] == args.only]
    if not todo:
        print("no experiments configured (edit EXPERIMENTS)")
        return 2

    results = []
    for exp in todo:
        branch = exp["file"]
        print(f"=== {exp['opt_id']}: {branch} ===")
        if not os.path.isfile(branch):
            results.append({**exp, "status": "missing_file"})
            continue
        l0 = run_l0(branch)
        if not l0:
            results.append({**exp, "status": "l0_fail", "l0_pass": False})
            continue
        prof = run_msprof(
            branch, os.path.join("profiles", f"round{args.round}", exp["opt_id"])
        )
        if "error" in prof:
            results.append({**exp, "status": "msprof_fail", "l0_pass": True})
            print(f"  msprof FAILED: {prof['error']}")
            continue
        append_record(
            {
                "round": args.round,
                "candidate_id": exp["opt_id"],
                "parent_id": exp.get("parent_id"),
                "dispatch_path": DISPATCH_PATH,
                "workload": WORKLOAD,
                "duration_us": prof["duration_us"],
                "l0_pass": True,
                "msprof_raw_path": prof["msprof_raw_path"],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            }
        )
        results.append(
            {**exp, "status": "ok", "l0_pass": True, "duration_us": prof["duration_us"]}
        )
        print(f"  L0 pass, median Task Duration = {prof['duration_us']} us")

    # summary table: candidates vs current best (B2 structured backflow)
    base = None
    if os.path.isfile(PERF_RECORDS):
        with open(PERF_RECORDS) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            if r.get("candidate_id") == "baseline":
                base = r["duration_us"]
    print("\n| branch | l0 | duration_us | vs_best | status |")
    print("|---|---|---:|---:|---|")
    for r in results:
        dur = r.get("duration_us")
        rel = ""
        if dur is not None and base:
            rel = f"{(base - dur) / base * 100:+.1f}%"
        print(
            f"| {r['opt_id']} | {r.get('l0_pass', '-')} | "
            f"{dur if dur is not None else '-'} | {rel} | {r['status']} |"
        )
    print(f"\n(current best baseline for comparison: {base} us)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
