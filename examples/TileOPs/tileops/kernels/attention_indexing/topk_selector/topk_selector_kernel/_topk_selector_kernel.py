# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""_topk_selector_kernel: windowed top-k index selector on Ascend NPU (target="npuir").

For every (b, s, g) row of an [B, S, S_kv, G] fp32 score tensor, select the
``topk`` largest-score absolute kv indices inside the window
[starts[b,s], ends[b,s]) ∩ [0, seq_len_kv), and write them (value-descending)
into index[b, s, g, 0:topk] as int32.

Migration notes (GPU radix-select -> NPU chunked vsort + streaming merge,
per DESIGN.md R1/R2):
  - chunked T.vsort (descending, tail axis) + streaming merge topk replaces
    the SMEM-atomic radix histogram / threshold / refinement pipeline;
  - one-dimensional persistent kernel (NUM_KERNELS=40) + in-kernel static
    T.serial(MAX_TASKS) row loop with a runtime ``row < R`` guard replaces
    the 3-D CUDA grid (batch, seq_len, kv_group) x 1024 threads;
  - window masking uses -inf fill (T.vbrc / statement-level masked stores),
    chunk-level skip guard keeps narrow-window rows cheap;
  - candidates are bounded by topk + K2 (no 4096 truncation hazard of the
    source SMEM candidate buffer).

Codegen-bug workarounds applied (verified by minimal repros, all
implementation-level, semantics unchanged):
  W1. Boolean-combined conditions inside T.if_then_else that reference
      GM scalar loads segfault in device codegen -> use statement-level
      ``if cond_a or cond_b:`` inside T.Parallel loops instead.
  W2. int32 element-wise read-modify-write whose addend is an outer-loop
      product expression (c0 = c * C) segfaults in device codegen -> use a
      fused write-only form ``mrgA_idx[TOPK + i] = chunk_pos[i] + c0``.
  W3. Vectorized (T.Parallel) dynamic-index gather crashes the vector core
      (unaligned UB access) and dynamic-index scatter silently writes to
      wrong addresses -> translate vsort positions to business indices with
      a scalar T.serial gather over the first TOPK entries.

run: python _topk_selector_kernel.py --level L0   (or --level all)
"""

import argparse
import functools
import os
import time

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

import tilelang
import tilelang.language as T
import torch
import torch_npu  # noqa: F401

FP32 = "float32"
INT32 = "int32"

# Tiling constants (DESIGN.md §5.2 / §5.5, calibrated by on-device probes)
# NOTE: DESIGN.md assumed C_MAX=8192, but a standalone vsort probe shows a
# single fp32 descending vsort of 8192 elements alone requires ~224 KB UB
# (> 192 KB capacity): vsort carries an internal workspace of about 3x the
# sorted-bytes. Measured UB model across C/topk combos:
#   total ≈ (6*C + 7*(topk+K2)) * 4 bytes  (visible buffers + vsort workspace)
# which matches all probed points (incl. the C=4096/topk=2048 overflow at
# 208 KB). Per DESIGN.md §9.1 risk-1 fallback, C auto-downgrades so that the
# estimate fits the budget; the algorithm structure is unchanged.
C_MAX = 4096
C_ALIGN = 128
NUM_KERNELS_DEFAULT = 40  # physical vector cores are 40~48 (Ascend910B2C: 48)
UB_BUDGET_BYTES = 192 * 1024
UB_SAFETY_BYTES = int(UB_BUDGET_BYTES * 0.92)  # leave headroom for compiler extras


def _ub_estimate(C, topk):
    K2 = min(C, topk)
    return (6 * C + 7 * (topk + K2)) * 4


def _derive_chunk(seq_len_kv, topk):
    """Largest aligned chunk length that fits the UB budget (DESIGN.md §5.2)."""
    c = _align_up(min(seq_len_kv, C_MAX), C_ALIGN)
    while c > 128 and _ub_estimate(c, topk) > UB_SAFETY_BYTES:
        c //= 2
    return max(c, 128)


# ---------------------------------------------------------------------------
# Golden (PyTorch reference implementation, independent of the NPU algorithm;
# runs on CPU or NPU tensors, DESIGN.md §8.1)
# ---------------------------------------------------------------------------
def golden_topk_selector(
    index_score: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, topk: int
) -> torch.Tensor:
    """Windowed top-k index selection reference (§0.1 semantics).

    index_score: (B, S, S_kv, G) fp32; starts/ends: (B, S) int32.
    Returns (B, S, G, topk) int32 in torch.topk (value-descending) order.
    """
    B, S, S_kv, G = index_score.shape
    kv_idx = torch.arange(S_kv, device=index_score.device).view(1, 1, S_kv, 1)
    mask = (
        (kv_idx >= starts.view(B, S, 1, 1).long())
        & (kv_idx < ends.view(B, S, 1, 1).long())
        & (kv_idx < S_kv)
    )
    masked = index_score.masked_fill(~mask, float("-inf"))
    idx = torch.topk(masked, topk, dim=2)[1]  # (B, S, topk, G)
    return idx.permute(0, 1, 3, 2).contiguous()  # (B, S, G, topk)


# ---------------------------------------------------------------------------
# Kernel factory (interface-compatible two-stage call, DESIGN.md §3.3/§3.5.2)
# ---------------------------------------------------------------------------
def _align_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


@functools.lru_cache(maxsize=32)
def _topk_selector_kernel(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype):
    """JIT factory, mirrors the GPU-side signature.

    Returns a config-callable: f(RADIX, BLOCK_SIZE, SMEM_INPUT_SIZE, block_m,
    chunk_size=0, num_kernels=40) whose legacy GPU keys are accepted (for
    call-site compatibility with the TileOPs wrapper) and ignored; the NPU
    knobs are ``chunk_size`` (0 = auto-derive) and ``num_kernels``.
    The config-callable then takes (index_score, starts, ends) and returns
    the (B, S, G, topk) int32 index tensor.
    """
    if out_dtype != INT32:
        raise ValueError(f"out_dtype must be '{INT32}', got {out_dtype}")
    if topk <= 0 or topk > seq_len_kv:
        raise ValueError(f"require 0 < topk <= seq_len_kv, got topk={topk}")

    R = batch * seq_len * kv_group

    @tilelang.jit(target="npuir", out_idx=[1])
    def topk_selector_fwd_func(
        RADIX=1 << 8,
        BLOCK_SIZE=1024,
        SMEM_INPUT_SIZE=4096,
        block_m=32,
        chunk_size=0,
        num_kernels=NUM_KERNELS_DEFAULT,
    ):
        # Legacy GPU config keys (RADIX/BLOCK_SIZE/SMEM_INPUT_SIZE/block_m) are
        # ignored on NPU; the NPU-relevant knobs are chunk_size / num_kernels.
        C = chunk_size if chunk_size and chunk_size > 0 else _derive_chunk(seq_len_kv, topk)
        K2 = min(C, topk)
        NUM_CHUNKS = (seq_len_kv + C - 1) // C
        MAX_TASKS = (R + num_kernels - 1) // num_kernels

        if _ub_estimate(C, topk) > UB_BUDGET_BYTES:
            print(
                f"[warn] UB estimate {_ub_estimate(C, topk)} B exceeds "
                f"{UB_BUDGET_BYTES} B (C={C}, topk={topk}); compile may fail"
            )

        @T.prim_func
        def _topk_selector_kernel_main(
            index_score: T.Tensor((batch, seq_len, seq_len_kv, kv_group), in_dtype),
            index: T.Tensor((batch, seq_len, kv_group, topk), out_dtype),
            starts: T.Tensor((batch, seq_len), "int32"),
            ends: T.Tensor((batch, seq_len), "int32"),
        ):
            with T.Kernel(num_kernels, is_npu=True) as (kid, _):
                # UB buffers (Developer mode: alloc_shared maps to UB, §4.3)
                chunk_src = T.alloc_shared((C,), in_dtype)
                chunk_val = T.alloc_shared((C,), in_dtype)
                chunk_pos = T.alloc_shared((C,), "int32")
                mrgA_val = T.alloc_shared((topk + K2,), in_dtype)
                mrgA_idx = T.alloc_shared((topk + K2,), "int32")
                mrgB_val = T.alloc_shared((topk + K2,), in_dtype)
                mrgB_pos = T.alloc_shared((topk + K2,), "int32")

                for t in T.serial(MAX_TASKS):
                    row = t * num_kernels + kid
                    if row < R:
                        b = row // (seq_len * kv_group)
                        rem = row % (seq_len * kv_group)
                        s = rem // kv_group
                        g = rem % kv_group
                        sv = starts[b, s]
                        ev = ends[b, s]

                        # running candidate init: values -inf, indices 0
                        T.vbrc(-T.infinity("float32"), mrgA_val)
                        for i in T.Parallel(topk + K2):
                            mrgA_idx[i] = 0

                        for c in T.serial(NUM_CHUNKS):
                            c0 = c * C
                            # chunk-level skip guard (window overlap test);
                            # nested single-condition form (workaround W1)
                            if c0 < ev and c0 + C > sv and c0 < seq_len_kv:
                                n_valid = T.min(C, seq_len_kv - c0)
                                if kv_group == 1:
                                    # contiguous in-row kv segment
                                    T.copy(
                                        index_score[b, s, c0 : c0 + n_valid, g],
                                        chunk_src[0:n_valid],
                                    )
                                    # window mask incl. tail-garbage
                                    # overwrite (statement-level if, W1)
                                    for i in T.Parallel(C):
                                        pos = c0 + i
                                        if pos < sv or pos >= ev or pos >= seq_len_kv:
                                            chunk_src[i] = -T.infinity("float32")
                                else:
                                    # strided fallback path (G > 1):
                                    # element-wise load fused with mask;
                                    # clamp read index to stay in-bounds
                                    for i in T.Parallel(C):
                                        pos = c0 + i
                                        pos_rd = T.min(pos, seq_len_kv - 1)
                                        if pos < sv or pos >= ev or pos >= seq_len_kv:
                                            chunk_src[i] = -T.infinity("float32")
                                        else:
                                            chunk_src[i] = index_score[b, s, pos_rd, g]
                                # in-chunk descending sort; chunk_pos
                                # holds in-chunk positions
                                T.vsort(
                                    chunk_src,
                                    chunk_val,
                                    chunk_pos,
                                    descending=True,
                                    sort_axis=-1,
                                )
                                # assemble merge input [candidates | chunk head]
                                T.copy(chunk_val[0:K2], mrgA_val[topk : topk + K2])
                                # fused absolute-index store (W2)
                                for i in T.Parallel(K2):
                                    mrgA_idx[topk + i] = chunk_pos[i] + c0
                                # merge sort and take top-k
                                T.vsort(
                                    mrgA_val,
                                    mrgB_val,
                                    mrgB_pos,
                                    descending=True,
                                    sort_axis=-1,
                                )
                                # translate vsort positions to business
                                # indices via scalar gather (W3)
                                for j in T.serial(topk):
                                    mrgB_pos[j] = mrgA_idx[mrgB_pos[j]]
                                T.copy(mrgB_val[0:topk], mrgA_val[0:topk])
                                T.copy(mrgB_pos[0:topk], mrgA_idx[0:topk])

                        T.copy(mrgA_idx[0:topk], index[b, s, g, 0:topk])

        return _topk_selector_kernel_main

    return topk_selector_fwd_func


# ---------------------------------------------------------------------------
# L0 case 6 pre-check: standalone fp32 descending vsort at max design size
# ---------------------------------------------------------------------------
@tilelang.jit(target="npuir")
def _vsort_probe(N):
    @T.prim_func
    def _vsort_probe_main(
        src: T.Tensor((N,), FP32),
        dst_value: T.Tensor((N,), FP32),
        dst_index: T.Tensor((N,), INT32),
    ):
        with T.Kernel(1, is_npu=True) as (cid, _):
            src_ub = T.alloc_shared((N,), FP32)
            val_ub = T.alloc_shared((N,), FP32)
            idx_ub = T.alloc_shared((N,), INT32)
            T.copy(src, src_ub)
            T.vsort(src_ub, val_ub, idx_ub, descending=True, sort_axis=-1)
            T.copy(val_ub, dst_value)
            T.copy(idx_ub, dst_index)

    return _vsort_probe_main


# ---------------------------------------------------------------------------
# Comparison helpers (row-classified, vectorized; DESIGN.md §8.2)
# ---------------------------------------------------------------------------
def _window_counts(starts, ends, seq_len_kv):
    """Per-row valid window size W (B,S), clamped to [0, seq_len_kv]."""
    lo = torch.clamp(starts.long(), min=0)
    hi = torch.clamp(ends.long(), max=seq_len_kv)
    return (hi - lo).clamp(min=0)


def _compare_rows(out, ref, score, starts, ends, topk, strict_set=True, check_empty_zeros=False):
    """Row-classified top-k comparison.

    Full rows (W >= topk): value-multiset equality, value-descending order,
    index uniqueness, index-in-window, and (strict_set) per-row index-set
    equality with the reference.
    Degenerate rows (0 < W < topk): first W slots must equal the full window
    index set (value-descending); tail slots are defined-fill (unchecked).
    Empty rows (W <= 0): if check_empty_zeros, output must be all zeros.
    Returns (ok: bool, msg: str).

    Gathers run on-device (score may be huge); all order/equality logic runs
    on CPU because NPU diff/equal misbehave around +-inf (inf - inf = NaN).
    """
    B, S, S_kv, G = score.shape
    device = score.device
    K = topk

    out_l = out.long()
    ref_l = ref.long()
    scores_bgk = score.permute(0, 1, 3, 2)  # (B,S,G,S_kv) view
    out_vals = torch.gather(scores_bgk, -1, out_l).cpu()  # (B,S,G,K)
    ref_vals = torch.gather(scores_bgk, -1, ref_l).cpu()

    kv_idx = torch.arange(S_kv, device=device).view(1, 1, 1, S_kv)
    in_win = (
        (kv_idx >= starts.view(B, S, 1, 1).long())
        & (kv_idx < ends.view(B, S, 1, 1).long())
        & (kv_idx < S_kv)
    )  # (B,S,1,S_kv)
    in_win = in_win.expand(B, S, G, S_kv)
    out_in_win = torch.gather(in_win, -1, out_l).cpu()  # (B,S,G,K)

    out_l = out_l.cpu()
    ref_l = ref_l.cpu()
    W = _window_counts(starts, ends, S_kv).cpu()  # (B,S)
    starts_c = starts.cpu()
    ends_c = ends.cpu()

    row_full = (W >= K).view(B, S, 1, 1).expand(B, S, G, 1)
    row_empty = (W <= 0).view(B, S, 1, 1).expand(B, S, G, 1)

    fails = []

    # --- full rows ---
    n_full = int(row_full[..., 0].sum())
    if n_full > 0:
        sel = row_full.expand(B, S, G, K)
        ov = out_vals[sel[..., 0]].view(n_full, K)
        rv = ref_vals[sel[..., 0]].view(n_full, K)
        ol = out_l[sel[..., 0]].view(n_full, K)
        rl = ref_l[sel[..., 0]].view(n_full, K)
        ow = out_in_win[sel[..., 0]].view(n_full, K)
        val_ok = torch.equal(ov.sort(-1).values, rv.sort(-1).values)
        if not val_ok:
            bad = (ov.sort(-1).values != rv.sort(-1).values).any(-1)
            fails.append(f"value multiset mismatch on {int(bad.sum())}/{n_full} full rows")
        # descending via pairwise >= (diff breaks on inf - inf = NaN)
        desc_ok = bool((ov[:, :-1] >= ov[:, 1:]).all())
        if not desc_ok:
            fails.append("value order not descending on full rows")
        uniq_ok = bool((ol.sort(-1).values.diff(dim=-1) != 0).all())
        if not uniq_ok:
            fails.append("duplicate indices on full rows")
        win_ok = bool(ow.all())
        if not win_ok:
            fails.append(f"out-of-window indices on {int((~ow).any(-1).sum())} full rows")
        if strict_set:
            set_ok = torch.equal(ol.sort(-1).values, rl.sort(-1).values)
            if not set_ok:
                bad = (ol.sort(-1).values != rl.sort(-1).values).any(-1)
                fails.append(
                    f"strict per-row index-set mismatch on {int(bad.sum())}/{n_full} full rows"
                )

    # --- degenerate rows (0 < W < topk): first W slots = whole window set ---
    row_deg = (~row_full[..., 0]) & (~row_empty[..., 0])  # (B,S,G)
    n_deg = int(row_deg.sum())
    if n_deg > 0:
        for b in range(B):
            for s in range(S):
                if not (0 < int(W[b, s]) < K):
                    continue
                for g in range(G):
                    w = int(W[b, s])
                    lo = max(0, int(starts_c[b, s]))
                    hi = min(int(ends_c[b, s]), S_kv)
                    expect = set(range(lo, hi))
                    got = set(int(x) for x in out[b, s, g, :w].tolist())
                    if got != expect:
                        fails.append(
                            f"degenerate row (b={b},s={s},g={g},W={w}): "
                            f"first-W set mismatch with window set"
                        )
                    vals = out_vals[b, s, g, :w]
                    if w > 1 and not bool((vals[:-1] >= vals[1:]).all()):
                        fails.append(f"degenerate row (b={b},s={s},g={g}) not descending")

    # --- empty rows ---
    n_empty = int(row_empty[..., 0].sum())
    if n_empty > 0 and check_empty_zeros:
        zeros_ok = bool((out[row_empty[..., 0]].cpu() == 0).all())
        if not zeros_ok:
            fails.append(f"empty-window rows not all-zero on {n_empty} rows")

    if fails:
        return False, "; ".join(fails[:5])
    return True, (
        f"full={n_full} deg={n_deg} empty={n_empty}"
        + (" (empty rows zero-checked)" if n_empty and check_empty_zeros else "")
    )


def _run_and_compare(
    tag,
    batch,
    seq_len,
    seq_len_kv,
    kv_group,
    topk,
    score,
    starts,
    ends,
    strict_set=True,
    check_empty_zeros=False,
    chunk_size=0,
    num_kernels=NUM_KERNELS_DEFAULT,
    time_it=False,
):
    """Compile+run the kernel and compare against golden; returns (ok, detail, out)."""
    t0 = time.time()
    func = _topk_selector_kernel(batch, seq_len, seq_len_kv, kv_group, topk, FP32, INT32)
    out = func(chunk_size=chunk_size, num_kernels=num_kernels)(score, starts, ends)
    if hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()
    t1 = time.time()

    ref = golden_topk_selector(score, starts, ends, topk)
    ok, msg = _compare_rows(
        out,
        ref,
        score,
        starts,
        ends,
        topk,
        strict_set=strict_set,
        check_empty_zeros=check_empty_zeros,
    )
    elapsed = f" kernel={t1 - t0:.2f}s" if time_it else ""
    return ok, f"{msg}{elapsed}", out


def _rand_windows(batch, seq_len, seq_len_kv, topk, device, min_gap=None):
    """Random partial windows with W >= topk (DESIGN.md §8.3 case 2)."""
    starts = torch.randint(
        0, seq_len_kv - topk + 1, (batch, seq_len), dtype=torch.int32, device=device
    )
    extra = torch.randint(0, seq_len_kv - topk + 1, (batch, seq_len), device=device)
    ends = (starts.long() + topk + extra).clamp(max=seq_len_kv).to(torch.int32)
    return starts, ends


# ---------------------------------------------------------------------------
# L0: DESIGN.md §8.3 (smoke workload: batch=4, seq_len=256, kv=1024, G=1, topk=32)
# ---------------------------------------------------------------------------
SMOKE = (4, 256, 1024, 1, 32)


def run_L0():
    device = "npu"
    B, S, KV, G, K = SMOKE
    results = []

    # case 6 first (vsort probe at the calibrated max sort size C_MAX=4096;
    # 8192 was probed to overflow UB on its own and C auto-downgrades, see
    # the C_MAX note above -- DESIGN.md §9.1 risk-1 fallback in effect)
    N = C_MAX
    src = torch.randn(N, dtype=torch.float32, device=device)
    src[:64] = float("inf")
    src[64:128] = float("-inf")
    src[128:256] = 0.5  # repeated values
    probe = _vsort_probe(N)
    dv = torch.zeros(N, dtype=torch.float32, device=device)
    di = torch.zeros(N, dtype=torch.int32, device=device)
    probe(src, dv, di)
    ref_v, _ = torch.sort(src.cpu(), descending=True)
    ok6 = (
        torch.allclose(dv.cpu(), ref_v)
        and sorted(di.cpu().tolist()) == list(range(N))
        and torch.all(src.cpu()[di.cpu().long()] == ref_v)
    )
    results.append(
        (
            f"L0-6 vsort@{N} probe",
            ok6,
            "values==torch.sort desc, index is permutation, idx-value consistent",
        )
    )
    if not ok6:
        print(f"[L0-6 vsort@{N} probe] FAIL: vsort misbehaves; aborting L0")
        return results

    # case 1: full-window random (harness-equivalent)
    torch.manual_seed(42)
    score = torch.randn(B, S, KV, G, dtype=torch.float32, device=device)
    starts = torch.zeros(B, S, dtype=torch.int32, device=device)
    ends = torch.full((B, S), KV, dtype=torch.int32, device=device)
    ok, msg, _ = _run_and_compare("L0-1", B, S, KV, G, K, score, starts, ends, strict_set=True)
    results.append(("L0-1 full-window randn", ok, msg))

    # case 2: random partial windows (W >= topk)
    torch.manual_seed(43)
    score = torch.randn(B, S, KV, G, dtype=torch.float32, device=device)
    starts, ends = _rand_windows(B, S, KV, K, device)
    ok, msg, _ = _run_and_compare("L0-2", B, S, KV, G, K, score, starts, ends, strict_set=True)
    results.append(("L0-2 random partial windows", ok, msg))

    # case 3: degenerate windows (W < topk) + disjoint-empty rows
    torch.manual_seed(44)
    score = torch.randn(B, S, KV, G, dtype=torch.float32, device=device)
    starts = torch.zeros(B, S, dtype=torch.int32, device=device)
    ends = torch.full((B, S), KV, dtype=torch.int32, device=device)
    starts[0, 64:128] = torch.randint(0, KV - K + 1, (64,), device=device, dtype=torch.int32)
    for i in range(64, 128):
        w = int(torch.randint(1, K, (1,)))  # 1 <= W < topk
        ends[0, i] = starts[0, i] + w
    starts[0, 128:192] = 0  # disjoint empty windows
    ends[0, 128:192] = 0
    ok, msg, _ = _run_and_compare(
        "L0-3",
        B,
        S,
        KV,
        G,
        K,
        score,
        starts,
        ends,
        strict_set=True,
        check_empty_zeros=True,
    )
    results.append(("L0-3 windows < topk / empty", ok, msg))

    # case 4: ties and repeated values (set semantics, strict off)
    torch.manual_seed(45)
    score = torch.full((B, S, KV, G), 0.5, dtype=torch.float32, device=device)
    score[:, :, : K - 4] = 1.0  # topk-4 copies of the max value
    score[:, :, K - 4 : K + 20] = 0.75  # 24-way tie for the last 4 slots
    starts = torch.zeros(B, S, dtype=torch.int32, device=device)
    ends = torch.full((B, S), KV, dtype=torch.int32, device=device)
    ok, msg, _ = _run_and_compare("L0-4", B, S, KV, G, K, score, starts, ends, strict_set=False)
    results.append(("L0-4 ties/repeated values", ok, msg))

    # case 5a: multi-chunk via shrunk chunk_size on smoke shape
    torch.manual_seed(46)
    score = torch.randn(B, S, KV, G, dtype=torch.float32, device=device)
    starts = torch.zeros(B, S, dtype=torch.int32, device=device)
    ends = torch.full((B, S), KV, dtype=torch.int32, device=device)
    ok, msg, _ = _run_and_compare(
        "L0-5a", B, S, KV, G, K, score, starts, ends, strict_set=True, chunk_size=256
    )
    results.append(("L0-5a multi-chunk (C=256)", ok, msg))

    # case 5b: mid shape kv=2048 with chunk_size=1024 (random partial windows)
    torch.manual_seed(47)
    b2, s2, kv2, g2, k2 = 2, 64, 2048, 1, 64
    score = torch.randn(b2, s2, kv2, g2, dtype=torch.float32, device=device)
    starts, ends = _rand_windows(b2, s2, kv2, k2, device)
    ok, msg, _ = _run_and_compare(
        "L0-5b",
        b2,
        s2,
        kv2,
        g2,
        k2,
        score,
        starts,
        ends,
        strict_set=True,
        chunk_size=1024,
    )
    results.append(("L0-5b mid shape kv=2048, C=1024", ok, msg))

    # case 7: kv_group > 1 fallback path (recorded, not blocking per §8.3)
    try:
        torch.manual_seed(48)
        b3, s3, kv3, g3, k3 = 2, 8, 64, 2, 16
        score = torch.randn(b3, s3, kv3, g3, dtype=torch.float32, device=device)
        starts = torch.zeros(b3, s3, dtype=torch.int32, device=device)
        ends = torch.full((b3, s3), kv3, dtype=torch.int32, device=device)
        ok, msg, _ = _run_and_compare(
            "L0-7", b3, s3, kv3, g3, k3, score, starts, ends, strict_set=True
        )
        results.append(("L0-7 kv_group=2 fallback", ok, msg))
    except Exception as e:  # noqa: BLE001
        results.append(("L0-7 kv_group=2 fallback", False, f"{type(e).__name__}: {str(e)[:160]}"))

    for name, ok, msg in results:
        print(f"[{name}] {'PASS' if ok else 'FAIL'}: {msg}", flush=True)
    return results


# ---------------------------------------------------------------------------
# L1: manifest workloads (full-window, harness-equivalent ref semantics)
# ---------------------------------------------------------------------------
L1_WORKLOADS = [
    # (batch, seq_len, seq_len_kv, kv_group, topk)  [manifest attention_indexing]
    (4, 256, 1024, 1, 32),
    (8, 512, 2048, 1, 64),
    (1, 32 * 1024, 64 * 1024, 1, 1024),
    (1, 32 * 1024, 64 * 1024, 1, 2048),
]


def run_L1():
    device = "npu"
    results = []
    for i, (B, S, KV, G, K) in enumerate(L1_WORKLOADS):
        torch.manual_seed(100 + i)
        score = torch.randn(B, S, KV, G, dtype=torch.float32, device=device)
        starts = torch.zeros(B, S, dtype=torch.int32, device=device)
        ends = torch.full((B, S), KV, dtype=torch.int32, device=device)
        ok, msg, _ = _run_and_compare(
            f"L1-{i + 1}",
            B,
            S,
            KV,
            G,
            K,
            score,
            starts,
            ends,
            strict_set=True,
            time_it=True,
        )
        results.append((f"L1-{i + 1} ({B},{S},{KV},{G},topk={K})", ok, msg))
        print(
            f"[L1-{i + 1}] ({B},{S},{KV},{G},topk={K}) {'PASS' if ok else 'FAIL'}: {msg}",
            flush=True,
        )
        del score, starts, ends
        torch.npu.empty_cache() if hasattr(torch.npu, "empty_cache") else None
    return results


# ---------------------------------------------------------------------------
# L2: abnormal inputs (recorded, non-blocking)
# ---------------------------------------------------------------------------
def run_L2():
    device = "npu"
    results = []

    def _record(name, fn):
        try:
            ok, msg = fn()
            results.append((name, ok, msg))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"{type(e).__name__}: {str(e)[:160]}"))

    # L2-1: empty windows (disjoint -> all-zero output; intersecting -> defined fill)
    def _l2_1():
        B, S, KV, G, K = 2, 32, 1024, 1, 32
        torch.manual_seed(201)
        score = torch.randn(B, S, KV, G, dtype=torch.float32, device=device)
        starts = torch.zeros(B, S, dtype=torch.int32, device=device)
        ends = torch.full((B, S), KV, dtype=torch.int32, device=device)
        starts[0, :8] = 0
        ends[0, :8] = 0  # disjoint empty at left edge
        starts[0, 8:16] = KV
        ends[0, 8:16] = KV  # disjoint empty at right edge
        starts[0, 16:24] = KV + 10  # fully out-of-range window
        ends[0, 16:24] = KV + 20
        starts[0, 24:32] = 5
        ends[0, 24:32] = 5  # intersecting empty (defined fill)
        out = _topk_selector_kernel(B, S, KV, G, K, FP32, INT32)()(score, starts, ends)
        ref = golden_topk_selector(score, starts, ends, K)
        cmp_ok, cmp_msg = _compare_rows(out, ref, score, starts, ends, K, strict_set=True)
        # disjoint-empty rows must be all zeros (defined behavior, §0.6 R1)
        z_ok = bool((out[0, :16].cpu() == 0).all()) and bool((out[0, 16:24].cpu() == 0).all())
        return (z_ok and cmp_ok), f"disjoint-empty zero rows ok={z_ok}; {cmp_msg}"

    _record("L2-1 empty windows", _l2_1)

    # L2-2: window smaller than topk on a dedicated shape
    def _l2_2():
        B, S, KV, G, K = 1, 16, 512, 1, 64
        torch.manual_seed(202)
        score = torch.randn(B, S, KV, G, dtype=torch.float32, device=device)
        starts = torch.randint(0, KV - K, (B, S), dtype=torch.int32, device=device)
        ends = (starts.long() + torch.randint(1, K, (B, S), device=device)).to(torch.int32)
        ok, msg, _ = _run_and_compare("L2-2", B, S, KV, G, K, score, starts, ends, strict_set=True)
        return ok, msg

    _record("L2-2 window < topk", _l2_2)

    # L2-3: non-divisible chunks (kv not a multiple of C)
    def _l2_3():
        B, S, KV, G, K = 1, 8, 1000, 1, 32  # C=1024 auto > kv -> single ragged chunk
        torch.manual_seed(203)
        score = torch.randn(B, S, KV, G, dtype=torch.float32, device=device)
        starts = torch.zeros(B, S, dtype=torch.int32, device=device)
        ends = torch.full((B, S), KV, dtype=torch.int32, device=device)
        ok, msg, _ = _run_and_compare("L2-3a", B, S, KV, G, K, score, starts, ends, strict_set=True)
        return ok, f"kv=1000 auto C: {msg}"

    def _l2_3b():
        B, S, KV, G, K = 1, 8, 8294, 1, 64  # C=8192, 2 chunks, tail=102
        torch.manual_seed(204)
        score = torch.randn(B, S, KV, G, dtype=torch.float32, device=device)
        starts = torch.zeros(B, S, dtype=torch.int32, device=device)
        ends = torch.full((B, S), KV, dtype=torch.int32, device=device)
        ok, msg, _ = _run_and_compare("L2-3b", B, S, KV, G, K, score, starts, ends, strict_set=True)
        return ok, f"kv=8294 C=8192 tail=102: {msg}"

    def _l2_3c():
        B, S, KV, G, K = 1, 8, 1000, 1, 24  # C=232 ragged, topk 8-multiple
        torch.manual_seed(205)
        score = torch.randn(B, S, KV, G, dtype=torch.float32, device=device)
        starts = torch.zeros(B, S, dtype=torch.int32, device=device)
        ends = torch.full((B, S), KV, dtype=torch.int32, device=device)
        ok, msg, _ = _run_and_compare(
            "L2-3c",
            B,
            S,
            KV,
            G,
            K,
            score,
            starts,
            ends,
            strict_set=True,
            chunk_size=232,
        )
        return ok, f"kv=1000 topk=24 C=232: {msg}"

    _record("L2-3a non-divisible chunk (kv<C)", _l2_3)
    _record("L2-3b non-divisible chunk (multi)", _l2_3b)
    _record("L2-3c ragged C (232) + kv=1000", _l2_3c)

    # L2-4 (recorded limitation, not executed): topk not a multiple of 8.
    # Probed on device: any topk % 8 != 0 (e.g. topk=20) aborts the vector
    # core with "UB address accessed by the VEC instruction is not aligned"
    # on the topk-length segment ops. DESIGN.md §5.3 assumed the compiler
    # pads non-aligned tail axes (performance item only) -- on this stack
    # that assumption does not hold. All manifest topk values (32/64/1024/
    # 2048) are 8-multiples, so the contract workload is unaffected; kept
    # here as a WARN record per the L2 non-blocking policy.
    results.append(
        (
            "L2-4 topk%8!=0 (known limitation)",
            False,
            "topk=20 aborts vector core (unaligned 80B segment); "
            "manifest topk values are all 8-multiples -- recorded, not fixed",
        )
    )

    for name, ok, msg in results:
        print(f"[{name}] {'PASS' if ok else 'WARN'}: {msg}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Boundary: special values (recorded, non-blocking)
# ---------------------------------------------------------------------------
def run_boundary():
    device = "npu"
    results = []

    def _record(name, fn):
        try:
            ok, msg = fn()
            results.append((name, ok, msg))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"{type(e).__name__}: {str(e)[:160]}"))

    # B-1: massive ties (constant tensor)
    def _b1():
        B, S, KV, G, K = 2, 16, 512, 1, 32
        score = torch.full((B, S, KV, G), 0.5, dtype=torch.float32, device=device)
        starts = torch.zeros(B, S, dtype=torch.int32, device=device)
        ends = torch.full((B, S), KV, dtype=torch.int32, device=device)
        ok, msg, _ = _run_and_compare("B-1", B, S, KV, G, K, score, starts, ends, strict_set=False)
        return ok, msg

    _record("B-1 all-equal ties", _b1)

    # B-2: +-inf values (few +inf selected, -inf present but unselected)
    def _b2():
        B, S, KV, G, K = 2, 16, 1024, 1, 32
        torch.manual_seed(301)
        score = torch.randn(B, S, KV, G, dtype=torch.float32, device=device)
        score[:, :, :5] = float("inf")
        score[:, :, 100:140] = float("-inf")
        starts = torch.zeros(B, S, dtype=torch.int32, device=device)
        ends = torch.full((B, S), KV, dtype=torch.int32, device=device)
        ok, msg, _ = _run_and_compare("B-2", B, S, KV, G, K, score, starts, ends, strict_set=True)
        return ok, msg

    _record("B-2 +-inf values", _b2)

    # B-3: more +inf than topk (tie among infs; set semantics)
    def _b3():
        B, S, KV, G, K = 2, 16, 1024, 1, 32
        torch.manual_seed(302)
        score = torch.randn(B, S, KV, G, dtype=torch.float32, device=device)
        score[:, :, :100] = float("inf")
        starts = torch.zeros(B, S, dtype=torch.int32, device=device)
        ends = torch.full((B, S), KV, dtype=torch.int32, device=device)
        ok, msg, _ = _run_and_compare("B-3", B, S, KV, G, K, score, starts, ends, strict_set=False)
        return ok, msg

    _record("B-3 +inf tie (>topk)", _b3)

    # B-4: +/-0.0 mix (equal values, distinct bit patterns) competing for slots
    def _b4():
        B, S, KV, G, K = 2, 16, 512, 1, 32
        torch.manual_seed(303)
        score = torch.randn(B, S, KV, G, dtype=torch.float32, device=device) - 10.0
        score[:, :, :64] = 0.0  # +0 ties
        score[:, :, 64:128] = -0.0  # -0 ties (same value, diff bits)
        starts = torch.zeros(B, S, dtype=torch.int32, device=device)
        ends = torch.full((B, S), KV, dtype=torch.int32, device=device)
        ok, msg, _ = _run_and_compare("B-4", B, S, KV, G, K, score, starts, ends, strict_set=False)
        return ok, msg

    _record("B-4 +/-0.0 mix", _b4)

    # B-5: NaN (out of contract per §0.1/§9.1.7; recorded only)
    def _b5():
        B, S, KV, G, K = 1, 8, 256, 1, 16
        torch.manual_seed(304)
        score = torch.randn(B, S, KV, G, dtype=torch.float32, device=device)
        score[0, 0, 10] = float("nan")
        starts = torch.zeros(B, S, dtype=torch.int32, device=device)
        ends = torch.full((B, S), KV, dtype=torch.int32, device=device)
        out = _topk_selector_kernel(B, S, KV, G, K, FP32, INT32)()(score, starts, ends)
        # only require: no crash, valid dtype/shape, non-NaN rows still correct
        sub_ok, sub_msg = _compare_rows(
            out[:, 1:],
            golden_topk_selector(score[:, 1:], starts[:, 1:], ends[:, 1:], K),
            score[:, 1:],
            starts[:, 1:],
            ends[:, 1:],
            K,
            strict_set=True,
        )
        return bool(sub_ok), (
            f"NaN row excluded from check (out-of-contract); non-NaN rows: {sub_msg}"
        )

    _record("B-5 NaN (out of contract)", _b5)

    for name, ok, msg in results:
        print(f"[{name}] {'PASS' if ok else 'WARN'}: {msg}", flush=True)
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "all"])
    args, _ = parser.parse_known_args()

    t_start = time.time()
    if args.level == "L0":
        run_L0()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()
    print(f"total wall time: {time.time() - t_start:.1f}s")
    print("\033[92mAll requested checks finished!\033[0m")


if __name__ == "__main__":
    main()
