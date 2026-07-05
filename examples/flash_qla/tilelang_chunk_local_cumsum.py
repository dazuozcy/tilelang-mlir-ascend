
import os

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

DTYPE_TO_STR = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    cu_cpu = cu_seqlens.cpu()
    seqlens = torch.diff(cu_cpu)
    num_chunks_per_seq = (seqlens + chunk_size - 1) // chunk_size
    batch_idx = torch.repeat_interleave(
        torch.arange(len(seqlens), dtype=cu_cpu.dtype), num_chunks_per_seq
    )
    chunk_idx = torch.cat(
        [torch.arange(n, dtype=cu_cpu.dtype) for n in num_chunks_per_seq.tolist()]
    )
    return torch.stack([batch_idx, chunk_idx], dim=1).to(cu_seqlens.device)


@tilelang.jit(target="npuir")
def tilelang_chunk_local_cumsum(
    H,
    chunk_size,
    accum_dtype,
    g_dtype,
    seqlen_dtype,
    is_varlen,
    reverse,
):
    data_batch_size = T.dynamic("data_batch_size")
    num_cu_seqlens = T.dynamic("num_cu_seqlens")
    num_tokens = T.dynamic("num_tokens")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size

    g_shape = (data_batch_size, num_tokens, H)

    @T.macro
    def kernel_body(
        bb,
        batch_idx,
        chunk_idx,
        seq_start_idx,
        seq_end_idx,
        g_raw,
        g_cumsum,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S

        gT_shared = T.alloc_shared((block_S, H), dtype=g_dtype)
        g_acc = T.alloc_fragment((block_S, H), dtype=accum_dtype)
        g_cs = T.alloc_fragment((block_S, H), dtype=accum_dtype)
        g_out = T.alloc_shared((block_S, H), dtype=g_dtype)

        if not reverse:
            if right <= seq_end_idx:
                T.copy(g_raw[bb, left:right, 0:H], gT_shared)
            else:
                for j, i in T.Parallel(block_S, H):
                    if left + j < seq_end_idx:
                        gT_shared[j, i] = g_raw[bb, left + j, i]
                    else:
                        gT_shared[j, i] = 0

            T.vcast(gT_shared, g_acc)
            T.cumsum(g_acc, g_cs, dim=0, reverse=False)
            T.vcast(g_cs, g_out)

            if right <= seq_end_idx:
                T.copy(g_out, g_cumsum[bb, left:right, 0:H])
            else:
                for j, i in T.Parallel(block_S, H):
                    if left + j < seq_end_idx:
                        g_cumsum[bb, left + j, i] = g_out[j, i]
        else:
            for j, i in T.Parallel(block_S, H):
                src_pos = left + block_S - 1 - j
                if src_pos < seq_end_idx:
                    gT_shared[j, i] = g_raw[bb, src_pos, i]
                else:
                    gT_shared[j, i] = 0

            T.vcast(gT_shared, g_acc)
            T.cumsum(g_acc, g_cs, dim=0, reverse=False)
            T.vcast(g_cs, g_out)

            for j, i in T.Parallel(block_S, H):
                dst_pos = left + block_S - 1 - j
                if dst_pos < seq_end_idx:
                    g_cumsum[bb, dst_pos, i] = g_out[j, i]

    if is_varlen:

        @T.prim_func
        def tilelang_chunk_local_cumsum_kernel(
            g_raw: T.Tensor(g_shape, dtype=g_dtype),
            cu_seqlens: T.Tensor([num_cu_seqlens], dtype=seqlen_dtype),
            chunk_indices: T.Tensor([num_chunks, 2], dtype=seqlen_dtype),
            g_cumsum: T.Tensor(g_shape, dtype=g_dtype),
        ):
            _ = T.meta_var(seqlen_dtype)  # help parser resolve closure dtype
            with T.Kernel(num_chunks, is_npu=True) as (bc, _):
                bb = 0
                batch_idx = chunk_indices[bc, 0]
                chunk_idx = chunk_indices[bc, 1]
                seq_start_idx = cu_seqlens[batch_idx]
                seq_end_idx = cu_seqlens[batch_idx + 1]

                kernel_body(
                    bb,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    g_raw,
                    g_cumsum,
                )

                left = seq_start_idx + chunk_idx * block_S
                if batch_idx == num_cu_seqlens - 2:
                    for j, i in T.Parallel(block_S, H):
                        if left + j >= seq_end_idx and left + j < num_tokens:
                            g_cumsum[bb, left + j, i] = 0

    else:
        num_chunks_per_seq = (num_tokens + block_S - 1) // block_S

        @T.prim_func
        def tilelang_chunk_local_cumsum_kernel(
            g_raw: T.Tensor(g_shape, dtype=g_dtype),
            g_cumsum: T.Tensor(g_shape, dtype=g_dtype),
        ):
            with T.Kernel(data_batch_size * num_chunks_per_seq, is_npu=True) as (bc, _):
                bb = bc % data_batch_size
                batch_idx = bb
                chunk_idx = bc // data_batch_size
                seq_start_idx = 0
                seq_end_idx = num_tokens

                kernel_body(
                    bb,
                    batch_idx,
                    chunk_idx,
                    seq_start_idx,
                    seq_end_idx,
                    g_raw,
                    g_cumsum,
                )

    return tilelang_chunk_local_cumsum_kernel


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    reverse: bool = False,
):
    batch_size, num_tokens, H = g.shape
    assert g.stride(-1) == 1

    if cu_seqlens is None:
        seqlen_dtype = "int32"
        is_varlen = False
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        seqlen_dtype = DTYPE_TO_STR[cu_seqlens.dtype]
        is_varlen = True

    g_cumsum = torch.empty_like(g)

    tilelang_chunk_local_cumsum_kernel = tilelang_chunk_local_cumsum(
        H,
        chunk_size,
        accum_dtype="float32",
        g_dtype=DTYPE_TO_STR[g.dtype],
        seqlen_dtype=seqlen_dtype,
        is_varlen=is_varlen,
        reverse=reverse,
    )
    if is_varlen:
        tilelang_chunk_local_cumsum_kernel(g, cu_seqlens, chunk_indices, g_cumsum)
    else:
        tilelang_chunk_local_cumsum_kernel(g, g_cumsum)

    return g_cumsum


def chunk_local_cumsum_torch_ref(
    g: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    reverse: bool = False,
):
    batch_size, num_tokens, H = g.shape
    g_cumsum = torch.zeros_like(g)

    if cu_seqlens is None:
        ranges = [(b, 0, num_tokens) for b in range(batch_size)]
    else:
        ranges = [
            (b, int(cu_seqlens[b]), int(cu_seqlens[b + 1]))
            for b in range(cu_seqlens.shape[0] - 1)
        ]

    for batch_idx, seq_start, seq_end in ranges:
        bb = 0 if cu_seqlens is not None else batch_idx
        seqlen = seq_end - seq_start
        num_c = (seqlen + chunk_size - 1) // chunk_size
        for c in range(num_c):
            left = seq_start + c * chunk_size
            right = min(left + chunk_size, seq_end)
            chunk = g[bb, left:right, :].float()
            if reverse:
                cs = torch.flip(torch.cumsum(torch.flip(chunk, [0]), dim=0), [0])
            else:
                cs = torch.cumsum(chunk, dim=0)
            g_cumsum[bb, left:right, :] = cs.to(g.dtype)
    return g_cumsum


def _run_case(
    batch_size,
    num_tokens,
    H,
    chunk_size,
    reverse,
    is_varlen,
    dtype,
    rtol,
    atol,
    seed=0,
):
    torch.manual_seed(seed)
    tilelang.cache.clear_cache()

    if is_varlen:
        g = torch.randn(1, num_tokens, H, dtype=dtype, device="npu")
        num_chunks_total = num_tokens // chunk_size
        boundaries = torch.sort(torch.randint(1, num_chunks_total, (batch_size - 1,)))[
            0
        ]
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                (boundaries * chunk_size).to(torch.int32),
                torch.tensor([num_tokens], dtype=torch.int32),
            ]
        ).npu()
        ref = chunk_local_cumsum_torch_ref(
            g.cpu(), chunk_size, cu_seqlens.cpu(), reverse
        )
        out = chunk_local_cumsum(g, chunk_size, cu_seqlens, reverse)
    else:
        g = torch.randn(batch_size, num_tokens, H, dtype=dtype, device="npu")
        ref = chunk_local_cumsum_torch_ref(g.cpu(), chunk_size, None, reverse)
        out = chunk_local_cumsum(g, chunk_size, None, reverse)

    torch.testing.assert_close(out.cpu(), ref, rtol=rtol, atol=atol)
    tag = f"{'varlen' if is_varlen else 'nonvarlen'} rev={reverse} {dtype} "
    tag += f"shape=({batch_size},{num_tokens},{H}) cs={chunk_size}"
    print(f"[PASS] {tag}")


def _run_varlen_case(
    H,
    chunk_size,
    reverse,
    dtype,
    rtol,
    atol,
    seq_lens,
    seed=0,
):
    """Varlen test with explicit per-sequence lengths (may be non-divisible
    by chunk_size).  num_tokens is set to the last chunk's right boundary so
    that the padding-zeroing logic is exercised without leaving garbage
    beyond the kernel's coverage (matching original kernel behaviour)."""
    torch.manual_seed(seed)
    tilelang.cache.clear_cache()

    cu_seqlens_cpu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
        dtype=torch.int32,
    )
    total_len = int(cu_seqlens_cpu[-1].item())
    last_start = int(cu_seqlens_cpu[-2].item())
    last_seqlen = seq_lens[-1]
    last_chunk_right = (
        last_start + ((last_seqlen + chunk_size - 1) // chunk_size) * chunk_size
    )
    num_tokens = last_chunk_right

    cu_seqlens = cu_seqlens_cpu.npu()
    g = torch.randn(1, num_tokens, H, dtype=dtype, device="npu")
    ref = chunk_local_cumsum_torch_ref(g.cpu(), chunk_size, cu_seqlens.cpu(), reverse)
    out = chunk_local_cumsum(g, chunk_size, cu_seqlens, reverse)

    # Compare valid sequence positions
    torch.testing.assert_close(
        out.cpu()[:, :total_len, :], ref[:, :total_len, :], rtol=rtol, atol=atol
    )
    # Padding region [total_len, num_tokens) should be zeroed by the kernel
    if num_tokens > total_len:
        pad = out.cpu()[:, total_len:num_tokens, :]
        torch.testing.assert_close(pad, torch.zeros_like(pad), rtol=0, atol=0)

    tag = f"varlen(custom) rev={reverse} {dtype} H={H} cs={chunk_size} "
    tag += f"nt={num_tokens} seqs={seq_lens}"
    print(f"[PASS] {tag}")


def main():
    fp16 = (torch.float16, 1e-3, 1e-3)
    bf16 = (torch.bfloat16, 2e-2, 2e-2)

    # Case 1: non-varlen, reverse=False, divisible
    _run_case(2, 128, 64, 64, False, False, *fp16)
    # Case 2: non-varlen, reverse=False, tail (not divisible)
    _run_case(2, 100, 64, 64, False, False, *fp16)
    # Case 3: non-varlen, reverse=True, divisible
    _run_case(2, 128, 64, 64, True, False, *fp16)
    # Case 4: non-varlen, reverse=True, tail
    _run_case(2, 100, 64, 64, True, False, *fp16)
    # Case 5: varlen, reverse=False, padding
    _run_case(4, 256, 64, 64, False, True, *fp16)
    # Case 6: varlen, reverse=True, padding
    _run_case(4, 256, 64, 64, True, True, *fp16)
    # Case 7: non-varlen, bf16
    _run_case(2, 128, 64, 64, False, False, *bf16)
    # Case 8: varlen, bf16
    _run_case(3, 192, 64, 64, False, True, *bf16)
    # Case 9: non-varlen, reverse=True, bf16, tail
    _run_case(2, 100, 32, 64, True, False, *bf16)

    # Edge cases: non-divisible seq lengths + padding zeroing
    # Case 10: varlen, seqs not divisible by chunk_size, with padding
    _run_varlen_case(64, 64, False, *fp16, seq_lens=[100, 70, 50])
    # Case 11: varlen, reverse, non-divisible + padding
    _run_varlen_case(64, 64, True, *fp16, seq_lens=[100, 70, 50])
    # Case 12: varlen, large H, padding, single-element tail chunk
    _run_varlen_case(128, 64, False, *fp16, seq_lens=[65, 70])
    # Case 13: varlen, reverse, bf16, non-divisible + padding
    _run_varlen_case(64, 64, True, *bf16, seq_lens=[100, 70, 50])
    # Case 14: single batch single chunk
    _run_case(1, 64, 16, 64, False, False, *fp16)
    # Case 15: reverse, single batch single chunk
    _run_case(1, 64, 16, 64, True, False, *fp16)

    print("\n\033[92mAll tests passed!\033[0m")


if __name__ == "__main__":
    main()
