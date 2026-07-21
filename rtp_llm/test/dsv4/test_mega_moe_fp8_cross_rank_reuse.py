"""Stress DeepGEMM MegaMoE final-combine correctness across eight ranks.

This test gives each rank deliberately uneven work, repeatedly reuses the
same symmetric buffer, and checks both the returned output and the retained
source-token-ordered final-combine input.  That separates an L1/L2 or remote
write error from a final-combine load/reduction/output-store error.

Production-shape FP8 regression run::

    torchrun --standalone --nproc-per-node=8 \
        test_mega_moe_fp8_cross_rank_reuse.py --rounds 1000

Use ``--weight-format fp4`` to exercise the FP8-activation/FP4-weight kernel.
For an unfixed DeepGEMM wheel, add ``--expect-mismatch`` to turn a reproduced
single-row corruption into the expected negative-test outcome.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

DEEP_GEMM_PATH = os.environ.get("DEEP_GEMM_SITE_PACKAGES")
if DEEP_GEMM_PATH:
    sys.path.insert(0, DEEP_GEMM_PATH)
os.environ.setdefault(
    "DG_JIT_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), f"deep_gemm_jit_{os.getuid()}"),
)
os.makedirs(os.environ["DG_JIT_CACHE_DIR"], exist_ok=True)

import deep_gemm
import torch
import torch.distributed as dist


NUM_EXPERTS = 256
NUM_TOPK = 8
HIDDEN = 6144
INTER = 2048
FP8_BLOCK = 128
FP4_BLOCK = 32
T_SMALL = 6144
T_LARGE = 8192


def _make_weights(e_local: int, device: torch.device, weight_format: str):
    generator = torch.Generator(device=device)
    generator.manual_seed(20260721 + dist.get_rank())

    def make_fp8_weight(n: int, k: int):
        weight = torch.randint(
            -2,
            3,
            (e_local, n, k),
            dtype=torch.int8,
            device=device,
            generator=generator,
        ).to(torch.float8_e4m3fn)
        raw_scale = torch.ones(
            (e_local, n // FP8_BLOCK, k // FP8_BLOCK),
            dtype=torch.float32,
            device=device,
        )
        packed_scale = deep_gemm.transform_sf_into_required_layout(
            raw_scale, n, k, (FP8_BLOCK, FP8_BLOCK), e_local
        )
        return weight, packed_scale

    def make_fp4_weight(n: int, k: int):
        weight = torch.randint(
            -128,
            127,
            (e_local, n, k // 2),
            dtype=torch.int8,
            device=device,
            generator=generator,
        )
        raw_scale = torch.ones(
            (e_local, n, k // FP4_BLOCK),
            dtype=torch.float32,
            device=device,
        )
        packed_scale = deep_gemm.transform_sf_into_required_layout(
            raw_scale, n, k, (1, FP4_BLOCK), e_local
        )
        return weight, packed_scale

    make_weight = make_fp8_weight if weight_format == "fp8" else make_fp4_weight
    l1 = make_weight(2 * INTER, HIDDEN)
    l2 = make_weight(HIDDEN, INTER)
    if weight_format == "fp8":
        return deep_gemm.transform_weights_for_mega_moe_fp8(l1, l2)
    return deep_gemm.transform_weights_for_mega_moe(l1, l2)


def _make_inputs(tokens: int, rank: int, seed: int, device: torch.device):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + rank * 100003)
    x = torch.randn(
        (tokens, HIDDEN), dtype=torch.bfloat16, device=device, generator=generator
    )
    x_fp8, x_sf = deep_gemm.per_token_cast_to_fp8(
        x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    topk_idx = torch.randint(
        0,
        NUM_EXPERTS,
        (tokens, NUM_TOPK),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    topk_weights = torch.rand(
        (tokens, NUM_TOPK),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return x_fp8, x_sf, topk_idx, topk_weights


def _pack(buf, inputs, tokens: int) -> None:
    x_fp8, x_sf, topk_idx, topk_weights = inputs
    buf.x[:tokens].copy_(x_fp8[:tokens])
    buf.x_sf[:tokens].copy_(x_sf[:tokens])
    buf.topk_idx[:tokens].copy_(topk_idx[:tokens])
    buf.topk_weights[:tokens].copy_(topk_weights[:tokens])


def _call(
    buf, l1, l2, tokens: int, post_barrier: bool, weight_format: str
) -> torch.Tensor:
    y = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
    if weight_format == "fp8":
        deep_gemm.fp8_fp8_mega_moe(
            y,
            l1,
            l2,
            buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            fast_math=False,
            assume_all_topk_valid=True,
        )
    else:
        deep_gemm.fp8_fp4_mega_moe(
            y,
            l1,
            l2,
            buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            fast_math=False,
        )
    if post_barrier:
        # Enqueued on the current CUDA stream: no host synchronization, but no
        # rank may reuse the shared remote combine/workspace storage early.
        buf.handle.barrier(channel=0)
    return y


def _combine_view(buf, tokens: int) -> torch.Tensor:
    combine_offset = (
        buf.l2_acts_sf.data_ptr()
        - buf.buffer.data_ptr()
        + buf.l2_acts_sf.numel() * buf.l2_acts_sf.element_size()
    )
    combine_numel = NUM_TOPK * buf.num_max_tokens_per_rank * HIDDEN
    combine_num_bytes = combine_numel * torch.bfloat16.itemsize
    if combine_offset + combine_num_bytes > buf.buffer.numel():
        raise RuntimeError(
            "combine buffer is outside the symmetric allocation: "
            f"offset={combine_offset} bytes={combine_num_bytes} "
            f"capacity={buf.buffer.numel()}"
        )
    return (
        buf.buffer.narrow(0, combine_offset, combine_num_bytes)
        .view(torch.bfloat16)
        .view(NUM_TOPK, buf.num_max_tokens_per_rank, HIDDEN)[:, :tokens]
    )


def _ordered_combine_output(
    buf, tokens: int, weight_format: str, chunk_cols: int = 512
) -> torch.Tensor:
    combine = _combine_view(buf, tokens)
    result = torch.empty(
        (tokens, HIDDEN), dtype=torch.bfloat16, device=combine.device
    )
    weights = buf.topk_weights[:tokens]
    for col_begin in range(0, HIDDEN, chunk_cols):
        col_end = min(col_begin + chunk_cols, HIDDEN)
        reference = torch.zeros(
            (tokens, col_end - col_begin),
            dtype=torch.float32,
            device=combine.device,
        )
        for slot_idx in range(NUM_TOPK):
            contribution = combine[slot_idx, :, col_begin:col_end].float()
            if weight_format == "fp8":
                reference = torch.addcmul(
                    reference,
                    contribution,
                    weights[:, slot_idx : slot_idx + 1],
                )
            else:
                # The FP4 kernel applies routing weights before writing the
                # per-slot combine buffer; final combine is an ordered sum.
                reference += contribution
        result[:, col_begin:col_end] = reference.to(torch.bfloat16)
    return result


def _diff_summary(actual: torch.Tensor, expected: torch.Tensor):
    diff = actual != expected
    changed = int(torch.count_nonzero(diff).item())
    changed_rows = torch.nonzero(torch.any(diff, dim=1), as_tuple=False).flatten()
    max_abs = 0.0
    if changed:
        max_abs = float((actual.float() - expected.float()).abs().max().item())
    return changed, int(changed_rows.numel()), max_abs, changed_rows


def main() -> None:
    global NUM_EXPERTS, NUM_TOPK, HIDDEN, INTER, T_SMALL, T_LARGE

    parser = argparse.ArgumentParser()
    parser.add_argument("--post-barrier", action="store_true")
    parser.add_argument("--expect-mismatch", action="store_true")
    parser.add_argument("--weight-format", choices=("fp8", "fp4"), default="fp8")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--num-experts", type=int, default=NUM_EXPERTS)
    parser.add_argument("--num-topk", type=int, default=NUM_TOPK)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--inter", type=int, default=INTER)
    parser.add_argument("--small-tokens", type=int, default=T_SMALL)
    parser.add_argument("--large-tokens", type=int, default=T_LARGE)
    args = parser.parse_args()

    NUM_EXPERTS = args.num_experts
    NUM_TOPK = args.num_topk
    HIDDEN = args.hidden
    INTER = args.inter
    T_SMALL = args.small_tokens
    T_LARGE = args.large_tokens

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if NUM_EXPERTS % world_size != 0:
        raise RuntimeError(
            f"num_experts={NUM_EXPERTS} must be divisible by world_size={world_size}"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    l1, l2 = _make_weights(NUM_EXPERTS // world_size, device, args.weight_format)
    get_symm_buffer = (
        deep_gemm.get_symm_buffer_for_mega_moe_fp8
        if args.weight_format == "fp8"
        else deep_gemm.get_symm_buffer_for_mega_moe
    )
    buf = get_symm_buffer(
        group=dist.group.WORLD,
        num_experts=NUM_EXPERTS,
        num_max_tokens_per_rank=T_LARGE,
        num_topk=NUM_TOPK,
        hidden=HIDDEN,
        intermediate_hidden=INTER,
        use_fp8_dispatch=True,
        activation="swiglu",
    )

    target = _make_inputs(T_SMALL, rank, 111, device)
    skew_small = _make_inputs(T_SMALL, rank, 222, device)
    skew_large = _make_inputs(T_LARGE, rank, 333, device)

    # Compile and establish a synchronized bitwise reference.
    _pack(buf, target, T_SMALL)
    reference = _call(
        buf, l1, l2, T_SMALL, post_barrier=True, weight_format=args.weight_format
    ).clone()
    reference_combine = _ordered_combine_output(buf, T_SMALL, args.weight_format)
    torch.cuda.synchronize(device)
    dist.barrier()

    mismatch_rounds = 0
    max_abs = 0.0
    for round_idx in range(args.rounds):
        skew_tokens = T_SMALL if rank == 0 else T_LARGE
        _pack(buf, skew_small if rank == 0 else skew_large, skew_tokens)
        _call(
            buf,
            l1,
            l2,
            skew_tokens,
            post_barrier=args.post_barrier,
            weight_format=args.weight_format,
        )

        # Deliberately do not synchronize between the asymmetric call and the
        # target call. Rank 0 reaches reuse much earlier than rank 1.
        _pack(buf, target, T_SMALL)
        current = _call(
            buf,
            l1,
            l2,
            T_SMALL,
            post_barrier=args.post_barrier,
            weight_format=args.weight_format,
        ).clone()

        # The kernel has completed before these same-stream PyTorch kernels
        # read its retained [topk, source_token, hidden] combine buffer.
        current_combine = _ordered_combine_output(buf, T_SMALL, args.weight_format)

        # Keep attempts isolated without hiding the race inside each pair.
        buf.handle.barrier(channel=1)
        torch.cuda.synchronize(device)
        changed, changed_rows, round_max, row_ids = _diff_summary(
            current, reference
        )
        raw_vs_combine = _diff_summary(current, current_combine)
        combine_vs_reference = _diff_summary(current_combine, reference_combine)
        if changed:
            mismatch_rounds += 1
            max_abs = max(max_abs, round_max)
            print(
                f"rank={rank} round={round_idx} changed={changed}/{current.numel()} "
                f"changed_rows={changed_rows} max_abs={round_max} "
                f"row_ids={row_ids[:32].tolist()} "
                f"raw_vs_current_combine={{elements:{raw_vs_combine[0]},"
                f"rows:{raw_vs_combine[1]},max_abs:{raw_vs_combine[2]}}} "
                f"current_combine_vs_reference={{elements:{combine_vs_reference[0]},"
                f"rows:{combine_vs_reference[1]},max_abs:{combine_vs_reference[2]}}}",
                flush=True,
            )
        dist.barrier()

    mismatch_tensor = torch.tensor(
        [mismatch_rounds], dtype=torch.int64, device=device
    )
    dist.all_reduce(mismatch_tensor, op=dist.ReduceOp.SUM)
    total_mismatch_rounds = int(mismatch_tensor.item())
    if rank == 0:
        print(
            f"weight_format={args.weight_format} post_barrier={args.post_barrier} "
            f"rounds={args.rounds} "
            f"rank_mismatch_rounds={total_mismatch_rounds} max_abs={max_abs}",
            flush=True,
        )

    dist.destroy_process_group()
    mismatch_observed = total_mismatch_rounds != 0
    if mismatch_observed != args.expect_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
