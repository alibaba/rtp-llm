"""Exercise the fixed CUDA tiles with Qwen3.6 and Qwen3-8B shapes."""

import argparse

import torch
from torch.nn.functional import linear
from gdn_test import bit_mismatches

from rtp_llm.models_py.modules.base.cuda.norm import FusedQKRMSNorm
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.models_py.triton_kernels.attention.batch_invariant import (
    batch_invariant_paged_attention,
)
from rtp_llm.models_py.triton_kernels.common.batch_invariant_linear import (
    batch_invariant_linear,
)
from rtp_llm.ops import ActivationType, HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import rtp_llm_ops
from rtp_llm.utils.model_weight import W


def check_projections(qwen3_8b):
    vocab_size = 151936 if qwen3_8b else 248320
    hybrid_dims = (
        (2048, 1),
        (2048, 256),
        (2048, 1024),
        (2048, 4096),
        (2048, 5120),
        (2048, 12352),
        (4096, 2048),
        (512, 2048),
        (2048, vocab_size),
    )
    dims = (
        ((4096, 6144), (4096, 4096), (4096, 24576), (12288, 4096), (4096, vocab_size))
        if qwen3_8b
        else hybrid_dims
    )
    for k, n in dims:
        x = torch.randn((193, k), device="cuda", dtype=torch.bfloat16) * 0.125
        # RTP stores regular projections in [in, out] order and the LM head in
        # [out, in] order. Exercise both strides.
        head = n == vocab_size
        weight_shape = (n, k) if head else (k, n)
        weight = torch.randn(weight_shape, device="cuda", dtype=x.dtype) * 0.03125
        if not head:
            weight = weight.T
        target = x[42:43].clone()
        expected = batch_invariant_linear(target, weight, output_fp32=head)[0]
        assert expected.dtype == (torch.float32 if head else torch.bfloat16)
        reference = linear(target.float(), weight.float())[0]
        torch.testing.assert_close(
            expected.float(),
            reference,
            rtol=1e-4 if head else 0.004,
            atol=1e-5 if qwen3_8b else 3e-6,
        )
        batches = ((x[:64], 42), (x[:65].roll(22, 0), 64), (x, 42))
        for batch, slot in batches:
            actual = batch_invariant_linear(batch, weight, output_fp32=head)[slot]
            assert bit_mismatches(expected, actual) == 0, (k, n, batch.shape[0], slot)
        print(f"projection {k} x {n} {expected.dtype}: equal", flush=True)
        if qwen3_8b:
            native = rtp_llm_ops.cublas_gemm_bf16_bf16_fp32 if head else linear
            solo = native(target, weight)[0]
            differences = [
                bit_mismatches(solo, native(batch, weight)[slot])
                for batch, slot in batches
            ]
            print(f"regular projection {k} x {n}, 64/65/193 rows: {differences}")


def check_dense_mlp():
    x = torch.randn((193, 4096), device="cuda", dtype=torch.bfloat16) * 0.125
    weights = {
        key: torch.randn(shape, device=x.device, dtype=x.dtype) * 0.03125
        for key, shape in ((W.ffn_w13, (4096, 24576)), (W.ffn_w2, (12288, 4096)))
    }
    for invariant in (False, True):
        hw = HWKernelConfig()
        hw.batch_invariant = invariant
        mlp = DenseMLP(ActivationType.Swiglu, ParallelismConfig(), weights, None, hw)
        expected = mlp(x[42:43])[0]
        differences = [
            bit_mismatches(expected, mlp(batch)[row])
            for batch, row in ((x[:64], 42), (x[:65].roll(22, 0), 64), (x, 42))
        ]
        print(f"dense MLP, invariant={invariant}, 64/65/193 rows: {differences}")
        if invariant:
            assert differences == [0, 0, 0]


def check_qk_norm(head_dim, query_heads, kv_heads):
    qkv = torch.randn(
        (193, (query_heads + 2 * kv_heads) * head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    weights = torch.rand((2, head_dim), dtype=qkv.dtype, device=qkv.device)
    norm = FusedQKRMSNorm(*weights, query_heads, kv_heads, head_dim)
    expected = norm(qkv[42:43].clone())[0]
    for batch, slot in ((qkv[:64], 42), (qkv[:65].roll(22, 0), 64), (qkv, 42)):
        assert bit_mismatches(expected, norm(batch.clone())[slot]) == 0
    print("Q/K RMSNorm: equal for solo and 64/65/193 rows", flush=True)


def check_attention(trtllm_baseline, dims, query_heads, kv_heads):
    q_len, noise_len = 131, 257
    page_size = 64
    query = torch.randn(
        (q_len + noise_len, query_heads, dims), device="cuda", dtype=torch.bfloat16
    )
    cache = torch.randn(
        (10, 2, kv_heads, page_size, dims), device="cuda", dtype=query.dtype
    )
    page_ids = torch.tensor(
        [[3, 1, 2, 0, 0], [6, 5, 8, 4, 7]], dtype=torch.int32, device="cuda"
    )
    scale = dims**-0.5
    positions = torch.arange(1, q_len + 1, dtype=torch.int32, device="cuda")
    request = torch.zeros_like(positions)
    expected = batch_invariant_paged_attention(
        query[:q_len], cache, page_ids[:1], request, positions, scale
    )
    combined_lengths = torch.cat(
        (torch.arange(1, noise_len + 1, device="cuda", dtype=torch.int32), positions)
    )
    combined_requests = torch.cat(
        (torch.zeros_like(combined_lengths[:noise_len]), request + 1)
    )
    reordered = torch.cat((query[q_len:], query[:q_len]))
    mixed = batch_invariant_paged_attention(
        reordered, cache, page_ids.flip(0), combined_requests, combined_lengths, scale
    )
    assert bit_mismatches(expected, mixed[noise_len:]) == 0
    for start, end in ((0, 64), (64, 128), (128, 131), (130, 131)):
        split = batch_invariant_paged_attention(
            query[start:end],
            cache,
            page_ids[:1],
            request[start:end],
            positions[start:end],
            scale,
        )
        assert bit_mismatches(expected[start:end], split) == 0

    # Check causal indexing across noncontiguous pages against FP32 attention.
    k, v = (
        x.reshape(kv_heads, -1, dims)[:, :q_len]
        .repeat_interleave(query_heads // kv_heads, dim=0)
        .float()
        for x in cache[page_ids[0, :3]].permute(1, 2, 0, 3, 4)
    )
    q = query[:q_len].transpose(0, 1).float()
    scores = (q @ k.transpose(-1, -2)) * scale
    mask = torch.ones((q_len, q_len), dtype=torch.bool, device="cuda").triu(1)
    reference = (scores.masked_fill(mask, -torch.inf).softmax(-1) @ v).transpose(0, 1)
    torch.testing.assert_close(expected.float(), reference, rtol=0.008, atol=0.006)
    print("paged attention: equal for reordered rows and prefill cuts", flush=True)

    if trtllm_baseline:
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache

        work = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")

        def trt(q, pages, q_lengths, lengths):
            def ints(values):
                return torch.tensor(values, dtype=torch.int32, device="cuda")

            def cu(values):
                return torch.nn.functional.pad(
                    ints(values).cumsum(0, dtype=torch.int32), (1, 0)
                )

            return trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=cache,
                workspace_buffer=work,
                block_tables=pages,
                seq_lens=ints(lengths),
                max_q_len=max(q_lengths),
                max_kv_len=max(lengths),
                bmm1_scale=scale,
                bmm2_scale=1.0,
                batch_size=len(lengths),
                cum_seq_lens_q=cu(q_lengths),
                cum_seq_lens_kv=cu(lengths),
                window_left=-1,
                sinks=None,
                out_dtype=q.dtype,
            )

        solo = trt(query[:q_len], page_ids[:1], [q_len], [q_len])
        mixed = trt(reordered, page_ids.flip(0), [noise_len, q_len], [noise_len, q_len])
        tail = trt(query[128:q_len], page_ids[:1], [q_len - 128], [q_len])
        print(
            "TRTLLM baseline mixed differences:",
            bit_mismatches(solo, mixed[noise_len:]),
        )
        print("TRTLLM baseline split differences:", bit_mismatches(solo[128:], tail))


def check_decode_attention(trtllm_baseline, dims, query_heads, kv_heads):
    torch.manual_seed(537)
    q = torch.randn((33, query_heads, dims), dtype=torch.bfloat16, device="cuda")
    cache = torch.randn((512, 2, kv_heads, 64, dims), dtype=q.dtype, device="cuda")
    target_pages = torch.arange(1, 257, device="cuda", dtype=torch.int32)
    peer_pages = torch.arange(256, 512, device="cuda", dtype=torch.int32)
    table = torch.cat((target_pages[None], peer_pages.expand(32, -1)))
    requests = torch.arange(33, device="cuda", dtype=torch.int32)
    scale = dims**-0.5
    for target in (193, 4097):
        lengths = torch.tensor(
            [target] + [16380] * 32, device="cuda", dtype=torch.int32
        )
        solo = batch_invariant_paged_attention(
            q[:1], cache, table[:1], requests[:1], lengths[:1], scale
        )
        for size in (2, 9, 33):
            mixed = batch_invariant_paged_attention(
                q[:size], cache, table[:size], requests[:size], lengths[:size], scale
            )
            assert bit_mismatches(solo, mixed[:1]) == 0

        if trtllm_baseline:
            from flashinfer.decode import trtllm_batch_decode_with_kv_cache

            workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")

            def decode(size, max_seq_len):
                return trtllm_batch_decode_with_kv_cache(
                    query=q[:size],
                    kv_cache=cache,
                    workspace_buffer=workspace,
                    block_tables=table[:size],
                    seq_lens=lengths[:size],
                    max_seq_len=max_seq_len,
                    bmm1_scale=scale,
                    bmm2_scale=1.0,
                    window_left=-1,
                    sinks=None,
                    out_dtype=q.dtype,
                    q_len_per_req=1,
                )

            solo = decode(1, target)
            different = bit_mismatches(solo, decode(2, 16380)[:1])
            print(
                f"TRTLLM decode baseline, target KV length {target}: {different} differences"
            )
    print("paged decode attention: equal with longer peers", flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trtllm-baseline", action="store_true")
    parser.add_argument("--qwen3-8b", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(157)
    torch.backends.cuda.matmul.allow_tf32 = False
    dims = (128, 32, 8) if args.qwen3_8b else (256, 16, 2)
    check_projections(args.qwen3_8b)
    if args.qwen3_8b:
        check_dense_mlp()
        check_qk_norm(*dims)
    check_attention(args.trtllm_baseline, *dims)
    check_decode_attention(args.trtllm_baseline, *dims)


if __name__ == "__main__":
    main()
