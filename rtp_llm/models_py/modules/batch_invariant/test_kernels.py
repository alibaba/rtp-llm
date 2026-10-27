"""GPU contract tests: bit patterns across execution shapes, plus accuracy."""

import pytest
import torch

from rtp_llm.models_py.modules.batch_invariant import (
    BatchInvariantPolicy,
    construction_policy,
)
from rtp_llm.models_py.modules.batch_invariant.activation import silu_and_mul
from rtp_llm.models_py.modules.batch_invariant.kernels import (
    linear,
    logprob_normalizers,
    logsumexp,
    paged_attention,
    rmsnorm,
    selected_logprobs,
)
from rtp_llm.models_py.modules.batch_invariant.rope import base_rope_cache


def same_bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def test_policy_scope_does_not_leak():
    assert not construction_policy().enabled
    policy = BatchInvariantPolicy(True)
    with pytest.raises(RuntimeError):
        with policy.construction():
            assert construction_policy() is policy
            with BatchInvariantPolicy(False).construction():
                assert not construction_policy().enabled
            assert construction_policy() is policy
            raise RuntimeError("unwind construction")
    assert not construction_policy().enabled


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(127, 257), (512, 1024), (4096, 512)])
def test_linear_row_positions_and_batch_buckets(dtype, shape):
    torch.manual_seed(17)
    k, n = shape
    weight = torch.randn(n, k, device="cuda", dtype=dtype) / k**0.5
    x = torch.randn(1, k, device="cuda", dtype=dtype)
    bias = torch.randn(n, device="cuda", dtype=dtype)
    ref = linear(x, weight, bias, output_dtype=torch.float32)
    expected = torch.nn.functional.linear(
        x.double(), weight.double(), bias.double()
    ).float()
    torch.testing.assert_close(ref, expected, atol=2e-5, rtol=2e-4)
    for batch in (2, 7, 15, 16, 17, 31, 32, 33, 64, 127, 128, 129, 257):
        packed = torch.randn(batch, k, device="cuda", dtype=dtype)
        for idx in (0, batch // 2, batch - 1):
            packed[idx].copy_(x[0])
            same_bits(
                ref,
                linear(packed, weight, bias, output_dtype=torch.float32)[idx : idx + 1],
            )


@pytest.mark.parametrize("width", [128, 1024, 4096, 513])
def test_norm_and_logprob(width):
    torch.manual_seed(19)
    x = torch.randn(35, width, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(width, dtype=x.dtype, device=x.device)
    norm = rmsnorm(x, w, 1e-6)
    target = torch.arange(35, device=x.device) % width
    logits = x.float() * 7
    lp = selected_logprobs(logits, target)
    expected_norm = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        * w.float()
    ).bfloat16()
    torch.testing.assert_close(norm, expected_norm, rtol=0.008, atol=0.008)
    torch.testing.assert_close(
        logsumexp(logits), logits.logsumexp(-1, keepdim=True), atol=8e-6, rtol=1e-6
    )
    for index in (0, 7, 16, 34):
        same_bits(norm[index : index + 1], rmsnorm(x[index : index + 1], w, 1e-6))
        same_bits(
            lp[index : index + 1],
            selected_logprobs(logits[index : index + 1], target[index : index + 1]),
        )
    order = torch.randperm(35, device=x.device)
    same_bits(lp[order], selected_logprobs(logits[order], target[order]))


def test_logprob_preserves_shift_before_log_sum():
    # Adding log(2) to this maximum rounds it away in FP32. The selected
    # logprob must still be -log(2), rather than zero.
    logits = torch.tensor([[16777216.0, 16777216.0, -16777216.0]], device="cuda")
    target = torch.zeros(1, device="cuda", dtype=torch.long)
    expected = torch.tensor([[-0.6931471805599453]], device="cuda")
    same_bits(selected_logprobs(logits, target), expected)
    normalizers = logprob_normalizers(logits)
    assert normalizers.shape == (1, 2) and normalizers.dtype == torch.float32
    for batch in (3, 33):
        same_bits(
            selected_logprobs(logits.repeat(batch, 1), target.repeat(batch)),
            expected.repeat(batch, 1),
        )


def test_rope_cache_isolated_across_model_configurations():
    device = torch.cuda.current_device()
    first = base_rope_cache(128, 10000.0, 1025, device)
    saved = first.clone()
    for dim, theta, length in (
        (64, 10000.0, 97),
        (128, 1000000.0, 1025),
        (256, 10000.0, 33),
    ):
        other = base_rope_cache(dim, theta, length, device)
        assert other.shape == (length, dim)
        assert other.data_ptr() != first.data_ptr()
        # Native FP32 storage must contain BF16-rounded values exactly.
        same_bits(other, other.bfloat16().float())
        same_bits(first, saved)
    same_bits(base_rope_cache(128, 10000.0, 1025, device), saved)


@pytest.mark.parametrize("width", [3, 128, 1025, 12288])
def test_swiglu_rounding_batch_order_and_graph(width):
    torch.manual_seed(97)
    x = torch.randn(7, width * 2, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.silu(x[:, :width]) * x[:, width:]
    same_bits(silu_and_mul(x), expected)
    for row in (0, 3, 6):
        same_bits(silu_and_mul(x[row : row + 1]), expected[row : row + 1])
    order = torch.tensor([6, 2, 4, 1, 5, 0, 3], device="cuda")
    same_bits(silu_and_mul(x[order]), expected[order])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        silu_and_mul(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = silu_and_mul(x)
    graph.replay()
    same_bits(output, expected)
    x.add_(0.25)
    graph.replay()
    same_bits(output, torch.nn.functional.silu(x[:, :width]) * x[:, width:])


@pytest.mark.parametrize("width", [64, 513, 4096, 8192])
def test_residual_norm_batch_order_and_graph(width):
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    operation = rtp_llm_ops.batch_invariant_post_residual_rmsnorm
    torch.manual_seed(239)
    x = torch.randn(33, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(width, device="cuda", dtype=x.dtype)
    reference = operation(x, weight, 1e-6)
    expected = (
        x.double()
        * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-6)
        * weight.double()
    ).bfloat16()
    torch.testing.assert_close(reference, expected, rtol=0.008, atol=0.008)
    for row in (0, 16, 32):
        same_bits(operation(x[row : row + 1], weight, 1e-6), reference[row : row + 1])
    order = torch.randperm(33, device="cuda")
    same_bits(operation(x[order], weight, 1e-6), reference[order])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        operation(x, weight, 1e-6)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = operation(x, weight, 1e-6)
    graph.replay()
    same_bits(output, reference)
    x.add_(0.25)
    graph.replay()
    same_bits(output, operation(x, weight, 1e-6))


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_attention_batch_order_pages_and_graph(head_dim):
    torch.manual_seed(23)
    h, kh, page, length = 8, 2, 16, 97
    blocks = (length + page - 1) // page
    cache = torch.randn(
        blocks * 2, 2, kh, page, head_dim, device="cuda", dtype=torch.bfloat16
    )
    table = torch.arange(blocks * 2, dtype=torch.int32, device="cuda").view(2, blocks)
    q = torch.randn(length, h, head_dim, device="cuda", dtype=torch.bfloat16)
    req = torch.zeros(length, dtype=torch.int32, device="cuda")
    pos = torch.arange(length, dtype=torch.int32, device="cuda")
    ref = paged_attention(q, cache, table, req, pos)
    # Independent dense FP64 attention for correctness, including causal tails.
    keys = (
        cache[:blocks, 0]
        .permute(1, 0, 2, 3)
        .reshape(kh, -1, head_dim)[:, :length]
        .repeat_interleave(h // kh, 0)
        .double()
    )
    vals = (
        cache[:blocks, 1]
        .permute(1, 0, 2, 3)
        .reshape(kh, -1, head_dim)[:, :length]
        .repeat_interleave(h // kh, 0)
        .double()
    )
    for i in (0, 15, 16, 31, 32, 63, 96):
        expected = (
            (q[i].double()[:, None, :] * keys[:, : i + 1]).sum(-1) / head_dim**0.5
        ).softmax(-1)
        expected = (expected[:, :, None] * vals[:, : i + 1]).sum(1).bfloat16()
        torch.testing.assert_close(ref[i], expected, atol=0.008, rtol=0.008)
        same_bits(
            ref[i : i + 1],
            paged_attention(q[i : i + 1], cache, table, req[i : i + 1], pos[i : i + 1]),
        )
    for chunk in (3, 16, 33):
        actual = torch.cat(
            [
                paged_attention(
                    q[i : i + chunk],
                    cache,
                    table,
                    req[i : i + chunk],
                    pos[i : i + chunk],
                )
                for i in range(0, length, chunk)
            ]
        )
        same_bits(ref, actual)
    order = torch.randperm(length, device="cuda")
    same_bits(
        ref[order], paged_attention(q[order], cache, table, req[order], pos[order])
    )
    permutation = torch.randperm(blocks * 2, device="cuda")
    remapped = torch.empty_like(table.flatten(), dtype=torch.long)
    remapped[permutation] = torch.arange(blocks * 2, device="cuda")
    same_bits(
        ref,
        paged_attention(q, cache[permutation], remapped[table.long()].int(), req, pos),
    )
    # Actual graph replay, including newly supplied values, not merely recapture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        paged_attention(q, cache, table, req, pos)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = paged_attention(q, cache, table, req, pos)
    graph.replay()
    same_bits(ref, graph_out)
    q.copy_(q.flip(0))
    graph.replay()
    same_bits(paged_attention(q, cache, table, req, pos), graph_out)


def test_logprob_extreme_logits_and_full_vocab():
    torch.manual_seed(31)
    x = torch.randn(9, 151936, device="cuda") * 50
    ids = torch.tensor([0, 16, 1023, 1024, 10000, 151935, 1, 7, 900], device="cuda")
    actual = selected_logprobs(x, ids)
    expected = (
        x.double().gather(1, ids[:, None]) - x.double().logsumexp(-1, keepdim=True)
    ).float()
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=1e-6)
    for i in range(9):
        same_bits(actual[i : i + 1], selected_logprobs(x[i : i + 1], ids[i : i + 1]))


def test_factory_policy_isolation_and_live_weight_update():
    from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
    from rtp_llm.models_py.modules.batch_invariant.linear import BatchInvariantLinear
    from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory

    weight = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    with BatchInvariantPolicy(True).construction():
        strict = LinearFactory.create_linear(weight, None, None, None)
        norm = RMSNorm(torch.ones(128, device="cuda", dtype=torch.bfloat16))
    with BatchInvariantPolicy(False).construction():
        ordinary = LinearFactory.create_linear(weight, None, None, None)
        assert not isinstance(ordinary, BatchInvariantLinear)
        assert norm.batch_invariant_policy.enabled
    assert isinstance(strict, BatchInvariantLinear)
    x = torch.randn(17, 128, device="cuda", dtype=torch.bfloat16)
    old = strict(x)
    weight.zero_()
    updated = strict(x)
    assert torch.count_nonzero(old) > 0
    assert torch.count_nonzero(updated) == 0
    same_bits(updated[:1], strict(x[:1]))
    with (
        BatchInvariantPolicy(True).construction(),
        pytest.raises(ValueError, match="compatible Linear"),
    ):
        LinearFactory.create_linear(weight.half(), None, None, None)


@pytest.mark.parametrize(
    "heads,kv_heads,dim", [(8, 2, 64), (28, 4, 128), (8, 8, 256), (32, 1, 128)]
)
def test_unified_packed_ragged_chunks_and_uninitialized_tail(heads, kv_heads, dim):
    """Packed Tensor Core rows must agree with isolated/reordered query rows."""
    torch.manual_seed(137)
    lengths, prefixes, page = [65, 129, 3], [31, 896, 7], 16
    blocks = (max(n + p for n, p in zip(lengths, prefixes)) + page - 1) // page
    table = torch.arange(3 * blocks, dtype=torch.int32, device="cuda").view(3, blocks)
    cache = torch.randn(
        3 * blocks, 2, kv_heads, page, dim, device="cuda", dtype=torch.bfloat16
    )
    # No kernel may read uninitialized entries past the last logical KV token.
    for req, (n, p) in enumerate(zip(lengths, prefixes)):
        for pos in range(n + p, blocks * page):
            cache[req * blocks + pos // page, :, :, pos % page] = float("nan")
    q = torch.randn(sum(lengths), heads, dim, device="cuda", dtype=torch.bfloat16)
    requests = torch.repeat_interleave(
        torch.arange(3, device="cuda", dtype=torch.int32),
        torch.tensor(lengths, device="cuda"),
    )
    positions = torch.cat(
        [
            torch.arange(p, p + n, device="cuda", dtype=torch.int32)
            for n, p in zip(lengths, prefixes)
        ]
    )
    starts = torch.tensor([0, 65, 194, 197], dtype=torch.int32, device="cuda")
    expected = paged_attention(q, cache, table, requests, positions)
    actual = paged_attention(q, cache, table, requests, positions, query_start=starts)
    same_bits(expected, actual)
    permutation = torch.randperm(len(q), device="cuda")
    same_bits(
        expected[permutation],
        paged_attention(
            q[permutation], cache, table, requests[permutation], positions[permutation]
        ),
    )
    start = 0
    for req, length in enumerate(lengths):
        for chunk in (1, 3, 17, 64):
            parts = []
            for i in range(0, length, chunk):
                end = min(i + chunk, length)
                part = slice(start + i, start + end)
                parts.append(
                    paged_attention(
                        q[part],
                        cache,
                        table[req : req + 1],
                        torch.zeros(end - i, device="cuda", dtype=torch.int32),
                        positions[part],
                        query_start=torch.tensor(
                            [0, end - i], device="cuda", dtype=torch.int32
                        ),
                    )
                )
            same_bits(expected[start : start + length], torch.cat(parts))
        start += length
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        paged_attention(q, cache, table, requests, positions, query_start=starts)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = paged_attention(
            q, cache, table, requests, positions, query_start=starts
        )
    graph.replay()
    same_bits(expected, captured)
    positions.sub_(1)
    graph.replay()
    same_bits(paged_attention(q, cache, table, requests, positions), captured)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_blas_strides_alignment_graph_and_global_isolation(transposed, output_dtype):
    torch.manual_seed(211)
    k, n = 512, 1024
    w = torch.randn(k, n, device="cuda", dtype=torch.bfloat16).T
    if not transposed:
        w = w.contiguous()
    # A nonzero storage offset exercises the plan's pointer alignment key.
    storage = torch.randn(1 + 129 * k, device="cuda", dtype=torch.bfloat16)
    x = storage[1:].view(129, k)
    before = (
        torch.backends.cuda.preferred_blas_library(),
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.matmul.fp32_precision,
    )
    result = linear(x, w, output_dtype=output_dtype)
    for i in (0, 7, 64, 128):
        same_bits(result[i : i + 1], linear(x[i : i + 1], w, output_dtype=output_dtype))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        linear(x, w, output_dtype=output_dtype)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = linear(x, w, output_dtype=output_dtype)
    graph.replay()
    same_bits(result, captured)
    w.zero_()
    graph.replay()
    same_bits(torch.zeros_like(result), captured)
    assert before == (
        torch.backends.cuda.preferred_blas_library(),
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.matmul.fp32_precision,
    )
