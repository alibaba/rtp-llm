import json

import pytest
import torch

from rtp_llm.models_py.modules.kimi_k3.native_kda import (
    flash_kda_paged_prefill,
    vllm_kda_paged_prefill,
    plan_state_sequences,
    StateSegment,
    plan_cula_checkpoint_groups,
)
from rtp_llm.models_py.modules.kimi_k3 import native_kda as kda_module


def test_state_sequence_plans():
    # Exercise the actual CPU plan, including short, ragged and padded request batches.
    for batch in (1, 2, 3, 7, 8, 9):
        lengths = [129 + i for i in range(batch)]
        cu = [0]
        for n in lengths:
            cu.append(cu[-1] + n)
        plans = plan_state_sequences(
            cu, [0] * batch, [[2 * i + 1, 2 * i + 2] for i in range(batch)], 128
        )
        assert len(plans) == batch
        for i, p in enumerate(plans):
            assert [(s.start, s.end) for s in p.segments] == [
                (cu[i], cu[i] + 128),
                (cu[i] + 128, cu[i + 1]),
            ]
    assert not plan_state_sequences([0, 17], [0], [[0]], 128)[0].segments
    for blocks in ([-1, 2], [-1, -1]):
        plan = plan_state_sequences([0, 129], [0], [blocks], 128)[0]
        assert plan.initial_block is None
        assert [(s.start, s.end, s.cache_block) for s in plan.segments] == [
            (0, 128, blocks[0]),
            (128, 129, blocks[1]),
        ]
    for args in [
        ([0, 2], [-1], [[1]], 128),
        ([0, 257], [0], [[1]], 128),
        ([0, 129], [0], [[1, 0]], 128),
        ([0, 1], [128], [[0, 2]], 128),
        ([0, 1], [128], [[-1, 2]], 128),
        ([0, 129], [0], [[-1, 0]], 128),
        ([0, 129], [0], [[-2, 2]], 128),
    ]:
        try:
            plan_state_sequences(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid state plan accepted")


def test_cula_checkpoint_groups_bound_per_call_workspace():
    segments = tuple(
        StateSegment(index * 4096, (index + 1) * 4096, index + 1)
        for index in range(9)
    )
    groups = plan_cula_checkpoint_groups(segments, 4096)
    assert [len(group) for group in groups] == [4, 4, 1]
    assert sum(groups, ()) == segments

    partial = (StateSegment(0, 3968, 1),) + tuple(
        StateSegment(3968 + index * 4096, 3968 + (index + 1) * 4096, index + 2)
        for index in range(7)
    )
    groups = plan_cula_checkpoint_groups(partial, 4096)
    assert [len(group) for group in groups] == [1, 4, 3]
    assert sum(groups, ()) == partial


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("retain_first_checkpoint", [True, False])
@torch.inference_mode()
def test_cula_paged_prefill_matches_independent_vllm_backend(retain_first_checkpoint):
    pytest.importorskip("cula.kda")
    assert hasattr(kda_module, "cula_kda_paged_prefill")
    torch.manual_seed(923)
    heads, dim = 2, 128
    lengths = [4167, 193, 32]
    cu_seqlens = [0, 4167, 4360, 4392]
    projections = [
        torch.randn(cu_seqlens[-1], heads, dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    beta = torch.randn(cu_seqlens[-1], heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda") * 0.1
    dt_bias = torch.randn(heads, dim, device="cuda") * 0.01
    storage = torch.randn(10, heads * dim * dim + 512, device="cuda")
    reference_storage = storage.clone()
    actual_cache = storage[:, : heads * dim * dim].view(10, heads, dim, dim)
    reference_cache = reference_storage[:, : heads * dim * dim].view(10, heads, dim, dim)
    tables = [[1, 2], [5, 6], [0, 0]]
    if not retain_first_checkpoint:
        tables[0][0] = -1
    arguments = (*projections, beta, a_log, dt_bias, -5.0)
    reference = vllm_kda_paged_prefill(
        *arguments, reference_cache, cu_seqlens, [0, 4096, 0], tables, 4096
    )
    actual = kda_module.cula_kda_paged_prefill(
        *arguments, actual_cache, cu_seqlens, [0, 4096, 0], tables, 4096
    )
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=5e-4)
    torch.testing.assert_close(storage, reference_storage, rtol=1e-3, atol=5e-4)
    assert torch.count_nonzero(actual[cu_seqlens[2] :]).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_cula_paged_prefill_finishes_partial_prefix_block():
    pytest.importorskip("cula.kda")
    torch.manual_seed(924)
    tokens, heads, dim = 4167, 2, 128
    projections = [
        torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    beta = torch.randn(tokens, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda") * 0.1
    dt_bias = torch.randn(heads, dim, device="cuda") * 0.01
    storage = torch.randn(3, heads * dim * dim + 512, device="cuda")
    reference_storage = storage.clone()
    actual_cache = storage[:, : heads * dim * dim].view(3, heads, dim, dim)
    reference_cache = reference_storage[:, : heads * dim * dim].view(3, heads, dim, dim)
    arguments = (*projections, beta, a_log, dt_bias, -5.0)
    metadata = ([0, tokens], [128], [[1, 2]], 4096)
    reference = vllm_kda_paged_prefill(*arguments, reference_cache, *metadata)
    actual = kda_module.cula_kda_paged_prefill(*arguments, actual_cache, *metadata)
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=5e-4)
    torch.testing.assert_close(storage, reference_storage, rtol=1e-3, atol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_cula_paged_prefill_multiple_bounded_groups_match_vllm():
    pytest.importorskip("cula.kda")
    torch.manual_seed(925)
    tokens, heads, dim = 9 * 4096 + 71, 2, 128
    projections = [
        torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    beta = torch.randn(tokens, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda") * 0.1
    dt_bias = torch.randn(heads, dim, device="cuda") * 0.01
    storage = torch.randn(12, heads * dim * dim + 512, device="cuda")
    reference_storage = storage.clone()
    actual_cache = storage[:, : heads * dim * dim].view(12, heads, dim, dim)
    reference_cache = reference_storage[:, : heads * dim * dim].view(
        12, heads, dim, dim
    )
    arguments = (*projections, beta, a_log, dt_bias, -5.0)
    metadata = ([0, tokens], [0], [list(range(1, 11))], 4096)
    reference = vllm_kda_paged_prefill(*arguments, reference_cache, *metadata)
    actual = kda_module.cula_kda_paged_prefill(
        *arguments, actual_cache, *metadata
    )
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=5e-4)
    torch.testing.assert_close(storage, reference_storage, rtol=1e-3, atol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("backend", ["flashkda", "vllm_triton"])
@pytest.mark.parametrize("checkpoint_mode", ["full", "final", "none"])
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("tail_length", [1, 193])
def test_native_kda_paged_state_layout(backend, checkpoint_mode, heads, tail_length):
    flash_kda = pytest.importorskip("flash_kda") if backend == "flashkda" else None
    torch.manual_seed(922)
    lengths = [4167, tail_length, 32]
    cu = [0, 4167, 4167 + tail_length, 4199 + tail_length]
    dim = 128
    xs = [
        torch.randn(cu[-1], heads, dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    beta = torch.randn(cu[-1], heads, device="cuda", dtype=torch.bfloat16)
    alog = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.01
    # Non-contiguous outer stride matches RTP's combined conv/recurrent storage.
    storage = torch.randn(
        10, heads * dim * dim + 512, device="cuda", dtype=torch.float32
    )
    cache = storage[:, : heads * dim * dim].view(10, heads, dim, dim)
    before = storage.clone()
    tables = [[1, 2], [5, 6], [0, 0]]
    if checkpoint_mode != "full":
        tables[0][0] = -1
    if checkpoint_mode == "none":
        tables[0][1] = tables[1][1] = -1
    input_before = [x.clone() for x in [*xs, beta]]
    paged = flash_kda_paged_prefill if backend == "flashkda" else vllm_kda_paged_prefill
    out = paged(
        *xs, beta, alog, bias, -5.0, cache, cu, [0, 4096, 0], tables, 4096
    )

    def native(request, end):
        start = cu[request]
        end = start + end
        n = end - start
        q, k, v, g = [x[start:end].unsqueeze(0).contiguous() for x in xs]
        initial = (
            torch.zeros(1, heads, dim, dim, device="cuda")
            if request == 0
            else before[5, : heads * dim * dim]
            .view(heads, dim, dim)
            .transpose(-1, -2)
            .unsqueeze(0)
            .contiguous()
        )
        if backend == "vllm_triton":
            from rtp_llm.models_py.modules.kimi_k3.vllm_kda.kda.chunk import chunk_kda_with_fused_gate

            output, final = chunk_kda_with_fused_gate(
                q, k, v.clone(), g,
                raw_beta=beta[start:end].unsqueeze(0).contiguous(),
                A_log=alog, g_bias=bias.flatten(), initial_state=initial,
                output_final_state=True, lower_bound=-5.0,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=torch.tensor([0, n], device="cuda", dtype=torch.int32),
            )
            return output[0], final[0].transpose(-1, -2)
        final = torch.empty_like(initial)
        output = torch.empty_like(v)
        workspace = torch.empty(
            flash_kda.get_workspace_size(n, heads, 1), dtype=torch.uint8, device="cuda"
        )
        torch.ops.flash_kda.fwd(
            q,
            k,
            v,
            g,
            beta[start:end].unsqueeze(0).clone(memory_format=torch.contiguous_format),
            dim**-0.5,
            output,
            workspace,
            alog,
            bias,
            -5.0,
            initial,
            final,
        )
        return output[0], final[0].transpose(-1, -2)

    for actual, original in zip([*xs, beta], input_before):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)

    for request in (0, 1):
        expected, state = native(request, lengths[request])
        torch.testing.assert_close(
            out[cu[request] : cu[request + 1]], expected, rtol=0, atol=0
        )
        if checkpoint_mode != "none":
            torch.testing.assert_close(cache[[2, 6][request]], state, rtol=0, atol=0)
    _, boundary_state = native(0, 4096)
    if checkpoint_mode == "full":
        torch.testing.assert_close(cache[1], boundary_state, rtol=0, atol=0)
    assert torch.count_nonzero(out[cu[2] :]).item() == 0
    written = (
        {1, 2, 6}
        if checkpoint_mode == "full"
        else {2, 6} if checkpoint_mode == "final" else set()
    )
    for block in set(range(10)) - written:
        torch.testing.assert_close(storage[block], before[block], rtol=0, atol=0)
    torch.testing.assert_close(
        storage[:, heads * dim * dim :], before[:, heads * dim * dim :], rtol=0, atol=0
    )
    print(
        json.dumps(
            {
                "passed": True,
                "plan_batch_sizes": [1, 2, 3, 7, 8, 9],
                "real_requests": 2,
                "virtual_requests": 1,
                "block_size": 4096,
                "prefix_lengths": [0, 4096, 0],
                "ragged_lengths": lengths,
                "outputs_and_states": "bitwise identical to independent native full-request calls",
                "unused_cache_and_conv_storage": "unchanged",
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("advertised", [None, False])
def test_rejects_storage_only_fp32_backend(monkeypatch, advertised):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "flash_kda", SimpleNamespace())
    namespace = SimpleNamespace()
    if advertised is not None:
        namespace.supports_fp32_recurrence = lambda: advertised
    monkeypatch.setattr(torch.ops, "flash_kda", namespace)
    with pytest.raises(RuntimeError, match="FP32 recurrent accumulation"):
        flash_kda_paged_prefill(*([None] * 13))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp32_state_survives_zero_update():
    flash_kda = pytest.importorskip("flash_kda")
    assert torch.ops.flash_kda.supports_fp32_recurrence()
    # Zero keys/values and an underflowed gate produce exactly no state update.
    # A storage-only FP32 implementation fails by rounding the initial state.
    lengths = [17, 129]
    heads, dim = 2, 128
    total = sum(lengths)
    qkv = torch.zeros(1, total, heads, dim, device="cuda", dtype=torch.bfloat16)
    gate = torch.full_like(qkv, -100)
    beta = torch.zeros(1, total, heads, device="cuda", dtype=torch.bfloat16)
    initial = torch.full(
        (2, heads, dim, dim), 0.1234567, device="cuda", dtype=torch.float32
    )
    final, checkpoint = torch.empty_like(initial), torch.empty_like(initial)
    output = torch.empty_like(qkv)
    workspace = torch.empty(
        flash_kda.get_workspace_size(total, heads, 2), device="cuda", dtype=torch.uint8
    )
    torch.ops.flash_kda.fwd(
        qkv,
        qkv,
        qkv,
        gate,
        beta,
        dim**-0.5,
        output,
        workspace,
        torch.zeros(heads, device="cuda"),
        torch.zeros(heads, dim, device="cuda"),
        -5.0,
        initial,
        final,
        torch.tensor([0, 17, total], device="cuda", dtype=torch.int64),
        checkpoint,
        torch.tensor([16, 128], device="cuda", dtype=torch.int64),
    )
    torch.testing.assert_close(final, initial, rtol=0, atol=0)
    torch.testing.assert_close(checkpoint, initial, rtol=0, atol=0)
    assert torch.count_nonzero(output).item() == 0
