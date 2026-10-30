import pytest
import torch

from rtp_llm.models_py.triton_kernels.sparse_mla import (
    fused_indexer_cp_assemble as assemble,
)

SKIP_REASON = "CUDA required"


def _has_cuda():
    return torch.cuda.is_available()


def test_master_switch_on_uses_one_packed_all_gather(monkeypatch):
    from rtp_llm.models_py.triton_kernels.sparse_mla import (
        fused_indexer_cp_assemble as assemble,
    )

    q_bytes = torch.arange(12, dtype=torch.uint8).view(3, 4)
    local_q = q_bytes.view(torch.float8_e4m3fn)
    local_s = torch.tensor(
        [[101, 102], [103, 104], [105, 106]],
        dtype=torch.uint8,
    )
    calls = []

    def fake_all_gather(payload, group=None, **kwargs):
        calls.append((payload.clone(), group, kwargs))
        return payload

    monkeypatch.setattr(assemble, "cp_opt_enabled", lambda: True)
    monkeypatch.setattr(assemble, "all_gather", fake_all_gather)
    gathered, gathered_s, q_cols, s_cols = assemble._all_gather_indexer_k(
        local_q, local_s, group="tp"
    )

    assert len(calls) == 1
    assert calls[0][2]["role"] == "indexer_k_packed"
    assert gathered_s is None
    assert tuple(gathered.shape) == (3, 6)
    assert (q_cols, s_cols) == (4, 2)
    assert torch.equal(gathered[:, :q_cols], q_bytes)
    assert torch.equal(gathered[:, q_cols:], local_s)


def test_master_switch_off_uses_two_separate_all_gathers(monkeypatch):
    from rtp_llm.models_py.triton_kernels.sparse_mla import (
        fused_indexer_cp_assemble as assemble,
    )

    local_q = torch.arange(12, dtype=torch.uint8).view(3, 4).view(torch.float8_e4m3fn)
    local_s = torch.arange(6, dtype=torch.uint8).view(3, 2)
    calls = []

    def fake_all_gather(payload, group=None, **kwargs):
        calls.append((payload, group, kwargs))
        return payload

    monkeypatch.setattr(assemble, "cp_opt_enabled", lambda: False)
    monkeypatch.setattr(assemble, "all_gather", fake_all_gather)
    gathered_q, gathered_s, q_cols, s_cols = assemble._all_gather_indexer_k(
        local_q, local_s, group="tp"
    )

    assert [call[2]["role"] for call in calls] == [
        "indexer_k_quant",
        "indexer_k_scale",
    ]
    assert gathered_q is local_q
    assert gathered_s is local_s
    assert (q_cols, s_cols) == (4, 2)


@pytest.mark.skipif(not _has_cuda(), reason=SKIP_REASON)
@pytest.mark.parametrize(
    "actual_lens,padded_lens",
    [
        ([0], [0]),
        ([0], [5]),
        ([7], [7]),
        ([3], [8]),
        ([0, 3, 0, 2], [4, 4, 4, 4]),
        ([513, 1, 0], [768, 256, 256]),
    ],
)
@pytest.mark.parametrize("width,scale_width", [(128, 4), (256, 8), (17, 3)])
def test_fused_padded_wire_bytes(actual_lens, padded_lens, width, scale_width):
    from rtp_llm.models_py.triton_kernels.sparse_mla.fused_indexer_cp_assemble import (
        _pack_padded_k_scale,
    )

    n, rows = sum(actual_lens), sum(padded_lens)
    # Arbitrary bytes cover FP8 NaNs too: no numeric conversion is allowed.
    k = torch.randint(0, 256, (n, width), dtype=torch.uint8, device="cuda")
    scales = torch.randint(0, 256, (n, scale_width), dtype=torch.uint8, device="cuda")
    mapping = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
    ref = torch.zeros((rows, width + scale_width), dtype=torch.uint8, device="cuda")
    src = dst = 0
    for actual, padded in zip(actual_lens, padded_lens):
        mapping[dst : dst + actual] = torch.arange(src, src + actual, device="cuda")
        ref[dst : dst + actual, :width] = k[src : src + actual]
        ref[dst : dst + actual, width:] = scales[src : src + actual]
        src += actual
        dst += padded
    for repeat in range(3):
        result = _pack_padded_k_scale(
            k.view(torch.float8_e4m3fn), scales, rows, mapping
        )
        assert torch.equal(result, ref)
    if n == rows or n == 0:
        result = _pack_padded_k_scale(k, scales, rows)
        assert torch.equal(result, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("k_width,s_width", [(128, 4), (65, 3)])
def test_zero_scatter_exact(k_width, s_width):
    actual_rows, padded_rows = 17, 24
    k = torch.randint(0, 256, (actual_rows, k_width), device="cuda", dtype=torch.uint8)
    s = torch.randint(0, 256, (actual_rows, s_width), device="cuda", dtype=torch.uint8)
    dst = torch.randperm(padded_rows, device="cuda")[:actual_rows]
    inverse = torch.full((padded_rows,), -1, device="cuda", dtype=torch.int64)
    inverse[dst] = torch.arange(actual_rows, device="cuda")
    out_k = torch.empty((padded_rows, k_width), device="cuda", dtype=torch.uint8)
    out_s = torch.empty((padded_rows, s_width), device="cuda", dtype=torch.uint8)
    assemble._fused_zero_scatter_kernel[(padded_rows,)](
        k,
        s,
        out_k,
        out_s,
        inverse,
        k_width,
        s_width,
        padded_rows,
        assemble._next_power_of_2(k_width),
        assemble._next_power_of_2(s_width),
    )
    ref_k = torch.zeros_like(out_k)
    ref_s = torch.zeros_like(out_s)
    ref_k.index_copy_(0, dst, k)
    ref_s.index_copy_(0, dst, s)
    assert torch.equal(out_k, ref_k)
    assert torch.equal(out_s, ref_s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("k_width,s_width", [(128, 4), (65, 3)])
def test_separate_restore_exact(k_width, s_width):
    rows, output_rows = 37, 23
    k = torch.randint(0, 256, (rows, k_width), device="cuda", dtype=torch.uint8)
    s = torch.randint(0, 256, (rows, s_width), device="cuda", dtype=torch.uint8)
    indices = torch.randperm(rows, device="cuda")[:output_rows].long()
    out_k = torch.empty((output_rows, k_width), device="cuda", dtype=torch.uint8)
    out_s = torch.empty((output_rows, s_width), device="cuda", dtype=torch.uint8)
    assemble._gather_restore_separate_kernel[(output_rows,)](
        k,
        s,
        out_k,
        out_s,
        indices,
        k_width,
        s_width,
        output_rows,
        assemble._next_power_of_2(k_width),
        assemble._next_power_of_2(s_width),
    )
    assert torch.equal(out_k, k.index_select(0, indices))
    assert torch.equal(out_s, s.index_select(0, indices))
