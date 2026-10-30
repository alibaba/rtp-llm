"""Indexer CP local scatter/restore kernels and separate gather dispatch."""

import pytest
import torch

from rtp_llm.models_py.triton_kernels.sparse_mla import fused_indexer_cp_assemble as assemble


def test_indexer_uses_two_separate_all_gathers(monkeypatch):
    local_k = torch.arange(12, dtype=torch.uint8).view(3, 4).view(torch.float8_e4m3fn)
    local_s = torch.arange(6, dtype=torch.uint8).view(3, 2)
    calls = []

    def fake_all_gather(payload, group=None, **kwargs):
        calls.append((payload, group, kwargs))
        return payload

    monkeypatch.setattr(assemble, "all_gather", fake_all_gather)
    gathered_k, gathered_s, k_cols, s_cols = assemble._all_gather_indexer_k(
        local_k, local_s, group="tp"
    )
    assert [call[2] for call in calls] == [{}, {}]
    assert gathered_k is local_k
    assert gathered_s is local_s
    assert (k_cols, s_cols) == (4, 2)


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
        k, s, out_k, out_s, inverse, k_width, s_width, padded_rows,
        assemble._next_power_of_2(k_width), assemble._next_power_of_2(s_width),
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
        k, s, out_k, out_s, indices, k_width, s_width, output_rows,
        assemble._next_power_of_2(k_width), assemble._next_power_of_2(s_width),
    )
    assert torch.equal(out_k, k.index_select(0, indices))
    assert torch.equal(out_s, s.index_select(0, indices))
