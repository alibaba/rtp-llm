"""Unit tests for fused_indexer_cp_assemble Triton kernels.

Tests the scatter and gather kernels against the reference PyTorch
implementations in _indexer_cp_assembler.py.
"""

import pytest
import torch

SKIP_REASON = "CUDA required for Triton kernels"


def _has_cuda():
    return torch.cuda.is_available()


def _build_dst_idx(actual_lens, padded_lens, device):
    """Reference dst_idx computation (same as in flashmla_sparse_cp_impl plan())."""
    B = int(actual_lens.numel())
    total_actual = int(actual_lens.sum().item())
    req_ids = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=torch.int64),
        actual_lens,
        output_size=total_actual,
    )
    actual_cu = torch.zeros(B, dtype=torch.int64, device=device)
    padded_cu = torch.zeros(B, dtype=torch.int64, device=device)
    actual_cu[1:] = torch.cumsum(actual_lens[:-1], dim=0)
    padded_cu[1:] = torch.cumsum(padded_lens[:-1], dim=0)
    in_req_pos = torch.arange(
        total_actual, device=device, dtype=torch.int64
    ) - actual_cu.index_select(0, req_ids)
    return padded_cu.index_select(0, req_ids) + in_req_pos


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

    local_q = torch.arange(12, dtype=torch.uint8).view(3, 4).view(
        torch.float8_e4m3fn
    )
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
class TestScatterToPadded:
    """Test _scatter_to_padded_kernel correctness."""

    def test_basic_scatter(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_indexer_cp_assemble import (
            _next_power_of_2,
            _scatter_to_padded_kernel,
        )

        device = torch.device("cuda")
        D = 128
        actual_lens = torch.tensor([3, 2], dtype=torch.int64, device=device)
        padded_lens = torch.tensor([4, 4], dtype=torch.int64, device=device)
        total_actual = 5
        total_padded = 8

        dst_idx = _build_dst_idx(actual_lens, padded_lens, device)

        actual_data = torch.randn(total_actual, D, device=device).to(torch.uint8)
        padded_ref = torch.zeros(total_padded, D, dtype=torch.uint8, device=device)
        padded_ref.index_copy_(0, dst_idx, actual_data)

        padded_triton = torch.zeros(total_padded, D, dtype=torch.uint8, device=device)
        BLOCK_D = _next_power_of_2(D)
        _scatter_to_padded_kernel[(total_actual,)](
            actual_data, padded_triton, dst_idx, D, total_actual, BLOCK_D
        )

        assert torch.equal(padded_triton, padded_ref)

    def test_scatter_small_dim(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_indexer_cp_assemble import (
            _next_power_of_2,
            _scatter_to_padded_kernel,
        )

        device = torch.device("cuda")
        D = 4  # scale dimension
        actual_lens = torch.tensor([5, 3, 4], dtype=torch.int64, device=device)
        padded_lens = torch.tensor([8, 8, 8], dtype=torch.int64, device=device)
        total_actual = 12
        total_padded = 24

        dst_idx = _build_dst_idx(actual_lens, padded_lens, device)

        actual_data = torch.randint(
            0, 255, (total_actual, D), dtype=torch.uint8, device=device
        )
        padded_ref = torch.zeros(total_padded, D, dtype=torch.uint8, device=device)
        padded_ref.index_copy_(0, dst_idx, actual_data)

        padded_triton = torch.zeros(total_padded, D, dtype=torch.uint8, device=device)
        BLOCK_D = _next_power_of_2(D)
        _scatter_to_padded_kernel[(total_actual,)](
            actual_data, padded_triton, dst_idx, D, total_actual, BLOCK_D
        )

        assert torch.equal(padded_triton, padded_ref)


@pytest.mark.skipif(not _has_cuda(), reason=SKIP_REASON)
class TestGatherRestore:
    """Test _gather_restore_kernel correctness."""

    def test_basic_gather(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_indexer_cp_assemble import (
            _gather_restore_kernel,
            _next_power_of_2,
        )

        device = torch.device("cuda")
        D = 128
        gathered_rows = 32
        chunk_T = 20

        gathered = torch.randn(gathered_rows, D, device=device).to(torch.uint8)
        restore_indices = torch.randint(
            0, gathered_rows, (chunk_T,), dtype=torch.int64, device=device
        )

        out_ref = gathered[restore_indices].contiguous()

        out_triton = torch.empty(chunk_T, D, dtype=torch.uint8, device=device)
        BLOCK_D = _next_power_of_2(D)
        _gather_restore_kernel[(chunk_T,)](
            gathered, out_triton, restore_indices, D, chunk_T, BLOCK_D
        )

        assert torch.equal(out_triton, out_ref)

    def test_gather_small_dim(self):
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_indexer_cp_assemble import (
            _gather_restore_kernel,
            _next_power_of_2,
        )

        device = torch.device("cuda")
        D = 4
        gathered_rows = 64
        chunk_T = 48

        gathered = torch.randint(
            0, 255, (gathered_rows, D), dtype=torch.uint8, device=device
        )
        restore_indices = torch.randint(
            0, gathered_rows, (chunk_T,), dtype=torch.int64, device=device
        )

        out_ref = gathered[restore_indices].contiguous()

        out_triton = torch.empty(chunk_T, D, dtype=torch.uint8, device=device)
        BLOCK_D = _next_power_of_2(D)
        _gather_restore_kernel[(chunk_T,)](
            gathered, out_triton, restore_indices, D, chunk_T, BLOCK_D
        )

        assert torch.equal(out_triton, out_ref)


@pytest.mark.skipif(not _has_cuda(), reason=SKIP_REASON)
class TestEndToEndAgainstReference:
    """End-to-end test comparing Triton fused path against asm reference."""

    def _make_plan(self, actual_lens, padded_lens, cp_size, device):
        """Create a minimal plan-like object for testing."""
        from types import SimpleNamespace

        from rtp_llm.models_py.modules.dsv4.cp import cp_padded_local_kv_lens
        from rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_assembler import (
            build_kv_allgather_restore_indices,
        )

        total_actual = int(actual_lens.sum().item())
        total_local = int(padded_lens.sum().item())
        per_req_total = actual_lens * cp_size  # approximate total lens

        restore_indices = torch.arange(total_local, dtype=torch.int64, device=device)

        return SimpleNamespace(
            per_req_actual_local_kv_lens=actual_lens,
            per_req_local_kv_lens=padded_lens,
            total_local_T=total_local,
            total_actual_local_T=total_actual,
            restore_indices=restore_indices,
            cp_ctx=SimpleNamespace(cp_size=cp_size),
        )

    def test_scatter_matches_asm_copy_actual_to_padded(self):
        """Verify scatter kernel produces same result as asm.copy_actual_indexer_k_to_padded."""
        from rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_assembler import (
            copy_actual_indexer_k_to_padded,
        )
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_indexer_cp_assemble import (
            _next_power_of_2,
            _scatter_to_padded_kernel,
        )

        device = torch.device("cuda")
        head_dim = 128
        scale_dim = 4
        actual_lens = torch.tensor([10, 7, 12, 5], dtype=torch.int64, device=device)
        padded_lens = torch.tensor([16, 8, 16, 8], dtype=torch.int64, device=device)
        total_actual = int(actual_lens.sum().item())
        total_padded = int(padded_lens.sum().item())

        plan = self._make_plan(actual_lens, padded_lens, cp_size=4, device=device)

        actual_k = torch.randint(
            0, 255, (total_actual, head_dim), dtype=torch.uint8, device=device
        ).view(torch.float8_e4m3fn)
        actual_s = torch.randint(
            0, 255, (total_actual, scale_dim), dtype=torch.uint8, device=device
        )

        # Reference: asm
        ref_k = torch.zeros(
            total_padded, head_dim, dtype=torch.float8_e4m3fn, device=device
        )
        ref_s = torch.zeros(total_padded, scale_dim, dtype=torch.uint8, device=device)
        copy_actual_indexer_k_to_padded(
            plan=plan,
            actual_k_quant=actual_k,
            actual_k_scale=actual_s,
            padded_k_quant=ref_k,
            padded_k_scale=ref_s,
        )

        # Triton scatter
        dst_idx = _build_dst_idx(actual_lens, padded_lens, device)
        triton_k = torch.zeros(
            total_padded, head_dim, dtype=torch.float8_e4m3fn, device=device
        )
        triton_s = torch.zeros(
            total_padded, scale_dim, dtype=torch.uint8, device=device
        )

        BLOCK_K = _next_power_of_2(head_dim)
        _scatter_to_padded_kernel[(total_actual,)](
            actual_k.view(torch.uint8),
            triton_k.view(torch.uint8),
            dst_idx,
            head_dim,
            total_actual,
            BLOCK_K,
        )
        BLOCK_S = _next_power_of_2(scale_dim)
        _scatter_to_padded_kernel[(total_actual,)](
            actual_s,
            triton_s,
            dst_idx,
            scale_dim,
            total_actual,
            BLOCK_S,
        )

        assert torch.equal(triton_k.view(torch.uint8), ref_k.view(torch.uint8))
        assert torch.equal(triton_s, ref_s)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
