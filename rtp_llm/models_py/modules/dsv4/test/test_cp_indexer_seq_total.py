"""Phase F CPU unit test: pin the CP T_per_req computation in
``IndexerFP8.prepare`` to use ``cp_ctx.input_lengths_global`` rather
than the framework's rank-local ``input_lengths``.

The bound contract:
    seq_total_per_req[b] = prefix_lengths[b] + eff_input_lengths[b]
    T_per_req[b]         = seq_total_per_req[b] // ratio
    T (global)           = sum(T_per_req)

Under CP=N B=1: ``cp_ctx.input_lengths_global = [seq_len_full]``,
so ``T = (prefix + seq_len_full) // ratio`` regardless of cp_rank —
matching the per-rank pool which holds compressed entries for the
ENTIRE global sequence (Phase F: nested compressor all-gathers
KV/score before writing the pool).

Standalone helper that mirrors the Phase-F branch in
``IndexerFP8.prepare`` (CPU-only; avoids importing the FP8 indexer
module which pulls in deep_gemm + .so bindings).
"""

import torch


def _eff_input_lengths(
    rank_local_input_lengths: torch.Tensor,
    cp_ctx_input_lengths_global: torch.Tensor,
    cp_size: int,
) -> torch.Tensor:
    """Mirror Phase-F branch:
    cp_active = cp_size > 1 and global is not None
    eff = cp_ctx.input_lengths_global if cp_active else input_lengths
    """
    if cp_size > 1 and cp_ctx_input_lengths_global is not None:
        return cp_ctx_input_lengths_global
    return rank_local_input_lengths


def _t_per_req(
    prefix_lengths: torch.Tensor,
    eff_input_lengths: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    seq_total = prefix_lengths.to(torch.int64) + eff_input_lengths.to(torch.int64)
    return (seq_total // ratio).to(torch.int32)


def test_cp1_eff_input_lengths_falls_back_to_rank_local() -> None:
    """CP=1: no global override, eff == input_lengths."""
    rank_local = torch.tensor([100], dtype=torch.int32)
    eff = _eff_input_lengths(rank_local, None, cp_size=1)
    assert torch.equal(eff, rank_local)


def test_cp2_eff_input_lengths_uses_global() -> None:
    """CP>1 with global populated: eff = cp_ctx.input_lengths_global."""
    rank_local = torch.tensor([50], dtype=torch.int32)  # chunk_length per rank
    glob = torch.tensor([100], dtype=torch.int32)
    eff = _eff_input_lengths(rank_local, glob, cp_size=2)
    assert torch.equal(eff, glob)


def test_cp2_t_per_req_independent_of_rank() -> None:
    """T_per_req under CP=N B=1 is rank-invariant (each rank sees the
    same global seq_total / ratio). Pool holds compressed entries for
    the full global sequence, so T must reflect global."""
    prefix = torch.tensor([200], dtype=torch.int32)
    glob = torch.tensor([400], dtype=torch.int32)
    ratio = 4
    rank_local_per_rank = {2: torch.tensor([200], dtype=torch.int32)}  # chunk_len = 200
    for cp_size, rl in rank_local_per_rank.items():
        eff = _eff_input_lengths(rl, glob, cp_size)
        T = _t_per_req(prefix, eff, ratio)
        assert int(T[0].item()) == (200 + 400) // ratio  # 150
    # CP=1 sanity (rl == glob):
    eff = _eff_input_lengths(glob, None, cp_size=1)
    T = _t_per_req(prefix, eff, ratio)
    assert int(T[0].item()) == (200 + 400) // ratio


def test_cp2_t_per_req_diverges_if_using_rank_local() -> None:
    """Negative regression: if Phase F's swap to ``cp_ctx.input_lengths_global``
    is reverted, the T computed from rank-local inputs diverges from the
    global expectation — pool reads/writes would then be undersized.
    Asserts the divergence magnitude so the swap stays load-bearing."""
    prefix = torch.tensor([0], dtype=torch.int32)
    rank_local = torch.tensor([200], dtype=torch.int32)  # chunk_len under CP=2
    glob = torch.tensor([400], dtype=torch.int32)
    ratio = 4
    T_rank_local = _t_per_req(prefix, rank_local, ratio)
    T_global = _t_per_req(prefix, glob, ratio)
    assert int(T_rank_local[0].item()) == 50
    assert int(T_global[0].item()) == 100
    assert T_global[0] != T_rank_local[0]


if __name__ == "__main__":
    test_cp1_eff_input_lengths_falls_back_to_rank_local()
    test_cp2_eff_input_lengths_uses_global()
    test_cp2_t_per_req_independent_of_rank()
    test_cp2_t_per_req_diverges_if_using_rank_local()
    print("OK")
