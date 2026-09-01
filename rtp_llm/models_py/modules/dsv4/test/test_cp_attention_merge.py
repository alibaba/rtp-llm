"""CPU UTs for CP-sharded attention LSE/O merge.

These tests validate the math needed to avoid gathering cached KV during
cache-hit prefill: local attention over disjoint KV shards can be merged into
the exact full attention result with only per-query LSE and output exchange.
"""

import importlib.util
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _import_merge_module():
    name = "_dsv4_cp_attention_merge_for_test"
    spec = importlib.util.spec_from_file_location(
        name,
        _REPO_ROOT / "rtp_llm/models_py/modules/dsv4/fp8/_cp_attention_merge.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MERGE = _import_merge_module()


def _dense_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference attention returning (O, LSE).

    q: [T, H, D], k/v: [N, H, D], mask: optional [T, N] boolean.
    """
    T, H, D = q.shape
    N = k.shape[0]
    if N == 0:
        return (
            torch.zeros((T, H, D), dtype=q.dtype, device=q.device),
            torch.full((T, H), float("-inf"), dtype=torch.float32, device=q.device),
        )

    logits = torch.einsum("thd,nhd->thn", q.float(), k.float()) * (D**-0.5)
    if mask is not None:
        logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
    out = torch.einsum("thn,nhd->thd", probs, v.float()).to(q.dtype)
    return out, lse


def _merge_split_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    shards: list[torch.Tensor],
    mask: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    outs = []
    lses = []
    for idx in shards:
        local_mask = mask.index_select(1, idx) if mask is not None else None
        out, lse = _dense_attn(
            q, k.index_select(0, idx), v.index_select(0, idx), local_mask
        )
        if out_dtype is not None:
            out = out.to(out_dtype)
        outs.append(out)
        lses.append(lse)
    return MERGE.merge_lse_output(torch.stack(outs, dim=0), torch.stack(lses, dim=0))


def _rr_block_shards(n_keys: int, cp_size: int, block_size: int) -> list[torch.Tensor]:
    shards: list[list[int]] = [[] for _ in range(cp_size)]
    for token in range(n_keys):
        block = token // block_size
        shards[block % cp_size].append(token)
    return [torch.tensor(s, dtype=torch.long) for s in shards]


def test_two_shards_matches_full_dense_attention() -> None:
    torch.manual_seed(11)
    q = torch.randn(7, 3, 16)
    k = torch.randn(13, 3, 16)
    v = torch.randn(13, 3, 16)
    full_out, full_lse = _dense_attn(q, k, v)
    merged_out, merged_lse = _merge_split_attention(
        q,
        k,
        v,
        [
            torch.tensor([0, 2, 4, 6, 8, 10, 12]),
            torch.tensor([1, 3, 5, 7, 9, 11]),
        ],
    )
    torch.testing.assert_close(merged_lse, full_lse, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(merged_out, full_out, rtol=1e-5, atol=1e-5)


def test_uneven_and_empty_shards_match_full_attention() -> None:
    torch.manual_seed(12)
    q = torch.randn(5, 2, 8)
    k = torch.randn(4, 2, 8)
    v = torch.randn(4, 2, 8)
    shards = [
        torch.tensor([0, 3]),
        torch.tensor([], dtype=torch.long),
        torch.tensor([1]),
        torch.tensor([2]),
    ]
    full_out, full_lse = _dense_attn(q, k, v)
    merged_out, merged_lse = _merge_split_attention(q, k, v, shards)
    torch.testing.assert_close(merged_lse, full_lse, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(merged_out, full_out, rtol=1e-5, atol=1e-5)


def test_all_empty_rows_produce_zero_output_and_negative_inf_lse() -> None:
    local_outs = torch.randn(3, 4, 2, 8)
    local_lse = torch.full((3, 4, 2), float("-inf"))
    merged_out, merged_lse = MERGE.merge_lse_output(local_outs, local_lse)
    assert torch.equal(merged_out, torch.zeros_like(merged_out))
    assert torch.equal(merged_lse, torch.full_like(merged_lse, float("-inf")))


def test_bf16_local_outputs_keep_bf16_with_fp32_lse() -> None:
    torch.manual_seed(13)
    q = torch.randn(6, 4, 16)
    k = torch.randn(17, 4, 16)
    v = torch.randn(17, 4, 16)
    shards = _rr_block_shards(n_keys=17, cp_size=3, block_size=4)
    full_out, full_lse = _dense_attn(q, k, v)
    merged_out, merged_lse = _merge_split_attention(
        q, k, v, shards, out_dtype=torch.bfloat16
    )
    assert merged_out.dtype == torch.bfloat16
    assert merged_lse.dtype == torch.float32
    torch.testing.assert_close(merged_lse, full_lse, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(merged_out.float(), full_out, rtol=4e-3, atol=4e-3)


def test_prefix_plus_input_causal_mask_with_rr_blocks_matches_full_attention() -> None:
    torch.manual_seed(14)
    prefix_len = 19
    input_len = 7
    n_keys = prefix_len + input_len
    q = torch.randn(input_len, 3, 16)
    k = torch.randn(n_keys, 3, 16)
    v = torch.randn(n_keys, 3, 16)

    key_pos = torch.arange(n_keys)
    query_pos = torch.arange(prefix_len, prefix_len + input_len)
    causal_mask = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
    shards = _rr_block_shards(n_keys=n_keys, cp_size=4, block_size=5)

    full_out, full_lse = _dense_attn(q, k, v, causal_mask)
    merged_out, merged_lse = _merge_split_attention(q, k, v, shards, causal_mask)
    torch.testing.assert_close(merged_lse, full_lse, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(merged_out, full_out, rtol=1e-5, atol=1e-5)


def test_varlen_batch_compressed_rr_and_replicated_swa_logical_shards_match_full_attention() -> (
    None
):
    """Attention-module contract for the raw-Q/O/LSE merge path.

    The compressed region is physically page-RR sharded. SWA_KV is physically
    replicated, but the optimized attention path must still partition SWA
    keys logically before merging O/LSE, otherwise replicated SWA keys would
    be counted once per CP rank. This test builds a B>1 varlen request batch
    and validates that the disjoint rank-local key sets exactly recover full
    attention.
    """
    torch.manual_seed(15)
    cp_size = 4
    block_size = 2
    compress_ratio = 4
    window_size = 4
    prefix_lengths = torch.tensor([7, 3], dtype=torch.long)
    input_lengths = torch.tensor([3, 2], dtype=torch.long)
    total_lens = prefix_lengths + input_lengths

    req_ids = []
    query_pos = []
    for req, (prefix, input_len) in enumerate(zip(prefix_lengths, input_lengths)):
        req_ids.extend([req] * int(input_len.item()))
        query_pos.extend(
            range(int(prefix.item()), int(prefix.item()) + int(input_len.item()))
        )
    req_id_per_token = torch.tensor(req_ids, dtype=torch.long)
    query_pos_t = torch.tensor(query_pos, dtype=torch.long)

    keys: list[tuple[str, int, int]] = []
    for req, total_len in enumerate(total_lens.tolist()):
        for cmp_idx in range(total_len // compress_ratio):
            keys.append(("cmp", req, cmp_idx))
        for pos in range(total_len):
            keys.append(("swa", req, pos))

    T = int(query_pos_t.numel())
    H = 3
    D = 8
    q = torch.randn(T, H, D)
    k = torch.randn(len(keys), H, D)
    v = torch.randn(len(keys), H, D)

    full_mask = torch.zeros((T, len(keys)), dtype=torch.bool)
    rank_masks = [torch.zeros_like(full_mask) for _ in range(cp_size)]
    for row in range(T):
        req = int(req_id_per_token[row].item())
        qpos = int(query_pos_t[row].item())
        max_cmp = min(
            (qpos + 1) // compress_ratio, int(total_lens[req].item()) // compress_ratio
        )
        swa_start = max(0, qpos - window_size + 1)
        for col, (kind, key_req, key_pos) in enumerate(keys):
            if key_req != req:
                continue
            if kind == "cmp":
                visible = key_pos < max_cmp
                owner = (key_pos // block_size) % cp_size
            else:
                visible = swa_start <= key_pos <= qpos
                # SWA is replicated, so owner is a logical compute partition,
                # not a storage owner.
                owner = key_pos % cp_size
            if visible:
                full_mask[row, col] = True
                rank_masks[owner][row, col] = True

    union = torch.zeros_like(full_mask)
    overlap = torch.zeros_like(full_mask)
    for mask in rank_masks:
        overlap |= union & mask
        union |= mask
    assert torch.equal(union, full_mask)
    assert not overlap.any()

    full_out, full_lse = _dense_attn(q, k, v, full_mask)
    outs = []
    lses = []
    for mask in rank_masks:
        out, lse = _dense_attn(q, k, v, mask)
        outs.append(out)
        lses.append(lse)
    merged_out, merged_lse = MERGE.merge_lse_output(
        torch.stack(outs, dim=0), torch.stack(lses, dim=0)
    )
    torch.testing.assert_close(merged_lse, full_lse, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(merged_out, full_out, rtol=1e-5, atol=1e-5)


def test_per_shard_topk_is_not_global_topk_equivalent() -> None:
    """Documents the CSA/indexer hazard.

    If each CP rank independently keeps its local top1, merging the resulting
    O/LSE attends two keys here, while global top1 should attend only key 0.
    Runtime use of merge_lse_output must therefore only happen after the
    candidate key set has already been globally partitioned.
    """
    q = torch.ones(1, 1, 1)
    k = torch.tensor([[[10.0]], [[9.0]]])
    v = torch.tensor([[[0.0]], [[10.0]]])

    global_top1_mask = torch.tensor([[True, False]])
    full_top1_out, _ = _dense_attn(q, k, v, global_top1_mask)
    merged_local_top1_out, _ = _merge_split_attention(
        q,
        k,
        v,
        [torch.tensor([0]), torch.tensor([1])],
    )

    assert float((merged_local_top1_out - full_top1_out).abs().max().item()) > 1.0


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
