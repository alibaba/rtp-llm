"""Phase 2B-1 sanity: HCA-K + INDEXER-K paged write slot wiring.

Validates that ``build_decode_metadata`` (eager) computes the right
boundary-only slot mappings for HCA_KV (ratio=128) and INDEXER_KV
(ratio=4), and that ``write_kv_to_pool(..., mask_negative=True)``
honors the ``-1`` sentinel for non-boundary tokens.

Bypasses package init (loads modules off disk) — same trick as the
Phase 2A test.
"""

from __future__ import annotations

import importlib.util
import sys

import torch


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_BASE = (
    "/home/serina.wzq/RTP-LLM/github-opensource/rtp_llm/models_py/modules/dsv4/decode"
)
pl = _load(f"{_BASE}/pool_layout.py", "_pl_p2b")
psm = _load(f"{_BASE}/pool_slot_mapping.py", "_psm_p2b")
kw = _load(f"{_BASE}/kv_write_decode_op.py", "_kw_p2b")


def _shim(name, target):
    sys.modules[name] = target


_shim("rtp_llm", type(sys)("rtp_llm"))
_shim("rtp_llm.models_py", type(sys)("rtp_llm.models_py"))
_shim("rtp_llm.models_py.modules", type(sys)("rtp_llm.models_py.modules"))
_shim("rtp_llm.models_py.modules.dsv4", type(sys)("rtp_llm.models_py.modules.dsv4"))
_shim(
    "rtp_llm.models_py.modules.dsv4.decode",
    type(sys)("rtp_llm.models_py.modules.dsv4.decode"),
)
_shim("rtp_llm.models_py.modules.dsv4.decode.pool_layout", pl)
_shim("rtp_llm.models_py.modules.dsv4.decode.pool_slot_mapping", psm)
_shim("rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op", kw)
dam = _load(f"{_BASE}/decode_attn_metadata.py", "_dam_p2b")


HEAD_DIM = 512
IDX_HEAD_DIM = 128
WINDOW_SIZE = 128
MAX_SEQ_LEN = 8192
INDEX_TOPK = 64


def _build_pool(num_blocks: int, eb: int, vec_dim: int) -> torch.Tensor:
    bpe = vec_dim * 2  # bf16
    return torch.zeros(num_blocks, eb * bpe, dtype=torch.uint8)


def test_hca_paged_slots_boundary_only():
    """HCA: ratio=128. Slot is -1 except when (start_pos+1)%128 == 0."""
    device = torch.device("cpu")
    swa_bt = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    hca_bt = torch.tensor([[20, 21], [22, 23], [24, 25]], dtype=torch.int32)
    # boundary iff (sp+1) % 128 == 0
    # req 0 sp=127 (sp+1=128 → boundary, cmp_idx=0)
    # req 1 sp=200 (sp+1=201 → not boundary)
    # req 2 sp=255 (sp+1=256 → boundary, cmp_idx=1)
    seq_lens = torch.tensor([127, 200, 255], dtype=torch.int32)

    paged_bts = {pl.SWA_KV: swa_bt, pl.HCA_KV: hca_bt}
    paged_ebs = {pl.SWA_KV: 256, pl.HCA_KV: 2}

    meta = dam.build_decode_metadata(
        start_pos=seq_lens,
        q_len=1,
        window_size=WINDOW_SIZE,
        head_dim=HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
        compress_ratios=[128],
        index_topk=INDEX_TOPK,
        device=device,
        paged_block_tables=paged_bts,
        paged_pool_entries_per_block=paged_ebs,
    )

    hca_slot = meta.pool_write_slot_mappings[pl.HCA_KV]
    # req 0: cmp_idx=0 → block_table[0,0]=20, offset 0  → 20*2+0 = 40
    # req 1: not boundary → -1
    # req 2: cmp_idx=1 → block_table[2,0]=24, offset 1  → 24*2+1 = 49
    expected = torch.tensor([40, -1, 49], dtype=torch.long)
    assert torch.equal(hca_slot, expected), (hca_slot, expected)

    # Now write 3 K vectors with mask_negative=True; only slots 40 and 49
    # should be touched.
    pool = _build_pool(num_blocks=64, eb=2, vec_dim=HEAD_DIM)
    desc = pl.build_pool_descriptor(pool, pl.HCA_KV, HEAD_DIM, IDX_HEAD_DIM, coff=1)
    k = torch.randn(3, HEAD_DIM, dtype=torch.bfloat16)
    kw.write_kv_to_pool(k, hca_slot, desc.view(), mask_negative=True)
    v = desc.view()
    assert torch.equal(v[40], k[0])
    assert torch.equal(v[49], k[2])
    # All other slots should still be zero.
    assert torch.count_nonzero(v[0]) == 0
    assert torch.count_nonzero(v[100]) == 0


def test_indexer_paged_slots_share_csa_boundary():
    """INDEXER_KV uses ratio=4 boundary, same as CSA_KV."""
    device = torch.device("cpu")
    swa_bt = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)
    idx_bt = torch.tensor([[30, 31], [32, 33]], dtype=torch.int32)
    # boundary iff (sp+1) % 4 == 0
    # req 0 sp=3 (sp+1=4 → boundary, cmp_idx=0)
    # req 1 sp=4 (sp+1=5 → not boundary)
    seq_lens = torch.tensor([3, 4], dtype=torch.int32)

    paged_bts = {pl.SWA_KV: swa_bt, pl.INDEXER_KV: idx_bt}
    paged_ebs = {pl.SWA_KV: 256, pl.INDEXER_KV: 64}

    meta = dam.build_decode_metadata(
        start_pos=seq_lens,
        q_len=1,
        window_size=WINDOW_SIZE,
        head_dim=HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
        compress_ratios=[4],
        index_topk=INDEX_TOPK,
        device=device,
        paged_block_tables=paged_bts,
        paged_pool_entries_per_block=paged_ebs,
    )

    idx_slot = meta.pool_write_slot_mappings[pl.INDEXER_KV]
    # req 0: cmp_idx=0 → block_table[0,0]=30, offset 0 → 30*64+0 = 1920
    # req 1: not boundary → -1
    expected = torch.tensor([1920, -1], dtype=torch.long)
    assert torch.equal(idx_slot, expected), (idx_slot, expected)


if __name__ == "__main__":
    test_hca_paged_slots_boundary_only()
    print("[1] HCA-K boundary-only slots + masked write OK")
    test_indexer_paged_slots_share_csa_boundary()
    print("[2] INDEXER-K shares ratio=4 boundary with CSA OK")
    print("ALL OK")
