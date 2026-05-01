"""Phase 2B-2a: byte-equivalence between legacy register_buffer SWA read
and paged-pool SWA read (via global slot ids + indirect attn).

This test stays CPU-only (Triton kernel needs CUDA, so the actual
``translate_local_to_global_slots`` call is skipped on CPU; we
re-implement the same logic in pure torch and check byte-equivalence
of the output gather pattern). The kernel itself is exercised
end-to-end by the running server.

Validates:
  1. ``swa_abs_idx`` is computed correctly (left-aligned, -1 padded).
  2. The torch reference of the global-slot translation matches the
     formula ``block_table[req, abs//E] * E + abs % E``.
  3. Indirect ``index_select`` on the paged pool returns the same K
     vectors as a direct request-local read of the SWA register_buffer
     populated by Phase 2A's dual-write.
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
pl = _load(f"{_BASE}/pool_layout.py", "_pl_p2b2a")
psm = _load(f"{_BASE}/pool_slot_mapping.py", "_psm_p2b2a")
kw = _load(f"{_BASE}/kv_write_decode_op.py", "_kw_p2b2a")


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
dam = _load(f"{_BASE}/decode_attn_metadata.py", "_dam_p2b2a")


HEAD_DIM = 512
WINDOW_SIZE = 128
MAX_SEQ_LEN = 8192
INDEX_TOPK = 64


def _torch_translate_to_global(
    req_id: torch.Tensor,  # [T] int32
    block_table: torch.Tensor,  # [B, max_blocks] int32
    local_idx: torch.Tensor,  # [T, K] int32, -1 sentinel
    block_size: int,
) -> torch.Tensor:
    """Pure-torch reference of triton_convert_req_index_to_global_index."""
    valid = local_idx >= 0
    safe_local = torch.where(local_idx >= 0, local_idx, torch.zeros_like(local_idx))
    block_id = safe_local // block_size
    in_block = safe_local % block_size
    in_bounds = (block_id >= 0) & (block_id < block_table.shape[1])
    safe_block_id = torch.where(in_bounds, block_id, torch.zeros_like(block_id))
    base = block_table[
        req_id.long().unsqueeze(-1).expand_as(safe_block_id), safe_block_id.long()
    ]
    out = base * block_size + in_block
    return torch.where(valid & in_bounds, out, torch.full_like(out, -1))


def test_swa_abs_idx_population():
    """Phase 2B-2a metadata field: left-aligned valid abs positions, -1 pad."""
    device = torch.device("cpu")
    swa_bt = torch.tensor([[10, 11], [12, 13]], dtype=torch.int32)
    paged_bts = {pl.SWA_KV: swa_bt}
    paged_ebs = {pl.SWA_KV: 256}

    # req 0 sp=0  → only abs_pos=0 valid
    # req 1 sp=300 → window covers [173..300], all 128 valid
    seq_lens = torch.tensor([0, 300], dtype=torch.int32)

    meta = dam.build_decode_metadata(
        start_pos=seq_lens,
        q_len=1,
        window_size=WINDOW_SIZE,
        head_dim=HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
        compress_ratios=[0],
        index_topk=INDEX_TOPK,
        device=device,
        paged_block_tables=paged_bts,
        paged_pool_entries_per_block=paged_ebs,
    )
    assert meta.swa_abs_idx is not None
    assert meta.swa_abs_idx.shape == (2, 1, WINDOW_SIZE)
    # req 0: only [0] valid — first entry is 0, rest -1
    assert meta.swa_abs_idx[0, 0, 0].item() == 0
    assert (meta.swa_abs_idx[0, 0, 1:] == -1).all()
    # req 1: [173..300] all valid
    assert meta.swa_abs_idx[1, 0, 0].item() == 173
    assert meta.swa_abs_idx[1, 0, -1].item() == 300
    assert (meta.swa_abs_idx[1, 0] >= 0).all()


def test_paged_swa_read_matches_legacy_ring_buffer():
    """Write K via Phase 2A dual-write to BOTH a register_buffer ring AND
    the paged SWA pool, then verify a paged read using global slot ids
    returns the same vectors as the register_buffer ring read."""
    device = torch.device("cpu")
    bs = 3
    swa_bt = torch.tensor([[40, 41], [42, 43], [44, 45]], dtype=torch.int32)

    # Per-step decode: walk each request through abs positions [0..N) and
    # at each step write the new K to both buffers, then verify the
    # post-write window read matches.
    N = 200  # > WINDOW_SIZE=128 to exercise both partial and full ring
    swa_pool = torch.zeros(64, 256 * HEAD_DIM * 2, dtype=torch.uint8)
    swa_desc = pl.build_pool_descriptor(swa_pool, pl.SWA_KV, HEAD_DIM, 128, coff=1)
    register_buf = torch.zeros(bs, WINDOW_SIZE, HEAD_DIM, dtype=torch.bfloat16)

    paged_bts = {pl.SWA_KV: swa_bt}
    paged_ebs = {pl.SWA_KV: 256}

    rng = torch.Generator().manual_seed(42)
    for sp in range(N):
        # All 3 reqs at the same step for simplicity.
        seq_lens = torch.tensor([sp, sp, sp], dtype=torch.int32)
        meta = dam.build_decode_metadata(
            start_pos=seq_lens,
            q_len=1,
            window_size=WINDOW_SIZE,
            head_dim=HEAD_DIM,
            max_seq_len=MAX_SEQ_LEN,
            compress_ratios=[0],
            index_topk=INDEX_TOPK,
            device=device,
            paged_block_tables=paged_bts,
            paged_pool_entries_per_block=paged_ebs,
        )
        k = torch.randn(bs, HEAD_DIM, dtype=torch.bfloat16, generator=rng)
        # legacy ring write
        ring_slot = sp % WINDOW_SIZE
        register_buf[:, ring_slot] = k
        # paged dual-write
        kw.write_kv_to_pool(
            k,
            meta.pool_write_slot_mappings[pl.SWA_KV],
            swa_desc.view(),
            mask_negative=False,
        )

    # After N steps, read back via both routes and compare.
    sp = N - 1
    seq_lens = torch.tensor([sp, sp, sp], dtype=torch.int32)
    meta = dam.build_decode_metadata(
        start_pos=seq_lens,
        q_len=1,
        window_size=WINDOW_SIZE,
        head_dim=HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
        compress_ratios=[0],
        index_topk=INDEX_TOPK,
        device=device,
        paged_block_tables=paged_bts,
        paged_pool_entries_per_block=paged_ebs,
    )

    # Paged read: translate swa_abs_idx → global slots, gather from pool view.
    T = bs * 1
    swa_local = meta.swa_abs_idx.reshape(T, WINDOW_SIZE)
    req_id = torch.arange(bs, dtype=torch.int32)
    swa_global = _torch_translate_to_global(req_id, swa_bt, swa_local, 256)

    pool_view = swa_desc.view()  # [num_slots, D]
    safe_global = torch.where(
        swa_global >= 0,
        swa_global,
        torch.zeros_like(swa_global),
    ).to(torch.long)
    paged_kv = pool_view.index_select(0, safe_global.view(-1)).view(
        T, WINDOW_SIZE, HEAD_DIM
    )
    paged_mask = (swa_global >= 0).unsqueeze(-1)
    paged_kv = torch.where(paged_mask, paged_kv, torch.zeros_like(paged_kv))

    # Legacy read: register_buffer indexed by ring slots from topk_window_idxs.
    legacy_idxs = meta.topk_window_idxs[:bs, 0]  # [bs, win] int32, ring slots
    legacy_kv = torch.zeros(bs, WINDOW_SIZE, HEAD_DIM, dtype=torch.bfloat16)
    for r in range(bs):
        for k_pos in range(WINDOW_SIZE):
            ridx = int(legacy_idxs[r, k_pos].item())
            if ridx >= 0:
                legacy_kv[r, k_pos] = register_buf[r, ridx]
            # else: -1 ring slot → keep zero (matches paged -1-mask behavior)

    paged_kv_b = paged_kv.view(bs, 1, WINDOW_SIZE, HEAD_DIM).squeeze(1)
    assert torch.equal(
        paged_kv_b, legacy_kv
    ), "paged SWA read should byte-equal legacy register_buffer ring read"


if __name__ == "__main__":
    test_swa_abs_idx_population()
    print("[1] swa_abs_idx left-aligned, -1 padded OK")
    test_paged_swa_read_matches_legacy_ring_buffer()
    print("[2] paged SWA read == legacy register_buffer ring read OK")
    print("ALL OK")
