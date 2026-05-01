"""Phase 2 sanity: end-to-end SWA dual-write.

Validates:
  1. ``DSv4DecodeFmhaImpl.prepare`` correctly snapshots the SWA
     ``block_table`` from the framework attention inputs and computes
     the matching paged write slot mapping.
  2. ``write_kv_to_pool`` with that slot mapping writes K rows whose
     bytes match what the legacy register_buffer SWA write produces
     when read back through ``PoolDescriptor.view()``.
  3. Eager-path ``build_decode_metadata`` produces identical paged
     fields when given the same inputs (no impl involved).

Bypasses the rtp_llm package init (which pulls C++ ops linked against
a different torch build) by importing the decode submodules directly
from disk.
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
pl = _load(f"{_BASE}/pool_layout.py", "_pl_phase2")
psm = _load(f"{_BASE}/pool_slot_mapping.py", "_psm_phase2")
kw = _load(f"{_BASE}/kv_write_decode_op.py", "_kw_phase2")


# decode_attn_metadata + decode_fmha_impl pull in pool_layout via package
# import — we have to register them under the full dotted path so their
# `from rtp_llm.models_py...pool_layout import ...` works during exec.
def _shim(name: str, target):
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

dam = _load(f"{_BASE}/decode_attn_metadata.py", "_dam_phase2")
_shim("rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata", dam)
dfi = _load(f"{_BASE}/decode_fmha_impl.py", "_dfi_phase2")


HEAD_DIM = 512
WINDOW_SIZE = 128
MAX_SEQ_LEN = 8192
INDEX_TOPK = 64


class _MockAttnInputs:
    """Minimal stand-in for ``PyAttentionInputs`` populated for decode."""

    def __init__(
        self,
        sequence_lengths: torch.Tensor,
        swa_block_table: torch.Tensor,
        device: torch.device,
    ):
        self.sequence_lengths = sequence_lengths.to(device)
        # gid 6 = SWA_KV (from ATTN_TYPE_TO_GROUP_ID[7]). Other groups can
        # be empty for this Phase 2 SWA-only test.
        empty = torch.empty(0, 0, dtype=torch.int32, device=device)
        self.kv_cache_kernel_block_id_device_by_group = [
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            swa_block_table.to(device),
        ]


def _build_swa_pool(num_blocks: int = 64, device: str = "cpu") -> torch.Tensor:
    eb = 256  # SWA entries_per_block (POOL_LAYOUT.md)
    bpe = HEAD_DIM * 2  # bf16
    return torch.zeros(num_blocks, eb * bpe, dtype=torch.uint8, device=device)


def test_impl_prepare_populates_swa_paged_metadata():
    """Validate end-to-end: impl.prepare → metadata holds SWA block_table
    and the right global slot mapping."""
    device = torch.device("cpu")
    max_bs = 4

    # Build an SWA pool descriptor (num_blocks=64 → plenty for max_bs=4 × 2 blocks).
    swa_pool = _build_swa_pool(num_blocks=64, device="cpu")
    swa_desc = pl.build_pool_descriptor(
        swa_pool,
        pl.SWA_KV,
        HEAD_DIM,
        indexer_head_dim=128,
        coff=1,
    )
    assert swa_desc is not None
    assert swa_desc.entries_per_block == 256

    # Per-request block table: req r owns blocks [10+r*2, 10+r*2+1].
    swa_bt = torch.tensor(
        [[10, 11], [12, 13], [14, 15], [16, 17]],
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([0, 100, 250, 500], dtype=torch.int32)

    cfg = dfi.DSv4DecodeFmhaImplConfig(
        max_batch_size=max_bs,
        q_len=1,
        window_size=WINDOW_SIZE,
        head_dim=HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
        compress_ratios=[0, 0, 4, 128],  # ensure both ratios in the dict
        index_topk=INDEX_TOPK,
        paged_pool_specs={pl.SWA_KV: (256, 2)},
    )
    impl = dfi.DSv4DecodeFmhaImpl(cfg, device=device)
    inputs = _MockAttnInputs(seq_lens, swa_bt, device)
    impl.prepare(inputs)

    meta = impl.metadata
    assert pl.SWA_KV in meta.pool_block_tables
    bt_in_meta = meta.pool_block_tables[pl.SWA_KV]
    assert bt_in_meta.shape == (max_bs, 2)
    assert torch.equal(bt_in_meta, swa_bt)

    # Expected global slot for each request:
    #   slot = block_table[r, abs_pos // 256] * 256 + abs_pos % 256
    expected = torch.tensor(
        [
            10 * 256 + 0,  # req 0, pos 0   → block 10, offset 0
            12 * 256 + 100,  # req 1, pos 100 → block 12, offset 100
            14 * 256 + 250,  # req 2, pos 250 → block 14, offset 250
            17 * 256 + 244,  # req 3, pos 500 = block 17 (=block_table[3,1]) offset 244
        ],
        dtype=torch.long,
    )
    swa_slot = meta.pool_write_slot_mappings[pl.SWA_KV][:max_bs]
    assert torch.equal(swa_slot, expected), (swa_slot, expected)

    # Now exercise the paged write through the same view used by attention.
    k = torch.randn(max_bs, HEAD_DIM, dtype=torch.bfloat16)
    kw.write_kv_to_pool(k, swa_slot, swa_desc.view(), mask_negative=False)
    v = swa_desc.view()
    for i, s in enumerate(expected.tolist()):
        assert torch.equal(v[s], k[i]), f"slot {s} mismatch (req {i})"


def test_eager_build_metadata_matches_impl():
    """build_decode_metadata (eager) should produce the same SWA paged
    fields as impl.prepare for the same inputs."""
    device = torch.device("cpu")
    swa_bt = torch.tensor(
        [[20, 21], [22, 23], [24, 25]],
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([0, 75, 312], dtype=torch.int32)
    paged_bts = {pl.SWA_KV: swa_bt}
    paged_ebs = {pl.SWA_KV: 256}

    meta = dam.build_decode_metadata(
        start_pos=seq_lens,
        q_len=1,
        window_size=WINDOW_SIZE,
        head_dim=HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
        compress_ratios=[0, 0, 4, 128],
        index_topk=INDEX_TOPK,
        device=device,
        paged_block_tables=paged_bts,
        paged_pool_entries_per_block=paged_ebs,
    )

    expected = torch.tensor(
        [
            20 * 256 + 0,
            22 * 256 + 75,
            25 * 256 + 56,  # req 2: pos 312 → block_table[2, 1]=25, offset 56
        ],
        dtype=torch.long,
    )
    assert pl.SWA_KV in meta.pool_block_tables
    swa_slot = meta.pool_write_slot_mappings[pl.SWA_KV]
    assert torch.equal(swa_slot, expected), (swa_slot, expected)


def test_register_buffer_vs_paged_byte_equivalence():
    """The dual-write goal: paged pool data == register_buffer data
    (for the SWA window slice) after one decode step."""
    device = torch.device("cpu")
    bs = 3
    swa_bt = torch.tensor([[30, 31], [32, 33], [34, 35]], dtype=torch.int32)
    # All requests are still inside the first SWA block (pos < 256), so
    # ring slot in register_buffer = pos % WINDOW_SIZE = pos % 128.
    seq_lens = torch.tensor([0, 50, 127], dtype=torch.int32)

    swa_pool = _build_swa_pool(num_blocks=64)
    swa_desc = pl.build_pool_descriptor(
        swa_pool,
        pl.SWA_KV,
        HEAD_DIM,
        128,
        coff=1,
    )

    paged_bts = {pl.SWA_KV: swa_bt}
    paged_ebs = {pl.SWA_KV: 256}
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

    # Mimic attention.forward_decode: register_buffer write + paged write.
    k = torch.randn(bs, HEAD_DIM, dtype=torch.bfloat16)
    register_buffer = torch.zeros(bs, WINDOW_SIZE, HEAD_DIM, dtype=torch.bfloat16)
    # Legacy ring-slot write
    ring_slots = (seq_lens % WINDOW_SIZE).long()
    for r in range(bs):
        register_buffer[r, ring_slots[r]] = k[r]

    kw.write_kv_to_pool(
        k,
        meta.pool_write_slot_mappings[pl.SWA_KV],
        swa_desc.view(),
        mask_negative=False,
    )

    # Paged read at the global slot must equal the register_buffer at the
    # ring slot (same K vector landed in both places).
    v = swa_desc.view()
    for r in range(bs):
        ring = int(ring_slots[r].item())
        global_slot = int(meta.pool_write_slot_mappings[pl.SWA_KV][r].item())
        assert torch.equal(
            v[global_slot], register_buffer[r, ring]
        ), f"req {r}: paged@{global_slot} != register_buffer@{ring}"


if __name__ == "__main__":
    test_impl_prepare_populates_swa_paged_metadata()
    print("[1] impl.prepare wires SWA paged metadata OK")
    test_eager_build_metadata_matches_impl()
    print("[2] eager build_decode_metadata SWA paged fields OK")
    test_register_buffer_vs_paged_byte_equivalence()
    print("[3] dual-write byte equivalence OK")
    print("ALL OK")
