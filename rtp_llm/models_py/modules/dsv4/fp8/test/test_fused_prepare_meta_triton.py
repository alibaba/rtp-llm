"""UT: numerical equivalence of fused Triton prepare vs Python path.

Compares every persistent buffer written by
``fused_update_decode_meta_pure`` against the Python reference inside
``update_decode_metadata_in_place_fp8``. Both paths use the same
pre-allocated metadata dataclass; we run them on two independently
allocated copies to avoid cross-contamination, then assert elementwise
``torch.equal`` over the ``[:bs]`` / ``[:bs*q_len]`` prefixes.

Dev-box note: ``rtp_llm/models_py/modules/__init__.py`` pulls in the
whole framework (compiled ops + base layers). This UT only needs the
two pure-Python files under ``dsv4/fp8/decode``, so it loads them via
``importlib`` directly from disk — same code, no package init chain.

Run (bypassing package init):
  CUDA_VISIBLE_DEVICES=0 /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/fp8/test/test_fused_prepare_meta_triton.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_DECODE_DIR = _THIS_DIR.parent / "decode"


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ``_fused_prepare_meta_triton`` is imported lazily from inside
# ``update_decode_metadata_in_place_fp8`` — pre-register it so that
# import resolves to our direct load instead of traversing the package.
_attn_type_mod = _load_module(
    "rtp_llm.models_py.modules.dsv4.attn_type",
    _THIS_DIR.parent.parent / "attn_type.py",
)
_fused_mod = _load_module(
    "rtp_llm.models_py.modules.dsv4.fp8.decode._fused_prepare_meta_triton",
    _DECODE_DIR / "_fused_prepare_meta_triton.py",
)
# pool_slot_mapping imports attn_type from the package path;
# pre-register it via the loaded attn_type module above so the import
# resolves without triggering the full framework __init__ chain.
_pool_mod = _load_module(
    "rtp_llm.models_py.modules.dsv4.fp8.decode.pool_slot_mapping",
    _DECODE_DIR / "pool_slot_mapping.py",
)
_translator_mod = _load_module(
    "rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator",
    _DECODE_DIR / "paged_topk_translator.py",
)
_meta_mod = _load_module(
    "rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata",
    _DECODE_DIR / "decode_attn_metadata.py",
)

allocate_decode_metadata_fp8 = _meta_mod.allocate_decode_metadata_fp8
update_decode_metadata_in_place_fp8 = _meta_mod.update_decode_metadata_in_place_fp8


# V4-Flash config: ratios {4, 128}, window=128, index_topk=512.
_V4_COMPRESS_RATIOS = [0, 0] + [4, 128] * 20 + [4, 0]
_WINDOW = 128
_HEAD_DIM = 512
_INDEX_TOPK = 512


def _alloc(max_bs: int, q_len: int, max_seq_len: int, device: torch.device):
    return allocate_decode_metadata_fp8(
        max_batch_size=max_bs,
        q_len=q_len,
        window_size=_WINDOW,
        head_dim=_HEAD_DIM,
        max_seq_len=max_seq_len,
        compress_ratios=_V4_COMPRESS_RATIOS,
        index_topk=_INDEX_TOPK,
        device=device,
    )


def _run_update(meta, start_pos, fused: bool):
    prev = os.environ.get("DSV4_FUSED_PREPARE")
    os.environ["DSV4_FUSED_PREPARE"] = "1" if fused else "0"
    try:
        update_decode_metadata_in_place_fp8(meta, start_pos)
    finally:
        if prev is None:
            os.environ.pop("DSV4_FUSED_PREPARE", None)
        else:
            os.environ["DSV4_FUSED_PREPARE"] = prev


def _assert_equal(tc, name, a, b):
    if torch.equal(a, b):
        return
    diff_mask = a != b
    n_diff = int(diff_mask.sum().item())
    idx = torch.nonzero(diff_mask.reshape(-1), as_tuple=False).reshape(-1)[:8]
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    samples = [
        f"  [{int(i.item())}] py={int(flat_a[i].item())} fused={int(flat_b[i].item())}"
        for i in idx
    ]
    tc.fail(
        f"mismatch on '{name}': {n_diff}/{a.numel()} elements differ.\n"
        + "\n".join(samples)
    )


def _compare(tc, meta_py, meta_fu, bs: int, q_len: int):
    T = bs * q_len
    _assert_equal(tc, "start_pos", meta_py.start_pos[:bs], meta_fu.start_pos[:bs])
    _assert_equal(
        tc,
        "cache_seqlens_i32",
        meta_py.cache_seqlens_i32[:bs],
        meta_fu.cache_seqlens_i32[:bs],
    )
    _assert_equal(
        tc,
        "slot_mapping_swa",
        meta_py.slot_mapping_swa[:T],
        meta_fu.slot_mapping_swa[:T],
    )
    for r in (4, 128):
        _assert_equal(
            tc,
            f"slot_mapping_compressed[{r}]",
            meta_py.slot_mapping_compressed[r][:T],
            meta_fu.slot_mapping_compressed[r][:T],
        )
        _assert_equal(
            tc,
            f"compressed_lens[{r}]",
            meta_py.compressed_lens[r][:bs],
            meta_fu.compressed_lens[r][:bs],
        )
        _assert_equal(
            tc,
            f"topk_total_by_ratio[{r}]",
            meta_py.topk_total_by_ratio[r][:bs],
            meta_fu.topk_total_by_ratio[r][:bs],
        )
    _assert_equal(
        tc,
        "topk_window_idxs",
        meta_py.topk_window_idxs[:bs],
        meta_fu.topk_window_idxs[:bs],
    )
    _assert_equal(
        tc,
        "swa_abs_idx",
        meta_py.swa_abs_idx[:bs],
        meta_fu.swa_abs_idx[:bs],
    )
    _assert_equal(
        tc,
        "topk_buffer_compressed",
        meta_py.topk_buffer_compressed[:bs],
        meta_fu.topk_buffer_compressed[:bs],
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedPrepareMetaEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0")

    def _run_case(self, bs: int, q_len: int, max_seq_len: int, start_pos_values):
        meta_py = _alloc(bs, q_len, max_seq_len, self.device)
        meta_fu = _alloc(bs, q_len, max_seq_len, self.device)

        start_pos = torch.tensor(
            start_pos_values, dtype=torch.int32, device=self.device
        )
        self.assertEqual(start_pos.shape[0], bs)

        _run_update(meta_py, start_pos, fused=False)
        _run_update(meta_fu, start_pos, fused=True)

        _compare(self, meta_py, meta_fu, bs, q_len)

    def test_early_decode_all_zero(self):
        self._run_case(bs=4, q_len=1, max_seq_len=4096, start_pos_values=[0, 0, 0, 0])

    def test_small_batch_mixed_positions(self):
        self._run_case(
            bs=8,
            q_len=1,
            max_seq_len=4096,
            start_pos_values=[0, 1, 63, 127, 128, 511, 4095, 12287],
        )

    def test_bs1_large_pos(self):
        self._run_case(bs=1, q_len=1, max_seq_len=65536, start_pos_values=[60000])

    def test_bs128_randomised(self):
        torch.manual_seed(0)
        bs = 128
        max_seq_len = 65664
        vals = torch.randint(0, max_seq_len - 1, (bs,), dtype=torch.int32).tolist()
        vals[0] = 0
        vals[1] = 1
        vals[2] = 127
        vals[3] = 128
        vals[4] = 511
        vals[5] = max_seq_len - 2
        self._run_case(bs=bs, q_len=1, max_seq_len=max_seq_len, start_pos_values=vals)

    def test_qlen_gt_1(self):
        self._run_case(
            bs=4, q_len=2, max_seq_len=4096, start_pos_values=[0, 127, 511, 2048]
        )


# ---------------------------------------------------------------------------
# Phase 2b: fused pool_write_slot_mapping equivalence.
# ---------------------------------------------------------------------------
# The base suite above runs ``update_decode_metadata_in_place_fp8`` WITHOUT
# ``paged_block_tables``, so phase2b is skipped.  This suite wires up the
# paged-pool path: it allocates with ``paged_pool_specs`` so that
# ``meta.pool_block_tables`` / ``pool_write_slot_mappings`` exist, seeds
# the block tables with random valid block ids, then compares the fused
# Triton kernel's outputs to the Python ``compute_kv_pool_slot_mapping``
# loop.

# Pool specs mirror V4-Flash runtime geometry enough for numerical test:
#   SWA:     E=256, max_blocks per req fits a single window
#   CSA:     E=64,  ratio=4
#   INDEXER: E=64,  ratio=4 (shares CSA's boundary)
#   HCA:     E=2,   ratio=128
_SWA_E = 256
_CSA_E = 64
_IDX_E = 64
_HCA_E = 2


def _alloc_with_paged_pools(
    max_bs: int, q_len: int, max_seq_len: int, device: torch.device
):
    from rtp_llm.models_py.modules.dsv4.attn_type import (
        CSA_KV,
        HCA_KV,
        INDEXER_KV,
        SWA_KV,
    )

    # max_blocks_per_req per pool — pick with headroom so clamp isn't
    # excercised on valid positions in the tests below.
    paged_pool_specs = {
        SWA_KV: (_SWA_E, max(1, (_WINDOW + _SWA_E - 1) // _SWA_E)),
        CSA_KV: (_CSA_E, max(1, (max_seq_len // 4 + _CSA_E - 1) // _CSA_E)),
        INDEXER_KV: (
            _IDX_E,
            max(1, (max_seq_len // 4 + _IDX_E - 1) // _IDX_E),
        ),
        HCA_KV: (
            _HCA_E,
            max(1, (max_seq_len // 128 + _HCA_E - 1) // _HCA_E),
        ),
    }
    return allocate_decode_metadata_fp8(
        max_batch_size=max_bs,
        q_len=q_len,
        window_size=_WINDOW,
        head_dim=_HEAD_DIM,
        max_seq_len=max_seq_len,
        compress_ratios=_V4_COMPRESS_RATIOS,
        index_topk=_INDEX_TOPK,
        device=device,
        paged_pool_specs=paged_pool_specs,
    )


def _seed_block_tables(meta, seed: int = 0):
    """Fill meta.pool_block_tables with deterministic non-zero block ids.
    Returns (paged_block_tables_arg, paged_pool_entries_per_block_arg).
    """
    from rtp_llm.models_py.modules.dsv4.attn_type import (
        CSA_KV,
        HCA_KV,
        INDEXER_KV,
        SWA_KV,
    )

    gen = torch.Generator(device=meta.pool_block_tables[SWA_KV].device).manual_seed(
        seed
    )
    for at, bt in meta.pool_block_tables.items():
        # Distinct ranges per pool so cross-pool aliasing bugs surface.
        base = {SWA_KV: 1000, CSA_KV: 2000, INDEXER_KV: 3000, HCA_KV: 4000}[at]
        rand = torch.randint(
            0, 10000, bt.shape, generator=gen, dtype=torch.int32, device=bt.device
        )
        bt.copy_(rand + base)

    paged_pool_entries_per_block = {
        SWA_KV: _SWA_E,
        CSA_KV: _CSA_E,
        INDEXER_KV: _IDX_E,
        HCA_KV: _HCA_E,
    }
    # paged_block_tables arg is the same tensor reference — caller
    # would pass the framework's live block table; for the test the
    # pre-seeded meta buffers are fine.
    paged_block_tables = dict(meta.pool_block_tables)
    return paged_block_tables, paged_pool_entries_per_block


def _run_update_paged(meta, start_pos, paged_bt, paged_e, fused: bool):
    prev = os.environ.get("DSV4_FUSED_PREPARE")
    os.environ["DSV4_FUSED_PREPARE"] = "1" if fused else "0"
    try:
        update_decode_metadata_in_place_fp8(
            meta,
            start_pos,
            paged_block_tables=paged_bt,
            paged_pool_entries_per_block=paged_e,
        )
    finally:
        if prev is None:
            os.environ.pop("DSV4_FUSED_PREPARE", None)
        else:
            os.environ["DSV4_FUSED_PREPARE"] = prev


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedPhase2bEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0")

    def _run_case(self, bs: int, q_len: int, max_seq_len: int, start_pos_values):
        meta_py = _alloc_with_paged_pools(bs, q_len, max_seq_len, self.device)
        meta_fu = _alloc_with_paged_pools(bs, q_len, max_seq_len, self.device)

        paged_bt_py, paged_e = _seed_block_tables(meta_py, seed=42)
        paged_bt_fu, _ = _seed_block_tables(meta_fu, seed=42)

        start_pos = torch.tensor(
            start_pos_values, dtype=torch.int32, device=self.device
        )
        self.assertEqual(start_pos.shape[0], bs)

        _run_update_paged(meta_py, start_pos, paged_bt_py, paged_e, fused=False)
        _run_update_paged(meta_fu, start_pos, paged_bt_fu, paged_e, fused=True)

        from rtp_llm.models_py.modules.dsv4.attn_type import (
            CSA_KV,
            HCA_KV,
            INDEXER_KV,
            SWA_KV,
        )

        T = bs * q_len
        for at, label in (
            (SWA_KV, "SWA"),
            (CSA_KV, "CSA"),
            (INDEXER_KV, "INDEXER"),
            (HCA_KV, "HCA"),
        ):
            _assert_equal(
                self,
                f"pool_write_slot_mappings[{label}]",
                meta_py.pool_write_slot_mappings[at][:T],
                meta_fu.pool_write_slot_mappings[at][:T],
            )

    def test_phase2b_small_batch(self):
        self._run_case(
            bs=4, q_len=1, max_seq_len=4096, start_pos_values=[0, 3, 127, 2048]
        )

    def test_phase2b_boundary_cases(self):
        # start_pos values that hit both on-boundary and off-boundary cases
        # for CSA (ratio=4) and HCA (ratio=128).
        self._run_case(
            bs=8,
            q_len=1,
            max_seq_len=4096,
            start_pos_values=[0, 1, 2, 3, 127, 255, 511, 1023],
        )

    def test_phase2b_bs128(self):
        torch.manual_seed(0)
        bs = 128
        max_seq_len = 65664
        vals = torch.randint(0, max_seq_len - 1, (bs,), dtype=torch.int32).tolist()
        vals[0] = 0
        vals[1] = 3  # just before ratio=4 boundary
        vals[2] = 127  # just before ratio=128 boundary
        vals[3] = 255
        self._run_case(bs=bs, q_len=1, max_seq_len=max_seq_len, start_pos_values=vals)


if __name__ == "__main__":
    unittest.main()
