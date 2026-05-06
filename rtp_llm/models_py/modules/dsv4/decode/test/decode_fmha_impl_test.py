"""Stage 3A — DSv4DecodeFmhaImpl wiring test.

Verifies the CUDA-graph FMHA impl built by
``DeepSeekV4Model.prepare_fmha_impl(is_cuda_graph=True)`` exposes the
contract the C++ ``CudaGraphRunner`` relies on:

  * ``support_cuda_graph()`` returns True
  * ``prepare_cuda_graph(attn_inputs)`` is callable
  * After ``prepare_cuda_graph``, the impl's metadata reflects the new
    ``start_pos`` (sequence_lengths) and reuses the same buffer addresses
    (no realloc — required for replay correctness)

Synthetic ``PyAttentionInputs``-like stub used here so the test can run
on the CPU dev box. The real PyAttentionInputs comes from C++ pybind in
production but for our purposes we only access ``sequence_lengths``,
``input_lengths``, and ``device``.
"""

import os
import sys
import unittest
from dataclasses import dataclass

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.decode.decode_fmha_impl import (
    DSv4DecodeFmhaImpl,
    DSv4DecodeFmhaImplConfig,
)
from rtp_llm.models_py.modules.dsv4.decode.forward import build_paged_pool_specs
from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import write_kv_to_pool
from rtp_llm.models_py.modules.dsv4.decode.pool_slot_mapping import (
    compute_kv_pool_slot_mapping,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import PoolBackedModule
from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV, INDEXER_STATE, SWA_KV


@dataclass
class _StubAttnInputs:
    """Minimal PyAttentionInputs-shaped stub for unit tests."""

    sequence_lengths: torch.Tensor
    kv_cache_kernel_block_id_device_by_group: list = None


class _StubAttn:
    def __init__(self, entries_by_type):
        self._kv_cache = None
        self._entries_by_type = entries_by_type

    def _pool_entries_per_block(self, attn_type):
        return self._entries_by_type.get(attn_type, 0)


class _StubLayer:
    def __init__(self, attn):
        self.attn = attn


class _StubV4:
    def __init__(self, entries_by_type):
        self.layers = [_StubLayer(_StubAttn(entries_by_type))]


class _StubKvCache:
    def __init__(self, group_region_names):
        self.group_region_names = group_region_names


def _make_impl(bs: int = 4) -> DSv4DecodeFmhaImpl:
    cfg = DSv4DecodeFmhaImplConfig(
        max_batch_size=bs,
        q_len=1,
        window_size=8,
        head_dim=32,
        max_seq_len=64,
        compress_ratios=[4, 128],
        index_topk=4,
    )
    return DSv4DecodeFmhaImpl(cfg, device=torch.device("cpu"))


class TestDSv4DecodeFmhaImpl(unittest.TestCase):

    def test_support_cuda_graph(self):
        impl = _make_impl()
        self.assertTrue(
            impl.support_cuda_graph(),
            "support_cuda_graph() must be True so CudaGraphRunner engages",
        )
        self.assertTrue(callable(getattr(impl, "prepare_cuda_graph", None)))

    def test_metadata_allocated_at_construction(self):
        impl = _make_impl(bs=4)
        meta = impl.metadata
        self.assertEqual(meta.batch_size, 4)
        self.assertEqual(meta.start_pos.shape, (4,))
        self.assertEqual(meta.slot_mapping_swa.shape, (4,))
        self.assertEqual(meta.topk_window_idxs.shape, (4, 1, 8))
        # Per-ratio buffers
        self.assertIn(4, meta.slot_mapping_compressed)
        self.assertIn(128, meta.slot_mapping_compressed)
        self.assertEqual(meta.slot_mapping_compressed[4].shape, (4,))

    def test_prepare_cuda_graph_updates_metadata(self):
        impl = _make_impl(bs=4)
        attn_in = _StubAttnInputs(
            sequence_lengths=torch.tensor([3, 5, 7, 11], dtype=torch.int32),
        )
        impl.prepare_cuda_graph(attn_in)

        # start_pos written
        expected = torch.tensor([3, 5, 7, 11], dtype=torch.int32)
        self.assertTrue(torch.equal(impl.metadata.start_pos, expected))

        # SWA slot mapping = req * window + (start_pos % window):
        # window=8 → [0*8 + 3, 1*8 + 5, 2*8 + 7, 3*8 + 3] = [3, 13, 23, 27]
        self.assertTrue(
            torch.equal(
                impl.metadata.slot_mapping_swa,
                torch.tensor([3, 13, 23, 27], dtype=torch.int32),
            )
        )

    def test_prepare_cuda_graph_no_realloc_across_calls(self):
        """Critical replay-correctness invariant: data_ptr() of every
        metadata tensor must be unchanged across calls. The captured
        graph holds the old pointer; a realloc would silently make
        replay read stale memory."""
        impl = _make_impl(bs=4)
        before = {
            "start_pos": impl.metadata.start_pos.data_ptr(),
            "slot_swa": impl.metadata.slot_mapping_swa.data_ptr(),
            "topk_window": impl.metadata.topk_window_idxs.data_ptr(),
        }
        for r, t in impl.metadata.slot_mapping_compressed.items():
            before[f"slot_cmp[{r}]"] = t.data_ptr()
        for r, t in impl.metadata.topk_total_by_ratio.items():
            before[f"topk_total[{r}]"] = t.data_ptr()

        # Multiple updates with different start_pos
        impl.prepare_cuda_graph(
            _StubAttnInputs(
                sequence_lengths=torch.tensor([0, 1, 2, 3], dtype=torch.int32)
            )
        )
        impl.prepare_cuda_graph(
            _StubAttnInputs(
                sequence_lengths=torch.tensor([10, 20, 30, 40], dtype=torch.int32)
            )
        )
        impl.prepare_cuda_graph(
            _StubAttnInputs(
                sequence_lengths=torch.tensor([5, 5, 5, 5], dtype=torch.int32)
            )
        )

        after = {
            "start_pos": impl.metadata.start_pos.data_ptr(),
            "slot_swa": impl.metadata.slot_mapping_swa.data_ptr(),
            "topk_window": impl.metadata.topk_window_idxs.data_ptr(),
        }
        for r, t in impl.metadata.slot_mapping_compressed.items():
            after[f"slot_cmp[{r}]"] = t.data_ptr()
        for r, t in impl.metadata.topk_total_by_ratio.items():
            after[f"topk_total[{r}]"] = t.data_ptr()

        for k in before:
            self.assertEqual(
                before[k], after[k], f"buffer {k} reallocated by prepare_cuda_graph"
            )

    def test_warmup_clamp(self):
        """Synthetic warmup probes can pass start_pos beyond max_seq_len-1.
        prepare must clamp so downstream RoPE / kv_cache indexing stays
        in range."""
        impl = _make_impl(bs=2)
        impl.prepare_cuda_graph(
            _StubAttnInputs(
                sequence_lengths=torch.tensor([1000, 5000], dtype=torch.int32),
            )
        )
        max_s = impl.config.max_seq_len  # 64
        self.assertTrue(bool((impl.metadata.start_pos < max_s).all()))

    def test_paged_pool_specs_use_framework_block_table_width(self):
        """CUDA graph metadata must allocate the same block-table width as
        C++ capture inputs. DSV4 tail/fixed pools may only have two live
        blocks, but their block tables are still indexed by absolute segment."""
        width = 17
        kv_cache = _StubKvCache([SWA_KV, CSA_KV, INDEXER_STATE])
        v4 = _StubV4({SWA_KV: 256, CSA_KV: 64, INDEXER_STATE: 4})
        attn_inputs = _StubAttnInputs(
            sequence_lengths=torch.tensor([0], dtype=torch.int32),
            kv_cache_kernel_block_id_device_by_group=[
                torch.zeros(2, width, dtype=torch.int32),
                torch.zeros(2, width, dtype=torch.int32),
                torch.zeros(2, width, dtype=torch.int32),
            ],
        )

        specs = build_paged_pool_specs(
            kv_cache,
            v4,
            max_seq_len=8192,
            attn_inputs=attn_inputs,
        )

        self.assertEqual(specs[SWA_KV], (256, width))
        self.assertEqual(specs[CSA_KV], (64, width))
        self.assertEqual(specs[INDEXER_STATE], (4, width))

    def test_paged_block_table_copy_preserves_invalid_ids(self):
        """Synthetic CUDA graph capture starts with zero block tables. The
        DSv4 impl should preserve those sentinel ids in metadata instead of
        rewriting them to a valid physical block."""
        cfg = DSv4DecodeFmhaImplConfig(
            max_batch_size=2,
            q_len=1,
            window_size=8,
            head_dim=32,
            max_seq_len=64,
            compress_ratios=[4],
            index_topk=4,
            paged_pool_specs={SWA_KV: (256, 4), CSA_KV: (64, 4)},
            group_region_names=[SWA_KV, CSA_KV],
        )
        by_group = [
            torch.tensor([[0, -1, 11, 12], [0, 0, 21, 22]], dtype=torch.int32),
            torch.tensor([[0, 31, 32, 0], [0, 41, 42, 0]], dtype=torch.int32),
        ]
        impl = DSv4DecodeFmhaImpl(
            cfg,
            device=torch.device("cpu"),
            attn_inputs=_StubAttnInputs(
                sequence_lengths=torch.tensor([3, 7], dtype=torch.int32),
                kv_cache_kernel_block_id_device_by_group=by_group,
            ),
        )

        self.assertTrue(torch.equal(impl.metadata.pool_block_tables[SWA_KV], by_group[0]))
        self.assertTrue(torch.equal(impl.metadata.pool_block_tables[CSA_KV], by_group[1]))

    def test_pool_slot_helpers_skip_null_and_oob_blocks(self):
        block_table = torch.tensor([[0, -1, 2]], dtype=torch.int32)
        abs_pos = torch.tensor([0, 4, 8, 12, -1], dtype=torch.int32)

        mapped = compute_kv_pool_slot_mapping(
            block_table,
            abs_pos,
            entries_per_block=4,
        )

        self.assertTrue(
            torch.equal(mapped, torch.tensor([-1, -1, 8, -1, -1], dtype=torch.long))
        )

    def test_pool_backed_module_masks_oob_physical_slots(self):
        helper = PoolBackedModule()
        block_table = torch.tensor([[999, 1]], dtype=torch.int32)

        valid, safe_slot = helper._compute_pool_slots(
            bsz=1,
            T=4,
            block_table=block_table,
            eb=4,
            device=torch.device("cpu"),
            pool_rows=16,
        )

        self.assertFalse(bool(valid.any()))
        self.assertTrue(torch.equal(safe_slot, torch.zeros_like(safe_slot)))

    def test_write_kv_to_pool_skips_oob_slots(self):
        pool = torch.zeros(2, 1, dtype=torch.float32)
        src = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
        slots = torch.tensor([1, 2, -1], dtype=torch.long)

        write_kv_to_pool(src, slots, pool, mask_negative=True)

        self.assertTrue(torch.equal(pool[:, 0], torch.tensor([0.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
