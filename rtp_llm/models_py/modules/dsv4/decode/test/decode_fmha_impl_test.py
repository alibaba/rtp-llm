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


@dataclass
class _StubAttnInputs:
    """Minimal PyAttentionInputs-shaped stub for unit tests."""

    sequence_lengths: torch.Tensor


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


if __name__ == "__main__":
    unittest.main()
