import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.fla import gdn_recurrent_gating as fused
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_fwd,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires SM120",
)
class GdnRecurrentGatingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1548)
        enabled = patch.object(fused, "_ENABLED", True)
        enabled.start()
        self.addCleanup(enabled.stop)

    def case(self, batch, seq, dtype, length=9, inactive=False, block_size=8):
        mixed = torch.randn(batch, seq, 40, 128, device="cuda", dtype=torch.bfloat16)
        q, k, v = mixed.split((8, 8, 24), dim=2)
        ba = torch.randn(batch * seq, 48, device="cuda", dtype=torch.bfloat16)
        b, a = ba[:, :24], ba[:, 24:]
        al = torch.randn(24, device="cuda")
        db = torch.randn(24, device="cuda", dtype=torch.bfloat16)
        blocks = batch * 8 + 1
        original = (
            torch.randn(blocks, 24 * 128 * 128 + 128, device="cuda", dtype=dtype) * 0.01
        )
        old, new = original.clone(), original.clone()
        old_state = old[:, : 24 * 128 * 128].view(blocks, 24, 128, 128)
        new_state = new[:, : 24 * 128 * 128].view(blocks, 24, 128, 128)
        block_map = torch.arange(1, blocks, device="cuda", dtype=torch.int32).reshape(
            batch, 8
        )
        if inactive:
            block_map[-1].zero_()
        lengths = torch.full((batch,), length, device="cuda", dtype=torch.int32)

        def baseline():
            g, beta = fused_gdn_gating(al, a, b, db)
            return fused_recurrent_gated_delta_rule_fwd(
                q,
                k,
                v,
                g.reshape(batch, seq, 24),
                beta.reshape(batch, seq, 24),
                128**-0.5,
                old_state,
                block_map=block_map,
                seq_size_per_block=block_size,
                sequence_lengths=lengths,
                use_qk_l2norm_in_kernel=True,
            )[0]

        def candidate():
            return fused.fused_recurrent_with_gating(
                q, k, v, a, b, al, db, new_state, block_map, block_size, lengths
            )

        reference = baseline()
        actual = candidate()[0]
        valid = batch - 1 if inactive else batch
        self.assertTrue(torch.equal(reference[:valid], actual[:valid]))
        self.assertTrue(torch.equal(old, new))
        return ba, old, new, baseline, candidate

    def test_output_and_padded_state_match(self):
        for dtype in (torch.bfloat16, torch.float32):
            for batch, seq in (
                (1, 1),
                (3, 1),
                (8, 1),
                (17, 1),
                (32, 1),
                (1, 5),
                (2, 5),
                (3, 5),
                (4, 5),
            ):
                with self.subTest(dtype=dtype, batch=batch, seq=seq):
                    self.case(batch, seq, dtype)

    def test_cache_block_boundary_and_inactive_rows(self):
        for block_size in (8, 64):
            for length in (block_size - 1, block_size, block_size + 1):
                self.case(
                    3, 5, torch.float32, length, inactive=True, block_size=block_size
                )

    def test_actual_block_size_output_and_state(self):
        for dtype in (torch.bfloat16, torch.float32):
            for batch, seq in ((1, 1), (8, 1), (32, 1), (1, 5), (2, 5), (4, 5)):
                self.case(batch, seq, dtype, length=65, block_size=64)

    def test_graph_replay_reads_current_raw_gates(self):
        ba, old, new, baseline, candidate = self.case(1, 5, torch.float32)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = candidate()[0]
        old.copy_(new)
        ba.normal_()
        reference = baseline()
        graph.replay()
        self.assertTrue(torch.equal(reference, actual))
        self.assertTrue(torch.equal(old, new))
        graph.reset()

    def test_padded_state_head_falls_back(self):
        q = torch.empty(1, 5, 8, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.empty(1, 5, 24, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.empty(5, 24, device="cuda", dtype=torch.bfloat16)
        al = torch.empty(24, device="cuda", dtype=torch.float32)
        state = torch.empty(9, 24, 129, 128, device="cuda")[:, :, :128]
        block_map = torch.arange(1, 9, device="cuda", dtype=torch.int32).reshape(1, 8)
        lengths = torch.tensor([65], device="cuda", dtype=torch.int32)
        self.assertIsNone(
            fused.fused_recurrent_with_gating(
                q, q, v, a, a, al, al, state, block_map, 64, lengths
            )
        )

    def test_disabled_fallback(self):
        q = torch.empty(1, 1, 8, 128, device="cuda", dtype=torch.bfloat16)
        with patch.object(fused, "_ENABLED", False):
            self.assertIsNone(
                fused.fused_recurrent_with_gating(
                    q, q, q, None, None, None, None, None, None, 8, None
                )
            )


if __name__ == "__main__":
    unittest.main()
