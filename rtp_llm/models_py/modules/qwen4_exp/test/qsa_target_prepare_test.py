import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import (
    restore_indexer_cache,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_norm_rope import (
    try_indexer_norm_rope,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.rope_triton import try_build_text_rope
from rtp_llm.models_py.modules.qwen4_exp.qsa.target_prepare_triton import (
    try_prepare_target_window,
)
from rtp_llm.models_py.modules.qwen4_exp.test import (
    target_writer_fusion_test as fixture,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaTargetPrepareTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 target preparation")
        torch.manual_seed(163201)
        self.writer = fixture.TargetWriterFusionTest()
        self.config = SimpleNamespace(
            style="Mrope",
            dim=64,
            indexer_is_neox_style=True,
            index_factor=3,
            mrope_dim1=12,
            mrope_dim2=10,
            mrope_dim3=10,
            mrope_interleaved=True,
            base=10000.0,
            scale=1.0,
        )

    def tensors(self, prefixes, query, dtype=torch.float32):
        args = self.writer.case(prefixes, query, True)
        rows = len(prefixes) * query
        projected = torch.randn(rows, 640, device="cuda", dtype=torch.bfloat16)
        q = projected[:, :512].view(rows, 4, 128)
        args[0] = projected[:, 512:]
        gamma = torch.randn(128, device="cuda", dtype=dtype) * 0.07
        positions = (
            (args[1][:, None] + torch.arange(query, device="cuda")).flatten().long()
        )
        starts = positions - positions % 4
        frequency = 10000.0 ** (-2 * torch.arange(32, device="cuda") / 64)
        return q, gamma, positions, starts, frequency, args

    def call(self, tensors, candidate):
        q, gamma, positions, starts, frequency, args = tensors
        if candidate:
            raw, prefix, _, _, kg, kv, kt, state, st = args
            return try_prepare_target_window(
                q,
                raw,
                prefix,
                positions,
                starts,
                frequency,
                gamma,
                kg,
                kv,
                kt,
                state,
                st,
                rope_config=self.config,
                norm_eps=1e-6,
            )
        cosine, sine = try_build_text_rope(
            starts,
            self.config,
            "Mrope",
            dtype=q.dtype,
            device=q.device,
            inv_freq=frequency,
        )
        query_cos, query_sin = try_build_text_rope(
            positions,
            self.config,
            "Mrope",
            dtype=q.dtype,
            device=q.device,
            inv_freq=frequency,
        )
        args[2], args[3] = cosine, sine
        undo = self.writer.write(args, True)
        return try_indexer_norm_rope(q, query_cos, query_sin, gamma, 1e-6), undo

    def compare(self, tensors):
        args = tensors[-1]
        original_kv, original_state = args[5].clone(), args[7].clone()
        expected, before = self.call(tensors, False)
        reference_kv, reference_state = args[5].clone(), args[7].clone()
        restore_indexer_cache(before)
        actual, after = self.call(tensors, True)
        torch.testing.assert_close(
            actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
        )
        for actual_value, reference in (
            (args[5], reference_kv),
            (args[7], reference_state),
        ):
            torch.testing.assert_close(actual_value, reference, rtol=0, atol=0)
        for name in ("state_slots", "original_state", "kv_slots", "original_kv"):
            torch.testing.assert_close(
                getattr(after, name), getattr(before, name), rtol=0, atol=0
            )
        restore_indexer_cache(after)
        torch.testing.assert_close(args[5], original_kv, rtol=0, atol=0)
        torch.testing.assert_close(args[7], original_state, rtol=0, atol=0)

    def test_native_q_cache_snapshots_and_undo_on_every_tail_and_cross_page(self):
        for query in (1, 2, 3, 4):
            for prefixes in (
                [4],
                [5],
                [6],
                [7],
                [127],
                [128],
                [4093],
                [32749],
                [5, 127, 4093],
                [124, 125, 126, 127, 128, 129, 130],
            ):
                for dtype in (torch.bfloat16, torch.float32):
                    with self.subTest(query=query, prefixes=prefixes, dtype=dtype):
                        self.compare(self.tensors(prefixes, query, dtype))

    def test_nonunit_rope_scale_and_invalid_physical_slots_match_old_path(self):
        for scale in (0.5, 1.731, 4.0):
            self.config.scale = scale
            for query in (1, 4):
                tensors = self.tensors([5, 127, 128], query)
                tensors[-1][6][0].zero_()
                tensors[-1][8][1].zero_()
                tensors[-1][8][2].fill_(tensors[-1][7].shape[0] + 1)
                self.compare(tensors)

    def test_two_graphs_refresh_queries_gains_frequency_positions_and_pages(self):
        for query in (1, 4):
            tensors = self.tensors([5, 125, 126], query)
            q, gamma, positions, starts, frequency, args = tensors
            for _ in range(3):
                self.call(tensors, True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual, undo = self.call(tensors, True)
            for residue in range(4):
                args[1].copy_(
                    torch.tensor(
                        [4 + residue, 124 + residue, 128 + residue],
                        device="cuda",
                        dtype=torch.int32,
                    )
                )
                positions.copy_(
                    (args[1][:, None] + torch.arange(query, device="cuda")).flatten()
                )
                starts.copy_(positions - positions % 4)
                q.normal_()
                args[0].normal_()
                gamma.normal_(0, 0.1)
                args[4].normal_(0, 0.1)
                frequency.mul_(0.97)
                args[6][:, [0, 1]] = args[6][:, [1, 0]]
                args[8][:, [0, 1]] = args[8][:, [1, 0]]
                initial_kv, initial_state = args[5].clone(), args[7].clone()
                graph.replay()
                reference_kv, reference_state = args[5].clone(), args[7].clone()
                restore_indexer_cache(undo)
                torch.testing.assert_close(args[5], initial_kv, rtol=0, atol=0)
                torch.testing.assert_close(args[7], initial_state, rtol=0, atol=0)
                expected, before = self.call(tensors, False)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(args[5], reference_kv, rtol=0, atol=0)
                torch.testing.assert_close(args[7], reference_state, rtol=0, atol=0)
                restore_indexer_cache(before)
            graph.reset()

    def test_unsupported_layout_config_and_grad_decline_before_writes(self):
        tensors = list(self.tensors([7], 4))
        args = tensors[-1]
        original_kv, original_state = args[5].clone(), args[7].clone()
        for index in (5, 6, 7, 8):
            original = args[index]
            args[index] = original[:0] if index in (5, 7) else original[:, :0]
            self.assertIsNone(self.call(tensors, True))
            args[index] = original
        self.config.mrope_interleaved = False
        self.assertIsNone(self.call(tensors, True))
        self.config.mrope_interleaved = True
        tensors[0].requires_grad_()
        self.assertIsNone(self.call(tensors, True))
        torch.testing.assert_close(args[5], original_kv, rtol=0, atol=0)
        torch.testing.assert_close(args[7], original_state, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
