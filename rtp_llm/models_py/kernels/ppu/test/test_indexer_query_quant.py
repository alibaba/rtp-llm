"""Match the actual SGL DSA quantizer, including FP8 rounding boundaries."""

import json
import os
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla import fused_q_rope_quant as kernel


class IndexerQueryQuantTest(TestCase):
    @classmethod
    def setUpClass(cls):
        data = json.loads(
            Path(__file__).with_name("test_indexer_query_sgl_golden.json").read_text()
        )

        def tensor(name, dtype):
            return torch.tensor(data[name], dtype=dtype, device="cuda")

        cls.q = tensor("q_u16", torch.uint16).view(torch.bfloat16)
        cls.k = tensor("k_u16", torch.uint16).view(torch.bfloat16)
        cls.expected_q = tensor("expected_q_u8", torch.uint8)
        cls.expected_s = tensor("expected_scale_u32", torch.uint32).view(torch.float32)
        cls.expected_k = tensor("expected_k_u16", torch.uint16).view(torch.bfloat16)
        cls.positions = torch.zeros(len(cls.q), dtype=torch.int32, device="cuda")
        cls.cache = torch.cat((torch.ones(1, 32), torch.zeros(1, 32)), dim=1).cuda()

    def qk(self, q, k, scale_fmt="none"):
        return kernel.fused_qk_rope_quant(
            q,
            k,
            self.positions[: len(q)],
            self.cache,
            1,
            128,
            64,
            is_neox_style=False,
            scale_fmt=scale_fmt,
        )

    def test_both_query_entrypoints_match_sgl_for_both_dispatch_modes(self):
        for mode in ("legacy", "v2"):
            with self.subTest(mode=mode), patch.dict(
                os.environ, {"DSV4_FP8_QUANT_KERNEL": mode}
            ):
                q, s, k = self.qk(self.q, self.k)
                self.assertTrue(torch.equal(q.view(torch.uint8), self.expected_q))
                self.assertTrue(torch.equal(s, self.expected_s))
                self.assertTrue(torch.equal(k, self.expected_k))
                q_only, s_only = kernel.fused_q_rope_quant(
                    self.q,
                    self.positions,
                    self.cache,
                    1,
                    128,
                    64,
                    is_neox_style=False,
                    scale_fmt="none",
                )
                self.assertTrue(torch.equal(q_only.view(torch.uint8), self.expected_q))
                self.assertTrue(torch.equal(s_only, self.expected_s))

    def test_graph_replay_reads_new_values_on_nondefault_stream(self):
        q, k = self.q[:4].clone(), self.k[:4].clone()
        for _ in range(3):
            self.qk(q, k)
        torch.cuda.synchronize()
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):
            outputs = self.qk(q, k)
        for multiplier in (2, 4):
            q.copy_(self.q[:4] * multiplier)
            k.copy_(self.k[:4] * multiplier)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(
                torch.equal(outputs[0].view(torch.uint8), self.expected_q[:4])
            )
            self.assertTrue(torch.equal(outputs[1], self.expected_s[:4] * multiplier))
            self.assertTrue(torch.equal(outputs[2], self.expected_k[:4] * multiplier))

    def test_ue8m0_keeps_the_existing_backend_math(self):
        for mode in ("legacy", "v2"):
            with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": mode}):
                actual = self.qk(self.q, self.k, "ue8m0")
                with patch.object(kernel, "is_ppu", return_value=False):
                    previous = self.qk(self.q, self.k, "ue8m0")
                for a, b in zip(actual, previous):
                    self.assertTrue(
                        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    )

    def test_empty_inputs_have_valid_shapes(self):
        q, s, k = self.qk(self.q[:0], self.k[:0])
        self.assertEqual(q.shape, (0, 1, 128))
        self.assertEqual(s.shape, (0, 1, 1))
        self.assertEqual(k.shape, (0, 128))


if __name__ == "__main__":
    main()
