"""Native BF16 score boundaries, odd-width FP4 GEMM and Graph replay."""
import unittest

import torch
from rtp_llm.device.device_type import is_ppu


@unittest.skipUnless(is_ppu(), "requires a PPU")
class PpuFP4MqaLogitsTest(unittest.TestCase):
    def _inputs(self, rows, keys):
        torch.manual_seed(711)
        q = torch.randint(-128, 128, (rows, 64, 64), device="cuda", dtype=torch.int8)
        k = torch.randint(-128, 128, (keys, 64), device="cuda", dtype=torch.int8)
        qs = torch.full((rows, 64, 1), 0x7F7F7F7F, device="cuda", dtype=torch.int32)
        ks = torch.full((keys, 1), 0x7F7F7F7F, device="cuda", dtype=torch.int32)
        weights = torch.randn(rows, 64, device="cuda", dtype=torch.float32)
        starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
        ends = torch.full((rows,), keys, device="cuda", dtype=torch.int32)
        ends[0] = 0
        starts[-1] = keys // 2
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device="cuda", dtype=torch.bfloat16)

        def decode(packed):
            raw = packed.view(torch.uint8)
            codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2).long()
            return torch.where(codes < 8, lut[codes & 7], -lut[codes & 7])

        dot = torch.einsum("mhd,nd->mhn", decode(q), decode(k))
        return (q, qs), (k, ks), weights, starts, ends, dot

    def _reference(self, dot, weights, starts, ends):
        score = (dot.relu() * weights.bfloat16().unsqueeze(-1)).sum(dim=1)
        columns = torch.arange(score.shape[1], device=score.device)
        score.masked_fill_((columns[None] < starts[:, None]) | (columns[None] >= ends[:, None]), float("-inf"))
        selected = score.unsqueeze(0).topk(min(512, score.shape[1]), dim=-1).indices[0]
        out = torch.full((score.shape[0], 512), -1, device=score.device, dtype=torch.int32)
        out[:, :selected.shape[1]].copy_(selected.masked_fill((selected < starts[:, None]) | (selected >= ends[:, None]), -1))
        return score, out

    @torch.inference_mode()
    def test_odd_width_and_bounded_rows(self):
        from rtp_llm.models_py.kernels.ppu.attention.ppu_fp4_mqa_logits import fp8_fp4_mqa_logits, native_topk

        for rows, keys in [(2, 1), (7, 17), (7, 513), (2, 12288)]:
            with self.subTest(rows=rows, keys=keys):
                q, k, weights, starts, ends, dot = self._inputs(rows, keys)
                expected_score, expected_out = self._reference(dot, weights, starts, ends)
                score = fp8_fp4_mqa_logits(q, k, weights, starts, ends, logits_dtype=torch.bfloat16,
                                          scratch_bytes=2 * 64 * (keys + keys % 2))
                torch.testing.assert_close(score, expected_score, rtol=0, atol=0)
                self.assertTrue(torch.equal(native_topk(score, starts, ends), expected_out))

    @torch.inference_mode()
    def test_graph_changed_weights_and_ties(self):
        from rtp_llm.models_py.kernels.ppu.attention.ppu_fp4_mqa_logits import fp8_fp4_mqa_logits, native_topk

        q, k, weights, starts, ends, dot = self._inputs(7, 17)
        original = weights.clone()

        def run():
            score = fp8_fp4_mqa_logits(q, k, weights, starts, ends, logits_dtype=torch.bfloat16)
            return score, native_topk(score, starts, ends)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            score, out = run()
        for changed in [-original, torch.zeros_like(original), original]:
            weights.copy_(changed)
            graph.replay()
            expected_score, expected_out = self._reference(dot, weights, starts, ends)
            torch.testing.assert_close(score, expected_score, rtol=0, atol=0)
            self.assertTrue(torch.equal(out, expected_out))


if __name__ == "__main__":
    unittest.main()
