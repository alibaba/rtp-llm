"""FP8 selected-KV restoration, padding, repeatability and graph replay."""

from unittest import TestCase, main

import torch

from rtp_llm.models_py.kernels.ppu.mla_kv_quant import concat_and_cache_mla_fp8
from rtp_llm.models_py.kernels.ppu.sparse_mla_decode import sparse_mla_fp8_decode
from rtp_llm.models_py.utils.flash_mla_legacy import flash_mla_sparse_fwd


class SparseMlaDecodeTest(TestCase):
    def setUp(self):
        torch.manual_seed(1729)
        torch.set_num_threads(1)
        self.q = torch.randn(4, 64, 576, device="cuda").mul_(0.2).bfloat16()
        # Both block and row strides differ from contiguous 656-byte storage.
        self.cache = torch.zeros(9, 64, 1312, device="cuda", dtype=torch.uint8)[
            :, :, :656
        ]
        self.ckv = torch.randn(512, 512, device="cuda").bfloat16()
        self.kpe = torch.randn(512, 64, device="cuda").bfloat16()
        self.slots = torch.arange(64, 576, device="cuda", dtype=torch.int64)
        concat_and_cache_mla_fp8(self.ckv, self.kpe, self.cache, self.slots)
        self.indices = torch.full((4, 1, 2048), -1, device="cuda", dtype=torch.int32)
        self.indices[0, 0, :128] = self.slots[:128].int()
        self.indices[1, 0, :512] = self.slots.flip(0).int()
        self.indices[2, 0, :3] = torch.tensor([64, 575, 64], device="cuda")
        self.scale = 256**-0.5

    def reference(self):
        packed = self.cache.contiguous().view(-1, 656)
        values = packed[:, :512].contiguous().view(torch.float8_e4m3fn).float()
        scales = packed[:, 512:528].contiguous().view(torch.float32)
        kv = torch.cat(
            (
                (values.reshape(-1, 4, 128) * scales[:, :, None])
                .reshape(-1, 512)
                .bfloat16(),
                packed[:, 528:].contiguous().view(torch.bfloat16),
            ),
            dim=-1,
        ).unsqueeze(1)
        output = flash_mla_sparse_fwd(self.q, kv, self.indices, self.scale, d_v=512)[0]
        output.masked_fill_(~(self.indices >= 0).any(-1)[:, :, None], 0)
        return output

    def forward(self):
        return sparse_mla_fp8_decode(
            self.q, self.cache.unsqueeze(2), self.indices, self.scale
        )

    def test_restores_scaled_fp8_and_rope_with_ragged_empty_and_duplicate_keys(self):
        expected = self.reference()
        actual = self.forward()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(bool(torch.isfinite(actual).all()))
        for _ in range(32):
            self.assertTrue(torch.equal(self.forward(), expected))

    def test_graph_consumes_updated_queries_indices_and_cache(self):
        for _ in range(3):
            self.forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.forward()
        for factor in (0.5, 2.0):
            self.q.mul_(factor)
            concat_and_cache_mla_fp8(
                self.ckv * factor, self.kpe * factor, self.cache, self.slots
            )
            self.indices[0, 0, :128].copy_(self.slots[128:256].int())
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(output, self.reference()))
            for _ in range(8):
                previous = output.clone()
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(previous, output))

    def test_empty_queries_and_empty_selection(self):
        output = sparse_mla_fp8_decode(
            self.q[:0], self.cache, self.indices[:0], self.scale
        )
        self.assertEqual(output.shape, (0, 64, 512))
        output = sparse_mla_fp8_decode(
            self.q, self.cache, self.indices[:, :, :0], self.scale
        )
        self.assertEqual(torch.count_nonzero(output).item(), 0)

    def test_empty_mask_rejects_out_of_range_slots_and_changes_on_replay(self):
        self.indices.fill_(self.cache.shape[0] * self.cache.shape[1])
        self.assertEqual(torch.count_nonzero(self.forward()).item(), 0)
        for _ in range(3):
            self.forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.forward()
        for selected in [True, False, True]:
            self.indices.fill_(-1)
            if selected:
                self.indices[1, 0, :3] = self.slots[:3].int()
            graph.replay()
            self.assertTrue(torch.equal(actual, self.reference()))


if __name__ == "__main__":
    main()
