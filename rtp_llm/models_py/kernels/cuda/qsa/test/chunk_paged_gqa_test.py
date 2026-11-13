import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha import sparse_paged_gqa_attn
from rtp_llm.models_py.kernels.cuda.qsa.test.reference.sparse_paged import _reference


class ChunkPagedGqaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("SM120 is required")

    def fixture(self, fp8=False, heads=3, queries=17):
        torch.manual_seed(1773)
        b, h, s, d = 2, heads, queries, 256
        q = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16)
        cache = torch.randn(7, 2, 1, 128, d, device="cuda", dtype=torch.bfloat16)
        table = torch.tensor([[3, 6, 1], [5, 2, 4]], device="cuda", dtype=torch.int32)
        lengths = torch.full((b, s), 367, device="cuda", dtype=torch.int32)
        lengths[1, -1] = 0
        selected = torch.full((b, s, 2051), -1, device="cuda", dtype=torch.int32)
        selected[..., :11] = torch.arange(11, device="cuda", dtype=torch.int32) * 31
        selected[1, -1] = -1
        scale = None
        if fp8:
            cache = cache.to(torch.float8_e4m3fn)
            scale = torch.rand(7, 2, 1, 128, device="cuda") * 0.8 + 0.2
        return q, cache, table, lengths, selected, scale

    def run_attention(self, args, **kwargs):
        q, cache, table, lengths, selected, scale = args
        return sparse_paged_gqa_attn(
            q, cache, table, lengths, selected, page_size=128, kv_scale=scale, **kwargs
        )

    def packed_float64_reference(self, args):
        q, cache, table, lengths, selected, scale = args
        result = torch.zeros_like(q, dtype=torch.float64)
        for b in range(q.shape[0]):
            for s in range(q.shape[2]):
                # This fixture has one KV head, shared by all query heads.
                ids = selected[b, s, :-1]
                ids = ids[(ids >= 0) & (ids < lengths[b, s])].long()
                if not ids.numel():
                    continue
                page, offset = table[b, ids // 128].long(), ids % 128
                keys = cache[page, 0, 0, offset].double()
                values = cache[page, 1, 0, offset].double()
                if scale is not None:
                    keys *= scale[page, 0, 0, offset, None].double()
                    values *= scale[page, 1, 0, offset, None].double()
                scores = q[b, :, s].double() @ keys.T / q.shape[-1] ** 0.5
                result[b, :, s] = scores.softmax(-1) @ values
        return result

    def test_grouped_chunk_matches_independent_float_reference(self):
        for fp8 in (False, True):
            args = self.fixture(fp8)
            q, cache, table, lengths, selected, scale = args
            decoded = cache.float()
            if scale is not None:
                decoded = decoded * scale[..., None]
            expected = _reference(q, decoded, table, lengths, selected)
            actual = self.run_attention(args)
            # Keep a BF16-relative bound plus an absolute zero floor.
            torch.testing.assert_close(actual, expected, rtol=8e-3, atol=2e-3)
            with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
                legacy = self.run_attention(args)
            torch.testing.assert_close(actual, legacy, rtol=8e-3, atol=2e-3)
            torch.testing.assert_close(
                actual[1, :, -1], torch.zeros_like(actual[1, :, -1])
            )

    def test_packed_count_gate_and_replay_consume_live_metadata(self):
        args = list(self.fixture(True))
        q, cache, table, lengths, selected, scale = args
        counts = (selected >= 0).sum(-1, keepdim=True).int()
        args[4] = torch.cat([selected, counts], dim=-1)
        gate = torch.randn(2, 17, 3, 256, device="cuda", dtype=torch.bfloat16)
        options = dict(selected_has_count=True, output_gate=gate)
        for _ in range(3):
            self.run_attention(args, **options)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.run_attention(args, graph_capture=True, **options)
        for gain in (0.25, 1.0, 2.0):
            q.normal_().mul_(gain)
            scale.uniform_(0.2, 1.0)
            graph.replay()
            # Graph replay must exactly match the same single-PV eager path.
            expected = self.run_attention(args, **options)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # Check numerical accuracy independently, rather than comparing
            # two differently rounded kernels. Match the BF16 output/gate
            # boundaries, but compute attention itself in FP64.
            reference = self.packed_float64_reference(args).bfloat16().float()
            reference *= gate.transpose(1, 2).float().sigmoid().bfloat16().float()
            torch.testing.assert_close(actual.float(), reference, rtol=8e-3, atol=2e-3)
        args[4][..., :-1].fill_(-1)
        args[4][..., -1].zero_()
        lengths.zero_()
        graph.replay()
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
        graph.reset()

    def test_trusted_path_skips_redundant_value_scan_and_keeps_shape_checks(self):
        args = self.fixture(True)
        expected = self.run_attention(args)
        with patch(
            "rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha._validate_sparse_paged_indices",
            side_effect=AssertionError("redundant selected scan"),
        ):
            actual = self.run_attention(args, validate_selected=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        invalid_shape = list(args)
        invalid_shape[3] = invalid_shape[3][:, :-1]
        with self.assertRaisesRegex(ValueError, "kv_lens must"):
            self.run_attention(invalid_shape, validate_selected=False)
        invalid_value = list(args)
        invalid_value[4] = invalid_value[4].clone()
        invalid_value[4][0, 0, 0] = 367
        with self.assertRaisesRegex(ValueError, "outside its row"):
            self.run_attention(invalid_value)

    def test_single_pv_pipeline_replay_crosses_tile_boundaries(self):
        from rtp_llm.models_py.kernels.cuda.qsa.chunk_paged_gqa import chunk_paged_gqa

        args = list(self.fixture(True, heads=12, queries=129))
        q, cache, table, lengths, selected, scale = args
        counts = (
            torch.tensor(
                [0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2048, 2051],
                device="cuda",
                dtype=torch.int32,
            )
            .repeat(20)[: 2 * 129]
            .reshape(2, 129)
        )
        slots = torch.arange(2051, device="cuda", dtype=torch.int32)
        args[4] = torch.empty((2, 129, 2052), device="cuda", dtype=torch.int32)
        gate = torch.randn(2, 129, 12, 256, device="cuda", dtype=torch.bfloat16)

        def refresh(shift):
            active = counts.roll(shift, dims=1).clone()
            active[1, -1] = 0
            args[4][..., :-1] = torch.where(
                slots < active[..., None], (slots * 31) % 367, -1
            )
            args[4][..., -1] = active

        for block_n, warps, stages in ((32, 4, 3), (64, 2, 1)):
            with self.subTest(block_n=block_n, warps=warps, stages=stages):
                refresh(0)

                def run(output_gate=gate):
                    return chunk_paged_gqa(
                        q,
                        cache,
                        table,
                        lengths,
                        args[4],
                        scale=scale,
                        selected_has_count=True,
                        output_gate=output_gate,
                        block_n=block_n,
                        num_warps=warps,
                        num_stages=stages,
                    )

                run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = run()
                for shift in (1, 7):
                    refresh(shift)
                    scale.uniform_(0.2, 1.0)
                    graph.replay()
                    torch.testing.assert_close(actual, run(), rtol=0, atol=0)
                    # Validate attention accuracy before the extra BF16 gate
                    # rounding, then check fusion against separate torch ops.
                    ungated = run(output_gate=None)
                    reference = self.packed_float64_reference(args)
                    torch.testing.assert_close(
                        ungated.float(), reference.float(), rtol=8e-3, atol=2e-3
                    )
                    torch.testing.assert_close(
                        actual,
                        ungated * torch.sigmoid(gate.transpose(1, 2)),
                        rtol=0,
                        atol=0,
                    )
                args[4][..., :-1].fill_(-1)
                args[4][..., -1].zero_()
                graph.replay()
                torch.testing.assert_close(
                    actual, torch.zeros_like(actual), rtol=0, atol=0
                )
                graph.reset()

    def test_long_fp8_prefill_graph_refreshes_tuned_default(self):
        args = list(self.fixture(True, heads=12, queries=1024))
        q, cache, table, lengths, selected, scale = args
        selected[..., :2048] = torch.arange(2048, device="cuda") % 367
        selected[1, -1] = -1
        counts = (selected >= 0).sum(-1, keepdim=True).int()
        args[4] = torch.cat([selected, counts], dim=-1)
        self.run_attention(args, selected_has_count=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.run_attention(
                args, selected_has_count=True, graph_capture=True
            )
        rows = torch.tensor([0, 31, 511, 1023], device="cuda")
        for gain in (0.5, 1.5):
            q.normal_().mul_(gain)
            scale.uniform_(0.2, 1.0)
            table.copy_(table.flip(1))
            graph.replay()
            expected = self.run_attention(args, selected_has_count=True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            sampled = [
                q.index_select(2, rows),
                cache,
                table,
                lengths.index_select(1, rows),
                args[4].index_select(1, rows),
                scale,
            ]
            reference = self.packed_float64_reference(sampled)
            torch.testing.assert_close(
                actual.index_select(2, rows).float(),
                reference.float(),
                rtol=8e-3,
                atol=2e-3,
            )
        graph.reset()


if __name__ == "__main__":
    unittest.main()
