"""Real FP8 H8 fused Q-B vs current generic GEMM + Q-only split/RoPE.

Optional paired graph benchmark covers only Q-B/RoPE -> absorbed Wkc, not
whole CMP, Indexer, TRT, output projection, TP communication, or MoE.
Run on an explicitly reserved SM100/103 GPU with the pinned RTP wheel.
"""

import json
import math
import os
import statistics
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.hybrid.test.glm5_cmp_tp8_gpu_test import (
    _quantize_block_weight,
)
from rtp_llm.models_py.triton_kernels.sparse_mla import glm5_cmp_q_b as fused
from rtp_llm.models_py.triton_kernels.sparse_mla.glm5_cmp_q_rope import split_q_rope


class LoaderContractTest(unittest.TestCase):
    def test_capture_cannot_initialize_extension(self):
        fused._load_extension.cache_clear()
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Warm up"):
                fused._load_extension()

    def test_capture_cannot_initialize_another_device(self):
        fused._prepare_device.cache_clear()
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "on this device"):
                fused._prepare_device(0)

    def test_public_metadata_errors_do_not_initialize_cuda(self):
        operands = [torch.empty(4) for _ in range(8)]
        with patch.object(fused, "_prepare_device") as prepare:
            with self.assertRaisesRegex(ValueError, "is_neox_style"):
                fused.q_b_proj_h8(*operands[:6], out=operands[6:], is_neox_style=1)
            with self.assertRaisesRegex(ValueError, "alias"):
                fused.q_b_proj_h8(
                    *operands[:6],
                    out=(operands[0], operands[7]),
                    is_neox_style=False,
                    enable_pdl=True,
                )
            with self.assertRaisesRegex(ValueError, "activation must be CUDA"):
                fused.q_b_proj_h8(
                    *operands[:6],
                    out=operands[6:],
                    is_neox_style=False,
                    enable_pdl=True,
                )
            prepare.assert_not_called()


class Fixture:
    def __init__(self, rows, neox=False, positions_dtype=torch.int32):
        from rtp_kernel import glm5 as ops

        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        self.ops, self.rows, self.neox = ops, rows, neox
        generator = torch.Generator(device="cuda").manual_seed(941 + rows)
        self.generator = generator

        def random(shape, scale=1.0):
            return (
                torch.randn(shape, generator=generator, device="cuda") * scale
            ).bfloat16()

        self.input = random((rows, 2048))
        self.activation, self.scale = sgl_per_token_group_quant_fp8(
            self.input,
            group_size=128,
            eps=1.0e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        self.weight, self.weight_scale = _quantize_block_weight(
            random((2048, 2048), 1 / math.sqrt(2048))
        )
        self.wkc = random((8, 192, 512), 1 / math.sqrt(192))
        angles = torch.randn((1024, 32), generator=generator, device="cuda")
        self.cos_sin = torch.cat((angles.cos(), angles.sin()), 1)
        self.positions = torch.randint(
            1024, (rows,), dtype=positions_dtype, generator=generator, device="cuda"
        )
        self.projected = torch.empty((rows, 2048), device="cuda", dtype=torch.bfloat16)
        self.old_nope = torch.empty((rows, 8, 192), device="cuda", dtype=torch.bfloat16)
        self.old_query = torch.full(
            (rows, 8, 576), 13.0, device="cuda", dtype=torch.bfloat16
        )
        self.new_nope = torch.empty_like(self.old_nope)
        self.new_query = torch.full_like(self.old_query, 13.0)

    def old(self, absorb=True):
        self.ops.indexer_q_proj(
            self.activation,
            self.scale,
            self.weight,
            self.weight_scale,
            out=self.projected,
        )
        split_q_rope(
            self.projected,
            self.cos_sin,
            self.positions,
            q_nope_out=self.old_nope,
            q_out=self.old_query,
            is_neox_style=self.neox,
        )
        if absorb:
            self.ops.absorbed_q_nope_bmm(self.old_nope, self.wkc, out=self.old_query)

    def new(self, absorb=True, enable_pdl=True):
        fused.q_b_proj_h8(
            self.activation,
            self.scale,
            self.weight,
            self.weight_scale,
            self.cos_sin,
            self.positions,
            out=(self.new_nope, self.new_query),
            is_neox_style=self.neox,
            enable_pdl=enable_pdl,
        )
        if absorb:
            self.ops.absorbed_q_nope_bmm(self.new_nope, self.wkc, out=self.new_query)

    def assert_equal(self, case):
        torch.testing.assert_close(
            self.new_nope, self.old_nope, atol=0, rtol=0, msg=case
        )
        torch.testing.assert_close(
            self.new_query, self.old_query, atol=0, rtol=0, msg=case
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires reserved SM100/103 GPU")
class FusedH8Test(unittest.TestCase):
    def test_projection_prefix_and_wkc_bitexact(self):
        for rows in (1, 2, 4, 6, 8, 16, 17, 24, 32, 63, 64, 65, 128, 129, 256):
            for neox in (False, True):
                for dtype in (torch.int32, torch.int64):
                    with self.subTest(rows=rows, neox=neox, dtype=dtype):
                        fixture = Fixture(rows, neox, dtype)
                        fixture.old(absorb=False)
                        fixture.new(absorb=False)
                        fixture.assert_equal(
                            "projection including untouched latent prefix"
                        )
                        self.assertTrue(
                            torch.all(fixture.new_query[..., :512] == 13).item()
                        )
                        fixture.old()
                        fixture.new()
                        fixture.assert_equal("absorbed Wkc, PDL on")
                        original_pdl = fixture.ops.get_pdl()
                        try:
                            fixture.ops.set_pdl(False)
                            fixture.new(enable_pdl=False)
                            fixture.assert_equal("absorbed Wkc, PDL off")
                        finally:
                            fixture.ops.set_pdl(original_pdl)

    def test_graph_changed_inputs_positions_and_side_stream(self):
        fixture = Fixture(8, True, torch.int64)
        fixture.old()
        fixture.new()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fixture.old()
            fixture.new()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                fixture.new()
            torch.cuda.current_stream().wait_stream(side)
        for iteration in range(3):
            fixture.positions.add_(11).remainder_(1024)
            # Same addresses, genuinely changed FP8 data and scale values.
            payload = fixture.activation.view(torch.uint8)
            payload.copy_(torch.roll(payload, iteration + 1, 1))
            if iteration == 1:
                fixture.scale.copy_(torch.roll(fixture.scale, 1, 1))
            fixture.old()
            graph.replay()
            fixture.assert_equal(f"changed-input replay {iteration}")

    def test_reject_bad_shapes_dtype_alias(self):
        fixture = Fixture(4)
        fixture.new()
        arguments = (
            fixture.activation,
            fixture.scale,
            fixture.weight,
            fixture.weight_scale,
            fixture.cos_sin,
            fixture.positions,
        )
        with self.assertRaisesRegex((ValueError, RuntimeError), "alias"):
            fused.q_b_proj_h8(
                *arguments,
                out=(fixture.new_query[..., :192], fixture.new_query),
                is_neox_style=False,
            )
        with self.assertRaisesRegex(RuntimeError, "scales"):
            fused.q_b_proj_h8(
                fixture.activation,
                fixture.scale.contiguous(),
                *arguments[2:],
                out=(fixture.new_nope, fixture.new_query),
                is_neox_style=False,
            )
        with self.assertRaisesRegex(RuntimeError, "weight"):
            fused.q_b_proj_h8(
                *arguments[:2],
                fixture.weight[:1024],
                *arguments[3:],
                out=(fixture.new_nope, fixture.new_query),
                is_neox_style=False,
            )

    @unittest.skipUnless(
        os.getenv("GLM5_CMP_QB_H8_BENCHMARK") == "1", "benchmark opt-in"
    )
    def test_paired_qb_wkc_graph_benchmark(self):
        records = []
        for rows in (1, 2, 4, 6, 8, 16, 64, 256):
            fixture = Fixture(rows)
            for _ in range(3):
                fixture.old()
                fixture.new()
            fixture.assert_equal("benchmark warmup")
            graphs = []
            for forward in (fixture.old, fixture.new):
                graph = torch.cuda.CUDAGraph()
                start, end = (
                    torch.cuda.Event(enable_timing=True, external=True)
                    for _ in range(2)
                )
                with torch.cuda.graph(graph):
                    start.record()
                    forward()
                    end.record()
                graphs.append((graph, start, end))
            # First graph instantiation/replay is not steady-state inference.
            for _ in range(10):
                for graph, _, _ in graphs:
                    graph.replay()
            torch.cuda.synchronize()
            samples = [[], []]
            for sample in range(31):
                for index in (0, 1) if sample % 2 == 0 else (1, 0):
                    graph, start, end = graphs[index]
                    graph.replay()
                    end.synchronize()
                    samples[index].append(start.elapsed_time(end) * 1000)
            fixture.assert_equal("benchmark graph")
            medians = [statistics.median(values) for values in samples]
            records.append(
                {
                    "rows": rows,
                    "heads": 8,
                    "old_median_us": medians[0],
                    "fused_median_us": medians[1],
                    "speedup": medians[0] / medians[1],
                    "old_samples_us": samples[0],
                    "fused_samples_us": samples[1],
                }
            )
        print("Q_B_WKC_LOCAL_GRAPH " + json.dumps(records), flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
