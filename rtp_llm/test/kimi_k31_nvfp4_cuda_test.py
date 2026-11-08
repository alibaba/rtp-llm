"""Native FP4 Linear: online weights, dynamic activations and Graph replay."""

import unittest
from types import SimpleNamespace as NS

import torch

from rtp_llm.models.kimi_k3.nvfp4_tensor import (
    OnlineNvfp4Projection,
    nvfp4_global_scale,
)


def decode(codes, swizzled, outer):
    n, k = codes.shape[0], codes.shape[1] * 2
    rows, columns = (n + 127) // 128 * 128, (k // 16 + 3) // 4 * 4
    scales = swizzled.view(torch.float8_e4m3fn).reshape(
        rows // 128, columns // 4, 32, 4, 4
    )
    scales = scales.permute(0, 3, 2, 1, 4).reshape(rows, columns)[:n, : k // 16]
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=codes.device,
    )
    indices = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2).long()
    return table[indices] * scales.float().repeat_interleave(16, -1) * outer


def flashinfer_dynamic_quant(input):
    from flashinfer import fp4_quantize
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import Fp4Activation

    dense = input.contiguous()
    if dense.data_ptr() % 16:
        dense = dense.clone(memory_format=torch.contiguous_format)
    scale = nvfp4_global_scale(dense)
    values, block_scales = fp4_quantize(dense, scale.reciprocal())
    return Fp4Activation(values, block_scales, scale)


@unittest.skipUnless(
    torch.cuda.is_available(), "requires selected CUDA13 SM100/SM103 host"
)
class Nvfp4KernelReplayTest(unittest.TestCase):
    def test_decode_attn_res_fuses_moe_output_add(self):
        from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual
        from rtp_llm.models_py.triton_kernels.moe.output_add import add_moe_output

        def rand(*shape):
            return torch.randn(shape, device="cuda", dtype=torch.bfloat16)

        width = 7168
        residual = KimiK3AttentionResidual(rand(width), rand(width), 1e-6)
        norm = rand(width)
        for mode in ("bf16", "fp8", "fp4"):
            for write in (-1, 2):
                with self.subTest(mode=mode, write=write):
                    prefix, routed, shared = (rand(32, width) for _ in range(3))
                    anchors = rand(32, 3, width)
                    original = add_moe_output(routed, shared, prefix)
                    expected_bank = anchors.clone()
                    kwargs = dict(
                        num_blocks=2, block_write_idx=write,
                        output_norm_weight=norm, output_norm_eps=1e-6,
                    )
                    fn = residual.forward_fp8 if mode == "fp8" else residual
                    if mode == "fp4":
                        kwargs.update(quantize_nvfp4=True, rowwise_nvfp4=True)
                    expected = fn(original, expected_bank, **kwargs)
                    actual = fn(prefix, anchors, delta=routed, shared_delta=shared, **kwargs)
                    self.assertTrue(torch.equal(prefix, original))
                    self.assertTrue(torch.equal(anchors, expected_bank))
                    if mode == "fp8":
                        self.assertTrue(torch.equal(
                            actual.values.view(torch.uint8), expected.values.view(torch.uint8),
                        ))
                        self.assertTrue(torch.equal(actual.scale_wire, expected.scale_wire))
                    elif mode == "fp4":
                        self.assertTrue(torch.equal(actual.wire, expected.wire))
                    else:
                        self.assertTrue(torch.equal(actual, expected))

    def projection(self, n=64, k=64, module="dsa.input", dynamic=True):
        from rtp_llm.config.quant_config import ModelOptFp4Config
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import (
            CudaFp4GEMMLinear,
        )

        original = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / 8
        p = OnlineNvfp4Projection(NS(name="w"), module=module)
        tensors = p._postprocess({"w": original}, "cuda", None)
        linear = CudaFp4GEMMLinear(
            weight=tensors["w"],
            weight_scales=tensors[p.local_key],
            weight_scale_2=tensors[p.global_key],
            input_scales=None if dynamic else torch.tensor([0.01], device="cuda"),
            quant_config=ModelOptFp4Config(
                bits=4,
                group_size=16,
                is_quanted=False,
                activation_scheme="dynamic" if dynamic else "static",
            ),
        )
        weight = decode(tensors["w"], tensors[p.local_key], tensors[p.global_key])
        quant_error = (weight[:n] - original.T.float()).norm() / original.float().norm()
        self.assertLess(quant_error.item(), 0.15)
        return linear, weight, int(tensors[p.logical_n_key])

    def check(self, linear, weight, x, actual=None):
        from flashinfer import fp4_quantize

        outer = (
            nvfp4_global_scale(x) if linear.dynamic_input_scale else linear.input_scale
        )
        codes, scales = fp4_quantize(x.contiguous(), outer.reciprocal())
        activation = decode(codes, scales, outer)
        quant_error = (activation - x.float()).norm() / x.float().norm().clamp_min(1e-8)
        self.assertLess(quant_error.item(), 0.15)
        expected = activation @ weight.T
        actual = linear(x) if actual is None else actual
        relative = (actual.float() - expected).norm() / expected.norm().clamp_min(1e-8)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLess(relative.item(), 0.005)
        self.assertEqual(actual.dtype, torch.bfloat16)

    def test_dynamic_strided_zero_and_empty_inputs(self):
        linear, weight, _ = self.projection()
        for m in (1, 7, 33, 128):
            x = torch.randn(m, 128, device="cuda", dtype=torch.bfloat16)[:, ::2]
            self.check(linear, weight, x)
        self.check(
            linear, weight, torch.zeros(7, 64, device="cuda", dtype=torch.bfloat16)
        )
        self.check(
            linear, weight, torch.randn(7, 64, device="cuda", dtype=torch.float16)
        )
        self.assertEqual(
            linear(torch.empty(0, 64, device="cuda", dtype=torch.bfloat16)).shape,
            (0, 64),
        )
        self.assertEqual(
            linear(torch.empty(0, 64, device="cuda", dtype=torch.float16)).dtype,
            torch.bfloat16,
        )

    def test_kda_forget_strided_quantization_and_preallocated_gemm(self):
        from rtp_llm.models_py.triton_kernels.common.nvfp4 import quantize_strided_nvfp4

        linear, weight, _ = self.projection(k=128)
        for rows, stride, offset, magnitude in (
            (1, 420, 13, 0),
            (7, 432, 16, 1),
            (129, 421, 13, 64),
            (8193, 421, 13, 1),
        ):
            with self.subTest(rows=rows, stride=stride, offset=offset, magnitude=magnitude):
                source = torch.randn(rows, stride, device="cuda", dtype=torch.bfloat16)
                view = source[:, offset : offset + 128].mul_(magnitude)
                reference = flashinfer_dynamic_quant(view)
                actual = quantize_strided_nvfp4(view)
                torch.testing.assert_close(actual.scale, reference.scale, atol=0, rtol=0)
                self.assertTrue(torch.equal(actual.values, reference.values))
                self.assertTrue(torch.equal(actual.block_scales, reference.block_scales))
                output = torch.empty((rows * 2, linear.output_size), device="cuda",
                                     dtype=torch.bfloat16).narrow(0, rows, rows)
                projected = linear.forward_quantized(actual, out=output)
                self.assertEqual(projected.data_ptr(), output.data_ptr())
                self.check(linear, weight, view.clone(memory_format=torch.contiguous_format), projected)

    def test_kda_forget_strided_quantization_graph_replay(self):
        from rtp_llm.models_py.triton_kernels.common.nvfp4 import quantize_strided_nvfp4

        source = torch.randn(129, 421, device="cuda", dtype=torch.bfloat16)
        view = source[:, 13:141]
        quantize_strided_nvfp4(view)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = quantize_strided_nvfp4(view)

        view.mul_(-2)
        graph.replay()
        reference = flashinfer_dynamic_quant(view)
        torch.testing.assert_close(actual.scale, reference.scale, atol=0, rtol=0)
        self.assertTrue(torch.equal(actual.values, reference.values))
        self.assertTrue(torch.equal(actual.block_scales, reference.block_scales))

    def test_decode_dynamic_quant_matches_flashinfer(self):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import CudaFp4GEMMLinear

        for rows, width, extra, offset, magnitude in (
            (1, 64, 13, 1, 0),
            (32, 7168, 0, 0, 1),
            (255, 7168, 9, 1, 1),
            (256, 1536, 16, 8, 4),
            (129, 1536, 7, 3, 0),
            (256, 7168, 0, 0, 1),
        ):
            with self.subTest(rows=rows, width=width, offset=offset, magnitude=magnitude):
                source = torch.randn(rows, width + extra, device="cuda", dtype=torch.bfloat16)
                view = source[:, offset:offset + width].mul_(magnitude)
                expected = flashinfer_dynamic_quant(view)
                actual = CudaFp4GEMMLinear.quantize_dynamic(view)
                torch.testing.assert_close(actual.scale, expected.scale, rtol=0, atol=0)
                self.assertTrue(torch.equal(actual.values, expected.values))
                self.assertTrue(torch.equal(actual.block_scales, expected.block_scales))

    def test_decode_wide_dynamic_quant_graph_replay(self):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import CudaFp4GEMMLinear

        for width in (1536, 7168):
            with self.subTest(width=width):
                source = torch.randn(256, width + 16, device="cuda", dtype=torch.bfloat16)
                view = source[:, 8:8 + width]
                CudaFp4GEMMLinear.quantize_dynamic(view)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = CudaFp4GEMMLinear.quantize_dynamic(view)
                view.mul_(-2)
                view[-1, -1] = 19
                graph.replay()
                expected = flashinfer_dynamic_quant(view)
                torch.testing.assert_close(actual.scale, expected.scale, rtol=0, atol=0)
                self.assertTrue(torch.equal(actual.values, expected.values))
                self.assertTrue(torch.equal(actual.block_scales, expected.block_scales))

    def test_kda_attn_res_quantizes_directly_into_the_ag_packet(self):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_comm_fusion import (
            _pack_nvfp4,
        )
        from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual

        def rand(*shape):
            return torch.randn(shape, device="cuda", dtype=torch.bfloat16)

        width = 7168
        linear, _, _ = self.projection(k=width, module="kda.input")
        residual = KimiK3AttentionResidual(rand(width), rand(width), 1e-6)
        norm = rand(width)
        for rows, blocks, write, has_delta, magnitude in (
            (1, 0, 0, False, 0),
            (129, 0, 0, False, 1),
            (257, 2, -1, True, 1),
            (513, 8, 8, True, 1),
            (8192, 1, -1, True, 1),
        ):
            with self.subTest(rows=rows, blocks=blocks, write=write, has_delta=has_delta):
                prefix = rand(rows, width).mul_(magnitude)
                anchors = rand(rows, max(blocks, write + 1, 1), width)
                kwargs = dict(
                    delta=rand(rows, width) if has_delta else None,
                    output_norm_weight=norm,
                    output_norm_eps=1e-6,
                    num_blocks=blocks,
                    block_write_idx=write,
                )
                expected_prefix, expected_anchors = prefix.clone(), anchors.clone()
                normalized = residual(expected_prefix, expected_anchors, **kwargs)
                reference = flashinfer_dynamic_quant(normalized)
                reference_wire, _ = _pack_nvfp4(reference)
                expected = linear.forward_quantized(reference)
                for split in (False, True):
                    actual_prefix, actual_anchors = prefix.clone(), anchors.clone()
                    packet = residual(
                        actual_prefix, actual_anchors, quantize_nvfp4=True,
                        split_nvfp4_quant=split, **kwargs,
                    )
                    self.assertTrue(torch.equal(actual_prefix, expected_prefix))
                    self.assertTrue(torch.equal(actual_anchors, expected_anchors))
                    self.assertTrue(torch.equal(packet.wire, reference_wire), f"split={split}")
                    actual = linear.forward_quantized(packet.unpack(packet.wire))
                    self.assertTrue(torch.equal(actual, expected))

    def test_dsa_attn_res_produces_bf16_fp8_and_nvfp4_with_original_rounding(self):
        from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_comm_fusion import _pack_nvfp4

        def rand(*shape):
            return torch.randn(shape, device="cuda", dtype=torch.bfloat16)

        rows, width = 259, 7168
        residual = KimiK3AttentionResidual(rand(width), rand(width), 1e-6)
        prefix, anchors = rand(rows, width), rand(rows, 2, width)
        kwargs = dict(
            output_norm_weight=rand(width), output_norm_eps=1e-6,
            delta=rand(rows, width), num_blocks=2,
        )
        expected, expected_fp8 = residual(prefix.clone(), anchors.clone(), also_quantize=True, **kwargs)
        reference, _ = _pack_nvfp4(flashinfer_dynamic_quant(expected))
        for split in (False, True):
            actual, actual_fp8, fp4 = residual(
                prefix.clone(), anchors.clone(), also_quantize=True, quantize_nvfp4=True,
                split_nvfp4_quant=split, **kwargs,
            )
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(
                actual_fp8.values.view(torch.uint8), expected_fp8.values.view(torch.uint8),
            ))
            self.assertTrue(torch.equal(actual_fp8.scale_wire[:, :rows], expected_fp8.scale_wire[:, :rows]))
            self.assertTrue(torch.equal(fp4.wire, reference), f"split={split}")

    def test_kda_rowwise_attn_res_packet_and_prepared_gemm(self):
        from flashinfer import fp4_quantize
        from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual

        def rand(*shape):
            return torch.randn(shape, device="cuda", dtype=torch.bfloat16)

        width = 7168
        residual = KimiK3AttentionResidual(rand(width), rand(width), 1e-6)
        linear, weight, _ = self.projection(k=width, module="kda.input")
        norm = rand(width)

        def group_scales(scales, rows):
            return scales.view(torch.uint8).reshape(
                (rows + 127) // 128, width // 64, 32, 4, 4,
            ).permute(0, 3, 2, 1, 4).reshape(-1, width // 16)

        for rows, blocks, write in ((1, 0, 0), (129, 0, 0), (257, 2, -1), (513, 8, 8)):
            with self.subTest(rows=rows, blocks=blocks, write=write):
                prefix = rand(rows, width)
                prefix[0].zero_()
                prefix[-1].mul_(7)
                anchors = rand(rows, max(1, blocks, write + 1), width)
                delta = rand(rows, width) if blocks else None
                args = dict(output_norm_weight=norm, output_norm_eps=1e-6,
                            num_blocks=blocks, block_write_idx=write, delta=delta)
                ref_prefix, ref_anchors = prefix.clone(), anchors.clone()
                expected = residual(ref_prefix, ref_anchors, **args)
                packet = residual(prefix, anchors, quantize_nvfp4=True, rowwise_nvfp4=True, **args)
                activation = packet.unpack(packet.wire)
                self.assertTrue(torch.equal(prefix, ref_prefix))
                self.assertTrue(torch.equal(anchors, ref_anchors))
                scale = expected.abs().amax(dim=-1).float().clamp_min(1e-8) * (1.0 / 2688.0)
                torch.testing.assert_close(activation.scale, scale, rtol=0, atol=0)
                actual_scales = group_scales(activation.block_scales, rows)
                self.assertEqual(torch.count_nonzero(actual_scales[rows:]).item(), 0)
                for row in {0, min(rows - 1, 17), rows - 1}:
                    values, scales = fp4_quantize(
                        expected[row:row + 1].contiguous(), scale[row:row + 1].reciprocal(),
                    )
                    self.assertTrue(torch.equal(activation.values[row:row + 1], values))
                    self.assertTrue(torch.equal(actual_scales[row], group_scales(scales, 1)[0]))
                replay = decode(
                    activation.values, activation.block_scales, activation.scale[:, None],
                ) @ weight.T
                out = expected.new_empty((rows, linear.output_size))
                launch, = linear.prepare_quantized_gemms(activation, out, num_streams=1)
                actual = linear.forward_quantized(activation, out=out, gemm=launch)
                relative = (actual.float() - replay).norm() / replay.norm().clamp_min(1e-8)
                self.assertLess(relative.item(), 0.005)

    def test_dsa_q_norm_shares_fp8_encoding_and_nvfp4_partials(self):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import sgl_per_token_group_quant_fp8
        from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_fp8_quant_triton import rmsnorm_fp8_quant_ue8m0
        from rtp_llm.models_py.triton_kernels.common.layernorm_gated import independent_q_kv_rmsnorm
        from rtp_llm.models_py.triton_kernels.common.nvfp4 import quantize_nvfp4_from_partials

        rows, width = 129, 1536
        x = torch.randint(-15, 16, (rows, 2048), device="cuda").bfloat16()[:, :width]
        weight = torch.randn(width, dtype=torch.bfloat16, device="cuda")
        expected = independent_q_kv_rmsnorm(
            x, weight, 1e-6, q_width=width, kv_width=512, full_rows=rows,
        )
        partials = torch.empty((rows + 3) // 4, device="cuda", dtype=torch.float32)
        actual, fp8, fp8_scale = rmsnorm_fp8_quant_ue8m0(x, weight, eps=1e-6, amax_partials=partials)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(partials.max(), expected.abs().amax().float(), atol=0, rtol=0)
        expected_fp8, expected_scale = sgl_per_token_group_quant_fp8(
            expected, group_size=128, eps=1e-4, column_major_scales=True,
            scale_tma_aligned=True, scale_ue8m0=True,
        )
        self.assertTrue(torch.equal(fp8.view(torch.uint8), expected_fp8.view(torch.uint8)))
        self.assertTrue(torch.equal(fp8_scale, expected_scale))
        linear, _, _ = self.projection(k=width)
        fp4 = quantize_nvfp4_from_partials(actual, partials)
        reference = linear.quantize_dynamic(expected)
        self.assertTrue(torch.equal(fp4.values, reference.values))
        self.assertTrue(torch.equal(fp4.block_scales.view(-1), reference.block_scales.view(-1)))
        torch.testing.assert_close(fp4.scale, reference.scale, atol=0, rtol=0)

    def test_dsa_nvfp4_pull_rebases_misaligned_scale_rows(self):
        from flashinfer import fp4_quantize
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_comm_fusion import (
            _pack_nvfp4, PackedNvfp4Activation,
        )
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import Fp4Activation
        from rtp_llm.models_py.modules.kimi_k3.natural_dsa import (
            NaturalPairedProjection, split_natural_projection,
        )
        from rtp_llm.models_py.modules.hybrid.natural_zigzag_plan import RowSegment
        from rtp_llm.models_py.triton_kernels.common.nvfp4 import pull_nvfp4_rows

        width = 7168
        linear, _, _ = self.projection(n=416, k=width)
        q, _, gate = split_natural_projection(linear, 64, 96, 256, 0)
        paired = NaturalPairedProjection(q, gate)
        intervals = ((3, 131), (129, 1), (128, 129))
        total_rows = sum(rows for _, rows in intervals)
        packet = PackedNvfp4Activation.empty(total_rows, width, linear.weight.device)
        values, block_scales, _ = packet.unpack(packet.wire)
        target = Fp4Activation(
            values, block_scales, values.new_empty((total_rows,), dtype=torch.float32),
        )
        reference = []
        offset = 0
        for peer, (start, rows) in enumerate(intervals):
            hidden = torch.randn(271, width, device="cuda", dtype=torch.bfloat16) * (peer + 1)
            activation = linear.quantize_dynamic(hidden)
            wire, _ = _pack_nvfp4(activation)
            source = PackedNvfp4Activation(wire, hidden.shape)
            pull_nvfp4_rows(source, target, RowSegment(peer, start, offset, rows))
            x = hidden[start:start + rows]
            codes, scales = fp4_quantize(x, activation.scale.reciprocal())
            self.assertTrue(torch.equal(target.values[offset:offset + rows], codes))
            self.assertTrue(torch.equal(target.scale[offset:offset + rows], activation.scale.expand(rows)))
            reference.append(paired.nvfp4.forward_quantized(Fp4Activation(codes, scales, activation.scale)))
            offset += rows
        output = torch.cat(paired.forward_nvfp4(target), dim=-1)
        self.assertTrue(torch.equal(output, torch.cat(reference)))

    def test_decode_packet_assembles_rank_local_scales_and_indexer_inputs(self):
        for rowwise in (False, True):
            with self.subTest(rowwise=rowwise):
                self._check_decode_packet(rowwise)

    def _check_decode_packet(self, rowwise):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_comm_fusion import (
            DsaDecodeActivation, PackedNvfp4Activation,
        )
        from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual
        from rtp_llm.models_py.triton_kernels.common.nvfp4 import unpack_decode_nvfp4

        def rand(*shape):
            return torch.randn(shape, device="cuda", dtype=torch.bfloat16)

        width = 7168
        linear, _, _ = self.projection(n=128, k=width)
        residual = KimiK3AttentionResidual(rand(width), rand(width), 1e-6)
        norm = rand(width)
        for rows in (3, 32, 129):
            dsa_packets = []
            kda_packets = []
            shard_outputs = []
            for _ in range(8):
                prefix, anchors = rand(rows, width), rand(rows, 2, width)
                kwargs = dict(num_blocks=2, output_norm_weight=norm, output_norm_eps=1e-6)
                packet = residual(
                    prefix.clone(), anchors.clone(), quantize_nvfp4=True,
                    decode_head_gate_columns=96, rowwise_nvfp4=rowwise, **kwargs,
                )
                self.assertIsInstance(packet, DsaDecodeActivation)
                fp4, _, head_gate = packet.views()
                head_gate.copy_(torch.randn_like(head_gate))
                kda = residual(
                    prefix, anchors, quantize_nvfp4=True, rowwise_nvfp4=rowwise, **kwargs,
                )
                self.assertIsInstance(kda, PackedNvfp4Activation)
                self.assertTrue(torch.equal(fp4.wire, kda.wire))
                if rowwise:
                    ref = residual(
                        prefix, anchors, quantize_nvfp4=True, decode_head_gate_columns=96, **kwargs,
                    )
                    self.assertTrue(torch.equal(packet.local_hidden, ref.local_hidden))
                    _, fp8, _ = packet.views()
                    _, ref_fp8, _ = ref.views()
                    self.assertTrue(torch.equal(
                        fp8.values.view(torch.uint8), ref_fp8.values.view(torch.uint8),
                    ))
                    self.assertTrue(torch.equal(fp8.scale_wire[:, :rows], ref_fp8.scale_wire[:, :rows]))
                shard_outputs.append(linear.forward_quantized(kda.unpack(kda.wire)))
                kda_packets.append(kda)
                dsa_packets.append(packet)
            for packets in (kda_packets, dsa_packets):
                wire = torch.cat([p.wire for p in packets])
                fp4, fp8, gate = unpack_decode_nvfp4(wire, packets[0], len(packets))
                local_activations = [p.unpack(p.wire) for p in kda_packets]
                self.assertTrue(torch.equal(fp4.scale, torch.cat([
                    a.scale.expand(rows) for a in local_activations
                ])))
                torch.testing.assert_close(linear.forward_quantized(fp4), torch.cat(shard_outputs), atol=0, rtol=0)
                if isinstance(packets[0], DsaDecodeActivation):
                    views = [p.views() for p in packets]
                    self.assertTrue(torch.equal(
                        fp8[0].view(torch.uint8),
                        torch.cat([v.values.view(torch.uint8) for _, v, _ in views]),
                    ))
                    self.assertTrue(torch.equal(
                        fp8[1], torch.cat([v.scale_wire.T[:rows] for _, v, _ in views]),
                    ))
                    self.assertTrue(torch.equal(gate, torch.cat([g for _, _, g in views])))

    def test_decode_qkv_norm_emits_fp8_and_nvfp4_partials(self):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import sgl_per_token_group_quant_fp8
        from rtp_llm.models_py.triton_kernels.common.layernorm_gated import fused_q_kv_rmsnorm
        from rtp_llm.models_py.triton_kernels.common.nvfp4 import quantize_nvfp4_from_partials

        qkv = torch.randn(255, 2112, device="cuda", dtype=torch.bfloat16)
        q, kv, _ = qkv.split((1536, 512, 64), dim=-1)
        weights = [torch.randn(x.shape[1], device=x.device, dtype=x.dtype) for x in (q, kv)]
        ref_q, ref_kv = fused_q_kv_rmsnorm(q, kv, *weights, 1e-6)
        actual_q, actual_kv, (fp8, scales), amax = fused_q_kv_rmsnorm(
            q, kv, *weights, 1e-6, quantize_q=True,
        )
        torch.testing.assert_close(actual_q, ref_q, atol=0, rtol=0)
        torch.testing.assert_close(actual_kv, ref_kv, atol=0, rtol=0)
        expected, expected_scales = sgl_per_token_group_quant_fp8(
            ref_q, group_size=128, eps=1e-4, column_major_scales=True,
            scale_tma_aligned=True, scale_ue8m0=True,
        )
        self.assertTrue(torch.equal(fp8.view(torch.uint8), expected.view(torch.uint8)))
        self.assertTrue(torch.equal(scales, expected_scales))
        linear, _, _ = self.projection(k=1536)
        reference = linear.quantize_dynamic(ref_q)
        actual = quantize_nvfp4_from_partials(actual_q, amax)
        for got, wanted in zip(actual, reference):
            self.assertTrue(torch.equal(got.view(-1), wanted.view(-1)))

    def test_kda_padding_and_graph_recomputes_activation_scale(self):
        linear, weight, logical_n = self.projection(n=580, module="kda.input")
        self.assertEqual((logical_n, linear.output_size), (580, 608))
        x = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):
                linear(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = linear(x)
        for scale in (0.0, 0.001, 1.0, 100.0):
            x.copy_(torch.randn_like(x) * scale)
            graph.replay()
            self.check(linear, weight, x, actual)

    def test_static_linear_retains_existing_scale_contract(self):
        linear, weight, _ = self.projection(dynamic=False)
        self.check(
            linear, weight, torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
        )

    def test_dynamic_scale_on_decode_side_stream_graph_replay(self):
        from rtp_llm.models_py.utils.stream_dag import StreamDag

        first, first_weight, _ = self.projection()
        query, query_weight, _ = self.projection()
        state = NS(input=torch.randn(7, 64, device="cuda", dtype=torch.bfloat16))
        dag = StreamDag(torch.cuda.current_device(), side_streams=("query",))
        dag.add("input", lambda s: setattr(s, "hidden", first(s.input)))
        dag.add("query", lambda s: setattr(s, "query", query(s.hidden)),
                stream="query", after=("input",))
        dag.add("join", lambda s: setattr(s, "output", s.query.clone()), after=("query",))
        dag.finalize()
        for _ in range(3):
            dag.run(state, multi_stream=True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dag.run(state, multi_stream=True)
        for scale in (0.0, 0.001, 1.0, 100.0):
            state.input.copy_(torch.randn_like(state.input) * scale)
            graph.replay()
            self.check(first, first_weight, state.input, state.hidden)
            self.check(query, query_weight, state.hidden, state.output)

    def test_natural_q_gate_and_kv_keep_the_original_nvfp4_operands(self):
        from rtp_llm.models_py.modules.kimi_k3.natural_dsa import (
            NaturalPairedProjection,
            split_natural_projection,
        )

        linear, _, _ = self.projection(n=416)
        q, kv, gate = split_natural_projection(linear, 64, 96, 256, 0)
        merged = NaturalPairedProjection(q, gate)
        self.assertEqual(merged.nvfp4.weight.dtype, torch.uint8)
        self.assertEqual(kv.weight.dtype, torch.uint8)
        x = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
        expected = linear(x).float()
        q_out, g_out = merged(x)
        actual = torch.cat((q_out, kv(x), g_out), -1).float()
        self.assertLess(((actual - expected).norm() / expected.norm()).item(), 0.005)


if __name__ == "__main__":
    unittest.main()
