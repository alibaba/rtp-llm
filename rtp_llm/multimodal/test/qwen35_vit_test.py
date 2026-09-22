import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeVisionModel as HFVisionModel,
)

from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe import qwen3_5_moe_vit as vit
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
    Qwen3_5MoeVisionConfig,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.vision_linear import (
    QKVParallelLinear,
    UnquantizedLinearMethod,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.vision_parameter import (
    ModelWeightParameter,
)


class Qwen35VitTest(unittest.TestCase):
    def config(self):
        config = Qwen3_5MoeVisionConfig(
            depth=2,
            hidden_size=32,
            num_heads=4,
            intermediate_size=48,
            patch_size=2,
            temporal_patch_size=2,
            num_position_embeddings=16,
            out_hidden_size=64,
        )
        config._attn_implementation = "sdpa"
        config.vit_attention_backend = "sdpa"
        return config

    def test_pretrained_config_reads_vision_section(self):
        vision = self.config().to_dict()
        vision["out_hidden_size"] = 4096
        for config in (
            vision,
            {
                "model_type": "qwen3_5_moe",
                "text_config": {"hidden_size": 4096},
                "vision_config": vision,
            },
        ):
            with self.subTest(nested="vision_config" in config):
                with tempfile.TemporaryDirectory() as directory:
                    Path(directory, "config.json").write_text(json.dumps(config))
                    loaded = Qwen3_5MoeVisionConfig.from_pretrained(directory)
                for name in ("hidden_size", "depth", "num_heads", "out_hidden_size"):
                    self.assertEqual(getattr(loaded, name), vision[name])

    def make_model(self, config=None):
        config = self.config() if config is None else config
        # vLLM layers allocate uninitialized weights; always load a checkpoint
        # before checking numerical behavior.
        checkpoint = HFVisionModel(copy.deepcopy(config)).state_dict()
        model = vit.Qwen3_5MoeVisionModel(config)
        model.load_weights(checkpoint.items())
        return model.eval()

    @torch.inference_mode()
    def test_no_deepstack_reuses_merger_output(self):
        model = self.make_model()
        pixels, grid = torch.randn(24, 24), torch.tensor([[1, 4, 6]])
        output = []
        handle = model.merger.register_forward_hook(
            lambda module, args, value: output.append(value)
        )
        try:
            with patch.object(vit, "HAS_TRITON", False):
                result = model(pixels, grid).pooler_output
            self.assertEqual(len(output), 1)
            self.assertEqual(result.data_ptr(), output[0].data_ptr())
            torch.testing.assert_close(result, output[0], atol=0, rtol=0)
        finally:
            handle.remove()

    @torch.inference_mode()
    def test_patch_gemm_preserves_conv3d_weights_and_order(self):
        torch.manual_seed(11)
        layer = vit.Qwen3_5MoeVisionPatchEmbed(self.config())
        pixels = torch.randn(48, 24)
        expected = F.conv3d(
            pixels.view(48, 3, 2, 2, 2),
            layer.proj.weight,
            layer.proj.bias,
            stride=(2, 2, 2),
        ).view(48, 32)
        torch.testing.assert_close(layer(pixels), expected, atol=1e-6, rtol=1e-5)

    @torch.inference_mode()
    def test_checkpoint_names_and_shapes_are_compatible(self):
        config = self.config()
        original = HFVisionModel(copy.deepcopy(config))
        ported = vit.Qwen3_5MoeVisionModel(config)
        loaded = ported.load_weights(original.state_dict().items())
        self.assertEqual(loaded, set(original.state_dict()))
        qkv = ported.blocks[0].attn.qkv
        self.assertIsInstance(qkv, QKVParallelLinear)
        self.assertIsInstance(qkv.weight, ModelWeightParameter)
        self.assertIsInstance(qkv.quant_method, UnquantizedLinearMethod)
        self.assertEqual((qkv.weight.tp_rank, qkv.weight.tp_size), (0, 1))
        # Device/dtype conversion must retain loader metadata.
        ported.double()
        self.assertIsInstance(qkv.weight, ModelWeightParameter)
        self.assertTrue(callable(qkv.weight.weight_loader))
        ported.load_weights(original.state_dict().items())
        self.assertEqual(
            sum(p.numel() for p in original.parameters()),
            sum(p.numel() for p in ported.parameters()),
        )

    @torch.inference_mode()
    def test_packed_different_grids_preserve_media_isolation(self):
        torch.manual_seed(7)
        model = self.make_model()
        grids = [torch.tensor([[2, 4, 6]]), torch.tensor([[1, 6, 4]])]
        pixels = [torch.randn(48, 24), torch.randn(24, 24) + 2]
        with patch.object(vit, "HAS_TRITON", False):
            separate = [model(x, g).pooler_output for x, g in zip(pixels, grids)]
            together = model(torch.cat(pixels), torch.cat(grids))
        self.assertEqual(together.last_hidden_state.shape, (72, 32))
        self.assertEqual(together.pooler_output.shape, (18, 64))
        torch.testing.assert_close(
            together.pooler_output, torch.cat(separate), atol=2e-6, rtol=1e-4
        )
        self.assertTrue(torch.isfinite(together.pooler_output).all())
        self.assertFalse(torch.equal(separate[0][:6], separate[1]))

    @torch.inference_mode()
    def test_prepared_metadata_matches_eager(self):
        model = self.make_model()
        x, grid = torch.randn(24, 24), torch.tensor([[1, 4, 6]])
        with patch.object(vit, "HAS_TRITON", False):
            metadata = model.prepare_graph_metadata(grid, x)
            eager = model(x, grid).pooler_output
            prepared = model(x, grid, _graph_metadata=metadata).pooler_output
        self.assertEqual(metadata["max_seqlen"].device.type, "cpu")
        torch.testing.assert_close(eager, prepared, atol=0, rtol=0)

    @torch.inference_mode()
    def test_fused_and_separate_qkv_checkpoint_loading_agree(self):
        fused = QKVParallelLinear(8, 4, 2, disable_tp=True)
        separate = QKVParallelLinear(8, 4, 2, disable_tp=True)
        weight = torch.randn(24, 8)
        bias = torch.randn(24)
        list(fused.load_weights([("weight", weight), ("bias", bias)]))
        for shard_id, w, b in zip("qkv", weight.chunk(3), bias.chunk(3)):
            separate.weight.weight_loader(separate.weight, w, shard_id)
            separate.bias.weight_loader(separate.bias, b, shard_id)
        torch.testing.assert_close(fused.weight, separate.weight, atol=0, rtol=0)
        torch.testing.assert_close(fused.bias, separate.bias, atol=0, rtol=0)
        x = torch.randn(5, 2, 8)
        actual, deferred_bias = separate(x)
        self.assertIsNone(deferred_bias)
        torch.testing.assert_close(actual, F.linear(x, weight, bias))

    @torch.inference_mode()
    def test_incomplete_checkpoint_fails_before_changing_weights(self):
        model = self.make_model()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        incomplete = {name: value + 1 for name, value in before.items()}
        del incomplete["blocks.0.attn.qkv.weight"]
        with self.assertRaisesRegex(ValueError, "checkpoint keys mismatch"):
            model.load_weights(incomplete.items())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.inference_mode()
    def test_cuda_rotary_matches_fp32_reference(self):
        torch.manual_seed(19)
        x = torch.randn(2, 97, 16, 72, device="cuda", dtype=torch.bfloat16)
        angle = torch.randn(97, 36, device="cuda")
        c, s = angle.cos().bfloat16(), angle.sin().bfloat16()
        a, b = x.float().chunk(2, dim=-1)
        cf, sf = c.float()[None, :, None, :], s.float()[None, :, None, :]
        expected = torch.cat((a * cf - b * sf, b * cf + a * sf), dim=-1).bfloat16()
        actual = vit.ApplyRotaryEmb()(x, c, s)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.inference_mode()
    def test_cuda_graph_replay_uses_new_pixels(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("FA4 graph check requires SM100 or newer")
        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.vision_graph import (
            VisionGraphCache,
        )

        config = self.config()
        config.hidden_size = 288
        config.num_heads = 4
        config.intermediate_size = 432
        config.depth = 1
        config.vit_attention_backend = "fa4"
        model = self.make_model(config).cuda().bfloat16().eval()
        pixels = torch.randn(24, 24, device="cuda", dtype=torch.bfloat16)
        grid = torch.tensor([[1, 4, 6]])
        cache = VisionGraphCache(model, enabled=True, max_entries=1, max_patches=128)
        expected = model(pixels, grid).pooler_output
        cache.run(pixels, grid)
        captured = cache.run(pixels, grid)
        self.assertEqual(cache.stats()["capture"], 1)
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
        changed = pixels + 0.5
        replayed = cache.run(changed, grid)
        torch.testing.assert_close(
            replayed, model(changed, grid).pooler_output, atol=0, rtol=0
        )
        self.assertFalse(torch.equal(replayed, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.inference_mode()
    def test_cuda_rotary_reads_strided_qk_and_preserves_head_size(self):
        torch.manual_seed(31)
        packed = torch.randn(97, 3, 16, 72, device="cuda", dtype=torch.bfloat16)
        original = packed.clone()
        qk = packed[:, :2].permute(1, 0, 2, 3)
        self.assertFalse(qk.is_contiguous())
        self.assertEqual(
            qk.untyped_storage().data_ptr(), packed.untyped_storage().data_ptr()
        )
        for rotary_dim in (64, 72):
            angle = torch.randn(97, rotary_dim // 2, device="cuda")
            c, s = angle.cos().bfloat16(), angle.sin().bfloat16()
            a, b = qk[..., :rotary_dim].float().chunk(2, dim=-1)
            cf, sf = c.float()[None, :, None, :], s.float()[None, :, None, :]
            expected = torch.cat(
                (a * cf - b * sf, b * cf + a * sf, qk[..., rotary_dim:].float()),
                dim=-1,
            ).bfloat16()
            actual = vit.ApplyRotaryEmb()(qk, c, s)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(actual.shape, (2, 97, 16, 72))
        torch.testing.assert_close(packed, original, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.inference_mode()
    def test_cuda_dense_and_varlen_preserve_segment_isolation(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("Dense FA4 optimization targets SM103")
        torch.manual_seed(37)
        for lengths, layout in [((1024, 1024), "dense"), ((1024, 512), "varlen")]:
            tokens = sum(lengths)
            packed = torch.randn(tokens, 3, 16, 72, device="cuda", dtype=torch.bfloat16)
            qk = packed[:, :2].permute(1, 0, 2, 3)
            angle = torch.randn(tokens, 36, device="cuda")
            c, s = angle.cos().bfloat16(), angle.sin().bfloat16()
            q, k = vit.ApplyRotaryEmb()(qk, c, s).unbind(0)
            self.assertEqual(q.shape[-1], 72)
            self.assertEqual(k.shape[-1], 72)
            v = packed[:, 2]
            cu = torch.tensor([0, lengths[0], tokens], dtype=torch.int32, device="cuda")
            attn = vit.VisionAttention(72**-0.5)
            attn.backend = "fa4"
            args = (cu, torch.tensor(max(lengths), dtype=torch.int32), lengths)
            actual = attn(q[None], k[None], v[None], *args)[0]
            self.assertEqual(attn.last_layout, layout)
            reference = []
            offset = 0
            for length in lengths:
                qs, ks, vs = [
                    t[offset : offset + length].float().transpose(0, 1)[None]
                    for t in (q, k, v)
                ]
                reference.append(
                    F.scaled_dot_product_attention(
                        qs, ks, vs, scale=72**-0.5, dropout_p=0.0
                    )[0]
                    .transpose(0, 1)
                    .bfloat16()
                )
                offset += length
            torch.testing.assert_close(
                actual, torch.cat(reference), atol=2e-3, rtol=2e-2
            )
            changed_v = v.clone()
            changed_v[lengths[0] :] += 10
            changed = attn(q[None], k[None], changed_v[None], *args)[0]
            torch.testing.assert_close(
                changed[: lengths[0]], actual[: lengths[0]], atol=0, rtol=0
            )
            self.assertFalse(torch.equal(changed[lengths[0] :], actual[lengths[0] :]))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.inference_mode()
    def test_cuda_dense_graph_replay_uses_new_pixels(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("Dense FA4 optimization targets SM103")
        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.vision_graph import (
            VisionGraphCache,
        )

        config = self.config()
        config.hidden_size = 1152
        config.num_heads = 16
        config.intermediate_size = 128
        config.depth = 1
        config.vit_attention_backend = "fa4"
        model = self.make_model(config).cuda().bfloat16().eval()
        pixels = torch.randn(1024, 24, device="cuda", dtype=torch.bfloat16)
        grid = torch.tensor([[1, 32, 32]])
        cache = VisionGraphCache(model, enabled=True, max_entries=1, max_patches=1024)
        expected = model(pixels, grid).pooler_output
        self.assertEqual(model.blocks[0].attn.attn.last_layout, "dense")
        cache.run(pixels, grid)
        captured = cache.run(pixels, grid)
        self.assertEqual(cache.stats()["capture"], 1)
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
        changed = pixels + 0.5
        replayed = cache.run(changed, grid)
        torch.testing.assert_close(
            replayed, model(changed, grid).pooler_output, atol=0, rtol=0
        )
        self.assertFalse(torch.equal(replayed, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.inference_mode()
    def test_packed_qk_rotary_matches_general_kernel(self):
        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe import vision_kernels

        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for length in (1, 7, 129, 513):
                with self.subTest(dtype=dtype, length=length):
                    qkv = torch.randn(length, 3, 16, 72, device="cuda", dtype=dtype)
                    before = qkv.clone()
                    packed = qkv[:, :2].permute(1, 0, 2, 3)
                    # Dense input takes the general rotary kernel as reference.
                    dense = packed.clone(memory_format=torch.contiguous_format)
                    factors = torch.randn(length, 36, device="cuda", dtype=dtype)
                    cos, sin = factors.cos(), factors.sin()
                    expected = vision_kernels.apply_rotary(dense, cos, sin)
                    actual = vision_kernels.apply_rotary(packed, cos, sin)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    torch.testing.assert_close(qkv, before, atol=0, rtol=0)
                    if dtype != torch.float32:
                        self.assertTrue(actual.is_contiguous())
                    self.assertNotEqual(actual.data_ptr(), packed.data_ptr())

        # In-place and offset variants must retain their general-kernel behavior.
        qkv = torch.randn(129, 3, 16, 72, device="cuda", dtype=torch.bfloat16)
        packed = qkv[:, :2].permute(1, 0, 2, 3)
        factors = torch.randn(131, 36, device="cuda", dtype=torch.bfloat16)
        cos, sin = factors.cos(), factors.sin()
        expected = vision_kernels.apply_rotary(
            packed.contiguous(), cos, sin, seqlen_offsets=2
        )
        actual = vision_kernels.apply_rotary(packed, cos, sin, seqlen_offsets=2)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        expected = vision_kernels.apply_rotary(packed.contiguous(), cos, sin)
        actual = vision_kernels.apply_rotary(packed, cos, sin, inplace=True)
        self.assertEqual(actual.data_ptr(), packed.data_ptr())
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_mlp_cpu_autograd_preserves_unfused_path(self):
        torch.manual_seed(41)
        mlp = self.make_model().blocks[0].mlp
        x = torch.randn(5, 2, 32, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        with patch.object(
            torch,
            "_addmm_activation",
            side_effect=AssertionError("CPU/autograd must retain the unfused path"),
        ):
            actual = mlp(x)
        expected = mlp.linear_fc2(mlp.act_fn(mlp.linear_fc1(reference_x)))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        parameters = tuple(p for p in mlp.parameters() if p.requires_grad)
        actual_grad = torch.autograd.grad(actual.square().sum(), (x, *parameters))
        expected_grad = torch.autograd.grad(
            expected.square().sum(), (reference_x, *parameters)
        )
        for actual_value, expected_value in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual_value, expected_value, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.inference_mode()
    def test_cuda_mlp_fuses_fc1_for_half_inputs_and_preserves_shapes(self):
        torch.manual_seed(43)
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            mlp = self.make_model().blocks[0].mlp.cuda().to(dtype)
            inputs = [
                torch.randn(9, 32, device="cuda", dtype=dtype),
                torch.randn(3, 5, 32, device="cuda", dtype=dtype),
                torch.randn(3, 5, 64, device="cuda", dtype=dtype)[..., ::2],
                torch.randn(3, 5, 32, device="cuda", dtype=dtype).transpose(0, 1),
            ]
            for x in inputs:
                with self.subTest(dtype=dtype, shape=x.shape, stride=x.stride()):
                    with patch.object(
                        torch, "_addmm_activation", wraps=torch._addmm_activation
                    ) as fused:
                        actual = mlp(x)
                    fused.assert_called_once()
                    self.assertTrue(fused.call_args.kwargs["use_gelu"])
                    self.assertEqual(actual.shape, x.shape)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(torch.isfinite(actual).all())
                    # A layout change must not change values. This compares
                    # the same fused math, not old FC1's intermediate rounding.
                    contiguous = mlp(x.contiguous())
                    torch.testing.assert_close(actual, contiguous, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_mlp_grad_and_custom_linear_method_use_original_path(self):
        torch.manual_seed(47)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        mlp = self.make_model().blocks[0].mlp.cuda().to(dtype)
        with torch.enable_grad():
            x = torch.randn(7, 1, 32, device="cuda", dtype=dtype, requires_grad=True)
            with patch.object(
                torch,
                "_addmm_activation",
                side_effect=AssertionError("grad-enabled MLP must not use fused op"),
            ):
                actual = mlp(x)
            expected = mlp.linear_fc2(mlp.act_fn(mlp.linear_fc1(x)))
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            actual.float().square().sum().backward()
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())

        class OffsetLinearMethod(UnquantizedLinearMethod):
            def __init__(self):
                self.calls = 0

            def apply(self, layer, x, bias=None):
                self.calls += 1
                return super().apply(layer, x, bias) + 0.5

        method = OffsetLinearMethod()
        mlp.linear_fc1.quant_method = method
        with torch.inference_mode():
            x = torch.randn(7, 1, 32, device="cuda", dtype=dtype)
            expected = mlp.linear_fc2(
                mlp.act_fn(
                    F.linear(x, mlp.linear_fc1.weight, mlp.linear_fc1.bias) + 0.5
                )
            )
            with patch.object(
                torch,
                "_addmm_activation",
                side_effect=AssertionError("custom linear method must not be bypassed"),
            ):
                actual = mlp(x)
            self.assertEqual(method.calls, 1)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
