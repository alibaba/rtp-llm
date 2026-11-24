"""Native NVFP4 byte preservation and real MegaMoE/graph regression tests."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

# Bazel remote actions may omit HOME; give DeepJIT an explicit cache root.
os.environ.setdefault(
    "DG_JIT_CACHE_DIR",
    os.path.join(os.environ.get("TEST_TMPDIR", tempfile.gettempdir()), "deep_gemm"),
)

from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig  # noqa: E402
from rtp_llm.model_loader.native_mega_moe_nvfp4_weight import (  # noqa: E402
    NativeMegaMoeNvfp4Weight,
)
from rtp_llm.utils.model_weight import (  # noqa: E402
    CkptWeightInfo,
    W,
    identity,
    stack_,
    stack_moe_w1,
)


class NativeNvfp4LoaderTest(unittest.TestCase):
    def test_explicit_nvfp4_never_falls_back_to_mxfp4(self):
        from rtp_llm.model_loader.native_mega_moe_nvfp4_weight import (
            OnlineMegaMoeNvfp4Weight,
        )
        from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
            OnlineMegaMoeFp4FromNvfp4Weight,
            OnlineMegaMoeFp4Weight,
            wrap_moe_for_mega_moe,
        )
        from rtp_llm.model_loader.per_group_fp4_quant_weight import PerGroupFp4Weight

        source = MoeAtomicWeight(
            W.moe_w2,
            [
                CkptWeightInfo(
                    "model.layers.{i}.mlp.experts.{expert_id}.down_proj.weight",
                    identity,
                )
            ],
            stack_,
            config=MoeConfig(expert_num=4),
        )
        native_source = object.__new__(PerGroupFp4Weight)
        native_source.name, native_source.kernel = W.moe_w2, source
        with patch.dict(
            os.environ,
            {"MOE_STRATEGY": "mega_moe", "RTP_LLM_MOE_WEIGHT_FORMAT": "nvfp4"},
        ):
            self.assertIsInstance(
                wrap_moe_for_mega_moe(source), OnlineMegaMoeNvfp4Weight
            )
            self.assertIsInstance(
                wrap_moe_for_mega_moe(native_source), NativeMegaMoeNvfp4Weight
            )
        with patch.dict(
            os.environ,
            {"MOE_STRATEGY": "mega_moe_se", "RTP_LLM_MOE_WEIGHT_FORMAT": "nvfp4"},
        ):
            for weight in (source, native_source):
                with self.assertRaisesRegex(
                    ValueError, "requires MOE_STRATEGY=mega_moe"
                ):
                    wrap_moe_for_mega_moe(weight)
        with patch.dict(
            os.environ,
            {"MOE_STRATEGY": "mega_moe", "RTP_LLM_MOE_WEIGHT_FORMAT": "auto"},
        ):
            self.assertIs(type(wrap_moe_for_mega_moe(source)), OnlineMegaMoeFp4Weight)
            self.assertIs(
                type(wrap_moe_for_mega_moe(native_source)),
                OnlineMegaMoeFp4FromNvfp4Weight,
            )

    def test_native_moe_activation_dtype_resolves_auto_before_validation(self):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.ops import DataType

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"MOE_STRATEGY": "mega_moe", "RTP_LLM_MOE_WEIGHT_FORMAT": "nvfp4"},
        ):
            for act_type in ("auto", "AUTO", None, "bf16"):
                with self.subTest(act_type=act_type):
                    config = ModelConfig()
                    config.ckpt_path = directory
                    config.config_dtype = "bfloat16"
                    config.init_precision_config(None, act_type)
                    self.assertEqual(config.data_type, DataType.TYPE_BF16)
            config = ModelConfig()
            config.ckpt_path = directory
            config.config_dtype = "bfloat16"
            with self.assertRaisesRegex(ValueError, "requires BF16 activations"):
                config.init_precision_config(None, "fp16")

    def test_mxfp4_nibbles_and_both_scale_encodings(self):
        from rtp_llm.model_loader.native_mega_moe_nvfp4_weight import (
            dequantize_mxfp4_expert,
        )

        codes = torch.arange(16, dtype=torch.uint8).repeat(4)
        packed = (codes[::2] | (codes[1::2] << 4)).view(1, 32)
        expected = (
            torch.tensor(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
            )
            .repeat(4)
            .view(1, 2, 32)
        )
        expected *= torch.tensor([0.25, 8.0]).view(1, 2, 1)
        raw_scales = torch.tensor([[125, 130]], dtype=torch.uint8)
        for scales in (
            raw_scales,
            raw_scales.view(torch.float8_e8m0fnu),
            torch.tensor([[0.25, 8.0]]),
        ):
            actual = dequantize_mxfp4_expert(packed, scales)
            torch.testing.assert_close(
                actual, expected.view(1, 64).bfloat16(), rtol=0, atol=0
            )
        with self.assertRaisesRegex(ValueError, "scale shape"):
            dequantize_mxfp4_expert(packed, raw_scales[:, :1])

    def test_offline_mxfp4_model_weight_pass_is_opt_in(self):
        from rtp_llm.model_loader.model_weight_info import _apply_mega_moe_fp4_wrappers
        from rtp_llm.model_loader.native_mega_moe_nvfp4_weight import (
            OnlineMegaMoeNvfp4FromMxfp4Weight,
        )
        from rtp_llm.model_loader.offline_modelopt_fp4_quant_weight import (
            OfflineMegaMoeFp4MoeWeight,
        )

        raw = MoeAtomicWeight(
            W.moe_w2,
            [CkptWeightInfo("model.layers.{i}.mlp.experts.w2_weight")],
            stack_,
            config=MoeConfig(expert_num=4),
            stacked_ckpt_keys=True,
        )
        database = SimpleNamespace(
            get_pretrain_tensor_names=lambda: [
                "model.layers.0.mlp.experts.w2_weight_scale"
            ]
        )
        config = SimpleNamespace(scale_fmt="ue8m0")
        for mode, cls in (
            ("auto", OfflineMegaMoeFp4MoeWeight),
            ("nvfp4", OnlineMegaMoeNvfp4FromMxfp4Weight),
        ):
            with self.subTest(mode=mode), patch.dict(
                os.environ,
                {"MOE_STRATEGY": "mega_moe", "RTP_LLM_MOE_WEIGHT_FORMAT": mode},
            ):
                weights = SimpleNamespace(layer_weights=[[raw]])
                result = _apply_mega_moe_fp4_wrappers(weights, database, config)
                self.assertIs(type(result.layer_weights[0][0]), cls)

    def test_preserves_codes_scales_and_selected_expert_order(self):
        for key in (W.moe_w1, W.moe_w2):
            projections = (
                ("up_proj", "gate_proj") if key == W.moe_w1 else ("down_proj",)
            )
            kernel = MoeAtomicWeight(
                key,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.mlp.experts.{expert_id}." + p + ".weight",
                        identity,
                    )
                    for p in projections
                ],
                stack_moe_w1 if key == W.moe_w1 else stack_,
                config=MoeConfig(expert_num=4),
            )
            loader = NativeMegaMoeNvfp4Weight(kernel)
            tensors = {}
            for expert in (3, 1):
                for j, p in enumerate(projections):
                    base = f"model.layers.3.mlp.experts.{expert}.{p}"
                    tensors[base + ".weight"] = (
                        torch.arange(256).reshape(16, 16) + expert + j
                    ).to(torch.uint8)
                    tensors[base + ".weight_scale"] = torch.full(
                        (16, 2), 0.5 + j, dtype=torch.float8_e4m3fn
                    )
                    tensors[base + ".weight_scale_2"] = torch.tensor(
                        0.123 * (expert + 1) * (j + 1)
                    )
            source = SimpleNamespace(
                load_tensor=lambda name, dtype: [tensors[name].to(dtype)]
            )
            cfg = SimpleNamespace(get_selected_experts=lambda *args: [3, 1])
            with patch(
                "rtp_llm.model_loader.online_modelopt_fp4_quant_weight.dequantize_nvfp4_expert",
                side_effect=AssertionError("must not dequantize"),
            ):
                raw = loader._load_raw_tensor(source, 3, "cpu", cfg)
            for local, expert in enumerate((3, 1)):
                expected = torch.cat(
                    [
                        tensors[f"model.layers.3.mlp.experts.{expert}.{p}.weight"]
                        for p in projections
                    ]
                )
                self.assertTrue(
                    torch.equal(raw[key][local].view(torch.uint8), expected)
                )
                expected_sf = torch.cat(
                    [
                        tensors[
                            f"model.layers.3.mlp.experts.{expert}.{p}.weight_scale"
                        ].view(torch.uint8)
                        for p in projections
                    ]
                )
                self.assertTrue(
                    torch.equal(
                        raw[loader.scale.name][local].view(torch.uint8), expected_sf
                    )
                )
                expected_gsf = torch.stack(
                    [
                        tensors[
                            f"model.layers.3.mlp.experts.{expert}.{p}.weight_scale_2"
                        ]
                        for p in projections
                    ]
                )
                torch.testing.assert_close(
                    raw[loader.global_scale_name][local].reshape(-1),
                    expected_gsf,
                    rtol=0,
                    atol=0,
                )

    def test_full_loader_preserves_native_dtypes_with_ep4(self):
        from rtp_llm.model_loader.ffn_weight import MoeWeight

        for rank in range(4):
            kernels, tensors, expected = [], {}, {}
            for name, projections in (
                (W.moe_w1, ("up_proj", "gate_proj")),
                (W.moe_w2, ("down_proj",)),
            ):
                src = MoeAtomicWeight(
                    name,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.mlp.experts.{expert_id}." + p + ".weight",
                            identity,
                        )
                        for p in projections
                    ],
                    stack_moe_w1 if name == W.moe_w1 else stack_,
                    config=MoeConfig(expert_num=4),
                )
                native = NativeMegaMoeNvfp4Weight(src)
                kernels.append(native)
                for j, p in enumerate(projections):
                    base = f"model.layers.3.mlp.experts.{rank}.{p}"
                    tensors[base + ".weight"] = torch.full(
                        (4, 32), 0xBA, dtype=torch.uint8
                    )
                    tensors[base + ".weight_scale"] = torch.full(
                        (4, 4), 1.5 + j, dtype=torch.float8_e4m3fn
                    )
                    tensors[base + ".weight_scale_2"] = torch.tensor(
                        0.1234567 * (rank + 1) * (j + 1)
                    )
                expected[native.global_scale_name] = torch.tensor(
                    [
                        [
                            0.1234567 * (rank + 1) * (j + 1)
                            for j in range(len(projections))
                        ]
                    ]
                )
            cfg = SimpleNamespace(
                get_selected_experts=lambda *_: [rank],
                merge_lora=False,
                tp_size=4,
                tp_rank=rank,
                ep_size=4,
                ep_rank=rank,
                dp_size=1,
                dp_rank=0,
                ffn_tp_size=1,
                ffn_tp_rank=0,
                hidden_size=64,
                head_num=1,
                head_num_kv=1,
                size_per_head=64,
                moe_pure_tp_mode=False,
                bit=4,
                compute_dtype=torch.bfloat16,
                exported_device=SimpleNamespace(shuffle_moe_weight=lambda t, *_: t),
            )
            source = SimpleNamespace(
                load_tensor=lambda name, dtype: [tensors[name].to(dtype)]
            )
            actual = MoeWeight(kernels, MoeConfig(expert_num=4)).load(
                source, 3, "cpu", cfg
            )
            for name in (W.moe_w1, W.moe_w2):
                self.assertEqual(actual[name].dtype, torch.int8)
                self.assertTrue((actual[name].view(torch.uint8) == 0xBA).all())
            for name in (W.moe_s1, W.moe_s2):
                self.assertEqual(actual[name].dtype, torch.float8_e4m3fn)
            for name in (W.moe_w1_s2, W.moe_w2_s2):
                self.assertEqual(actual[name].dtype, torch.float32)
                torch.testing.assert_close(
                    actual[name].reshape(-1), expected[name].reshape(-1), rtol=0, atol=0
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class NativeNvfp4KernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch.distributed as dist

        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("SM100/SM103 required")
        cls.directory = tempfile.TemporaryDirectory()
        dist.init_process_group(
            "nccl",
            init_method="file://" + cls.directory.name + "/init",
            rank=0,
            world_size=1,
        )

    @classmethod
    def tearDownClass(cls):
        import torch.distributed as dist

        dist.destroy_process_group()
        cls.directory.cleanup()

    def test_fused_activation_pack_matches_eager(self):
        from triton.testing import do_bench_cudagraph

        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_nvfp4 import (
            quantize_native_nvfp4_input,
        )
        from rtp_llm.models_py.modules.glm5_mega_moe.nvfp4_input_triton import (
            quantize_input,
        )

        torch.manual_seed(42)
        for rows, hidden in (
            (1, 4096),
            (32, 7168),
            (64, 4096),
            (128, 7168),
            (1024, 7168),
        ):
            value = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
            value[0, :16] = -0.0
            with patch.dict(os.environ, {"RTP_LLM_FUSED_NVFP4_INPUT": "0"}):
                expected = quantize_native_nvfp4_input(value)
                actual = quantize_input(value)
                for reference, fused in zip(expected, actual):
                    torch.testing.assert_close(fused, reference, rtol=0, atol=0)
                with torch.cuda.stream(torch.cuda.Stream()):
                    old_ms = do_bench_cudagraph(
                        lambda: quantize_native_nvfp4_input(value), rep=10
                    )
                    new_ms = do_bench_cudagraph(lambda: quantize_input(value), rep=10)
                print(
                    f"NVFP4 input rows={rows} hidden={hidden}: eager={old_ms*1000:.2f}us fused={new_ms*1000:.2f}us",
                    flush=True,
                )

    def test_fp8_conversion_preserves_projection_scales_and_accuracy(self):
        from rtp_llm.model_loader.native_mega_moe_nvfp4_weight import (
            OnlineMegaMoeNvfp4FromFp8Weight,
        )
        from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
            dequantize_nvfp4_expert,
        )

        torch.manual_seed(37)
        for name in (W.moe_w1, W.moe_w2):
            for block, exponent in ((128, False), (32, True)):
                rows, hidden, experts = 256, 256, 2
                weight = (torch.randn(experts, rows, hidden, device="cuda") * 8).to(
                    torch.float8_e4m3fn
                )
                scale = torch.full(
                    (experts, rows if exponent else rows // block, hidden // block),
                    124.0 if exponent else 0.125,
                    device="cuda",
                )
                # Give gate a distinct global scale; a shared/incorrect scale breaks this check.
                scale[:, scale.shape[1] // 2 :] += 2 if exponent else 0.5
                loader = object.__new__(OnlineMegaMoeNvfp4FromFp8Weight)
                loader.name = name
                loader._source_block_size = block
                loader._scale_is_ue8m0_exponent = exponent
                loader.kernel = SimpleNamespace(
                    name=name, _load_raw_tensor=lambda *args: {name: weight}
                )
                loader.fp8_scale = SimpleNamespace(
                    name="input_scale",
                    _load_raw_tensor=lambda *args: {"input_scale": scale},
                )
                loader.out_scale = SimpleNamespace(name="scale")
                result = loader._load_raw_tensor(None, 3, "cuda", None)
                gsf = result[W.moe_w1_s2 if name == W.moe_w1 else W.moe_w2_s2]
                sf = (
                    torch.exp2(scale - 127)
                    if exponent
                    else scale.repeat_interleave(block, dim=1)
                )
                reference = (
                    (
                        weight.float().reshape(experts, rows, hidden // block, block)
                        * sf.unsqueeze(-1)
                    )
                    .reshape_as(weight)
                    .to(torch.bfloat16)
                )
                parts = 2 if name == W.moe_w1 else 1
                self.assertEqual(
                    tuple(gsf.shape), (experts, 2) if parts == 2 else (experts,)
                )
                for expert in range(experts):
                    for part in range(parts):
                        sl = slice(part * rows // parts, (part + 1) * rows // parts)
                        actual = dequantize_nvfp4_expert(
                            result[name][expert, sl].view(torch.uint8),
                            result["scale"][expert, sl],
                            gsf[expert, part] if parts == 2 else gsf[expert],
                        )
                        target = reference[expert, sl].float()
                        relative = (
                            actual.float() - target
                        ).square().mean().sqrt() / target.square().mean().sqrt()
                        self.assertTrue(torch.isfinite(actual).all())
                        self.assertLess(relative.item(), 0.13)
                if parts == 2:
                    self.assertTrue((gsf[:, 1] > gsf[:, 0] * 2).all())

    def test_fused_checkpoint_mxfp4_converts_independent_expert_projections(self):
        from rtp_llm.model_loader.native_mega_moe_nvfp4_weight import (
            OnlineMegaMoeNvfp4FromMxfp4Weight,
        )
        from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
            dequantize_nvfp4_expert,
        )

        def stack_fused(tensors):
            # A fused ModelOpt W1 is [gate | up], where RTP consumes [up | gate].
            gate, up = torch.stack(tensors).chunk(2, dim=1)
            return torch.cat((up, gate), dim=1)

        for name, ckpt, parts in (
            (W.moe_w1, "w13_weight", 2),
            (W.moe_w2, "w2_weight", 1),
        ):
            tensor_name = "model.layers.{i}.mlp.experts." + ckpt
            kernel = MoeAtomicWeight(
                name,
                [CkptWeightInfo(tensor_name)],
                stack_fused if parts == 2 else stack_,
                config=MoeConfig(expert_num=4),
                stacked_ckpt_keys=True,
            )
            layer_id, block_width, rows = 3, 128, 128
            key = tensor_name.format(i=layer_id)
            selected = [3, 1]
            bits = torch.full(
                (4, parts * rows, block_width // 2), 0xA2, dtype=torch.uint8
            )
            # All packed elements are [+1, -1]. Distinguish every expert and W1 half.
            raw_scales = torch.arange(123, 127, dtype=torch.uint8).reshape(4, 1, 1)
            raw_scales = raw_scales.expand(4, parts * rows, block_width // 32).clone()
            if parts == 2:
                raw_scales[:, rows:] += 3
            tensors = {key: bits, key + "_scale": raw_scales}
            loader = OnlineMegaMoeNvfp4FromMxfp4Weight(
                kernel, scale_dtype=torch.float8_e8m0fnu
            )
            cfg = SimpleNamespace(
                get_selected_experts=lambda *_: selected, compute_dtype=torch.bfloat16
            )
            source = SimpleNamespace(
                load_tensor=lambda name, dtype: [tensors[name].to(dtype)],
                has_tensor=tensors.__contains__,
            )
            raw = loader._load_raw_tensor(source, layer_id, "cuda", cfg)
            global_key = W.moe_w1_s2 if parts == 2 else W.moe_w2_s2
            gsf = raw[global_key].reshape(len(selected), parts)
            for local_id, expert_id in enumerate(selected):
                for part in range(parts):
                    # Fused source has the opposite projection order for W1.
                    src_part = parts - 1 - part
                    expected_scale = 2.0 ** (
                        int(raw_scales[expert_id, src_part * rows, 0]) - 127
                    )
                    row_slice = slice(rows * part, rows * (part + 1))
                    actual = dequantize_nvfp4_expert(
                        raw[name][local_id, row_slice].view(torch.uint8),
                        raw[loader.scale.name][local_id, row_slice],
                        gsf[local_id, part],
                    )
                    expected = (
                        torch.tensor(
                            [1, -1], dtype=torch.bfloat16, device="cuda"
                        ).repeat(rows, block_width // 2)
                        * expected_scale
                    )
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0)

    def test_kernel_reference_and_changed_input_graph_replay(self):
        import torch.distributed as dist
        from deep_gemm.utils import per_block_cast_to_nvfp4, per_token_cast_to_nvfp4

        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_nvfp4 import (
            GLM5MegaMoENVFP4,
            quantize_native_nvfp4_input,
        )

        torch.manual_seed(282)
        world, rank = dist.get_world_size(), dist.get_rank()
        e, d, n, topk = 4 * world, 256, 256, 2
        local = slice(4 * rank, 4 * (rank + 1))

        def quant(shape):
            p, s, g = [], [], []
            for i in range(e):
                x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.04
                gsf = torch.tensor(0.0011 * (i + 1), device="cuda", dtype=torch.float32)
                packed, sf = per_block_cast_to_nvfp4(x, weight_gsf=gsf)
                p.append(packed)
                s.append(sf.to(torch.float8_e4m3fn))
                g.append(gsf)
            return torch.stack(p).view(torch.int8), torch.stack(s), torch.stack(g)

        up, su, gu = quant((n, d))
        gate, sg, gg = quant((n, d))
        down, sd, gd = quant((d, n))
        # Independent non-power-of-two gate/up GSFs must survive intact.
        gg = gg * 1.7
        with patch.dict(os.environ, {"GLM5_MEGA_MOE_JIT_WARMUP_TOKENS": "1,16"}):
            moe = GLM5MegaMoENVFP4.from_params(
                0,
                dim=d,
                moe_inter_dim=n,
                n_routed_experts=e,
                n_activated_experts=topk,
                ep_size=world,
                ep_rank=rank,
                max_tokens_per_rank=128,
            )
            moe.setup_weights_from_nvfp4(
                torch.cat((up, gate), 1)[local],
                torch.cat((su.view(torch.uint8), sg.view(torch.uint8)), 1).view(
                    torch.float8_e4m3fn
                )[local],
                torch.stack((gu, gg), 1)[local],
                down[local],
                sd[local],
                gd[local],
            )
        tokens = 7 + rank
        x = torch.randn(tokens, d, device="cuda", dtype=torch.bfloat16)
        ids = (
            torch.arange(tokens * topk, device="cuda").reshape(tokens, topk) + rank
        ) % e
        weights = torch.softmax(torch.randn(tokens, topk, device="cuda"), dim=-1)
        for value in (x, torch.zeros_like(x), -torch.zeros_like(x)):
            expected_quant = per_token_cast_to_nvfp4(value, use_packed_ue4m3=True)
            actual_quant = quantize_native_nvfp4_input(value)
            for expected_part, actual_part in zip(expected_quant, actual_quant):
                torch.testing.assert_close(actual_part, expected_part, rtol=0, atol=0)
        y = moe(x, weights, ids).clone()

        magnitudes = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda"
        )

        def dequantize(packed, sf, gsf):
            # Keep FP32 here: checkpoint GSFs are not narrowed to BF16.
            bits = packed.view(torch.uint8)
            codes = torch.stack((bits & 15, bits >> 4), dim=-1).flatten(-2)
            values = magnitudes[(codes & 7).long()]
            values = torch.where((codes & 8) != 0, -values, values)
            return values * sf.float().repeat_interleave(16, dim=-1) * gsf

        p, s, g = per_token_cast_to_nvfp4(x)
        xd = dequantize(p, s, g[:, None])
        ref = torch.zeros_like(x, dtype=torch.float32)
        for expert in range(e):
            wg = dequantize(gate[expert], sg[expert], gg[expert])
            wu = dequantize(up[expert], su[expert], gu[expert])
            wd = dequantize(down[expert], sd[expert], gd[expert])
            gate_out = (xd @ wg.T).to(torch.bfloat16).float()
            up_out = (xd @ wu.T).to(torch.bfloat16).float()
            routing = ((ids == expert) * weights).sum(-1)
            hidden = torch.nn.functional.silu(gate_out) * up_out
            # The fused epilogue folds routing into the intermediate, scales
            # it by 16, then uses block-only NVFP4. L2 undoes that factor.
            hidden = hidden * routing[:, None] * 16.0
            hp, hs = per_block_cast_to_nvfp4(hidden)
            out = (dequantize(hp, hs, 1.0) @ wd.T) / 16.0
            ref += out.to(torch.bfloat16).float()
        ref = ref.to(torch.bfloat16).float()
        relative = (y.float() - ref).square().mean().sqrt() / ref.square().mean().sqrt()
        print(
            f"Native NVFP4 EP={world} rank={rank}: relative RMSE={relative.item():.6f}",
            flush=True,
        )
        self.assertLess(relative.item(), 0.03)
        clone = moe.clone_for_cuda_graph(share_mega_buf=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            clone(x, weights, ids)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = clone(x, weights, ids)
        for _ in range(3):
            x.normal_()
            ids.add_(1).remainder_(e)
            expected = moe(x, weights, ids).clone()
            graph.replay()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
