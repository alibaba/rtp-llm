"""Synthetic contracts for GLM53's offline FP8 requantization capability."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    MoeAtomicWeight,
    MoeConfig,
)
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    LoadQuantPerBlockFp8Weight,
    PerBlockFp8Weight,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.deepseek_v2 import DeepSeekV2Weight
from rtp_llm.models.glm5_3_flash import Glm53Flash, Glm53FlashWeight
from rtp_llm.models.glm5_3_flash_mtp import Glm53FlashMtp, Glm53FlashMtpWeight
from rtp_llm.models_py.kernels.cuda.fp8_kernel import fp8_kernel
from rtp_llm.utils.model_weight import CkptWeightInfo, W


def _fixture(shape, scale_value=1.06251):
    # The BF16 bridge rounds the normalized unit entries onto an FP8 midpoint.
    # Retain a block maximum so the two producers use the same UE8M0 scale.
    values = torch.ones(shape, dtype=torch.float32)
    values[..., ::128, ::128] = 448.0
    values[..., 1::128, 1::128] = -1.0
    values[..., 2::128, 2::128] = 0.0
    scale = torch.full(
        (*shape[:-2], shape[-2] // 128, shape[-1] // 128),
        scale_value,
        dtype=torch.float32,
    )
    return values.to(torch.float8_e4m3fn).cuda(), scale.cuda()


def _reference(weight, scale, bridge_dtype):
    """Independent CPU block indexing and frexp-based UE8M0 rounding.

    No production dequantization, quantization or scale-packing helper is used.
    Torch supplies the defined E4M3 round-to-nearest conversion.
    """
    n, k = weight.shape[-2:]
    w = weight.cpu().reshape(-1, n, k)
    s = scale.cpu().float().reshape(-1, n // 128, k // 128)
    quant, logical = torch.empty_like(w), torch.empty_like(s)
    for batch in range(w.shape[0]):
        for row in range(n // 128):
            for col in range(k // 128):
                rows = slice(row * 128, (row + 1) * 128)
                cols = slice(col * 128, (col + 1) * 128)
                x = w[batch, rows, cols].float() * s[batch, row, col]
                x = x.to(bridge_dtype).float()
                raw_scale = (x.abs().amax().clamp_min(1e-4) / 448.0).item()
                mantissa, exponent = math.frexp(raw_scale)
                rounded_scale = math.ldexp(
                    1.0, exponent - 1 if mantissa == 0.5 else exponent
                )
                quant[batch, rows, cols] = (
                    (x * torch.tensor(1.0 / rounded_scale, dtype=torch.float32))
                    .clamp(-448, 448)
                    .to(torch.float8_e4m3fn)
                )
                logical[batch, row, col] = rounded_scale
    return quant.view_as(weight), logical.view_as(scale)


def _unpack(packed, k):
    words = packed.cpu().to(torch.int64)
    exponent = torch.stack(
        [((words >> shift) & 255) for shift in (0, 8, 16, 24)], dim=-1
    ).flatten(-2)[..., : k // 128]
    return torch.pow(2.0, exponent.float() - 127)


class _IdentityDevice:
    def maybe_rewrite_weight_by_key(self, key, tensor, **kwargs):
        return tensor

    def shuffle_moe_weight(self, tensor, compute_dtype, name):
        return tensor


class TestGlm53Fp8Requant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("UNSUPPORTED: CUDA is required for real scale packing")
        if torch.cuda.get_device_capability()[0] not in (10, 12):
            raise RuntimeError("UNSUPPORTED: UE8M0 producer requires SM10/SM12")

    def assert_bytes_equal(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(
            torch.equal(
                actual.detach().cpu().reshape(-1).contiguous().view(torch.uint8),
                expected.detach().cpu().reshape(-1).contiguous().view(torch.uint8),
            )
        )

    def assert_packed(self, packed, expected, n, k):
        self.assertEqual(packed.dtype, torch.int32)
        self.assertEqual(packed.stride(-2), 1)
        if packed.shape[-1] > 1:
            self.assertGreaterEqual(packed.stride(-1), n)
        self.assertEqual(
            tuple(packed.shape), (*expected.shape[:-2], n, (k // 128 + 3) // 4)
        )
        self.assert_bytes_equal(_unpack(packed, k), expected.repeat_interleave(128, -2))

    def test_model_capability_producers(self):
        self.assertFalse(LoadConfig.model_construct().fp8_requant_preserve_fp32)
        cases = (
            (Glm53FlashWeight, SimpleNamespace(), True),
            (Glm53FlashMtpWeight, SimpleNamespace(is_glm53_mtp=True), True),
            (Glm53FlashMtpWeight, SimpleNamespace(is_glm53_mtp=False), False),
            (Glm53FlashMtpWeight, SimpleNamespace(), False),
        )
        for weight_type, model_config, expected in cases:
            with self.subTest(weight_type=weight_type, model_config=model_config):
                weight = object.__new__(weight_type)
                weight.model_config = model_config
                config = LoadConfig.model_construct()
                with patch.object(
                    DeepSeekV2Weight, "create_load_config", return_value=config
                ) as base:
                    result = weight.create_load_config("dtype", database="database")
                base.assert_called_once_with(
                    "dtype",
                    "database",
                    phy2log=None,
                    exported_device=None,
                    force_cpu_load_weights=False,
                    moe_pure_tp_preshard=False,
                )
                self.assertIs(result, config)
                self.assertEqual(result.fp8_requant_preserve_fp32, expected)
                self.assertEqual(
                    result.model_copy().fp8_requant_preserve_fp32, expected
                )

    def test_default_and_fp32_match_independent_references(self):
        with self.assertRaises(ValueError):
            fp8_kernel.requant_weight_ue8m0(
                torch.empty(0), torch.empty(0), dequant_dtype=torch.float16
            )
        witnessed_difference = False
        for shape in ((128, 512), (2, 256, 1024)):
            for scale_value in (1.06249, 1.06251, 1.0, 1e-8):
                with self.subTest(shape=shape, scale=scale_value):
                    weight, scale = _fixture(shape, scale_value)
                    original_w, original_s = weight.clone(), scale.clone()
                    outputs = []
                    for dtype in (torch.bfloat16, torch.float32):
                        kwargs = (
                            {} if dtype == torch.bfloat16 else {"dequant_dtype": dtype}
                        )
                        with patch.object(
                            fp8_kernel,
                            "block_quant_dequant",
                            wraps=fp8_kernel.block_quant_dequant,
                        ) as dequant:
                            quant, packed = fp8_kernel.requant_weight_ue8m0(
                                weight, scale, **kwargs
                            )
                        if len(shape) == 3 and dtype == torch.float32:
                            self.assertEqual(dequant.call_count, shape[0])
                            self.assertTrue(
                                all(
                                    call.args[0].ndim == 2
                                    for call in dequant.call_args_list
                                )
                            )
                        expected_w, expected_s = _reference(weight, scale, dtype)
                        self.assert_bytes_equal(quant, expected_w)
                        self.assert_packed(packed, expected_s, shape[-2], shape[-1])
                        outputs.append(quant.cpu().view(torch.uint8))
                    witnessed_difference |= not torch.equal(*outputs)
                    self.assert_bytes_equal(weight, original_w)
                    self.assert_bytes_equal(scale, original_s)
        self.assertTrue(
            witnessed_difference, "synthetic fixture must distinguish the bridge"
        )

    def _descriptor(self, offline, name=W.attn_gate_w):
        weights = [CkptWeightInfo("gate.weight")]
        if name in (W.moe_w1, W.ffn_w13):
            weights.append(CkptWeightInfo("up.weight"))
        if name in (W.moe_w1, W.moe_w2):
            source = MoeAtomicWeight(name, weights, config=MoeConfig(expert_num=2))
        elif name in (W.ffn_w13, W.ffn_w2):
            source = FfnAtomicWeight(name, weights, config=FfnConfig())
        else:
            source = AtomicWeight(name, weights)
        descriptor = WeightModule.create(
            source,
            Fp8BlockWiseQuantConfig(is_quanted=offline),
        )
        self.assertIsInstance(
            descriptor, PerBlockFp8Weight if offline else LoadQuantPerBlockFp8Weight
        )
        return descriptor

    def test_resolved_model_backend_owns_routed_conversion(self):
        self.assertFalse(LoadConfig.model_construct().fp8_defer_moe_requant)
        for model_type in (Glm53Flash, Glm53FlashMtp):
            for capability in (False, True):
                for strategy in (
                    "auto",
                    "fp8_per_block",
                    "mega_moe_fp8",
                    "mega_moe_fp8_se",
                ):
                    with self.subTest(
                        model=model_type, capability=capability, strategy=strategy
                    ):
                        config = LoadConfig.model_construct(
                            fp8_requant_preserve_fp32=capability
                        )
                        loader = SimpleNamespace(get_load_config=lambda: config)
                        model = object.__new__(model_type)
                        model.moe_config = SimpleNamespace(moe_strategy=strategy)
                        with patch.object(
                            BaseModel, "create_model_loader", return_value=loader
                        ) as base:
                            self.assertIs(model.create_model_loader(), loader)
                        base.assert_called_once_with()
                        self.assertEqual(
                            config.fp8_defer_moe_requant,
                            capability
                            and strategy in ("mega_moe_fp8", "mega_moe_fp8_se"),
                        )
                        self.assertEqual(
                            config.model_copy().fp8_defer_moe_requant,
                            config.fp8_defer_moe_requant,
                        )

    def _postprocess_checkpoint_weight(self, name, weight, scale, deferred):
        from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper

        descriptor = self._descriptor(True, name)
        config = LoadConfig.model_construct(
            fp8_requant_preserve_fp32=True,
            fp8_defer_moe_requant=deferred,
            exported_device=_IdentityDevice(),
        )
        with patch.object(
            deepgemm_wrapper, "is_deep_gemm_e8m0_used", return_value=True
        ):
            result = descriptor._postprocess(
                {descriptor.kernel.name: weight, descriptor.scale.name: scale},
                "cuda",
                config,
            )
        return result[descriptor.kernel.name], result[descriptor.scale.name]

    def test_loader_to_mega_moe_fp32_boundary(self):
        import deep_gemm

        from rtp_llm.models_py.kernels.cuda import fp8_kernel as exported_fp8
        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fp8 import GLM5MegaMoEFP8
        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fp8_se import (
            GLM5MegaMoEFP8SE,
        )

        # Validate the actual loader and model consumers up to the foreign
        # transform boundary. No symmetric buffer or full-model equivalence claim.
        # ARM's DeepGEMM package omits this optional backend; provide the mocked
        # boundary there too, without weakening checks on its real inputs.
        for shared in (False, True):
            prefix = () if shared else (2,)
            w13, s13 = _fixture((*prefix, 256, 512), 1.06251)
            w2, s2 = _fixture((*prefix, 512, 128), 1.06249)
            originals = [tensor.clone() for tensor in (w13, s13, w2, s2)]
            ref13, refs13 = _reference(w13, s13, torch.float32)
            ref2, refs2 = _reference(w2, s2, torch.float32)
            for deferred in (False, True):
                with self.subTest(shared=shared, deferred=deferred):
                    names = (W.ffn_w13, W.ffn_w2) if shared else (W.moe_w1, W.moe_w2)
                    loaded13, loaded_s13 = self._postprocess_checkpoint_weight(
                        names[0], w13, s13, deferred
                    )
                    loaded2, loaded_s2 = self._postprocess_checkpoint_weight(
                        names[1], w2, s2, deferred
                    )
                    expect_raw = deferred and not shared
                    self.assertEqual(
                        loaded_s13.dtype, torch.float32 if expect_raw else torch.int32
                    )
                    self.assertEqual(
                        loaded_s2.dtype, torch.float32 if expect_raw else torch.int32
                    )
                    cls = GLM5MegaMoEFP8SE if shared else GLM5MegaMoEFP8
                    module = object.__new__(cls)
                    torch.nn.Module.__init__(module)
                    module.cfg = SimpleNamespace(
                        n_local_experts=2, dim=512, moe_inter_dim=128, layer_id=0
                    )
                    with patch.object(module, "_setup_buffer_and_warmup"), patch.object(
                        deep_gemm,
                        "transform_weights_for_mega_moe_fp8",
                        side_effect=lambda a, b: (a, b),
                        create=True,
                    ) as transform, patch.object(
                        deep_gemm, "get_block_m_for_mega_moe_fp8", create=True
                    ), patch.object(
                        exported_fp8,
                        "requant_weight_ue8m0",
                        wraps=fp8_kernel.requant_weight_ue8m0,
                    ) as requant:
                        if shared:
                            module.setup_shared_expert_from_fp8(
                                loaded13, loaded_s13, loaded2, loaded_s2
                            )
                        else:
                            gate, up = loaded13.chunk(2, -2)
                            gate_s, up_s = loaded_s13.chunk(2, -2)
                            module.setup_weights_from_fp8(
                                gate, gate_s, loaded2, loaded_s2, up, up_s
                            )
                    transform.assert_called_once()
                    self.assertEqual(requant.call_count, 2 if expect_raw else 0)
                    for call in requant.call_args_list:
                        self.assertEqual(call.kwargs, {"dequant_dtype": torch.float32})
                    (actual13, actual_s13), (actual2, actual_s2) = (
                        transform.call_args.args
                    )
                    self.assert_bytes_equal(actual13, ref13)
                    self.assert_bytes_equal(actual2, ref2)
                    self.assert_packed(actual_s13, refs13, 256, 512)
                    self.assert_packed(actual_s2, refs2, 512, 128)
                    for tensor, original in zip((w13, s13, w2, s2), originals):
                        self.assert_bytes_equal(tensor, original)

            if shared:
                # A raw shared-expert input may bypass the generic loader.
                # Its owned converter must still preserve FP32 intermediates.
                module = object.__new__(GLM5MegaMoEFP8SE)
                torch.nn.Module.__init__(module)
                module.cfg = SimpleNamespace(
                    n_local_experts=2, dim=512, moe_inter_dim=128, layer_id=0
                )
                with patch.object(module, "_setup_buffer_and_warmup"), patch.object(
                    deep_gemm,
                    "transform_weights_for_mega_moe_fp8",
                    side_effect=lambda a, b: (a, b),
                    create=True,
                ) as transform, patch.object(
                    deep_gemm, "get_block_m_for_mega_moe_fp8", create=True
                ):
                    module.setup_shared_expert_from_fp8(w13, s13, w2, s2)
                (actual13, actual_s13), (actual2, actual_s2) = transform.call_args.args
                self.assert_bytes_equal(actual13, ref13)
                self.assert_bytes_equal(actual2, ref2)
                self.assert_packed(actual_s13, refs13, 256, 512)
                self.assert_packed(actual_s2, refs2, 512, 128)

    def test_shared_expert_rejects_missing_foreign_backend(self):
        from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fp8_se import (
            GLM5MegaMoEFP8SE,
        )

        module = object.__new__(GLM5MegaMoEFP8SE)
        torch.nn.Module.__init__(module)
        module.cfg = SimpleNamespace()
        with patch.dict("sys.modules", {"deep_gemm": SimpleNamespace()}):
            with self.assertRaisesRegex(
                RuntimeError, "requires DeepGEMM shared-expert support"
            ):
                module.setup_shared_expert_from_fp8(None, None, None, None)

    def test_real_descriptor_postprocess_routing(self):
        from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper
        from rtp_llm.models_py.kernels.cuda import fp8_kernel as exported_fp8

        weight, scale = _fixture((128, 512))
        # Each arm traverses the real composite + atomic postprocess consumers.
        # Online controls start after quantization; this is not online-loader E2E.
        for offline, capability, scale_dtype, hardware, expected_dtype in (
            (True, True, torch.float32, True, torch.float32),
            (True, False, torch.float32, True, torch.bfloat16),
            (False, True, torch.float32, True, torch.bfloat16),
            (True, True, torch.bfloat16, True, torch.bfloat16),
            (True, True, torch.int32, True, None),
            (False, True, torch.int32, True, None),
            (True, True, torch.float32, False, None),
        ):
            with self.subTest(
                offline=offline,
                capability=capability,
                scale_dtype=scale_dtype,
                hardware=hardware,
            ):
                descriptor = self._descriptor(offline)
                arm_w = weight.clone()
                if scale_dtype == torch.int32:
                    _, packed = fp8_kernel.requant_weight_ue8m0(weight, scale)
                    arm_s = packed.clone()
                else:
                    arm_s = scale.to(scale_dtype).clone()
                original_w, original_s = arm_w.clone(), arm_s.clone()
                config = LoadConfig.model_construct(
                    fp8_requant_preserve_fp32=capability,
                    exported_device=_IdentityDevice(),
                )
                with patch.object(
                    deepgemm_wrapper, "is_deep_gemm_e8m0_used", return_value=hardware
                ), patch.object(
                    exported_fp8,
                    "requant_weight_ue8m0",
                    wraps=fp8_kernel.requant_weight_ue8m0,
                ) as requant:
                    result = descriptor._postprocess(
                        {descriptor.kernel.name: arm_w, descriptor.scale.name: arm_s},
                        "cuda",
                        config,
                    )
                if expected_dtype is None:
                    requant.assert_not_called()
                    if scale_dtype == torch.int32:
                        self.assertIs(result[descriptor.kernel.name], arm_w)
                        self.assertIs(result[descriptor.scale.name], arm_s)
                else:
                    requant.assert_called_once()
                    self.assertEqual(
                        requant.call_args.kwargs,
                        (
                            {"dequant_dtype": torch.float32}
                            if expected_dtype == torch.float32
                            else {}
                        ),
                    )
                    expected_w, expected_s = _reference(arm_w, arm_s, expected_dtype)
                    self.assert_bytes_equal(result[descriptor.kernel.name], expected_w)
                    self.assert_packed(
                        result[descriptor.scale.name], expected_s, 128, 512
                    )
                self.assert_bytes_equal(arm_w, original_w)
                self.assert_bytes_equal(arm_s, original_s)

    def test_gate_up_cat_and_mn_major_layout(self):
        for shape in ((128, 1024), (2, 128, 1024)):
            with self.subTest(shape=shape):
                gate, gate_s = _fixture(shape, 1.06251)
                up, up_s = _fixture(shape, 1.06249)
                combined, combined_s = torch.cat([gate, up], -2), torch.cat(
                    [gate_s, up_s], -2
                )
                quant, packed = fp8_kernel.requant_weight_ue8m0(
                    combined, combined_s, dequant_dtype=torch.float32
                )
                gate_q, gate_p = fp8_kernel.requant_weight_ue8m0(
                    gate, gate_s, dequant_dtype=torch.float32
                )
                up_q, up_p = fp8_kernel.requant_weight_ue8m0(
                    up, up_s, dequant_dtype=torch.float32
                )
                self.assert_bytes_equal(quant, torch.cat([gate_q, up_q], -2))
                # Consumer restores the MN-major layout after concatenating scales.
                cat_p = (
                    torch.cat([gate_p, up_p], -2)
                    .transpose(-1, -2)
                    .contiguous()
                    .transpose(-1, -2)
                )
                self.assertEqual(cat_p.stride(-2), 1)
                self.assert_bytes_equal(packed, cat_p)
                ref_w, ref_s = _reference(combined, combined_s, torch.float32)
                self.assert_bytes_equal(quant, ref_w)
                self.assert_packed(cat_p, ref_s, combined.shape[-2], shape[-1])


if __name__ == "__main__":
    unittest.main()
