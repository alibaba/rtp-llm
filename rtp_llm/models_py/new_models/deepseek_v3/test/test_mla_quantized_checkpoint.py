"""Exercise quantized MLA using real, asymmetric checkpoint projections."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v3 import attention
from rtp_llm.models_py.quant_methods import fp8
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


class _MlaLoader(NewModelLoader):
    def _create_model(self):
        return attention.DeepSeekV32MlaAttention(**self.model_config["attention"])


def load_checkpoint(
    method, device, *, compressed, fp8_source=False, random_online=False
):
    h, q, r, k, p, v = (
        (1024, 128, 128, 128, 128, 128)
        if method.startswith("fp8_block")
        else (160, 48, 64, 32, 32, 32)
    )
    heads = 2
    shape = {
        ("q_a_proj" if compressed else "q_proj"): (
            q if compressed else heads * (k + p),
            h,
        ),
        "kv_a_proj_with_mqa": (r + p, h),
        "kv_b_proj": (heads * (k + v), r),
        "o_proj": (h, heads * v),
    }
    if compressed:
        shape["q_b_proj"] = (heads * (k + p), q)
    g = torch.Generator().manual_seed(418)
    data = {}
    for name, (n, cols) in shape.items():
        if fp8_source:
            data[name + ".weight"] = torch.randint(-4, 5, (n, cols), generator=g).to(
                torch.float8_e4m3fn
            )
            data[name + ".weight_scale_inv"] = (
                2.0
                ** torch.randint(
                    -3, 3, ((n + 127) // 128, (cols + 127) // 128), generator=g
                ).float()
            )
        elif method == "fp8_block_online" and not random_online:
            data[name + ".weight"] = torch.randint(
                -1, 2, (n, cols), generator=g
            ).bfloat16()
        else:
            data[name + ".weight"] = torch.randn(n, cols, generator=g).bfloat16()
    data["kv_a_layernorm.weight"] = torch.ones(r, dtype=torch.bfloat16)
    if compressed:
        data["q_a_layernorm.weight"] = torch.ones(q, dtype=torch.bfloat16)
    qc = QuantizationConfig(method)
    args = dict(
        hidden_size=h,
        num_heads=heads,
        q_lora_rank=q if compressed else 0,
        kv_lora_rank=r,
        nope_head_dim=k,
        rope_head_dim=p,
        v_head_dim=v,
        layer_idx=0,
        params_dtype=torch.bfloat16,
        quant_config=qc,
    )
    with tempfile.TemporaryDirectory() as root:
        save_file(data, str(Path(root) / "model.safetensors"))
        layer = _MlaLoader(
            {"model_type": "deepseek2", "attention": args},
            NewLoaderConfig(
                device=device,
                compute_dtype=torch.bfloat16,
                load_method="scratch",
                quant_config=qc,
            ),
            model_path=root,
        ).load()
    return layer, data


class QuantizedMlaTest(unittest.TestCase):
    def test_online_fused_and_prefill_round_fp8_only_once(self):
        from rtp_llm.models_py.kernels.cuda import fp8_quant

        def pack_cpu(s, mn):
            exp = ((s.view(torch.int32) >> 23) & 255).repeat_interleave(128, 0)[:mn]
            exp = torch.nn.functional.pad(exp, (0, -exp.shape[1] % 4))
            return (
                (exp.reshape(mn, -1, 4) << (torch.arange(4) * 8))
                .sum(-1)
                .int()
                .T.contiguous()
                .T
            )

        with mock.patch.object(torch.version, "hip", None), mock.patch.object(
            fp8, "_select_fp8_runtime_backend", return_value=fp8._CUDA_DEEP_GEMM
        ), mock.patch.object(
            fp8, "is_deep_gemm_e8m0_used", return_value=False
        ), mock.patch.object(
            attention, "is_deep_gemm_e8m0_used", return_value=True
        ), mock.patch.object(
            fp8_quant, "_transform_scale_ue8m0", side_effect=pack_cpu
        ):
            layer, data = load_checkpoint(
                "fp8_block_online", "cpu", compressed=True, random_online=True
            )
        fused = torch.cat([data["q_a_proj.weight"], data["kv_a_proj_with_mqa.weight"]])
        for source, w, s in (
            (
                fused,
                layer._fused_qkv_a_runtime.weight,
                layer._fused_qkv_a_runtime.weight_scale,
            ),
            (data["kv_b_proj.weight"], layer._kv_b_runtime_w, layer._kv_b_runtime_s),
        ):
            expected_w, scale = fp8_quant.per_block_cast_to_fp8(source, True)
            self.assertTrue(
                torch.equal(w.view(torch.uint8), expected_w.view(torch.uint8))
            )
            torch.testing.assert_close(s, pack_cpu(scale, w.shape[0]))
            first, sf = fp8_quant.per_block_cast_to_fp8(source, False)
            rounded = (
                first.float() * sf.repeat_interleave(128, 0).repeat_interleave(128, 1)
            ).bfloat16()
            old_w, _ = fp8_quant.per_block_cast_to_fp8(rounded, True)
            self.assertFalse(torch.equal(w.view(torch.uint8), old_w.view(torch.uint8)))
        # Decode must not use either of the quantized prefill views.
        source = data["kv_b_proj.weight"].view(2, 256, 128)
        torch.testing.assert_close(layer._kc_w, source[:, :128])
        torch.testing.assert_close(layer._vc_w, source[:, 128:].transpose(1, 2))

    @unittest.skipUnless(
        torch.cuda.is_available()
        and not torch.version.hip
        and torch.cuda.get_device_capability()[0] in (10, 12),
        "requires Blackwell DeepGEMM",
    )
    def test_blackwell_real_fused_fp8_forward(self):
        for method in ("fp8_block", "fp8_block_online"):
            with self.subTest(method=method):
                self._check_blackwell_fused_fp8(method)

    def _check_blackwell_fused_fp8(self, method):
        layer, data = load_checkpoint(
            method, "cuda:0", compressed=True, fp8_source=(method == "fp8_block")
        )
        fused = layer._fused_qkv_a_runtime
        self.assertEqual(fused.weight_scale.dtype, torch.int32)
        self.assertGreater(fused.weight_scale.shape[1], 1)
        self.assertEqual(fused.weight_scale.stride(0), 1)
        # These source scales and probes are powers of two, exactly
        # representable before and after Blackwell UE8M0 requantization.
        x = torch.zeros(16, 1024, dtype=torch.bfloat16, device="cuda")
        x[torch.arange(16), torch.arange(16) * 61] = 1
        expected = []
        for key in ("q_a_proj", "kv_a_proj_with_mqa"):
            w = data[key + ".weight"].float()
            if method == "fp8_block":
                w *= (
                    data[key + ".weight_scale_inv"]
                    .repeat_interleave(128, 0)
                    .repeat_interleave(128, 1)
                )
            expected.append(x.float() @ w.cuda().T)
        torch.testing.assert_close(fused(x), torch.cat(expected, dim=-1).bfloat16())

    def test_packed_fp8_fused_projection_preserves_tma_storage(self):
        # Stand-ins for the two CUDA-only kernels. All checkpoint reading, FP8
        # projection fusion, buffer registration and forward dispatch are real.
        observed = {}

        def pack(w, s):
            n, k = w.shape
            # Two packed columns and an intentionally padded leading dimension.
            result = torch.empty(k // 512, n + 4, dtype=torch.int32).T[:n]
            exps = (torch.log2(s).int() + 127).repeat_interleave(128, dim=0)[:n]
            shifts = torch.arange(4, dtype=torch.int32) * 8
            result.copy_((exps.reshape(n, -1, 4) << shifts).sum(dim=-1).int())
            observed["storage"] = result
            return w, result

        def gemm(a, b, out, **kwargs):
            w, s = b
            self.assertEqual(s.data_ptr(), observed["storage"].data_ptr())
            self.assertEqual(s.stride(), (1, w.shape[0] + 4))
            shifts = torch.arange(4, dtype=torch.int32) * 8
            exps = ((s[..., None] >> shifts) & 255).flatten(1)
            scales = (2.0 ** (exps - 127).float()).repeat_interleave(128, dim=1)
            out.copy_(a[0].float() @ (w.float() * scales).T)

        with mock.patch.object(torch.version, "hip", None), mock.patch.object(
            fp8, "_select_fp8_runtime_backend", return_value=fp8._CUDA_DEEP_GEMM
        ), mock.patch.object(
            fp8, "is_deep_gemm_e8m0_used", return_value=False
        ), mock.patch.object(
            attention, "is_deep_gemm_e8m0_used", return_value=True
        ), mock.patch.object(
            attention, "_resolve_requant_weight_ue8m0", return_value=pack
        ), mock.patch.object(
            attention,
            "_resolve_sgl_per_token_group_quant",
            return_value=lambda x, **kw: (x, None),
        ), mock.patch.object(
            attention, "_resolve_fp8_gemm_nt", return_value=gemm
        ):
            layer, data = load_checkpoint(
                "fp8_block", "cpu", compressed=True, fp8_source=True
            )
            x = torch.zeros(2, 1024, dtype=torch.bfloat16)
            x[0, 64], x[1, 896] = 1, -1
            expected = []
            for key in ("q_a_proj", "kv_a_proj_with_mqa"):
                s = (
                    data[key + ".weight_scale_inv"]
                    .repeat_interleave(128, 0)
                    .repeat_interleave(128, 1)
                )
                expected.append(x.float() @ (data[key + ".weight"].float() * s).T)
            torch.testing.assert_close(
                layer._fused_qkv_a_runtime(x), torch.cat(expected, 1).bfloat16()
            )

    @unittest.skipUnless(
        torch.cuda.is_available() and not torch.version.hip, "requires NVIDIA W8A16"
    )
    def test_int8_checkpoint_fused_and_prefill_projections(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
            MlaFlashInferPrefillOp,
        )

        def reference(x, w):
            w = w.to(x.device)
            s = w.abs().amax(dim=1).clamp_min(1e-8) / 128.0
            qw = (w / s[:, None]).round().clamp(-128, 127)
            return ((x.float() @ qw.float().T) * s.float()).bfloat16()

        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                layer, data = load_checkpoint("INT8", "cuda:0", compressed=compressed)
                q = data[("q_a_proj" if compressed else "q_proj") + ".weight"]
                a = torch.cat((q, data["kv_a_proj_with_mqa.weight"]))
                x = torch.eye(160, device="cuda", dtype=torch.bfloat16)[::19]
                kv = data["kv_b_proj.weight"]
                ref = kv.view(2, 64, 64)
                torch.testing.assert_close(layer._kc_w.cpu(), ref[:, :32])
                torch.testing.assert_close(
                    layer._vc_w.cpu(), ref[:, 32:].transpose(1, 2)
                )
                prefill = MlaFlashInferPrefillOp.__new__(MlaFlashInferPrefillOp)
                prefill.weights = [layer._build_mla_kernel_weights()]
                prefill.quant_config = layer.quant_config
                prefill.num_heads = 2
                prefill.qk_nope_head_dim = prefill.v_head_dim = 32
                layer.release_checkpoint_only_weights()
                torch.testing.assert_close(
                    layer._fused_qkv_a_runtime(x), reference(x, a)
                )
                y = x[:, :64].contiguous()
                torch.testing.assert_close(
                    prefill._project_kv_b(y, 0), reference(y, kv)
                )


if __name__ == "__main__":
    unittest.main()
