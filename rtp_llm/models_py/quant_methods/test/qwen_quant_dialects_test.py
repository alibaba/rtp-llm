"""Load complete Qwen checkpoint dialects into the production CPU owners."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import (
    AWQConfig,
    GPTQConfig,
    OmniQuantConfig,
    SmoothQuantConfig,
)
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.ops import ParallelismConfig


class QwenQuantDialectsTest(unittest.TestCase):
    def check_checkpoint(self, method, *, trt=False, backend="scratch"):
        h, f = 32, 64
        group = method in ("awq", "gptq")
        omni = method == "omni_quant"
        source = (
            (AWQConfig if method == "awq" else GPTQConfig)(
                bits=4, group_size=16, is_quanted=True
            )
            if group
            else OmniQuantConfig()
            if omni
            else SmoothQuantConfig()
        )
        cfg = SimpleNamespace(
            model_type="qwen_2" if trt else "qwen_7b",
            quant_config=source,
            activation_type="SiGLU",
            num_layers=1,
            is_mtp=False,
            vocab_size=h,
            hidden_size=h,
            inter_size=f,
            layernorm_eps=1e-6,
            attn_config=SimpleNamespace(head_num=4, kv_head_num=4, size_per_head=8),
            enable_fp32_lm_head=False,
            tie_word_embeddings=False,
            capture_aux_hidden_layer_ids=[],
            hidden_state_capture_layer_ids=[],
        )
        embedding = torch.arange(h * h).reshape(h, h).half()
        values = {
            "transformer.vocab_embedding.weight"
            if trt
            else "transformer.wte.weight": embedding,
            "transformer.ln_f.weight": torch.ones(h).half(),
            "lm_head.weight": -embedding,
        }
        p = "transformer.layers.0." if trt else "transformer.h.0."
        for norm in ("input_layernorm", "post_layernorm") if trt else ("ln_1", "ln_2"):
            values[p + norm + ".weight"] = torch.ones(h).half()
            if omni:
                values[p + norm + ".bias"] = torch.arange(h).half() / 16
        names = (
            ("attention.qkv", "attention.dense", "mlp.fc", "mlp.gate", "mlp.proj")
            if trt
            else ("attn.c_attn", "attn.c_proj", "mlp.w2", "mlp.w1", "mlp.c_proj")
        )
        params = []
        for i, (name, k, n) in enumerate(
            zip(names, (h, h, h, h, f), (3 * h, h, f, f, h))
        ):
            key = p + name + "."
            if group:
                # Different words per projection/row/column detect swapped
                # gate/up, and GPTQ (packed input) vs AWQ (packed output).
                dims = (k, n // 8) if method == "awq" else (k // 8, n)
                expected = {
                    "qweight": torch.arange(
                        dims[0] * dims[1], dtype=torch.int32
                    ).reshape(dims)
                    + 127 * i,
                    "qzeros": torch.arange(k // 16 * n // 8, dtype=torch.int32).reshape(
                        k // 16, n // 8
                    )
                    + 11 * i,
                    "scales": (
                        torch.arange(k // 16 * n).reshape(k // 16, n).half() + 1 + 7 * i
                    )
                    / 64,
                }
                values.update(
                    {key + field: tensor for field, tensor in expected.items()}
                )
                if method == "gptq":
                    values[key + "g_idx"] = torch.arange(k, dtype=torch.int32) // 16
            else:
                weight = ((torch.arange(n * k).reshape(n, k) + i * 19) % 123 - 59).to(
                    torch.int8
                )
                scale = (torch.arange(n).float() + i + 1) / 32
                expected = {"weight": weight, "weight_scale": scale.view(-1, 1)}
                stored = weight if trt and i in (0, 1, 4) else weight.t().contiguous()
                if not trt and i == 0:
                    # Exact grouped MHA shapes emitted by hf_qwen_convert.
                    stored, scale = stored.reshape(k, 3, h), scale.reshape(3, h)
                values[key + ("weight" if trt else "qweight")] = stored
                values[key + ("per_channel_scale" if trt else "scales")] = (
                    scale.unsqueeze(0) if trt else scale
                )
                if i == 1 or (i == 4 and not omni):
                    smooth = 1 + torch.arange(k).float() / 64
                    values[key + "smoother"] = smooth.unsqueeze(0) if trt else smooth
                if omni and i == 1:
                    values[key + "shift"] = torch.arange(k).float() / 128
            if i == 0 or (omni and i in (1, 2, 3)):
                values[key + "bias"] = (torch.arange(n).half() + 1 + i) / 8
            params.append(expected)
        with tempfile.TemporaryDirectory() as directory:
            # Old converter emits a PyTorch archive, not an HF shard index.
            if backend == "scratch":
                torch.save(values, Path(directory) / "model.pt")
            else:
                save_file(values, str(Path(directory) / "model.safetensors"))
            loader = NewModelLoader(
                cfg,
                NewLoaderConfig(
                    device="cpu",
                    compute_dtype=torch.float16,
                    load_method=backend,
                    parallelism_config=ParallelismConfig(),
                    quant_config=QuantizationConfig(method, source),
                ),
                model_path=directory,
            )
            model = loader.load()
        self.assertEqual(loader.loaded_method.value, backend)
        layer = model.layers[0]
        for indices, leaf in (
            ((0,), layer.self_attn.qkv_proj),
            ((1,), layer.self_attn.o_proj),
            ((2, 3), layer.mlp.gate_up_proj),
            ((4,), layer.mlp.down_proj),
        ):
            for field in params[indices[0]]:
                expected = torch.cat(
                    [params[i][field] for i in indices], dim=1 if group else 0
                )
                torch.testing.assert_close(
                    getattr(leaf, field), expected, rtol=0, atol=0
                )
            if leaf.bias is not None:
                expected = torch.cat([values[p + names[i] + ".bias"] for i in indices])
                torch.testing.assert_close(
                    leaf.bias.flatten(), expected, rtol=0, atol=0
                )
            for field in ("smoother", "shift"):
                if getattr(leaf, field, None) is not None:
                    torch.testing.assert_close(
                        getattr(leaf, field),
                        values[p + names[indices[0]] + "." + field].flatten(),
                    )
        self.assertTrue(torch.equal(model.embed_tokens.weight, embedding))
        self.assertTrue(torch.equal(model.lm_head.weight, -embedding))

    def test_qwen1_smoothquant(self):
        self.check_checkpoint("smooth_quant")

    def test_qwen1_omniquant(self):
        self.check_checkpoint("omni_quant")

    def test_qwen1_awq(self):
        self.check_checkpoint("awq")

    def test_qwen1_gptq(self):
        self.check_checkpoint("gptq")

    def test_qwen2_trt_smoothquant(self):
        self.check_checkpoint("smooth_quant", trt=True)

    @unittest.skipUnless(
        os.getenv("NEWLOADER_TEST_NATIVE_FST") == "1", "requires the native FST reader"
    )
    def test_fst_roundtrip(self):
        for method, trt in (
            ("smooth_quant", False),
            ("omni_quant", False),
            ("awq", False),
            ("gptq", False),
            ("smooth_quant", True),
        ):
            with self.subTest(method=method, trt=trt):
                self.check_checkpoint(method, trt=trt, backend="fastsafetensors")


if __name__ == "__main__":
    unittest.main()
