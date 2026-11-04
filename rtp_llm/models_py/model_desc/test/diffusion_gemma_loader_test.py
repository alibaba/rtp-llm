"""DiffusionGemma weight-info / loader round-trip test.

Builds a tiny HF-layout safetensors checkpoint (the exact key names from
google/diffusiongemma-26B-A4B-it: model.decoder.* backbone, packed 3D
experts, router scale/per_expert_scale, per-mode layer scalars,
self-conditioning, encoder.language_model tied references) and loads it
through the real ModelLoader with DiffusionGemmaWeightInfo. Verifies the
fusions (qkv merge with v=k duplication for global layers, [in, out]
transposes, tied lm_head -> embedding) elementwise.
"""

import ctypes
import glob
import json
import os
import tempfile
from unittest import TestCase, main

import torch


def _preload_libcuda_stub() -> None:
    extra_dirs = [d for d in os.environ.get("CUDA_STUB_LIB", "").split(":") if d]
    for soname in ("libcuda.so.1", "libnvidia-ml.so.1"):
        candidates = [soname]
        for base in extra_dirs:
            candidates.append(os.path.join(base, soname))
        candidates += sorted(glob.glob(f"/usr/local/cuda*/lib64/stubs/{soname}"))
        for path in candidates:
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue


_preload_libcuda_stub()

from safetensors.torch import save_file  # noqa: E402

from rtp_llm.config.model_config import ModelConfig  # noqa: E402
from rtp_llm.models.diffusion_gemma import DiffusionGemmaConfig  # noqa: E402
from rtp_llm.models.diffusion_gemma_weight import DiffusionGemmaWeightInfo  # noqa: E402
from rtp_llm.utils.model_weight import W  # noqa: E402


H = 32
SLIDING_DIM = 16
SLIDING_KV = 2
GLOBAL_DIM = 32
GLOBAL_KV = 1
HEADS = 4
INTER = 24
MOE_INTER = 12
EXPERTS = 6
LAYERS = 3
VOCAB = 64
EPS = 1e-6


def _hf_layout(layer_type, seed):
    generator = torch.Generator().manual_seed(seed)
    kv = GLOBAL_KV if layer_type == "full_attention" else SLIDING_KV
    head_dim = GLOBAL_DIM if layer_type == "full_attention" else SLIDING_DIM
    weights = {
        "self_attn.q_proj.weight": torch.randn(HEADS * head_dim, H, generator=generator) * 0.2,
        "self_attn.k_proj.weight": torch.randn(kv * head_dim, H, generator=generator) * 0.2,
        "self_attn.o_proj.weight": torch.randn(H, HEADS * head_dim, generator=generator) * 0.2,
        "self_attn.q_norm.weight": torch.ones(head_dim) + 0.02 * torch.randn(head_dim, generator=generator),
        "self_attn.k_norm.weight": torch.ones(head_dim) + 0.02 * torch.randn(head_dim, generator=generator),
        "input_layernorm.weight": torch.ones(H) + 0.05 * torch.randn(H, generator=generator),
        "post_attention_layernorm.weight": torch.ones(H) + 0.05 * torch.randn(H, generator=generator),
        "pre_feedforward_layernorm.weight": torch.ones(H) + 0.05 * torch.randn(H, generator=generator),
        "post_feedforward_layernorm.weight": torch.ones(H) + 0.05 * torch.randn(H, generator=generator),
        "pre_feedforward_layernorm_2.weight": torch.ones(H) + 0.05 * torch.randn(H, generator=generator),
        "post_feedforward_layernorm_1.weight": torch.ones(H) + 0.05 * torch.randn(H, generator=generator),
        "post_feedforward_layernorm_2.weight": torch.ones(H) + 0.05 * torch.randn(H, generator=generator),
        "mlp.gate_proj.weight": torch.randn(INTER, H, generator=generator) * 0.2,
        "mlp.up_proj.weight": torch.randn(INTER, H, generator=generator) * 0.2,
        "mlp.down_proj.weight": torch.randn(H, INTER, generator=generator) * 0.2,
        "experts.gate_up_proj": torch.randn(EXPERTS, 2 * MOE_INTER, H, generator=generator) * 0.2,
        "experts.down_proj": torch.randn(EXPERTS, H, MOE_INTER, generator=generator) * 0.2,
        "router.proj.weight": torch.randn(EXPERTS, H, generator=generator) * 0.5,
        "router.scale": torch.ones(H) + 0.1 * torch.randn(H, generator=generator),
        "router.per_expert_scale": torch.ones(EXPERTS) + 0.1 * torch.randn(EXPERTS, generator=generator),
        "layer_scalar": 0.9 + 0.2 * torch.rand(1, generator=generator),
    }
    if layer_type == "sliding_attention":
        weights["self_attn.v_proj.weight"] = (
            torch.randn(kv * head_dim, H, generator=generator) * 0.2
        )
    return weights


class DiffusionGemmaLoaderTest(TestCase):
    def test_weight_info_maps_hf_checkpoint(self):
        config = DiffusionGemmaConfig()
        config.num_layers = LAYERS
        config.hidden_size = H
        config.vocab_size = VOCAB
        config.max_seq_len = 128
        config.layernorm_eps = EPS
        config.attn_config.head_num = HEADS
        config.attn_config.kv_head_num = SLIDING_KV
        config.attn_config.size_per_head = SLIDING_DIM
        config.inter_size = INTER
        config.moe_inter_size = MOE_INTER
        config.expert_num = EXPERTS
        config.moe_k = 3
        config.dgm_layer_types = ["sliding_attention", "sliding_attention", "sliding_attention"]
        config.dgm_head_dim = SLIDING_DIM
        config.dgm_kv_head_num = SLIDING_KV

        layer_types = ["sliding_attention", "full_attention", "sliding_attention"]
        config.dgm_layer_types = layer_types

        hf_tensors = {}
        torch.manual_seed(3)
        embedding = torch.randn(VOCAB, H) * 0.1
        hf_tensors["model.decoder.embed_tokens.weight"] = embedding
        hf_tensors["model.decoder.norm.weight"] = torch.ones(H) + 0.05 * torch.randn(H)
        hf_tensors["model.decoder.self_conditioning.pre_norm.weight"] = (
            torch.ones(H) + 0.05 * torch.randn(H)
        )
        hf_tensors["model.decoder.self_conditioning.gate_proj.weight"] = (
            torch.randn(INTER, H) * 0.2
        )
        hf_tensors["model.decoder.self_conditioning.up_proj.weight"] = (
            torch.randn(INTER, H) * 0.2
        )
        hf_tensors["model.decoder.self_conditioning.down_proj.weight"] = (
            torch.randn(H, INTER) * 0.2
        )
        layers_hf = []
        for i, layer_type in enumerate(layer_types):
            hf = _hf_layout(layer_type, 200 + i)
            for key, value in hf.items():
                hf_tensors[f"model.decoder.layers.{i}.{key}"] = value
            # encoder-side tied layer scalar (independent buffer)
            enc_scalar = 1.1 + 0.2 * torch.rand(1, generator=torch.Generator().manual_seed(300 + i))
            hf_tensors[f"model.encoder.language_model.layers.{i}.layer_scalar"] = enc_scalar
            layers_hf.append((hf, enc_scalar))

        with tempfile.TemporaryDirectory() as ckpt:
            save_file(hf_tensors, os.path.join(ckpt, "model.safetensors"))
            with open(os.path.join(ckpt, "config.json"), "w") as f:
                json.dump({"canvas_length": 6, "text_config": {}}, f)

            from rtp_llm.ops import HWKernelConfig, ParallelismConfig

            weight_info = DiffusionGemmaWeightInfo(
                model_config=config,
                parallelism_config=ParallelismConfig(),
                hw_kernel_config=HWKernelConfig(),
                kv_cache_config=None,
            )
            info = weight_info._get_weight_info()

            # The weight info must reference exactly the checkpoint keys we
            # wrote (nothing more, nothing less).
            referenced = set()
            all_modules = list(info.weights) + [
                w for layer in info.layer_weights for w in layer
            ]
            for module in all_modules:
                for ck in module.weights:
                    name = ck.name if hasattr(ck, "name") else str(ck)
                    referenced.add(name.format(i=0))
            missing = referenced - set(hf_tensors)
            self.assertEqual(missing, set(), f"weight info references absent keys: {missing}")

            # Fusion checks on layer 0 (sliding) and a synthesized global layer.
            sliding = layers_hf[0][0]
            module_for = {}
            for module in info.layer_weights[0]:
                if module.weights:
                    module_for[module.name] = module

            qkv_module = module_for[W.attn_qkv_w]
            q = sliding["self_attn.q_proj.weight"]
            k = sliding["self_attn.k_proj.weight"]
            v = sliding["self_attn.v_proj.weight"]
            expected_qkv = torch.cat([q.T, k.T, v.T], dim=1).contiguous()
            fused = qkv_module.process_fun([q, k, v])
            torch.testing.assert_close(fused, expected_qkv)

            o_module = module_for[W.attn_o_w]
            torch.testing.assert_close(
                o_module.process_fun([sliding["self_attn.o_proj.weight"]]),
                sliding["self_attn.o_proj.weight"].T.contiguous(),
            )

            # global layer fusion: v slot duplicates k_proj (layer 1 module).
            module_for_g = {}
            for module in info.layer_weights[1]:
                if module.weights:
                    module_for_g[module.name] = module
            global_hf = layers_hf[1][0]
            gq = global_hf["self_attn.q_proj.weight"]
            gk = global_hf["self_attn.k_proj.weight"]
            expected_gqkv = torch.cat([gq.T, gk.T, gk.T], dim=1).contiguous()
            fused_g = module_for_g[W.attn_qkv_w].process_fun([gq, gk])
            torch.testing.assert_close(fused_g, expected_gqkv)
            self.assertNotIn("self_attn.v_proj.weight", global_hf)

            # packed experts stay in HF layout.
            moe_module = module_for[W.moe_w1]
            torch.testing.assert_close(
                moe_module.process_fun([sliding["experts.gate_up_proj"]]),
                sliding["experts.gate_up_proj"],
            )

            # router transpose and raw scales.
            torch.testing.assert_close(
                module_for[W.moe_gate].process_fun([sliding["router.proj.weight"]]),
                sliding["router.proj.weight"].T.contiguous(),
            )
            self.assertIs(module_for[W.dgm_router_scale].process_fun([sliding["router.scale"]]), sliding["router.scale"])

            # per-mode scalars map to their own checkpoint keys.
            enc_module = module_for[W.dgm_layer_scalar_enc]
            dec_module = module_for[W.dgm_layer_scalar_dec]
            self.assertIs(
                enc_module.process_fun([layers_hf[0][1]]), layers_hf[0][1]
            )
            self.assertIs(
                dec_module.process_fun([sliding["layer_scalar"]]), sliding["layer_scalar"]
            )

            # globals: embedding + tied lm_head + self-conditioning set.
            global_names = {module.name for module in info.weights}
            self.assertIn(W.embedding, global_names)
            self.assertIn(W.lm_head, global_names)
            self.assertIn(W.dgm_self_cond_pre_norm, global_names)


if __name__ == "__main__":
    main()
