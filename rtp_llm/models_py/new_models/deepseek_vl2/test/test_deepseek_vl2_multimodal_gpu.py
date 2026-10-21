"""Checkpoint-to-logits integration using the real CUDA MLA backend.

The checkpoint is seeded and reduced to one dense language layer, but keeps
kernel-supported MLA dimensions and the actual SigLIP architecture. No vision,
attention, loader, or language forward is replaced by a test implementation.
"""

import gc
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferPrefillImpl,
)
from rtp_llm.models_py.new_models.deepseek_vl2.test.test_deepseek_vl2_load import (
    _raw_language_config,
    _vision_config,
    _write_config,
)
from rtp_llm.models_py.new_models.deepseek_vl2.vision import (
    DeepSeekVLV2VisionModel,
    load_deepseek_vl2_vision,
)
from rtp_llm.ops import FMHAConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs
from safetensors.torch import save_file


def _language_checkpoint():
    raw = _raw_language_config(use_mla=True)
    raw.update(
        hidden_size=256,
        intermediate_size=512,
        max_position_embeddings=256,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        torch_dtype="bfloat16",
    )
    shapes = {
        "language.model.embed_tokens.weight": (8, 256),
        "language.model.layers.0.input_layernorm.weight": (256,),
        "language.model.layers.0.post_attention_layernorm.weight": (256,),
        "language.model.layers.0.self_attn.q_proj.weight": (384, 256),
        "language.model.layers.0.self_attn.kv_a_proj_with_mqa.weight": (576, 256),
        "language.model.layers.0.self_attn.kv_a_layernorm.weight": (512,),
        "language.model.layers.0.self_attn.kv_b_proj.weight": (512, 512),
        "language.model.layers.0.self_attn.o_proj.weight": (256, 256),
        "language.model.layers.0.mlp.gate_proj.weight": (512, 256),
        "language.model.layers.0.mlp.up_proj.weight": (512, 256),
        "language.model.layers.0.mlp.down_proj.weight": (256, 512),
        "language.model.norm.weight": (256,),
        "language.lm_head.weight": (8, 256),
    }
    weights = {
        name: (
            torch.ones(shape, dtype=torch.bfloat16)
            if len(shape) == 1
            else (torch.randn(shape) * 0.03).to(torch.bfloat16)
        )
        for name, shape in shapes.items()
    }
    return raw, weights


def _language_model_config(model_path):
    config = ModelConfig()
    config.model_type = "deepseek_vl_v2"
    config.ckpt_path = model_path
    config.hidden_size = 256
    config.num_layers = 1
    config.vocab_size = 8
    config.max_seq_len = 256
    config.layernorm_eps = 1e-6
    config.quant_config = None
    config.data_type = "bf16"
    config.tie_word_embeddings = False
    config.enable_fp32_lm_head = False
    config.attn_config.head_num = 2
    config.attn_config.kv_head_num = 2
    config.attn_config.size_per_head = 192
    config.attn_config.use_mla = True
    config.attn_config.q_lora_rank = 0
    config.attn_config.kv_lora_rank = 512
    config.attn_config.nope_head_dim = 128
    config.attn_config.rope_head_dim = 64
    config.attn_config.v_head_dim = 128
    config.attn_config.tokens_per_block = 64
    config.attn_config.kernel_tokens_per_block = 64
    config.attn_config.softmax_extra_scale = 1.0
    return config


class DeepSeekVLV2MultimodalGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError("This integration target requires its assigned CUDA GPU")

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def test_checkpoint_to_image_conditioned_language_logits(self):
        torch.manual_seed(1279)
        vision_config = _vision_config()
        vision_config["projector_config"]["n_embed"] = 256
        raw, weights = _language_checkpoint()
        with tempfile.TemporaryDirectory() as model_path:
            source = DeepSeekVLV2VisionModel(vision_config, torch.bfloat16).eval()
            weights.update(source.state_dict())
            _write_config(model_path, raw)
            save_file(weights, f"{model_path}/model.safetensors")
            del source, weights

            vision = load_deepseek_vl2_vision(
                vision_config=vision_config,
                model_path=model_path,
                compute_dtype=torch.bfloat16,
                device="cuda:0",
            )
            model = NewModelLoader(
                model_config=_language_model_config(model_path),
                load_config=NewLoaderConfig(
                    device="cuda:0",
                    compute_dtype=torch.bfloat16,
                    parallelism_config=ParallelismConfig(),
                    fmha_config=FMHAConfig(),
                ),
                model_path=model_path,
            ).load()

        self.assertFalse(model.training)
        self.assertFalse(vision.training)
        features = vision(torch.randn(2, 3, 384, 384, device="cuda"))
        self.assertEqual(tuple(features.shape), (2, 196, 256))
        self.assertTrue(torch.isfinite(features).all())
        token_count = 198  # BOS, 196 image features, one text token.
        attention_inputs = PyAttentionInputs()
        attention_inputs.is_prefill = True
        attention_inputs.input_lengths = torch.tensor([token_count], dtype=torch.int32)
        attention_inputs.prefix_lengths = torch.zeros(1, dtype=torch.int32)
        attention_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        block_ids = torch.arange(4, dtype=torch.int32).reshape(1, 4)
        attention_inputs.kv_cache_block_id = block_ids
        attention_inputs.kv_cache_block_id_device = block_ids.cuda()
        attention_inputs.kv_cache_kernel_block_id = block_ids
        attention_inputs.kv_cache_kernel_block_id_device = block_ids.cuda()
        attention_inputs.cu_seqlens = torch.tensor([0, token_count], dtype=torch.int32)
        attention_inputs.cu_seqlens_device = attention_inputs.cu_seqlens.cuda()
        attention_inputs.cu_kv_seqlens_device = attention_inputs.cu_seqlens_device
        attention_inputs.total_tokens = token_count
        text_mask = torch.zeros(token_count, dtype=torch.bool, device="cuda")
        text_mask[0] = text_mask[-1] = True
        inputs = types.SimpleNamespace(
            input_ids=torch.ones(token_count, dtype=torch.long, device="cuda"),
            attention_inputs=attention_inputs,
            embedding_inputs=types.SimpleNamespace(text_tokens_mask=text_mask),
            multimodal_inputs=types.SimpleNamespace(
                multimodal_features=[features[0]],
                mm_features_locs=torch.tensor([1], dtype=torch.int32),
            ),
        )
        fmha = model.prepare_fmha_impl(inputs)
        self.assertIsInstance(fmha, MlaFlashInferPrefillImpl)
        with patch.object(fmha, "forward", wraps=fmha.forward) as attention_forward:
            first = model.lm_head(model(inputs, fmha).hidden_states).float()
            inputs.multimodal_inputs.multimodal_features = [features[1]]
            second = model.lm_head(model(inputs, fmha).hidden_states).float()
        self.assertEqual(attention_forward.call_count, 2)
        self.assertEqual(tuple(first.shape), (token_count, 8))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.isfinite(second).all())
        # The last token is identical text in both requests: only attending to
        # the different preceding images can change its output.
        self.assertFalse(torch.equal(first[-1], second[-1]))


if __name__ == "__main__":
    unittest.main()
