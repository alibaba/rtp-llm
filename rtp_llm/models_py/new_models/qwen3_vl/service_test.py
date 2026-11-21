import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
    Qwen3VLVisionModel,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

from rtp_llm.models.qwen3_vl import QWen3_VL
from rtp_llm.models.qwen3_vl_moe import QWen3_VL_MOE
from rtp_llm.models_py.new_models.qwen3_vl.language import (
    Qwen3VLForCausalLM,
    Qwen3VLMoeForCausalLM,
)
from rtp_llm.models_py.qwen3_moe_checkpoint_test import weights
from rtp_llm.models_py.serving_route_checkpoint_test import service
from rtp_llm.multimodal.multimodal_mixin_factory import MultimodalMixinFactory


def checkpoint(root, moe, *, vision_hidden_size=16):
    model_type = "qwen3_vl_moe" if moe else "qwen3_vl"
    vision = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=vision_hidden_size,
        intermediate_size=32,
        out_hidden_size=4,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
    )
    vision._attn_implementation = "eager"
    visual = Qwen3VLVisionModel(vision).eval()
    config = dict(
        model_type=model_type,
        vision_start_token_id=3,
        vision_end_token_id=4,
        text_config=dict(
            num_hidden_layers=1,
            hidden_size=4,
            intermediate_size=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=8,
            moe_intermediate_size=3,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            norm_topk_prob=True,
            rope_scaling={"mrope_section": [1, 0, 0]},
        ),
        vision_config=vision.to_dict(),
    )
    (root / "config.json").write_text(json.dumps(config))
    tok = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "<|image_pad|>": 1,
                "<|video_pad|>": 2,
                "<|vision_start|>": 3,
                "<|vision_end|>": 4,
            },
            unk_token="[UNK]",
        )
    )
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=1),
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]"),
        video_processor=Qwen3VLVideoProcessor(patch_size=2, temporal_patch_size=1),
    )
    processor.save_pretrained(root)
    data = weights()
    layer = "model.layers.0.mlp."
    if moe:
        # HF's packed VL expert contract is [E,H,2M] and [E,M,H].
        gateup = [
            torch.cat(
                [
                    data.pop(f"{layer}experts.{i}.{p}_proj.weight")[:3]
                    for p in ("gate", "up")
                ]
            ).T
            for i in range(4)
        ]
        down = [
            data.pop(f"{layer}experts.{i}.down_proj.weight")[:, :3].T for i in range(4)
        ]
        data[layer + "experts.gate_up_proj"] = torch.stack(gateup).contiguous()
        data[layer + "experts.down_proj"] = torch.stack(down).contiguous()
    else:
        data.pop(layer + "gate.weight")
        for i in range(4):
            for p in ("gate", "up", "down"):
                w = data.pop(f"{layer}experts.{i}.{p}_proj.weight")
                if i == 0:
                    data[layer + p + "_proj.weight"] = w
    tensors = {
        k.replace("model.", "model.language_model.", 1): v for k, v in data.items()
    }
    tensors.update(
        {
            "model.visual." + k: v.detach().clone()
            for k, v in visual.state_dict().items()
        }
    )
    save_file(tensors, str(root / "model.safetensors"))
    with torch.no_grad():
        for p in visual.parameters():
            p.copy_(p.half().float())
    return model_type, visual, tensors


class _Cpu:
    def _get_device_str(self):
        return "cpu"


class _Dense(_Cpu, QWen3_VL):
    pass


class _Moe(_Cpu, QWen3_VL_MOE):
    pass


class _CpuPrefill:
    # CPU SDPA for one request at zero rotary positions (identity MRoPE),
    # without a CUDA cache. The actual loaded decoder/FFN are unmodified.
    def forward(self, qkv, kv_cache, layer_id):
        q, k, v = qkv.split((4, 2, 2), dim=-1)
        q = q.reshape(-1, 2, 2).transpose(0, 1)
        k = k.unsqueeze(0).expand(2, -1, -1)
        v = v.unsqueeze(0).expand(2, -1, -1)
        return (
            torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            .transpose(0, 1)
            .contiguous()
        )


class _CpuExpertMath(torch.nn.Module):
    """Test-only BF16/FP32 CPU math in lieu of the GPU-only fused executor."""

    topk_ids_dtype = torch.int64

    def __init__(self, experts):
        super().__init__()
        self.w13 = experts.w13.detach()
        self.w2 = experts.w2.detach()

    def forward(self, hidden_states, topk_weights, topk_ids, **kwargs):
        output = torch.zeros_like(hidden_states)
        for t, row in enumerate(hidden_states):
            for slot, expert in enumerate(topk_ids[t]):
                up, gate = torch.nn.functional.linear(row, self.w13[expert]).chunk(2)
                value = torch.nn.functional.linear(
                    up * torch.nn.functional.silu(gate), self.w2[expert]
                )
                output[t] += value * topk_weights[t, slot]
        return output


class _VLServiceFixture(unittest.TestCase):
    @torch.inference_mode()
    def check_vl(self, *, native_output):
        for moe, server, owner in (
            (False, _Dense, Qwen3VLForCausalLM),
            (True, _Moe, Qwen3VLMoeForCausalLM),
        ):
            with tempfile.TemporaryDirectory() as temp:
                model_type, reference, tensors = checkpoint(Path(temp), moe)
                for method in ("scratch", "fastsafetensors"):
                    with self.subTest(model_type=model_type, method=method):
                        config = server._create_config(temp)
                        config.model_type = model_type
                        config.tokenizer_path = temp
                        config.data_type = "fp32"
                        fixture = service(temp, method)
                        parallel = fixture.parallelism_config
                        parallel.local_world_size = 1
                        parallel.pp_stage_layer_counts = []
                        model = server.from_config(
                            model_config=config,
                            parallelism_config=parallel,
                            kv_cache_config=SimpleNamespace(
                                multi_task_prompt="", multi_task_prompt_str=""
                            ),
                            fmha_config=None,
                            moe_config=None,
                            max_generate_batch_size=1,
                            hw_kernel_config=SimpleNamespace(
                                enable_cuda_graph=False, use_swizzleA=False
                            ),
                            load_method=method,
                            merge_lora=False,
                            vit_config=None,
                        )
                        self.assertIsInstance(model.py_model, owner)
                        self.assertEqual(
                            model.model_weights_loader.loaded_method.value, method
                        )
                        torch.testing.assert_close(
                            model.py_model.lm_head.weight, tensors["lm_head.weight"]
                        )
                        if moe:
                            torch.testing.assert_close(
                                model.py_model.layers[0].mlp.experts.w2[2],
                                tensors[
                                    "model.language_model.layers.0.mlp.experts.down_proj"
                                ][2].T,
                            )
                        engine = SimpleNamespace(
                            load_config=SimpleNamespace(load_method=method)
                        )
                        mm = MultimodalMixinFactory._create_multimodal_mixin(
                            config, engine, None, "cpu"
                        )
                        pixels = torch.randn(4, 12)
                        grid = torch.tensor([[1, 2, 2]])
                        features, _, extra = mm.mm_part.embedding([pixels, grid])
                        expected = reference(pixels, grid_thw=grid)
                        torch.testing.assert_close(features, expected.pooler_output)
                        torch.testing.assert_close(
                            extra, torch.stack(expected.deepstack_features).reshape(-1)
                        )
                        inputs = SimpleNamespace(
                            input_ids=torch.tensor([1, 128]),
                            embedding_inputs=SimpleNamespace(
                                text_tokens_mask=torch.tensor([1, 0])
                            ),
                            multimodal_inputs=SimpleNamespace(
                                multimodal_features=[features],
                                mm_features_locs=torch.tensor([1]),
                                mm_extra_input=[extra],
                            ),
                        )
                        hidden, deepstack, locs = (
                            model.py_model._embed_multimodal_inputs(inputs)
                        )
                        torch.testing.assert_close(
                            hidden[0],
                            tensors["model.language_model.embed_tokens.weight"][1],
                        )
                        torch.testing.assert_close(hidden[1:], features)
                        result = model.py_model._inject_deepstack_after_layer(
                            hidden.clone(), deepstack, locs, 0
                        )
                        torch.testing.assert_close(
                            result[1:], features + extra.reshape_as(features)
                        )
                        if not native_output:
                            continue
                        inputs.capture_hidden_states = False
                        prefill = _CpuPrefill()
                        # Independently assemble main's reference layer/deepstack/final
                        # ordering using checkpoint-derived inputs, not the VL forward.
                        text_and_image = torch.cat(
                            (
                                tensors["model.language_model.embed_tokens.weight"][
                                    1:2
                                ],
                                expected.pooler_output,
                            )
                        )
                        layer = model.py_model.layers[0]
                        if moe:
                            experts = layer.mlp.experts
                            experts.fused_moe = _CpuExpertMath(experts)
                            ref_hidden, residual = layer(
                                text_and_image,
                                torch.zeros_like(text_and_image),
                                prefill,
                                None,
                            )
                        else:
                            ref_hidden = layer(text_and_image, prefill, None)
                        ref_hidden[1:] = ref_hidden[1:] + expected.deepstack_features[0]
                        ref_hidden = (
                            model.py_model.norm(ref_hidden, residual)[0]
                            if moe
                            else model.py_model.norm(ref_hidden)
                        )
                        actual = model.py_model(inputs, prefill).hidden_states
                        torch.testing.assert_close(actual, ref_hidden)
                        self.assertTrue(torch.isfinite(actual).all())


class PublicVLTest(_VLServiceFixture):
    def test_checkpoint_vision_and_deepstack(self):
        self.check_vl(native_output=False)


class VLNativeOutputTest(_VLServiceFixture):
    def test_vl(self):
        self.check_vl(native_output=True)


if __name__ == "__main__":
    unittest.main()
