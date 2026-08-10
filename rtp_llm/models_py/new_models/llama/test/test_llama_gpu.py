import types
import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.llama import LlamaForCausalLM
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.ops.compute_ops import PyModelInputs


class LlamaGpuTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA or ROCm GPU")
    def test_bf16_decoder_forward_matches_torch_reference(self):
        dtype = torch.bfloat16
        config = types.SimpleNamespace(
            model_type="llama",
            num_layers=1,
            vocab_size=8,
            hidden_size=4,
            inter_size=4,
            attn_config=types.SimpleNamespace(
                head_num=2,
                kv_head_num=1,
                size_per_head=2,
            ),
            layernorm_eps=1e-6,
            enable_fp32_lm_head=False,
            tie_word_embeddings=True,
            attention_bias=False,
        )
        parallelism = types.SimpleNamespace(
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            prefill_cp_config=types.SimpleNamespace(
                is_enabled=lambda: False,
                is_prefill_enabled=lambda: False,
            ),
            ffn_disaggregate_config=types.SimpleNamespace(
                enable_ffn_disaggregate=False
            ),
            get_attn_tp_size=lambda: 1,
            get_attn_tp_rank=lambda: 0,
            get_ffn_tp_size=lambda: 1,
            get_ffn_tp_rank=lambda: 0,
        )
        load_config = NewLoaderConfig(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            ffn_tp_size=1,
            ffn_tp_rank=0,
            lm_head_tp_size=1,
            lm_head_tp_rank=0,
            compute_dtype=dtype,
            device="cuda",
            quant_config=QuantizationConfig("none"),
            parallelism_config=parallelism,
        )
        weights = {
            "model.embed_tokens.weight": torch.arange(32, dtype=dtype).reshape(8, 4),
            "model.layers.0.input_layernorm.weight": torch.ones(4, dtype=dtype),
            "model.layers.0.self_attn.q_proj.weight": torch.arange(
                16, dtype=dtype
            ).reshape(4, 4),
            "model.layers.0.self_attn.k_proj.weight": torch.arange(
                8, dtype=dtype
            ).reshape(2, 4),
            "model.layers.0.self_attn.v_proj.weight": torch.arange(
                8, 16, dtype=dtype
            ).reshape(2, 4),
            "model.layers.0.self_attn.o_proj.weight": torch.eye(4, dtype=dtype),
            "model.layers.0.post_attention_layernorm.weight": torch.ones(
                4, dtype=dtype
            ),
            "model.layers.0.mlp.gate_proj.weight": torch.ones(4, 4, dtype=dtype),
            "model.layers.0.mlp.up_proj.weight": torch.full((4, 4), 0.125, dtype=dtype),
            "model.layers.0.mlp.down_proj.weight": torch.full(
                (4, 4), 0.25, dtype=dtype
            ),
            "model.norm.weight": torch.ones(4, dtype=dtype),
        }
        model = LlamaForCausalLM(config, load_config)
        model.load_weights(weights)
        NewModelLoader._validate_loaded_weights(model)
        model = model.cuda().eval()

        class QOnlyFmha:
            fmha_params = None

            def __init__(self, q_size):
                self.q_size = q_size

            def forward(self, qkv, kv_cache, layer_idx):
                return qkv[..., : self.q_size]

        def rms_norm(x, weight):
            normalized = x.float() * torch.rsqrt(
                x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6
            )
            return (normalized * weight.float()).to(dtype)

        input_ids = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
        inputs = PyModelInputs(input_ids=input_ids)
        fmha = QOnlyFmha(model.layers[0].self_attn.qkv_proj.q_size)
        outputs = model(inputs, fmha_impl=fmha)

        device_weights = {name: value.cuda() for name, value in weights.items()}
        hidden = F.embedding(
            input_ids.long(), device_weights["model.embed_tokens.weight"]
        )
        residual = hidden
        hidden = rms_norm(
            hidden, device_weights["model.layers.0.input_layernorm.weight"]
        )
        hidden = F.linear(
            hidden, device_weights["model.layers.0.self_attn.q_proj.weight"]
        )
        hidden = F.linear(
            hidden, device_weights["model.layers.0.self_attn.o_proj.weight"]
        )
        hidden = residual + hidden
        residual = hidden
        hidden = rms_norm(
            hidden,
            device_weights["model.layers.0.post_attention_layernorm.weight"],
        )
        gate = F.linear(hidden, device_weights["model.layers.0.mlp.gate_proj.weight"])
        up = F.linear(hidden, device_weights["model.layers.0.mlp.up_proj.weight"])
        hidden = F.linear(
            F.silu(gate.float()) * up.float(),
            device_weights["model.layers.0.mlp.down_proj.weight"].float(),
        ).to(dtype)
        expected = rms_norm(residual + hidden, device_weights["model.norm.weight"])

        torch.testing.assert_close(
            outputs.hidden_states, expected, rtol=2e-2, atol=2e-2
        )


if __name__ == "__main__":
    unittest.main()
