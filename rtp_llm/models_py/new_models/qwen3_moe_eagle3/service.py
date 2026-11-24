"""Speculative executor owner for NVIDIA's full-vocabulary Qwen3 MoE EAGLE3."""

from functools import partial

import torch

from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.linear import QKVParallelLinear, RowParallelLinear
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.model_base import select_fmha_impl_for_layer
from rtp_llm.models_py.new_models.qwen3.language import Qwen3MLP
from rtp_llm.models_py.new_models.qwen3_moe_eagle3.body import Eagle3DraftBody
from rtp_llm.models_py.weight_mapper import WeightsMapper


class _Attention(RtpModule):
    def __init__(self, hidden, heads, kv_heads, dim, dtype, *, size, rank):
        super().__init__()
        kwargs = dict(
            tp_size=size,
            tp_rank=rank,
            bias=False,
            params_dtype=dtype,
            quant_config=None,
        )
        self.qkv_proj = QKVParallelLinear(
            hidden_size=2 * hidden,
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=dim,
            prefix="midlayer.self_attn.qkv_proj",
            **kwargs,
        )
        self.o_proj = RowParallelLinear(
            input_size=heads * dim,
            output_size=hidden,
            prefix="midlayer.self_attn.o_proj",
            **kwargs,
        )
        self.local_width = self.qkv_proj.q_size

    def forward(self, x, fmha, cache):
        attended = fmha.forward(self.qkv_proj(x), cache, 0)
        return self.o_proj(attended.reshape(x.shape[0], self.local_width))


class Qwen3MoeEagle3ForCausalLM(GptModelBase, RtpModule):
    runtime_parameter_paths = {
        "embedding": "embed_tokens.weight",
        "lm_head": "lm_head.weight",
    }
    WEIGHTS_MAPPER = WeightsMapper(
        exact_mapping={"model.embed_tokens.weight": "embed_tokens.weight"},
        prefix_mapping={key: "body." + key for key in ("fc.", "midlayer.", "norm.")},
    )

    def __init__(self, config, load_config):
        super().__init__(
            config,
            load_config.parallelism_config,
            weight=None,
            max_generate_batch_size=load_config.max_generate_batch_size,
            fmha_config=load_config.fmha_config,
            py_hw_kernel_config=load_config.py_hw_kernel_config,
        )
        if getattr(load_config.quant_config, "quant_type", "none") != "none":
            raise ValueError(
                "Qwen3 EAGLE3 release draft requires unquantized floating weights"
            )
        h, dtype = config.hidden_size, load_config.compute_dtype
        attn = config.attn_config
        self.capture_ids = tuple(config.capture_aux_hidden_layer_ids)
        ordered = sorted(self.capture_ids)
        self.slot_order = tuple(ordered.index(i) for i in self.capture_ids)

        def make_mlp(hidden, intermediate, dtype):
            return Qwen3MLP(
                hidden,
                intermediate,
                load_config.ffn_tp_size,
                load_config.ffn_tp_rank,
                params_dtype=dtype,
                prefix="midlayer.mlp",
            )

        self.body = Eagle3DraftBody(
            h,
            config.inter_size,
            attn.head_num,
            attn.kv_head_num,
            attn.size_per_head,
            config.layernorm_eps,
            dtype,
            attention=partial(
                _Attention, size=load_config.attn_tp_size, rank=load_config.attn_tp_rank
            ),
            mlp=make_mlp,
        )
        with torch.device("meta"):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                h,
                load_config.attn_tp_size,
                load_config.attn_tp_rank,
                dtype,
                allow_vocab_padding=True,
            )
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                h,
                load_config.lm_head_tp_size,
                load_config.lm_head_tp_rank,
                torch.float32 if config.enable_fp32_lm_head else dtype,
            )
        self.requires_grad_(False)

    def prepare_borrowed_parameters(self, aliases):
        if set(aliases) != set(self.runtime_parameter_paths.values()):
            raise ValueError("EAGLE3 requires both target vocabulary parameters")

    def load_weights(self, weights):
        iterator = weights.items() if isinstance(weights, dict) else weights
        super().load_weights(self.WEIGHTS_MAPPER.apply(iterator))

    def validate_runtime_device(self, device):
        self.prepare_borrowed_parameters(
            getattr(self, "_rtp_borrowed_parameter_bindings", {})
        )

    def runtime_weight_view(self):
        return {
            role: self.get_parameter(path)
            for role, path in self.runtime_parameter_paths.items()
        }

    def project_target_hidden(self, captured):
        if self.slot_order != (0, 1, 2):
            blocks = captured.chunk(3, -1)
            captured = torch.cat([blocks[i] for i in self.slot_order], -1)
        return self.body.fc(captured)

    def forward(self, inputs, fmha_impl=None):
        from rtp_llm.ops.compute_ops import PyModelOutputs

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        state = inputs.input_hiddens
        if state is None or state.shape[-1] != self.body.hidden_size:
            raise ValueError(
                "EAGLE3 executor must supply projected target or recurrent draft states"
            )
        out = self.body(
            self.embed_tokens(inputs.input_ids),
            state,
            select_fmha_impl_for_layer(fmha_impl, self.kv_cache, 0),
            self.kv_cache.get_layer_cache(0) if self.kv_cache is not None else None,
        )
        self._mtp_target_hidden_states = out.recurrent_hidden
        result = PyModelOutputs(out.logits_input)
        result.mtp_target_hidden_states = out.recurrent_hidden
        return result
