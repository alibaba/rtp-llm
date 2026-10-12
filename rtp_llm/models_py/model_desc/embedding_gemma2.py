"""Bidirectional EmbeddingGemma 2 layers with projection-only PLE.

Reuse Gemma4 norm/RoPE math, the shared Embedding and DenseMLP, and the existing
embedding executor. Attention has no persistent state and is never causal.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models.embedding_gemma2_weight import (
    EMBEDDING_PROJECTION,
    PLE_GATE,
    PLE_NORM,
    PLE_OUTPUT,
    PLE_OUTPUT_NORM,
    PLE_PROJECTION,
)
from rtp_llm.models_py.model_desc.block_map import get_primary_attention_inputs
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    DenseMLP,
    Embedding,
    LinearFactory,
    MultimodalEmbeddingInjector,
)
from rtp_llm.models_py.modules.gemma4.core import (
    Gemma4RMSNorm,
    Gemma4RopeTable,
    apply_gemma4_rope,
    gemma4_rms_norm,
)
from rtp_llm.ops.compute_ops import PyModelOutputs
from rtp_llm.utils.model_weight import W


@dataclass
class EmbeddingGemma2AttentionPlan:
    lengths: tuple[int, ...]
    positions: torch.Tensor
    masks: dict[int, torch.Tensor]


class EmbeddingGemma2Attention(nn.Module):
    def __init__(self, config, weights, geometry, hw_kernel_config=None):
        super().__init__()
        self.head_num = config.attn_config.head_num
        self.head_dim = geometry["head_dim"]
        self.kv_heads = geometry["kv_heads"]
        self.window = geometry["window"]
        self.eps = config.layernorm_eps
        self.qkv_proj = LinearFactory.create_linear_from_weights(
            weights, W.attn_qkv_w, hw_kernel_config=hw_kernel_config
        )
        self.o_proj = LinearFactory.create_linear_from_weights(
            weights, W.attn_o_w, hw_kernel_config=hw_kernel_config
        )
        self.q_norm = Gemma4RMSNorm(weights[W.q_ln_gamma], self.eps)
        self.k_norm = Gemma4RMSNorm(weights[W.k_ln_gamma], self.eps)
        self.rope = Gemma4RopeTable(self.head_dim, geometry["rope_theta"], 1.0)

    def forward(self, hidden_states, plan):
        q_size = self.head_num * self.head_dim
        kv_size = self.kv_heads * self.head_dim
        q, k, v = self.qkv_proj(hidden_states).split((q_size, kv_size, kv_size), -1)
        q = self.q_norm(q.reshape(-1, self.head_num, self.head_dim))
        k = self.k_norm(k.reshape(-1, self.kv_heads, self.head_dim))
        v = gemma4_rms_norm(v.reshape(-1, self.kv_heads, self.head_dim), None, self.eps)
        cos, sin = self.rope.cos_sin(plan.positions)
        q = apply_gemma4_rope(q, cos, sin)
        k = apply_gemma4_rope(k, cos, sin)
        output = []
        start = 0
        groups = self.head_num // self.kv_heads
        for length in plan.lengths:
            end = start + length
            query = q[start:end].transpose(0, 1)
            key = k[start:end].repeat_interleave(groups, dim=1).transpose(0, 1)
            value = v[start:end].repeat_interleave(groups, dim=1).transpose(0, 1)
            # The checkpoint reference rounds QK scores and probabilities to
            # BF16. SDPA's FP32 intermediates can exceed the embedding accuracy
            # budget after 24 layers, so preserve these explicit boundaries.
            scores = query @ key.transpose(-2, -1)
            if self.window:
                scores = scores.masked_fill(
                    ~plan.masks[length], torch.finfo(query.dtype).min
                )
            probabilities = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
            output.append(
                (probabilities @ value).transpose(0, 1).reshape(length, q_size)
            )
            start = end
        return self.o_proj(torch.cat(output, dim=0))


class EmbeddingGemma2Layer(nn.Module):
    def __init__(
        self, config, parallelism_config, weights, geometry, hw_kernel_config=None
    ):
        super().__init__()
        self.self_attn = EmbeddingGemma2Attention(
            config, weights, geometry, hw_kernel_config
        )
        self.input_norm = Gemma4RMSNorm(weights[W.pre_ln_gamma], config.layernorm_eps)
        self.post_attention_norm = Gemma4RMSNorm(
            weights[W.post_ln_gamma], config.layernorm_eps
        )
        self.pre_ffn_norm = Gemma4RMSNorm(
            weights[W.pre_ffn_ln_gamma], config.layernorm_eps
        )
        self.post_ffn_norm = Gemma4RMSNorm(
            weights[W.post_ffn_ln_gamma], config.layernorm_eps
        )
        self.mlp = DenseMLP(
            config.activation_type, parallelism_config, weights, None, hw_kernel_config
        )
        self.ple_gate = LinearFactory.create_linear_from_weights(
            weights, PLE_GATE, hw_kernel_config=hw_kernel_config
        )
        self.ple_output = LinearFactory.create_linear_from_weights(
            weights, PLE_OUTPUT, hw_kernel_config=hw_kernel_config
        )
        self.ple_norm = Gemma4RMSNorm(weights[PLE_OUTPUT_NORM], config.layernorm_eps)
        self.register_buffer("layer_scalar", weights[W.layer_scalar], persistent=False)

    def forward(self, hidden_states, per_layer_input, plan):
        hidden_states = hidden_states + self.post_attention_norm(
            self.self_attn(self.input_norm(hidden_states), plan)
        )
        hidden_states = hidden_states + self.post_ffn_norm(
            self.mlp(self.pre_ffn_norm(hidden_states))
        )
        gate = (
            F.gelu(self.ple_gate(hidden_states), approximate="tanh") * per_layer_input
        )
        hidden_states = hidden_states + self.ple_norm(self.ple_output(gate))
        return hidden_states * self.layer_scalar


class EmbeddingGemma2Model(GptModelBase):
    def __init__(
        self, config, parallelism_config, weights, max_generate_batch_size, **kwargs
    ):
        super().__init__(
            config, parallelism_config, weights, max_generate_batch_size, **kwargs
        )
        if parallelism_config.world_size != 1:
            raise ValueError("EmbeddingGemma2 currently requires single-rank execution")
        embed_weight = weights.get_global_weight(W.embedding)
        if embed_weight.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(
                "EmbeddingGemma2 requires BF16 or FP32; FP16 is numerically unsafe"
            )
        self.embed_tokens = Embedding(config, parallelism_config, embed_weight)
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.register_buffer(
            "embed_scale",
            torch.tensor(
                config.input_embedding_scalar,
                dtype=embed_weight.dtype,
                device=embed_weight.device,
            ),
            persistent=False,
        )
        self.ple_dim = config.mm_related_params.config["embeddinggemma2_ple_dim"]
        self.ple_layer_num = len(
            config.mm_related_params.config["embeddinggemma2_layers"]
        )
        self.ple_projection = LinearFactory.create_linear_from_weights(
            weights.global_weights,
            PLE_PROJECTION,
            hw_kernel_config=self.py_hw_kernel_config,
        )
        self.ple_norm = Gemma4RMSNorm(
            weights.get_global_weight(PLE_NORM), config.layernorm_eps
        )
        self.output_norm = Gemma4RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), config.layernorm_eps
        )
        self.embedding_projection = LinearFactory.create_linear_from_weights(
            weights.global_weights,
            EMBEDDING_PROJECTION,
            hw_kernel_config=self.py_hw_kernel_config,
        )
        self.layers = nn.ModuleList(
            [
                EmbeddingGemma2Layer(
                    config,
                    parallelism_config,
                    weights.weights[i],
                    geometry,
                    self.py_hw_kernel_config,
                )
                for i, geometry in enumerate(
                    config.mm_related_params.config["embeddinggemma2_layers"][
                        : self.layer_num
                    ]
                )
            ]
        )

    def prepare_fmha_impl(
        self, inputs, is_cuda_graph=False, cuda_graph_selection_mode=None
    ):
        if is_cuda_graph:
            raise ValueError(
                "EmbeddingGemma2 encoder CUDA graph support is not implemented"
            )
        attn = get_primary_attention_inputs(inputs)
        if attn.sequence_lengths.numel():
            raise ValueError("EmbeddingGemma2 is an encoder and cannot decode")
        lengths = tuple(int(n) for n in attn.input_lengths.cpu().tolist())
        if not lengths or min(lengths) <= 0:
            raise ValueError("EmbeddingGemma2 requires non-empty input sequences")
        if max(lengths) > self.config.max_seq_len:
            raise ValueError(
                "EmbeddingGemma2 input exceeds the configured context length"
            )
        device = self.embed_scale.device
        positions = torch.cat(
            [torch.arange(n, device=device, dtype=torch.int32) for n in lengths]
        )
        window = self.config.attn_config.sliding_window
        # Metadata is prepared once for the complete encoder call, outside layers.
        masks = {}
        for n in set(lengths):
            pos = torch.arange(n, device=device)
            masks[n] = (pos[:, None] - pos[None, :]).abs() <= window
        return EmbeddingGemma2AttentionPlan(lengths, positions, masks)

    def forward(self, inputs, fmha_impl=None):
        plan = fmha_impl if fmha_impl is not None else self.prepare_fmha_impl(inputs)
        if inputs.input_ids.numel() != sum(plan.lengths):
            raise ValueError(
                "EmbeddingGemma2 packed token count differs from input lengths"
            )
        embedding_inputs = getattr(inputs, "embedding_inputs", None)
        text_mask = (
            embedding_inputs.text_tokens_mask if embedding_inputs is not None else None
        )
        embeddings = (
            self.embed_tokens(inputs.input_ids, text_tokens_mask=text_mask)
            * self.embed_scale
        )
        multimodal = getattr(inputs, "multimodal_inputs", None)
        if multimodal is not None and multimodal.multimodal_features:
            # HF derives PLE from the final initial embeddings, after media
            # replacement. Vision features are already in text hidden space
            # and must not receive the word-embedding sqrt(hidden_size) scale.
            embeddings = self.multimodal_embedding_injector(
                embeddings, multimodal.multimodal_features, multimodal.mm_features_locs
            )
        outputs = []
        offset = 0
        for length in plan.lengths:
            hidden_states = embeddings[offset : offset + length]
            request_plan = EmbeddingGemma2AttentionPlan(
                (length,), plan.positions[offset : offset + length], plan.masks
            )
            ple = self.ple_projection(hidden_states) * self.config.hidden_size**-0.5
            ple = self.ple_norm(ple.reshape(-1, self.ple_layer_num, self.ple_dim))
            for i, layer in enumerate(self.layers):
                hidden_states = layer(hidden_states, ple[:, i], request_plan)
            outputs.append(self.embedding_projection(self.output_norm(hidden_states)))
            offset += length
        return PyModelOutputs(torch.cat(outputs, dim=0))
