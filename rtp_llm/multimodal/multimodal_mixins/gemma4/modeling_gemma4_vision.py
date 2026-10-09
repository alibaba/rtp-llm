"""Gemma4 vision tower (ViT) for RTP-LLM multimodal serving.

Pure-torch port of the HF transformers 5.5.0 Gemma4VisionModel math
(modeling_gemma4.py). The module tree is named so that state_dict keys match
the checkpoint under the single prefix "model.":

  model.vision_tower.patch_embedder.input_proj.weight        [1152, 768]
  model.vision_tower.patch_embedder.position_embedding_table [2, 10240, 1152]
  model.vision_tower.encoder.layers.N.{input,post_attention,pre_feedforward,post_feedforward}_layernorm.weight
  model.vision_tower.encoder.layers.N.self_attn.{q,k,v,o}_proj.linear.weight
  model.vision_tower.encoder.layers.N.self_attn.{q,k}_norm.weight [72]
  model.vision_tower.std_bias / std_scale [1152]
  model.embed_vision.embedding_projection.weight [2816, 1152]

Pipeline: scale 2*(x-0.5) -> input_proj -> + 2D position embeddings
-> 27 x (norm -> attn(q/k norm, 2D rope, scale=1.0, bidirectional+padding mask)
-> norm -> +res -> norm -> MLP(gelu_tanh) -> norm -> +res)
-> 3x3 avg pool -> * sqrt(1152) -> strip padding -> (x-std_bias)*std_scale
-> scale-less RMSNorm -> embedding_projection (1152 -> 2816).
"""

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class Gemma4VisionRMSNorm(nn.Module):
    """HF Gemma4RMSNorm: fp32 rms norm, optional fp32 scale, cast back."""

    def __init__(self, weight: Optional[torch.Tensor], eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        if weight is not None:
            self.weight = nn.Parameter(weight.float())
        else:
            self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = x.float().pow(2).mean(-1, keepdim=True) + self.eps
        out = x.float() * torch.pow(ms, -0.5)
        if self.weight is not None:
            out = out * self.weight
        return out.type_as(x)


def _linear(weight: torch.Tensor) -> nn.Linear:
    out_features, in_features = weight.shape
    lin = nn.Linear(in_features, out_features, bias=False)
    lin.weight = nn.Parameter(weight)
    return lin


class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, weights: Dict[str, torch.Tensor], hidden_size: int):
        super().__init__()
        self.input_proj = _linear(weights["patch_embedder.input_proj.weight"])
        self.position_embedding_table = nn.Parameter(
            weights["patch_embedder.position_embedding_table"]
        )
        self.position_embedding_size = self.position_embedding_table.shape[1]

    def _position_embeddings(
        self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor
    ) -> torch.Tensor:
        clamped = pixel_position_ids.clamp(min=0)
        # Each spatial one-hot row selects exactly one table entry. Gather
        # those entries directly to avoid a video-sized one-hot allocation.
        pos = (
            self.position_embedding_table[0][clamped[..., 0]]
            + self.position_embedding_table[1][clamped[..., 1]]
        )
        return torch.where(padding_positions.unsqueeze(-1), 0.0, pos)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = 2 * (pixel_values - 0.5)
        hidden = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        return hidden + self._position_embeddings(pixel_position_ids, padding_positions)


class Gemma4VisionRotaryEmbedding(nn.Module):
    """2D spatial rope: independent inv_freq per spatial dim (theta=100)."""

    def __init__(self, head_dim: int, theta: float, device: torch.device):
        super().__init__()
        spatial_dim = head_dim // 2
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, spatial_dim, 2, dtype=torch.float32, device=device)
                / spatial_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [B, P, 2] -> cos/sin [B, P, head_dim]
        inv = self.inv_freq[None, :, None].float()
        all_cos, all_sin = [], []
        for i in range(2):
            pos = position_ids[:, :, i][:, None, :].float()
            freqs = (inv @ pos).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())
        return torch.cat(all_cos, dim=-1), torch.cat(all_sin, dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_multidim_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # x: [B, P, H, D] (D=head_dim); cos/sin: [B, P, D]; split D into 2 spatial
    # halves and rotate each with its own cos/sin slice (HF
    # apply_multidimensional_rope with ndim=2).
    ndim = 2
    d = x.shape[-1]
    per = 2 * (d // (2 * ndim))
    xs = x.split(per, dim=-1)
    cs = cos.split(per, dim=-1)
    ss = sin.split(per, dim=-1)
    out = []
    for xi, ci, si in zip(xs, cs, ss):
        ci = ci.unsqueeze(2)
        si = si.unsqueeze(2)
        out.append(xi * ci + _rotate_half(xi) * si)
    return torch.cat(out, dim=-1)


class Gemma4VisionAttention(nn.Module):
    def __init__(self, weights: Dict[str, torch.Tensor], prefix: str, eps: float):
        super().__init__()
        p = f"{prefix}self_attn."
        self.q_proj = _linear(weights[p + "q_proj.linear.weight"])
        self.k_proj = _linear(weights[p + "k_proj.linear.weight"])
        self.v_proj = _linear(weights[p + "v_proj.linear.weight"])
        self.o_proj = _linear(weights[p + "o_proj.linear.weight"])
        self.q_norm = Gemma4VisionRMSNorm(weights[p + "q_norm.weight"], eps)
        self.k_norm = Gemma4VisionRMSNorm(weights[p + "k_norm.weight"], eps)
        self.v_norm = Gemma4VisionRMSNorm(None, eps)
        self.scaling = 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        head_dim = self.q_norm.weight.shape[0]
        hidden_shape = (*input_shape, -1, head_dim)

        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        q = _apply_multidim_rope(q, cos, sin).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        k = _apply_multidim_rope(k, cos, sin).transpose(1, 2)
        v = self.v_norm(self.v_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

        attention_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask
        attention_weights = F.softmax(
            attention_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        attn_output = torch.matmul(attention_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(self, weights: Dict[str, torch.Tensor], layer_idx: int, eps: float):
        super().__init__()
        p = f"encoder.layers.{layer_idx}."
        self.self_attn = Gemma4VisionAttention(weights, p, eps)
        self.input_layernorm = Gemma4VisionRMSNorm(
            weights[p + "input_layernorm.weight"], eps
        )
        self.post_attention_layernorm = Gemma4VisionRMSNorm(
            weights[p + "post_attention_layernorm.weight"], eps
        )
        self.pre_feedforward_layernorm = Gemma4VisionRMSNorm(
            weights[p + "pre_feedforward_layernorm.weight"], eps
        )
        self.post_feedforward_layernorm = Gemma4VisionRMSNorm(
            weights[p + "post_feedforward_layernorm.weight"], eps
        )
        self.gate_proj = _linear(weights[p + "mlp.gate_proj.linear.weight"])
        self.up_proj = _linear(weights[p + "mlp.up_proj.linear.weight"])
        self.down_proj = _linear(weights[p + "mlp.down_proj.linear.weight"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.down_proj(
            F.gelu(self.gate_proj(hidden_states), approximate="tanh")
            * self.up_proj(hidden_states)
        )
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class Gemma4VisionEncoder(nn.Module):
    def __init__(self, weights: Dict[str, torch.Tensor], num_layers: int, eps: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [Gemma4VisionEncoderLayer(weights, i, eps) for i in range(num_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attention_mask)
        return hidden_states


class Gemma4VisionPooler(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.root_hidden_size = hidden_size**0.5

    def _avg_pool_by_positions(
        self, hidden_states: torch.Tensor, pixel_position_ids: torch.Tensor, length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {tuple(hidden_states.shape)} to {length}: "
                f"{k}^2 * {length} != {input_seq_len}"
            )
        clamped = pixel_position_ids.clamp(min=0)
        max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, padding_positions


class Gemma4VisionTower(nn.Module):
    """patch_embedder -> encoder -> pooler -> standardize (HF Gemma4VisionModel)."""

    def __init__(
        self,
        config: dict,
        weights: Dict[str, torch.Tensor],
        device: torch.device,
    ):
        super().__init__()
        self.hidden_size = int(config["hidden_size"])  # 1152
        self.head_dim = int(config.get("head_dim", 72))
        self.num_layers = int(config["num_hidden_layers"])  # 27
        self.pooling_kernel_size = int(config.get("pooling_kernel_size", 3))
        rope_theta = float(config.get("rope_theta", 100.0))
        eps = float(config.get("rms_norm_eps", 1e-6))
        self.standardize = bool(config.get("standardize", True))

        self.patch_embedder = Gemma4VisionPatchEmbedder(weights, self.hidden_size)
        self.rotary_emb = Gemma4VisionRotaryEmbedding(self.head_dim, rope_theta, device)
        self.encoder = Gemma4VisionEncoder(weights, self.num_layers, eps)
        self.pooler = Gemma4VisionPooler(self.hidden_size)
        if self.standardize:
            # Checkpoint-native dtype (bf16): (x-std_bias)*std_scale runs in the
            # model dtype like HF; fp32 buffers would promote the stream.
            self.register_buffer("std_bias", weights["std_bias"])
            self.register_buffer("std_scale", weights["std_scale"])

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        output_length = pixel_values.shape[-2] // (
            self.pooling_kernel_size * self.pooling_kernel_size
        )
        padding_positions = (pixel_position_ids == -1).all(dim=-1)

        hidden_states = self.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )
        cos, sin = self.rotary_emb(pixel_position_ids)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)
        # Bidirectional mask: additive -inf at padding key positions.
        attention_mask = torch.zeros(
            hidden_states.shape[0],
            1,
            1,
            hidden_states.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        attention_mask = attention_mask.masked_fill(
            padding_positions[:, None, None, :], float("-inf")
        )
        hidden_states = self.encoder(hidden_states, cos, sin, attention_mask)

        hidden_states, pooler_mask = self.pooler(
            hidden_states, pixel_position_ids, padding_positions, output_length
        )
        hidden_states = hidden_states[pooler_mask]
        if self.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale
        return hidden_states


class Gemma4MultimodalEmbedder(nn.Module):
    """Scale-less RMSNorm -> Linear(vision_hidden -> text_hidden)."""

    def __init__(self, weights: Dict[str, torch.Tensor], eps: float):
        super().__init__()
        self.embedding_pre_projection_norm = Gemma4VisionRMSNorm(None, eps)
        self.embedding_projection = _linear(
            weights["embed_vision.embedding_projection.weight"]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(
            self.embedding_pre_projection_norm(inputs_embeds)
        )


class Gemma4VisionModel(nn.Module):
    """vision_tower + embed_vision; state_dict keys match ckpt prefix "model.".

    forward(pixel_values, pixel_position_ids) -> soft-token embeddings in the
    language model's hidden space, [num_valid_soft_tokens, text_hidden_size].

    weights=None constructs empty parameters (filled by the mixin's load flow
    via state_dict key mapping); pass a real weight dict for direct use/tests.
    """

    def __init__(
        self,
        config: dict,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if weights is None:
            weights = _empty_weights(config, device)
        eps = float(config.get("rms_norm_eps", 1e-6))
        self.vision_tower = Gemma4VisionTower(config, weights, device)
        self.embed_vision = Gemma4MultimodalEmbedder(weights, eps)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embed_vision(self.vision_tower(pixel_values, pixel_position_ids))


def _empty_weights(config: dict, device: torch.device) -> Dict[str, torch.Tensor]:
    hidden = int(config["hidden_size"])
    inter = int(config.get("intermediate_size", 4304))
    head_dim = int(config.get("head_dim", 72))
    layers = int(config["num_hidden_layers"])
    patch = int(config.get("patch_size", 16))
    pos_size = int(config.get("position_embedding_size", 10240))
    text_hidden = int(config["text_hidden_size"])

    def e(*shape):
        return torch.empty(*shape, device=device)

    weights: Dict[str, torch.Tensor] = {
        "patch_embedder.input_proj.weight": e(hidden, 3 * patch * patch),
        "patch_embedder.position_embedding_table": e(2, pos_size, hidden),
        "std_bias": e(hidden),
        "std_scale": e(hidden),
        "embed_vision.embedding_projection.weight": e(text_hidden, hidden),
    }
    for i in range(layers):
        p = f"encoder.layers.{i}."
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            weights[p + norm + ".weight"] = e(hidden)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weights[p + f"self_attn.{proj}.linear.weight"] = e(hidden, hidden)
        weights[p + "self_attn.q_norm.weight"] = e(head_dim)
        weights[p + "self_attn.k_norm.weight"] = e(head_dim)
        weights[p + "mlp.gate_proj.linear.weight"] = e(inter, hidden)
        weights[p + "mlp.up_proj.linear.weight"] = e(inter, hidden)
        weights[p + "mlp.down_proj.linear.weight"] = e(hidden, inter)
    return weights


VISION_CONFIG_DEFAULTS = {
    "hidden_size": 1152,
    "text_hidden_size": 2816,
    "head_dim": 72,
    "num_hidden_layers": 27,
    "intermediate_size": 4304,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "position_embedding_size": 10240,
    "rope_theta": 100.0,
    "rms_norm_eps": 1e-6,
    "standardize": True,
}
