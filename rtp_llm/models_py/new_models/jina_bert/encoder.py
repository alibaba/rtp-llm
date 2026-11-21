"""Direct checkpoint owners for jina-bert-v2-qk-post-norm.

Jina is an encoder, not a causal GPT block: Q/K normalization spans the
hidden dimension before the head split and the position bias is symmetric.
"""

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.module_base import RtpModule, _mark_loaded
from rtp_llm.models_py.read_spec import ReadSpec, make_sharded_read_spec


class _Names:
    @staticmethod
    def map_name(key):
        key = key.removeprefix("bert.")
        prefix, _, suffix = key.rpartition(".")
        if prefix.endswith(
            ("embeddings.LayerNorm", "attention.output.LayerNorm", "layer_norm_2")
        ):
            suffix = {"gamma": "weight", "beta": "bias"}.get(suffix, suffix)
            return prefix + "." + suffix
        return key


def _alibi_slopes(heads):
    # For non-powers of two, append the alternating extra slopes from 2*p.
    p = 1 << (heads.bit_length() - 1)
    base = 2 ** (-8.0 / p)
    slopes = [base ** (i + 1) for i in range(p)]
    extra = 2 ** (-4.0 / p)
    return slopes + [extra ** (2 * i + 1) for i in range(heads - p)]


def _sum_tp(value, size, pg):
    if size == 1:
        return value
    if pg is None:
        from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

        return all_reduce(value, Group.TP)
    if torch.distributed.get_world_size(pg) != size:
        raise RuntimeError("Jina TP group size differs from the checkpoint partition")
    torch.distributed.all_reduce(value, group=pg)
    return value


class JinaEncoderLayer(RtpModule):
    def __init__(self, cfg, dtype, tp_size=1, tp_rank=0, ffn_size=1, ffn_rank=0):
        super().__init__()
        hidden = cfg.hidden_size
        heads = cfg.attn_config.head_num
        self.tp_size, self.tp_rank = tp_size, tp_rank
        self.ffn_size, self.ffn_rank = ffn_size, ffn_rank
        if heads < 1 or hidden % heads or heads % tp_size or cfg.inter_size % ffn_size:
            raise ValueError(
                "Jina attention heads and intermediate width must divide evenly across TP"
            )
        self.heads = heads // tp_size
        self.head_width = hidden // heads
        self.full_width = hidden
        self.inter = cfg.inter_size
        self.slopes = tuple(
            _alibi_slopes(heads)[tp_rank * self.heads : (tp_rank + 1) * self.heads]
        )
        norm = lambda: nn.LayerNorm(hidden, eps=cfg.layernorm_eps, dtype=dtype)
        proj = lambda: nn.Linear(hidden, hidden // tp_size, dtype=dtype)
        self.attention = nn.Module()
        attn = nn.Module()
        self.attention.add_module("self", attn)
        attn.query, attn.key, attn.value = proj(), proj(), proj()
        local_norm = lambda: nn.LayerNorm(
            hidden // tp_size, eps=cfg.layernorm_eps, dtype=dtype
        )
        attn.layer_norm_q, attn.layer_norm_k = local_norm(), local_norm()
        self.attention.output = nn.Module()
        self.attention.output.dense = nn.Linear(hidden // tp_size, hidden, dtype=dtype)
        self.attention.output.LayerNorm = norm()
        self.layer_norm_1, self.layer_norm_2 = norm(), norm()
        self.mlp = nn.Module()
        self.mlp.up_gated_layer = nn.Linear(
            hidden, 2 * cfg.inter_size // ffn_size, bias=False, dtype=dtype
        )
        self.mlp.down_layer = nn.Linear(cfg.inter_size // ffn_size, hidden, dtype=dtype)

    def checkpoint_read_spec(self, name, shape):
        size, rank = (
            (self.ffn_size, self.ffn_rank)
            if name.startswith("mlp.")
            else (self.tp_size, self.tp_rank)
        )
        if size == 1:
            return None
        if name == "mlp.up_gated_layer.weight":
            if shape != (2 * self.inter, self.full_width):
                raise ValueError("invalid fused Jina GLU weight shape")
            width = self.inter // size
            begin = width * rank
            rows = tuple(range(begin, begin + width)) + tuple(
                range(self.inter + begin, self.inter + begin + width)
            )
            return ReadSpec(name, name, shape, row_indices=rows)
        if name.startswith("attention.self."):
            return make_sharded_read_spec(name, shape, 0, size, rank)
        if name in ("attention.output.dense.weight", "mlp.down_layer.weight"):
            return make_sharded_read_spec(name, shape, 1, size, rank)
        return None

    def _normalize(self, value, layer, pg):
        if self.tp_size == 1:
            return layer(value)
        fp = value.float()
        mean = _sum_tp(fp.sum(-1, keepdim=True), self.tp_size, pg) / self.full_width
        centered = fp - mean
        var = (
            _sum_tp(centered.square().sum(-1, keepdim=True), self.tp_size, pg)
            / self.full_width
        )
        return (
            centered * torch.rsqrt(var + layer.eps) * layer.weight.float()
            + layer.bias.float()
        ).to(value.dtype)

    def _row(self, x, proj, pg, size):
        return _sum_tp(F.linear(x, proj.weight), size, pg) + proj.bias

    def forward(self, hidden, pg=None, sequence_ids=None):
        batch, seq, width = hidden.shape
        attn = self.attention.self

        def split(x):
            return x.view(batch, seq, self.heads, self.head_width).transpose(1, 2)

        q = split(self._normalize(attn.query(hidden), attn.layer_norm_q, pg))
        k = split(self._normalize(attn.key(hidden), attn.layer_norm_k, pg))
        v = split(attn.value(hidden))
        # Cap the temporary bias allocation even for long code inputs.
        pos = torch.arange(seq, device=hidden.device)
        slopes = torch.tensor(self.slopes, device=hidden.device, dtype=hidden.dtype)[
            :, None, None
        ]
        chunk = max(1, min(seq, (1 << 20) // max(1, seq * self.heads)))
        parts = []
        for start in range(0, seq, chunk):
            distance = (pos[start : start + chunk, None] - pos[None, :]).abs()
            bias = -slopes * distance
            if sequence_ids is not None:
                same_sequence = (
                    sequence_ids[start : start + chunk, None] == sequence_ids[None, :]
                )
                bias = bias.masked_fill(~same_sequence, float("-inf"))
            parts.append(
                F.scaled_dot_product_attention(
                    q[:, :, start : start + chunk], k, v, attn_mask=bias
                )
            )
        output = torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]
        output = self._row(
            output.transpose(1, 2).reshape(batch, seq, width // self.tp_size),
            self.attention.output.dense,
            pg,
            self.tp_size,
        )
        output = self.attention.output.LayerNorm(hidden + output)
        residual = self.layer_norm_1(hidden + output)
        up, gate = self.mlp.up_gated_layer(residual).chunk(2, dim=-1)
        return self.layer_norm_2(
            residual
            + self._row(up * F.gelu(gate), self.mlp.down_layer, pg, self.ffn_size)
        )


class JinaEncoder(RtpModule):
    """Encoder returns token states; the embedding service owns pooling."""

    WEIGHTS_MAPPER = _Names()
    supports_custom_weight_mappings = True

    def __init__(self, cfg, load):
        super().__init__()
        self.init_encoder(cfg, load)

    def init_encoder(self, cfg, load):
        dtype = load.compute_dtype
        if getattr(load.quant_config, "quant_type", "none") not in (None, "", "none"):
            raise NotImplementedError(
                "Jina direct encoder expects floating-point checkpoint projections"
            )
        if getattr(load.parallelism_config, "pp_size", 1) != 1:
            raise NotImplementedError(
                "Jina direct encoder has no PP stage ownership contract"
            )
        self._custom = {
            source: (str(i), target)
            for i, (target, source) in enumerate(load.custom_weight_mappings)
        }
        self.custom_params = nn.ParameterDict()
        self.embeddings = nn.Module()
        self.embeddings.word_embeddings = nn.Embedding(
            cfg.vocab_size, cfg.hidden_size, dtype=dtype
        )
        self.embeddings.token_type_embeddings = nn.Embedding(
            cfg.type_vocab_size, cfg.hidden_size, dtype=dtype
        )
        self.embeddings.LayerNorm = nn.LayerNorm(
            cfg.hidden_size, eps=cfg.layernorm_eps, dtype=dtype
        )
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            JinaEncoderLayer(
                cfg,
                dtype,
                load.attn_tp_size,
                load.attn_tp_rank,
                load.ffn_tp_size,
                load.ffn_tp_rank,
            )
            for _ in range(cfg.num_layers)
        )
        self.requires_grad_(False)

    def load_weights(self, weights):
        entries = weights.items() if isinstance(weights, dict) else weights

        def mapped():
            for source, tensor in entries:
                target = self.WEIGHTS_MAPPER.map_name(source)
                custom = self._custom.get(source, self._custom.get(target))
                if custom is not None:
                    key, _ = custom
                    if key in self.custom_params or not tensor.is_floating_point():
                        raise ValueError(
                            f"invalid or duplicate Jina downstream tensor {source}"
                        )
                    owning = self.embeddings.word_embeddings.weight
                    self.custom_params[key] = nn.Parameter(
                        tensor.detach().to(
                            device=owning.device, dtype=owning.dtype, copy=True
                        ),
                        requires_grad=False,
                    )
                    _mark_loaded(self.custom_params, key)
                else:
                    yield target, tensor

        super().load_weights(mapped())
        missing = [
            source
            for source, (key, _) in self._custom.items()
            if key not in self.custom_params
        ]
        if missing:
            raise RuntimeError(f"missing Jina downstream tensors: {missing}")

    def custom_weight_view(self):
        return {
            target: self.custom_params[key] for key, target in self._custom.values()
        }

    def checkpoint_weight_name_filter(self):
        # Exclude inactive HF task heads and nonpersistent index buffers only.
        def should_load(key):
            if key in self._custom or self.WEIGHTS_MAPPER.map_name(key) in self._custom:
                return True
            key = key.removeprefix("bert.")
            return not (
                key.startswith(("pooler.", "cls.", "classifier."))
                or key in ("embeddings.position_ids", "embeddings.token_type_ids")
            )

        return should_load

    def encode(self, ids, token_types=None, scale=1.0, pg=None, sequence_ids=None):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("expected non-empty rectangular Jina token sequences")
        if token_types is None:
            token_types = torch.zeros_like(ids)
        if token_types.shape != ids.shape:
            raise ValueError("token type ids must have the same shape as input ids")
        h = self.embeddings.LayerNorm(
            self.embeddings.word_embeddings(ids) * scale
            + self.embeddings.token_type_embeddings(token_types)
        )
        for layer in self.encoder.layer:
            h = layer(h, pg, sequence_ids)
        return h

    def encode_packed(self, ids, token_types, lengths, scale=1.0):
        if ids.ndim != 1 or any(n <= 0 for n in lengths) or sum(lengths) != ids.numel():
            raise ValueError("inconsistent Jina packed input lengths")
        if token_types is not None and token_types.shape != ids.shape:
            raise ValueError("inconsistent Jina packed token type ids")
        parts = []
        type_chunks = None if token_types is None else token_types.split(lengths)
        for i, piece in enumerate(ids.split(lengths)):
            types = None if type_chunks is None else type_chunks[i].unsqueeze(0)
            parts.append(self.encode(piece.unsqueeze(0), types, scale).squeeze(0))
        return torch.cat(parts, dim=0)
