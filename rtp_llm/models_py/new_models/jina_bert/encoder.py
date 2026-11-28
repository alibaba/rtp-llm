"""Direct checkpoint owners for jina-bert-v2-qk-post-norm.

Jina is an encoder, not a causal GPT block: Q/K normalization spans the
hidden dimension before the head split and the position bias is symmetric.
"""

from functools import partial
from itertools import accumulate

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.downstream_heads import (
    check_encoder_handler_alias,
    unpack_handler_mapping,
)
from rtp_llm.models_py.module_base import RtpModule, _mark_loaded
from rtp_llm.models_py.read_spec import (
    ReadSpec,
    make_sharded_read_spec,
    unwrap_checkpoint_tensor,
)


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


def _sdpa(q, k, v, slopes, sequence_ids=None):
    # Portable fallback. Bound both the bias and the score tile even if SDPA
    # selects its math implementation; never raise this budget for long inputs.
    batch, heads, seq, _ = q.shape
    chunk = max(1, (1 << 20) // (batch * heads * seq))
    pos = torch.arange(seq, device=q.device)
    slopes = slopes.to(q.dtype)[:, None, None]
    parts = []
    for start in range(0, seq, chunk):
        section = slice(start, start + chunk)
        distance = (pos[section, None] - pos).abs()
        bias = -slopes * distance
        if sequence_ids is not None:
            bias.masked_fill_(sequence_ids[section, None] != sequence_ids, float("-inf"))
        parts.append(
            F.scaled_dot_product_attention(q[:, :, section], k, v, attn_mask=bias)
        )
    return torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]


def _packed_sdpa(q, k, v, slopes, *, ranges):
    # Keep projections and TP collectives batched; only attention needs splits.
    parts = [
        _sdpa(q[:, :, lo:hi], k[:, :, lo:hi], v[:, :, lo:hi], slopes)
        for lo, hi in ranges
    ]
    return torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]


def _packed_flash(q, k, v, slopes, *, ends, max_len):
    from flash_attn import flash_attn_varlen_func

    # FlashAttention applies -slope * abs(i-j) within each sequence, without
    # materializing any score/bias matrix. Q/K/V are [total, local_heads, dim].
    out = flash_attn_varlen_func(
        *(x.squeeze(0).transpose(0, 1) for x in (q, k, v)),
        ends,
        ends,
        max_len,
        max_len,
        causal=False,
        alibi_slopes=slopes,
    )
    return out.transpose(0, 1).unsqueeze(0)


def _packed_attention(ids, ends, host_ends, is_graph, dtype, head_width):
    total = ids.numel()
    if not total:
        raise ValueError("expected non-empty Jina tokens")
    use_flash = False
    if ids.is_cuda and dtype in (torch.float16, torch.bfloat16) and head_width <= 256:
        from rtp_llm.utils.flash_attn_utils import can_use_flash_attn

        use_flash = can_use_flash_attn(ids.device)
    # On replay only the device boundaries are live. Never bake lengths from
    # the capture fixture into a Python loop or a kernel's launch bound.
    ranges = None
    if not is_graph and (host_ends is not None or not use_flash):
        offsets = host_ends.tolist() if host_ends is not None else ends.tolist()
        offsets.append(total)  # any trailing graph padding is an isolated sequence
        if offsets[0] != 0 or any(b < a for a, b in zip(offsets, offsets[1:])):
            raise ValueError("invalid Jina cumulative query lengths")
        ranges = [(a, b) for a, b in zip(offsets, offsets[1:]) if a < b]

    if use_flash:
        ends = ends.to(device=ids.device, dtype=torch.int32, non_blocking=True)
        ends = torch.cat((ends, ends.new_full((1,), total)))
        max_len = total if ranges is None else max(b - a for a, b in ranges)
        return partial(_packed_flash, ends=ends, max_len=max_len)

    if ranges is not None:
        return partial(_packed_sdpa, ranges=ranges)
    ends = ends.to(device=ids.device, dtype=torch.int32, non_blocking=True)
    sequence_ids = torch.bucketize(
        torch.arange(total, device=ids.device, dtype=ends.dtype), ends[1:], right=True
    )
    return partial(_sdpa, sequence_ids=sequence_ids)


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

    def forward(self, hidden, pg, attention, slopes):
        batch, seq, width = hidden.shape
        attn = self.attention.self

        def split(x):
            return x.view(batch, seq, self.heads, self.head_width).transpose(1, 2)

        q = split(self._normalize(attn.query(hidden), attn.layer_norm_q, pg))
        k = split(self._normalize(attn.key(hidden), attn.layer_norm_k, pg))
        v = split(attn.value(hidden))
        output = attention(q, k, v, slopes)
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
        cp = getattr(load.parallelism_config, "prefill_cp_config", None)
        if load.tp_size > 1 and cp is not None and cp.is_enabled():
            # Jina's bidirectional SDPA sees only this worker's tokens. It
            # has no cross-rank K/V exchange for a split prefill sequence.
            raise ValueError(
                "Jina bidirectional attention requires a complete sequence; "
                "context parallel prefill is unsupported"
            )
        for role, source, *_ in load.custom_weight_mappings:
            check_encoder_handler_alias(
                role, source, self.WEIGHTS_MAPPER.map_name(source)
            )
        self._custom = {
            source: (str(i), target, requested_dtype)
            for i, (target, source, requested_dtype) in enumerate(
                map(unpack_handler_mapping, load.custom_weight_mappings)
            )
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
                    key, _, dtype = custom
                    tensor, _ = unwrap_checkpoint_tensor(tensor)
                    if key in self.custom_params or not tensor.is_floating_point():
                        raise ValueError(
                            f"invalid or duplicate Jina downstream tensor {source}"
                        )
                    owning = self.embeddings.word_embeddings.weight
                    self.custom_params[key] = nn.Parameter(
                        tensor.detach().to(
                            device=owning.device, dtype=dtype or owning.dtype, copy=True
                        ),
                        requires_grad=False,
                    )
                    _mark_loaded(self.custom_params, key)
                else:
                    yield target, tensor

        super().load_weights(mapped())
        missing = [
            source
            for source, (key, _, _) in self._custom.items()
            if key not in self.custom_params
        ]
        if missing:
            raise RuntimeError(f"missing Jina downstream tensors: {missing}")

    def custom_weight_view(self):
        return {
            target: self.custom_params[key] for key, target, _ in self._custom.values()
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

    def encode(
        self, ids, token_types=None, scale=1.0, pg=None, sequence_ids=None, *, attention=None
    ):
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
        if self.encoder.layer:
            slopes = torch.tensor(
                self.encoder.layer[0].slopes, device=h.device, dtype=torch.float32
            )
            if attention is None:
                attention = partial(_sdpa, sequence_ids=sequence_ids)
            for layer in self.encoder.layer:
                h = layer(h, pg, attention, slopes)
        return h

    def encode_cumulative(
        self, ids, token_types, ends, scale=1.0, *, host_ends=None, is_graph=False
    ):
        attention = None
        if self.encoder.layer:
            attention = _packed_attention(
                ids,
                ends,
                host_ends,
                is_graph,
                self.embeddings.word_embeddings.weight.dtype,
                self.encoder.layer[0].head_width,
            )
        return self.encode(
            ids.unsqueeze(0),
            None if token_types is None else token_types.unsqueeze(0),
            scale,
            attention=attention,
        ).squeeze(0)

    def encode_packed(self, ids, token_types, lengths, scale=1.0):
        if ids.ndim != 1 or any(n <= 0 for n in lengths) or sum(lengths) != ids.numel():
            raise ValueError("inconsistent Jina packed input lengths")
        if token_types is not None and token_types.shape != ids.shape:
            raise ValueError("inconsistent Jina packed token type ids")
        ends = torch.tensor(
            list(accumulate(lengths, initial=0)), dtype=torch.int32, device="cpu"
        )
        return self.encode_cumulative(ids, token_types, ends, scale, host_ends=ends)
