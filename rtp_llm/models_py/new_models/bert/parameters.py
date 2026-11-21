"""HF-shaped post-LayerNorm BERT encoder parameters, common to BERT/RoBERTa."""

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.read_spec import make_sharded_read_spec, unwrap_checkpoint_tensor


class HiddenShardedTable(RtpModule):
    def __init__(self, rows, width, tp, rank, dtype):
        super().__init__()
        if rows <= 0 or tp <= 0 or not 0 <= rank < tp or width % tp:
            raise ValueError("Invalid BERT embedding table partition")
        self.shape = (rows, width)
        self.tp, self.rank = tp, rank
        self.weight = nn.Parameter(
            torch.empty(rows, width // tp, dtype=dtype), requires_grad=False
        )

    def checkpoint_read_spec(self, name, shape):
        if name != "weight" or tuple(shape) != self.shape:
            raise ValueError(
                f"Wrong BERT table {name} shape {shape}; expected {self.shape}"
            )
        return make_sharded_read_spec(name, shape, 1, self.tp, self.rank)

    def load_weights(self, weights):
        for name, value in weights.items():
            tensor, spec = unwrap_checkpoint_tensor(value)
            if spec is None or not spec.is_local_shard:
                view = self.checkpoint_read_spec(name, tensor.shape)
                tensor = tensor[view.slices]
            if name != "weight" or not tensor.is_floating_point():
                raise ValueError("Invalid BERT embedding weight")
            copy_weight_(self.weight.data, tensor, name)
            self._mark_weight_loaded(name)


class BertNorm(RtpModule):
    def __init__(self, width, eps, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, dtype=dtype), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(width, dtype=dtype), requires_grad=False)
        self.variance_epsilon = eps

    @property
    def beta(self):
        return self.bias

    @property
    def input_scale(self):
        # Static BERT quantizes the post-norm activation at the consuming
        # projection. Expose that exact scale without a second Parameter.
        consumer = getattr(self, "_scale_consumer", None)
        return consumer().input_scale if consumer is not None else None

    def forward(self, value, residual=None):
        if value.is_cuda:
            from rtp_llm.models_py.modules import AddBiasResLayerNorm, LayerNorm

            if residual is None:
                return LayerNorm.forward(self, value)
            return AddBiasResLayerNorm.forward(
                self, value, residual, value.new_empty(0)
            )
        if residual is not None:
            value = value + residual
        return F.layer_norm(
            value, self.weight.shape, self.weight, self.bias, self.variance_epsilon
        )


class BertEmbeddings(RtpModule):
    def __init__(self, vocab, positions, types, h, eps, tp, rank, dtype):
        super().__init__()
        self.tp_size = tp
        self.word_embeddings = HiddenShardedTable(vocab, h, tp, rank, dtype)
        self.position_embeddings = HiddenShardedTable(positions, h, tp, rank, dtype)
        self.token_type_embeddings = HiddenShardedTable(types, h, tp, rank, dtype)
        self.LayerNorm = BertNorm(h, eps, dtype)

    @property
    def weight(self):
        return self.word_embeddings.weight

    def forward(self, tokens, embed):
        types = embed.combo_tokens_type_ids
        pos = embed.combo_position_ids
        pos_table = self.position_embeddings.weight
        type_table = self.token_type_embeddings.weight
        if tokens.is_cuda:
            from rtp_llm.models_py.modules import EmbeddingBert

            hidden = EmbeddingBert.forward(
                self,
                tokens,
                pos,
                pos_table,
                types,
                type_table,
                embed.input_embedding_scalar,
            )
        else:
            hidden = (
                F.embedding(tokens.long(), self.weight) * embed.input_embedding_scalar
            )
            if pos is not None and pos.numel():
                hidden = hidden + F.embedding(pos.long(), pos_table)
            if types is not None and types.numel():
                hidden = hidden + F.embedding(types.long(), type_table)
            if self.tp_size > 1:
                from rtp_llm.models_py.distributed.collective_torch import (
                    Group,
                    all_gather,
                )

                hidden = (
                    all_gather(hidden, group=Group.TP)
                    .reshape(self.tp_size, tokens.numel(), -1)
                    .transpose(0, 1)
                    .reshape(tokens.numel(), -1)
                )
        return self.LayerNorm(hidden)


class BertLayer(RtpModule):
    def __init__(self, config, load, index, *, input_major=False):
        super().__init__()
        h, ffn = config.hidden_size, config.inter_size
        heads, dim = config.attn_config.head_num, config.attn_config.size_per_head
        if heads * dim != h:
            raise ValueError("BERT attention heads must span hidden_size")
        self.index = index
        quant = load.quant_config
        if getattr(quant, "quant_type", None) == "pertensor_quant":
            from .static_int8 import BertStaticInt8Config

            quant = BertStaticInt8Config("pertensor_quant", input_major=input_major)
        common = dict(quant_config=quant, params_dtype=load.compute_dtype, bias=True)
        attn = dict(tp_size=load.attn_tp_size, tp_rank=load.attn_tp_rank, **common)
        mlp = dict(tp_size=load.ffn_tp_size, tp_rank=load.ffn_tp_rank, **common)
        p = f"encoder.layer.{index}"
        self.qkv = QKVParallelLinear(
            hidden_size=h,
            num_heads=heads,
            num_kv_heads=heads,
            head_dim=dim,
            prefix=p + ".attention.self",
            **attn,
        )
        self.attn_output = RowParallelLinear(
            h, h, prefix=p + ".attention.output.dense", **attn
        )
        self.attn_norm = BertNorm(h, config.layernorm_eps, load.compute_dtype)
        self.intermediate = ColumnParallelLinear(
            h, ffn, prefix=p + ".intermediate.dense", **mlp
        )
        self.output = RowParallelLinear(ffn, h, prefix=p + ".output.dense", **mlp)
        self.output_norm = BertNorm(h, config.layernorm_eps, load.compute_dtype)

    def forward(self, hidden, fmha, cache):
        packed = self.qkv(hidden)
        context = (
            fmha.forward(packed, cache, self.index)
            .reshape(hidden.shape[0], -1)
            .contiguous()
        )
        attention = self.attn_norm(self.attn_output(context), hidden)
        return self.output_norm(
            self.output(self.intermediate.forward_with_bias_gelu(attention)), attention
        )


class PackedBertAttention:
    """CPU bidirectional attention; batch boundaries are never attended across."""

    def __init__(self, lengths, heads, dim):
        self.lengths = tuple(int(n) for n in lengths)
        self.heads, self.dim = heads, dim

    def forward(self, packed, cache, index):
        if sum(self.lengths) != packed.shape[0] or any(n <= 0 for n in self.lengths):
            raise ValueError("Invalid packed BERT sequence lengths")
        out = []
        for sample in packed.split(self.lengths, dim=0):
            q, k, v = (
                x.transpose(0, 1)
                for x in sample.reshape(-1, 3, self.heads, self.dim).unbind(1)
            )
            out.append(F.scaled_dot_product_attention(q, k, v).transpose(0, 1))
        return torch.cat(out, dim=0)
