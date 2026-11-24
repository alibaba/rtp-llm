"""Checkpoint parameter tree for Flash on M890P. The provider builds the execution tree."""

import torch
from torch import nn

from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.modules.dsv4.fp8.linear_owner import Dsv4FP8Parameters
from rtp_llm.models_py.new_models.deepseek_v4.ppu_linear import PpuOwnedFP8Linear
from rtp_llm.models_py.read_spec import make_sharded_read_spec, unwrap_checkpoint_tensor


def _parameter(shape, dtype):
    return nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


class _ScalarWeight(RtpModule):
    def __init__(self, shape, dtype):
        super().__init__()
        self.weight = _parameter(shape, dtype)


class _BF16Projection(nn.Linear):
    # NewLoader fills every element before runtime publication.
    def reset_parameters(self):
        pass


class _CompressorParameters(RtpModule):
    def __init__(self, dim, head_dim, ratio, *, indexer=False):
        super().__init__()
        width = head_dim * (2 if ratio == 4 else 1)
        self.ape = _parameter((ratio, width), torch.float32)
        self.norm = _ScalarWeight(
            (head_dim,), torch.float32 if indexer else torch.bfloat16
        )
        self.wkv = _BF16Projection(
            dim, width, bias=False, dtype=torch.bfloat16
        ).requires_grad_(False)
        self.wgate = _BF16Projection(
            dim, width, bias=False, dtype=torch.bfloat16
        ).requires_grad_(False)


class _IndexerParameters(RtpModule):
    def __init__(self, args, provider):
        super().__init__()
        self.wq_b = PpuOwnedFP8Linear(
            args.q_lora_rank,
            args.index_n_heads * args.index_head_dim,
            platform_provider=provider,
        )
        # M890P FP4 score kernels consume the BF16, NOT score-folded, projection.
        self.weights_proj = _ScalarWeight(
            (args.index_n_heads, args.dim), torch.bfloat16
        )
        self.compressor = _CompressorParameters(
            args.dim, args.index_head_dim, 4, indexer=True
        )


class PpuAttentionParameters(RtpModule):
    def __init__(self, args, layer_id, provider):
        super().__init__()
        if provider is None:
            raise ValueError("M890P attention needs a role-specific provider")
        ratio = args.compress_ratios[layer_id]
        if ratio not in (0, 4, 128):
            raise ValueError(f"Unsupported Flash attention compression ratio {ratio}")
        if getattr(args, "commit_only", False):
            raise ValueError(
                "The M890P Flash target/MTP path does not host a DSpARK commit-only model"
            )
        tp, rank = args.tp_size, args.tp_rank
        if tp < 1 or not 0 <= rank < tp or args.n_heads % tp or args.o_groups % tp:
            raise ValueError("Invalid M890P attention head/group TP geometry")
        if (args.n_heads * args.head_dim) % args.o_groups:
            raise ValueError("M890P attention heads must divide across output groups")
        if (ratio and args.head_dim != 512) or (
            ratio == 4 and args.index_head_dim != 128
        ):
            raise ValueError(
                "M890P compressed attention needs main head512 and C4 index head128"
            )
        object.__setattr__(self, "_provider", provider)
        self._args, self._layer_id = args, layer_id
        dense = dict(tp_size=tp, tp_rank=rank, platform_provider=provider)
        self.wq_a = PpuOwnedFP8Linear(args.dim, args.q_lora_rank, **dense)
        self.wq_b = PpuOwnedFP8Linear(
            args.q_lora_rank, args.n_heads * args.head_dim, partition="column", **dense
        )
        self.wkv = PpuOwnedFP8Linear(args.dim, args.head_dim, **dense)
        # Grouped wo_a has its own provider kernel, not a dense FP8 executor.
        self.wo_a = Dsv4FP8Parameters(
            args.n_heads * args.head_dim // args.o_groups,
            args.o_groups * args.o_lora_rank,
            partition="column",
            tp_size=tp,
            tp_rank=rank,
        )
        self.wo_b = PpuOwnedFP8Linear(
            args.o_groups * args.o_lora_rank, args.dim, partition="row", **dense
        )
        self.attn_sink = _parameter((args.n_heads // tp,), torch.float32)
        self.q_norm = _ScalarWeight((args.q_lora_rank,), torch.bfloat16)
        self.kv_norm = _ScalarWeight((args.head_dim,), torch.bfloat16)
        self.compressor = (
            _CompressorParameters(args.dim, args.head_dim, ratio) if ratio else None
        )
        self.indexer = _IndexerParameters(args, provider) if ratio == 4 else None

    def checkpoint_read_spec(self, name, shape):
        if name == "attn_sink":
            args = self._args
            if tuple(shape) != (args.n_heads,):
                raise ValueError(
                    f"Expected {args.n_heads} attention sink heads, got {shape}"
                )
            return make_sharded_read_spec(name, shape, 0, args.tp_size, args.tp_rank)
        prefix, _, subname = name.partition(".")
        submodule = self._modules.get(prefix)
        if subname and isinstance(submodule, RtpModule):
            spec = submodule.resolve_checkpoint_read_spec(subname, tuple(shape))
            return spec.for_checkpoint_name(name) if spec is not None else None
        return None

    def load_weights(self, weights):
        items = weights.items() if isinstance(weights, dict) else weights
        for name, value in items:
            if name != "attn_sink":
                super().load_weights([(name, value)])
                continue
            if name in getattr(self, "_rtp_loaded_weight_names", ()):
                raise RuntimeError("Duplicate PPU attention sink")
            tensor, applied = unwrap_checkpoint_tensor(value)
            spec = self.checkpoint_read_spec(
                name, applied.source_shape if applied else tuple(tensor.shape)
            )
            if applied is None:
                tensor, _ = unwrap_checkpoint_tensor(spec.materialize(tensor))
            elif applied.slices != spec.slices or applied.shard_rank != spec.shard_rank:
                raise ValueError("PPU attention sink belongs to another TP rank")
            copy_weight_(self.attn_sink, tensor, name)
            self._mark_weight_loaded(name)

    def bind_runtime(self):
        self.validate_weights_loaded()
        self.wkv.validate_runtime_device(self.wkv.weight.device)
        from rtp_llm.models_py.new_models.deepseek_v4.backbone import _attention_kwargs

        # The provider is the source of truth for prefill and decode. In particular
        # decode owns the streams and hooks for merged Q/KV and overlapping indexing.
        return self._provider.build_attention(
            None, owned=self, **_attention_kwargs(self._args, self._layer_id)
        )


def create_ppu_attention(args, layer_id, platform_provider):
    return PpuAttentionParameters(args, layer_id, platform_provider)
