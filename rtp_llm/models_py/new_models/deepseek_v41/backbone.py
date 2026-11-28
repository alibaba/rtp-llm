"""Direct V4.1 block/transformer owners over unchanged main execution."""

import os
import weakref
from functools import partial

import torch
from torch import nn

from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.dsv41.bounded_replay import (
    enabled as bounded_replay_enabled,
)
from rtp_llm.models_py.new_models.deepseek_v4.load_geometry import make_lm_head, vocabulary_allocation
from rtp_llm.models_py.new_models.deepseek_v41.attention import (
    OwnedV41Attention,
    OwnedV41CommitAttention,
)
from rtp_llm.models_py.new_models.deepseek_v41.globals import OwnedV41Matrix
from rtp_llm.models_py.new_models.deepseek_v41.moe import OwnedV41MoE
from rtp_llm.models_py.weight_mapper import WeightsMapper


class OwnedDelayedHC(RtpModule):
    def __init__(self, args, layer_id, name):
        RtpModule.__init__(self)
        self.dim, self.hc_mult = args.dim, args.hc_mult
        self.hc_sinkhorn_iters, self.norm_eps, self.hc_eps = (
            args.hc_sinkhorn_iters,
            args.norm_eps,
            args.hc_eps,
        )
        self.layer_id, self.name = layer_id, name
        width = args.hc_mult * (args.hc_mult + 2)
        self.fn = nn.Parameter(
            torch.empty(width, args.hc_mult * args.dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.base = nn.Parameter(
            torch.empty(width, dtype=torch.float32), requires_grad=False
        )
        self.scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self._previous_ref = None
        self.pre_mix_out = None

    def set_previous(self, previous):
        self._previous_ref = weakref.ref(previous) if previous is not None else None


class _HeadReference(nn.Module):
    """Parameter-free delayed head reference during checkpoint staging."""

    def __init__(self, last_ffn):
        super().__init__()
        self._last_ffn_ref = weakref.ref(last_ffn)


class OwnedV41Block(RtpModule):
    attention_type = OwnedV41Attention
    commit_attention_type = OwnedV41CommitAttention
    hc_type = OwnedDelayedHC
    norm_type = RMSNorm

    def __init__(self, args, layer_id, shared_attention, *, checkpoint_path=None):
        RtpModule.__init__(self)
        self.layer_id, self.fp8_kv_cache = layer_id, args.fp8_kv_cache
        self.engram = self.engram_hashes = self.engram_token_mask = None
        self._cp_sync_after_attn_done = False
        self.attn = (
            self.commit_attention_type if args.commit_only else self.attention_type
        )(args, layer_id, shared_attention)
        self._prefill_fast_hc_impls_cached = None
        if args.commit_only:
            self.ffn = self.attn_norm = self.ffn_norm = self.attn_hc = self.ffn_hc = (
                None
            )
            return
        prefix = getattr(args, "checkpoint_layer_prefix", "layers")
        vision_layers = getattr(args, "vision_router_bias_layers", {}).get(prefix, ())
        self.ffn = OwnedV41MoE(
            layer_id,
            args.dim,
            args.moe_inter_dim,
            args.n_routed_experts,
            args.n_activated_experts,
            args.n_shared_experts,
            score_func=args.score_func,
            route_scale=args.route_scale,
            swiglu_limit=args.swiglu_limit,
            n_hash_layers=0,
            vocab_size=args.vocab_size,
            prefix=f"{prefix}.{layer_id}.ffn",
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            ep_size=args.ep_size,
            ep_rank=args.ep_rank,
            max_tokens_per_rank=args.max_tokens_per_rank,
            is_decode_role=args.is_decode_role,
            strategy=args.moe_strategy,
            model_type="deepseek_v41",
            shared_fp8_block_size=32,
            has_vision_router_bias=layer_id in vision_layers,
        )
        self.attn_norm = self.norm_type(args.dim, args.norm_eps, torch.bfloat16)
        self.ffn_norm = self.norm_type(args.dim, args.norm_eps, torch.bfloat16)
        self.attn_hc = self.hc_type(args, layer_id, "attn")
        self.ffn_hc = self.hc_type(args, layer_id, "ffn")
        # Engram is populated by the layer-local host source loader after
        # ordinary parameters; never stage its ~94 GiB tables onto a GPU.


class OwnedV41Transformer(RtpModule):
    block_type = OwnedV41Block
    head_type = _HeadReference
    norm_type = RMSNorm

    def __init__(
        self,
        args,
        *,
        checkpoint_path=None,
        layer_prefix="layers",
        global_prefix="",
        head_dtype=torch.bfloat16,
    ):
        RtpModule.__init__(self)
        if (
            not args.v41_config
            or args.n_layers < 1
            or len(args.compress_ratios) < args.n_layers
        ):
            raise ValueError(
                "V4.1 requires its actual config and per-layer attention ratios"
            )
        if args.n_mtp_layers != 0:
            raise ValueError("V4.1 draft is a separate model owner")
        self.args, self.max_seq_len, self.hc_mult = args, args.max_seq_len, args.hc_mult
        self.fp8_kv_cache, self.commit_only = args.fp8_kv_cache, args.commit_only
        self.swa_bounded_replay = bounded_replay_enabled()
        if self.swa_bounded_replay and (
            not args.fp8_kv_cache
            or os.environ.get("DSV41_CED", "0").lower()
            not in ("1", "true", "yes", "on")
        ):
            raise ValueError(
                "DSV41_SWA_BOUNDED_REPLAY requires V4.1 FP8 and DSV41_CED=1"
            )
        self._checkpoint_layer_prefix, self._checkpoint_global_prefix = (
            layer_prefix + ".",
            global_prefix,
        )
        args.checkpoint_layer_prefix = layer_prefix
        shared_attention = {}
        self.layers = nn.ModuleList(
            self.create_block(args, i, shared_attention, checkpoint_path)
            for i in range(args.n_layers)
        )
        if not self.commit_only:
            previous = None
            for layer in self.layers:
                layer.attn_hc.set_previous(previous)
                layer.ffn_hc.set_previous(layer.attn_hc)
                previous = layer.ffn_hc
        self.create_global_modules(args, head_dtype)
        exact = {
            f"{layer_prefix}.{i}.hc_{kind}_{suffix}": f"layers.{i}.{kind}_hc.{suffix}"
            for i in range(args.n_layers)
            for kind in ("attn", "ffn")
            for suffix in ("fn", "base", "scale")
        }
        prefixes = {layer_prefix + ".": "layers."}
        if global_prefix:
            prefixes.update(
                {
                    global_prefix + name + ".": name + "."
                    for name in ("embed", "head", "norm")
                }
            )
        self.WEIGHTS_MAPPER = WeightsMapper(
            exact_mapping=exact, prefix_mapping=prefixes
        )
        self.register_buffer("_mtp_hidden_buffer", None, persistent=False)
        self.register_buffer("_mtp_last_hidden_buffer", None, persistent=False)
        self._mtp_hidden_valid_tokens = self._mtp_last_hidden_valid_tokens = 0
        self._prefill_ws_q_rows = self._prefill_ws_q_dim = (
            self._prefill_ws_full_rows
        ) = self._prefill_ws_main_w = self._prefill_ws_idx_w = 0
        self.capture_aux_hidden_layer_ids = ()

    def create_block(self, args, layer_id, shared_attention, checkpoint_path):
        return self.block_type(
            args, layer_id, shared_attention, checkpoint_path=checkpoint_path
        )

    def create_global_modules(self, args, head_dtype):
        if self.commit_only:
            self.embed = self.head = self.norm = self.head_hc = None
            return
        with vocabulary_allocation(args, "v4.embed.weight"):
            self.embed = OwnedV41Matrix(
                args.vocab_size,
                args.dim,
                shard_dim=1,
                tp_size=args.embedding_tp_size,
                tp_rank=args.embedding_tp_rank,
                dtype=torch.bfloat16,
            )
        self.norm = self.norm_type(args.dim, args.norm_eps, torch.bfloat16)
        with vocabulary_allocation(args, "v4.head.weight"):
            self.head = make_lm_head(args, head_dtype)
        self.head_hc = self.head_type(self.layers[-1].ffn_hc)

    @property
    def head_weight(self):
        return None if self.head is None else self.head.weight

    def checkpoint_weight_name_filter(self):
        if self.commit_only:
            required = {
                f"{self._checkpoint_layer_prefix}{i}.attn.{name}"
                for i in range(len(self.layers))
                for name in ("wkv.weight", "wkv.scale", "kv_norm.weight")
            }
            return required.__contains__
        globals_ = {
            self._checkpoint_global_prefix + name
            for name in ("embed.weight", "head.weight", "norm.weight")
        }
        return lambda name: name in globals_ or (
            ".engram." not in name
            and any(
                name.startswith(f"{self._checkpoint_layer_prefix}{i}.")
                for i in range(len(self.layers))
            )
        )

    def checkpoint_read_spec(self, name, shape):
        return self.mapped_child_read_spec(name, shape)

    def load_weights(self, weights):
        items = weights.items() if isinstance(weights, dict) else weights
        RtpModule.load_weights(self, self.WEIGHTS_MAPPER.apply(items))


def _checkpoint_native_hc(args, layer_id, name):
    from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCUnit

    # Use the exact executor: its type is part of the mega_mhc and decode
    # transition contract. Register its final contiguous views as the direct
    # checkpoint destinations. No second tensor set or post-load copy.
    width = args.hc_mult * (args.hc_mult + 2)
    unit = DelayedHCUnit(
        torch.empty(width, args.hc_mult * args.dim, dtype=torch.float32),
        torch.empty(width, dtype=torch.float32),
        torch.empty(3, dtype=torch.float32),
        dim=args.dim,
        hc_mult=args.hc_mult,
        hc_sinkhorn_iters=args.hc_sinkhorn_iters,
        norm_eps=args.norm_eps,
        hc_eps=args.hc_eps,
        layer_id=layer_id,
        name=name,
    )
    for key in ("fn", "base", "scale"):
        setattr(unit, key, nn.Parameter(getattr(unit, key), requires_grad=False))
    return unit


def _checkpoint_cuda_norm(dim, eps, dtype, *, implementation=None):
    from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm as CudaRMSNorm

    return CudaRMSNorm(
        nn.Parameter(torch.empty(dim, dtype=dtype), requires_grad=False),
        eps,
        implementation=implementation,
    )


def _target_block_norm_factory(device):
    # CPU/meta is only the initial parameter location; select for serving.
    if (
        device is not None
        and torch.device(device).type == "cuda"
        and torch.version.cuda is not None
    ):
        from rtp_llm.models_py.new_models.deepseek_v4.cuda_rms import (
            native_rmsnorm,
            use_native_rms,
        )

        if use_native_rms(device):
            return partial(_checkpoint_cuda_norm, implementation=native_rmsnorm)
    return _checkpoint_cuda_norm


def runtime_backbone_class(*, target_block_device=None):
    """Compose the real main runtime only for production construction.

    Pure checkpoint staging does not import native runtime. Production does
    not substitute an ABI or a CPU attention implementation on import failure.
    """
    from rtp_llm.models_py.modules.dsv41.block import Block
    from rtp_llm.models_py.modules.dsv41.fp8.attention import (
        _configure_flash_mla_l2_persist,
    )
    from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8
    from rtp_llm.models_py.modules.dsv41.fp8.attention_v41_commit import (
        CommitOnlyAttentionV41FP8,
    )
    from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCHead
    from rtp_llm.models_py.modules.dsv41.transformer import V4Transformer

    class RuntimeAttention(AttentionV41FP8, OwnedV41Attention):
        load_weights = OwnedV41Attention.load_weights

        def __init__(self, args, layer_id, shared_attention):
            # The owner bypasses AttentionFP8.__init__, but runs its MLA kernels.
            _configure_flash_mla_l2_persist()
            OwnedV41Attention.__init__(self, args, layer_id, shared_attention)

    class RuntimeCommitAttention(CommitOnlyAttentionV41FP8, OwnedV41CommitAttention):
        __init__ = OwnedV41CommitAttention.__init__
        load_weights = OwnedV41CommitAttention.load_weights

    class RuntimeBlock(Block, OwnedV41Block):
        __init__ = OwnedV41Block.__init__
        _dsv41_native_decode_block = Block
        attention_type = RuntimeAttention
        commit_attention_type = RuntimeCommitAttention
        hc_type = staticmethod(_checkpoint_native_hc)
        norm_type = staticmethod(_target_block_norm_factory(target_block_device))

    class RuntimeTransformer(V4Transformer, OwnedV41Transformer):
        __init__ = OwnedV41Transformer.__init__
        block_type = RuntimeBlock
        head_type = DelayedHCHead
        norm_type = staticmethod(_checkpoint_cuda_norm)

    return RuntimeTransformer
