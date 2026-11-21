"""Direct checkpoint ownership for the existing DeepSeek V4 backbone.

Execution methods are inherited from the existing runtime, not reimplemented.
This module is deliberately not a registry entry: runtime/config integration
and its backend validation must be completed before enabling an entire model.
"""

from dataclasses import dataclass

import torch
from torch import nn

from rtp_llm.models_py.layers.embedding import VocabParallelEmbedding
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.dsv4.block import Block
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8
from rtp_llm.models_py.modules.dsv4.transformer import V4Args, V4Transformer
from rtp_llm.models_py.new_models.deepseek_v4.checkpoint_namespace import (
    V4CheckpointNamespace,
)
from rtp_llm.models_py.new_models.deepseek_v4.commit_attention import (
    OwnedCommitOnlyAttentionFP8,
)
from rtp_llm.models_py.new_models.deepseek_v4.hc import OwnedHCHead, OwnedHCUnit
from rtp_llm.models_py.new_models.deepseek_v4.load_geometry import make_lm_head
from rtp_llm.models_py.weight_mapper import WeightsMapper


@dataclass
class OwnedV4Args(V4Args):
    lm_head_tp_size: int = 1
    lm_head_tp_rank: int = 0
    cuda_norm: bool = False
    cuda_native_rms: bool = False


def _checkpoint_norm(args, dtype):
    if not args.cuda_norm:
        return RMSNorm(args.dim, args.norm_eps, dtype)
    from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm as CudaRMSNorm

    # The CUDA V4 prefill runner recognizes this runtime class and writes
    # the result back into its workspace. Only change who owns the weight.
    weight = nn.Parameter(torch.empty(args.dim, dtype=dtype), requires_grad=False)
    if args.cuda_native_rms:
        from .cuda_rms import native_rmsnorm

        return CudaRMSNorm(weight, args.norm_eps, implementation=native_rmsnorm)
    return CudaRMSNorm(weight, args.norm_eps)


def _attention_kwargs(args, layer_id):
    return dict(
        layer_id=layer_id,
        dim=args.dim,
        n_heads=args.n_heads,
        q_lora_rank=args.q_lora_rank,
        head_dim=args.head_dim,
        rope_head_dim=args.rope_head_dim,
        o_lora_rank=args.o_lora_rank,
        o_groups=args.o_groups,
        window_size=args.window_size,
        compress_ratio=args.compress_ratios[layer_id],
        compress_rope_theta=args.compress_rope_theta,
        rope_theta=args.rope_theta,
        rope_factor=args.rope_factor,
        beta_fast=args.beta_fast,
        beta_slow=args.beta_slow,
        original_seq_len=args.original_seq_len,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        index_n_heads=args.index_n_heads,
        index_head_dim=args.index_head_dim,
        index_topk=args.index_topk,
        norm_eps=args.norm_eps,
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
    )


class OwnedV4Block(Block, RtpModule):
    def __init__(self, args, layer_id, *, params_dtype=torch.bfloat16):
        # Block.__init__ is the legacy ModelWeights constructor; retain only
        # its execution methods and initialize the direct parameter tree here.
        RtpModule.__init__(self)
        self.layer_id = layer_id
        self.fp8_kv_cache = args.fp8_kv_cache
        self.commit_only = bool(args.commit_only)
        self._cp_sync_after_attn_done = False
        self._mega_csa_adapter = None
        self._mega_hca_adapter = None
        attn_cls = OwnedCommitOnlyAttentionFP8 if self.commit_only else AttentionFP8
        self.attn = attn_cls(**_attention_kwargs(args, layer_id))
        if self.commit_only:
            self.ffn = self.attn_norm = self.ffn_norm = None
            self.attn_hc = self.ffn_hc = None
            self._prefill_fast_hc_impls_cached = None
            return
        self.ffn = self.create_ffn(args, layer_id)
        self.attn_norm = _checkpoint_norm(args, params_dtype)
        self.ffn_norm = _checkpoint_norm(args, params_dtype)
        for kind in ("attn", "ffn"):
            setattr(
                self,
                kind + "_hc",
                OwnedHCUnit(
                    dim=args.dim,
                    hc_mult=args.hc_mult,
                    hc_sinkhorn_iters=args.hc_sinkhorn_iters,
                    norm_eps=args.norm_eps,
                    hc_eps=args.hc_eps,
                    layer_id=layer_id,
                    name=kind,
                ),
            )
        self._prefill_fast_hc_impls_cached = self._resolve_prefill_fast_hc_impls()

    def create_ffn(self, args, layer_id):
        # The shared FP8_FP4 owner retains the existing chunked runtime. It is
        # not the ModelOpt/NVFP4 block16 implementation.
        from rtp_llm.models_py.new_models.deepseek_v4.moe import OwnedFp8Fp4MoE

        return OwnedFp8Fp4MoE(
            layer_id=layer_id,
            dim=args.dim,
            moe_inter_dim=args.moe_inter_dim,
            n_routed_experts=args.n_routed_experts,
            n_activated_experts=args.n_activated_experts,
            n_shared_experts=args.n_shared_experts,
            score_func=args.score_func,
            route_scale=args.route_scale,
            swiglu_limit=args.swiglu_limit,
            n_hash_layers=args.n_hash_layers,
            vocab_size=args.vocab_size,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            ep_size=args.ep_size,
            ep_rank=args.ep_rank,
            world_size=args.world_size,
            world_rank=args.world_rank,
            max_tokens_per_rank=args.max_tokens_per_rank,
            is_decode_role=args.is_decode_role,
            strategy=args.moe_strategy,
            n_physical_experts=args.n_physical_experts,
            prefix=f"layers.{layer_id}.ffn",
            shared_fp8_block_size=128,
        )


class OwnedV4Transformer(V4Transformer, RtpModule):
    """Runtime-compatible parameter tree, with override points for draft roles."""

    def __init__(
        self,
        args: OwnedV4Args,
        *,
        params_dtype=torch.bfloat16,
        head_dtype=None,
        layer_prefix="layers",
        global_prefix="",
    ):
        RtpModule.__init__(self)
        if args.n_layers <= 0 or len(args.compress_ratios) < args.n_layers:
            raise ValueError("V4 must provide an attention ratio for each target layer")
        if args.n_mtp_layers != 0:
            raise ValueError("V4 backbone does not embed a second MTP model")
        if params_dtype != torch.bfloat16:
            raise ValueError("V4 native FP8/FP4 runtime requires BF16 hidden states")
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.hc_mult = args.hc_mult
        self.fp8_kv_cache = args.fp8_kv_cache
        self.commit_only = bool(args.commit_only)
        self._checkpoint_layer_prefix = layer_prefix + "."
        self._checkpoint_global_prefix = global_prefix
        self.layers = nn.ModuleList(
            self.create_block(args, index, params_dtype=params_dtype)
            for index in range(args.n_layers)
        )
        self.create_global_modules(args, params_dtype, head_dtype or params_dtype)
        exact = {
            global_prefix + "hc_head_" + suffix: "head_hc." + suffix
            for suffix in ("fn", "base", "scale")
        }
        exact.update(
            {
                f"{layer_prefix}.{index}.hc_{kind}_{suffix}": f"layers.{index}.{kind}_hc.{suffix}"
                for index in range(args.n_layers)
                for kind in ("attn", "ffn")
                for suffix in ("fn", "base", "scale")
            }
        )
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
        self._dbg_step = 0
        self._mega_csa_runtime = None
        self._mega_decode_enabled = False
        self.register_buffer("_mtp_hidden_buffer", None, persistent=False)
        self.register_buffer("_mtp_last_hidden_buffer", None, persistent=False)
        self._mtp_hidden_valid_tokens = self._mtp_last_hidden_valid_tokens = 0
        self._prefill_ws_q_rows = self._prefill_ws_q_dim = (
            self._prefill_ws_full_rows
        ) = 0
        self._prefill_ws_main_w = self._prefill_ws_idx_w = 0
        self.capture_aux_hidden_layer_ids = ()

    def create_block(self, args, layer_id, *, params_dtype):
        return OwnedV4Block(args, layer_id, params_dtype=params_dtype)

    def create_global_modules(self, args, params_dtype, head_dtype):
        if self.commit_only:
            self.embed = self.norm = self.head = self.head_hc = None
            return
        # Existing V4 Python execution embeds full hidden vectors. Attention
        # TP and head TP are independent; do not slice embedding hidden_dim.
        self.embed = VocabParallelEmbedding(
            args.vocab_size, args.dim, params_dtype=params_dtype
        )
        self.norm = _checkpoint_norm(args, params_dtype)
        self.head = make_lm_head(args, head_dtype)
        self.head_hc = OwnedHCHead(
            dim=args.dim,
            hc_mult=args.hc_mult,
            norm_eps=args.norm_eps,
            hc_eps=args.hc_eps,
        )

    @property
    def head_weight(self):
        return None if self.head is None else self.head.weight

    def checkpoint_weight_name_filter(self):
        return V4CheckpointNamespace(
            layer_prefix=self._checkpoint_layer_prefix,
            global_prefix=self._checkpoint_global_prefix,
            layer_count=len(self.layers),
            commit_only=self.commit_only,
        ).accepts

    def load_standalone_checkpoint(self, weights):
        # The production wrapper maps checkpoint names before dispatch. Do not
        # override load_weights here: a custom child loader cuts off RtpModule's
        # tree traversal, hiding the expert's skip and projection's TP range.
        items = weights.items() if isinstance(weights, dict) else weights
        self.load_weights(self.WEIGHTS_MAPPER.apply(items))

    def moe_chunk_plan(self, tokens, device):
        from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
            synchronized_moe_chunk_plan,
        )

        return synchronized_moe_chunk_plan(self.layers, tokens, device)
