"""Checkpoint-owned V4 Flash backbone with M890P block/transformer execution."""

import torch
from torch import nn

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4.ppu.block import PpuBlock
from rtp_llm.models_py.modules.dsv4.ppu.transformer import PpuV4Transformer
from rtp_llm.models_py.new_models.deepseek_v4.backbone import OwnedV4Transformer
from rtp_llm.models_py.new_models.deepseek_v4.hc import _hc_parameters
from rtp_llm.models_py.new_models.deepseek_v4.load_geometry import make_lm_head


def _norm(args, dtype):
    return RMSNorm(
        nn.Parameter(torch.empty(args.dim, dtype=dtype), requires_grad=False),
        args.norm_eps,
    )


class _HCParameters(RtpModule):
    def __init__(self, args):
        super().__init__()
        self.fn, self.base, self.scale = _hc_parameters(
            args.dim, args.hc_mult, head=False
        )

    def bind_runtime(self, args, provider, layer_id, name):
        return provider.build_hc_unit(
            self.fn,
            self.base,
            self.scale,
            dim=args.dim,
            hc_mult=args.hc_mult,
            hc_sinkhorn_iters=args.hc_sinkhorn_iters,
            norm_eps=args.norm_eps,
            hc_eps=args.hc_eps,
            layer_id=layer_id,
            name=name,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
        )


class _BoundChild:
    """Lookups see native execution; registration always sees the parameter owner."""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, block, owner=None):
        if block is None:
            return self
        bound = block._bound_children
        return (block._modules if bound is None else bound)[self.name]


class PpuOwnedV4Block(PpuBlock, RtpModule):
    """Keep parameters in the checkpoint tree; expose native runtime children."""

    def __init__(self, args, layer_id, *, platform_provider, params_dtype):
        from .ppu_attention import create_ppu_attention
        from .ppu_moe import create_ppu_moe

        RtpModule.__init__(self)
        self.layer_id = layer_id
        self.tp_size, self.tp_rank, self.dim = args.tp_size, args.tp_rank, args.dim
        self.fp8_kv_cache = args.fp8_kv_cache
        self._platform_provider = platform_provider
        self._cp_sync_after_attn_done = False
        self._mega_csa_adapter = self._mega_hca_adapter = None
        self._bound_children = None
        self.attn = create_ppu_attention(args, layer_id, platform_provider)
        self.ffn = create_ppu_moe(args, layer_id, platform_provider)
        self.attn_norm = _norm(args, params_dtype)
        self.ffn_norm = _norm(args, params_dtype)
        self.attn_hc = _HCParameters(args)
        self.ffn_hc = _HCParameters(args)
        self._prefill_fast_hc_impls_cached = None

    # NewLoader dispatches into registered children, so parameter names and
    # ownership do not change when the runtime binds. Serving code uses normal
    # attribute access; in particular its exact PPU executor type checks still
    # see the original modules rather than wrapper or CUDA classes.
    attn = _BoundChild()
    ffn = _BoundChild()
    attn_hc = _BoundChild()
    ffn_hc = _BoundChild()

    def bind_runtime(self, args):
        if self._bound_children is None:
            children = {
                "attn": self.attn.bind_runtime(),
                "ffn": self.ffn.bind_runtime(args),
            }
            for name in ("attn", "ffn"):
                key = name + "_hc"
                children[key] = self._modules[key].bind_runtime(
                    args, self._platform_provider, self.layer_id, name
                )
            self._bound_children = children
            self._prefill_fast_hc_impls_cached = self._resolve_prefill_fast_hc_impls()

    def _apply(self, fn, recurse=True):
        self._bound_children = None
        self._prefill_fast_hc_impls_cached = None
        return super()._apply(fn, recurse=recurse)


class PpuOwnedV4Transformer(PpuV4Transformer, OwnedV4Transformer):
    def __init__(self, args, *, platform_provider, **kwargs):
        if platform_provider is None:
            raise ValueError("PPU V4 requires an instance-owned provider")
        if args.commit_only:
            raise ValueError(
                "PPU Flash target/MTP backbone is not a DSpARK commit role"
            )
        object.__setattr__(self, "platform_provider", platform_provider)
        OwnedV4Transformer.__init__(self, args, **kwargs)

    def create_block(self, args, layer_id, *, params_dtype):
        return PpuOwnedV4Block(
            args,
            layer_id,
            platform_provider=self.platform_provider,
            params_dtype=params_dtype,
        )

    def create_global_modules(self, args, params_dtype, head_dtype):
        from .ppu_globals import PpuHiddenShardEmbedding

        self.embed = PpuHiddenShardEmbedding(
            args.vocab_size, args.dim, args.tp_size, args.tp_rank, dtype=params_dtype
        )
        self.norm = _norm(args, params_dtype)
        self.head = make_lm_head(args, head_dtype)
        self.head_hc = self.platform_provider.build_hc_head(
            *_hc_parameters(args.dim, args.hc_mult, head=True),
            dim=args.dim,
            hc_mult=args.hc_mult,
            norm_eps=args.norm_eps,
            hc_eps=args.hc_eps,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
        )

    def bind_runtime(self):
        from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_owned import (
            initialize_owned_mega_decode,
        )

        for block in self.layers:
            block.bind_runtime(self.args)
        # Same optional SM100/103 capability + environment guard as main;
        # this is separate from the PPU provider's own FP4 MoE strategy.
        initialize_owned_mega_decode(self, self.args)
