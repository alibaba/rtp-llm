"""PPU module construction; every Transformer execution method is inherited."""

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.ppu.block import SharedPPUBlock
from rtp_llm.models_py.modules.dsv41.ppu.layers import (
    PPUSharedHCHead,
    PPUSharedRMSNorm,
    PPUSharedEmbedding,
)
from rtp_llm.models_py.modules.dsv41.transformer import V4Transformer
from rtp_llm.utils.model_weight import W


class SharedPPUTransformer(V4Transformer):
    """No private layer loop and no legacy core/runner delegation.

    The construction interface supplies real weights,
    mHC predecessor chain, MoE ops and framework global tensors.
    """

    def __init__(self, args, views, ops, *, enpm_group=None, block_factory=SharedPPUBlock):
        nn.Module.__init__(self)
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.hc_mult = args.hc_mult
        self.commit_only = bool(args.commit_only)
        self.fp8_kv_cache = args.fp8_kv_cache
        # N2 must explicitly bind its public CED/bounded policy. No private
        # planner is installed, and none of the old PPU cache is constructed.
        self.swa_bounded_replay = False
        self.layers = nn.ModuleList(
            block_factory(views, i, ops, enpm_group=enpm_group)
            for i in range(args.n_layers)
        )
        previous = None
        for layer in self.layers:
            if not self.commit_only:
                layer.attn_hc.set_previous(previous)
                previous = layer.ffn_hc
        gw = views.weights.global_weights
        self.embed = None if self.commit_only else PPUSharedEmbedding(gw[W.embedding])
        self.norm = None if self.commit_only else PPUSharedRMSNorm(gw[W.final_ln_gamma], args.norm_eps)
        self.head_weight = None if self.commit_only else gw[W.lm_head]
        if self.head_weight is not None and self.head_weight.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("shared PPU LM head must be FP32 or BF16")
        self.head_hc = None if self.commit_only else PPUSharedHCHead(previous, ops)
        self._dbg_step = 0
        self.register_buffer("_mtp_hidden_buffer", None, persistent=False)
        self.register_buffer("_mtp_last_hidden_buffer", None, persistent=False)
        self._mtp_hidden_valid_tokens = 0
        self._mtp_last_hidden_valid_tokens = 0
        self._prefill_ws_q_rows = 0
        self._prefill_ws_q_dim = 0
        self._prefill_ws_full_rows = 0
        self._prefill_ws_main_w = 0
        self._prefill_ws_idx_w = 0
        self.capture_aux_hidden_layer_ids = ()
        self._ppu_runtime = None

    def _propagate_cp_ctx(self, cp_ctx):
        super()._propagate_cp_ctx(cp_ctx)
        runtime = self._ppu_runtime
        if runtime is not None and runtime.current is not None:
            runtime.current.bind_context(cp_ctx)

    def _hc_head_reduce(self, x):
        reduced = super()._hc_head_reduce(x)
        state = self._ppu_runtime.current if self._ppu_runtime is not None else None
        return state.restore_head(reduced) if state is not None and state.prefill else reduced
