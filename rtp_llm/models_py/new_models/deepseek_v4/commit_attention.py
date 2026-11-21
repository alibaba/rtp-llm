"""Checkpoint-owned SWA state for the existing V4 DSpARK commit runtime.

Like main's CommitOnlyAttentionFP8, this deliberately does not initialize the
proposal attention constructor. No Q/O, compressor or indexer parameters are
allocated on a dedicated prefill-commit worker.
"""

import torch
from torch import nn

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    _DSV4_FP8_INDEXER_ENTRY_BYTES,
    _DSV4_FP8_KV_ENTRY_BYTES,
    AttentionFP8,
)
from rtp_llm.models_py.modules.dsv4.fp8.linear_owner import Dsv4FP8Linear
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    HCA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
    SWA_KV,
)


class OwnedCommitOnlyAttentionFP8(AttentionFP8):
    def __init__(
        self,
        *,
        layer_id,
        dim,
        n_heads,
        q_lora_rank,
        head_dim,
        rope_head_dim,
        o_lora_rank,
        o_groups,
        window_size,
        compress_ratio,
        compress_rope_theta,
        rope_theta,
        rope_factor,
        beta_fast,
        beta_slow,
        original_seq_len,
        max_batch_size,
        max_seq_len,
        index_n_heads,
        index_head_dim,
        index_topk,
        norm_eps=1e-6,
        tp_size=1,
        tp_rank=0,
    ):
        del (
            compress_rope_theta,
            original_seq_len,
            max_batch_size,
            index_n_heads,
            index_topk,
        )
        RtpModule.__init__(self)
        if compress_ratio != 0:
            raise ValueError("V4 commit-only attention requires SWA (compress_ratio=0)")
        if (
            isinstance(tp_size, bool)
            or not isinstance(tp_size, int)
            or tp_size <= 0
            or isinstance(tp_rank, bool)
            or not isinstance(tp_rank, int)
            or not 0 <= tp_rank < tp_size
        ):
            raise ValueError("Invalid V4 commit-only attention TP partition")
        if n_heads % tp_size or o_groups % tp_size:
            raise ValueError("V4 heads and output groups must be divisible by TP")
        self.layer_id = layer_id
        self.dim = dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.window_size = window_size
        self.compress_ratio = 0
        self.eps = norm_eps
        self.softmax_scale = head_dim**-0.5
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.n_heads = n_heads // tp_size
        self.n_groups = o_groups // tp_size
        self._checkpoint_owned = True
        self._checkpoint_shapes = {}

        # KV is replicated across attention ranks, as in the full V4 owner.
        self.wkv = Dsv4FP8Linear(dim, head_dim)
        self.kv_norm = nn.Parameter(
            torch.empty(head_dim, dtype=torch.bfloat16), requires_grad=False
        )
        self.attn_sink = None
        self.wq_a = self.wq_b = self.wo_b = None
        self.wo_a_w = self.wo_a_s = self.q_norm = None
        self._wo_a_stk_w = self._wo_a_stk_s = None
        self.compressor = self.indexer = None
        self._rope_base = rope_theta
        self._rope_o_seq_len = 0
        self._rope_factor = rope_factor
        self._rope_beta_fast = beta_fast
        self._rope_beta_slow = beta_slow
        self._rope_dim = rope_head_dim
        self._rope_max_seq_len = max_seq_len
        self._fp8_decode_op = None
        self._cp_ctx = None
        self._prefill_meta_shared = None
        self._kv_cache = None
        self._block_tables_by_type = None
        self._pool_spec = {
            SWA_KV: (torch.uint8, _DSV4_FP8_KV_ENTRY_BYTES),
            CSA_KV: (torch.uint8, _DSV4_FP8_KV_ENTRY_BYTES),
            HCA_KV: (torch.uint8, _DSV4_FP8_KV_ENTRY_BYTES),
            INDEXER_KV: (torch.uint8, _DSV4_FP8_INDEXER_ENTRY_BYTES),
            CSA_STATE: (torch.float32, 4 * head_dim),
            HCA_STATE: (torch.float32, 2 * head_dim),
            INDEXER_STATE: (torch.float32, 4 * index_head_dim),
        }
        self.init_rope_cache(self.kv_norm.device)

    def load_weights(self, weights):
        items = weights.items() if isinstance(weights, dict) else weights
        for name, value in items:
            if name not in ("wkv.weight", "wkv.scale", "kv_norm.weight"):
                raise RuntimeError(f"Unexpected V4 commit-only tensor {name!r}")
            super().load_weights([(name, value)])

    def validate_runtime_device(self, device):
        self.wkv.validate_runtime_device(device)

    def process_weights_after_loading(self):
        self.validate_weights_loaded()
        self.validate_runtime_device(self.kv_norm.device)
        self.init_rope_cache(self.kv_norm.device)

    def forward(self, *args, **kwargs):
        raise RuntimeError("Commit-only attention cannot execute proposal attention")
