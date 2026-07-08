"""Two-pass user-profile attention impl (no custom mask).

Replaces the dense-custom_mask FlashInfer path for the BERT reranker
user-profile branch. Numerics live in bert_uqi_two_pass_core (unit-tested by
file-path import against the eager oracle AND the old custom_mask path);
this module is the rtp_llm-facing shell:

  - BertUqiTwoPassAttnOp: ONE per BertModel, persistent across requests —
    wrappers (and their 8MB int-workspace + 8MB pinned buffers) are created
    once, plan() re-runs per request (FlashInfer's documented reuse pattern).
  - BertUqiTwoPassImpl: thin per-request facade satisfying FMHAImplBase.
    Exposes uqi_perm / uqi_inv_perm for BertModel.forward to permute hidden
    states into [A...|B...] per-sequence layout before the layer stack.
"""

from typing import Optional

import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

from rtp_llm.models_py.modules.factory.attention.block_mask import (
    BertUqiTwoPassSchedule,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.bert_uqi_two_pass_core import (
    plan_two_pass,
    run_two_pass,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    get_py_flashinfer_workspace_buffer,
    release_py_flashinfer_workspace_buffer,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    ParamsBase,
    PyAttentionInputs,
    get_scalar_type,
    rtp_llm_ops,
)


class BertUqiTwoPassAttnOp(object):
    """Persistent two-pass attention op. Create once, prepare() per request."""

    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.g_workspace_buffer_p1 = get_py_flashinfer_workspace_buffer()
        self.g_workspace_buffer_p2 = get_py_flashinfer_workspace_buffer()
        self.local_head_num = attn_configs.head_num
        self.local_kv_head_num = attn_configs.kv_head_num
        self.head_dim_qk = attn_configs.size_per_head
        self.head_dim_vo = attn_configs.size_per_head
        self.page_size = attn_configs.kernel_tokens_per_block
        self.wrapper_p1 = BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer_p1, backend="auto"
        )
        self.wrapper_p2 = BatchPrefillWithRaggedKVCacheWrapper(
            self.g_workspace_buffer_p2, backend="auto"
        )
        self.schedule: Optional[BertUqiTwoPassSchedule] = None
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()

    def __del__(self):
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer_p1)
        release_py_flashinfer_workspace_buffer(self.g_workspace_buffer_p2)

    def prepare(
        self, attn_inputs: PyAttentionInputs, schedule: BertUqiTwoPassSchedule
    ) -> ParamsBase:
        # fill_params mirrors PyFlashinferPrefillAttnOp.prepare (C++ side reads
        # these from PyModelOutputs). Fresh params object per request — cheap,
        # and avoids aliasing a params object the previous step may still hold.
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        kv_block_id_host = attn_inputs.kv_cache_kernel_block_id_host
        if kv_block_id_host is None:
            kv_block_id_host = torch.empty(0, dtype=torch.int32)
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            kv_block_id_host,
            self.page_size,
        )
        self.schedule = schedule
        plan_two_pass(
            self.wrapper_p1,
            self.wrapper_p2,
            schedule,
            self.local_head_num,
            self.local_kv_head_num,
            self.head_dim_qk,
            get_scalar_type(attn_inputs.dtype),
        )
        return self.fmha_params

    def forward(
        self, qkv: torch.Tensor, kv_cache: Optional[LayerKVCache]
    ) -> torch.Tensor:
        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(
            qkv,
            [
                self.head_dim_qk * self.local_head_num,
                self.head_dim_qk * self.local_kv_head_num,
                self.head_dim_vo * self.local_kv_head_num,
            ],
            dim=-1,
        )
        q = q.reshape(q.shape[0], self.local_head_num, self.head_dim_qk)
        k = k.reshape(k.shape[0], self.local_kv_head_num, self.head_dim_qk)
        v = v.reshape(v.shape[0], self.local_kv_head_num, self.head_dim_vo)
        return run_two_pass(self.wrapper_p1, self.wrapper_p2, self.schedule, q, k, v)


class BertUqiTwoPassImpl(FMHAImplBase):
    """Per-request facade over the persistent op."""

    def __init__(
        self,
        op: BertUqiTwoPassAttnOp,
        attn_inputs: PyAttentionInputs,
        schedule: BertUqiTwoPassSchedule,
    ) -> None:
        self.fmha_impl = op
        self.attn_inputs = attn_inputs
        self.fmha_params = op.prepare(attn_inputs, schedule)
        # BertModel.forward reads these to permute hidden states (None => identity)
        self.uqi_perm = schedule.perm
        self.uqi_inv_perm = schedule.inv_perm

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        return self.fmha_impl.forward(qkv, kv_cache)

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        # Never registered with the factory — constructed explicitly by BertModel.
        return attn_inputs.is_prefill

    def support_cuda_graph(self) -> bool:
        return False
