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

from rtp_llm.models_py.modules.factory.attention.bert_uqi_schedule import (
    BertUqiTwoPassSchedule,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.bert_uqi_two_pass_core import (
    plan_two_pass,
    run_two_pass,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPrefillImpl,
    _host_i32,
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
        kv_block_id_host = attn_inputs.kv_cache_kernel_block_id
        if kv_block_id_host is None:
            kv_block_id_host = torch.empty(0, dtype=torch.int32)
        self.fmha_params.fill_params(
            _host_i32(attn_inputs.prefix_lengths),
            _host_i32(attn_inputs.sequence_lengths),
            _host_i32(attn_inputs.input_lengths),
            _host_i32(kv_block_id_host),
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


class BertUqiEncoderGraphImpl(PyFlashinferPrefillImpl):
    """CUDA UQI graphs with one FlashInfer masked attention per layer."""

    def __init__(self, configs, attn_inputs, token_capacity, max_sequence_length=None):
        if configs.is_causal:
            raise ValueError("BERT user-segment graph requires noncausal attention")
        if torch.version.hip is not None:
            raise ValueError("UQI encoder graph currently supports CUDA only")
        self.token_capacity = token_capacity
        self.max_sequence_length = max_sequence_length or token_capacity
        self.request_capacity = attn_inputs.input_lengths.numel()
        super().__init__(
            configs,
            attn_inputs,
            is_cuda_graph=True,
            token_capacity=token_capacity,
            max_seq_len=self.max_sequence_length,
            mask_builder=self._mask_plan_args,
        )

    def _mask_plan_args(self, attn_inputs, lengths, padding_slots, for_capture):
        from rtp_llm.models_py.modules.factory.attention.bert_uqi_schedule import (
            build_uqi_graph_layout,
        )

        lengths = lengths.tolist()
        if not for_capture:
            starts = attn_inputs.uqi_b_starts
            b_lengths = attn_inputs.uqi_b_lens
            if (
                starts is None
                or b_lengths is None
                or starts.is_cuda
                or b_lengths.is_cuda
            ):
                raise ValueError("UQI graph requires host segment bounds")
            if (
                starts.numel() > self.request_capacity
                or starts.numel() != b_lengths.numel()
            ):
                raise ValueError("invalid UQI graph request metadata")
            active = starts.numel()
            if any(lengths[active:]):
                raise ValueError("missing UQI graph segment bounds")
            starts = starts.tolist() + [-1] * (len(lengths) - active)
            b_lengths = b_lengths.tolist() + [0] * (len(lengths) - active)
        else:
            # Synthetic startup inputs contain no user segment. Always pass a
            # mask, even here, so capture never specializes to an unmasked path.
            starts, b_lengths = [-1] * len(lengths), [0] * len(lengths)
        layout = build_uqi_graph_layout(lengths, starts, b_lengths, self.token_capacity)
        padding = layout.pop()[0]
        limit = self.max_sequence_length or self.token_capacity
        for i in range(padding_slots):
            size = min(limit, max(0, padding - i * limit))
            layout.append((size, size, 0))
        masks = []
        for n, start, count in layout:
            pos = torch.arange(n, device="cuda")
            is_b = (pos >= start) & (pos < start + count)
            # A -> A; B -> A+B. Separate ragged sequences isolate padding.
            masks.append((is_b[:, None] | ~is_b[None, :]).reshape(-1))
        return {"custom_mask": torch.cat(masks)}
