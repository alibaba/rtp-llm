"""Fixed-capacity, noncausal encoder attention for Transformer-only graphs.

Planning and host metadata updates happen outside capture. The last sequence
owns padding tokens, so bidirectional real sequences never attend to padding.
BertModel selects this implementation explicitly for graph attention.
"""

import torch
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops.compute_ops import ParamsBase


class BertEncoderGraphParams(ParamsBase):
    def check_recycle(self):
        return True


class BertEncoderGraphImpl(FMHAImplBase):
    def __init__(
        self,
        configs,
        attn_inputs,
        token_capacity,
        max_sequence_length=None,
        custom_mask=False,
        allocate_indptr=True,
    ):
        if configs.is_causal:
            raise ValueError("BERT encoder graph requires noncausal attention")
        self.heads = configs.head_num
        self.kv_heads = configs.kv_head_num
        self.dim = configs.size_per_head
        self.dtype = configs.dtype
        self.token_capacity = token_capacity
        self.max_sequence_length = max_sequence_length
        self.request_capacity = attn_inputs.input_lengths.numel()
        self.fmha_params = BertEncoderGraphParams()
        self.cu_seqlens_host = None
        self.cu_seqlens_device = None
        if allocate_indptr:
            self.cu_seqlens_host = torch.empty(
                self.request_capacity + 2, dtype=torch.int32, pin_memory=True
            )
            self.cu_seqlens_device = torch.empty_like(
                self.cu_seqlens_host, device="cuda"
            )
        self.wrapper = None
        if torch.version.hip is None:
            from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

            # Each graph owns its planner workspace and indptr storage. Replanning
            # another bucket must not overwrite this graph's execution plan.
            self.workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
            self.kv_cu_seqlens_device = torch.empty_like(self.cu_seqlens_device)
            mask_buffers = {}
            if custom_mask:
                # Each sequence is packed separately and may round up one byte.
                if token_capacity * token_capacity > torch.iinfo(torch.int32).max:
                    raise ValueError("UQI mask exceeds int32 mask-indptr capacity")
                mask_bytes = (
                    (token_capacity * token_capacity + 7) // 8
                    + self.request_capacity
                    + 1
                )
                self.custom_mask_buffer = torch.empty(
                    mask_bytes, dtype=torch.uint8, device="cuda"
                )
                self.mask_indptr_buffer = torch.empty_like(self.cu_seqlens_device)
                mask_buffers = dict(
                    custom_mask_buf=self.custom_mask_buffer,
                    mask_indptr_buf=self.mask_indptr_buffer,
                )
            self.wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace,
                use_cuda_graph=True,
                qo_indptr_buf=self.cu_seqlens_device,
                kv_indptr_buf=self.kv_cu_seqlens_device,
                backend="fa2",
                **mask_buffers,
            )
        # Capture uses one synthetic sequence as long as the aggregate bucket.
        # Only real replay sequences are constrained by position-table length.
        self._prepare(attn_inputs, for_capture=True)

    def prepare_cuda_graph(self, attn_inputs):
        self._prepare(attn_inputs, for_capture=False)

    def _prepare(self, attn_inputs, for_capture):
        """Plan synthetic capture inputs or validate and plan a real replay batch."""
        lengths = attn_inputs.input_lengths
        if lengths.is_cuda:
            raise ValueError("BERT graph planning requires host input lengths")
        if lengths.numel() != self.request_capacity:
            raise ValueError("BERT graph request capacity changed")
        prefix = attn_inputs.prefix_lengths
        if prefix is not None and prefix.numel() and bool(prefix.any()):
            raise ValueError("BERT encoder graph does not support prefix KV")
        if bool((lengths < 0).any()):
            raise ValueError("BERT graph sequence lengths must be nonnegative")
        if (
            not for_capture
            and self.max_sequence_length is not None
            and bool((lengths > self.max_sequence_length).any())
        ):
            raise ValueError("BERT graph sequence exceeds position-table capacity")
        real_tokens = int(lengths.sum())
        if real_tokens > self.token_capacity:
            raise ValueError("BERT graph token capacity exceeded")
        self.cu_seqlens_host[0] = 0
        torch.cumsum(lengths, 0, out=self.cu_seqlens_host[1:-1])
        self.cu_seqlens_host[-1] = self.token_capacity
        if self.wrapper is None:
            self.cu_seqlens_device.copy_(self.cu_seqlens_host, non_blocking=True)
        else:
            self.wrapper.plan(
                self.cu_seqlens_host,
                self.cu_seqlens_host,
                self.heads,
                self.kv_heads,
                self.dim,
                self.dim,
                causal=False,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                o_data_type=self.dtype,
                disable_split_kv=True,
                **self._mask_plan_args(attn_inputs, for_capture),
            )

    def _mask_plan_args(self, attn_inputs, for_capture):
        return {}

    def forward(self, qkv, kv_cache, layer_idx=0):
        if kv_cache is not None:
            raise ValueError("BERT encoder graph cannot use KV cache")
        q, k, v = torch.split(
            qkv.reshape(self.token_capacity, -1),
            [self.heads * self.dim, self.kv_heads * self.dim, self.kv_heads * self.dim],
            dim=-1,
        )
        q = q.reshape(self.token_capacity, self.heads, self.dim)
        k = k.reshape(self.token_capacity, self.kv_heads, self.dim)
        v = v.reshape(self.token_capacity, self.kv_heads, self.dim)
        if self.wrapper is not None:
            output = self.wrapper.run(q, k, v)
        else:
            from aiter import flash_attn_varlen_func

            output = flash_attn_varlen_func(
                q,
                k,
                v,
                self.cu_seqlens_device,
                self.cu_seqlens_device,
                self.token_capacity,
                self.token_capacity,
                dropout_p=0.0,
                causal=False,
            )
        return output.reshape(self.token_capacity, self.heads * self.dim)

    @staticmethod
    def support(configs, attn_inputs):
        return attn_inputs.is_prefill and not configs.is_causal

    def support_cuda_graph(self):
        return True


class BertUqiEncoderGraphImpl(BertEncoderGraphImpl):
    """CUDA UQI graphs with one FlashInfer masked attention per layer."""

    def __init__(self, configs, attn_inputs, token_capacity, max_sequence_length=None):
        if torch.version.hip is not None:
            raise ValueError("UQI encoder graph currently supports CUDA only")
        super().__init__(
            configs, attn_inputs, token_capacity, max_sequence_length, custom_mask=True
        )

    def _mask_plan_args(self, attn_inputs, for_capture):
        from rtp_llm.models_py.modules.factory.attention.bert_uqi_graph_layout import (
            build_uqi_graph_layout,
        )

        lengths = attn_inputs.input_lengths.tolist()
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
        masks = []
        for n, start, count in layout:
            pos = torch.arange(n, device="cuda")
            is_b = (pos >= start) & (pos < start + count)
            # A -> A; B -> A+B. Separate ragged sequences isolate padding.
            masks.append((is_b[:, None] | ~is_b[None, :]).reshape(-1))
        return {"custom_mask": torch.cat(masks)}
