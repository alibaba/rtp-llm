"""Fixed-capacity, noncausal encoder attention for Transformer-only graphs.

Planning and host metadata updates happen outside capture. The last sequence
owns padding tokens, so bidirectional real sequences never attend to padding.
This implementation is selected explicitly by VisionBert, not by the factory.
"""

import torch
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops.compute_ops import ParamsBase


class BertEncoderGraphParams(ParamsBase):
    def check_recycle(self):
        return True


class BertEncoderGraphImpl(FMHAImplBase):
    def __init__(self, configs, attn_inputs, token_capacity, max_sequence_length=None):
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
        self.cu_host = torch.empty(
            self.request_capacity + 2, dtype=torch.int32, pin_memory=True
        )
        self.cu_device = torch.empty_like(self.cu_host, device="cuda")
        self.wrapper = None
        if torch.version.hip is None:
            from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

            # Each graph owns its planner workspace and indptr storage. Replanning
            # another bucket must not overwrite this graph's execution plan.
            self.workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
            self.kv_cu_device = torch.empty_like(self.cu_device)
            self.wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace,
                use_cuda_graph=True,
                qo_indptr_buf=self.cu_device,
                kv_indptr_buf=self.kv_cu_device,
                backend="fa2",
            )
        # Capture uses one synthetic sequence as long as the aggregate bucket.
        # Only real replay sequences are constrained by position-table length.
        self._prepare(attn_inputs, validate_sequence_length=False)

    def prepare_cuda_graph(self, attn_inputs):
        self._prepare(attn_inputs, validate_sequence_length=True)

    def _prepare(self, attn_inputs, validate_sequence_length):
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
            validate_sequence_length
            and self.max_sequence_length is not None
            and bool((lengths > self.max_sequence_length).any())
        ):
            raise ValueError("BERT graph sequence exceeds position-table capacity")
        real_tokens = int(lengths.sum())
        if real_tokens > self.token_capacity:
            raise ValueError("BERT graph token capacity exceeded")
        self.cu_host[0] = 0
        torch.cumsum(lengths, 0, out=self.cu_host[1:-1])
        self.cu_host[-1] = self.token_capacity
        if self.wrapper is None:
            self.cu_device.copy_(self.cu_host, non_blocking=True)
        else:
            self.wrapper.plan(
                self.cu_host,
                self.cu_host,
                self.heads,
                self.kv_heads,
                self.dim,
                self.dim,
                causal=False,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                o_data_type=self.dtype,
                disable_split_kv=True,
            )

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
                self.cu_device,
                self.cu_device,
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
