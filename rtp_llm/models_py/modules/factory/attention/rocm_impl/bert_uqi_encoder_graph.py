"""Fixed-buffer HIP Graph adapter for the existing AITER varlen attention.

Pass one separates A/B; pass two reads each complete sequence. Both use full
query buffers so live B-token counts never change captured operation shapes.
"""
import torch

from rtp_llm.models_py.modules.factory.attention.bert_encoder_graph import BertEncoderGraphImpl
from rtp_llm.models_py.modules.factory.attention.bert_uqi_graph_layout import (
    BertUqiAiterGraphLayout,
)


class BertUqiRocmEncoderGraphImpl(BertEncoderGraphImpl):
    def __init__(self, configs, attn_inputs, token_capacity, max_sequence_length=None):
        # The base constructor invokes _prepare for the startup profile.
        self._host_layout = None
        self._host_metadata = None
        self._device_metadata = None
        self.uqi_perm = None
        self.uqi_inv_perm = None
        self.first_cu = None
        self.full_cu = None
        self.b_mask = None
        self.valid_mask = None
        self.aiter_max_seq_len = max_sequence_length or token_capacity
        super().__init__(
            configs, attn_inputs, token_capacity, max_sequence_length,
            allocate_indptr=False,
        )

    def _prepare(self, attn_inputs, validate_sequence_length):
        # Validate here and upload only the metadata used by this adapter.
        host_lengths = attn_inputs.input_lengths
        if host_lengths.is_cuda:
            raise ValueError("BERT graph planning requires host input lengths")
        if host_lengths.numel() != self.request_capacity:
            raise ValueError("BERT graph request capacity changed")
        prefix = attn_inputs.prefix_lengths
        if prefix is not None and prefix.numel() and bool(prefix.any()):
            raise ValueError("BERT encoder graph does not support prefix KV")
        lengths = host_lengths.tolist()
        if any(n < 0 for n in lengths) or sum(lengths) > self.token_capacity:
            raise ValueError("BERT graph token capacity exceeded")
        limit = self.aiter_max_seq_len
        if not validate_sequence_length:
            # Startup profiles use one bucket-sized synthetic sequence. Split
            # it into legal request slots for fixed max_seqlen AITER planning.
            lengths = [min(limit, max(0, self.token_capacity - i * limit))
                       for i in range(self.request_capacity)]
            if sum(lengths) != self.token_capacity:
                raise ValueError("UQI bucket exceeds request capacity * max_seq_len")
            starts, counts = [-1] * len(lengths), [0] * len(lengths)
        else:
            starts, counts = attn_inputs.uqi_b_starts, attn_inputs.uqi_b_lens
            if starts is None or counts is None or starts.is_cuda or counts.is_cuda:
                raise ValueError("UQI graph requires host segment bounds")
            active = starts.numel()
            if active > len(lengths) or counts.numel() != active or any(lengths[active:]):
                raise ValueError("invalid UQI graph segment metadata")
            starts = starts.tolist() + [-1] * (len(lengths) - active)
            counts = counts.tolist() + [0] * (len(lengths) - active)
        if self._host_layout is None:
            self._host_layout = BertUqiAiterGraphLayout(
                self.token_capacity, self.request_capacity, limit
            )
            self._host_metadata = torch.frombuffer(
                self._host_layout.buffer, dtype=torch.uint8
            )
            self._device_metadata = torch.empty_like(self._host_metadata, device="cuda")
            layout, device = self._host_layout, self._device_metadata
            self.uqi_perm = device[: layout.inv_offset].view(torch.int64)
            self.uqi_inv_perm = device[layout.inv_offset : layout.first_offset].view(torch.int64)
            self.first_cu = device[layout.first_offset : layout.full_offset].view(torch.int32)
            self.full_cu = device[layout.full_offset : layout.b_mask_offset].view(torch.int32)
            self.b_mask = device[layout.b_mask_offset : layout.valid_mask_offset].view(torch.bool).view(-1, 1)
            self.valid_mask = device[layout.valid_mask_offset :].view(torch.bool).view(-1, 1)
        self._host_layout.update(lengths, starts, counts)
        # One blocking upload replaces six temporary tensors/copies. Completing
        # the copy before return also makes reuse of the host byte buffer safe
        # for callers that prepare multiple layouts without an intervening replay.
        self._device_metadata.copy_(self._host_metadata)

    def forward(self, qkv, kv_cache, layer_idx=0):
        if kv_cache is not None:
            raise ValueError("UQI encoder graph cannot use KV cache")
        from aiter import flash_attn_varlen_func

        q, k, v = torch.split(qkv.reshape(self.token_capacity, -1),
                             [self.heads * self.dim, self.kv_heads * self.dim,
                              self.kv_heads * self.dim], dim=-1)
        q = q.reshape(self.token_capacity, self.heads, self.dim)
        k = k.reshape(self.token_capacity, self.kv_heads, self.dim)
        v = v.reshape(self.token_capacity, self.kv_heads, self.dim)
        def run(cu):
            return flash_attn_varlen_func(
                q, k, v, cu, cu, self.aiter_max_seq_len, self.aiter_max_seq_len,
                dropout_p=0.0, causal=False,
            ).reshape(self.token_capacity, self.heads * self.dim)
        local = run(self.first_cu)
        full = run(self.full_cu)
        result = torch.where(self.b_mask, full, local)
        return torch.where(self.valid_mask, result, 0.)
