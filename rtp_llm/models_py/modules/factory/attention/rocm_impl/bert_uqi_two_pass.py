"""ROCm facade for BERT profile attention; metadata is request-local."""
import torch

from rtp_llm.models_py.modules.factory.attention.bert_uqi_schedule import (
    BertUqiAiterGraphLayout,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.modules.factory.attention.rocm_impl.bert_uqi_two_pass_core import (
    prepare_two_pass, run_two_pass,
)
from rtp_llm.ops.compute_ops import ParamsBase


class BertUqiParams(ParamsBase):
    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    def check_recycle(self):
        return True


class BertUqiTwoPassAttnOp:
    def __init__(self, attn_configs):
        self.heads = attn_configs.head_num
        self.kv_heads = attn_configs.kv_head_num
        self.dim = attn_configs.size_per_head

    def forward(self, qkv, plan):
        qkv = qkv.reshape(qkv.shape[0], (self.heads + 2 * self.kv_heads) * self.dim)
        q, k, v = torch.split(
            qkv, [self.heads * self.dim, self.kv_heads * self.dim, self.kv_heads * self.dim],
            dim=-1,
        )
        n = qkv.shape[0]
        q = q.reshape(n, self.heads, self.dim)
        k = k.reshape(n, self.kv_heads, self.dim)
        v = v.reshape(n, self.kv_heads, self.dim)
        return run_two_pass(plan, q, k, v)


class BertUqiTwoPassImpl(FMHAImplBase):
    def __init__(self, op, attn_inputs, schedule):
        self.fmha_impl = op
        self.attn_inputs = attn_inputs
        # ROCm tensors use the cuda device namespace. The current device is
        # selected by the engine, including when host-only metadata is supplied.
        device = schedule.perm.device if schedule.perm is not None else torch.device("cuda", torch.cuda.current_device())
        self.fmha_params = BertUqiParams(prepare_two_pass(schedule, device))
        self.uqi_perm = schedule.perm
        self.uqi_inv_perm = schedule.inv_perm

    def forward(self, qkv, kv_cache, layer_idx=0):
        return self.fmha_impl.forward(qkv, self.fmha_params.plan)

    @staticmethod
    def support(attn_configs, attn_inputs):
        return attn_inputs.is_prefill

    def support_cuda_graph(self):
        return False


# Fixed-buffer graph attention; request planning runs outside capture.


class BertUqiRocmEncoderGraphImpl(FMHAImplBase):
    def __init__(self, configs, attn_inputs, token_capacity, max_sequence_length=None):
        if configs.is_causal:
            raise ValueError("BERT user-segment graph requires noncausal attention")
        self.heads, self.kv_heads, self.dim = (
            configs.head_num,
            configs.kv_head_num,
            configs.size_per_head,
        )
        self.token_capacity = token_capacity
        self.request_capacity = attn_inputs.input_lengths.numel()
        self.fmha_params = BertUqiParams(None)
        self._host_layout = None
        self._host_metadata = None
        self._device_metadata = None
        self.uqi_perm = None
        self.uqi_inv_perm = None
        self.split_cu_seqlens = None
        self.full_cu_seqlens = None
        self.b_mask = None
        self.valid_mask = None
        self.aiter_max_seq_len = max_sequence_length or token_capacity
        self._prepare(attn_inputs, for_capture=True)

    def prepare_cuda_graph(self, attn_inputs):
        self._prepare(attn_inputs, for_capture=False)

    def support_cuda_graph(self):
        return True

    @staticmethod
    def support(configs, attn_inputs):
        return attn_inputs.is_prefill

    def _prepare(self, attn_inputs, for_capture):
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
        if for_capture:
            # Split startup tokens into legal requests; fixed padding slots
            # cover any remaining aggregate capacity.
            lengths = [
                min(limit, max(0, self.token_capacity - i * limit))
                for i in range(self.request_capacity)
            ]
            starts, b_lengths = [-1] * len(lengths), [0] * len(lengths)
        else:
            starts, b_lengths = attn_inputs.uqi_b_starts, attn_inputs.uqi_b_lens
            if (
                starts is None
                or b_lengths is None
                or starts.is_cuda
                or b_lengths.is_cuda
            ):
                raise ValueError("UQI graph requires host segment bounds")
            active = starts.numel()
            if (
                active > len(lengths)
                or b_lengths.numel() != active
                or any(lengths[active:])
            ):
                raise ValueError("invalid UQI graph segment metadata")
            starts = starts.tolist() + [-1] * (len(lengths) - active)
            b_lengths = b_lengths.tolist() + [0] * (len(lengths) - active)
        if self._host_layout is None:
            self._host_layout = BertUqiAiterGraphLayout(
                self.token_capacity, self.request_capacity, limit
            )
            self._host_metadata = torch.frombuffer(
                self._host_layout.buffer, dtype=torch.uint8
            )
            self._device_metadata = torch.empty_like(self._host_metadata, device="cuda")
            layout, device_metadata = self._host_layout, self._device_metadata
            self.uqi_perm = device_metadata[: layout.inv_perm_offset].view(torch.int64)
            self.uqi_inv_perm = device_metadata[
                layout.inv_perm_offset : layout.split_indptr_offset
            ].view(torch.int64)
            self.split_cu_seqlens = device_metadata[
                layout.split_indptr_offset : layout.full_indptr_offset
            ].view(torch.int32)
            self.full_cu_seqlens = device_metadata[
                layout.full_indptr_offset : layout.b_mask_offset
            ].view(torch.int32)
            self.b_mask = (
                device_metadata[layout.b_mask_offset : layout.valid_mask_offset]
                .view(torch.bool)
                .view(-1, 1)
            )
            self.valid_mask = (
                device_metadata[layout.valid_mask_offset :].view(torch.bool).view(-1, 1)
            )
        self._host_layout.update(lengths, starts, b_lengths)
        # One blocking upload replaces six temporary tensors/copies. Completing
        # the copy before return also makes reuse of the host byte buffer safe
        # for callers that prepare multiple layouts without an intervening replay.
        self._device_metadata.copy_(self._host_metadata)

    def forward(self, qkv, kv_cache, layer_idx=0):
        if kv_cache is not None:
            raise ValueError("UQI encoder graph cannot use KV cache")
        from aiter import flash_attn_varlen_func

        q, k, v = torch.split(
            qkv.reshape(self.token_capacity, -1),
            [self.heads * self.dim, self.kv_heads * self.dim, self.kv_heads * self.dim],
            dim=-1,
        )
        q = q.reshape(self.token_capacity, self.heads, self.dim)
        k = k.reshape(self.token_capacity, self.kv_heads, self.dim)
        v = v.reshape(self.token_capacity, self.kv_heads, self.dim)

        def run_attention(cu_seqlens):
            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                self.aiter_max_seq_len,
                self.aiter_max_seq_len,
                dropout_p=0.0,
                causal=False,
            ).reshape(self.token_capacity, self.heads * self.dim)

        split_output = run_attention(self.split_cu_seqlens)
        full_output = run_attention(self.full_cu_seqlens)
        result = torch.where(self.b_mask, full_output, split_output)
        return torch.where(self.valid_mask, result, 0.0)
