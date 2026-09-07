"""Single-pass BERT profile attention using FlashInfer's public mask API."""

import math
from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPrefillAttnOp,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs


class BertUqiAttention(FMHAImplBase):
    """One persistent wrapper; plan once per batch, run once per layer."""

    def __init__(self, config: AttentionConfigs) -> None:
        if config.is_causal or config.need_rope_kv_cache:
            raise ValueError("BERT UQI requires non-causal attention without a KV cache")
        if config.dtype not in (torch.float16, torch.bfloat16) or config.q_scaling <= 0:
            raise ValueError("BERT UQI requires FP16/BF16 and positive q_scaling")
        self.op = PyFlashinferPrefillAttnOp(config)
        self.heads = config.head_num
        self.kv_heads = config.kv_head_num
        self.head_dim = config.size_per_head
        self.scale = (
            config.softmax_extra_scale / config.q_scaling / math.sqrt(self.head_dim)
        )
        self.fmha_params = None

    def prepare(self, inputs: PyAttentionInputs, device: torch.device) -> None:
        # custom_mask's segment_packbits requires device indptrs in FlashInfer.
        # Build the mask on CPU once; no token scans, permutations or GPU mask
        # construction kernels are needed in the model's forward path.
        indptr = inputs.cu_seqlens_device[: inputs.input_lengths.size(0) + 1]
        self.op.prefill_wrapper.plan(
            indptr,
            indptr,
            self.heads,
            self.kv_heads,
            self.head_dim,
            head_dim_vo=self.head_dim,
            custom_mask=inputs.bert_uqi_mask.to(device, non_blocking=True),
            causal=False,
            sm_scale=self.scale,
            q_data_type=self.op.dtype,
            kv_data_type=self.op.dtype,
            o_data_type=self.op.dtype,
        )

    def forward(
        self, qkv: torch.Tensor, kv_cache: Optional[LayerKVCache], layer_idx: int = 0
    ) -> torch.Tensor:
        if kv_cache is not None:
            raise ValueError("BERT UQI does not accept a KV cache")
        q, k, v = qkv.reshape(qkv.shape[0], -1, self.head_dim).split(
            (self.heads, self.kv_heads, self.kv_heads), dim=1
        )
        return self.op.prefill_wrapper.run(q, k, v)

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        return False  # Selected only by BertModel, never by the global factory.
