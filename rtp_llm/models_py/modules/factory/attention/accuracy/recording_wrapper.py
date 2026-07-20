from typing import Any, Dict, Optional, Tuple

import torch

from rtp_llm.models_py.modules.factory.attention.accuracy.attention_record import (
    AttentionLayerRecord,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs


class RecordingWrapper(FMHAImplBase):
    def __init__(
        self,
        inner_impl: FMHAImplBase,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ) -> None:
        self.inner = inner_impl
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.layer_records: Dict[int, AttentionLayerRecord] = {}

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        q, k, v = self._split(qkv)
        q = q.detach().clone()
        k = k.detach().clone()
        v = v.detach().clone()
        out = self.inner.forward(qkv, kv_cache, layer_idx)
        # impls return either [T, Hq, D] (paged) or [T, Hq*D] (FMHAv2/XQA)
        # the record copy is always flattened to packed [T, Hq*D]
        # But the model still gets inner's raw out
        out_rec = out.reshape(out.shape[0], -1)
        self.layer_records[layer_idx] = AttentionLayerRecord(
            layer_idx=layer_idx,
            q=q,
            k=k,
            v=v,
            output=out_rec.detach().clone(),
        )
        return out

    def _split(
        self, qkv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.attn_configs.head_num
        kv_h = self.attn_configs.kv_head_num
        d = self.attn_configs.size_per_head
        q, k, v = torch.split(qkv, [h * d, kv_h * d, kv_h * d], dim=-1)
        return q.reshape(-1, h, d), k.reshape(-1, kv_h, d), v.reshape(-1, kv_h, d)

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def support_cuda_graph(self) -> bool:
        return False

    @property
    def fmha_params(self) -> Any:
        return self.inner.fmha_params
