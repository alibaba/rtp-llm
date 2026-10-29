from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.attention.accuracy.golden_kv_history import (
    GoldenKVHistoryView,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, rtp_llm_ops
from rtp_llm.ops.fused_rope_kvcache_op import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQOut,
)


class GoldenSDPAImpl(FMHAImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        history: GoldenKVHistoryView,
    ) -> None:
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.history = history
        self.head_num = attn_configs.head_num
        self.kv_head_num = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scale = self.head_dim**-0.5
        self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id,
            attn_configs.kernel_tokens_per_block,
            False,
        )
        self.cache_writer = GoldenCacheWriter(
            attn_configs, attn_inputs, self.fmha_params
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        q, k, v = self._split_qkv(qkv)
        self.cache_writer.write(
            qkv, k, v, kv_cache
        )  # quantized write, candidate-only; golden never reads
        assert self.kv_head_num > 0 and self.head_num % self.kv_head_num == 0
        attention_inputs = self.attn_inputs
        outputs = []
        if attention_inputs.is_prefill:
            cu = attention_inputs.cu_seqlens.tolist()
            prefix_lengths = attention_inputs.prefix_lengths.tolist()
            input_lengths = attention_inputs.input_lengths.tolist()
            assert len(prefix_lengths) == len(self.history)
            for batch_idx in range(len(prefix_lengths)):
                s, e = cu[batch_idx], cu[batch_idx + 1]
                prefix_len = prefix_lengths[batch_idx]
                assert e - s == input_lengths[batch_idx]
                seg_k, seg_v = k[s:e], v[s:e]
                if prefix_len > 0:  # prefix: prepend this sequence's golden history
                    history_k, history_v = self.history.get_prefix(batch_idx, layer_idx)
                    seg_k = torch.cat([history_k[:prefix_len], seg_k], 0)
                    seg_v = torch.cat([history_v[:prefix_len], seg_v], 0)
                else:  # plain: store this [P+C] segment as golden history for later decode
                    self.history.store_plain(batch_idx, layer_idx, seg_k, seg_v)
                outputs.append(
                    self._sdpa(q[s:e], seg_k, seg_v, prefix_len=prefix_len, causal=True)
                )
        else:  # decode: one row per sequence
            sequence_lengths = attention_inputs.sequence_lengths.tolist()
            assert len(sequence_lengths) == len(self.history)
            for batch_idx in range(len(sequence_lengths)):
                history_k, history_v = self.history.get_decode_history(
                    batch_idx, layer_idx
                )
                assert history_k.shape[0] == sequence_lengths[batch_idx]
                seg_k = torch.cat([history_k, k[batch_idx : batch_idx + 1]], 0)
                seg_v = torch.cat([history_v, v[batch_idx : batch_idx + 1]], 0)
                self.history.append_decode(
                    batch_idx,
                    layer_idx,
                    k[batch_idx : batch_idx + 1],
                    v[batch_idx : batch_idx + 1],
                )
                outputs.append(
                    self._sdpa(
                        q[batch_idx : batch_idx + 1],
                        seg_k,
                        seg_v,
                        prefix_len=history_k.shape[0],
                        causal=False,
                    )
                )
        return torch.cat(outputs, 0).reshape(qkv.shape[0], -1)

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        prefix_len: int,
        causal: bool,
    ) -> torch.Tensor:
        # q [Lq,Hq,D], k/v [Lk,Hkv,D] -> [1,H,L,D]
        # NOTE: cuda12.9 torch=2.8.0; torch<2.13 SDPA returns WRONG results for >64K token inputs,
        # so each segment's kv seqlen must stay <= 65536
        assert (
            k.shape[0] <= 65536
        ), "golden SDPA requires kv seqlen <= 64K on torch<2.13"
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        attn_mask = None
        if causal:  # query i may see kv [0, prefix_len + i]
            Lq, Lk = q.shape[-2], k.shape[-2]
            m = torch.zeros(Lq, Lk, dtype=torch.bool, device=q.device)
            m[:, prefix_len:] = torch.triu(
                torch.ones(Lq, Lq, dtype=torch.bool, device=q.device), 1
            )
            attn_mask = ~m  # True = visible
        gqa = self.head_num != self.kv_head_num
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=self.scale, enable_gqa=gqa
        )
        return out.squeeze(0).transpose(0, 1).contiguous()  # [Lq, Hq, D]

    def _split_qkv(
        self, qkv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, kv_h, d = self.head_num, self.kv_head_num, self.head_dim
        q, k, v = torch.split(qkv, [h * d, kv_h * d, kv_h * d], dim=-1)
        return q.reshape(-1, h, d), k.reshape(-1, kv_h, d), v.reshape(-1, kv_h, d)

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        return True

    def support_cuda_graph(self) -> bool:
        return False


class GoldenCacheWriter:
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        fmha_params: Any,
    ) -> None:
        self.kv_dtype = attn_configs.kv_cache_dtype
        if self.kv_dtype == KvCacheDataType.BASE:
            self.op = KVCacheWriteOp(
                attn_configs.kv_head_num,
                attn_configs.size_per_head,
                attn_configs.kernel_tokens_per_block,
            )
            self.op.set_params(fmha_params)
            self.mode = "base"
        elif self.kv_dtype == KvCacheDataType.FP8:
            self.op = (
                FusedRopeKVCachePrefillOpQOut(attn_configs)
                if attn_inputs.is_prefill
                else FusedRopeKVCacheDecodeOp(attn_configs)
            )
            self.params = self.op.prepare(attn_inputs)
            self.mode = "fp8"
        else:
            raise NotImplementedError(
                f"GoldenCacheWriter: unsupported kv_cache_dtype={self.kv_dtype}; "
                f"only BASE and FP8 are supported"
            )

    def write(
        self,
        qkv: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> None:
        if kv_cache is None:
            raise RuntimeError(
                "GoldenCacheWriter.write: kv_cache is None in accuracy check"
            )
        if self.mode == "base":
            self.op.forward(k, v, kv_cache)
        else:
            self.op.forward(qkv, kv_cache, self.params)
