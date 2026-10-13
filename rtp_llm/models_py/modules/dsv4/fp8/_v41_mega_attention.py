"""FlashMLA mega attention over RTP's existing V4.1 KV pools and indices."""

import logging
from copy import copy
from typing import NamedTuple

import torch

from rtp_llm.model_loader.weight_memory_saver import feature_weights_region
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range


class MegaAttentionOutput(NamedTuple):
    data: torch.Tensor
    scales: torch.Tensor


class MegaAttention:
    @classmethod
    def create(cls, attn):
        from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear

        supported = (
            type(attn.wq_b) is V41MXFP8Linear
            and attn.wq_b.weight.is_cuda
            and attn.head_dim == 512
            and attn.rope_head_dim == 64
            and attn.n_heads in (64, 128)
            and attn.n_heads == attn.n_groups * 8
            and torch.cuda.get_device_capability(attn.wq_b.weight.device)[0] == 10
        )
        if not supported:
            raise RuntimeError(
                "V4.1 mega attention requires SM10x, MXFP8 Q weights, "
                "64/128 local heads, head_dim=512, rope_dim=64 "
                "and 8 heads per group"
            )
        import flash_mla.cuda as native
        from flash_mla import fused_norm_rope_attn_rope_cast

        if not all(
            hasattr(native, name)
            for name in (
                "fused_norm_rope_attn_rope_cast_fwd",
                "fused_norm_rope_attn_rope_cast_decode",
                "permute_q_b_proj",
                "permute_wv_proj",
            )
        ):
            raise ImportError("FlashMLA was built without mega attention")

        result = cls(attn, fused_norm_rope_attn_rope_cast)
        logging.info(
            "[dsv41] mega attention enabled: layer=%s heads=%s groups=%s",
            attn.layer_id,
            attn.n_heads,
            attn.n_groups,
        )
        return result

    def __init__(self, attn, kernels):
        self.kernels = kernels
        # Keep checkpoint/source tensors in their original layout. Only the
        # resident Q linear gets a copy; weight reload can restore it normally.
        attn.wq_b = copy(attn.wq_b)
        with feature_weights_region():
            attn.wq_b.weight, attn.wq_b.weight_scales = kernels.permute_q_b_proj(
                (attn.wq_b.weight, attn.wq_b.weight_scales), attn.n_heads, 512
            )
            self.weight, self.weight_scales = kernels.permute_wv_proj(
                (attn._wo_a_stk_w, attn._wo_a_stk_s), 8, 512
            )
        self.groups = attn.n_groups
        self.rank = attn.o_lora_rank
        self.refresh_rope(attn)

    def refresh_rope(self, attn):
        cache = attn._shared_attention.setdefault("mega_attention_rope", {})
        key = id(attn.freqs_cis)
        if key not in cache:
            with feature_weights_region():
                cache[key] = torch.cat(
                    (attn.freqs_cis.real, attn.freqs_cis.imag), dim=-1
                ).contiguous()
        self.cos_sin_cache = cache[key]

    def reload(self, attn):
        # Parent reload has restored the standard Q weights. Copy the new
        # permutations into the existing addresses for captured CUDA graphs.
        q_weight, q_scales = self.kernels.permute_q_b_proj(
            (attn.wq_b.weight, attn.wq_b.weight_scales), attn.n_heads, 512
        )
        attn.wq_b.weight.copy_(q_weight)
        attn.wq_b.weight_scales.copy_(q_scales)
        weight, scales = self.kernels.permute_wv_proj(
            (attn._wo_a_stk_w, attn._wo_a_stk_s), 8, 512
        )
        self.weight.copy_(weight)
        self.weight_scales.copy_(scales)
        self.cos_sin_cache.copy_(
            torch.cat((attn.freqs_cis.real, attn.freqs_cis.imag), dim=-1)
        )

    def _arguments(self, attn, q, positions):
        return dict(
            enable_q_norm=False,
            rms_norm_eps=0.0,
            token_positions=positions.reshape(-1).to(dtype=torch.int32).contiguous(),
            is_rope_neox_style=False,
            rope_dim=64,
            cos_sin_cache=self.cos_sin_cache,
            n_wv_group=self.groups,
            num_per_channels=32,
            use_tma_aligned_col_major_sf=True,
            round_sf=True,
            use_packed_ue8m0=True,
            q=q.reshape(-1, attn.n_heads, 512),
            sm_scale=attn.softmax_scale,
            attn_sink=attn.attn_sink,
        )

    def prefill(self, attn, q, positions, kv, indices, lengths):
        with record_function_range("dsv41.attn.mega.prefill"):
            data, scales, _, _ = self.kernels.prefill(
                **self._arguments(attn, q, positions),
                kv=kv,
                indices=indices,
                topk_length=lengths,
            )
        return MegaAttentionOutput(data, scales)

    def decode(
        self,
        attn,
        q,
        positions,
        swa_pool,
        swa_indices,
        global_pool=None,
        global_indices=None,
    ):
        with record_function_range("dsv41.attn.mega.decode"):
            data, scales, _ = self.kernels.decode(
                **self._arguments(attn, q, positions),
                k_cache=swa_pool.unsqueeze(2),
                indices_in_kvcache=swa_indices.reshape(q.shape[0] * q.shape[1], -1),
                extra_k_cache=None if global_pool is None else global_pool.unsqueeze(2),
                extra_indices_in_kvcache=global_indices,
            )
        return MegaAttentionOutput(data, scales)

    def project(self, attn, pair, out=None):
        import deep_gemm

        projected = torch.empty(
            pair.data.shape[0],
            self.groups,
            self.rank,
            dtype=torch.bfloat16,
            device=pair.data.device,
        )
        with record_function_range("dsv41.attn.mega.wo_a"):
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                pair,
                (self.weight, self.weight_scales),
                projected,
                recipe=(1, 1, 32),
            )
        return attn.wo_b(projected.flatten(1), out=out)
