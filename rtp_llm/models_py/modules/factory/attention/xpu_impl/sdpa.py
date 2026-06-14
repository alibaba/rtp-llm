"""XPU FMHA implementation using PyTorch scaled_dot_product_attention.

This provides a portable attention backend for Intel XPU (GPU) devices.
It delegates to PyTorch's F.scaled_dot_product_attention, which uses
oneDNN/oneMKL kernels on Intel GPUs.

Includes RoPE (Rotary Position Embedding) via shared helper from
vllm_flash_attn module.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.xpu_impl.vllm_flash_attn import (
    _apply_rope, _need_rope, _split_qkv_and_rope, _write_to_paged_cache,
    _get_flash_attn_varlen, _UNSUPPORTED_ROPE_STYLES,
)
from rtp_llm.ops import RopeStyle

logger = logging.getLogger(__name__)


class XpuSdpaPrefillImpl(FMHAImplBase):
    """Prefill attention using PyTorch SDPA on Intel XPU with RoPE."""

    def __init__(self, attn_configs, attn_inputs, parallelism_config=None):
        self.fmha_params = None
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.rope_config = attn_configs.rope_config
        self.need_rope = _need_rope(attn_configs)
        # PD disaggregation: notify cache_store after writing KV so the decode
        # role can fetch the prefill output via P2P RPC.  Same as VllmPrefill.
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @staticmethod
    def support(attn_configs, attn_inputs):
        if not attn_inputs.is_prefill:
            return False
        rope_style = getattr(getattr(attn_configs, "rope_config", None), "style", RopeStyle.No)
        if rope_style in _UNSUPPORTED_ROPE_STYLES:
            return False
        # Prefix-cache prefill (prefix_lengths > 0) requires reading cached K/V
        # and extending the attention context.  Not yet implemented on XPU:
        # fall through to a different impl or fail-fast at runtime.
        # prefix_lengths > 0 means a prefix-cache hit: the impl must load
        # cached K/V and extend the attention context, which is not yet
        # implemented here.  Return False so the factory can select an impl
        # that handles it (or fail-fast at a higher level).
        prefix_lengths = getattr(attn_inputs, 'prefix_lengths', None)
        if prefix_lengths is not None and prefix_lengths.numel() > 0:
            if (prefix_lengths > 0).any():
                return False
        return True

    def forward(self, qkv, kv_cache=None, layer_idx=0):
        total_tokens = qkv.shape[0]
        q, k, v = _split_qkv_and_rope(
            qkv, self.attn_inputs, self.num_heads, self.num_kv_heads,
            self.head_dim, self.rope_config, self.need_rope,
        )

        # Write K,V to paged cache for future decode steps.
        # IMPORTANT: when prefix-cache is hit (prefix_lengths[i] > 0), the new
        # tokens must be written at offset = prefix_lengths[i], not 0, or they
        # will clobber the cached prefix blocks at the wrong positions.
        if kv_cache is not None:
            block_ids_all = self.attn_inputs.kv_cache_block_id_device
            if block_ids_all is None:
                block_ids_all = self.attn_inputs.kv_cache_block_id_host
            if block_ids_all is not None and block_ids_all.numel() > 0:
                input_lengths = self.attn_inputs.input_lengths
                prefix_lengths = getattr(self.attn_inputs, 'prefix_lengths', None)
                pl_cpu = prefix_lengths.cpu() if (prefix_lengths is not None and prefix_lengths.numel() > 0) else None
                if input_lengths is not None and input_lengths.numel() > 1:
                    in_cpu = input_lengths.cpu()
                    offsets = torch.cat([torch.zeros(1, dtype=torch.int32), in_cpu.cumsum(0)])
                    for req_idx in range(in_cpu.numel()):
                        start = int(offsets[req_idx])
                        end = int(offsets[req_idx + 1])
                        bids = block_ids_all[req_idx].cpu()
                        start_pos = int(pl_cpu[req_idx]) if pl_cpu is not None and pl_cpu.numel() > req_idx else 0
                        _write_to_paged_cache(
                            k[start:end], v[start:end], kv_cache, bids, start_pos,
                            self.num_kv_heads, self.head_dim,
                        )
                else:
                    bids = block_ids_all[0].cpu()
                    start_pos = int(pl_cpu[0]) if pl_cpu is not None and pl_cpu.numel() > 0 else 0
                    _write_to_paged_cache(k, v, kv_cache, bids, start_pos,
                                          self.num_kv_heads, self.head_dim)

            # PD disaggregation: notify cache_store the KV blocks are ready.
            common.apply_write_cache_store(
                self.write_cache_store_impl, self.attn_inputs, kv_cache
            )

        # Use flash_attn_varlen for batched variable-length attention
        flash_attn_varlen = _get_flash_attn_varlen()
        cu_seqlens = self.attn_inputs.cu_seqlens
        if cu_seqlens is None or cu_seqlens.numel() <= 1:
            cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device=qkv.device)
        else:
            cu_seqlens = cu_seqlens.to(device=qkv.device, dtype=torch.int32)
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        output = flash_attn_varlen(
            q.contiguous(), k.contiguous(), v.contiguous(),
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=True,
        )
        return output.reshape(total_tokens, -1)


class XpuSdpaDecodeImpl(FMHAImplBase):
    """Decode attention using PyTorch SDPA on Intel XPU with RoPE.

    Uses paged LayerKVCache for KV history across decode steps.
    Only supports batch_size==1: uses seq_lengths[0]/block_ids[0] throughout.
    """

    def __init__(self, attn_configs, attn_inputs, parallelism_config=None):
        self.fmha_params = None
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.rope_config = attn_configs.rope_config
        self.need_rope = _need_rope(attn_configs)

    @staticmethod
    def support(attn_configs, attn_inputs):
        if attn_inputs.is_prefill:
            return False
        rope_style = getattr(getattr(attn_configs, "rope_config", None), "style", RopeStyle.No)
        if rope_style in _UNSUPPORTED_ROPE_STYLES:
            return False
        # Only supports single-request decode; multi-request needs per-row
        # sequence_lengths/block_ids which are not yet wired up here.
        batch_size = getattr(attn_inputs, 'batch_size', None)
        if batch_size is None:
            seq_lens = getattr(attn_inputs, 'sequence_lengths', None)
            if seq_lens is not None and seq_lens.numel() > 1:
                return False
        elif batch_size > 1:
            return False
        return True

    def forward(self, qkv, kv_cache=None, layer_idx=0):
        from rtp_llm.models_py.modules.factory.attention.xpu_impl.vllm_flash_attn import (
            _split_qkv_and_rope, _write_to_paged_cache, _read_from_paged_cache,
        )

        total_tokens = qkv.shape[0]

        # Compute position_ids from sequence_lengths
        if self.need_rope:
            seq_lengths = self.attn_inputs.sequence_lengths
            if seq_lengths is not None and seq_lengths.numel() > 0:
                start_pos = int(seq_lengths[0].item())
            else:
                start_pos = 0
            self.attn_inputs.position_ids = torch.arange(
                start_pos, start_pos + total_tokens,
                dtype=torch.long, device=qkv.device,
            )

        q, k_new, v_new = _split_qkv_and_rope(
            qkv, self.attn_inputs, self.num_heads, self.num_kv_heads,
            self.head_dim, self.rope_config, self.need_rope,
        )

        # Use paged KV cache
        if kv_cache is not None:
            block_ids_all = self.attn_inputs.kv_cache_block_id_device
            if block_ids_all is None:
                block_ids_all = self.attn_inputs.kv_cache_block_id_host
            if block_ids_all is not None and block_ids_all.numel() > 0:
                seq_lengths = self.attn_inputs.sequence_lengths
                start_pos = int(seq_lengths[0].item()) if seq_lengths is not None and seq_lengths.numel() > 0 else 0
                bids = block_ids_all[0].cpu()
                _write_to_paged_cache(k_new, v_new, kv_cache, bids, start_pos,
                                      self.num_kv_heads, self.head_dim)
                total_len = start_pos + total_tokens
                k, v = _read_from_paged_cache(kv_cache, bids, total_len,
                                              self.num_kv_heads, self.head_dim)
            else:
                k, v = k_new, v_new
        else:
            k, v = k_new, v_new

        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        output = output.squeeze(0).transpose(0, 1)

        return output.reshape(total_tokens, -1)
