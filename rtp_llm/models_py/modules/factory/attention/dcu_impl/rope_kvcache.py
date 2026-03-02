import logging
from typing import Optional, Any, List

import torch
import lightop
from lightop import op as lightop_ops

from rtp_llm.ops import PyAttentionInputs, FMHAType, KVCache, AttentionConfigs


logger = logging.getLogger(__name__)


class FusedRopeKVCachePrefillOp:
    def __init__(self, attn_configs: AttentionConfigs):
        self.attn_configs = attn_configs
        self.attn_configs.rope_config.max_pos = 40960
        self.cos_sin_cache = self._compute_cos_sin_cache()

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.attn_configs.rope_config.dim, 2, dtype=self.attn_configs.dtype) / self.attn_configs.rope_config.dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.attn_configs.rope_config.base)
        t = torch.arange(self.attn_configs.rope_config.max_pos, dtype=self.attn_configs.dtype)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        cache = cache.cuda()
        return cache

    def prepare(self, attn_inputs: PyAttentionInputs):
        block_size = self.attn_configs.tokens_per_block
        kv_cache_block_id_host = None
        if attn_inputs.kv_cache_block_id_host.numel() > 0:
            kv_cache_block_id_host = attn_inputs.kv_cache_block_id_host
        cu_seqlens = attn_inputs.cu_seqlens

        # 构造positions
        positions = torch.ones(cu_seqlens[-1].item(), dtype=torch.long, device='cpu')
        first_indices = cu_seqlens[:-1]
        reset_values = torch.cat([torch.tensor([0]), attn_inputs.input_lengths[:-1]])
        positions[first_indices] = 1 - reset_values 
        positions = positions.cumsum(0) - 1
        
        # 构造slot_mapping
        block_indices = positions // block_size
        seq_ids = torch.repeat_interleave(
            torch.arange(len(attn_inputs.input_lengths)), 
            attn_inputs.input_lengths,
        )
        physical_block_ids = kv_cache_block_id_host[0][seq_ids, block_indices]
        block_offsets = positions % block_size
        slot_mapping = physical_block_ids * block_size + block_offsets

        # 移动到GPU
        positions = positions.to('cuda')
        slot_mapping = slot_mapping.to('cuda')

        #logging.info(f"DtkRopeKVCachePrefillOp prepare: \n{attn_inputs.kv_cache_block_id_host=}\n{attn_inputs.kv_cache_block_id_device=}\n{attn_inputs.prefix_lengths=}\n{attn_inputs.sequence_lengths=}\n{attn_inputs.input_lengths=}\n{cu_seqlens=}\n{positions=}\n{slot_mapping=}")
        # 4. 准备注意力参数字典
        attn_params = {
            'head_num': self.attn_configs.head_num,
            'kv_head_num': self.attn_configs.kv_head_num,
            'size_per_head': self.attn_configs.size_per_head,
            'tokens_per_block': self.attn_configs.tokens_per_block,
            'positions': positions,
            'slot_mapping': slot_mapping,
            'k_scale': torch.tensor(1.0, device='cuda'),
            'v_scale': torch.tensor(1.0, device='cuda'),
        }
        
        return attn_params
    
    def forward(self, qkv: torch.Tensor, kv_cache: Optional[KVCache], params: Optional[Any]) -> torch.Tensor:
        q, k, v = qkv.split([params["head_num"]*params["size_per_head"], 
                             params["kv_head_num"]*params["size_per_head"], 
                             params["kv_head_num"]*params["size_per_head"]], dim=-1)
        # print(f"{params['positions'].shape=}, {q.shape=}, {k.shape=}, {self.cos_sin_cache.shape=}, {self.attn_configs.rope_config.is_neox_style=}")
        lightop_ops.rotary_embedding_fuse(params["positions"], q, k, params["size_per_head"], self.cos_sin_cache, self.attn_configs.rope_config.is_neox_style)
        k_reshaped = k.reshape(-1, params["kv_head_num"], params["size_per_head"])
        v_reshaped = v.reshape(-1, params["kv_head_num"], params["size_per_head"])
        block_num = kv_cache.kv_cache_base.shape[1]
        k_cache = kv_cache.kv_cache_base[0].reshape(block_num, params["kv_head_num"], params["tokens_per_block"], params["size_per_head"])
        v_cache = kv_cache.kv_cache_base[1].reshape(block_num, params["kv_head_num"], params["size_per_head"], params["tokens_per_block"])
        
        # print(f"prefill before cache: {k_reshaped.shape=}, {v_reshaped.shape=}, k:{k_reshaped[0,0,:10].detach().cpu().tolist()}, v:{v_reshaped[0,0,:10].detach().cpu().tolist()}, {k_cache.shape=}, {k_cache.stride()=}, {v_cache.shape=}, {v_cache.stride()=}, {k_cache.is_contiguous()=}, {v_cache.is_contiguous()=}, {params['slot_mapping']=}")
         
        lightop.reshape_and_cache_cuda(k_reshaped, 
                                v_reshaped,
                                k_cache,
                                v_cache,
                                params["slot_mapping"],
                                'auto',
                                params["k_scale"],
                                params["v_scale"],
                        )
        # print(f"prefill after cache: k={k_cache[1,0,:,0].detach().cpu().tolist()}, v={v_cache[1,0,0,:].detach().cpu().tolist()}")
        return q, k, v


class FusedRopeKVCacheDecodeOp:
    def __init__(self, attn_configs: AttentionConfigs):
        self.attn_configs = attn_configs
        self.attn_configs.rope_config.max_pos = 40960
        self.cos_sin_cache = self._compute_cos_sin_cache()

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute the inverse frequency. Borrowd from vllm"""
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.attn_configs.rope_config.dim, 2, dtype=self.attn_configs.dtype) / self.attn_configs.rope_config.dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.attn_configs.rope_config.base)
        t = torch.arange(self.attn_configs.rope_config.max_pos, dtype=self.attn_configs.dtype)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        cache = cache.cuda()
        return cache

    def prepare(self, attn_inputs: PyAttentionInputs):
        block_size = self.attn_configs.tokens_per_block
        kv_cache_block_id_host = None
        if attn_inputs.kv_cache_block_id_host.numel() > 0:
            kv_cache_block_id_host = attn_inputs.kv_cache_block_id_host

        # 构造positions
        positions = attn_inputs.sequence_lengths.to(dtype=torch.long, copy=True)
        
        # 构造slot_mapping
        # seq_lens = attn_inputs.sequence_lengths
        block_indices = positions // block_size
        seq_ids = torch.repeat_interleave(
            torch.arange(len(attn_inputs.input_lengths)), 
            1,
        )
        physical_block_ids = kv_cache_block_id_host[0][seq_ids, block_indices]
        block_offsets = positions % block_size
        slot_mapping = physical_block_ids * block_size + block_offsets

        # 移动到GPU
        positions = positions.to('cuda')
        slot_mapping = slot_mapping.to('cuda')
        
        attn_params = {
            'k_scale': torch.tensor(1.0, device='cuda'),
            'v_scale': torch.tensor(1.0, device='cuda'),
            'head_num': self.attn_configs.head_num,
            'kv_head_num': self.attn_configs.kv_head_num,
            'size_per_head': self.attn_configs.size_per_head,
            'tokens_per_block': self.attn_configs.tokens_per_block,
            'positions': positions,
            'slot_mapping': slot_mapping
        }
        
        return attn_params

    def forward(self, qkv: torch.Tensor, kv_cache: Optional[KVCache], params: Optional[Any]) -> torch.Tensor:
        q, k, v = qkv.split([params["head_num"]*params["size_per_head"], 
                             params["kv_head_num"]*params["size_per_head"], 
                             params["kv_head_num"]*params["size_per_head"]], dim=-1)
        lightop_ops.rotary_embedding_fuse(params["positions"], q, k, params["size_per_head"], self.cos_sin_cache, self.attn_configs.rope_config.is_neox_style)
        #print(f"after rope: q: {q[0, :10].detach().cpu().tolist()}, k: {k[0, :10].detach().cpu().tolist()}")
        k_reshaped = k.reshape(-1, params["kv_head_num"], params["size_per_head"])
        v_reshaped = v.reshape(-1, params["kv_head_num"], params["size_per_head"])
        block_num = kv_cache.kv_cache_base.shape[1]
        k_cache = kv_cache.kv_cache_base[0].reshape(block_num, params["kv_head_num"], params["tokens_per_block"], params["size_per_head"])
        v_cache = kv_cache.kv_cache_base[1].reshape(block_num, params["kv_head_num"], params["size_per_head"], params["tokens_per_block"])
        #print(f"decode before cache: {k_reshaped.shape=}, {v_reshaped.shape=}, k:{k_reshaped[0,0,:10].detach().cpu().tolist()}, v:{v_reshaped[0,0,:10].detach().cpu().tolist()}, {k_cache.shape=}, {k_cache.stride()=}, {v_cache.shape=}, {v_cache.stride()=}, {k_cache.is_contiguous()=}, {v_cache.is_contiguous()=}, {params['slot_mapping']=}")
        lightop.reshape_and_cache_cuda(k_reshaped, 
                                   v_reshaped, 
                                   k_cache,
                                   v_cache,
                                   params["slot_mapping"],
                                   'auto',
                                   params["k_scale"],
                                   params["v_scale"])
        #print(f"after cache: k={k_cache[1,0,:,0].detach().cpu().tolist()}, v={v_cache[1,0,0,:].detach().cpu().tolist()}")
        return q
