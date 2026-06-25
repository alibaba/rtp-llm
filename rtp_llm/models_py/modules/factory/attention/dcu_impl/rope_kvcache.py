import logging
from typing import Optional, Any, List, Tuple

import torch
import lightop
from lightop import op as lightop_ops

from rtp_llm.ops import PyAttentionInputs, FMHAType, KVCache, AttentionConfigs


logger = logging.getLogger(__name__)


# DCU re-creates the FMHA impl per forward (see DcuDecodeImpl/DcuPrefillImpl in attention.py),
# so the rope cos/sin table — a [max_pos, dim] CUDA tensor — would be re-allocated on every
# request and leak under PyTorch's caching allocator until OOM. Cache by shape/dtype params.
_COS_SIN_CACHE: dict[Tuple[int, int, float, torch.dtype], torch.Tensor] = {}


def _get_or_build_cos_sin_cache(
    dim: int, max_pos: int, base: float, dtype: torch.dtype
) -> torch.Tensor:
    key = (dim, max_pos, base, dtype)
    cache = _COS_SIN_CACHE.get(key)
    if cache is not None:
        return cache
    # Build the position table in fp32, never in the activation dtype. bf16 has a
    # 7-bit mantissa and cannot represent integers past 256 exactly (step becomes
    # 2 at 256, 4 at 512, ...), so `t = arange(max_pos, dtype=bf16)` aliases
    # adjacent positions and RoPE can no longer distinguish them -- attention then
    # degenerates into repetition a few hundred tokens in. Compute t / inv_freq /
    # freqs in fp32; only cast the final cos/sin (range [-1, 1], lossless in bf16)
    # to the kernel dtype.
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype).cuda()
    _COS_SIN_CACHE[key] = cache
    return cache


class FusedRopeKVCachePrefillOp:
    def __init__(self, attn_configs: AttentionConfigs):
        self.attn_configs = attn_configs
        self.attn_configs.rope_config.max_pos = 40960
        #self.cos_sin_cache = self._compute_cos_sin_cache()
        self.cos_sin_cache = _get_or_build_cos_sin_cache(
            self.attn_configs.rope_config.dim,
            self.attn_configs.rope_config.max_pos,
            self.attn_configs.rope_config.base,
            self.attn_configs.dtype,
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute the inverse frequency (fp32; see _get_or_build_cos_sin_cache)."""
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.attn_configs.rope_config.dim, 2, dtype=torch.float32) / self.attn_configs.rope_config.dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache (fp32 math, cast result to act dtype)."""
        inv_freq = self._compute_inv_freq(self.attn_configs.rope_config.base)
        t = torch.arange(self.attn_configs.rope_config.max_pos, dtype=torch.float32)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        cache = cache.to(self.attn_configs.dtype).cuda()
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
        #self.cos_sin_cache = self._compute_cos_sin_cache()
        self.cos_sin_cache = _get_or_build_cos_sin_cache(
            self.attn_configs.rope_config.dim,
            self.attn_configs.rope_config.max_pos,
            self.attn_configs.rope_config.base,
            self.attn_configs.dtype,
        )
        # Persistent tensors for cuda_graph replay; allocated on first prepare().
        self._positions: Optional[torch.Tensor] = None
        self._slot_mapping: Optional[torch.Tensor] = None
        self._k_scale: Optional[torch.Tensor] = None
        self._v_scale: Optional[torch.Tensor] = None

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        # fp32: integer positions/angles must not be quantized in bf16 (bf16 cannot
        # represent integers > 256 exactly, corrupting RoPE past a few hundred tokens).
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.attn_configs.rope_config.dim, 2, dtype=torch.float32) / self.attn_configs.rope_config.dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.attn_configs.rope_config.base)
        t = torch.arange(self.attn_configs.rope_config.max_pos, dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # Build in fp32, store as the model dtype (cos/sin in [-1,1] are lossless in bf16).
        cache = torch.cat((cos, sin), dim=-1).to(self.attn_configs.dtype)
        cache = cache.cuda()
        return cache

    def _compute_positions_and_slots(self, attn_inputs: PyAttentionInputs):
        block_size = self.attn_configs.tokens_per_block
        kv_cache_block_id_host = attn_inputs.kv_cache_block_id_host

        positions = attn_inputs.sequence_lengths.to(dtype=torch.long, copy=True)
        block_indices = positions // block_size
        seq_ids = torch.repeat_interleave(
            torch.arange(len(attn_inputs.input_lengths)),
            1,
        )
        physical_block_ids = kv_cache_block_id_host[0][seq_ids, block_indices]
        block_offsets = positions % block_size
        slot_mapping = physical_block_ids * block_size + block_offsets
        return positions, slot_mapping

    def _has_valid_inputs(self, attn_inputs: PyAttentionInputs) -> bool:
        """Return False when attn_inputs is a dummy placeholder (cuda_graph init)."""
        seq_lens = getattr(attn_inputs, "sequence_lengths", None)
        if seq_lens is None:
            return False
        kv_host = getattr(attn_inputs, "kv_cache_block_id_host", None)
        if kv_host is None or kv_host.numel() == 0:
            return False
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        if not self._has_valid_inputs(attn_inputs):
            # During cuda_graph init the inputs are dummy placeholders.
            # Return whatever persistent params we have (or a minimal dict).
            if self._positions is not None:
                return {
                    'k_scale': self._k_scale,
                    'v_scale': self._v_scale,
                    'head_num': self.attn_configs.head_num,
                    'kv_head_num': self.attn_configs.kv_head_num,
                    'size_per_head': self.attn_configs.size_per_head,
                    'tokens_per_block': self.attn_configs.tokens_per_block,
                    'positions': self._positions,
                    'slot_mapping': self._slot_mapping,
                }
            # First-ever call with dummy inputs: allocate zero-filled placeholders.
            batch_size = attn_inputs.input_lengths.size(0)
            self._positions = torch.zeros(batch_size, dtype=torch.long, device='cuda')
            self._slot_mapping = torch.zeros(batch_size, dtype=torch.long, device='cuda')
            self._k_scale = torch.tensor(1.0, device='cuda')
            self._v_scale = torch.tensor(1.0, device='cuda')
            return {
                'k_scale': self._k_scale,
                'v_scale': self._v_scale,
                'head_num': self.attn_configs.head_num,
                'kv_head_num': self.attn_configs.kv_head_num,
                'size_per_head': self.attn_configs.size_per_head,
                'tokens_per_block': self.attn_configs.tokens_per_block,
                'positions': self._positions,
                'slot_mapping': self._slot_mapping,
            }

        positions, slot_mapping = self._compute_positions_and_slots(attn_inputs)

        positions_cuda = positions.to('cuda')
        slot_mapping_cuda = slot_mapping.to('cuda')

        # Allocate persistent tensors on first call; reuse (copy in-place) on subsequent calls
        # so that any cuda_graph captured against these addresses stays valid.
        if self._positions is None or self._positions.shape != positions_cuda.shape:
            self._positions = positions_cuda.clone()
            self._slot_mapping = slot_mapping_cuda.clone()
            self._k_scale = torch.tensor(1.0, device='cuda')
            self._v_scale = torch.tensor(1.0, device='cuda')
        else:
            self._positions.copy_(positions_cuda)
            self._slot_mapping.copy_(slot_mapping_cuda)

        attn_params = {
            'k_scale': self._k_scale,
            'v_scale': self._v_scale,
            'head_num': self.attn_configs.head_num,
            'kv_head_num': self.attn_configs.kv_head_num,
            'size_per_head': self.attn_configs.size_per_head,
            'tokens_per_block': self.attn_configs.tokens_per_block,
            'positions': self._positions,
            'slot_mapping': self._slot_mapping,
        }
        return attn_params

    def update_kv_cache_offset(self, attn_inputs: PyAttentionInputs):
        """Update positions and slot_mapping in-place for cuda_graph replay."""
        if self._positions is None:
            return
        positions, slot_mapping = self._compute_positions_and_slots(attn_inputs)
        self._positions.copy_(positions.to('cuda'))
        self._slot_mapping.copy_(slot_mapping.to('cuda'))

    def forward(self, qkv: torch.Tensor, kv_cache: Optional[KVCache], params: Optional[Any]) -> torch.Tensor:
        q, k, v = qkv.split([params["head_num"]*params["size_per_head"],
                             params["kv_head_num"]*params["size_per_head"],
                             params["kv_head_num"]*params["size_per_head"]], dim=-1)
        lightop_ops.rotary_embedding_fuse(params["positions"], q, k, params["size_per_head"], self.cos_sin_cache, self.attn_configs.rope_config.is_neox_style)
        k_reshaped = k.reshape(-1, params["kv_head_num"], params["size_per_head"])
        v_reshaped = v.reshape(-1, params["kv_head_num"], params["size_per_head"])
        block_num = kv_cache.kv_cache_base.shape[1]
        k_cache = kv_cache.kv_cache_base[0].reshape(block_num, params["kv_head_num"], params["tokens_per_block"], params["size_per_head"])
        v_cache = kv_cache.kv_cache_base[1].reshape(block_num, params["kv_head_num"], params["size_per_head"], params["tokens_per_block"])
        lightop.reshape_and_cache_cuda(k_reshaped,
                                   v_reshaped,
                                   k_cache,
                                   v_cache,
                                   params["slot_mapping"],
                                   'auto',
                                   params["k_scale"],
                                   params["v_scale"])
        return q
