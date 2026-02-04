import logging
from dataclasses import dataclass
from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.common import (
    apply_rope_and_kv_cache,
    copy_kv_cache_offset,
    create_write_cache_store_op,
    write_to_cache_store,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    KVCache,
    PyAttentionInputs,
)


@dataclass
class XQAParams:
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    batch_size: int
    max_seq_len: int
    q_scale: float = 1.0
    kv_scale: float = 1.0
    o_scale: float = 1.0


class XQADecodeImpl(FMHAImplBase):
    """XQA Decode implementation with integrated XQA wrapper functionality."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs
    ) -> None:
        # Store config and attn_inputs
        self.config = attn_configs
        self.attn_inputs = attn_inputs
        self.cu_qseqlens = attn_inputs.cu_seqlens
        assert not self.attn_inputs.is_prefill, "XQA is not supported for prefill"
        
        # Initialize workspace buffer and semaphores if supported
        if self._check_support():
            self.workspace_buffer = torch.zeros(
                248 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
            self.semaphores = torch.zeros(
                8 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
        
        # Initialize rope and cache operations
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOp(attn_configs)
        self.input_lengths = attn_inputs.input_lengths
        self.cu_seq_lens = attn_inputs.cu_seqlens
        self.fmha_params = None
        self.rope_params = None

        self.write_cache_store_impl = create_write_cache_store_op(attn_inputs)
        
        # Prepare parameters using wrapper's prepare method
        self.fmha_params = self._prepare_xqa_params(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

    def _check_support(self) -> bool:
        """Check if XQA is supported for given configuration."""
        group_size = self.config.head_num // self.config.kv_head_num
        input_type_supported = self.config.dtype in [torch.bfloat16, torch.float16]
        output_type_supported = self.config.dtype in [
            torch.bfloat16,
            torch.float16,
            torch.float8_e4m3fn,
        ]
        group_size_supported = 1 <= group_size <= 16
        head_dim_supported = self.config.size_per_head in [64, 128, 256]
        page_size_supported = self.config.tokens_per_block in [16, 32, 64, 128]
        return (
            input_type_supported
            and output_type_supported
            and group_size_supported
            and head_dim_supported
            and page_size_supported
        )

    def _prepare_xqa_params(
        self,
        attn_inputs: PyAttentionInputs,
        q_scale: float = 1.0,
        kv_scale: float = 1.0,
        o_scale: float = 1.0,
    ) -> XQAParams:
        """Prepare XQA parameters for attention computation."""
        return XQAParams(
            page_table=attn_inputs.kv_cache_block_id_device,
            seq_lens=attn_inputs.sequence_lengths,
            batch_size=attn_inputs.sequence_lengths.size(0),
            max_seq_len=(
                attn_inputs.sequence_lengths.max().item() + 1
                if attn_inputs.sequence_lengths.numel() > 0
                else 0
            ),
            q_scale=q_scale,
            kv_scale=kv_scale,
            o_scale=o_scale,
        )

    def _init_spec_mask(self, q_4d: torch.Tensor):
        """Initialize speculative decoding mask for XQA."""
        q_len_per_req = q_4d.shape[1]
        batch_size = q_4d.shape[0]
        
        if q_len_per_req > 1:
            num_packed_masks_per_token = (q_len_per_req + 31) // 32
            q_indices = torch.arange(
                q_len_per_req, device=q_4d.device, dtype=torch.int32
            ).unsqueeze(1)
            kv_indices = torch.arange(
                q_len_per_req, device=q_4d.device, dtype=torch.int32
            ).unsqueeze(0)
            causal_bool_mask = kv_indices <= q_indices

            padded_seq_len = num_packed_masks_per_token * 32
            if padded_seq_len > q_len_per_req:
                padding = torch.zeros(
                    q_len_per_req,
                    padded_seq_len - q_len_per_req,
                    device=q_4d.device,
                    dtype=torch.bool,
                )
                causal_bool_mask = torch.cat([causal_bool_mask, padding], dim=1)

            causal_bool_mask = causal_bool_mask.view(
                q_len_per_req, num_packed_masks_per_token, 32
            )
            bit_positions = torch.tensor(
                [1 << i for i in range(32)], device=q_4d.device, dtype=torch.int64
            )
            mask_uint32 = (
                (causal_bool_mask.to(torch.int64) * bit_positions)
                .sum(dim=-1)
                .to(torch.uint32)
            )
            mask_uint32 = (
                mask_uint32.unsqueeze(0)
                .expand(batch_size, q_len_per_req, num_packed_masks_per_token)
                .contiguous()
            )
            mask = mask_uint32.view(torch.uint16)
            return mask
        else:
            return None

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """检查当前实现是否支持给定的输入。"""
        group_size = attn_configs.head_num // attn_configs.kv_head_num
        input_type_supported = attn_configs.dtype in [torch.bfloat16, torch.float16]
        output_type_supported = attn_configs.dtype in [
            torch.bfloat16,
            torch.float16,
            torch.float8_e4m3fn,
        ]
        group_size_supported = 1 <= group_size <= 16
        head_dim_supported = attn_configs.size_per_head in [64, 128, 256]
        page_size_supported = attn_configs.tokens_per_block in [16, 32, 64, 128]
        return (
            input_type_supported
            and output_type_supported
            and group_size_supported
            and head_dim_supported
            and page_size_supported
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        need_rope_kv_cache: bool = True,
    ) -> torch.Tensor:
        """执行前向传播计算。"""
        # Apply RoPE and KV cache operations
        fmha_input = apply_rope_and_kv_cache(
            qkv, kv_cache, self.rope_kvcache_impl, self.rope_params, need_rope_kv_cache
        )
        
        # Write to cache store if needed
        write_to_cache_store(kv_cache, self.attn_inputs, self.write_cache_store_impl)
        
        # Execute XQA forward pass directly
        q = fmha_input
        
        # Extract K/V cache in HND layout
        k_cache = kv_cache.kv_cache_base[:, 0, ...]  # [num_pages, num_kv_heads, page_size, head_dim]
        v_cache = kv_cache.kv_cache_base[:, 1, ...]
        page_table = self.fmha_params.page_table
        seq_lens = self.fmha_params.seq_lens
        num_kv_heads = k_cache.shape[1]
        page_size = k_cache.shape[2]
        kv_layout = "HND"

        # Get sequence lengths and validate they're uniform
        seqlens = torch.diff(self.attn_inputs.decode_cu_seqlens_d).cpu().tolist()
        assert (
            len(set(seqlens)) == 1
        ), f"All sequences must have the same length for XQA, got lengths: {seqlens}"
        
        q_len_per_req = seqlens[0]
        batch_size = len(seqlens)
        q_4d = q.reshape(batch_size, q_len_per_req, q.shape[1], q.shape[2])

        # Prepare sequence lengths for XQA kernel
        if seq_lens.dim() == 1:
            new_seq_lens = seq_lens + q_len_per_req
            seq_lens_4d = new_seq_lens.unsqueeze(1).to(torch.uint32).to(q.device)
        else:
            new_seq_lens = seq_lens[:, 0] + q_len_per_req
            seq_lens_4d = new_seq_lens.to(torch.uint32).to(q.device)

        # Check if PDL optimization is available (SM90+)
        enable_pdl = False
        try:
            compute_capability = torch.cuda.get_device_capability(q.device)
            enable_pdl = compute_capability[0] >= 9
        except Exception as e:
            logging.warning(
                f"[XQA] Failed to get GPU compute capability, PDL optimization disabled: {e}"
            )

        # Initialize speculative mask and prepare tensors
        spec_mask = self._init_spec_mask(q_4d)
        q_4d = q_4d.unsqueeze(1).contiguous()
        output = torch.zeros_like(q_4d)

        # Import and call XQA kernel
        from flashinfer.xqa import xqa

        q_scale = self.fmha_params.q_scale
        kv_scale = self.fmha_params.kv_scale
        o_scale = self.fmha_params.o_scale
        rcp_out_scale = 1.0 / o_scale if o_scale != 1.0 else 1.0

        xqa(
            q_4d,
            k_cache,
            v_cache,
            page_table,
            seq_lens_4d,
            output,
            workspace_buffer=self.workspace_buffer,
            semaphores=self.semaphores,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            kv_layout=kv_layout,
            enable_pdl=enable_pdl,
            q_seq_len=q_len_per_req,
            mask=spec_mask,
            nb_sub_seq_per_seq=1,
            use_qgmma=True,
            sinks=None,
            q_scale=q_scale,
            kv_scale=kv_scale,
            rcp_out_scale=rcp_out_scale,
        )
        return output

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        # Update FMHA params
        new_fmha_params = self._prepare_xqa_params(attn_inputs)
        # Copy updated parameters to existing params
        self.fmha_params.page_table = new_fmha_params.page_table
        self.fmha_params.seq_lens = new_fmha_params.seq_lens
        self.fmha_params.batch_size = new_fmha_params.batch_size
        self.fmha_params.max_seq_len = new_fmha_params.max_seq_len
        
        # Update rope params
        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        copy_kv_cache_offset(self.rope_params.kv_cache_offset, new_rope_params.kv_cache_offset)
