import logging
from typing import Any, Optional, Type
from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHADecodeImplBase,
)
from rtp_llm.ops import AttentionConfigs, FMHAType, FMHAConfig, KvCacheDataType
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    PyAttentionInputs,
    XQAAttnOp,
    KVCache
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
        

class XQAImpl(FMHADecodeImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs
    ) -> None:
        super().__init__(
            XQAAttnOp(attn_configs),
            FusedRopeKVCacheDecodeOp(attn_configs),
            attn_inputs,
        )

    def create_params(self, attn_inputs: PyAttentionInputs):
        assert self.fmha_impl is not None
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        assert self.rope_kvcache_impl is not None
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.XQA

    def support_cuda_graph(self) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs):
        self._update_trt_params(attn_inputs)


class XQADecodeImpl(FMHADecodeImplBase):

    def __init__(
        self, 
        attn_configs: AttentionConfigs, 
        attn_inputs: PyAttentionInputs,
    ) -> None:
        # Create XQAWrapper
        xqa_wrapper = XQAWrapper(attn_configs, attn_inputs)
        super().__init__(
            xqa_wrapper,
            FusedRopeKVCacheDecodeOp(attn_configs),
            attn_inputs,
        )

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.XQA

    def support_cuda_graph(self) -> bool:
        return True


class XQAWrapper:
    def __init__(
        self,
        config: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ):
        self.config = config
        self.attn_inputs = attn_inputs
        self.cu_qseqlens = attn_inputs.cu_seqlens
        assert (
           not self.attn_inputs.is_prefill
        ), "XQA is not supported"
        group_size = self.config.head_num // self.config.kv_head_num
        

        kv_cache_type_supported = self.config.kv_cache_dtype in [
            KvCacheDataType.BASE,
            KvCacheDataType.FP8,
        ]

        input_type_supported = self.config.dtype in [torch.bfloat16, torch.float16]
        output_type_supported = self.config.dtype in [torch.bfloat16, torch.float16, torch.float8_e4m3fn]
        group_size_supported = (1<= group_size <= 16)
        head_dim_supported = self.config.size_per_head in [64, 128, 256]
        page_size_supported = self.config.tokens_per_block in [16, 32, 64, 128]
        assert (
            input_type_supported
            and output_type_supported
            and group_size_supported
            and head_dim_supported
            and page_size_supported
        ), "XQA is not supported"

        # init workspace_buffer and semaphores
        self.workspace_buffer = torch.zeros(
            248 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
        self.semaphores = torch.zeros(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    @staticmethod
    def support(attn_inputs: PyAttentionInputs) -> bool:
        return True

    def prepare(self, attn_inputs: PyAttentionInputs, q_scale: float = 1.0, kv_scale: float = 1.0, o_scale: float = 1.0) -> XQAParams:
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

    def init_spec_mask(self, q_4d: torch.Tensor):
        # init spec mask
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

    def forward(
        self,
        q: torch.Tensor, #[total_tokens, num_heads, head_dim] 
        kv_cache: KVCache,
        fmha_params: XQAParams,
    ) -> torch.Tensor:
        # [num_pages, num_kv_heads, page_size, head_dim] - HND layout
        k_cache = kv_cache.kv_cache_base[:, 0, ...]
        v_cache = kv_cache.kv_cache_base[:, 1, ...]
        page_table = fmha_params.page_table
        seq_lens = fmha_params.seq_lens  # cpu device
        num_kv_heads = k_cache.shape[1]
        page_size = k_cache.shape[2]
        kv_layout = "HND"
       
        seqlens = torch.diff(self.attn_inputs.decode_cu_seqlens_d).cpu().tolist()
        # Assert all sequences have the same length for XQA
        assert len(set(seqlens)) == 1, \
            f"All sequences must have the same length for XQA, got lengths: {seqlens}"
        q_len_per_req = seqlens[0]
        batch_size = len(seqlens)
        q_4d = q.reshape(batch_size, q_len_per_req, q.shape[1], q.shape[2])
        
        if seq_lens.dim() == 1:
            new_seq_lens = seq_lens + q_len_per_req
            seq_lens_4d = (
                new_seq_lens.unsqueeze(1).to(torch.uint32).to(q.device)
            )  # [batch_size] -> [batch_size, 1]
        else:
            new_seq_lens = seq_lens[:, 0] + q_len_per_req
            seq_lens_4d = new_seq_lens.to(torch.uint32).to(q.device)

        enable_pdl = False
        try:
            compute_capability = torch.cuda.get_device_capability(q.device)
            enable_pdl = compute_capability[0] >= 9  # SM90+
        except Exception as e:
            logging.warning(f"[XQA] Failed to get GPU compute capability, PDL optimization disabled: {e}")
            enable_pdl = False
        spec_mask = self.init_spec_mask(q_4d)
        q_4d = q_4d.unsqueeze(1).contiguous()
        output = torch.zeros_like(q_4d)

        # when nb_sub_seq_per_seq is None, xqa will use the best config for the current gpu.
        # https://code.alibaba-inc.com/foundation_models/flashinfer/blob/main/best_config/NVIDIA_L20X_XQA_inbf16_cachefp8_outbf16_ps64_hd128_nq12_nkv1.json
        from flashinfer.xqa import xqa
        
        # Get scale parameters from fmha_params
        q_scale = fmha_params.q_scale
        kv_scale = fmha_params.kv_scale
        o_scale = fmha_params.o_scale
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


def get_xqa_impl() -> Type[FMHADecodeImplBase]:
    """
    Select the appropriate XQA implementation based on CUDA version and flashinfer availability.
    
    Returns XQADecodeImpl if CUDA >= 12.8 and flashinfer.xqa is available,
    otherwise falls back to XQAImpl.
    """
    try:
        major, minor = map(int, torch.version.cuda.split('.')[:2])
        if (major, minor) >= (12, 8):
            try:
                from flashinfer.xqa import xqa
                logging.info("CUDA >= 12.8 and flashinfer.xqa available, using XQADecodeImpl")
                return XQADecodeImpl
            except (ImportError, AttributeError) as e:
                logging.info(f"CUDA >= 12.8 but flashinfer.xqa not available ({e}), falling back to XQAImpl")
                return XQAImpl
        else:
            logging.info(f"CUDA version {major}.{minor} < 12.8, using XQAImpl")
            return XQAImpl
    except Exception as e:
        logging.warning(f"Failed to check CUDA version ({e}), using XQAImpl")
        return XQAImpl