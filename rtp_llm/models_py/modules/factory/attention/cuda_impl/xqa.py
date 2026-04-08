import logging
from dataclasses import dataclass
from typing import Any, Optional, Type

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import (
    AttentionConfigs,
    FMHAConfig,
    FMHAType,
    KvCacheDataType,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    LayerKVCache,
    PyAttentionInputs,
    XQAAttnOp,
)

# Constants
DEFAULT_XQA_WORKSPACE_SIZE_MB = 248

# Global workspace buffer pool
_g_xqa_workspace_pool: list[torch.Tensor] = []
_g_xqa_pool_lock = __import__("threading").Lock()

# Global shared workspace buffer for CUDA graph mode
_g_xqa_shared_workspace_buffer: torch.Tensor = None
_g_xqa_shared_workspace_lock = __import__("threading").Lock()


def get_xqa_shared_workspace_buffer(device: str = "cuda") -> torch.Tensor:
    """Get or create the global shared XQA workspace buffer.

    This function returns a single shared workspace buffer that can be reused
    across all attention instances. This is especially useful in CUDA graph mode
    to reduce memory consumption.

    Args:
        device: CUDA device to allocate buffer on (default: "cuda")

    Returns:
        Global shared workspace buffer tensor of size DEFAULT_XQA_WORKSPACE_SIZE_MB
    """
    global _g_xqa_shared_workspace_buffer
    with _g_xqa_shared_workspace_lock:
        if _g_xqa_shared_workspace_buffer is None:
            _g_xqa_shared_workspace_buffer = torch.zeros(
                DEFAULT_XQA_WORKSPACE_SIZE_MB * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            )
        return _g_xqa_shared_workspace_buffer


def get_xqa_workspace_buffer(
    device: str = "cuda", shared: bool = False
) -> torch.Tensor:
    """Get an XQA workspace buffer from the pool or shared buffer.

    This function manages workspace buffers to support multiple concurrent instances.
    When shared=True, it returns a global shared buffer for all instances.
    When shared=False, it returns a buffer from the pool for exclusive use.

    Args:
        device: CUDA device to allocate buffer on (default: "cuda")
        shared: If True, return the global shared buffer; if False, get from pool

    Returns:
        Workspace buffer tensor of size DEFAULT_XQA_WORKSPACE_SIZE_MB
    """
    if shared:
        return get_xqa_shared_workspace_buffer(device)

    with _g_xqa_pool_lock:
        if _g_xqa_workspace_pool:
            return _g_xqa_workspace_pool.pop()
        else:
            return torch.zeros(
                DEFAULT_XQA_WORKSPACE_SIZE_MB * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            )


def release_xqa_workspace_buffer(buffer: torch.Tensor, shared: bool = False) -> None:
    """Release an XQA workspace buffer back to the pool.

    Args:
        buffer: The workspace buffer to release
        shared: If True, this is a shared buffer and should not be released; if False, return to pool
    """
    if shared:
        # Shared buffer is never released, just skip
        return

    with _g_xqa_pool_lock:
        _g_xqa_workspace_pool.append(buffer)


@dataclass
class XQAParams:
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    batch_size: int
    max_seq_len: int
    q_scale: float = 1.0
    kv_scale: float = 1.0
    o_scale: float = 1.0


class XQAImpl(FMHAImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = XQAAttnOp(attn_configs)
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOp(attn_configs)

        self.attn_inputs = attn_inputs

        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        fmha_impl = XQAAttnOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        common.update_trt_params(
            self.fmha_impl,
            self.rope_kvcache_impl,
            self.fmha_params,
            self.rope_params,
            attn_inputs,
        )


class XQADecodeImpl(FMHAImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.fmha_impl = XQAWrapper(attn_configs, attn_inputs)
        self.rope_kvcache_impl = FusedRopeKVCacheDecodeOp(attn_configs)

        self.attn_inputs = attn_inputs

        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        if attn_inputs.is_prefill:
            return False
        group_size = attn_configs.head_num // attn_configs.kv_head_num
        return (
            attn_configs.dtype in [torch.bfloat16, torch.float16, torch.float8_e4m3fn]
            and 1 <= group_size <= 16
            and attn_configs.size_per_head in [64, 128, 256]
            and attn_configs.tokens_per_block in [16, 32, 64, 128]
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        new_fmha_params = self.fmha_impl.prepare(attn_inputs)
        self.fmha_params.page_table = new_fmha_params.page_table
        self.fmha_params.seq_lens = new_fmha_params.seq_lens
        self.fmha_params.batch_size = new_fmha_params.batch_size
        self.fmha_params.max_seq_len = new_fmha_params.max_seq_len

        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        new_offset = new_rope_params.kv_cache_offset
        old_offset = self.rope_params.kv_cache_offset
        common.copy_kv_cache_offset(old_offset, new_offset)


class XQAWrapper:
    def __init__(
        self,
        config: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
    ):
        self.config = config
        self.attn_inputs = attn_inputs
        self.cu_qseqlens = attn_inputs.cu_seqlens
        assert not self.attn_inputs.is_prefill, "XQA is not supported"
        # attention_inputs is not used
        # init workspace_buffer and semaphores
        self.use_shared_buffer = config.shared_attn_workspace_buffer
        self.workspace_buffer = get_xqa_workspace_buffer(shared=self.use_shared_buffer)
        self.semaphores = torch.zeros(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    def __del__(self):
        """Release workspace buffer back to pool when object is destroyed."""
        if hasattr(self, "workspace_buffer") and hasattr(self, "use_shared_buffer"):
            release_xqa_workspace_buffer(
                self.workspace_buffer, shared=self.use_shared_buffer
            )

    def support(self, attn_inputs: Any) -> bool:
        group_size = self.config.head_num // self.config.kv_head_num
        input_type_supported = self.config.dtype in [torch.bfloat16, torch.float16]
        output_type_supported = self.config.dtype in [
            torch.bfloat16,
            torch.float16,
            torch.float8_e4m3fn,
        ]
        group_size_supported = 1 <= group_size <= 16
        head_dim_supported = self.config.size_per_head in [64, 128, 256]
        page_size_supported = self.config.kernel_tokens_per_block in [16, 32, 64, 128]
        return (
            input_type_supported
            and output_type_supported
            and group_size_supported
            and head_dim_supported
            and page_size_supported
        )

    def prepare(
        self,
        attn_inputs: PyAttentionInputs,
        q_scale: float = 1.0,
        kv_scale: float = 1.0,
        o_scale: float = 1.0,
    ) -> XQAParams:
        return XQAParams(
            page_table=attn_inputs.kv_cache_kernel_block_id_device,
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
        q: torch.Tensor,  # [total_tokens, num_heads, head_dim]
        kv_cache: LayerKVCache,
        fmha_params: XQAParams,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        k_cache = kv_cache.kv_cache_base[:, 0, ...]
        v_cache = kv_cache.kv_cache_base[:, 1, ...]
        page_table = fmha_params.page_table
        seq_lens = fmha_params.seq_lens
        num_kv_heads = k_cache.shape[1]
        page_size = k_cache.shape[2]
        kv_layout = "HND"

        seqlens = torch.diff(self.attn_inputs.decode_cu_seqlens_d).cpu().tolist()
        assert (
            len(set(seqlens)) == 1
        ), f"All sequences must have the same length for XQA, got lengths: {seqlens}"
        q_len_per_req = seqlens[0]
        batch_size = len(seqlens)
        q_4d = q.reshape(batch_size, q_len_per_req, q.shape[1], q.shape[2])

        if seq_lens.dim() == 1:
            new_seq_lens = seq_lens + q_len_per_req
            seq_lens_4d = new_seq_lens.unsqueeze(1).to(torch.uint32).to(q.device)
        else:
            new_seq_lens = seq_lens[:, 0] + q_len_per_req
            seq_lens_4d = new_seq_lens.to(torch.uint32).to(q.device)

        enable_pdl = False
        try:
            compute_capability = torch.cuda.get_device_capability(q.device)
            enable_pdl = compute_capability[0] >= 9
        except Exception as e:
            logging.warning(
                f"[XQA] Failed to get GPU compute capability, PDL optimization disabled: {e}"
            )

        spec_mask = self.init_spec_mask(q_4d)
        q_4d = q_4d.unsqueeze(1).contiguous()
        output = torch.zeros_like(q_4d)

        try:
            from rtp_kernel.xqa import xqa
        except ImportError:
            from flashinfer.xqa import xqa

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


def get_xqa_impl() -> Type[FMHAImplBase]:
    """
    Select the appropriate XQA implementation based on CUDA version and flashinfer availability.

    Returns XQADecodeImpl if CUDA >= 12.8 and flashinfer.xqa is available,
    otherwise falls back to XQAImpl.
    """
    logging.info(f"using XQA Kernel implementation")
    return XQAImpl
    # TODO: cudagraph bazel ut cant pass
    # try:
    #     major, minor = map(int, torch.version.cuda.split(".")[:2])
    #     if (major, minor) >= (12, 8):
    #         try:
    #             from flashinfer.xqa import xqa

    #             logging.info(
    #                 "CUDA >= 12.8 and flashinfer.xqa available, using XQADecodeImpl"
    #             )
    #             return XQADecodeImpl
    #         except (ImportError, AttributeError) as e:
    #             logging.info(
    #                 f"CUDA >= 12.8 but flashinfer.xqa not available ({e}), falling back to XQAImpl"
    #             )
    #             return XQAImpl
    #     else:
    #         logging.info(f"CUDA version {major}.{minor} < 12.8, using XQAImpl")
    #         return XQAImpl
    # except Exception as e:
    #     logging.warning(f"Failed to check CUDA version ({e}), using XQAImpl")
    #     return XQAImpl
