import logging
from dataclasses import dataclass
from typing import Any, Optional, Type

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.utils.arch import get_num_device_sms
from rtp_llm.ops import (
    AttentionConfigs,
    FMHAConfig,
    FMHAType,
    KvCacheDataType,
    ParallelismConfig,
    RopeStyle,
)
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    LayerKVCache,
    PyAttentionInputs,
    XQAAttnOp,
    rtp_llm_ops,
)

# Constants
DEFAULT_XQA_WORKSPACE_SIZE_MB = 248

# Global workspace buffer pool
_g_xqa_workspace_pool: list[torch.Tensor] = []
_g_xqa_pool_lock = __import__("threading").Lock()


def get_xqa_workspace_buffer(device: str = "cuda") -> torch.Tensor:
    with _g_xqa_pool_lock:
        if _g_xqa_workspace_pool:
            return _g_xqa_workspace_pool.pop()
        else:
            return torch.zeros(
                DEFAULT_XQA_WORKSPACE_SIZE_MB * 1024 * 1024,
                dtype=torch.uint8,
                device=device,
            )


def release_xqa_workspace_buffer(buffer: torch.Tensor) -> None:
    with _g_xqa_pool_lock:
        _g_xqa_workspace_pool.append(buffer)


@dataclass
class XQAParams:
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    batch_size: int
    max_seq_len: int
    # FP8 KV cache uses direct cast (no dynamic scaling), so per-block scale is always 1.0.
    # See fused_rope_kvcache_kernel.cu: s_max is hardcoded to 128, stored scale = 128/128 = 1.0.
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
        self.attn_configs = attn_configs
        self.use_fp8_per_token_head = (
            attn_configs.kv_cache_dtype == KvCacheDataType.FP8
            and getattr(attn_configs, "fp8_kv_cache_scale_mode", "per_tensor")
            == "per_token_head"
        )
        self.fmha_impl = XQAAttnOp(attn_configs)
        if self.use_fp8_per_token_head:
            self.rope_kvcache_impl = None
            self.rope_impl = (
                None
                if attn_configs.rope_config.style == RopeStyle.No
                else MhaRotaryEmbeddingOp(attn_configs)
            )
            self.kv_cache_write_op = KVCacheWriteOp(
                num_kv_heads=attn_configs.kv_head_num,
                head_size=attn_configs.size_per_head,
                token_per_block=attn_configs.kernel_tokens_per_block,
                fp8_kv_cache_scale_mode="per_token_head",
                kv_cache_dtype=attn_configs.kv_cache_dtype,
            )
        else:
            self.rope_kvcache_impl = FusedRopeKVCacheDecodeOp(attn_configs)
            self.rope_impl = None
            self.kv_cache_write_op = None

        self.attn_inputs = attn_inputs

        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        if self.use_fp8_per_token_head:
            self.rope_params = rtp_llm_ops.FlashInferMlaAttnParams()
            self.rope_params.fill_params(
                attn_inputs.prefix_lengths,
                attn_inputs.sequence_lengths,
                attn_inputs.input_lengths,
                attn_inputs.kv_cache_kernel_block_id_host,
                attn_configs.kernel_tokens_per_block,
            )
            if self.rope_impl is not None:
                self.rope_impl.set_params(self.rope_params)
            self.kv_cache_write_op.set_params(self.rope_params)
        else:
            self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        # C++ XQAParams.sequence_lengths shares storage with this tensor.
        # Keep a reference so prepare_cuda_graph can update it in-place.
        self._captured_seq_lens = attn_inputs.sequence_lengths

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        fmha_impl = XQAAttnOp(attn_configs)
        return fmha_impl.support(attn_inputs)

    def _split_qkv(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head_dim = self.attn_configs.size_per_head
        q, k, v = torch.split(
            qkv.reshape(qkv.shape[0], -1),
            [
                head_dim * self.attn_configs.head_num,
                head_dim * self.attn_configs.kv_head_num,
                head_dim * self.attn_configs.kv_head_num,
            ],
            dim=-1,
        )
        q = q.reshape(q.shape[0], self.attn_configs.head_num, head_dim)
        k = k.reshape(q.shape[0], self.attn_configs.kv_head_num, head_dim)
        v = v.reshape(q.shape[0], self.attn_configs.kv_head_num, head_dim)
        return q, k, v

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if self.use_fp8_per_token_head:
            assert (
                kv_cache is not None
            ), "kv_cache is required for FP8 per-token-head XQA"
            if self.need_rope_kv_cache and self.rope_impl is not None:
                q, k, v = self.rope_impl.forward(qkv)
            else:
                q, k, v = self._split_qkv(qkv)
            assert self.kv_cache_write_op is not None
            self.kv_cache_write_op.forward(k, v, kv_cache)
            fmha_input = q
        elif self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        return self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        if self.use_fp8_per_token_head:
            self.rope_params.fill_params(
                attn_inputs.prefix_lengths,
                attn_inputs.sequence_lengths,
                attn_inputs.input_lengths,
                attn_inputs.kv_cache_kernel_block_id_host,
                self.attn_configs.kernel_tokens_per_block,
                True,
            )
            new_fmha_params = self.fmha_impl.prepare(attn_inputs)
            common.copy_kv_cache_offset(
                self.fmha_params.kv_cache_offset,
                new_fmha_params.kv_cache_offset,
            )
        else:
            common.update_trt_params(
                self.fmha_impl,
                self.rope_kvcache_impl,
                self.fmha_params,
                self.rope_params,
                attn_inputs,
            )
        # update_trt_params only copies kv_cache_offset. The TRT XQA kernel also
        # reads sequence_lengths via the captured data_ptr(), so we must update
        # the data in-place at the address recorded during CUDA graph capture.
        new_seq_lens = attn_inputs.sequence_lengths
        n = min(self._captured_seq_lens.numel(), new_seq_lens.numel())
        self._captured_seq_lens[:n].copy_(new_seq_lens[:n], non_blocking=True)


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
        if attn_configs.kv_cache_dtype == KvCacheDataType.INT8:
            return False
        if torch.cuda.get_device_capability()[0] not in [9, 10, 12]:
            return False
        group_size = attn_configs.head_num // attn_configs.kv_head_num
        return (
            attn_configs.dtype in [torch.bfloat16, torch.float16, torch.float8_e4m3fn]
            and 1 <= group_size <= 16
            and attn_configs.size_per_head in [64, 128, 256]
            and attn_configs.kernel_tokens_per_block in [16, 32, 64, 128]
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
        self.fmha_impl.prepare_for_cuda_graph_replay(attn_inputs)

        new_fmha_params = self.fmha_impl.make_params(attn_inputs)
        # page_table 来自 attn_inputs.kv_cache_kernel_block_id_device，在生产路径中
        # 由 C++ CudaGraphRunner 预分配固定地址 tensor 并通过 in-place D2D copy 更新内容，
        # 因此这里赋值后 data_ptr 与 graph 捕获时一致，replay 安全。
        self.fmha_params.page_table = new_fmha_params.page_table
        self.fmha_params.seq_lens = new_fmha_params.seq_lens
        self.fmha_params.batch_size = new_fmha_params.batch_size
        self.fmha_params.max_seq_len = new_fmha_params.max_seq_len

        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        new_offset = new_rope_params.kv_cache_offset
        old_offset = self.rope_params.kv_cache_offset
        common.copy_kv_cache_offset(old_offset, new_offset)


def _load_xqa_fn():
    """Load the XQA kernel function, handling rtp_kernel/flashinfer API differences.

    Always prefer flashinfer.xqa.xqa directly over rtp_kernel.xqa.xqa, because
    rtp_kernel's wrapper calls load_xqa_best_config() which overrides the caller's
    nb_sub_seq_per_seq / use_qgmma with a config-file lookup (defaulting to
    nb_sub_seq=4 when no config file exists). XQAWrapper.forward() already
    computes these values adaptively, so the rtp_kernel override is harmful.
    """
    import inspect

    try:
        from flashinfer.xqa import xqa as fi_xqa

        needs_sf = "k_sf_cache" in inspect.signature(fi_xqa).parameters
        return fi_xqa, needs_sf
    except ImportError:
        pass

    from rtp_kernel.xqa import xqa

    return xqa, False


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
        self.workspace_buffer = get_xqa_workspace_buffer()
        self.semaphores = torch.zeros(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")

        self.enable_pdl = False
        self._sm_count = 1
        try:
            compute_capability = torch.cuda.get_device_capability()
            self.enable_pdl = compute_capability[0] >= 9
            self._sm_count = get_num_device_sms()
        except Exception as e:
            logging.warning(
                f"[XQA] Failed to get GPU compute capability, PDL optimization disabled: {e}"
            )

        self._xqa_fn, self._xqa_needs_sf_cache = _load_xqa_fn()

        self._batch_size: int = 0
        self._q_len_per_req: int = 0
        self._spec_mask: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._seq_lens_4d: Optional[torch.Tensor] = None

    def __del__(self):
        release_xqa_workspace_buffer(self.workspace_buffer)

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

    def _compute_batch_geometry(self, attn_inputs: PyAttentionInputs) -> None:
        cu_seqlens = attn_inputs.decode_cu_seqlens_host
        seqlens = torch.diff(cu_seqlens).tolist()
        assert (
            len(set(seqlens)) == 1
        ), f"All sequences must have the same length for XQA, got lengths: {seqlens}"
        self._q_len_per_req = seqlens[0]
        self._batch_size = len(seqlens)

    def _compute_spec_mask(self, device: str = "cuda") -> None:
        q_len = self._q_len_per_req
        bs = self._batch_size
        if q_len <= 1:
            self._spec_mask = None
            return
        num_packed_masks_per_token = (q_len + 31) // 32
        q_indices = torch.arange(q_len, device=device, dtype=torch.int32).unsqueeze(1)
        kv_indices = torch.arange(q_len, device=device, dtype=torch.int32).unsqueeze(0)
        causal_bool_mask = kv_indices <= q_indices

        padded_seq_len = num_packed_masks_per_token * 32
        if padded_seq_len > q_len:
            padding = torch.zeros(
                q_len,
                padded_seq_len - q_len,
                device=device,
                dtype=torch.bool,
            )
            causal_bool_mask = torch.cat([causal_bool_mask, padding], dim=1)

        causal_bool_mask = causal_bool_mask.view(q_len, num_packed_masks_per_token, 32)
        bit_positions = torch.tensor(
            [1 << i for i in range(32)], device=device, dtype=torch.int64
        )
        mask_uint32 = (
            (causal_bool_mask.to(torch.int64) * bit_positions)
            .sum(dim=-1)
            .to(torch.uint32)
        )
        mask_uint32 = (
            mask_uint32.unsqueeze(0)
            .expand(bs, q_len, num_packed_masks_per_token)
            .contiguous()
        )
        self._spec_mask = mask_uint32.view(torch.uint16)

    def _alloc_buffers(
        self, num_heads: int, head_dim: int, dtype: torch.dtype, device: str = "cuda"
    ) -> None:
        bs = self._batch_size
        q_len = self._q_len_per_req
        self._output_buffer = torch.zeros(
            bs,
            1,
            q_len,
            num_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self._seq_lens_4d = torch.zeros(bs, 1, dtype=torch.uint32, device=device)

    def _update_seq_lens_4d(self, seq_lens: torch.Tensor) -> None:
        """Update _seq_lens_4d from CPU seq_lens. Must be called OUTSIDE CUDA graph capture."""
        assert self._seq_lens_4d is not None
        bs = self._batch_size
        new_seq_lens = (seq_lens[:bs] + self._q_len_per_req).to(torch.uint32)
        self._seq_lens_4d[:bs].copy_(new_seq_lens.unsqueeze(1))

    def make_params(
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

    def prepare(
        self,
        attn_inputs: PyAttentionInputs,
        q_scale: float = 1.0,
        kv_scale: float = 1.0,
        o_scale: float = 1.0,
    ) -> XQAParams:
        self._compute_batch_geometry(attn_inputs)
        self._compute_spec_mask()
        self._alloc_buffers(
            self.config.head_num,
            self.config.size_per_head,
            self.config.dtype,
        )
        self._update_seq_lens_4d(attn_inputs.sequence_lengths)
        return self.make_params(attn_inputs, q_scale, kv_scale, o_scale)

    def prepare_for_cuda_graph_replay(self, attn_inputs: PyAttentionInputs) -> None:
        self._update_seq_lens_4d(attn_inputs.sequence_lengths)

    def forward(
        self,
        q: torch.Tensor,  # [total_tokens, num_heads, head_dim]
        kv_cache: LayerKVCache,
        fmha_params: XQAParams,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        paged_kv_cache = kv_cache.kv_cache_base
        if paged_kv_cache.dim() == 2:
            paged_kv_cache = common.reshape_paged_kv_cache(
                paged_kv_cache,
                self.config.kv_head_num,
                self.config.kernel_tokens_per_block,
                self.config.size_per_head,
            )
        k_cache = paged_kv_cache[:, 0, ...]
        v_cache = paged_kv_cache[:, 1, ...]
        page_table = fmha_params.page_table
        num_kv_heads = k_cache.shape[1]
        page_size = k_cache.shape[2]

        batch_size = self._batch_size
        q_len_per_req = self._q_len_per_req

        q_4d = q.reshape(batch_size, q_len_per_req, q.shape[1], q.shape[2])

        seq_lens_4d = self._seq_lens_4d[:batch_size]

        q_4d = q_4d.unsqueeze(1).contiguous()
        assert self._output_buffer is not None
        self._output_buffer.zero_()
        output = self._output_buffer

        q_scale = fmha_params.q_scale
        kv_scale = fmha_params.kv_scale
        o_scale = fmha_params.o_scale
        rcp_out_scale = 1.0 / o_scale if o_scale != 1.0 else 1.0

        # Adaptive multi-block: split KV across CTAs to fill SMs at small batch sizes.
        # Matches TRT-LLM XQA's formula in mha.cu. flashinfer >= 0.6.11 auto-tunes
        # internally (this becomes a no-op), but older versions (e.g. 0.6.6) respect
        # this parameter and need it for bs=1 performance.
        _XQA_TILE_TOKENS = 64
        nb_sub_seq = max(
            1,
            min(
                self._sm_count // max(1, batch_size * num_kv_heads),
                (fmha_params.max_seq_len + _XQA_TILE_TOKENS - 1) // _XQA_TILE_TOKENS,
            ),
        )

        xqa_kwargs = dict(
            q=q_4d,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            seq_lens=seq_lens_4d,
            output=output,
            workspace_buffer=self.workspace_buffer,
            semaphores=self.semaphores,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            kv_layout="HND",
            enable_pdl=self.enable_pdl,
            q_seq_len=q_len_per_req,
            mask=self._spec_mask,
            nb_sub_seq_per_seq=nb_sub_seq,
            use_qgmma=True,
            sinks=None,
            q_scale=q_scale,
            kv_scale=kv_scale,
            rcp_out_scale=rcp_out_scale,
        )
        if self._xqa_needs_sf_cache:
            xqa_kwargs["k_sf_cache"] = None
            xqa_kwargs["v_sf_cache"] = None
        self._xqa_fn(**xqa_kwargs)
        return output


def get_xqa_impl() -> Type[FMHAImplBase]:
    """
    Select the appropriate XQA implementation based on CUDA version and flashinfer availability.

    Returns XQADecodeImpl if CUDA >= 12.8 and flashinfer.xqa is available,
    otherwise falls back to XQAImpl.
    """
    try:
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        if (major, minor) >= (12, 8):
            try:
                from flashinfer.xqa import xqa

                logging.info(
                    "CUDA >= 12.8 and flashinfer.xqa available, using XQADecodeImpl"
                )
                return XQADecodeImpl
            except (ImportError, AttributeError) as e:
                logging.info(
                    f"CUDA >= 12.8 but flashinfer.xqa not available ({e}), falling back to XQAImpl"
                )
                return XQAImpl
        else:
            logging.info(f"CUDA version {major}.{minor} < 12.8, using XQAImpl")
            return XQAImpl
    except Exception as e:
        logging.warning(f"Failed to check CUDA version ({e}), using XQAImpl")
        return XQAImpl
