"""Experimental TRTLLM-gen sparse Decode with RTP's existing 656-byte KV.

Only selected KV is decoded/requantized into temporary plain E4M3 storage.
The persistent cache, Prefill/PD writers, RoPE and Indexer remain unchanged.
This is deliberately NOT the native-576 SGLang benchmark path, nor a lossless
replacement for FlashMLA. Enable explicitly with GLM5_SPARSE_DECODE_BACKEND.
"""

import logging
from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaFp8Op,
    SparseMlaImpl,
    SparseMlaOp,
    _is_multi_token_decode,
    _topk_2d,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.trtllm_kv_compat import (
    convert_selected_kv,
    mask_empty_output,
)
from rtp_llm.ops import KvCacheDataType


class TrtllmSparseMlaFp8Op(SparseMlaFp8Op):
    """Same paged-input/prepare contract as FlashMLA, different inner kernel.

    One instance owns its scratch across sequential layers. Graph instances do
    not share writable buffers, including the TRT semaphore workspace. CMP
    calls this very same forward after its existing side-stream join.
    """

    WORKSPACE_BYTES = 128 * 1024 * 1024
    backend_name = "trtllm_gen_656_compat"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            self.num_heads not in (8, 16, 32, 64)
            or self.kv_lora_rank != 512
            or self.qk_rope_head_dim != 64
            or self.qk_nope_head_dim != 192
            or self.token_per_block != 64
            or self.top_k != 2048
        ):
            raise ValueError(
                "TRT sparse Decode requires H=8/16/32/64, latent=512, "
                "NoPE=192, RoPE=64, page=64, TopK=2048"
            )
        # Lazy import: default FlashMLA and Prefill never load this backend.
        try:
            from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "Experimental TRT sparse Decode requires FlashInfer with the "
                "TRTLLM-gen MLA API (tested: flashinfer-python 0.6.12 and its "
                "matching cubin package); it is not supplied by the CUDA13 lock"
            ) from error

        self._decode = trtllm_batch_decode_with_kv_cache_mla
        self._capacity: Optional[int] = None
        self._device: Optional[torch.device] = None
        self.workspace_buffer = None
        self._seq_lens = None

    def _reserve(self, tokens: int, device: torch.device) -> None:
        if self._capacity == tokens and self._device == device:
            return
        if self._capacity is not None and self.use_cuda_graph:
            raise RuntimeError(
                "TRT sparse Decode cannot change captured scratch shape/device: "
                f"{self._capacity}/{self._device} -> {tokens}/{device}"
            )
        if device.type != "cuda":
            raise ValueError("TRT sparse Decode requires CUDA tensors")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare TRT sparse scratch before graph capture")
            capability = torch.cuda.get_device_capability(device)
            if capability not in ((10, 0), (10, 3)):
                raise ValueError(
                    f"TRT sparse Decode requires SM100/SM103, got {capability}"
                )

        opts = {"device": device}
        # No materialized BF16 selected cache or permanent native-cache mirror.
        self.selected_kv = torch.empty(
            (tokens, self.top_k, 576), dtype=torch.float8_e4m3fn, **opts
        )
        self.q_fp8 = torch.empty(
            (tokens, self.num_heads, 576), dtype=torch.float8_e4m3fn, **opts
        )
        self.source_indices = torch.empty(
            (tokens, self.top_k), dtype=torch.int64, **opts
        )
        self.physical_indices = torch.empty(
            (tokens, self.top_k), dtype=torch.int32, **opts
        )
        self.valid_counts = torch.empty((tokens,), dtype=torch.int32, **opts)
        self.trt_seq_lens = torch.empty_like(self.valid_counts)
        self.output = torch.empty(
            (tokens, self.num_heads, 512), dtype=torch.bfloat16, **opts
        )
        if self.workspace_buffer is None or self._device != device:
            # TRT's multi-block semaphore workspace must initially be zero.
            self.workspace_buffer = torch.zeros(
                self.WORKSPACE_BYTES, dtype=torch.uint8, **opts
            )
        self._capacity = tokens
        self._device = device
        self._selected_paged = self.selected_kv.view(-1, 1, 64, 576)
        self._query_view = self.q_fp8.view(tokens, 1, self.num_heads, 576)
        self._indices_view = self.physical_indices.view(tokens, 1, self.top_k)
        self._output_view = self.output.view(tokens, 1, self.num_heads, 512)
        scratch_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.selected_kv,
                self.q_fp8,
                self.source_indices,
                self.physical_indices,
                self.valid_counts,
                self.trt_seq_lens,
                self.output,
                self.workspace_buffer,
            )
        )
        logging.info(
            "TRT_SPARSE_DECODE_PREPARED: experimental=1 cache=rtp656 "
            "heads=%d tokens=%d scratch_bytes=%d cuda_graph=%s",
            self.num_heads,
            tokens,
            scratch_bytes,
            self.use_cuda_graph,
        )

    def plan(self, mla_params, block_table, attn_inputs=None) -> None:
        # Do not call SparseMlaFp8Op.plan: its FlashMLA scheduler/gather are
        # unrelated to TRT. Keep the inherited type for device-only prepare.
        SparseMlaOp.plan(self, mla_params, block_table, attn_inputs)
        if attn_inputs is not None and (
            getattr(attn_inputs, "is_prefill", False)
            and not _is_multi_token_decode(attn_inputs)
        ):
            raise ValueError("TRT sparse compatibility backend is Decode/MTP only")
        req_ids = mla_params.batch_indice_d
        # SparseMlaParams aliases expanded_seq_lens to kvlen_d for q=1, and
        # supplies per-query causal lengths for target verify / draft extend.
        lengths = getattr(mla_params, "expanded_seq_lens", None)
        if lengths is None or lengths.numel() == 0:
            lengths = mla_params.kvlen_d
        tokens = req_ids.numel()
        if (
            req_ids.ndim != 1
            or lengths.ndim != 1
            or lengths.numel() != tokens
            or req_ids.dtype != torch.int32
            or lengths.dtype != torch.int32
            or block_table.dtype != torch.int32
            or block_table.ndim != 2
            or lengths.device != req_ids.device
            or block_table.device != req_ids.device
        ):
            raise ValueError("TRT sparse requires device int32 per-query metadata")
        self._seq_lens = lengths
        self._reserve(tokens, req_ids.device)

    def forward(
        self, q, kv, topk_indices, kv_scale=None, layer_id=0, physical_indices=None
    ):
        """Consume already-rotated absorbed Q; return BF16 latent attention.

        All work stays on the caller's CUDA stream, including when called by
        CMP. No CPU reads of lengths/indices and no new synchronization.
        """
        if self._capacity is None or self._seq_lens is None:
            raise RuntimeError("Call plan() before TRT sparse forward")
        if (
            q.shape != (self._capacity, self.num_heads, 576)
            or q.dtype != torch.bfloat16
            or q.device != self._device
        ):
            raise ValueError("TRT sparse expects prepared BF16 Q [T,H,576]")
        if topk_indices.ndim not in (2, 3) or (
            topk_indices.ndim == 3 and topk_indices.shape[1] != 1
        ):
            raise ValueError("TRT sparse TopK must be [T,K] or [T,1,K]")
        topk = _topk_2d(topk_indices)
        if topk.shape != (self._capacity, self.top_k):
            raise ValueError("TRT sparse TopK does not match the prepared shape")
        if physical_indices is not None:
            if physical_indices.ndim not in (2, 3) or (
                physical_indices.ndim == 3 and physical_indices.shape[1] != 1
            ):
                raise ValueError("TRT sparse physical_indices must be [T,K] or [T,1,K]")
            physical_indices = _topk_2d(physical_indices)
            if (
                physical_indices.shape != topk.shape
                or physical_indices.dtype != torch.int32
                or physical_indices.device != self._device
                or not physical_indices.is_contiguous()
            ):
                raise ValueError(
                    "TRT sparse physical_indices must be contiguous device int32 "
                    "matching TopK"
                )
        if self._capacity == 0:
            return self.output
        if kv.dtype not in (torch.uint8, torch.float8_e4m3fn):
            raise ValueError("TRT sparse expects the existing packed RTP FP8 KV")
        if (
            kv.ndim not in (3, 4)
            or kv.shape[1] != self.token_per_block
            or kv.shape[-1] != 656
            or (kv.ndim == 4 and kv.shape[2] != 1)
        ):
            raise ValueError("TRT sparse expects paged KV [N,64,656] or [N,64,1,656]")
        # C++ cache tensors can expose the byte storage as FP8 rather than uint8.
        # Reinterpret only; never numerically cast the packed BF16/scale bytes.
        kv_bytes = kv.view(torch.uint8) if kv.dtype != torch.uint8 else kv
        convert_selected_kv(
            q,
            kv_bytes,
            topk,
            self.mla_params.batch_indice_d,
            self.block_table,
            self._seq_lens,
            q_out=self.q_fp8,
            kv_out=self.selected_kv,
            source_indices=self.source_indices,
            indices_out=self.physical_indices,
            counts_out=self.valid_counts,
            lengths_out=self.trt_seq_lens,
            physical_indices=physical_indices,
        )
        self._decode(
            query=self._query_view,
            kv_cache=self._selected_paged,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=self._indices_view,
            seq_lens=self.trt_seq_lens,
            max_seq_len=self.top_k,
            sparse_mla_top_k=self.top_k,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            backend="trtllm-gen",
            out=self._output_view,
        )
        mask_empty_output(self.output, self.valid_counts)
        return self.output


class TrtllmSparseMlaImpl(SparseMlaImpl):
    """Retain RTP's normal/CMP producers and device-only MTP preparation."""

    def __init__(self, *args, **kwargs):
        if "fmha_impl" in kwargs:
            raise ValueError("TRT sparse op must not be overridden")
        super().__init__(*args, fmha_impl=TrtllmSparseMlaFp8Op, **kwargs)

    @classmethod
    def support_parallelism_config(cls, parallelism_config) -> bool:
        if not super().support_parallelism_config(parallelism_config):
            return False
        # KVCacheManager creates CPSlotMapper from these two fields regardless
        # of CP method/role. PREFILL_CP metadata alone does not imply a full
        # block table: use raw TP size, not get_attn_tp_size/prefill_cp_size.
        return parallelism_config is None or not (
            parallelism_config.prefill_cp_config.kv_cache_sharded
            and parallelism_config.tp_size > 1
        )

    @classmethod
    def support(cls, attn_configs, attn_inputs) -> bool:
        return (
            attn_configs.is_sparse
            and attn_configs.use_mla
            and attn_configs.kv_cache_dtype == KvCacheDataType.FP8
            and (not attn_inputs.is_prefill or _is_multi_token_decode(attn_inputs))
            and attn_configs.head_num in (8, 16, 32, 64)
            and attn_configs.kv_lora_rank == 512
            and attn_configs.nope_head_dim == 192
            and attn_configs.rope_head_dim == 64
            and attn_configs.kernel_tokens_per_block == 64
            and attn_configs.indexer_topk == 2048
        )
