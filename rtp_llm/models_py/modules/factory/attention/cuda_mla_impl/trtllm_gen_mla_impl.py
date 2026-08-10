"""MLA decode implementation backed by the FlashInfer trtllm-gen kernel.

The trtllm-gen kernel consumes the absorbed query ([..., kv_lora_rank +
rope_head_dim]) and the unified MLA KV cache layout ([num_pages, page_size,
kv_lora_rank + rope_head_dim]) directly, which matches the RTP-LLM MLA cache
without any layout conversion. On Blackwell (sm100/sm103) FlashInfer
dispatches this API to the trtllm-gen MLA decode kernel, which is
significantly faster than the FlashInfer fa2 MLA decode kernel for small and
medium batches.

The trtllm-gen dispatcher requires the local query-head count to fit within
one selected Q tile or be an exact multiple of that tile. This adapter mirrors
the upstream selection rules and pads the head dimension to the smallest
executable count when needed, then slices the output back to the model's head
count. It never silently falls back to FlashInfer FA2 under the ``trtllm_gen``
backend name. Kimi K3 TP8 supplies 12 local heads and needs no padding; TP4
supplies 24 and is padded to 32.

The implementation reuses MlaFlashInferImplBase for RoPE, KV cache writes and
cache store publishing; only the attention operator differs from the default
FlashInfer decode path.
"""

import logging
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferImplBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.paged_mla_decode import (
    AbsorbedPagedMlaDecodeOp,
    PagedMlaDecodeImplMixin,
    PagedMlaDecodeMetadata,
    get_mla_decode_kernel,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs

from .rope_emb_new import NewMlaRotaryEmbeddingOp

_TRTLLM_MLA_API = None
try:
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla as _TRTLLM_MLA_API
except ImportError as e:  # pragma: no cover - depends on flashinfer version
    logging.warning(f"flashinfer trtllm-gen MLA API not available: {e}")

# The trtllm-gen kernel requires the per-sequence block count to be a
# multiple of this token granularity.
_TRTLLM_BLOCK_ALIGNMENT_TOKENS = 128

# trtllm_gen_counter_workspace needs >= 8MB; keep a comfortable margin.
_TRTLLM_WORKSPACE_BYTES = 64 * 1024 * 1024

# The trtllm-gen KV tile step used by its multi-CTA utilization heuristic.
_TRTLLM_STEP_KV = 256

# Above this per-CTA KV length the trtllm-gen dispatcher leaves the
# SwapsMmaAb kernel family (see useSwapsMmaAbMlaGenKernel in flashinfer's
# fmhaKernels.cuh).
_TRTLLM_SWAPS_MAX_SEQ_PER_CTA_KV = 1024

_g_trtllm_workspaces: Dict[int, torch.Tensor] = {}
_g_trtllm_warmup_keys: set[tuple[int, int, int, int, int]] = set()


def _get_trtllm_workspace(device: torch.device) -> torch.Tensor:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    workspace = _g_trtllm_workspaces.get(device_index)
    if workspace is None:
        workspace = torch.zeros(
            _TRTLLM_WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
        _g_trtllm_workspaces[device_index] = workspace
    return workspace


def _is_blackwell(device: Optional[torch.device] = None) -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major == 10


def _trtllm_gen_uses_swaps_kernel(
    num_heads: int, batch_size: int, max_seq_len: int, num_sms: int
) -> bool:
    if num_heads <= 32:
        return True
    # One CTA per token covers ceil(num_heads / 16) head tiles.
    num_ctas = batch_size * ((num_heads + 15) // 16)
    max_ctas_per_seq_kv = (max_seq_len + _TRTLLM_STEP_KV - 1) // _TRTLLM_STEP_KV
    ctas_per_seq_kv = min(max_ctas_per_seq_kv, max(1, num_sms // num_ctas))
    seq_len_per_cta_kv = (max_seq_len + ctas_per_seq_kv - 1) // ctas_per_seq_kv
    return (
        seq_len_per_cta_kv <= _TRTLLM_SWAPS_MAX_SEQ_PER_CTA_KV and num_ctas <= num_sms
    )


def _trtllm_gen_head_count_supported(
    num_heads: int, batch_size: int, max_seq_len: int, num_sms: int
) -> bool:
    if (
        num_heads <= 0
        or num_heads > 128
        or batch_size <= 0
        or max_seq_len <= 0
        or num_sms <= 0
    ):
        return False
    if _trtllm_gen_uses_swaps_kernel(
        num_heads, batch_size, max_seq_len, num_sms
    ):
        tile_size_q = 8 if num_heads <= 8 else 16
    else:
        tile_size_q = 64
    heads_per_cta = min(num_heads, tile_size_q)
    return num_heads % heads_per_cta == 0


def trtllm_gen_dispatch_num_heads(
    num_heads: int, batch_size: int, max_seq_len: int, num_sms: int
) -> Optional[int]:
    """Return the smallest executable trtllm-gen query-head count.

    The upstream dispatcher picks an 8-, 16- or 64-head Q tile and rejects a
    head count larger than the tile when it is not a multiple of the tile.
    Padding with zero-valued query heads is semantically neutral because MLA
    heads are independent until the output projection, where the padded
    results are discarded.
    """
    if (
        num_heads <= 0
        or num_heads > 128
        or batch_size <= 0
        or max_seq_len <= 0
        or num_sms <= 0
    ):
        return None
    for dispatch_heads in range(num_heads, 129):
        if _trtllm_gen_head_count_supported(
            dispatch_heads, batch_size, max_seq_len, num_sms
        ):
            return dispatch_heads
    return None


def trtllm_gen_kernel_supported(
    num_heads: int, batch_size: int, max_seq_len: int, num_sms: int
) -> bool:
    """Whether the adapter can execute this shape, including head padding."""
    return (
        trtllm_gen_dispatch_num_heads(
            num_heads, batch_size, max_seq_len, num_sms
        )
        is not None
    )


class TrtllmGenMlaDecodeOp(AbsorbedPagedMlaDecodeOp):
    """Decode attention operator that always executes trtllm-gen."""

    def __init__(
        self,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        token_per_block: int,
        softmax_extra_scale: float,
        weights: List[Dict[str, torch.Tensor]],
        max_bs: int = 0,
        max_context_len: int = 0,
        is_cuda_graph: bool = False,
    ):
        if _TRTLLM_MLA_API is None:
            raise RuntimeError(
                "TrtllmGenMlaDecodeOp requires flashinfer.mla."
                "trtllm_batch_decode_with_kv_cache_mla"
            )
        super().__init__(
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            qk_nope_head_dim,
            token_per_block,
            softmax_extra_scale,
            weights,
            max_bs=max_bs,
            is_cuda_graph=is_cuda_graph,
        )
        self.backend_name = "trtllm_gen"

        self._batch_size = 0
        self._padded_blocks = 0
        self._max_seq_len = 0
        self._max_context_len = max_context_len
        self._kernel = "trtllm"
        self._dispatch_num_heads = num_heads

        device = torch.device("cuda", torch.cuda.current_device())
        self._device = device
        self._num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self._workspace: Optional[torch.Tensor] = None
        self._query_padded: Optional[torch.Tensor] = None
        self._attn_output: Optional[torch.Tensor] = None

        self._metadata = PagedMlaDecodeMetadata(
            token_per_block,
            _TRTLLM_BLOCK_ALIGNMENT_TOKENS,
            max_bs,
            max_context_len,
            is_cuda_graph,
            device,
        )
        self._sync_metadata_views()

        if is_cuda_graph and max_bs > 0:
            assert self._metadata.block_tables is not None
            padded_blocks = self._metadata.block_tables.size(1)
            dispatch_num_heads = trtllm_gen_dispatch_num_heads(
                num_heads, max_bs, max(max_context_len, token_per_block), self._num_sms
            )
            if dispatch_num_heads is None:
                raise RuntimeError(
                    "trtllm-gen MLA does not support the CUDA Graph capture shape: "
                    f"heads={num_heads}, batch={max_bs}, "
                    f"max_seq_len={max_context_len}, sms={self._num_sms}"
                )
            self._set_dispatch_num_heads(dispatch_num_heads)
            self._ensure_trtllm_buffers(max_bs, padded_blocks, device)
            self._ensure_trtllm_ready()
            self._attn_output = torch.empty(
                (max_bs, self._dispatch_num_heads, kv_lora_rank),
                dtype=torch.bfloat16,
                device=device,
            )
            if self._dispatch_num_heads != self.num_heads:
                self._query_padded = torch.empty(
                    (
                        max_bs,
                        self._dispatch_num_heads,
                        kv_lora_rank + qk_rope_head_dim,
                    ),
                    dtype=torch.bfloat16,
                    device=device,
                )

    def _align_blocks(self, num_blocks: int) -> int:
        return self._metadata.align_blocks(num_blocks)

    def _sync_metadata_views(self) -> None:
        """Keep compatibility attributes while metadata ownership stays shared."""
        self._block_tables = self._metadata.block_tables
        self._seq_lens = self._metadata.seq_lens
        self._graph_col_idx = self._metadata.column_indices

    def _ensure_trtllm_buffers(
        self, batch_size: int, padded_blocks: int, device: torch.device
    ) -> None:
        if device != self._metadata.device:
            raise ValueError(
                f"trtllm-gen metadata device changed from {self._metadata.device} "
                f"to {device}"
            )
        self._metadata.ensure_capacity(batch_size, padded_blocks)
        self._sync_metadata_views()

    def _set_dispatch_num_heads(self, dispatch_num_heads: int) -> None:
        if dispatch_num_heads == self._dispatch_num_heads:
            return
        if self.use_cuda_graph and self._attn_output is not None:
            raise RuntimeError(
                "trtllm-gen dispatch head count cannot change under CUDA Graph"
            )
        self._dispatch_num_heads = dispatch_num_heads
        self._query_padded = None
        self._attn_output = None

    def _ensure_trtllm_ready(self) -> None:
        if self._workspace is None:
            self._workspace = _get_trtllm_workspace(self._device)
        device_index = self._device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        warmup_key = (
            device_index,
            self._dispatch_num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.token_per_block,
        )
        if warmup_key not in _g_trtllm_warmup_keys:
            self._warmup(self._device)
            _g_trtllm_warmup_keys.add(warmup_key)

    def _warmup(self, device: torch.device) -> None:
        # The first call loads cubins and initializes JIT state; it must
        # happen before any CUDA graph capture that includes this op.
        warm_blocks = self._align_blocks(1)
        kv_cache = torch.zeros(
            (
                warm_blocks,
                self.token_per_block,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        query = torch.zeros(
            (
                1,
                1,
                self._dispatch_num_heads,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        block_tables = torch.zeros((1, warm_blocks), dtype=torch.int32, device=device)
        seq_lens = torch.ones(1, dtype=torch.int32, device=device)
        assert self._workspace is not None
        _TRTLLM_MLA_API(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self._workspace,
            qk_nope_head_dim=self.kv_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=self.token_per_block,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=1.0,
            backend="trtllm-gen",
        )

    def plan(self, fmha_params: Any) -> None:
        """Validate the current shape and build padded trtllm-gen metadata."""
        batch_size = fmha_params.qo_indptr_h.numel() - 1
        kv_lens = fmha_params.kvlen_h.tolist()
        max_seq_len = max(kv_lens) if kv_lens else 0
        dispatch_max_seq_len = (
            max(self._max_context_len, self.token_per_block)
            if self.use_cuda_graph
            else max_seq_len
        )
        if self.use_cuda_graph and self._attn_output is not None:
            # Graph buffers are allocated from the maximum capture shape in
            # __init__. A smaller capture batch can select a different kernel
            # family, but the padded head extent must remain graph-static.
            dispatch_num_heads = self._dispatch_num_heads
            dispatch_supported = _trtllm_gen_head_count_supported(
                dispatch_num_heads,
                batch_size,
                dispatch_max_seq_len,
                self._num_sms,
            )
        else:
            dispatch_num_heads = trtllm_gen_dispatch_num_heads(
                self.num_heads, batch_size, dispatch_max_seq_len, self._num_sms
            )
            dispatch_supported = dispatch_num_heads is not None
        if not dispatch_supported or dispatch_num_heads is None:
            raise RuntimeError(
                "trtllm-gen MLA does not support the decode shape: "
                f"heads={self.num_heads}, batch={batch_size}, "
                f"max_seq_len={dispatch_max_seq_len}, sms={self._num_sms}"
            )
        self._set_dispatch_num_heads(dispatch_num_heads)
        self._kernel = "trtllm"
        self._ensure_trtllm_ready()
        self._metadata.plan(fmha_params)
        self._sync_metadata_views()
        self._batch_size = self._metadata.batch_size
        self._padded_blocks = self._metadata.padded_blocks
        self._max_seq_len = self._metadata.max_seq_len

    def _refresh_graph_block_tables(
        self, block_table: torch.Tensor, sequence_lengths_plus_1: torch.Tensor
    ) -> None:
        """Rebuild the padded block tables on device for one HybridCache group.

        CUDA graph capture forbids the host-side planning path, and every K3
        FULL group shares the batch shape and KV lengths, so only the block
        table content changes between groups. All shapes below are static,
        which keeps the masked copy replay-safe inside the captured graph.
        """
        self._metadata.refresh_cuda_graph(block_table, sequence_lengths_plus_1)

    def forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
    ) -> torch.Tensor:
        if self._batch_size <= 0:
            raise RuntimeError("trtllm-gen MLA plan() must be called before forward")
        q_absorbed = self._absorb_query(q_nope, q_pe, layer_id)
        num_tokens = q_absorbed.size(0)
        if num_tokens != self._batch_size:
            raise RuntimeError(
                "trtllm-gen MLA decode requires one query token per request: "
                f"tokens={num_tokens}, batch={self._batch_size}"
            )
        paged_kv = self._view_paged_kv(kv_cache)
        if q_absorbed.dtype != torch.bfloat16 or paged_kv.dtype != torch.bfloat16:
            raise RuntimeError(
                "trtllm-gen MLA requires BF16 query and KV cache, got "
                f"query={q_absorbed.dtype}, kv={paged_kv.dtype}"
            )
        attn_output = self._forward_trtllm(q_absorbed, paged_kv, num_tokens)
        return self._project_output(attn_output, layer_id)

    def refresh_cuda_graph_metadata(
        self,
        fmha_params: Any,
        block_table: torch.Tensor,
        sequence_lengths: torch.Tensor,
        seq_size_per_block: int,
    ) -> None:
        if seq_size_per_block != self.token_per_block:
            raise RuntimeError(
                f"trtllm-gen MLA page-size mismatch: impl={self.token_per_block}, "
                f"runtime={seq_size_per_block}"
            )
        del fmha_params
        self._refresh_graph_block_tables(block_table, sequence_lengths)

    def _ensure_output(self, num_tokens: int) -> torch.Tensor:
        if (
            self._attn_output is None
            or self._attn_output.size(0) < num_tokens
            or self._attn_output.size(1) != self._dispatch_num_heads
        ):
            if self.use_cuda_graph and self._attn_output is not None:
                raise RuntimeError(
                    "trtllm-gen MLA output buffer cannot grow under CUDA Graph"
                )
            self._attn_output = torch.empty(
                (num_tokens, self._dispatch_num_heads, self.kv_lora_rank),
                dtype=torch.bfloat16,
                device=self._device,
            )
        return self._attn_output[:num_tokens]

    def _pad_query_heads(self, query: torch.Tensor) -> torch.Tensor:
        if self._dispatch_num_heads == self.num_heads:
            return query
        num_tokens = query.size(0)
        if (
            self._query_padded is None
            or self._query_padded.size(0) < num_tokens
            or self._query_padded.size(1) != self._dispatch_num_heads
            or self._query_padded.dtype != query.dtype
            or self._query_padded.device != query.device
        ):
            if self.use_cuda_graph and self._query_padded is not None:
                raise RuntimeError(
                    "trtllm-gen padded query buffer cannot grow or change "
                    "dtype/device under CUDA Graph"
                )
            self._query_padded = torch.empty(
                (
                    num_tokens,
                    self._dispatch_num_heads,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ),
                dtype=query.dtype,
                device=query.device,
            )
        padded = self._query_padded[:num_tokens]
        padded.zero_()
        padded[:, : self.num_heads].copy_(query)
        return padded

    def _forward_trtllm(
        self, q_absorbed: torch.Tensor, paged_kv: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        # Under CUDA graph the value is baked into the captured graph while
        # kv lengths keep growing across replays, so use the static upper
        # bound; eager mode can pass the tight per-step maximum.
        max_seq_len = (
            max(self._max_context_len, self._max_seq_len)
            if self.use_cuda_graph
            else self._max_seq_len
        )
        assert self._workspace is not None
        query = self._pad_query_heads(q_absorbed)
        output = self._ensure_output(num_tokens)
        result = _TRTLLM_MLA_API(
            query=query.view(
                self._batch_size,
                1,
                self._dispatch_num_heads,
                query.size(-1),
            ),
            kv_cache=paged_kv,
            workspace_buffer=self._workspace,
            qk_nope_head_dim=self.kv_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=self._block_tables[: self._batch_size, : self._padded_blocks],
            seq_lens=self._seq_lens[: self._batch_size],
            max_seq_len=max_seq_len,
            out=output.view(
                self._batch_size,
                1,
                self._dispatch_num_heads,
                self.kv_lora_rank,
            ),
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=1.0,
            backend="trtllm-gen",
        )
        return result.view(
            self._batch_size, self._dispatch_num_heads, self.kv_lora_rank
        )[:, : self.num_heads]


class TrtllmGenMlaDecodeImpl(PagedMlaDecodeImplMixin, MlaFlashInferImplBase):
    """MLA decode impl selecting the FlashInfer trtllm-gen kernel."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        if attn_inputs.sequence_lengths.numel() > 0:
            max_bs = attn_inputs.sequence_lengths.size(0)
        else:
            max_bs = 0
        super().__init__(
            TrtllmGenMlaDecodeOp(
                attn_configs.head_num,
                attn_configs.kv_lora_rank,
                attn_configs.rope_head_dim,
                attn_configs.nope_head_dim,
                attn_configs.kernel_tokens_per_block,
                attn_configs.softmax_extra_scale,
                weights,
                max_bs=max_bs,
                max_context_len=max_seq_len,
                is_cuda_graph=is_cuda_graph,
            ),
            NewMlaRotaryEmbeddingOp(
                cos_sin_cache=cos_sin_cache,
                is_neox_style=attn_configs.rope_config.is_neox_style,
            ),
            MlaKVCacheWriteOp(
                kv_cache_dtype=attn_configs.kv_cache_dtype,
                clear_page_on_boundary=is_cuda_graph,
            ),
            attn_inputs,
            attn_configs.kernel_tokens_per_block,
            attn_configs,
            weights,
            cos_sin_cache,
            fmha_config,
            use_trt_fmha=False,
            quant_config=quant_config,
            max_seq_len=max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism_config,
        )

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        if get_mla_decode_kernel() != "trtllm_gen":
            return False
        if _TRTLLM_MLA_API is None:
            raise RuntimeError(
                "RTP_MLA_DECODE_KERNEL=trtllm_gen requires "
                "flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla"
            )
        if attn_inputs.is_prefill:
            return False
        if not attn_configs.use_mla:
            raise RuntimeError(
                "RTP_MLA_DECODE_KERNEL=trtllm_gen requires dense MLA attention"
            )
        if attn_configs.is_sparse:
            raise RuntimeError(
                "RTP_MLA_DECODE_KERNEL=trtllm_gen does not support sparse MLA"
            )
        if attn_configs.kv_cache_dtype != KvCacheDataType.BASE:
            raise RuntimeError(
                "RTP_MLA_DECODE_KERNEL=trtllm_gen requires BASE KV cache, "
                f"got {attn_configs.kv_cache_dtype}"
            )
        if (
            attn_configs.kv_lora_rank != 512
            or attn_configs.rope_head_dim != 64
            or not 0 < attn_configs.head_num <= 128
        ):
            raise RuntimeError(
                "RTP_MLA_DECODE_KERNEL=trtllm_gen does not support geometry "
                f"heads={attn_configs.head_num}, "
                f"kv_lora_rank={attn_configs.kv_lora_rank}, "
                f"rope_dim={attn_configs.rope_head_dim}"
            )
        page_size = attn_configs.kernel_tokens_per_block
        if page_size not in (32, 64):
            raise RuntimeError(
                "RTP_MLA_DECODE_KERNEL=trtllm_gen supports page sizes 32 or 64, "
                f"got {page_size}"
            )
        if not _is_blackwell():
            raise RuntimeError(
                "RTP_MLA_DECODE_KERNEL=trtllm_gen requires SM100 or SM103"
            )
        return True

    # prepare() is inherited from MlaFlashInferImplBase: it fills
    # FlashInferMlaAttnParams and calls fmha_impl.plan(), which this op
    # reuses to build its padded block tables and sequence lengths.
