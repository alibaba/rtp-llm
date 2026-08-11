"""MLA decode implementation backed by TokenSpeed's CuTe DSL kernel."""

import logging
import os
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferImplBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W

from .rope_emb_new import NewMlaRotaryEmbeddingOp

MLA_DECODE_KERNEL_ENV = "RTP_MLA_DECODE_KERNEL"
_MLA_DECODE_KERNELS = ("auto", "flashinfer", "tokenspeed_mla")


def _get_mla_decode_kernel() -> str:
    kernel = os.environ.get(MLA_DECODE_KERNEL_ENV, "auto")
    if kernel not in _MLA_DECODE_KERNELS:
        supported = ", ".join(_MLA_DECODE_KERNELS)
        raise RuntimeError(
            f"invalid {MLA_DECODE_KERNEL_ENV}={kernel!r}; expected one of: {supported}"
        )
    return kernel


class _TokenSpeedDecodeMetadata:
    """Stable block-table and sequence-length buffers for TokenSpeed MLA kernels."""

    def __init__(
        self,
        token_per_block: int,
        max_bs: int,
        max_context_len: int,
        use_cuda_graph: bool,
        device: torch.device,
    ) -> None:
        if token_per_block <= 0:
            raise ValueError(f"token_per_block must be positive, got {token_per_block}")

        self.token_per_block = token_per_block
        self.max_context_len = max_context_len
        self.use_cuda_graph = use_cuda_graph
        self.device = device

        self.block_tables: Optional[torch.Tensor] = None
        self.seq_lens: Optional[torch.Tensor] = None
        self.column_indices: Optional[torch.Tensor] = None
        self.batch_size = 0
        self.padded_blocks = 0
        self.max_seq_len = 0

        if use_cuda_graph and max_bs > 0:
            max_blocks = max(
                1,
                (max_context_len + token_per_block - 1) // token_per_block,
            )
            self.ensure_capacity(max_bs, max_blocks)

    def ensure_capacity(self, batch_size: int, padded_blocks: int) -> None:
        if (
            self.block_tables is None
            or self.block_tables.size(0) < batch_size
            or self.block_tables.size(1) < padded_blocks
        ):
            if self.use_cuda_graph and self.block_tables is not None:
                raise ValueError(
                    "TokenSpeed MLA decode metadata cannot grow under CUDA graph: "
                    f"need ({batch_size}, {padded_blocks}), have "
                    f"{tuple(self.block_tables.shape)}"
                )
            self.block_tables = torch.zeros(
                (batch_size, padded_blocks), dtype=torch.int32, device=self.device
            )
            self.seq_lens = torch.zeros(
                batch_size, dtype=torch.int32, device=self.device
            )
        if (
            self.column_indices is None
            or self.column_indices.numel() < padded_blocks
            or self.column_indices.device != self.device
        ):
            if self.use_cuda_graph and self.column_indices is not None:
                raise ValueError(
                    "TokenSpeed MLA decode column metadata cannot grow under CUDA graph"
                )
            self.column_indices = torch.arange(padded_blocks, device=self.device)

    def plan(self, fmha_params: Any) -> None:
        """Materialize dense block tables from RTP's compact FlashInfer metadata."""
        batch_size = fmha_params.qo_indptr_h.numel() - 1
        kv_lens = fmha_params.kvlen_h.tolist()
        max_seq_len = max(kv_lens) if kv_lens else 0
        needed_blocks = max(
            1, (max_seq_len + self.token_per_block - 1) // self.token_per_block
        )

        if self.use_cuda_graph:
            if self.block_tables is None or self.block_tables.size(0) < batch_size:
                raise ValueError(
                    f"TokenSpeed MLA graph metadata is too small for batch {batch_size}"
                )
            width = self.block_tables.size(1)
            if width < needed_blocks:
                raise ValueError(
                    f"TokenSpeed MLA graph metadata needs {needed_blocks} blocks, has {width}"
                )
        else:
            self.ensure_capacity(batch_size, needed_blocks)
            width = needed_blocks

        assert self.block_tables is not None
        assert self.seq_lens is not None
        assert self.column_indices is not None
        page_indices = fmha_params.page_indice_d
        page_indptr = fmha_params.decode_page_indptr_d
        row_starts = page_indptr[:batch_size].view(-1, 1)
        row_sizes = (page_indptr[1 : batch_size + 1] - page_indptr[:batch_size]).view(
            -1, 1
        )
        columns = self.column_indices[:width].view(1, -1)
        source_indices = row_starts + columns
        dense_tables = self.block_tables[:batch_size, :width]
        if page_indices.numel() == 0:
            dense_tables.zero_()
        else:
            source_indices = source_indices.clamp_max(page_indices.numel() - 1)
            dense_tables.copy_(page_indices[source_indices])
            dense_tables.masked_fill_(columns >= row_sizes, 0)
        self.seq_lens[:batch_size].copy_(fmha_params.kvlen_d)
        self.batch_size = batch_size
        self.padded_blocks = width
        self.max_seq_len = max_seq_len

    def refresh_cuda_graph(
        self, block_table: torch.Tensor, sequence_lengths: torch.Tensor
    ) -> None:
        """Refresh captured metadata in place from the selected cache group."""
        if not self.use_cuda_graph:
            raise RuntimeError(
                "TokenSpeed MLA graph metadata refresh requires CUDA graph"
            )
        assert self.block_tables is not None
        assert self.seq_lens is not None
        assert self.column_indices is not None
        batch_size = self.batch_size
        width = self.padded_blocks
        src = block_table[:batch_size]
        if src.dim() != 2 or src.size(1) < width:
            raise RuntimeError(
                "TokenSpeed MLA group refresh needs a block table of width "
                f">= {width}, got {tuple(src.shape)}"
            )
        kv_lens = sequence_lengths[:batch_size].to(torch.int32)
        live_blocks = (kv_lens + self.token_per_block - 1) // self.token_per_block
        dense_tables = self.block_tables[:batch_size, :width]
        dense_tables.copy_(src[:, :width])
        dense_tables.masked_fill_(
            self.column_indices[:width].view(1, -1) >= live_blocks.view(-1, 1),
            0,
        )
        self.seq_lens[:batch_size].copy_(kv_lens)


_TOKENSPEED_MLA_API = None
_TOKENSPEED_GET_NUM_SM = None
_TOKENSPEED_CAN_IMPLEMENT: Optional[Callable[..., None]] = None
_TOKENSPEED_IMPORT_ERROR: Optional[BaseException] = None
_TOKENSPEED_IMPORT_ATTEMPTED = False


def _ensure_tokenspeed_cutlass_compat() -> None:
    """Bridge the CuTe DSL 4.4 namespace used by TokenSpeed 0.2.3."""
    import cutlass.cute.nvgpu as nvgpu

    if hasattr(nvgpu, "OperandMajorMode"):
        return
    tcgen05 = getattr(nvgpu, "tcgen05", None)
    operand_major_mode = getattr(tcgen05, "OperandMajorMode", None)
    if operand_major_mode is None:
        raise ImportError(
            "tokenspeed-mla requires cutlass.cute.nvgpu.OperandMajorMode or "
            "cutlass.cute.nvgpu.tcgen05.OperandMajorMode"
        )
    # TokenSpeed 0.2.3 imports this symbol from the pre-4.4 namespace.
    nvgpu.OperandMajorMode = operand_major_mode


def _load_tokenspeed_mla() -> bool:
    """Load the optional backend without making it mandatory on other GPUs."""
    global _TOKENSPEED_MLA_API
    global _TOKENSPEED_GET_NUM_SM
    global _TOKENSPEED_CAN_IMPLEMENT
    global _TOKENSPEED_IMPORT_ERROR
    global _TOKENSPEED_IMPORT_ATTEMPTED

    if _TOKENSPEED_MLA_API is not None:
        return True
    if _TOKENSPEED_IMPORT_ATTEMPTED:
        return False
    _TOKENSPEED_IMPORT_ATTEMPTED = True
    try:
        _ensure_tokenspeed_cutlass_compat()
        decode_module = import_module("tokenspeed_mla.mla_decode")
        utils_module = import_module("tokenspeed_mla.utils")
        _TOKENSPEED_MLA_API = decode_module.tokenspeed_mla_decode
        _TOKENSPEED_GET_NUM_SM = utils_module.get_num_sm
        # TokenSpeed 0.2.3 performs this exact check before every launch but
        # does not export a public capability function yet. Prefer the public
        # spelling once the dependency provides it; keep the pinned-version
        # fallback here instead of duplicating its geometry rules in RTP.
        _TOKENSPEED_CAN_IMPLEMENT = getattr(
            decode_module,
            "tokenspeed_mla_decode_can_implement",
            getattr(decode_module, "_check_can_implement", None),
        )
        if _TOKENSPEED_CAN_IMPLEMENT is None:
            raise ImportError(
                "tokenspeed-mla does not expose a decode capability check"
            )
    except (ImportError, AttributeError) as e:  # pragma: no cover - deployment wheel
        _TOKENSPEED_IMPORT_ERROR = e
        logging.info("TokenSpeed MLA decode is unavailable: %s", e)
        _TOKENSPEED_MLA_API = None
        _TOKENSPEED_GET_NUM_SM = None
        _TOKENSPEED_CAN_IMPLEMENT = None
        return False
    return True


_g_tokenspeed_workspaces: Dict[tuple[int, int, int, int], torch.Tensor] = {}
_g_tokenspeed_warmup_keys: set[tuple[int, int, int, int, int, int, int, int]] = set()


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _tokenspeed_compute_capability(
    device: Optional[torch.device] = None,
) -> Optional[tuple[int, int]]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(device)


def _is_tokenspeed_blackwell(device: Optional[torch.device] = None) -> bool:
    return _tokenspeed_compute_capability(device) in ((10, 0), (10, 3))


def tokenspeed_mla_kernel_supported(
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    q_len: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> bool:
    """Delegate the complete geometry check to the pinned TokenSpeed package."""
    compute_capability = _tokenspeed_compute_capability(device)
    if compute_capability not in ((10, 0), (10, 3)):
        return False
    if not _load_tokenspeed_mla() or _TOKENSPEED_CAN_IMPLEMENT is None:
        return False
    try:
        _TOKENSPEED_CAN_IMPLEMENT(
            torch_dtype=dtype,
            page_size=page_size,
            num_heads=num_heads,
            seq_len_q=q_len,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            is_persistent=False,
            is_var_seq=True,
            is_var_split_kv=False,
            compute_capability=compute_capability,
        )
    except (AssertionError, TypeError, ValueError):
        return False
    return True


def _uniform_decode_q_len(attn_inputs: PyAttentionInputs) -> Optional[int]:
    """Return the rectangular decode q_len without reading device metadata."""
    input_lengths = getattr(attn_inputs, "input_lengths_host", None)
    if input_lengths is None or not input_lengths.numel():
        input_lengths = getattr(attn_inputs, "input_lengths", None)
        if input_lengths is None or not input_lengths.numel() or input_lengths.is_cuda:
            return 1
    values = [int(value) for value in input_lengths.tolist()]
    if not values:
        return 1
    if values[0] <= 0 or any(value != values[0] for value in values[1:]):
        return None
    return values[0]


def _get_tokenspeed_workspace(
    device: torch.device,
    num_heads: int,
    kv_lora_rank: int,
    max_q_len: int,
) -> torch.Tensor:
    """Allocate a split-KV upper bound without tying it to one batch size."""
    assert _TOKENSPEED_GET_NUM_SM is not None
    num_sms = int(_TOKENSPEED_GET_NUM_SM(device))
    key = (_device_index(device), num_heads, kv_lora_rank, max_q_len)
    workspace = _g_tokenspeed_workspaces.get(key)
    # B * split_kv never exceeds the active SM count. Each partial stores one
    # fp32 latent vector plus its normalization scalar per query head.
    required_bytes = num_sms * num_heads * max_q_len * (kv_lora_rank + 1) * 4
    if workspace is None or workspace.numel() < required_bytes:
        workspace = torch.empty(required_bytes, dtype=torch.int8, device=device)
        _g_tokenspeed_workspaces[key] = workspace
    return workspace


class TokenSpeedMlaDecodeOp:
    """Decode attention over RTP's physical paged MLA KV cache."""

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
        max_q_len: int = 1,
        max_context_len: int = 0,
        is_cuda_graph: bool = False,
    ) -> None:
        if not _load_tokenspeed_mla() or _TOKENSPEED_MLA_API is None:
            raise RuntimeError(
                "TokenSpeedMlaDecodeOp requires the tokenspeed-mla package"
            ) from _TOKENSPEED_IMPORT_ERROR
        if not weights or W.mla_kc not in weights[0] or W.mla_vc not in weights[0]:
            raise RuntimeError("TokenSpeed MLA decode requires projection weights")
        if max_q_len <= 0:
            raise ValueError(f"max_q_len must be positive, got {max_q_len}")
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.token_per_block = token_per_block
        self.bmm1_scale = (
            qk_nope_head_dim + qk_rope_head_dim
        ) ** -0.5 * softmax_extra_scale
        self.weights = weights
        self.use_cuda_graph = is_cuda_graph
        self._q_absorbed: Optional[torch.Tensor] = None
        self.backend_name = "tokenspeed_mla"
        self._batch_size = 0
        self._padded_blocks = 0
        self._max_seq_len = 0
        self._max_context_len = max_context_len
        self._max_q_len = max_q_len
        self._dtype = weights[0][W.mla_kc].dtype

        device = torch.device("cuda", torch.cuda.current_device())
        self._device = device
        if not tokenspeed_mla_kernel_supported(
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            token_per_block,
            max_q_len,
            self._dtype,
            device,
        ):
            raise ValueError(
                "unsupported TokenSpeed MLA configuration: "
                f"heads={num_heads}, kv_lora_rank={kv_lora_rank}, "
                f"rope_dim={qk_rope_head_dim}, page_size={token_per_block}, "
                f"q_len={max_q_len}, dtype={self._dtype}"
            )
        if is_cuda_graph and max_bs > 0:
            self._q_absorbed = torch.empty(
                (
                    max_bs * max_q_len,
                    num_heads,
                    kv_lora_rank + qk_rope_head_dim,
                ),
                dtype=self._dtype,
                device=device,
            )
        # TokenSpeed consumes the cache's physical page table directly. Its
        # metadata has one-block alignment and never expands a logical page.
        self._metadata = _TokenSpeedDecodeMetadata(
            token_per_block,
            max_bs,
            max_context_len,
            is_cuda_graph,
            device,
        )
        self._workspace = _get_tokenspeed_workspace(
            device, num_heads, kv_lora_rank, max_q_len
        )
        self._attn_output: Optional[torch.Tensor] = None
        if is_cuda_graph and max_bs > 0:
            self._attn_output = torch.empty(
                (max_bs * max_q_len, num_heads, kv_lora_rank),
                dtype=self._dtype,
                device=device,
            )
            self._warmup(max_bs, max_q_len, max(max_context_len, token_per_block))
        self._sync_metadata_views()

    def _absorb_query(
        self, q_nope: torch.Tensor, q_pe: torch.Tensor, layer_id: int
    ) -> torch.Tensor:
        q_nope = q_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        q_pe = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim)
        num_tokens = q_nope.size(0)
        if (
            self._q_absorbed is None
            or self._q_absorbed.size(0) < num_tokens
            or self._q_absorbed.dtype != q_nope.dtype
            or self._q_absorbed.device != q_nope.device
        ):
            if self.use_cuda_graph and self._q_absorbed is not None:
                raise RuntimeError(
                    "absorbed MLA query buffer cannot grow or change dtype/device "
                    "under CUDA graph"
                )
            self._q_absorbed = torch.empty(
                (
                    num_tokens,
                    self.num_heads,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ),
                dtype=q_nope.dtype,
                device=q_nope.device,
            )
        q_absorbed = self._q_absorbed[:num_tokens]
        q_absorbed[..., self.kv_lora_rank :].copy_(q_pe)
        torch.bmm(
            q_nope.transpose(0, 1),
            self.weights[layer_id][W.mla_kc],
            out=q_absorbed[..., : self.kv_lora_rank].transpose(0, 1),
        )
        return q_absorbed

    def _view_paged_kv(self, kv_cache: Optional[LayerKVCache]) -> torch.Tensor:
        if kv_cache is None:
            raise RuntimeError("absorbed paged MLA decode requires KV cache")
        return kv_cache.kv_cache_base.view(
            -1,
            self.token_per_block,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )

    def _project_output(self, attn_output: torch.Tensor, layer_id: int) -> torch.Tensor:
        attn_output = attn_output.view(-1, self.num_heads, self.kv_lora_rank)
        output = torch.bmm(
            attn_output.transpose(0, 1), self.weights[layer_id][W.mla_vc]
        )
        return output.transpose(0, 1)

    def _sync_metadata_views(self) -> None:
        self._block_tables = self._metadata.block_tables
        self._seq_lens = self._metadata.seq_lens

    def _warmup(self, batch_size: int, q_len: int, max_seq_len: int) -> None:
        """Compile the exact graph variant before CUDA capture begins."""
        key = (
            _device_index(self._device),
            self.num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.token_per_block,
            batch_size,
            q_len,
            max_seq_len,
        )
        if key in _g_tokenspeed_warmup_keys:
            return
        query = torch.zeros(
            (
                batch_size,
                q_len,
                self.num_heads,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ),
            dtype=self._dtype,
            device=self._device,
        )
        kv_cache = torch.zeros(
            (1, self.token_per_block, self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=self._dtype,
            device=self._device,
        )
        block_tables = torch.zeros(
            (batch_size, 1), dtype=torch.int32, device=self._device
        )
        seq_lens = torch.ones(batch_size, dtype=torch.int32, device=self._device)
        output = torch.empty(
            (batch_size, q_len, self.num_heads, self.kv_lora_rank),
            dtype=self._dtype,
            device=self._device,
        )
        _TOKENSPEED_MLA_API(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self._workspace,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            softmax_scale=self.bmm1_scale,
            out=output,
            is_var_seq=True,
            causal_mask=True,
            enable_pdl=False,
        )
        _g_tokenspeed_warmup_keys.add(key)

    def plan(self, fmha_params: Any) -> None:
        self._metadata.plan(fmha_params)
        self._sync_metadata_views()
        self._batch_size = self._metadata.batch_size
        self._padded_blocks = self._metadata.padded_blocks
        self._max_seq_len = self._metadata.max_seq_len

    def refresh_cuda_graph_metadata(
        self,
        fmha_params: Any,
        block_table: torch.Tensor,
        sequence_lengths: torch.Tensor,
        seq_size_per_block: int,
    ) -> None:
        del fmha_params
        if seq_size_per_block != self.token_per_block:
            raise RuntimeError(
                f"TokenSpeed MLA page-size mismatch: impl={self.token_per_block}, "
                f"runtime={seq_size_per_block}"
            )
        self._metadata.refresh_cuda_graph(block_table, sequence_lengths)

    def _ensure_output(self, num_tokens: int, dtype: torch.dtype) -> torch.Tensor:
        if (
            self._attn_output is None
            or self._attn_output.size(0) < num_tokens
            or self._attn_output.dtype != dtype
            or self._attn_output.device != self._device
        ):
            if self.use_cuda_graph and self._attn_output is not None:
                raise RuntimeError(
                    "TokenSpeed MLA output buffer cannot grow or change dtype/device "
                    "under CUDA graph"
                )
            self._attn_output = torch.empty(
                (num_tokens, self.num_heads, self.kv_lora_rank),
                dtype=dtype,
                device=self._device,
            )
        return self._attn_output[:num_tokens]

    def forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
    ) -> torch.Tensor:
        if self._batch_size <= 0:
            raise RuntimeError("plan() must be called before TokenSpeed MLA forward")
        q_absorbed = self._absorb_query(q_nope, q_pe, layer_id)
        num_tokens = q_absorbed.size(0)
        if num_tokens % self._batch_size != 0:
            raise RuntimeError(
                f"TokenSpeed MLA token count {num_tokens} is not divisible by "
                f"batch size {self._batch_size}"
            )
        q_len = num_tokens // self._batch_size
        if q_len <= 0:
            raise RuntimeError(f"TokenSpeed MLA requires positive q_len, got {q_len}")
        if self.use_cuda_graph and q_len != self._max_q_len:
            raise RuntimeError(
                "TokenSpeed MLA CUDA Graph query shape changed after capture: "
                f"captured q_len={self._max_q_len}, runtime q_len={q_len}"
            )
        if not tokenspeed_mla_kernel_supported(
            self.num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.token_per_block,
            q_len,
            q_absorbed.dtype,
            self._device,
        ):
            raise RuntimeError(
                "TokenSpeed MLA rejected the runtime configuration: "
                f"q_len={q_len}, dtype={q_absorbed.dtype}"
            )

        paged_kv = self._view_paged_kv(kv_cache)
        if paged_kv.dtype != q_absorbed.dtype:
            raise RuntimeError(
                f"TokenSpeed MLA query/KV dtype mismatch: {q_absorbed.dtype} vs "
                f"{paged_kv.dtype}"
            )
        if not self.use_cuda_graph and q_len != self._max_q_len:
            self._workspace = _get_tokenspeed_workspace(
                self._device, self.num_heads, self.kv_lora_rank, q_len
            )
        output = self._ensure_output(num_tokens, q_absorbed.dtype)
        max_seq_len = (
            max(self._max_context_len, self.token_per_block)
            if self.use_cuda_graph
            else max(self._max_seq_len, self.token_per_block)
        )
        attn_output = _TOKENSPEED_MLA_API(
            query=q_absorbed.view(
                self._batch_size, q_len, self.num_heads, q_absorbed.size(-1)
            ),
            kv_cache=paged_kv,
            workspace_buffer=self._workspace,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=self._block_tables[: self._batch_size, : self._padded_blocks],
            seq_lens=self._seq_lens[: self._batch_size],
            max_seq_len=max_seq_len,
            softmax_scale=self.bmm1_scale,
            out=output.view(self._batch_size, q_len, self.num_heads, self.kv_lora_rank),
            is_var_seq=True,
            causal_mask=True,
            enable_pdl=False,
        )
        return self._project_output(attn_output, layer_id)


class TokenSpeedMlaDecodeImpl(MlaFlashInferImplBase):
    """RTP attention-framework adapter for TokenSpeed MLA decode."""

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
        max_q_len = _uniform_decode_q_len(attn_inputs)
        if max_q_len is None:
            raise RuntimeError(
                "TokenSpeed MLA decode requires one uniform q_len per batch"
            )
        max_bs = (
            attn_inputs.sequence_lengths.size(0)
            if attn_inputs.sequence_lengths.numel() > 0
            else 0
        )
        super().__init__(
            TokenSpeedMlaDecodeOp(
                attn_configs.head_num,
                attn_configs.kv_lora_rank,
                attn_configs.rope_head_dim,
                attn_configs.nope_head_dim,
                attn_configs.kernel_tokens_per_block,
                attn_configs.softmax_extra_scale,
                weights,
                max_bs=max_bs,
                max_q_len=max_q_len,
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
            warmup_flashinfer=False,
        )

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        self.prepare(attn_inputs, forbid_realloc=True)

    def prepare_cuda_graph_group(self, attn_inputs: PyAttentionInputs) -> None:
        assert self.fmha_impl is not None
        assert self.fmha_params is not None
        self.attn_inputs = attn_inputs
        sequence_lengths = getattr(attn_inputs, "sequence_lengths_plus_1_d", None)
        block_table = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
        if sequence_lengths is None or sequence_lengths.numel() == 0:
            raise RuntimeError(
                "TokenSpeed MLA group refresh requires " "sequence_lengths_plus_1_d"
            )
        if block_table is None or block_table.numel() == 0:
            raise RuntimeError(
                "TokenSpeed MLA group refresh requires a device block table"
            )
        self.fmha_params.fill_decode_cuda_graph_params(
            sequence_lengths,
            block_table,
            self.seq_size_per_block,
        )
        self.fmha_impl.refresh_cuda_graph_metadata(
            self.fmha_params,
            block_table,
            sequence_lengths,
            self.seq_size_per_block,
        )

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        selector = _get_mla_decode_kernel()
        if selector == "flashinfer" or attn_inputs.is_prefill:
            return False

        def unsupported(reason: str) -> bool:
            if selector == "tokenspeed_mla":
                raise RuntimeError(
                    f"RTP_MLA_DECODE_KERNEL=tokenspeed_mla {reason}"
                ) from _TOKENSPEED_IMPORT_ERROR
            logging.info("TokenSpeed MLA auto selection fell back: %s", reason)
            return False

        if not attn_configs.use_mla:
            return unsupported("requires dense MLA attention")
        if attn_configs.is_sparse:
            return unsupported("does not support sparse MLA")
        if attn_configs.kv_cache_dtype != KvCacheDataType.BASE:
            return unsupported(
                f"requires BASE KV cache, got {attn_configs.kv_cache_dtype}"
            )
        if not _is_tokenspeed_blackwell():
            return unsupported("requires SM100 or SM103")
        if not _load_tokenspeed_mla():
            return unsupported("requires the tokenspeed-mla dependency")

        q_len = _uniform_decode_q_len(attn_inputs)
        if q_len is None:
            return unsupported("requires one uniform q_len per batch")
        dtype = getattr(attn_inputs, "dtype", torch.bfloat16)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.bfloat16
        if not tokenspeed_mla_kernel_supported(
            attn_configs.head_num,
            attn_configs.kv_lora_rank,
            attn_configs.rope_head_dim,
            attn_configs.kernel_tokens_per_block,
            q_len,
            dtype,
        ):
            return unsupported(
                "does not support the TokenSpeed configuration "
                f"heads={attn_configs.head_num}, "
                f"kv_lora_rank={attn_configs.kv_lora_rank}, "
                f"rope_dim={attn_configs.rope_head_dim}, "
                f"page_size={attn_configs.kernel_tokens_per_block}, "
                f"q_len={q_len}"
            )
        return True
