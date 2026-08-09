"""MLA decode implementation backed by TokenSpeed's CuTe DSL kernel."""

import logging
import os
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferImplBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.paged_mla_decode import (
    MLA_DECODE_KERNEL_ENV,
    AbsorbedPagedMlaDecodeOp,
    PagedMlaDecodeImplMixin,
    PagedMlaDecodeMetadata,
    get_mla_decode_kernel,
)
from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs

from .rope_emb_new import NewMlaRotaryEmbeddingOp

_TOKENSPEED_MLA_API = None
_TOKENSPEED_GET_NUM_SM = None
_TOKENSPEED_IMPORT_ERROR: Optional[BaseException] = None


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


if os.environ.get(MLA_DECODE_KERNEL_ENV, "auto") == "tokenspeed_mla":
    try:
        _ensure_tokenspeed_cutlass_compat()
        from tokenspeed_mla.mla_decode import tokenspeed_mla_decode
        from tokenspeed_mla.utils import get_num_sm

        _TOKENSPEED_MLA_API = tokenspeed_mla_decode
        _TOKENSPEED_GET_NUM_SM = get_num_sm
    except ImportError as e:  # pragma: no cover - depends on deployment wheel
        _TOKENSPEED_IMPORT_ERROR = e
        logging.info("TokenSpeed MLA decode is unavailable: %s", e)


_TOKENSPEED_MAX_Q_LEN = 4
_g_tokenspeed_workspaces: Dict[tuple[int, int, int, int], torch.Tensor] = {}
_g_tokenspeed_warmup_keys: set[tuple[int, int, int, int, int, int, int]] = set()


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _is_tokenspeed_blackwell(device: Optional[torch.device] = None) -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device)[0] == 10


def tokenspeed_mla_kernel_supported(
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
) -> bool:
    """Check the static shape contract exposed by TokenSpeed MLA decode."""
    return (
        0 < num_heads <= 128
        and kv_lora_rank == 512
        and qk_rope_head_dim == 64
        and page_size > 1
        and 128 % page_size == 0
    )


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


class TokenSpeedMlaDecodeOp(AbsorbedPagedMlaDecodeOp):
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
        max_context_len: int = 0,
        is_cuda_graph: bool = False,
    ) -> None:
        if _TOKENSPEED_MLA_API is None:
            raise RuntimeError(
                "TokenSpeedMlaDecodeOp requires the tokenspeed-mla package"
            ) from _TOKENSPEED_IMPORT_ERROR
        if not tokenspeed_mla_kernel_supported(
            num_heads, kv_lora_rank, qk_rope_head_dim, token_per_block
        ):
            raise ValueError(
                "unsupported TokenSpeed MLA geometry: "
                f"heads={num_heads}, kv_lora_rank={kv_lora_rank}, "
                f"rope_dim={qk_rope_head_dim}, page_size={token_per_block}"
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
            max_q_len=1,
            is_cuda_graph=is_cuda_graph,
        )
        self.backend_name = "tokenspeed_mla"
        self._batch_size = 0
        self._padded_blocks = 0
        self._max_seq_len = 0
        self._max_context_len = max_context_len

        device = torch.device("cuda", torch.cuda.current_device())
        self._device = device
        # TokenSpeed consumes the cache's physical page table directly. Its
        # metadata has one-block alignment and never expands a logical page.
        self._metadata = PagedMlaDecodeMetadata(
            token_per_block,
            token_per_block,
            max_bs,
            max_context_len,
            is_cuda_graph,
            device,
        )
        self._workspace = _get_tokenspeed_workspace(
            device, num_heads, kv_lora_rank, _TOKENSPEED_MAX_Q_LEN
        )
        self._attn_output: Optional[torch.Tensor] = None
        if is_cuda_graph and max_bs > 0:
            self._attn_output = torch.empty(
                (max_bs, num_heads, kv_lora_rank),
                dtype=torch.bfloat16,
                device=device,
            )
            self._warmup(max_bs, max(max_context_len, token_per_block))
        self._sync_metadata_views()

    def _sync_metadata_views(self) -> None:
        self._block_tables = self._metadata.block_tables
        self._seq_lens = self._metadata.seq_lens

    def _warmup(self, batch_size: int, max_seq_len: int) -> None:
        """Compile the exact graph variant before CUDA capture begins."""
        key = (
            _device_index(self._device),
            self.num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.token_per_block,
            batch_size,
            max_seq_len,
        )
        if key in _g_tokenspeed_warmup_keys:
            return
        query = torch.zeros(
            (batch_size, 1, self.num_heads, self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=torch.bfloat16,
            device=self._device,
        )
        kv_cache = torch.zeros(
            (1, self.token_per_block, self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=torch.bfloat16,
            device=self._device,
        )
        block_tables = torch.zeros(
            (batch_size, 1), dtype=torch.int32, device=self._device
        )
        seq_lens = torch.ones(batch_size, dtype=torch.int32, device=self._device)
        output = torch.empty(
            (batch_size, 1, self.num_heads, self.kv_lora_rank),
            dtype=torch.bfloat16,
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
        if not 1 <= q_len <= _TOKENSPEED_MAX_Q_LEN:
            raise RuntimeError(
                f"TokenSpeed MLA supports q_len 1..{_TOKENSPEED_MAX_Q_LEN}, got {q_len}"
            )
        if self.use_cuda_graph and q_len != 1:
            raise RuntimeError("TokenSpeed MLA CUDA Graph currently requires q_len=1")

        paged_kv = self._view_paged_kv(kv_cache)
        if paged_kv.dtype != q_absorbed.dtype:
            raise RuntimeError(
                f"TokenSpeed MLA query/KV dtype mismatch: {q_absorbed.dtype} vs "
                f"{paged_kv.dtype}"
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


class TokenSpeedMlaDecodeImpl(PagedMlaDecodeImplMixin, MlaFlashInferImplBase):
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

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        if get_mla_decode_kernel() != "tokenspeed_mla":
            return False
        if _TOKENSPEED_MLA_API is None:
            raise RuntimeError(
                "RTP_MLA_DECODE_KERNEL=tokenspeed_mla requires tokenspeed-mla"
            ) from _TOKENSPEED_IMPORT_ERROR
        if attn_inputs.is_prefill:
            return False
        if not attn_configs.use_mla:
            raise RuntimeError(
                "RTP_MLA_DECODE_KERNEL=tokenspeed_mla requires dense MLA attention"
            )
        if attn_configs.is_sparse:
            raise RuntimeError(
                "RTP_MLA_DECODE_KERNEL=tokenspeed_mla does not support sparse MLA"
            )
        if attn_configs.kv_cache_dtype != KvCacheDataType.BASE:
            raise RuntimeError(
                "RTP_MLA_DECODE_KERNEL=tokenspeed_mla requires BASE KV cache, "
                f"got {attn_configs.kv_cache_dtype}"
            )
        if not tokenspeed_mla_kernel_supported(
            attn_configs.head_num,
            attn_configs.kv_lora_rank,
            attn_configs.rope_head_dim,
            attn_configs.kernel_tokens_per_block,
        ):
            raise RuntimeError(
                "RTP_MLA_DECODE_KERNEL=tokenspeed_mla does not support geometry "
                f"heads={attn_configs.head_num}, "
                f"kv_lora_rank={attn_configs.kv_lora_rank}, "
                f"rope_dim={attn_configs.rope_head_dim}, "
                f"page_size={attn_configs.kernel_tokens_per_block}"
            )
        if not _is_tokenspeed_blackwell():
            raise RuntimeError(
                "RTP_MLA_DECODE_KERNEL=tokenspeed_mla requires SM100 or SM103"
            )
        return True
