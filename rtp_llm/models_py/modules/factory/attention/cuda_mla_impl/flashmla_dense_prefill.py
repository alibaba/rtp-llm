"""Dense causal MLA Prefill backed by FlashMLA's SM100 kernel.

Kimi K3 is a dense MLA model.  RTP's existing FlashMLA wrapper is sparse-only,
so it is not selected for K3 and the generic path falls back to FlashInfer.
This adapter keeps RTP's existing projections, RoPE/no-RoPE handling, cache
write and output projection unchanged; only the dense causal attention core is
replaced.
"""

import logging
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
from rtp_llm.ops import KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache
from rtp_llm.utils.model_weight import W


_FLASHMLA_WORKSPACES: Dict[int, torch.Tensor] = {}
_FLASHMLA_LOGGED_DEVICES: set[int] = set()


def _workspace(device: torch.device) -> torch.Tensor:
    """Return FlashMLA's reusable 32 MiB inference workspace per device."""

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    workspace = _FLASHMLA_WORKSPACES.get(device_index)
    if workspace is None:
        workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        _FLASHMLA_WORKSPACES[device_index] = workspace
    return workspace


class MlaFlashMLAPrefillOp:
    """FlashMLA dense-varlen Prefill core for the generic RTP MLA pipeline."""

    def __init__(
        self,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        page_size: int,
        softmax_extra_scale: float,
        use_mla: bool,
        weights: List[Dict[str, torch.Tensor]] | None,
        quant_config: Optional[object] = None,
        kv_cache_dtype: KvCacheDataType = KvCacheDataType.BASE,
    ) -> None:
        if weights is None:
            raise ValueError("FlashMLA Prefill requires MLA projection weights")
        if kv_cache_dtype != KvCacheDataType.BASE:
            raise ValueError(
                "Kimi K3 dense FlashMLA Prefill currently requires BF16 KV cache"
            )

        # Import lazily: selecting FlashInfer must not require FlashMLA, while an
        # explicitly selected FlashMLA backend must fail rather than fall back.
        try:
            import flash_mla.cuda as flash_mla_cuda
        except ImportError as error:
            raise RuntimeError(
                "KIMI_K3_MLA_BACKEND=flashmla requires the CUDA13 flash-mla package"
            ) from error

        self.flash_mla_cuda = flash_mla_cuda
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.page_size = page_size
        self.scale = (
            (qk_nope_head_dim + qk_rope_head_dim) ** -0.5
        ) * softmax_extra_scale
        self.weights = weights
        self.quant_config = quant_config
        self.use_mla = use_mla
        self.qo_indptr: Optional[torch.Tensor] = None
        self.kv_indptr: Optional[torch.Tensor] = None
        self.max_q_len = 0
        self.max_kv_len = 0
        self.has_reuse_cache = False

    def plan(self, mla_params: Any) -> None:
        qo_host = mla_params.qo_indptr_h
        kv_host = mla_params.prefill_ragged_kv_len_indptr_h
        qo_values = [int(value) for value in qo_host.tolist()]
        kv_values = [int(value) for value in kv_host.tolist()]
        if len(qo_values) != len(kv_values) or len(qo_values) < 2:
            raise ValueError(
                f"invalid FlashMLA indptr: qo={qo_values}, kv={kv_values}"
            )

        self.qo_indptr = mla_params.qo_indptr_d
        self.kv_indptr = mla_params.prefill_ragged_kv_len_indptr_d
        self.max_q_len = max(
            qo_values[index + 1] - qo_values[index]
            for index in range(len(qo_values) - 1)
        )
        self.max_kv_len = max(
            kv_values[index + 1] - kv_values[index]
            for index in range(len(kv_values) - 1)
        )
        reuse_pages = mla_params.reuse_cache_page_indice_d
        self.has_reuse_cache = reuse_pages is not None and reuse_pages.numel() != 0

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
    ) -> torch.Tensor:
        del kv_cache
        if self.has_reuse_cache:
            raise RuntimeError(
                "dense FlashMLA Prefill does not yet support prefix-cache reuse; "
                "run K3 with reuse_cache=0"
            )
        if self.qo_indptr is None or self.kv_indptr is None:
            raise RuntimeError("FlashMLA Prefill must be planned before forward")
        if q.shape[0] != compressed_kv.shape[0]:
            raise RuntimeError(
                "dense FlashMLA Prefill requires query and KV token counts to match"
            )

        kv_b_proj = LinearFactory.create_linear_from_weights(
            self.weights[layer_id],
            W.mla_kv_b_w,
            W.mla_kv_b_s,
            None,
            self.quant_config,
        )
        kv = kv_b_proj(compressed_kv).view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope = kv[..., : self.qk_nope_head_dim]
        value_states = kv[..., self.qk_nope_head_dim :]

        # K3's physical suffix is already prepared by the shared no-RoPE
        # adapter.  Materialize [NoPE | suffix] once for FlashMLA.
        k = q.new_empty(
            q.shape[0],
            self.num_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        k[..., : self.qk_nope_head_dim].copy_(k_nope)
        k[..., self.qk_nope_head_dim :].copy_(
            k_pe.view(-1, 1, self.qk_rope_head_dim)
        )

        out = torch.empty(
            q.shape[0],
            self.num_heads,
            self.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        # FlashMLA requires an LSE output even though inference does not consume
        # it.  The transposed allocation keeps sequence length contiguous, as
        # required by the pinned FlashMLA interface.
        lse = torch.empty(
            self.num_heads,
            q.shape[0],
            dtype=torch.float32,
            device=q.device,
        ).transpose(0, 1)
        self.flash_mla_cuda.dense_prefill_fwd(
            _workspace(q.device),
            q,
            k,
            value_states,
            self.qo_indptr,
            self.kv_indptr,
            out,
            lse,
            1,  # causal mask
            self.scale,
            self.max_q_len,
            self.max_kv_len,
            True,  # varlen packed layout
        )

        device_index = q.device.index if q.device.index is not None else 0
        if device_index not in _FLASHMLA_LOGGED_DEVICES:
            logging.info(
                "Kimi K3 dense Prefill MLA backend: FlashMLA "
                "device=%s heads=%d qk_dim=%d v_dim=%d",
                q.device,
                self.num_heads,
                self.qk_nope_head_dim + self.qk_rope_head_dim,
                self.v_head_dim,
            )
            _FLASHMLA_LOGGED_DEVICES.add(device_index)
        return out
