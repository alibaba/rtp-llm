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
from rtp_llm.ops.compute_ops import LayerKVCache, rtp_llm_ops
from rtp_llm.utils.model_weight import W


_FLASHMLA_WORKSPACES: Dict[int, torch.Tensor] = {}
_FLASHMLA_LOGGED_DEVICES: set[int] = set()
# Log each static execution shape once. KV lengths are intentionally excluded:
# target verification grows them every iteration while reusing the same plan
# shape, so including them turns this guard into a per-step INFO log.
_FLASHMLA_LOGGED_CONFIGS: set[tuple[int, int, tuple[int, ...], bool]] = set()


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
                "K3 Prefill requires the CUDA13 flash-mla package"
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
        self.reuse_cache_page_indice: Optional[torch.Tensor] = None
        self.batch_reuse_info_vec: Optional[torch.Tensor] = None
        self.total_kv_lens = 0
        self.batch_size = 0
        self.q_lens: List[int] = []
        self.kv_lens: List[int] = []

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
        self.q_lens = [
            qo_values[index + 1] - qo_values[index]
            for index in range(len(qo_values) - 1)
        ]
        self.kv_lens = [
            kv_values[index + 1] - kv_values[index]
            for index in range(len(kv_values) - 1)
        ]
        self.max_q_len = max(self.q_lens)
        self.max_kv_len = max(self.kv_lens)
        self.total_kv_lens = kv_values[-1]
        self.batch_size = len(self.q_lens)
        reuse_pages = mla_params.reuse_cache_page_indice_d
        self.has_reuse_cache = reuse_pages is not None and reuse_pages.numel() != 0
        self.reuse_cache_page_indice = reuse_pages
        self.batch_reuse_info_vec = mla_params.batch_reuse_info_vec_d

        reuse_host = mla_params.batch_reuse_info_vec_h
        if reuse_host is None or reuse_host.numel() == 0:
            prefix_lens = [0] * self.batch_size
        else:
            reuse_rows = reuse_host.reshape(-1, 4)
            if reuse_rows.shape[0] != self.batch_size:
                raise ValueError(
                    "FlashMLA batch reuse metadata disagrees with qo_indptr: "
                    f"batch={self.batch_size}, reuse_rows={reuse_rows.shape[0]}"
                )
            prefix_lens = [int(row[1]) for row in reuse_rows.tolist()]
        expected_kv_lens = [
            q_len + prefix_len
            for q_len, prefix_len in zip(self.q_lens, prefix_lens)
        ]
        if expected_kv_lens != self.kv_lens:
            raise ValueError(
                "FlashMLA Q/KV lengths disagree with cache reuse metadata: "
                f"expected={expected_kv_lens}, actual={self.kv_lens}"
            )

    def _gather_reused_kv(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_k_pe = k_pe.view(-1, self.qk_rope_head_dim)
        if not self.has_reuse_cache:
            return compressed_kv, flat_k_pe
        if kv_cache is None:
            raise RuntimeError("FlashMLA cache reuse requires an MLA KV cache")
        if self.reuse_cache_page_indice is None:
            raise RuntimeError("FlashMLA cache reuse has no page indices")
        if self.batch_reuse_info_vec is None:
            raise RuntimeError("FlashMLA cache reuse has no batch metadata")
        if self.qo_indptr is None:
            raise RuntimeError("FlashMLA cache reuse has no query indptr")
        if self.total_kv_lens < compressed_kv.shape[0]:
            raise RuntimeError(
                "FlashMLA total KV length is smaller than the current suffix: "
                f"kv={self.total_kv_lens}, suffix={compressed_kv.shape[0]}"
            )

        final_compressed_kv = torch.empty(
            (self.total_kv_lens, self.kv_lora_rank),
            dtype=compressed_kv.dtype,
            device=compressed_kv.device,
        )
        final_k_pe = torch.empty(
            (self.total_kv_lens, self.qk_rope_head_dim),
            dtype=flat_k_pe.dtype,
            device=flat_k_pe.device,
        )
        rtp_llm_ops.reuse_kv_cache_indexed_batched(
            final_compressed_kv,
            final_k_pe,
            compressed_kv,
            flat_k_pe.contiguous(),
            kv_cache.kv_cache_base,
            self.reuse_cache_page_indice,
            self.batch_reuse_info_vec,
            self.qo_indptr,
            self.page_size,
        )
        return final_compressed_kv, final_k_pe

    def _project_kv(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_b_proj = LinearFactory.create_linear_from_weights(
            self.weights[layer_id],
            W.mla_kv_b_w,
            W.mla_kv_b_s,
            None,
            self.quant_config,
        )
        expanded_dim = self.qk_nope_head_dim + self.v_head_dim
        kv = kv_b_proj(compressed_kv).view(-1, self.num_heads, expanded_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        value_states = kv[..., self.qk_nope_head_dim :]

        k = compressed_kv.new_empty(
            compressed_kv.shape[0],
            self.num_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        k[..., : self.qk_nope_head_dim].copy_(k_nope)
        k[..., self.qk_nope_head_dim :].copy_(
            k_pe.view(-1, 1, self.qk_rope_head_dim)
        )
        return k, value_states

    def _dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.qo_indptr is None or self.kv_indptr is None:
            raise RuntimeError("FlashMLA Prefill must be planned before forward")
        out = torch.empty(
            q.shape[0],
            self.num_heads,
            self.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
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
            1,
            self.scale,
            self.max_q_len,
            self.max_kv_len,
            True,
        )
        return out

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
    ) -> torch.Tensor:
        if self.qo_indptr is None or self.kv_indptr is None:
            raise RuntimeError("FlashMLA Prefill must be planned before forward")
        compressed_kv, k_pe = self._gather_reused_kv(compressed_kv, k_pe, kv_cache)
        if compressed_kv.shape[0] != self.total_kv_lens:
            raise RuntimeError(
                "FlashMLA gathered KV length disagrees with kv_indptr: "
                f"tensor={compressed_kv.shape[0]}, indptr={self.total_kv_lens}"
            )
        k, value_states = self._project_kv(compressed_kv, k_pe, layer_id)
        out = self._dense_attention(q, k, value_states)

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
        config = (
            device_index,
            self.batch_size,
            tuple(self.q_lens),
            self.has_reuse_cache,
        )
        if config not in _FLASHMLA_LOGGED_CONFIGS:
            logging.info(
                "Kimi K3 FlashMLA Prefill plan: batch=%d q_lens=%s "
                "kv_lens=%s reuse_cache=%s",
                self.batch_size,
                self.q_lens,
                self.kv_lens,
                self.has_reuse_cache,
            )
            _FLASHMLA_LOGGED_CONFIGS.add(config)
        return out
