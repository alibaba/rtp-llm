"""Model-wide graph-stable storage for the TP1 CSA megakernel path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .mega_csa_weights import (
    DIM,
    FRONT_OUT_DIM,
    HC,
    HC_MIX,
    HEAD_DIM,
    INDEX_HEADS,
    MAIN_HEADS,
    MQA_SPLIT_KV,
    Q_LORA_RANK,
)


@dataclass
class MegaCSALayerWorkspace:
    hc_partial: torch.Tensor
    hc_sum_sq: torch.Tensor
    collapsed: torch.Tensor
    pre: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    mix: torch.Tensor
    hidden_fp8: torch.Tensor
    hidden_sf: torch.Tensor
    front_out: torch.Tensor
    window_y: torch.Tensor
    indexer_weights: torch.Tensor
    q_lora_fp8: torch.Tensor
    q_lora_sf: torch.Tensor
    indexer_q: torch.Tensor
    indexer_folded_weights: torch.Tensor


@dataclass
class MegaCSASlotMappings:
    main_state_rows: torch.Tensor
    indexer_state_rows: torch.Tensor
    main_destinations: torch.Tensor
    indexer_destinations: torch.Tensor
    swa_destinations: torch.Tensor


class MegaCSARuntime:
    """Storage shared by all CSA layers in one transformer instance."""

    def __init__(self) -> None:
        self._step = 0
        self._metadata_id: Optional[int] = None
        self._active_is_cuda_graph = False
        self._layer_workspaces: Dict[Tuple[str, int, int], MegaCSALayerWorkspace] = {}
        self._logits: Dict[Tuple[str, int, int], torch.Tensor] = {}
        self._schedule_step = -1
        self._schedule_key: Optional[Tuple[str, int, int]] = None
        self._schedule: Optional[torch.Tensor] = None
        self._graph_schedule_history: list[torch.Tensor] = []
        self._rope_cache: Dict[
            Tuple[int, str, Tuple[int, ...]], Tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def begin_decode(self, metadata: Any) -> None:
        """Mark one Python forward for shared metadata and schedule lifetime."""
        self._step += 1
        self._metadata_id = id(metadata)
        self._active_is_cuda_graph = bool(getattr(metadata, "is_cuda_graph", False))

    @staticmethod
    def num_hc_splits(m: int, device: torch.device) -> int:
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        grid_size = max((m + 63) // 64, 1)
        num_block_k = (HC * DIM + 63) // 64
        return max(min(max(num_sms, 1) // grid_size, num_block_k // 4), 1)

    def layer_workspace(
        self, m: int, num_split: int, device: torch.device
    ) -> MegaCSALayerWorkspace:
        key = (str(device), m, num_split)
        workspace = self._layer_workspaces.get(key)
        if workspace is None:
            workspace = MegaCSALayerWorkspace(
                hc_partial=torch.empty(
                    num_split, m, HC_MIX, dtype=torch.float32, device=device
                ),
                hc_sum_sq=torch.empty(num_split, m, dtype=torch.float32, device=device),
                collapsed=torch.empty(m, DIM, dtype=torch.bfloat16, device=device),
                pre=torch.empty(m, HC, dtype=torch.float32, device=device),
                post=torch.empty(m, HC, dtype=torch.float32, device=device),
                comb=torch.empty(m, HC, HC, dtype=torch.float32, device=device),
                mix=torch.empty(m, HC_MIX, dtype=torch.float32, device=device),
                hidden_fp8=torch.empty(
                    m, DIM, dtype=torch.float8_e4m3fn, device=device
                ),
                hidden_sf=torch.empty(m, DIM // 128, dtype=torch.uint8, device=device),
                front_out=torch.empty(
                    m, FRONT_OUT_DIM, dtype=torch.bfloat16, device=device
                ),
                window_y=torch.empty(m, HEAD_DIM, dtype=torch.float32, device=device),
                indexer_weights=torch.empty(
                    m, INDEX_HEADS, dtype=torch.float32, device=device
                ),
                q_lora_fp8=torch.empty(
                    m, Q_LORA_RANK, dtype=torch.float8_e4m3fn, device=device
                ),
                q_lora_sf=torch.empty(
                    m, Q_LORA_RANK // 128, dtype=torch.uint8, device=device
                ),
                indexer_q=torch.empty(
                    m,
                    INDEX_HEADS,
                    128,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
                indexer_folded_weights=torch.empty(
                    m, INDEX_HEADS, dtype=torch.float32, device=device
                ),
            )
            self._layer_workspaces[key] = workspace
        return workspace

    def logits(self, m: int, score_capacity: int, device: torch.device) -> torch.Tensor:
        width = ((score_capacity + MQA_SPLIT_KV - 1) // MQA_SPLIT_KV) * MQA_SPLIT_KV
        key = (str(device), m, width)
        logits = self._logits.get(key)
        if logits is None:
            logits = torch.empty(m, width, dtype=torch.float32, device=device)
            self._logits[key] = logits
        return logits

    def slot_mappings(self, metadata: Any, m: int) -> MegaCSASlotMappings:
        from rtp_llm.models_py.modules.dsv4.attn_type import (
            CSA_KV,
            CSA_STATE,
            INDEXER_KV,
            INDEXER_STATE,
            SWA_KV,
        )

        if self._metadata_id != id(metadata):
            raise RuntimeError(
                "MegaCSARuntime.begin_decode() must run before the CSA layer loop"
            )

        state = metadata.compressor_state_slot_mappings
        writes = metadata.pool_write_slot_mappings
        sources = (
            state.get(CSA_STATE),
            state.get(INDEXER_STATE),
            writes.get(CSA_KV),
            writes.get(INDEXER_KV),
            writes.get(SWA_KV),
        )
        names = (
            "CSA_STATE",
            "INDEXER_STATE",
            "CSA_KV",
            "INDEXER_KV",
            "SWA_KV",
        )
        for name, source in zip(names, sources):
            if source is None or int(source.numel()) < m:
                raise RuntimeError(f"DSV4 mega metadata is missing {name} slots")
            if not source.is_cuda or not source.is_contiguous():
                raise TypeError(
                    f"DSV4 mega {name} slots must be contiguous CUDA tensors"
                )
            if source.dtype != torch.int64:
                raise TypeError(f"DSV4 mega {name} slots must be int64")

        return MegaCSASlotMappings(*sources)

    def mqa_schedule(
        self, context_lens: torch.Tensor, entries_per_block: int
    ) -> torch.Tensor:
        key = (str(context_lens.device), int(context_lens.numel()), entries_per_block)
        if self._schedule_step == self._step and self._schedule_key == key:
            assert self._schedule is not None
            return self._schedule

        import deep_gemm

        context_2d = context_lens.view(-1, 1).contiguous()
        schedule = deep_gemm.get_paged_mqa_logits_metadata(
            context_2d, entries_per_block, deep_gemm.get_num_sms()
        )
        if self._active_is_cuda_graph:
            # Every warmup/capture may bake a different schedule pointer into a
            # graph. Keep all of them alive for the transformer lifetime.
            self._graph_schedule_history.append(schedule)
        self._schedule = schedule
        self._schedule_key = key
        self._schedule_step = self._step
        return schedule

    def rope_tables(self, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if freqs_cis.dtype != torch.complex64 or freqs_cis.dim() != 2:
            raise TypeError("DSV4 mega requires contiguous complex64 freqs_cis [S,32]")
        key = (
            int(freqs_cis.data_ptr()),
            str(freqs_cis.device),
            tuple(int(value) for value in freqs_cis.shape),
        )
        tables = self._rope_cache.get(key)
        if tables is None:
            tables = (freqs_cis.real.contiguous(), freqs_cis.imag.contiguous())
            self._rope_cache[key] = tables
        return tables


__all__ = ["MegaCSARuntime", "MegaCSASlotMappings"]
