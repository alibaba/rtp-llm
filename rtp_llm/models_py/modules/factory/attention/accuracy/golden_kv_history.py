from typing import Dict, Sequence, Tuple

import torch

# (sequence_id, layer_idx)
HistoryKey = Tuple[int, int]
# (K, V)
HistoryValue = Tuple[torch.Tensor, torch.Tensor]


class GoldenKVHistory:
    """Scenario-level raw K/V history used by the golden attention implementation."""

    def __init__(self) -> None:
        self._history: Dict[HistoryKey, HistoryValue] = {}

    def bind_batch(self, batch_sequence_ids: Sequence[int]) -> "GoldenKVHistoryView":
        return GoldenKVHistoryView(self, batch_sequence_ids)

    def _store(
        self,
        sequence_id: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        key = (sequence_id, layer_idx)
        if key in self._history:
            raise RuntimeError(
                f"golden K/V history already exists for sequence {sequence_id}, layer {layer_idx}"
            )
        self._history[key] = (k.detach(), v.detach())

    def _get(self, sequence_id: int, layer_idx: int) -> HistoryValue:
        key = (sequence_id, layer_idx)
        if key not in self._history:
            raise RuntimeError(
                f"golden K/V history is missing for sequence {sequence_id}, layer {layer_idx}"
            )
        return self._history[key]

    def _append(
        self,
        sequence_id: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        history_k, history_v = self._get(sequence_id, layer_idx)
        self._history[(sequence_id, layer_idx)] = (
            torch.cat([history_k, k], dim=0),
            torch.cat([history_v, v], dim=0),
        )


class GoldenKVHistoryView:
    """Forward-level view that maps batch positions to scenario sequence IDs."""

    def __init__(
        self, history: GoldenKVHistory, batch_sequence_ids: Sequence[int]
    ) -> None:
        self._history = history
        self._batch_sequence_ids: Tuple[int, ...] = tuple(batch_sequence_ids)
        if len(set(self._batch_sequence_ids)) != len(self._batch_sequence_ids):
            raise ValueError("golden batch sequence IDs must be unique")

    def __len__(self) -> int:
        return len(self._batch_sequence_ids)

    def _sequence_id(self, batch_idx: int) -> int:
        if batch_idx < 0 or batch_idx >= len(self):
            raise IndexError(
                f"golden batch index {batch_idx} is out of range for batch size {len(self)}"
            )
        return self._batch_sequence_ids[batch_idx]

    def store_plain(
        self,
        batch_idx: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        self._history._store(self._sequence_id(batch_idx), layer_idx, k, v)

    def get_prefix(self, batch_idx: int, layer_idx: int) -> HistoryValue:
        return self._history._get(self._sequence_id(batch_idx), layer_idx)

    def get_decode_history(self, batch_idx: int, layer_idx: int) -> HistoryValue:
        return self._history._get(self._sequence_id(batch_idx), layer_idx)

    def append_decode(
        self,
        batch_idx: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        self._history._append(self._sequence_id(batch_idx), layer_idx, k, v)
