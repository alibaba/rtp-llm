import copy
import weakref
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.read_spec import (
    ReadSpec,
    make_sharded_read_spec,
    unwrap_checkpoint_tensor,
)
from rtp_llm.models_py.runtime_weight_update import prepare_leaf_weight_updates


def _positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")


def _validate_vocab_partition(
    vocab_size: int, tp_size: int, tp_rank: int, *, allow_padding: bool = False
) -> None:
    _positive_int(vocab_size, "vocab_size")
    _positive_int(tp_size, "tp_size")
    if isinstance(tp_rank, bool) or not isinstance(tp_rank, int):
        raise ValueError(f"tp_rank must be an integer, got {tp_rank!r}")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"Invalid TP partition: rank={tp_rank}, size={tp_size}")
    if vocab_size % tp_size != 0 and not allow_padding:
        raise ValueError(
            f"vocab_size={vocab_size} must be divisible by tp_size={tp_size}"
        )


class VocabParallelEmbedding(RtpModule):
    supports_independent_weight_updates = True

    def prepare_weight_updates(self, weights):
        # The manager stages the tied head separately. A conversion copy must
        # never follow the weak reference and write the live head.
        conversion = copy.copy(self)
        conversion._tied_output_head = None
        return prepare_leaf_weight_updates(conversion, weights)

    def tie_output_head(self, head: "ParallelLMHead") -> None:
        head._tied_checkpoint_source_rows = self.vocab_size
        if self.vocab_size < head.checkpoint_vocab_size:
            # Legacy pruning requires a source covering the output vocabulary.
            # A distinct explicit LM-head checkpoint may still satisfy it, so
            # defer failure to the model's missing-head completeness check.
            self._tied_output_head = None
            return
        self._tied_output_head = weakref.ref(head)

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        params_dtype: torch.dtype = torch.float16,
        allow_vocab_padding: bool = False,
    ):
        super().__init__()
        if not isinstance(allow_vocab_padding, bool):
            raise TypeError("allow_vocab_padding must be a bool")
        _validate_vocab_partition(
            vocab_size, tp_size, tp_rank, allow_padding=allow_vocab_padding
        )
        _positive_int(embedding_dim, "embedding_dim")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.vocab_size_per_partition = (vocab_size + tp_size - 1) // tp_size
        self.vocab_start_idx = min(tp_rank * self.vocab_size_per_partition, vocab_size)
        self.vocab_end_idx = min(
            self.vocab_start_idx + self.vocab_size_per_partition, vocab_size
        )
        self._tied_output_head = None
        self.weight = nn.Parameter(
            torch.empty(
                self.vocab_size_per_partition,
                embedding_dim,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

    def checkpoint_read_spec(
        self, name: str, shape: tuple[int, ...]
    ) -> ReadSpec | None:
        head = self._tied_output_head() if self._tied_output_head is not None else None
        if head is not None and name == "weight":
            if shape != (self.vocab_size, self.embedding_dim):
                raise ValueError(
                    "Tied embedding checkpoint shape does not match the model"
                )
            # Scatter is collective: a rank whose head happens to need only
            # local rows cannot scatter while its peer broadcasts the same key.
            # Only the ordinary aligned, unpruned partition can use scatter.
            if self.tp_size > 1 and (
                head.output_vocab_ids
                or head.tp_size != self.tp_size
                or head.vocab_size_per_partition != self.vocab_size_per_partition
                or head.tp_rank != self.tp_rank
            ):
                start, end = self.vocab_start_idx, self.vocab_end_idx
                extra = [
                    row for row in head.local_checkpoint_rows if not start <= row < end
                ]
                if extra:
                    rows = tuple(sorted((*range(start, end), *extra)))
                    return ReadSpec(
                        name, name, shape, partition="vocab", row_indices=rows
                    )
                # FST still broadcasts, then selects this contiguous range.
                # Scratch can read the range directly from the file.
                return ReadSpec(
                    name,
                    name,
                    shape,
                    partition="vocab",
                    slices=(slice(start, end), slice(None)),
                )
        if (
            self.tp_size <= 1
            or name != "weight"
            or shape != (self.vocab_size, self.embedding_dim)
        ):
            return None
        if self.vocab_size % self.tp_size:
            # Padding exists only in the destination parameter; never ask FST
            # uniform scatter to fabricate source rows beyond the checkpoint.
            return ReadSpec(
                name,
                name,
                shape,
                slices=(slice(self.vocab_start_idx, self.vocab_end_idx), slice(None)),
                partition="vocab",
            )
        return make_sharded_read_spec(name, shape, 0, self.tp_size, self.tp_rank)

    def prepare_local_checkpoint_weight(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pad a rank-local input-vocabulary slice without changing token IDs."""
        rows = self.vocab_end_idx - self.vocab_start_idx
        if tensor.shape != (rows, self.embedding_dim):
            raise ValueError(
                "Embedding local checkpoint shape does not match its row range"
            )
        if rows == self.vocab_size_per_partition:
            return tensor.contiguous()
        padded = tensor.new_zeros((self.vocab_size_per_partition, self.embedding_dim))
        padded[:rows].copy_(tensor)
        return padded

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, checkpoint_value in weights.items():
            tensor, read_spec = unwrap_checkpoint_tensor(checkpoint_value)
            if name != "weight":
                raise RuntimeError(f"Unsupported embedding tensor {name!r}")
            head = (
                self._tied_output_head() if self._tied_output_head is not None else None
            )
            if read_spec is not None and read_spec.row_indices is not None:
                local = _checkpoint_rows(
                    tensor,
                    read_spec,
                    tuple(range(self.vocab_start_idx, self.vocab_end_idx)),
                    self.vocab_size,
                    self.embedding_dim,
                )
                if not self._assign_weight(
                    self, "weight", self.prepare_local_checkpoint_weight(local)
                ):
                    raise RuntimeError("Failed to assign indexed embedding weight")
                if head is not None:
                    head.load_tied_checkpoint_weight(tensor, read_spec)
                continue
            expected_rows = (
                self.vocab_end_idx - self.vocab_start_idx
                if read_spec is not None
                else self.vocab_size
            )
            if read_spec is None and tensor.shape[0] != self.vocab_size:
                raise ValueError(
                    f"Embedding checkpoint rows must be {self.vocab_size}, "
                    f"got {tensor.shape[0]}"
                )
            if tensor.shape != (expected_rows, self.embedding_dim):
                raise ValueError(
                    f"Embedding checkpoint shape must be "
                    f"{(expected_rows, self.embedding_dim)}, got "
                    f"{tuple(tensor.shape)}"
                )
            local = (
                tensor
                if read_spec is not None
                else tensor[self.vocab_start_idx : self.vocab_end_idx]
            )
            if not self._assign_weight(
                self, "weight", self.prepare_local_checkpoint_weight(local)
            ):
                raise RuntimeError("Failed to assign embedding weight")
            if head is not None:
                head.load_tied_checkpoint_weight(tensor, read_spec)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return torch.nn.functional.embedding(input_ids, self.weight)

        mask = (input_ids >= self.vocab_start_idx) & (input_ids < self.vocab_end_idx)
        local_ids = torch.where(
            mask,
            input_ids - self.vocab_start_idx,
            torch.zeros_like(input_ids),
        )
        output = torch.nn.functional.embedding(local_ids, self.weight)
        output = output * mask.unsqueeze(-1)
        from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

        return all_reduce(output, group=Group.TP)


class ParallelLMHead(RtpModule):
    supports_independent_weight_updates = True

    def prepare_weight_updates(self, weights):
        return prepare_leaf_weight_updates(self, weights)

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        params_dtype: torch.dtype = torch.float16,
        output_vocab_ids: Optional[Sequence[int]] = None,
        output_vocab_padded_size: Optional[int] = None,
        tp_pad8: bool = False,
    ):
        super().__init__()
        _positive_int(vocab_size, "vocab_size")
        _positive_int(hidden_size, "hidden_size")
        _positive_int(tp_size, "tp_size")
        self.checkpoint_vocab_size = vocab_size
        self.output_vocab_ids = tuple(output_vocab_ids or ())
        if self.output_vocab_ids:
            if any(
                isinstance(row, bool)
                or not isinstance(row, int)
                or not 0 <= row < vocab_size
                for row in self.output_vocab_ids
            ):
                raise ValueError(
                    "Output vocabulary ids must be integers within the model vocabulary"
                )
            if tuple(sorted(set(self.output_vocab_ids))) != self.output_vocab_ids:
                raise ValueError("Output vocabulary ids must be sorted and unique")
            vocab_size = output_vocab_padded_size or len(self.output_vocab_ids)
            if vocab_size < len(self.output_vocab_ids):
                raise ValueError(
                    "Output vocabulary padding cannot remove selected rows"
                )
        elif output_vocab_padded_size:
            raise ValueError("Output vocabulary padding requires selected token ids")
        elif tp_pad8 or vocab_size % tp_size:
            # Preserve existing divisible layouts, and use the legacy pad8
            # alignment when an ordinary vocabulary needs TP padding.
            alignment = tp_size * 8
            vocab_size = ((vocab_size + alignment - 1) // alignment) * alignment
        _validate_vocab_partition(vocab_size, tp_size, tp_rank)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.vocab_size_per_partition = vocab_size // tp_size
        start = tp_rank * self.vocab_size_per_partition
        self.local_output_vocab_ids = self.output_vocab_ids[
            start : start + self.vocab_size_per_partition
        ]
        self._explicit_checkpoint_loaded = False
        self._tied_checkpoint_loaded = False
        self._tied_checkpoint_source_rows = None
        self.weight = nn.Parameter(
            torch.empty(
                self.vocab_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

    @property
    def local_checkpoint_rows(self) -> tuple[int, ...]:
        if self.output_vocab_ids:
            return self.local_output_vocab_ids
        start = min(
            self.tp_rank * self.vocab_size_per_partition, self.checkpoint_vocab_size
        )
        end = min(start + self.vocab_size_per_partition, self.checkpoint_vocab_size)
        return tuple(range(start, end))

    def checkpoint_read_spec(
        self, name: str, shape: tuple[int, ...]
    ) -> ReadSpec | None:
        if self.output_vocab_ids and name == "weight":
            if (
                len(shape) != 2
                or shape[0] < self.checkpoint_vocab_size
                or shape[1] != self.hidden_size
            ):
                raise ValueError(
                    "Pruned LM head checkpoint does not cover model vocabulary/hidden size"
                )
            return ReadSpec(
                name,
                name,
                shape,
                partition="vocab",
                row_indices=self.local_output_vocab_ids,
            )
        if (
            self.tp_size <= 1
            or name != "weight"
            or shape != (self.checkpoint_vocab_size, self.hidden_size)
        ):
            return None
        if self.vocab_size == self.checkpoint_vocab_size:
            return make_sharded_read_spec(name, shape, 0, self.tp_size, self.tp_rank)
        return self.tied_checkpoint_read_spec(name, shape)

    def tied_checkpoint_read_spec(self, name: str, shape: tuple[int, ...]) -> ReadSpec:
        """Plan a head read directly from its global embedding source (also PP)."""
        if (
            name != "weight"
            or len(shape) != 2
            or shape[0] < self.checkpoint_vocab_size
            or shape[1] != self.hidden_size
        ):
            raise ValueError(
                "Tied head source must cover the output vocabulary and hidden size"
            )
        if self.output_vocab_ids:
            return ReadSpec(
                name,
                name,
                shape,
                partition="vocab",
                row_indices=self.local_checkpoint_rows,
            )
        start = min(
            self.tp_rank * self.vocab_size_per_partition, self.checkpoint_vocab_size
        )
        end = start + len(self.local_checkpoint_rows)
        return ReadSpec(
            name,
            name,
            shape,
            partition="vocab",
            slices=(slice(start, end), slice(None)),
        )

    def prepare_local_checkpoint_weight(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pad already selected valid rows to this rank's final head storage."""
        rows = len(self.local_checkpoint_rows)
        if tensor.shape != (rows, self.hidden_size):
            raise ValueError("LM head local checkpoint rows do not match its read plan")
        if rows == self.vocab_size_per_partition:
            return tensor.contiguous()
        local = tensor.new_zeros((self.vocab_size_per_partition, self.hidden_size))
        local[:rows].copy_(tensor)
        return local

    def _prepare_checkpoint_weight(self, tensor, read_spec, *, tied=False):
        if tensor.ndim != 2:
            raise ValueError("LM head checkpoint must be a two-dimensional matrix")
        source_rows = (
            read_spec.source_shape[0] if read_spec is not None else tensor.shape[0]
        )
        if (
            not tied
            and not self.output_vocab_ids
            and source_rows != self.checkpoint_vocab_size
        ):
            raise ValueError(
                f"LM head checkpoint rows must be {self.checkpoint_vocab_size}, got {source_rows}"
            )
        selected = _checkpoint_rows(
            tensor,
            read_spec,
            self.local_checkpoint_rows,
            self.checkpoint_vocab_size,
            self.hidden_size,
        )
        return self.prepare_local_checkpoint_weight(selected)

    def _copy_weight(self, tensor: torch.Tensor, local_shard: bool = False) -> None:
        local = (
            self.prepare_local_checkpoint_weight(tensor)
            if local_shard
            else self._prepare_checkpoint_weight(tensor, None)
        )
        if not self._assign_weight(self, "weight", local):
            raise RuntimeError("Failed to assign LM head weight")

    def _copy_local_tied_weight(self, tensor: torch.Tensor) -> None:
        if self._tied_checkpoint_loaded or self._explicit_checkpoint_loaded:
            return
        if (
            self._tied_checkpoint_source_rows is not None
            and self._tied_checkpoint_source_rows < self.checkpoint_vocab_size
        ):
            raise ValueError(
                "Tied embedding source does not cover the configured output vocabulary"
            )
        if self.output_vocab_ids or self.vocab_size != self.checkpoint_vocab_size:
            raise RuntimeError(
                "Tied LM head with selected or padded rows was not loaded from the embedding checkpoint"
            )
        local_shape = tuple(self.weight.shape)
        global_shape = (self.vocab_size, self.hidden_size)
        if tuple(tensor.shape) == global_shape:
            start = self.tp_rank * self.vocab_size_per_partition
            tensor = tensor[start : start + self.vocab_size_per_partition]
        elif tuple(tensor.shape) != local_shape:
            raise ValueError(
                f"Tied LM head weight must have local shape {local_shape} or "
                f"global shape {global_shape}, got {tuple(tensor.shape)}"
            )
        if not self._assign_weight(self, "weight", tensor.contiguous()):
            raise RuntimeError("Failed to assign tied LM head weight")
        self._tied_checkpoint_loaded = True

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        for name, checkpoint_value in weights.items():
            tensor, read_spec = unwrap_checkpoint_tensor(checkpoint_value)
            if name != "weight":
                raise RuntimeError(f"Unsupported LM head tensor {name!r}")
            tied_update = (
                self._tied_checkpoint_loaded
                and not self._explicit_checkpoint_loaded
                and read_spec is None
                and tensor.shape[0] == self._tied_checkpoint_source_rows
            )
            local = self._prepare_checkpoint_weight(tensor, read_spec, tied=tied_update)
            if not self._assign_weight(self, "weight", local):
                raise RuntimeError("Failed to assign LM head weight")
            self._explicit_checkpoint_loaded = True

    def _copy_pruned_weight(
        self, tensor: torch.Tensor, read_spec: Optional[ReadSpec]
    ) -> None:
        local = self._prepare_checkpoint_weight(tensor, read_spec)
        if not self._assign_weight(self, "weight", local):
            raise RuntimeError("Failed to assign pruned LM head weight")

    def load_tied_checkpoint_weight(
        self, tensor: torch.Tensor, read_spec: Optional[ReadSpec] = None
    ) -> None:
        if not self._explicit_checkpoint_loaded:
            local = self._prepare_checkpoint_weight(tensor, read_spec, tied=True)
            if not self._assign_weight(self, "weight", local):
                raise RuntimeError("Failed to assign tied LM head weight")
            self._tied_checkpoint_source_rows = (
                read_spec.source_shape[0] if read_spec is not None else tensor.shape[0]
            )
            self._tied_checkpoint_loaded = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.nn.functional.linear(x, self.weight)
        if self.tp_size == 1:
            return output
        token_shape = output.shape[:-1]
        local_vocab_size = output.shape[-1]
        flat_output = output.reshape(-1, local_vocab_size)
        from rtp_llm.models_py.distributed.collective_torch import Group, all_gather

        gathered = all_gather(flat_output, group=Group.TP)
        gathered = (
            gathered.reshape(self.tp_size, flat_output.shape[0], local_vocab_size)
            .transpose(0, 1)
            .contiguous()
            .reshape(*token_shape, self.vocab_size)
        )
        return gathered


def _checkpoint_rows(
    tensor: torch.Tensor,
    read_spec: Optional[ReadSpec],
    rows: Sequence[int],
    vocab_size: int,
    hidden_size: int,
) -> torch.Tensor:
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            "Output vocabulary selection requires FP16, BF16 or FP32 weights"
        )
    if tensor.ndim != 2 or tensor.shape[1] != hidden_size:
        raise ValueError("Output vocabulary checkpoint hidden size mismatch")
    if read_spec is None:
        if tensor.shape[0] < vocab_size:
            raise ValueError(
                "Output vocabulary checkpoint does not cover model vocabulary"
            )
        offsets = rows
    else:
        if read_spec.source_shape[0] < vocab_size:
            raise ValueError("Checkpoint source does not cover the output vocabulary")
        if read_spec.row_indices is not None:
            if tensor.shape[0] != len(read_spec.row_indices):
                raise ValueError("Indexed checkpoint row count does not match ReadSpec")
            positions = {
                row: offset for offset, row in enumerate(read_spec.row_indices)
            }
            try:
                offsets = tuple(positions[row] for row in rows)
            except KeyError as error:
                raise ValueError(
                    "ReadSpec does not include a required vocabulary row"
                ) from error
        elif (
            read_spec.slices is not None
            and len(read_spec.slices) == 2
            and isinstance(read_spec.slices[0], slice)
            and read_spec.slices[1] == slice(None)
        ):
            start, end, step = read_spec.slices[0].indices(read_spec.source_shape[0])
            if step != 1 or tensor.shape[0] != max(0, end - start):
                raise ValueError("Vocabulary ReadSpec requires a contiguous row slice")
            offsets = tuple(row - start for row in rows)
            if any(offset < 0 or offset >= tensor.shape[0] for offset in offsets):
                raise ValueError("ReadSpec does not include a required vocabulary row")
        else:
            raise ValueError(
                "Vocabulary selection requires indexed rows or a row slice"
            )
    if not offsets:
        return tensor[:0]
    # An indexed head ReadSpec is already in output order. The embedding
    # partition is also a contiguous subrange of its sorted union ReadSpec.
    # Preserve a view in these cases instead of making a second large copy.
    if tuple(offsets) == tuple(range(offsets[0], offsets[0] + len(offsets))):
        return tensor.narrow(0, offsets[0], len(offsets))
    indices = torch.tensor(offsets, dtype=torch.long, device=tensor.device)
    return tensor.index_select(0, indices)


def output_vocab_head_kwargs(model_config):
    """Consume BaseModel's finalized token order and distributed padding."""

    def value(key, default=None):
        if isinstance(model_config, dict):
            return model_config.get(key, default)
        return getattr(model_config, key, default)

    if not value("enable_output_vocab_pruning", False):
        return {}
    ids = value("output_vocab_ids", ())
    if not ids:
        raise ValueError(
            "Output vocabulary must be finalized before NewLoader model construction"
        )
    return {
        "output_vocab_ids": ids,
        "output_vocab_padded_size": value("output_vocab_padded_size", 0),
    }
