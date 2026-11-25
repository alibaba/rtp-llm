from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple, Union

import torch

ReadIndex = Union[int, slice]


@dataclass(frozen=True)
class ReadSpec:
    """Describe how one checkpoint tensor is consumed by the current rank.

    ``slices`` is expressed in the checkpoint tensor's original layout.  A
    backend may execute it while reading (safetensors), while distributing
    (fastsafetensors ``get_sharded``), or as a compatibility fallback after a
    full tensor has been materialized.

    Optional ``shard_dim/count/rank`` describe a uniform transport chunk.
    Slices may further select within it. Omit the scatter hint for a slice
    spanning multiple chunks.

    ``row_indices`` expresses sorted global checkpoint rows for output-vocab
    pruning (including the tied embedding/head union). Scratch reads just
    their contiguous spans; FST currently selects from its full tensor buffer.
    It is mutually exclusive with ``slices`` and uniform-scatter metadata.
    """

    source_name: str
    output_name: str
    source_shape: Tuple[int, ...]
    load: bool = True
    slices: Optional[Tuple[ReadIndex, ...]] = None
    partition: str = "replicated"
    shard_dim: Optional[int] = None
    shard_count: int = 1
    shard_rank: int = 0
    target_ranks: Optional[Tuple[int, ...]] = None
    row_indices: Optional[Tuple[int, ...]] = None
    # Ordered, nonoverlapping half-open axis-0 spans. Unlike row_indices,
    # this is compact for fused QKV/gate-up projections with long row runs.
    segments: Optional[Tuple[Tuple[int, int], ...]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_name, str) or not self.source_name:
            raise ValueError("ReadSpec.source_name must be a non-empty string")
        if not isinstance(self.output_name, str) or not self.output_name:
            raise ValueError("ReadSpec.output_name must be a non-empty string")
        if not isinstance(self.source_shape, tuple) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in self.source_shape
        ):
            raise ValueError(f"Invalid ReadSpec source shape {self.source_shape!r}")
        if not isinstance(self.load, bool):
            raise TypeError("ReadSpec.load must be a bool")
        if self.segments is not None:
            if (
                self.slices is not None
                or self.row_indices is not None
                or self.shard_dim is not None
            ):
                raise ValueError(
                    "ReadSpec segments cannot be combined with slices, row_indices or shard_dim"
                )
            if not self.source_shape or not isinstance(self.segments, tuple):
                raise ValueError(
                    "ReadSpec segments require a tuple and non-scalar source"
                )
            previous_end = 0
            for segment in self.segments:
                if (
                    not isinstance(segment, tuple)
                    or len(segment) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in segment
                    )
                ):
                    raise ValueError(
                        "ReadSpec segments must be integer (start, end) tuples"
                    )
                start, end = segment
                if not previous_end <= start < end <= self.source_shape[0]:
                    raise ValueError(
                        "ReadSpec segments must be ordered nonoverlapping in-range spans"
                    )
                previous_end = end
        if self.row_indices is not None:
            if self.slices is not None or self.shard_dim is not None:
                raise ValueError(
                    "Indexed rows cannot be combined with slices or shard_dim"
                )
            if not self.source_shape or not isinstance(self.row_indices, tuple):
                raise ValueError(
                    "ReadSpec.row_indices requires a tuple and a non-scalar source"
                )
            if any(
                isinstance(row, bool)
                or not isinstance(row, int)
                or not 0 <= row < self.source_shape[0]
                for row in self.row_indices
            ):
                raise ValueError("ReadSpec row indices must be in the source row range")
            if tuple(sorted(set(self.row_indices))) != self.row_indices:
                raise ValueError("ReadSpec row indices must be sorted and unique")
        if self.slices is not None:
            if not isinstance(self.slices, tuple):
                raise TypeError("ReadSpec.slices must be a tuple or None")
            if len(self.slices) != len(self.source_shape):
                raise ValueError(
                    f"ReadSpec for {self.source_name!r} has {len(self.slices)} "
                    f"indices for a {len(self.source_shape)}-D tensor"
                )
            if any(
                isinstance(index, bool) or not isinstance(index, (int, slice))
                for index in self.slices
            ):
                raise TypeError("ReadSpec indices must be int or slice")
        if not isinstance(self.partition, str) or not self.partition:
            raise ValueError("ReadSpec.partition must be a non-empty string")
        if (
            isinstance(self.shard_count, bool)
            or not isinstance(self.shard_count, int)
            or self.shard_count <= 0
        ):
            raise ValueError("ReadSpec.shard_count must be a positive integer")
        if (
            isinstance(self.shard_rank, bool)
            or not isinstance(self.shard_rank, int)
            or not 0 <= self.shard_rank < self.shard_count
        ):
            raise ValueError(
                f"Invalid ReadSpec shard rank {self.shard_rank}/" f"{self.shard_count}"
            )
        if self.shard_dim is not None:
            if isinstance(self.shard_dim, bool) or not isinstance(self.shard_dim, int):
                raise TypeError("ReadSpec.shard_dim must be an integer axis")
            if not self.source_shape:
                raise ValueError("Scalar ReadSpec cannot have shard_dim")
            if not 0 <= self.shard_dim < len(self.source_shape):
                raise ValueError(
                    f"ReadSpec shard_dim={self.shard_dim} is invalid for "
                    f"shape {self.source_shape}"
                )
            if self.slices is None:
                raise ValueError("ReadSpec with shard_dim must define slices")
            self._scatter_axis_index()  # Reject impossible scatter hints on every backend.
        if self.target_ranks is not None:
            if not isinstance(self.target_ranks, tuple):
                raise TypeError("ReadSpec.target_ranks must be a tuple or None")
            if any(
                isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
                for rank in self.target_ranks
            ):
                raise ValueError(f"Invalid ReadSpec target ranks {self.target_ranks!r}")
            if tuple(sorted(set(self.target_ranks))) != self.target_ranks:
                raise ValueError("ReadSpec.target_ranks must be sorted and unique")

    @classmethod
    def full(cls, name: str, shape: Tuple[int, ...]) -> "ReadSpec":
        return cls(name, name, tuple(shape))

    @classmethod
    def skip(cls, name: str, shape: Tuple[int, ...]) -> "ReadSpec":
        return cls(name, name, tuple(shape), load=False, target_ranks=())

    @property
    def is_local_shard(self) -> bool:
        return (
            self.slices is not None
            or self.row_indices is not None
            or self.segments is not None
        )

    def for_checkpoint_name(self, name: str) -> "ReadSpec":
        return replace(self, source_name=name, output_name=name)

    def with_targets(self, target_ranks: Tuple[int, ...], load: bool) -> "ReadSpec":
        return replace(self, target_ranks=target_ranks, load=load)

    def materialize(self, source: Any):
        if not self.load:
            raise RuntimeError(
                f"Cannot materialize skipped checkpoint tensor {self.source_name!r}"
            )
        if self.segments is not None:
            pieces = [source[start:end] for start, end in self.segments]
            # CPU float8 cat is not universally implemented. Concatenate raw
            # bytes, preserving exact packed values and scale encodings.
            if not pieces:
                tensor = source[0:0]
            elif len(pieces) == 1:
                tensor = pieces[0]
            else:
                dtype = pieces[0].dtype
                tensor = torch.cat(
                    [
                        piece.contiguous().view(torch.uint8).reshape(piece.shape[0], -1)
                        for piece in pieces
                    ],
                    dim=0,
                )
                tensor = tensor.view(dtype).reshape(
                    sum(end - start for start, end in self.segments),
                    *self.source_shape[1:],
                )
        elif self.row_indices is not None:
            if isinstance(source, torch.Tensor):
                indices = torch.tensor(
                    self.row_indices, dtype=torch.long, device=source.device
                )
                tensor = source.index_select(0, indices)
            else:
                # safetensors slices do not implement index_select. Read only
                # consecutive requested rows, without materializing the full
                # vocabulary. This also supports an all-padding TP partition.
                spans = []
                for row in self.row_indices:
                    if spans and spans[-1][1] == row:
                        spans[-1] = (spans[-1][0], row + 1)
                    else:
                        spans.append((row, row + 1))
                pieces = [source[start:end] for start, end in spans]
                if len(pieces) == 1:
                    tensor = pieces[0]
                else:
                    tensor = torch.cat(pieces, dim=0) if pieces else source[0:0]
        else:
            tensor = source if self.slices is None else source[self.slices]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Read backend returned {type(tensor).__name__} for "
                f"{self.source_name!r}, expected torch.Tensor"
            )
        if not self.is_local_shard:
            return tensor
        return LoadedCheckpointTensor(tensor.contiguous(), self)

    def _scatter_axis_index(self):
        """Translate the checkpoint index to an index inside this rank's uniform chunk."""
        dim = self.shard_dim
        size = self.source_shape[dim]
        width, remainder = divmod(size, self.shard_count)
        if remainder:
            raise ValueError("ReadSpec uniform scatter requires a divisible dimension")
        offset = width * self.shard_rank
        index = self.slices[dim]
        if isinstance(index, int):
            row = index + size if index < 0 else index
            if offset <= row < offset + width:
                return row - offset
        else:
            rows = range(*index.indices(size))
            if rows.step <= 0:
                raise ValueError("ReadSpec scatter slices require a positive stride")
            if not rows:
                return slice(0, 0)
            if offset <= rows[0] and rows[-1] < offset + width:
                return slice(rows.start - offset, rows[-1] - offset + 1, rows.step)
        raise ValueError(
            "ReadSpec slices cross the hinted shard; omit shard_dim for a general slice"
        )

    def materialize_scattered(self, tensor: torch.Tensor):
        """Scatter only applied the uniform chunk, not any remaining model slices."""
        shape = list(self.source_shape)
        shape[self.shard_dim] //= self.shard_count
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"FST returned the wrong shard shape for {self.source_name!r}")
        indices = list(self.slices)
        indices[self.shard_dim] = self._scatter_axis_index()
        return self.wrap_sharded_tensor(tensor[tuple(indices)])

    def wrap_sharded_tensor(self, tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Read backend returned {type(tensor).__name__} for "
                f"{self.source_name!r}, expected torch.Tensor"
            )
        if not self.is_local_shard:
            return tensor
        return LoadedCheckpointTensor(tensor.contiguous(), self)


@dataclass(frozen=True)
class LoadedCheckpointTensor:
    """Tensor plus proof that the backend already applied ``ReadSpec``."""

    tensor: torch.Tensor
    read_spec: ReadSpec

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("LoadedCheckpointTensor.tensor must be torch.Tensor")
        if not self.read_spec.is_local_shard:
            raise ValueError(
                "LoadedCheckpointTensor requires a ReadSpec with a local slice"
            )


def unwrap_checkpoint_tensor(value: Any) -> Tuple[torch.Tensor, Optional[ReadSpec]]:
    if isinstance(value, LoadedCheckpointTensor):
        return value.tensor, value.read_spec
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"Checkpoint value must be torch.Tensor or LoadedCheckpointTensor, "
            f"got {type(value).__name__}"
        )
    return value, None


def make_sharded_read_spec(
    name: str,
    shape: Tuple[int, ...],
    dim: int,
    shard_count: int,
    shard_rank: int,
    partition: str = "tp",
) -> ReadSpec:
    shape = tuple(shape)
    if not shape:
        raise ValueError(f"Cannot shard scalar checkpoint tensor {name!r}")
    normalized_dim = dim if dim >= 0 else len(shape) + dim
    if not 0 <= normalized_dim < len(shape):
        raise ValueError(f"Invalid shard dim {dim} for {name!r} shape {shape}")
    if shard_count <= 0 or not 0 <= shard_rank < shard_count:
        raise ValueError(f"Invalid shard rank {shard_rank}/{shard_count}")
    if shape[normalized_dim] % shard_count:
        raise ValueError(
            f"Checkpoint tensor {name!r} dimension {normalized_dim}="
            f"{shape[normalized_dim]} is not divisible by {shard_count}"
        )
    shard_size = shape[normalized_dim] // shard_count
    start = shard_rank * shard_size
    indices = [slice(None)] * len(shape)
    indices[normalized_dim] = slice(start, start + shard_size)
    return ReadSpec(
        source_name=name,
        output_name=name,
        source_shape=shape,
        slices=tuple(indices),
        partition=partition,
        shard_dim=normalized_dim,
        shard_count=shard_count,
        shard_rank=shard_rank,
    )
