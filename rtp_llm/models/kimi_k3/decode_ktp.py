"""Experimental K3 Decode KDA-TP / MLA request-DP helpers.

This module deliberately contains no scheduler policy.  Phase one accepts a
fixed, q_len=1 batch and assigns one contiguous, equally-sized request range to
each rank.  Keeping the layout helper separate makes the ownership contract
unit-testable before the production scheduler learns about MLA owners.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import torch

from rtp_llm.ops import ParallelismConfig, RoleType
from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
    reduce_scatter,
)


_DECODE_KTP_ENV = "KIMI_K3_DECODE_KTP"
_KTP_DESCRIPTOR_VERSION = 1
_KTP_DESCRIPTOR_HEADER = 5


def decode_ktp_enabled(role_type: RoleType) -> bool:
    raw = os.environ.get(_DECODE_KTP_ENV, "0").strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{_DECODE_KTP_ENV} must be 0 or 1, got {raw!r}")
    return raw == "1" and role_type == RoleType.DECODE


def logical_tp1_parallelism(config: ParallelismConfig) -> ParallelismConfig:
    """Clone a rank config for replicated MLA weights and local MLA compute."""

    result = copy.copy(config)
    result.tp_size = 1
    result.tp_rank = 0
    # MLA does not consume FFN placement, but setting these fields prevents a
    # future generic attention helper from accidentally observing TP8 there.
    result.ffn_tp_size = 1
    result.ffn_tp_rank = 0
    return result


@dataclass(frozen=True)
class KdaParallelContext:
    """KDA-only tensor-parallel view over Decode DP workers."""

    size: int
    rank: int
    group: Group

    @classmethod
    def from_parallelism(cls, config: ParallelismConfig) -> "KdaParallelContext":
        if not decode_ktp_enabled(config.role_type):
            return cls(
                int(config.tp_size),
                int(config.tp_rank),
                Group.TP,
            )

        topology = {
            "tp_size": int(config.tp_size),
            "dp_size": int(config.dp_size),
            "ep_size": int(config.ep_size),
            "ktp_size": int(config.ktp_size),
            "world_size": int(config.world_size),
            "world_rank": int(config.world_rank),
            "dp_rank": int(config.dp_rank),
            "ktp_rank": int(config.ktp_rank),
        }
        world = topology["world_size"]
        if (
            topology["tp_size"] != 1
            or topology["dp_size"] != world
            or topology["ep_size"] != world
            or topology["ktp_size"] != world
            or world <= 1
            or topology["dp_rank"] != topology["world_rank"]
            or topology["ktp_rank"] != topology["world_rank"]
        ):
            raise ValueError(
                "KIMI_K3_DECODE_KTP requires Decode TP1/DP=EP=KTP=WORLD "
                f"with dp_rank=ktp_rank=world_rank, got {topology}"
            )
        return cls(world, topology["world_rank"], Group.KTP)

    def parallelism_config(self, config: ParallelismConfig) -> ParallelismConfig:
        """Clone the global DP config into the view consumed only by KDA."""

        result = copy.copy(config)
        result.tp_size = self.size
        result.tp_rank = self.rank
        return result


@dataclass(frozen=True)
class KtpStepDescriptor:
    """Fixed-width metadata contributed by one Decode DP worker.

    Request ids remain optional until the distributed KDA shadow registry is
    installed.  In that transition state, deterministic negative ids identify
    physical rows but are never valid registry keys.
    """

    rank: int
    step_epoch: int
    local_batch: int
    bucket: int
    request_ids: tuple[int, ...]
    generation_epochs: tuple[int, ...]
    valid_mask: tuple[bool, ...]

    @classmethod
    def build(
        cls,
        *,
        rank: int,
        step_epoch: int,
        local_batch: int,
        bucket: int,
        request_ids: Optional[Iterable[int]] = None,
        generation_epochs: Optional[Iterable[int]] = None,
        is_fake: bool = False,
    ) -> "KtpStepDescriptor":
        if rank < 0 or step_epoch < 0:
            raise ValueError(f"invalid KTP rank/epoch rank={rank} epoch={step_epoch}")
        if bucket <= 0:
            raise ValueError(f"KTP local bucket must be positive, got {bucket}")
        semantic_batch = 0 if is_fake else int(local_batch)
        if semantic_batch < 0 or semantic_batch > bucket:
            raise ValueError(
                "KTP local batch exceeds fixed bucket: "
                f"local_batch={semantic_batch}, bucket={bucket}"
            )

        ids = tuple(int(value) for value in request_ids or ())
        epochs = tuple(int(value) for value in generation_epochs or ())
        if is_fake:
            # The framework's fake stream may carry one placeholder request
            # row. It is a collective participant, never a cache owner.
            ids = ()
            epochs = ()
        if ids and len(ids) != semantic_batch:
            raise ValueError(
                f"KTP request id count {len(ids)} != local batch {semantic_batch}"
            )
        if epochs and len(epochs) != semantic_batch:
            raise ValueError(
                "KTP generation epoch count "
                f"{len(epochs)} != local batch {semantic_batch}"
            )
        if not ids:
            # Temporary Stage-2 row identity. Stage 3 replaces these values
            # with real (request_id, generation_epoch) registry keys.
            ids = tuple(-(rank * bucket + row + 2) for row in range(semantic_batch))
        if not epochs:
            epochs = (0,) * semantic_batch
        pad = bucket - semantic_batch
        return cls(
            rank=rank,
            step_epoch=step_epoch,
            local_batch=semantic_batch,
            bucket=bucket,
            request_ids=ids + (-1,) * pad,
            generation_epochs=epochs + (-1,) * pad,
            valid_mask=(True,) * semantic_batch + (False,) * pad,
        )

    @property
    def packed_width(self) -> int:
        return _KTP_DESCRIPTOR_HEADER + 3 * self.bucket

    def pack(self, device: torch.device) -> torch.Tensor:
        values = [
            _KTP_DESCRIPTOR_VERSION,
            self.rank,
            self.step_epoch,
            self.local_batch,
            self.bucket,
            *self.request_ids,
            *self.generation_epochs,
            *(int(value) for value in self.valid_mask),
        ]
        if len(values) != self.packed_width:
            raise RuntimeError("KTP descriptor has inconsistent fixed-width fields")
        return torch.tensor(values, dtype=torch.int64, device=device)

    @classmethod
    def unpack(cls, packed: torch.Tensor) -> "KtpStepDescriptor":
        values = [int(value) for value in packed.detach().cpu().tolist()]
        if len(values) < _KTP_DESCRIPTOR_HEADER:
            raise ValueError("KTP descriptor is shorter than its header")
        version, rank, step_epoch, local_batch, bucket = values[:5]
        if version != _KTP_DESCRIPTOR_VERSION:
            raise ValueError(
                f"unsupported KTP descriptor version {version}; "
                f"expected {_KTP_DESCRIPTOR_VERSION}"
            )
        expected = _KTP_DESCRIPTOR_HEADER + 3 * bucket
        if bucket <= 0 or len(values) != expected:
            raise ValueError(
                f"invalid KTP descriptor width={len(values)} bucket={bucket}"
            )
        ids_begin = _KTP_DESCRIPTOR_HEADER
        epoch_begin = ids_begin + bucket
        valid_begin = epoch_begin + bucket
        descriptor = cls(
            rank=rank,
            step_epoch=step_epoch,
            local_batch=local_batch,
            bucket=bucket,
            request_ids=tuple(values[ids_begin:epoch_begin]),
            generation_epochs=tuple(values[epoch_begin:valid_begin]),
            valid_mask=tuple(bool(value) for value in values[valid_begin:]),
        )
        descriptor.validate()
        return descriptor

    def validate(self) -> None:
        if self.bucket <= 0 or not 0 <= self.local_batch <= self.bucket:
            raise ValueError(
                f"invalid KTP descriptor batch={self.local_batch} bucket={self.bucket}"
            )
        if not (
            len(self.request_ids)
            == len(self.generation_epochs)
            == len(self.valid_mask)
            == self.bucket
        ):
            raise ValueError("KTP descriptor fields do not match its bucket")
        expected_mask = (True,) * self.local_batch + (False,) * (
            self.bucket - self.local_batch
        )
        if self.valid_mask != expected_mask:
            raise ValueError("KTP valid mask must be a contiguous local prefix")
        for idx in range(self.local_batch, self.bucket):
            if self.request_ids[idx] != -1 or self.generation_epochs[idx] != -1:
                raise ValueError("KTP padding rows must use -1 request/epoch sentinels")


@dataclass(frozen=True)
class KtpBatchPlan:
    """Rank-major fixed-bucket layout shared by every KTP participant."""

    descriptors: tuple[KtpStepDescriptor, ...]
    rank: int

    def __post_init__(self) -> None:
        if not self.descriptors or not 0 <= self.rank < len(self.descriptors):
            raise ValueError("invalid KTP batch plan rank or descriptor set")
        epoch = self.descriptors[0].step_epoch
        bucket = self.descriptors[0].bucket
        for expected_rank, descriptor in enumerate(self.descriptors):
            descriptor.validate()
            if descriptor.rank != expected_rank:
                raise ValueError(
                    "KTP descriptor rank order mismatch: "
                    f"slot={expected_rank}, descriptor={descriptor.rank}"
                )
            if descriptor.step_epoch != epoch:
                raise ValueError("KTP workers entered different decode step epochs")
            if descriptor.bucket != bucket:
                raise ValueError("KTP workers disagree on the fixed local bucket")

    @property
    def bucket(self) -> int:
        return self.descriptors[0].bucket

    @property
    def step_epoch(self) -> int:
        return self.descriptors[0].step_epoch

    @property
    def local_batch(self) -> int:
        return self.descriptors[self.rank].local_batch

    @property
    def physical_rows(self) -> int:
        return len(self.descriptors) * self.bucket

    @property
    def valid_rows(self) -> int:
        return sum(descriptor.local_batch for descriptor in self.descriptors)

    @property
    def local_physical_slice(self) -> slice:
        begin = self.rank * self.bucket
        return slice(begin, begin + self.bucket)

    @property
    def request_keys(self) -> tuple[Optional[tuple[int, int]], ...]:
        keys: list[Optional[tuple[int, int]]] = []
        for descriptor in self.descriptors:
            keys.extend(
                (request_id, generation_epoch) if valid else None
                for request_id, generation_epoch, valid in zip(
                    descriptor.request_ids,
                    descriptor.generation_epochs,
                    descriptor.valid_mask,
                )
            )
        return tuple(keys)

    def valid_mask(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [valid for item in self.descriptors for valid in item.valid_mask],
            dtype=torch.bool,
            device=device,
        )

    def pad_local_rows(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 0 or int(value.shape[0]) != self.local_batch:
            raise ValueError(
                "KTP local activation rows do not match descriptor: "
                f"shape={tuple(value.shape)}, local_batch={self.local_batch}"
            )
        if self.local_batch == self.bucket:
            return value.contiguous()
        padding = value.new_zeros(
            [self.bucket - self.local_batch] + list(value.shape[1:])
        )
        return torch.cat((value, padding), dim=0).contiguous()

    def trim_local_rows(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 0 or int(value.shape[0]) != self.bucket:
            raise ValueError(
                "KTP local physical shard does not match fixed bucket: "
                f"shape={tuple(value.shape)}, bucket={self.bucket}"
            )
        return value.narrow(0, 0, self.local_batch).contiguous()

    def all_gather_rows(self, local_value: torch.Tensor) -> torch.Tensor:
        gathered = all_gather(self.pad_local_rows(local_value), group=Group.KTP)
        if int(gathered.shape[0]) != self.physical_rows:
            raise RuntimeError(
                "KTP activation AllGather returned unexpected rows: "
                f"rows={gathered.shape[0]}, expected={self.physical_rows}"
            )
        return gathered

    def reduce_scatter_rows(self, global_partial: torch.Tensor) -> torch.Tensor:
        if global_partial.ndim == 0 or int(global_partial.shape[0]) != self.physical_rows:
            raise ValueError(
                "KTP global partial rows do not match rank-major layout: "
                f"shape={tuple(global_partial.shape)}, expected={self.physical_rows}"
            )
        return self.trim_local_rows(reduce_scatter(global_partial, group=Group.KTP))


def rendezvous_ktp_step(
    context: KdaParallelContext,
    *,
    step_epoch: int,
    local_batch: int,
    fixed_bucket: int,
    device: torch.device,
    request_ids: Optional[Iterable[int]] = None,
    generation_epochs: Optional[Iterable[int]] = None,
    is_fake: bool = False,
) -> KtpBatchPlan:
    """AllGather fixed-width descriptors and validate one global decode tick."""

    if context.group != Group.KTP or context.size <= 1:
        raise ValueError("Decode DP rendezvous requires an enabled KTP context")
    local = KtpStepDescriptor.build(
        rank=context.rank,
        step_epoch=step_epoch,
        local_batch=local_batch,
        bucket=fixed_bucket,
        request_ids=request_ids,
        generation_epochs=generation_epochs,
        is_fake=is_fake,
    )
    packed = local.pack(device)
    gathered = all_gather(packed, group=Group.KTP)
    expected_values = context.size * local.packed_width
    if gathered.ndim != 1 or gathered.numel() != expected_values:
        raise RuntimeError(
            "KTP descriptor AllGather returned unexpected shape: "
            f"shape={tuple(gathered.shape)}, expected_values={expected_values}"
        )
    rows = gathered.reshape(context.size, local.packed_width)
    descriptors = tuple(KtpStepDescriptor.unpack(row) for row in rows)
    plan = KtpBatchPlan(descriptors, context.rank)
    if plan.descriptors[context.rank] != local:
        raise RuntimeError("KTP gathered local descriptor was corrupted or reordered")
    return plan


@dataclass(frozen=True)
class DecodeOwnerLayout:
    global_batch: int
    local_batch: int
    start: int
    stop: int

    @classmethod
    def fixed(cls, global_batch: int, tp_size: int, tp_rank: int) -> "DecodeOwnerLayout":
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid TP placement size={tp_size} rank={tp_rank}")
        if global_batch < tp_size or global_batch % tp_size:
            raise ValueError(
                "KIMI_K3_DECODE_KTP requires BS >= TP and BS divisible by TP; "
                f"got BS={global_batch}, TP={tp_size}"
            )
        local = global_batch // tp_size
        start = tp_rank * local
        return cls(global_batch, local, start, start + local)

    def narrow(self, value: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return value.narrow(dim, self.start, self.local_batch).contiguous()


def _maybe_owner_rows(
    value: Optional[torch.Tensor], layout: DecodeOwnerLayout, *, dim: int = 0
) -> Optional[torch.Tensor]:
    if value is None or not value.numel():
        return value
    if value.shape[dim] != layout.global_batch:
        raise ValueError(
            "K3 Decode owner metadata has unexpected batch dimension: "
            f"shape={tuple(value.shape)} dim={dim} BS={layout.global_batch}"
        )
    return layout.narrow(value, dim)


def _owner_group_rows(
    values: Sequence[torch.Tensor], layout: DecodeOwnerLayout
) -> list[torch.Tensor]:
    return [_maybe_owner_rows(value, layout) for value in values]


def build_owner_attention_inputs(
    attention_inputs: PyAttentionInputs,
    layout: DecodeOwnerLayout,
    *,
    device: torch.device,
    global_query_tokens: int,
) -> PyAttentionInputs:
    """Return the local request view consumed by logical-TP1 MLA Decode."""

    if attention_inputs.is_prefill:
        raise ValueError("KIMI_K3_DECODE_KTP supports Decode only")
    if bool(getattr(attention_inputs, "is_target_verify", False)):
        raise ValueError("KIMI_K3_DECODE_KTP does not support target verify")
    if bool(getattr(attention_inputs, "is_cuda_graph", False)):
        raise ValueError("KIMI_K3_DECODE_KTP phase one does not support CUDA Graph")
    if getattr(attention_inputs, "cache_store_inputs", None) is not None:
        raise ValueError("KIMI_K3_DECODE_KTP phase one does not support cache-store")

    # Decode PyAttentionInputs intentionally has no cu_seqlens_host: the C++
    # boundary only materializes that host mirror for Prefill.  The packed
    # query tensor is authoritative here.  q_len=1 therefore means exactly
    # one query token for every request in the global owner layout.
    if global_query_tokens != layout.global_batch:
        raise ValueError(
            "KIMI_K3_DECODE_KTP phase one requires q_len=1: "
            f"tokens={global_query_tokens} BS={layout.global_batch}"
        )

    local = copy.copy(attention_inputs)
    tensor_fields = (
        "input_lengths",
        "prefix_lengths",
        "sequence_lengths",
        "sequence_lengths_plus_1_d",
        "input_lengths_host",
        "prefix_lengths_host",
        "sequence_lengths_host",
    )
    for name in tensor_fields:
        value = getattr(attention_inputs, name, None)
        if value is not None and value.numel():
            setattr(local, name, _maybe_owner_rows(value, layout))

    local_lengths_host = getattr(local, "input_lengths_host", None)
    if local_lengths_host is None or local_lengths_host.numel() != layout.local_batch:
        raise ValueError("KIMI_K3_DECODE_KTP requires host input lengths")

    local.cu_seqlens = torch.arange(
        layout.local_batch + 1, dtype=torch.int32, device=device
    )
    local.cu_seqlens_host = torch.arange(layout.local_batch + 1, dtype=torch.int32)
    sequence_host = getattr(local, "sequence_lengths_host", None)
    if sequence_host is None or not sequence_host.numel():
        raise ValueError("KIMI_K3_DECODE_KTP requires host sequence lengths")
    cu_kv = [0]
    for length in sequence_host.tolist():
        cu_kv.append(cu_kv[-1] + int(length))
    local.cu_kv_seqlens = torch.tensor(cu_kv, dtype=torch.int32, device=device)
    local.padding_offset = torch.zeros(
        layout.local_batch, dtype=torch.int32, device=device
    )
    local.total_tokens = layout.local_batch
    local.context_total_kv_length = cu_kv[-1]

    for name in (
        "kv_cache_block_id_host",
        "kv_cache_kernel_block_id_host",
        "kv_cache_kernel_block_id_device",
    ):
        value = getattr(attention_inputs, name, None)
        if value is None or not value.numel():
            continue
        batch_dim = 1 if name == "kv_cache_block_id_host" and value.ndim == 3 else 0
        setattr(local, name, _maybe_owner_rows(value, layout, dim=batch_dim))

    for name in (
        "kv_cache_block_id_host_by_group",
        "kv_cache_kernel_block_id_host_by_group",
        "kv_cache_kernel_block_id_device_by_group",
    ):
        values = getattr(attention_inputs, name, None)
        if values is not None:
            setattr(local, name, _owner_group_rows(values, layout))
    return local


__all__ = [
    "DecodeOwnerLayout",
    "KtpBatchPlan",
    "KtpStepDescriptor",
    "KdaParallelContext",
    "build_owner_attention_inputs",
    "decode_ktp_enabled",
    "logical_tp1_parallelism",
    "rendezvous_ktp_step",
]
