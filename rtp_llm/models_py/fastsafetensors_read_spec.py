import copy
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, Type

import torch

from rtp_llm.models_py.read_spec import ReadSpec

ReadSpecResolver = Callable[[str, Tuple[int, ...]], ReadSpec]


def validate_ep_process_group(
    process_group, parallelism, ep_size: int, ep_rank: int
) -> None:
    """Prove EP transport belongs to this complete PP stage, never WORLD.

    Matching size/rank alone is insufficient: two equal-sized stages own
    different checkpoint layers and must not exchange expert payloads.
    """
    from rtp_llm.models_py.distributed.rank_layout import Group, RankLayout

    layout = RankLayout.from_parallelism_config(parallelism)
    world_rank = torch.distributed.get_rank()
    if torch.distributed.get_world_size() != layout.world_size():
        raise RuntimeError("FST EP WORLD does not match RankLayout")
    if ep_size != layout.lane_stride():
        raise RuntimeError("FST EP requires ep_size to span one complete PP stage")
    coord = layout.coord_of(world_rank)
    if (
        parallelism.world_rank != world_rank
        or parallelism.pp_rank != coord.pp
        or parallelism.dp_rank != coord.dp
        or parallelism.tp_rank != coord.tp
        or ep_rank != layout.rank_in_group(Group.STAGE, world_rank)
    ):
        raise RuntimeError("FST EP rank coordinates do not match this PP stage")
    members = tuple(
        torch.distributed.get_global_rank(process_group, rank)
        for rank in range(torch.distributed.get_world_size(process_group))
    )
    expected = tuple(layout.group_of(Group.STAGE, world_rank))
    if members != expected or torch.distributed.get_rank(process_group) != ep_rank:
        raise RuntimeError(
            "FST EP process group does not match the owning PP stage: "
            f"members={members}, expected={expected}, ep_rank={ep_rank}"
        )


def validate_tp_process_group(process_group, tp_size: int, tp_rank: int) -> None:
    """Validate both the local partition and its global checkpoint consumers.

    A group with the right size/local rank can still join different DP replicas.
    Validate membership before FST creates its reader or enters any collective;
    otherwise translating peer ranks alone would faithfully communicate with the
    wrong replica. TP groups follow the runtime's contiguous DP-major layout.
    """
    group_size = torch.distributed.get_world_size(process_group)
    group_rank = torch.distributed.get_rank(process_group)
    if group_size != tp_size:
        raise RuntimeError(
            "FST TP process group size does not match NewLoader: "
            f"group_size={group_size}, tp_size={tp_size}"
        )
    if group_rank != tp_rank:
        raise RuntimeError(
            "FST TP process group rank does not match NewLoader: "
            f"group_rank={group_rank}, tp_rank={tp_rank}"
        )
    world_rank = torch.distributed.get_rank()
    first_rank = world_rank // tp_size * tp_size
    expected_ranks = tuple(range(first_rank, first_rank + tp_size))
    actual_ranks = tuple(
        torch.distributed.get_global_rank(process_group, rank)
        for rank in range(group_size)
    )
    if actual_ranks != expected_ranks:
        raise RuntimeError(
            "FST TP process group members do not match the runtime DP replica: "
            f"members={actual_ranks}, expected={expected_ranks}"
        )


_FLOAT8_DTYPES = tuple(
    getattr(torch, name)
    for name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
    if hasattr(torch, name)
)


def _transport_tensor(tensor):
    """Return an FST tensor wrapper with a NCCL-safe view of float8 bytes."""

    raw_tensor = getattr(tensor, "real_tensor", None)
    if (
        not isinstance(raw_tensor, torch.Tensor)
        or raw_tensor.dtype not in _FLOAT8_DTYPES
    ):
        return tensor
    transport = copy.copy(tensor)
    transport.real_tensor = raw_tensor.view(torch.uint8)
    return transport


class _Float8SafeProcessGroup:
    """Preserve float8 payload bits across collectives unsupported by NCCL.

    Some PyTorch/NCCL combinations reject float8 in broadcast, scatter, send,
    and recv. FST's tensor wrappers expose the backing tensor as
    ``real_tensor``; an equal-sized uint8 view makes those collectives portable
    without allocating or converting the payload.
    """

    def __init__(self, process_group):
        self._process_group = process_group

    def __getattr__(self, name):
        return getattr(self._process_group, name)

    def size(self):
        return self._process_group.size()

    def rank(self):
        return self._process_group.rank()

    def _global_rank(self, group_rank):
        """Translate FST's group-local peer rank for PyTorch collectives.

        fastsafetensors addresses peers relative to its process group.  Its
        PyTorch adapter forwards those values through the legacy ``src`` and
        ``dst`` arguments, which are global ranks even when ``group`` is a
        subgroup.  WORLD ranks are unchanged; fake and single-rank groups do
        not expose ``real_pg`` and therefore retain their existing behavior.
        """

        real_pg = getattr(self._process_group, "real_pg", None)
        if real_pg is None:
            return group_rank
        get_global_rank = getattr(torch.distributed, "get_global_rank", None)
        if get_global_rank is None:
            raise RuntimeError(
                "FST subgroup loading requires " "torch.distributed.get_global_rank()"
            )
        return get_global_rank(real_pg, group_rank)

    def broadcast(self, dst, rank):
        return self._process_group.broadcast(
            _transport_tensor(dst), self._global_rank(rank)
        )

    def scatter(self, dst, scatter_list, src):
        return self._process_group.scatter(
            _transport_tensor(dst),
            [_transport_tensor(tensor) for tensor in scatter_list],
            self._global_rank(src),
        )

    def send(self, tensor, dst_rank, tag):
        return self._process_group.send(
            _transport_tensor(tensor), self._global_rank(dst_rank), tag
        )

    def recv(self, tensor, src_rank, tag):
        return self._process_group.recv(
            _transport_tensor(tensor), self._global_rank(src_rank), tag
        )


def _install_float8_safe_process_group(file_buffer):
    if not isinstance(file_buffer.pg, _Float8SafeProcessGroup):
        file_buffer.pg = _Float8SafeProcessGroup(file_buffer.pg)
    return file_buffer.pg


class _TensorTransport:
    """Minimal FST tensor wrapper for routing an already-materialized row."""

    def __init__(self, tensor: torch.Tensor):
        self.real_tensor = tensor


def _stacked_expert_read_specs(
    slice_expander: Any,
    key: str,
    shape: Tuple[int, ...],
    pg_size: int,
    pg_rank: int,
) -> Tuple[ReadSpec, ...]:
    resolver = getattr(slice_expander, "stacked_read_specs", None)
    if not callable(resolver):
        raise RuntimeError(
            "FST stacked expert loading requires placement-aware ReadSpecs"
        )
    read_specs = tuple(resolver(key, shape))
    if len(read_specs) != shape[0]:
        raise RuntimeError(
            f"Stacked expert plan for {key!r} has {len(read_specs)} entries, "
            f"expected {shape[0]}"
        )

    experts_per_rank = shape[0] // pg_size
    expert_ids = []
    for read_spec in read_specs:
        if not isinstance(read_spec, ReadSpec):
            raise TypeError(
                f"Stacked expert plan for {key!r} contains "
                f"{type(read_spec).__name__}, expected ReadSpec"
            )
        if read_spec.source_name != key:
            raise RuntimeError(
                f"Stacked expert ReadSpec source {read_spec.source_name!r} "
                f"does not match {key!r}"
            )
        if read_spec.slices is None or not isinstance(read_spec.slices[0], int):
            raise RuntimeError(
                f"Stacked expert ReadSpec for {key!r} must select one expert row"
            )
        expert_id = read_spec.slices[0]
        if not 0 <= expert_id < shape[0]:
            raise RuntimeError(
                f"Stacked expert ReadSpec for {key!r} selects invalid expert "
                f"{expert_id}"
            )
        expected_owner = expert_id // experts_per_rank
        if (
            read_spec.shard_dim != 0
            or read_spec.shard_count != pg_size
            or read_spec.shard_rank != expected_owner
        ):
            raise RuntimeError(
                f"Stacked expert ReadSpec for {key!r} expert {expert_id} has "
                "an inconsistent canonical shard owner"
            )
        targets = read_spec.target_ranks
        if targets is None or not targets or any(rank >= pg_size for rank in targets):
            raise RuntimeError(
                f"Stacked expert ReadSpec for {key!r} expert {expert_id} has "
                f"invalid target ranks {targets}"
            )
        if read_spec.load != (pg_rank in targets):
            raise RuntimeError(
                f"Stacked expert ReadSpec for {key!r} expert {expert_id} is "
                f"inconsistent on rank {pg_rank}: load={read_spec.load}, "
                f"target_ranks={targets}"
            )
        expert_ids.append(expert_id)
    if tuple(sorted(expert_ids)) != tuple(range(shape[0])):
        raise RuntimeError(
            f"Stacked expert plan for {key!r} must describe every logical expert "
            "exactly once"
        )
    return read_specs


def _route_stacked_expert_rows(
    pg: _Float8SafeProcessGroup,
    key: str,
    local_experts: torch.Tensor,
    read_specs: Sequence[ReadSpec],
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Route only non-canonical expert replicas after the FST scatter."""

    pg_rank = pg.rank()
    pg_size = pg.size()
    experts_per_rank = len(read_specs) // pg_size
    expected_start = pg_rank * experts_per_rank
    rows: Dict[int, torch.Tensor] = {}

    def expert_id(read_spec: ReadSpec) -> int:
        assert read_spec.slices is not None
        value = read_spec.slices[0]
        assert isinstance(value, int)
        return value

    def canonical_row(logical_id: int) -> torch.Tensor:
        local_index = logical_id - expected_start
        if not 0 <= local_index < local_experts.shape[0]:
            raise RuntimeError(
                f"Rank {pg_rank} does not own canonical expert {logical_id} "
                f"from stacked tensor {key!r}"
            )
        return local_experts[local_index].contiguous()

    for read_spec in read_specs:
        logical_id = expert_id(read_spec)
        if read_spec.load and read_spec.shard_rank == pg_rank:
            rows[logical_id] = canonical_row(logical_id)

    # Pairwise rank ordering avoids a blocking send/send deadlock.  Every rank
    # derives the same send and receive lists from the immutable ReadSpecs.
    for peer_rank in range(pg_size):
        if peer_rank == pg_rank:
            continue
        sends = [
            read_spec
            for read_spec in read_specs
            if read_spec.shard_rank == pg_rank
            and read_spec.target_ranks is not None
            and peer_rank in read_spec.target_ranks
        ]
        receives = [
            read_spec
            for read_spec in read_specs
            if read_spec.shard_rank == peer_rank
            and read_spec.target_ranks is not None
            and pg_rank in read_spec.target_ranks
        ]

        def send_rows() -> None:
            for read_spec in sends:
                logical_id = expert_id(read_spec)
                pg.send(
                    _TensorTransport(canonical_row(logical_id)),
                    peer_rank,
                    logical_id,
                )

        def receive_rows() -> None:
            for read_spec in receives:
                logical_id = expert_id(read_spec)
                row = torch.empty_like(local_experts[0])
                pg.recv(_TensorTransport(row), peer_rank, logical_id)
                rows[logical_id] = row

        if pg_rank < peer_rank:
            send_rows()
            receive_rows()
        else:
            receive_rows()
            send_rows()

    for read_spec in read_specs:
        if not read_spec.load:
            continue
        logical_id = expert_id(read_spec)
        row = rows.get(logical_id)
        if row is None:
            raise RuntimeError(
                f"Stacked expert {logical_id} from {key!r} was not routed to "
                f"rank {pg_rank}"
            )
        yield read_spec.output_name, row


def iter_read_spec_tensors(
    file_buffer,
    keys,
    read_spec_resolver: ReadSpecResolver,
    safetensor_slice_expander: Optional[Any] = None,
    segmented_tensor_reader=None,
    name_filter=None,
    wo_a_trace=None,
) -> Iterator[Tuple[str, object]]:
    """Consume one FST file buffer using its routing/sharding primitives.

    ``get_sharded`` performs a source-side split followed by scatter.  It does
    not make FST's file copier range-aware, but it prevents every non-source
    rank from materializing the full tensor. ``push_tensor`` similarly routes
    a per-expert key directly when it has one owning EP rank. Stacked expert
    tensors use ``get_sharded(dim=0)`` before expansion into per-expert keys;
    placement-aware ReadSpecs then route only additional redundant rows.
    """

    pg = _install_float8_safe_process_group(file_buffer)
    pg_size = pg.size()
    pg_rank = pg.rank()
    all_ranks = tuple(range(pg_size))

    for key in keys:
        shape = tuple(file_buffer.get_shape(key))
        if (
            safetensor_slice_expander is not None
            and safetensor_slice_expander.handles_safetensor(key)
        ):
            # The shared-file planner verifies identical selected physical keys
            # across ranks before scheduling this collective.
            if name_filter is not None and not name_filter(key):
                continue
            experts_per_rank = (
                safetensor_slice_expander.num_experts
                // safetensor_slice_expander.ep_size
            )
            expected_start = pg_rank * experts_per_rank
            if (
                safetensor_slice_expander.ep_size != pg_size
                or safetensor_slice_expander.start_expert != expected_start
            ):
                raise RuntimeError(
                    f"FST stacked expert tensor {key!r} requires one WORLD rank "
                    "per EP rank with matching rank order"
                )
            if not hasattr(file_buffer, "get_sharded"):
                raise RuntimeError(
                    "fastsafetensors does not expose get_sharded required for "
                    f"stacked expert tensor {key!r}"
                )
            local_experts = file_buffer.get_sharded(key, 0)
            if not isinstance(local_experts, torch.Tensor):
                raise RuntimeError(f"FST did not return stacked expert tensor {key!r}")
            if local_experts.shape[0] != experts_per_rank:
                raise RuntimeError(
                    f"FST stacked expert shard {key!r} has "
                    f"{local_experts.shape[0]} experts, expected {experts_per_rank}"
                )
            read_specs = _stacked_expert_read_specs(
                safetensor_slice_expander, key, shape, pg_size, pg_rank
            )
            yield from _route_stacked_expert_rows(pg, key, local_experts, read_specs)
            continue

        read_spec = read_spec_resolver(key, shape)
        if read_spec.segments is not None:
            if not read_spec.load or (name_filter is not None and not name_filter(key)):
                continue
            if read_spec.target_ranks is not None:
                if (
                    any(rank >= pg_size for rank in read_spec.target_ranks)
                    or pg_rank not in read_spec.target_ranks
                ):
                    raise RuntimeError(
                        "Segmented FST ReadSpec has inconsistent target ranks"
                    )
            if segmented_tensor_reader is None:
                raise RuntimeError(
                    "Segmented FST ReadSpec requires native range reader; full tensor fallback is forbidden"
                )
            # No collective for this key on any rank. Empty/filtered ranks
            # can skip independently, while all other keys keep the old path.
            yield read_spec.output_name, segmented_tensor_reader(key, read_spec)
            continue
        targets = read_spec.target_ranks
        if targets is not None:
            if any(rank >= pg_size for rank in targets):
                raise ValueError(
                    f"ReadSpec for {key!r} targets ranks {targets}, but FST "
                    f"process group has size {pg_size}"
                )
            should_load = pg_rank in targets
            if read_spec.load != should_load:
                raise RuntimeError(
                    f"ReadSpec for {key!r} is inconsistent on rank {pg_rank}: "
                    f"load={read_spec.load}, target_ranks={targets}"
                )
            if not targets:
                continue
            can_route = hasattr(file_buffer, "push_tensor")
            # FST marks a tensor consumed after ``push_tensor`` and may free
            # its file buffer immediately.  A single targeted push is safe;
            # repeated pushes to TP/DP replicas are not.  Keep the latter on
            # the broadcast fallback until FST exposes multicast/subgroups.
            if len(targets) == 1 and targets != all_ranks and can_route:
                target_rank = targets[0]
                tensor = file_buffer.push_tensor(key, target_rank)
                if pg_rank == target_rank:
                    if not isinstance(tensor, torch.Tensor):
                        raise RuntimeError(
                            f"FST did not return routed tensor {key!r} on "
                            f"target rank {target_rank}"
                        )
                    yield read_spec.output_name, read_spec.materialize(tensor)
                continue
            if targets != all_ranks:
                tensor = file_buffer.get_tensor(key)
                if read_spec.load:
                    yield read_spec.output_name, read_spec.materialize(tensor)
                continue

        if not read_spec.load:
            continue
        can_scatter = (
            read_spec.shard_dim is not None
            and read_spec.shard_count == pg_size
            and read_spec.shard_rank == pg_rank
            and targets in (None, all_ranks)
            and hasattr(file_buffer, "get_sharded")
        )
        if wo_a_trace is not None and key == wo_a_trace._wo_a_key:
            yield read_spec.output_name, wo_a_trace.observe_wo_a_reader(
                file_buffer, key, read_spec, can_scatter
            )
            continue
        if can_scatter:
            tensor = file_buffer.get_sharded(key, read_spec.shard_dim)
            yield read_spec.output_name, read_spec.wrap_sharded_tensor(tensor)
            continue

        tensor = file_buffer.get_tensor(key)
        yield read_spec.output_name, read_spec.materialize(tensor)


def build_read_spec_parallel_loader(
    parallel_loader: Type, timing_context: Type
) -> Type:
    """Build a version-local ParallelLoader subclass without eager FST import."""

    class ReadSpecParallelLoader(parallel_loader):
        def __init__(
            self,
            *args,
            read_spec_resolver: ReadSpecResolver,
            safetensor_slice_expander: Optional[Any] = None,
            segmented_tensor_reader=None,
            name_filter=None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.read_spec_resolver = read_spec_resolver
            self.safetensor_slice_expander = safetensor_slice_expander
            self.segmented_tensor_reader = segmented_tensor_reader
            self.name_filter = name_filter

        def _consume_single_batch(self):
            with timing_context("wait_queue", self._log_message) as timer:
                batch_item = self.batch_queue.get()
                if self.queue_size == 0 and self.consumer_processed is not None:
                    self.consumer_processed.set()
                if batch_item is None:
                    self._log_error("get batch_item is None, will break")
                    return
                if isinstance(batch_item, Exception):
                    self._log_error("get batch_item is Exception, will raise")
                    raise batch_item
                batch = batch_item
                timer.batch_id = batch.batch_id
            queue_wait_time = timer.elapsed_ms

            if queue_wait_time / 1000 > 10:
                self._log_message(
                    f"Warning: Batch {batch.batch_id}: Queue wait took "
                    f"{queue_wait_time:.3f} ms",
                    is_error=True,
                )

            try:
                diag = getattr(self, "_rtp_v4_fst_trace", None)
                if diag is not None:
                    diag.point("consumer_received", batch.batch_id)
                self._log_message(
                    f"Batch {batch.batch_id}: tensor key len: {len(batch.keys)}"
                )
                key_context = (
                    diag.watch_keys(batch.batch_id, batch.keys)
                    if diag is not None
                    else nullcontext(batch.keys)
                )
                with timing_context(
                    "get_tensor", self._log_message, batch.batch_id
                ) as timer, key_context as keys:
                    yield from iter_read_spec_tensors(
                        batch.fb,
                        keys,
                        self.read_spec_resolver,
                        self.safetensor_slice_expander,
                        self.segmented_tensor_reader,
                        self.name_filter,
                        (
                            diag
                            if diag is not None and hasattr(diag, "_wo_a_local")
                            else None
                        ),
                    )
                get_tensor_time = timer.elapsed_ms
                if diag is not None:
                    diag.point("consumer_before_release", batch.batch_id)
            finally:
                with timing_context(
                    "fb.close", self._log_message, batch.batch_id
                ) as timer:
                    batch.fb.close()
                close_time = timer.elapsed_ms
            if diag is not None:
                diag.point("consumer_released", batch.batch_id)

            self._log_message(
                f"Batch {batch.batch_id} summary: "
                f"add_filenames={batch.add_filenames_time:.3f}ms, "
                f"copy_files={batch.copy_files_time:.3f}ms, "
                f"get_tensor={get_tensor_time:.3f}ms, "
                f"close={close_time:.3f}ms"
            )
            if self.queue_size < 0 and self.consumer_processed is not None:
                self.consumer_processed.set()

    ReadSpecParallelLoader.__name__ = "ReadSpecParallelLoader"
    return ReadSpecParallelLoader
