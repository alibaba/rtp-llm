"""Drive one complete Qwen3.8 split-refit stream with worker-owned CUDA leases."""

import asyncio
from dataclasses import dataclass

from q38_native32_control import SourceLease
from q38_refit_dense import (
    DEFAULT_MAX_DENSE_BYTES, DenseNativeAck, check_dense_spec, cuda_dense_receiver,
    cuda_dense_sender, validate_dense_records,
)
from q38_refit_groups import OwnedRefitGroups, normalize_uuid
from qwen38_ep32_transport import (
    MAX_PACKED_EXPERT_BYTES, ExpertFragment, ExpertNativeAck, ExpertPair, ExpertReceived,
    cuda_expert_receiver, cuda_expert_sender,
)
from rtp_llm.utils.qwen38_expert_shards import PACKED_SHAPES, ROLLOUT_EP, TRAIN_EP


EXPERT_ORDER = tuple(sorted(
    PACKED_SHAPES, key=lambda name: (int(name.split(".")[2]), name.rsplit(".", 1)[1])
))
TRAIN_TP = 8


@dataclass(frozen=True)
class SourceCoordinate:
    transaction: str
    step: int
    ordinal: int
    name: str


class TrainRefitWorker:
    """Install inside an assigned Megatron actor after its live model is bound."""

    def __init__(self, plan, rank, shapes, expert_weights, dense_weights,
                 max_dense_bytes=DEFAULT_MAX_DENSE_BYTES):
        if type(rank) is not int or not 0 <= rank < TRAIN_EP:
            raise ValueError("Invalid training refit rank")
        self.rank = rank
        self.groups = OwnedRefitGroups(plan, rank)
        self.expert = cuda_expert_sender(self.groups.experts, rank)
        self.dense = cuda_dense_sender(self.groups.dense, shapes, max_dense_bytes) if rank == 0 else None
        self.expert_weights = expert_weights
        self.dense_weights = dense_weights
        self.epoch = None
        self.last_step = -1

    def begin_step(self, transaction, step):
        import torch

        if self.epoch is not None or step != self.last_step + 1 or not transaction:
            raise RuntimeError("Unexpected training refit generation")
        torch.cuda.synchronize()
        self.epoch = (transaction, step)
        self.expert_iter = iter(self.expert_weights())
        self.dense_iter = iter(self.dense_weights()) if self.rank < TRAIN_TP else None

    def _check_epoch(self, transaction, step):
        if self.epoch != (transaction, step):
            raise ValueError("Training source request has the wrong transaction")

    def expert_prepare(self, coordinate):
        self._check_epoch(coordinate.transaction, coordinate.step)
        name, tensor = next(self.expert_iter)
        if name != coordinate.name:
            raise ValueError("Live expert export differs from the canonical per-layer order")
        return self.expert.prepare(transaction=coordinate.transaction, step=coordinate.step,
                                   ordinal=coordinate.ordinal, name=name, tensor=tensor)

    def expert_send(self, fragment):
        return self.expert.send(fragment)

    def expert_ack(self, ack):
        return self.expert.native_ack(ack)

    def expert_finish(self, transaction, step):
        self._check_epoch(transaction, step)
        if self.expert.state != "idle" or next(self.expert_iter, None) is not None:
            raise RuntimeError("Training expert export did not drain exactly")
        self.expert_iter = None

    def dense_prepare(self, transaction, step, ordinal):
        self._check_epoch(transaction, step)
        if self.rank != 0 or self.expert_iter is not None:
            raise RuntimeError("Only trainer zero may export a dense source after experts")
        name, tensor = next(self.dense_iter)
        return self.dense.prepare(transaction, step, ordinal, name, tensor)

    def dense_send(self, spec):
        return self.dense.send(spec)

    def dense_ack(self, acknowledgements):
        return self.dense.native_ack(acknowledgements)

    def dense_finish(self, transaction, step):
        """DP0 peers drain the Bridge collectives and yield no source tensors."""
        self._check_epoch(transaction, step)
        if self.rank >= TRAIN_TP or self.expert_iter is not None:
            raise RuntimeError("Dense Bridge participation belongs to the first TP group")
        if self.rank == 0 and self.dense.state != "idle":
            raise RuntimeError("Dense sender still holds a source lease")
        if next(self.dense_iter, None) is not None:
            raise RuntimeError("Unexpected source after canonical dense export")
        self.dense_iter = None

    def finish_step(self, transaction, step):
        self._check_epoch(transaction, step)
        if self.expert_iter is not None or self.dense_iter is not None:
            raise RuntimeError("Training export is still open")
        self.epoch = None
        self.last_step = step

    def close(self):
        if self.epoch is not None:
            raise RuntimeError("Refit generation still active or held")
        self.groups.close()


class RolloutRefitWorker:
    """One relay per RTP rank, bound to the same physical GPU as its consumer."""

    def __init__(self, plan, rollout_rank, native_incarnation, shapes,
                 max_dense_bytes=DEFAULT_MAX_DENSE_BYTES):
        if type(rollout_rank) is not int or not 0 <= rollout_rank < ROLLOUT_EP:
            raise ValueError("Invalid rollout refit rank")
        self.groups = OwnedRefitGroups(plan, TRAIN_EP + rollout_rank)
        identity = dict(gpu_uuid=plan.gpu_uuids[TRAIN_EP + rollout_rank],
                        native_incarnation=native_incarnation)
        self.expert = cuda_expert_receiver(self.groups.experts, rollout_rank, **identity)
        self.dense = cuda_dense_receiver(self.groups.dense, shapes, rollout_rank,
                                         max_bytes=max_dense_bytes, **identity)

    def expert_prepare(self, pair):
        if self.dense.slot.state != "idle":
            raise RuntimeError("Dense receive lease is still active")
        return self.expert.prepare(pair)

    def expert_receive(self):
        return self.expert.receive()

    def expert_ack(self, ack):
        return self.expert.native_ack(ack)

    def dense_prepare(self, spec):
        if self.expert.slot.state != "idle":
            raise RuntimeError("Expert receive lease is still active")
        return self.dense.prepare(spec)

    def dense_receive(self, spec):
        return self.dense.receive(spec)

    def dense_ack(self, ack):
        return self.dense.native_ack(ack)

    def close(self):
        if self.dense.slot.state != "idle" or self.expert.slot.state != "idle":
            raise RuntimeError("Native receive lease is still active or held")
        self.groups.close()


async def ray_refit_call(worker, method, *args):
    return await getattr(worker, method).remote(*args)


class SplitRefitStream:
    def __init__(self, trainers, receivers, native_incarnation, source_shapes, rollout_uuids,
                 *, call=ray_refit_call, operation_timeout=720,
                 max_dense_bytes=DEFAULT_MAX_DENSE_BYTES):
        self.trainers = tuple(trainers)
        self.receivers = tuple(receivers)
        self.gpu_uuids = tuple(normalize_uuid(value) for value in rollout_uuids)
        self.incarnation = native_incarnation
        self.shapes = {name: tuple(shape) for name, shape in source_shapes.items()}
        if (len(self.trainers) != TRAIN_EP or len(self.receivers) != ROLLOUT_EP
                or len(self.gpu_uuids) != ROLLOUT_EP or len(set(self.gpu_uuids)) != ROLLOUT_EP
                or not isinstance(self.incarnation, str) or not self.incarnation):
            raise ValueError("Refit requires all training workers and native GPU identities")
        if (len(self.shapes) != 1590 or any(name.startswith("mtp.") for name in self.shapes)
                or any(self.shapes.get(name) != shape for name, shape in PACKED_SHAPES.items())):
            raise ValueError("Refit requires the complete pinned main-policy source shapes")
        if not 0 < operation_timeout <= 1800 or not 0 < max_dense_bytes <= DEFAULT_MAX_DENSE_BYTES:
            raise ValueError("Refit operations and dense receives must be bounded")
        self.call = call
        self.timeout = operation_timeout
        self.max_dense_bytes = max_dense_bytes
        self.pending = set()
        self.held = None
        self.active = False
        self.poisoned = False
        self.closed = False
        self.last_step = -1

    def _submit(self, calls):
        tasks = []
        for worker, method, args in calls:
            task = asyncio.ensure_future(self.call(worker, method, *args))
            tasks.append(task)
            self.pending.add(task)
        return tasks

    async def _wait(self, tasks):
        if not tasks:
            return []
        done, still_running = await asyncio.wait(tasks, timeout=self.timeout,
                                                 return_when=asyncio.FIRST_EXCEPTION)
        # Preserve references to running operations on timeout or failure.
        for task in done:
            task.result()
        if still_running:
            raise TimeoutError("Split-refit workers did not acknowledge the operation")
        self.pending.difference_update(tasks)
        return [task.result() for task in tasks]

    async def _call(self, calls):
        return await self._wait(self._submit(calls))

    def _check_prepared(self, records, specs, max_bytes):
        for rank, (row, spec) in enumerate(zip(records, specs)):
            actual = (
                row["rank"], normalize_uuid(row["gpu_uuid"]), row["native_incarnation"],
                row["transaction"], row["step"], row["ordinal"], row["name"], row["sha256"],
                row["size_bytes"], row["prepared"],
            )
            expected = (
                rank, self.gpu_uuids[rank], self.incarnation, spec.transaction, spec.step,
                spec.ordinal, spec.name, spec.sha256, spec.validate(max_bytes), True,
            )
            if actual != expected:
                raise ValueError("Receiver prepare acknowledgement differs from its native source and GPU")

    async def _expert_lease(self, coordinate):
        fragments = await self._call(
            (worker, "expert_prepare", (coordinate,)) for worker in self.trainers
        )
        expected = (coordinate.transaction, coordinate.step, coordinate.ordinal, coordinate.name)
        for rank, fragment in enumerate(fragments):
            if (not isinstance(fragment, ExpertFragment) or fragment.train_rank != rank
                    or (fragment.transaction, fragment.step, fragment.ordinal, fragment.name) != expected):
                raise ValueError("Expert fragment differs from its training owner or source coordinate")
        pairs = tuple(ExpertPair(rank, tuple(fragments[2*rank:2*rank+2])) for rank in range(ROLLOUT_EP))
        prepared = await self._call(
            (worker, "expert_prepare", (pair,)) for worker, pair in zip(self.receivers, pairs)
        )
        self._check_prepared(prepared, [pair.spec for pair in pairs], MAX_PACKED_EXPERT_BYTES)
        routes = [(worker, "expert_receive", ()) for worker in self.receivers]
        routes += [(worker, "expert_send", (fragment,)) for worker, fragment in zip(self.trainers, fragments)]
        records = tuple((await self._call(routes))[:ROLLOUT_EP])
        for rank, record in enumerate(records):
            if (not isinstance(record, ExpertReceived) or record.receipt.pair != pairs[rank]
                    or normalize_uuid(record.receipt.gpu_uuid) != self.gpu_uuids[rank]
                    or record.receipt.native_incarnation != self.incarnation):
                raise ValueError("Expert receive differs from its assigned native transaction or GPU")

        async def release(acks):
            self._check_ack(coordinate, records, acks, ExpertNativeAck)
            calls = [(worker, "expert_ack", (ack,)) for worker, ack in zip(self.receivers, acks)]
            calls += [(worker, "expert_ack", (acks[rank//2],)) for rank, worker in enumerate(self.trainers)]
            await self._call(calls)
            self.held = None

        return SourceLease(coordinate.name, tuple(row.metadata for row in records),
                           tuple(pair.spec.sha256 for pair in pairs), release, expert_records=records)

    def _check_ack(self, coordinate, records, acks, ack_type):
        if (self.held != coordinate or not isinstance(acks, (tuple, list)) or len(acks) != len(records)
                or any(not isinstance(ack, ack_type) or ack.receipt != record.receipt
                       for record, ack in zip(records, acks))):
            self.poisoned = True
            raise RuntimeError("Refit transport did not receive the matching complete native acknowledgement")

    async def _dense_lease(self, coordinate, spec):
        prepared = await self._call((worker, "dense_prepare", (spec,)) for worker in self.receivers)
        self._check_prepared(prepared, (spec,) * ROLLOUT_EP, self.max_dense_bytes)
        routes = [(worker, "dense_receive", (spec,)) for worker in self.receivers]
        routes.append((self.trainers[0], "dense_send", (spec,)))
        records = tuple((await self._call(routes))[:ROLLOUT_EP])
        validate_dense_records(records, self.gpu_uuids, self.incarnation,
                               transaction=coordinate.transaction, step=coordinate.step,
                               ordinal=coordinate.ordinal, name=coordinate.name)
        if any(record.receipt.spec != spec for record in records):
            raise ValueError("Dense receive digest differs from the live training source")

        async def release(acks):
            self._check_ack(coordinate, records, acks, DenseNativeAck)
            calls = [(worker, "dense_ack", (ack,)) for worker, ack in zip(self.receivers, acks)]
            calls.append((self.trainers[0], "dense_ack", (acks,)))
            await self._call(calls)
            self.held = None

        return SourceLease(spec.name, tuple(row.metadata for row in records), (spec.sha256,) * ROLLOUT_EP,
                           release, dense_records=records)

    async def leases(self, step, control):
        if self.active or self.poisoned or self.closed or type(step) is not int or step != self.last_step + 1:
            raise RuntimeError("Invalid split-refit generation or existing held state")
        transaction = control.update_id
        if transaction != f"{self.incarnation}:{step+1}":
            raise RuntimeError("Native BEGIN_UPDATE must succeed before the first transfer")
        self.active = True
        try:
            await self._call((worker, "begin_step", (transaction, step)) for worker in self.trainers)
            for ordinal, name in enumerate(EXPERT_ORDER):
                coordinate = SourceCoordinate(transaction, step, ordinal, name)
                self.held = coordinate
                yield await self._expert_lease(coordinate)
                if self.held is not None:
                    raise RuntimeError("Expert stream advanced before its native ACK")
            await self._call((worker, "expert_finish", (transaction, step)) for worker in self.trainers)
            dense_peers = self._submit((worker, "dense_finish", (transaction, step))
                                       for worker in self.trainers[1:TRAIN_TP])
            dense_remaining = set(self.shapes) - set(EXPERT_ORDER)
            for ordinal in range(len(EXPERT_ORDER), len(self.shapes)):
                for peer in dense_peers:
                    if peer.done():
                        peer.result()
                spec, = await self._call([(self.trainers[0], "dense_prepare", (transaction, step, ordinal))])
                check_dense_spec(spec, self.shapes, self.max_dense_bytes)
                if ((spec.transaction, spec.step, spec.ordinal) != (transaction, step, ordinal)
                        or spec.name not in dense_remaining):
                    raise ValueError("Dense export did not cover a new canonical source for this generation")
                dense_remaining.remove(spec.name)
                coordinate = SourceCoordinate(transaction, step, ordinal, spec.name)
                self.held = coordinate
                yield await self._dense_lease(coordinate, spec)
                if self.held is not None:
                    raise RuntimeError("Dense stream advanced before its native ACK")
            root = self._submit([(self.trainers[0], "dense_finish", (transaction, step))])
            await self._wait(root + dense_peers)
            await self._call((worker, "finish_step", (transaction, step)) for worker in self.trainers)
            self.last_step = step
            self.active = False
        except BaseException:
            self.poisoned = True
            raise

    async def close(self):
        if self.active or self.poisoned or self.closed or self.pending or self.held is not None:
            raise RuntimeError("Refit worker close requires healthy drained leases")
        self.closed = True
        try:
            await self._call((worker, "close", ()) for worker in self.trainers + self.receivers)
        except BaseException:
            self.poisoned = True
            raise
