"""Forward-owned compact CP transport after native stage admission."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from ._v41_compact_cp_plan import (
    PRODUCERS,
    WIRE_VERSION,
    CompactCPPlan,
    resolve_geometry,
)
from ._v41_compact_cp_producer import produce_owner_rows, project_owner
from ._v41_compact_cp_transport import pack, prepare_metadata_batch, unpack

SHARED_KEY = "v41_compact_cp_forward"


@dataclass(frozen=True)
class Producer:
    attention: object
    plans: tuple
    metadata: tuple
    main_cache: object
    index_cache: object
    main_slots: torch.Tensor
    index_slots: torch.Tensor
    state_pool: torch.Tensor | None
    state_slots: torch.Tensor | None


@dataclass
class PendingGather:
    producer: Producer
    send: torch.Tensor
    receive: torch.Tensor
    work: object = None
    completion: object = None

    def launch(self):
        from rtp_llm.models_py.distributed import collective_torch
        from rtp_llm.models_py.distributed.collective_torch import Group

        from .attention_v41 import _get_cp_comm_stream

        stream = _get_cp_comm_stream(self.send.device)
        stream.wait_stream(torch.cuda.current_stream(self.send.device))
        # Keep allocator ownership even if enqueue or Work.wait raises.
        self.send.record_stream(stream)
        self.receive.record_stream(stream)
        with torch.cuda.stream(stream):
            self.work = torch.distributed.all_gather_into_tensor(
                self.receive,
                self.send,
                group=collective_torch._get_group(Group.TP),
                async_op=True,
            )
            self.work.wait()
            self.completion = torch.cuda.Event()
            self.completion.record(stream)

    def wait(self):
        if self.work is None:
            raise RuntimeError("Compact CP enqueue failed; worker teardown required")
        current = torch.cuda.current_stream(self.receive.device)
        if self.completion is not None:
            current.wait_event(self.completion)
        self.work.wait()
        return self.receive


class CompactCPForward:
    def __init__(self, mode, reason, dispatch, projection=None, producers=()):
        self.mode, self.reason, self.dispatch = mode, reason, dispatch
        self.projection = projection
        self.producers = tuple(producers)
        self.next_producer = 0
        self.active = None
        self.closed = False
        self.send = self.receive = None
        self.send_bytes = self.receive_bytes = 0
        if mode != "off":
            device = dispatch.input_ids.device
            elements = (
                1024 * 1024
                if mode == "projected"
                else max(p.plans[0].send_bytes for p in producers)
            )
            dtype = torch.float32 if mode == "projected" else torch.uint8
            self.send = torch.empty(elements, device=device, dtype=dtype)
            self.receive = torch.empty(4 * elements, device=device, dtype=dtype)

    @property
    def signature(self):
        return (
            "compact-cp",
            WIRE_VERSION,
            self.mode,
            self.reason,
            tuple(
                (
                    p.plans[0].layer,
                    p.plans[0].ratio,
                    p.plans[0].rows,
                    p.plans[0].send_bytes,
                )
                for p in self.producers
            ),
        )

    def start(self, attention, x):
        if (
            self.closed
            or self.mode == "off"
            or self.active is not None
            or self.next_producer >= len(self.producers)
        ):
            raise RuntimeError("Compact CP producer has no admitted transport slot")
        producer = self.producers[self.next_producer]
        if producer.attention is not attention:
            raise RuntimeError("Compact CP producer order differs from admission")
        cp = self.dispatch.cp
        plan = producer.plans[cp.cp_rank]
        owner = attention._owner()
        values, scores = project_owner(self.projection, x, owner, plan.ratio)
        if self.mode == "projected":
            width = 512 * plan.ratio
            send = self.send[: 1024 * width].view(1024, width)
            send[:, :512].copy_(values)
            if scores is not None:
                send[:, 512:].copy_(scores)
            received = self.receive[: 4096 * width].view(4096, width)
        else:
            metadata = producer.metadata[cp.cp_rank]
            frequencies = attention.freqs_cis[
                (metadata.positions // plan.ratio) * plan.ratio
            ]
            produced = produce_owner_rows(
                values,
                scores,
                owner,
                attention.eps,
                frequencies,
                attention.rope_head_dim,
                plan.ratio,
                keep_tail=bool(plan.tail_count),
            )
            send = pack(
                plan,
                metadata,
                produced.main,
                produced.index_payload,
                produced.index_scales,
                produced.state_tail,
                out=self.send[: plan.send_bytes],
            )
            received = self.receive[: 4 * plan.send_bytes]
        self.active = PendingGather(producer, send, received)
        self.active.launch()
        self.send_bytes += send.numel() * send.element_size()
        self.receive_bytes += received.numel() * received.element_size()

    def finish(self, attention):
        from rtp_llm.models_py.modules.dsv4.cp import _cp_restore_gathered_full_2d

        if self.active is None or self.active.producer.attention is not attention:
            raise RuntimeError("Compact CP completion has no matching producer")
        producer = self.active.producer
        received = self.active.wait()
        if self.mode == "projected":
            result = _cp_restore_gathered_full_2d(received, self.dispatch.cp)
        else:
            size = producer.plans[0].send_bytes
            main, index, state = unpack(
                producer.plans, producer.metadata, received.view(4, size)
            )
            producer.main_cache.scatter(main, producer.main_slots)
            producer.index_cache.scatter(index, producer.index_slots)
            if state is not None:
                producer.state_pool.index_copy_(0, producer.state_slots, state)
            result = None
        self.active = None
        self.next_producer += 1
        return result

    def drain(self):
        if self.active is not None:
            self.active.wait()
            self.active = None
        self.closed = True


def prepare(worker, dispatch, execution_round):
    if getattr(worker, "_compact_cp_failed", None) is not None:
        raise RuntimeError("Compact CP has an undrained failure; replace the worker")
    mode = getattr(worker, "_cp_compact_mode", "off")
    if mode == "off":
        return CompactCPForward("off", "disabled", dispatch)

    from rtp_llm.models_py.distributed import collective_torch
    from rtp_llm.models_py.distributed.collective_torch import Group
    from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    from .attention import bind_attn_cache

    device = worker._weight_device()
    mode, reason = resolve_geometry(
        mode,
        dispatch.cp,
        dispatch.spans,
        stage=worker._v4_args.pp_rank,
        generation_count=worker._gen_num_per_cycle,
        device_type=device.type,
        capability=torch.cuda.get_device_capability(device),
    )
    if mode == "off":
        return CompactCPForward(mode, reason, dispatch)
    kind = getattr(rtp_llm_ops, "V41CompactCpProjection", None)
    if kind is None or not kind.supported(device.index):
        return CompactCPForward("off", "projection-provider", dispatch)
    group = collective_torch._get_group(Group.TP)
    cp = dispatch.cp
    if (
        torch.distributed.get_backend(group) != "nccl"
        or torch.distributed.get_world_size(group) != 4
        or torch.distributed.get_rank(group) != cp.cp_rank
        or tuple(torch.distributed.get_process_group_ranks(group)) != (0, 1, 2, 3)
    ):
        raise ValueError("Compact CP requires the registered stage-zero CP4 group")
    projection = getattr(worker, "_compact_cp_projection", None)
    if projection is None:
        projection = kind(device.index)
        worker._compact_cp_projection = projection
    span = dispatch.spans[0]
    identity = span.identity
    layers = {layer.attn.layer_id: layer.attn for layer in worker.v4.layers}
    plan_groups = tuple(
        tuple(
            CompactCPPlan(
                mode,
                identity.request_id,
                identity.generation,
                identity.cache_incarnation,
                execution_round,
                span.chunk_id,
                span.start,
                rank,
                layer_id,
                ratio,
            )
            for rank in range(4)
        )
        for layer_id, ratio in PRODUCERS
    )
    all_metadata = prepare_metadata_batch(
        tuple(plan for group in plan_groups for plan in group), device
    )
    state_end = all_metadata[0].tail_identity[-1, :1] + 1
    producers = []
    for index, ((layer_id, ratio), plans) in enumerate(zip(PRODUCERS, plan_groups)):
        attention = layers[layer_id]
        owner = attention._owner()
        weights = (
            (owner.global_wkv, owner.global_wgate)
            if ratio == 2
            else (owner.global_wkv,)
        )
        if (
            not attention.is_kv_source
            or attention.compress_ratio != ratio
            or any(
                w.shape != (512, 5120)
                or w.dtype != torch.bfloat16
                or w.device != device
                or not w.is_contiguous()
                or w.data_ptr() % 256
                for w in weights
            )
        ):
            raise ValueError("Compact CP producer differs from the checkpoint contract")
        metadata = all_metadata[index * 4 : (index + 1) * 4]
        positions = torch.cat([meta.positions for meta in metadata])
        requests = torch.zeros_like(positions)
        with bind_attn_cache(attention, worker.kv_cache, dispatch.tables, cp_ctx=cp):
            main_tag = attention._global_region()
            main = V41PagedCache(
                attention._source_layer_cache(main_tag).kv_cache_base, main_tag, 0
            )
            index = V41PagedCache(
                attention._source_layer_cache("indexer_kv").kv_cache_base,
                "indexer_kv",
                0,
            )
            main_slots = attention._slots(main_tag, positions, requests)
            index_slots = attention._slots("indexer_kv", positions, requests)
            state_pool = state_slots = None
            if ratio == 2:
                tail_positions = metadata[0].tail_identity[:, 0].contiguous()
                state_pool = attention._source_pool("csa_state")
                state_slots = attention._slots(
                    "csa_state",
                    tail_positions,
                    torch.zeros_like(tail_positions),
                    state_end=state_end,
                )
            producers.append(
                Producer(
                    attention,
                    plans,
                    metadata,
                    main,
                    index,
                    main_slots,
                    index_slots,
                    state_pool,
                    state_slots,
                )
            )
    return CompactCPForward(mode, reason, dispatch, projection, producers)


@contextmanager
def execution_scope(worker, prepared):
    shared = worker.v4.layers[0].attn._shared_attention
    if SHARED_KEY in shared:
        raise RuntimeError("Compact CP forward scope is already active")
    if prepared.mode != "off":
        shared[SHARED_KEY] = prepared
    try:
        yield
        if prepared.next_producer != len(prepared.producers):
            raise RuntimeError("Compact CP forward did not complete every producer")
    finally:
        try:
            prepared.drain()
        finally:
            shared.pop(SHARED_KEY, None)
            if prepared.active is not None:
                worker._compact_cp_failed = prepared
            worker._compact_cp_last_stats = {
                "mode": prepared.mode,
                "reason": prepared.reason,
                "producers": prepared.next_producer,
                "send_bytes": prepared.send_bytes,
                "receive_bytes": prepared.receive_bytes,
                "transport_scratch_bytes": sum(
                    value.numel() * value.element_size()
                    for value in (prepared.send, prepared.receive)
                    if value is not None
                ),
                "drained": prepared.active is None,
            }
