"""Bounded synchronous CP4 projection exchange for unsharded prefill caches.

The scheduler supplies all four trusted row maps and full-cache plans, and owns
request-generation binding, old pair state and overwrite-safe SWA storage.
Queries, residuals and FFNs stay local. Projected SWA and compressor rows are
gathered before pooling so pair predecessors may belong to another CP rank.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, replace

import torch

from rtp_llm.models_py.modules.dsv4.v41_projection import V41KVInputs, _tensor
from rtp_llm.models_py.modules.dsv4.v41_stage import V41Stage
from rtp_llm.models_py.modules.dsv4.v41_stage_attention import V41ForwardRows


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()


@dataclass(frozen=True)
class V41CPPlan:
    world_ranks: tuple[int, ...]
    pp_rank: int
    generation: str
    rows_per_rank: int
    max_positions: int = 32768

    def __post_init__(self):
        if (
            type(self.world_ranks) is not tuple
            or len(self.world_ranks) != 4
            or len(set(self.world_ranks)) != 4
            or any(type(rank) is not int or rank < 0 for rank in self.world_ranks)
            or type(self.pp_rank) is not int
            or self.pp_rank not in (0, 1)
            or type(self.generation) is not str
            or not self.generation
            or type(self.rows_per_rank) is not int
            or not 1 <= self.rows_per_rank <= 32
            or type(self.max_positions) is not int
            or not 1 <= self.max_positions <= 32768
        ):
            raise ValueError(
                "V4.1 CP requires an exact PP-local CP4 plan with at most 32 rows per rank"
            )

    @property
    def fingerprint(self):
        return _digest({"protocol": "dsv41-unsharded-cp-v1", "plan": asdict(self)})


def _dispatch_rows(peers, plan, device):
    if len(peers) != 4 or any(not isinstance(rows, V41ForwardRows) for rows in peers):
        raise ValueError("V4.1 CP requires four trusted peer row maps")
    peers = tuple(rows.snapshot() for rows in peers)
    invocation = peers[0].invocation_id
    records = []
    for rows in peers:
        if (
            rows.invocation_id != invocation
            or rows.positions.numel() > plan.rows_per_rank
        ):
            raise ValueError("V4.1 CP peers disagree on invocation or row capacity")
        records.append(
            torch.stack(
                (
                    rows.request_ids.long(),
                    rows.positions.long(),
                    rows.valid_rows.long(),
                ),
                -1,
            )
            .cpu()
            .tolist()
        )
    padded = []
    for records_for_rank in records:
        padded.extend(
            records_for_rank
            + [[-1, -1, 0]] * (plan.rows_per_rank - len(records_for_rank))
        )
    values = torch.tensor(padded, dtype=torch.int64, device=device)
    full = V41ForwardRows(
        invocation,
        values[:, 0].contiguous(),
        values[:, 1].contiguous(),
        values[:, 2].bool(),
    ).snapshot()
    if bool((full.positions[full.valid_rows] >= plan.max_positions).any()):
        raise ValueError("V4.1 CP rows exceed the admitted context")
    # The initial exact path exchanges whole contiguous request chunks, without CED.
    for request in full.request_ids[full.valid_rows].unique().tolist():
        positions = (
            full.positions[full.valid_rows & (full.request_ids == request)]
            .sort()
            .values
        )
        if bool((positions[1:] != positions[:-1] + 1).any()):
            raise ValueError("V4.1 CP dispatch omits a row inside the request chunk")
    return peers, full, {"invocation": invocation, "rows": records}


class V41CPExchange:
    """Explicit NCCL group admission and one in-flight eager stage invocation.

    Every member calls admit/begin/run in the same order. Control consensus
    rejects peer metadata or stage-preflight failures before payload gathering.
    Numerical/device/collective failures poison the session and require caller
    worker/group teardown; no fallback, replay or graph capture is provided.
    """

    def __init__(self, plan, process_group, device):
        if not isinstance(plan, V41CPPlan) or process_group is None:
            raise ValueError("V4.1 CP requires an immutable plan and explicit group")
        self.plan, self.process_group, self.device = (
            plan,
            process_group,
            torch.device(device),
        )
        self._identity = None
        self._failed, self._active, self._round = False, None, 0

    def _runtime_identity(self):
        return self.plan, id(self.process_group), self.device

    def _check(self):
        if self._failed or self._identity != self._runtime_identity():
            raise RuntimeError("V4.1 CP requires a healthy admitted group/plan/device")
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("V4.1 CP exchange cannot run during capture")

    def _consensus(self, signature, error=None):
        dist = torch.distributed
        value = torch.tensor(
            [*signature, int(error is None)], dtype=torch.uint8, device=self.device
        ).view(1, 33)
        received = torch.empty(
            dist.get_world_size(self.process_group),
            33,
            dtype=torch.uint8,
            device=self.device,
        )
        try:
            dist.all_gather_into_tensor(received, value, group=self.process_group)
            received = received.cpu()
            if not bool((received[:, :32] == received[0, :32]).all()) or not bool(
                (received[:, 32] == 1).all()
            ):
                raise RuntimeError(
                    "V4.1 CP peer metadata or preflight mismatch"
                ) from error
        except Exception:
            self._failed = True
            raise

    def admit(self):
        dist, group, plan = torch.distributed, self.process_group, self.plan
        if self._failed or self._active is not None:
            raise RuntimeError("V4.1 CP session is failed or still active")
        if not dist.is_initialized() or self.device.type != "cuda":
            raise RuntimeError("V4.1 CP admission requires CUDA and initialized NCCL")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("V4.1 CP admission must precede graph capture")
            matches = (
                dist.get_backend(group) == "nccl"
                and dist.get_world_size(group) == 4
                and tuple(dist.get_process_group_ranks(group)) == plan.world_ranks
                and torch.cuda.get_device_capability(self.device) == (12, 0)
            )
            self._consensus(
                plan.fingerprint,
                None if matches else ValueError("CP membership/device mismatch"),
            )
        self.rank = dist.get_rank(group)
        self._identity = self._runtime_identity()

    def begin(self, peer_rows):
        self._check()
        if self._active is not None:
            raise RuntimeError("V4.1 CP already has an in-flight forward")
        error, signature, prepared = None, bytes(32), None
        try:
            prepared = _dispatch_rows(peer_rows, self.plan, self.device)
            signature = _digest(
                {
                    "plan": self.plan.fingerprint.hex(),
                    "round": self._round,
                    "dispatch": prepared[2],
                }
            )
        except Exception as exc:
            error = exc
        self._consensus(signature, error)
        peers, rows, _ = prepared
        self._active = V41CPForward(self, peers, rows, signature)
        return self._active


class V41CPForward:
    def __init__(self, session, peers, rows, signature):
        self.session, self.rows, self.signature = session, rows, signature
        local = peers[session.rank]
        self.local_rows = V41ForwardRows(
            rows.invocation_id,
            local.request_ids.to(session.device),
            local.positions.to(session.device),
            local.valid_rows.to(session.device),
        ).snapshot()
        begin = session.rank * session.plan.rows_per_rank
        self.local_slice = slice(begin, begin + local.positions.numel())
        self._next_layer = 0 if session.plan.pp_rank == 0 else 21
        self._stream = (
            torch.cuda.current_stream(session.device).cuda_stream
            if session.device.type == "cuda"
            else None
        )
        self._prepared = False

    def _check(self):
        self.session._check()
        if self.session._active is not self:
            raise RuntimeError("V4.1 CP forward is stale or completed")
        if (
            self._stream is not None
            and torch.cuda.current_stream(self.session.device).cuda_stream
            != self._stream
        ):
            raise RuntimeError("V4.1 CP forward must remain on its preparation stream")

    def prepare_stage(self, stage, state, rows, plans, windows, selections):
        self._check()
        error, prepared = None, None
        try:
            if (
                self._prepared
                or not isinstance(stage, V41Stage)
                or (
                    stage.role != "prefill"
                    or stage.pp_rank != self.session.plan.pp_rank
                    or stage.max_positions != self.session.plan.max_positions
                    or stage.device != self.session.device
                    or self.rows.positions.numel()
                    > min(op.kv.max_rows for op in stage.blocks)
                )
            ):
                raise ValueError("V4.1 CP forward does not match its stage")
            rows.require_same(self.local_rows)
            prepared = stage._prepare_forward(
                state, rows, plans, windows, selections, self.rows
            )
        except Exception as exc:
            error = exc
        self.session._consensus(
            _digest([self.signature.hex(), "stage-preflight"]), error
        )
        self._prepared = True
        self._stage, self._plans, self._arguments, self._selections = (
            stage,
            plans,
            prepared,
            selections,
        )
        return prepared

    def exchange(self, projected):
        self._check()
        error, fields, signature = None, (), bytes(32)
        try:
            if not self._prepared or not isinstance(projected, V41KVInputs):
                raise ValueError(
                    "V4.1 CP exchange requires a prepared stage and projected KV"
                )
            sources = projected.owner.sources
            if (
                sources.layer_id != self._next_layer
                or projected.owner
                is not self._stage.blocks[self._next_layer - self._stage.begin].kv
            ):
                raise ValueError("V4.1 CP projection has an out-of-order global layer")
            if not torch.equal(
                projected.positions, self.local_rows.positions
            ) or not torch.equal(projected.valid_rows, self.local_rows.valid_rows):
                raise ValueError("V4.1 CP projected rows disagree with dispatch")
            fields = [("swa", torch.bfloat16)]
            if sources.owns_kv:
                fields.append(
                    (
                        "values",
                        (
                            torch.float32
                            if sources.compress_ratio == 2
                            else torch.bfloat16
                        ),
                    )
                )
                if sources.compress_ratio == 2:
                    fields.append(("scores", torch.float32))
            for name, dtype in fields:
                value = getattr(projected, name)
                _tensor(
                    value,
                    (self.local_rows.positions.numel(), 512),
                    dtype,
                    self.session.device,
                    "CP " + name,
                )
                if not bool(torch.isfinite(value[self.local_rows.valid_rows]).all()):
                    raise ValueError("V4.1 CP live projections must be finite")
            signature = _digest([self.signature.hex(), "projection", asdict(sources)])
        except Exception as exc:
            error = exc
        self.session._consensus(signature, error)
        values = {}
        try:
            for name, dtype in fields:
                capacity = self.session.plan.rows_per_rank
                value = torch.zeros(
                    capacity, 512, device=self.session.device, dtype=dtype
                )
                value[: self.local_rows.positions.numel()] = torch.where(
                    self.local_rows.valid_rows[:, None], getattr(projected, name), 0
                )
                received = torch.empty(
                    4 * capacity, 512, device=self.session.device, dtype=dtype
                )
                torch.distributed.all_gather_into_tensor(
                    received, value, group=self.session.process_group
                )
                values[name] = received
        except Exception:
            self.session._failed = True
            raise
        self._next_layer += 1
        return replace(
            projected,
            **values,
            positions=self.rows.positions,
            valid_rows=self.rows.valid_rows
        )

    def run_prepared(self, stage, prepared, plans, selections):
        self._check()
        try:
            if (
                not self._prepared
                or stage is not self._stage
                or plans is not self._plans
                or prepared is not self._arguments
                or selections is not self._selections
            ):
                raise RuntimeError("V4.1 CP execution differs from its prepared stage")
            state, rows, hashes = prepared
            result = stage._run_prepared(
                state, rows, plans, hashes, selections, cp_forward=self
            )
            if self._next_layer != stage.end:
                raise RuntimeError("V4.1 CP stage did not exchange every local layer")
        except Exception:
            self.session._failed = True
            raise
        self.session._round += 1
        self.session._active = None
        return result

    def run(self, stage, state, rows, plans, *, windows=None, selections=None):
        plans = dict(plans)
        prepared = self.prepare_stage(stage, state, rows, plans, windows, selections)
        return self.run_prepared(stage, prepared, plans, selections)

    def local_plan(self, plan):
        """Keep the executed L20 cache binding while exporting only local rows."""
        return replace(
            plan,
            **{
                name: getattr(plan, name)[self.local_slice].contiguous()
                for name in (
                    "swa_writes",
                    "swa_reads",
                    "global_pages",
                    "index_pages",
                    "global_writes",
                    "index_writes",
                )
                if getattr(plan, name) is not None
            }
        )
