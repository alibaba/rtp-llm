import asyncio
import threading
from dataclasses import dataclass
from typing import Annotated

from .. import ExecutionError
from ..evidence.request_journal import RequestRecord, read_requests
from .step import StepContext
from .step_registry import Bounds, register_step


@dataclass(frozen=True)
class ClientRequestsArgs:
    targets: list[str] | None = None
    playback: Annotated[int, Bounds(minimum=1)] | None = None
    request_ids: list[Annotated[int, Bounds(minimum=1)]] | None = None
    outcome: str | None = None
    since_ms: Annotated[int, Bounds(minimum=0)] | None = None
    until_ms: Annotated[int, Bounds(minimum=0)] | None = None
    limit: Annotated[int, Bounds(minimum=0, maximum=1000)] = 20

    def __post_init__(self):
        if self.outcome is not None and self.outcome not in {
            "success",
            "error",
            "timeout",
            "cancelled",
        }:
            raise ValueError("Unknown Client outcome")
        if (
            self.since_ms is not None
            and self.until_ms is not None
            and self.since_ms > self.until_ms
        ):
            raise ValueError("since_ms must not exceed until_ms")
        if self.request_ids is not None and (
            not self.request_ids or len(set(self.request_ids)) != len(self.request_ids)
        ):
            raise ValueError("request_ids must be nonempty and unique")


@dataclass(frozen=True)
class RequestsResult:
    matched: int
    outcomes: dict[str, int]
    records: list[RequestRecord]
    truncated: bool


def playback_directories(root, topology, targets=None, playback=None):
    clients = {node["rpc"]: node for node in topology if node["role"] == "client"}
    chosen = list(clients) if targets is None else targets
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - clients.keys():
        raise ExecutionError("Expected nonempty, unique Client addresses")
    if playback is None:
        rounds = root / "components" / "client-0" / "coordinator"
        available = (
            [
                int(path.name)
                for path in rounds.iterdir()
                if path.is_dir() and path.name.isdecimal()
            ]
            if rounds.exists()
            else []
        )
        if not available:
            raise ExecutionError("No Client playback exists")
        playback = max(available)
    return [
        root / "components" / clients[target]["id"] / "playback" / str(playback)
        for target in chosen
    ]


@register_step("client/requests", effect="read")
async def client_requests(ctx: StepContext, args: ClientRequestsArgs) -> RequestsResult:
    """Read a sealed playback; filters limit returned details, never completeness validation."""
    directories = playback_directories(
        ctx.records.directory,
        ctx.runtime.topology.active(),
        args.targets,
        args.playback,
    )
    ids = None if args.request_ids is None else set(args.request_ids)

    cancelled = threading.Event()

    def read():
        details, outcomes = [], {
            key: 0 for key in ("success", "error", "timeout", "cancelled")
        }
        matched = 0
        for directory in directories:
            for record in read_requests(directory, cancelled):
                if (
                    ids is not None
                    and record.request_id not in ids
                    or args.outcome is not None
                    and record.outcome != args.outcome
                    or args.since_ms is not None
                    and record.sent_at_ms < args.since_ms
                    or args.until_ms is not None
                    and record.sent_at_ms >= args.until_ms
                ):
                    continue
                matched += 1
                outcomes[record.outcome] += 1
                if len(details) < args.limit:
                    details.append(record)
        return RequestsResult(matched, outcomes, details, matched > len(details))

    try:
        return await asyncio.to_thread(read)
    finally:
        cancelled.set()
