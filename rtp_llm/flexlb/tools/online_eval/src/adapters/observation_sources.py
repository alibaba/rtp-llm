"""Epoch-pinned Master, Mock HTTP and client-record observation sources."""
import json
import time
import urllib.request

from adapters.debug_client import DebugUnavailable
from runtime.observation import FrozenSnapshot

SOURCES = frozenset(
    {"engine_snapshot", "engine_requests", "client_records"}
)
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_DATASET_BYTES = 32 * 1024 * 1024


def _read_json(url, deadline):
    deadline.check()
    with urllib.request.urlopen(
        url, timeout=min(5.0, deadline.remaining())
    ) as response:
        body = response.read(MAX_SOURCE_BYTES + 1)
    if len(body) > MAX_SOURCE_BYTES:
        raise DebugUnavailable("source response exceeds byte budget")
    return json.loads(body)


class Sources:
    def __init__(self, ctx, params):
        self.ctx, self.params = ctx, params
        self.epoch = ctx.env_epoch
        self.env = ctx.env
        if self.env is None:
            raise DebugUnavailable("snapshot requires setup")
        self.mock_port = self.env.mock_http_port if {"engine_snapshot", "engine_requests"} & set(params["sources"]) else None
        self.custom = {name: ctx.services.snapshot(name, params) for name in params["sources"] if name not in SOURCES}
        self.records = None
        for kind in ("requests", "flow", "client"):
            if kind in params:
                self.records = ctx.resource(params[kind], kind)
                if not callable(getattr(self.records, "snapshot_records", None)):
                    raise DebugUnavailable(
                        f"{kind} does not implement snapshot_records"
                    )
        self.initial_members = {}

    def capture(self, deadline):
        started = self.ctx.clock()
        sources = {}
        for name in self.params["sources"]:
            deadline.check()
            sample = dict(
                source=name,
                env_epoch=self.epoch,
                started_mono=self.ctx.clock(),
                started_epoch_ms=int(time.time() * 1000),
                status="ok",
                errors=[],
                data=None,
            )
            try:
                if self.ctx.env_epoch != self.epoch or self.ctx.env is not self.env:
                    sample["status"] = "epoch_changed"
                    raise DebugUnavailable(
                        "environment changed; old cohort is not sampled from the new environment"
                    )
                if name in self.custom:
                    sample["data"] = self.custom[name].read(deadline)
                    json.dumps(sample["data"], allow_nan=False)
                elif name == "client_records":
                    from case_api.client_records import read_records
                    records = read_records(self.records, deadline, self.epoch)
                    sample["data"] = records  # Includes every nonterminal attempt.
                else:
                    route = "snapshot" if name == "engine_snapshot" else "requests"
                    data = _read_json(
                        f"http://127.0.0.1:{self.mock_port}/{route}", deadline
                    )
                    if name == "engine_snapshot":
                        if not isinstance(data, dict) or not isinstance(
                            data.get("engines"), list
                        ):
                            raise DebugUnavailable("missing engines snapshot")
                        members = {e["name"] for e in data["engines"]}
                    else:
                        if not isinstance(data, dict) or any(
                            not isinstance(v, dict) for v in data.values()
                        ):
                            raise DebugUnavailable("invalid engine requests snapshot")
                        members = set(data)
                    initial = self.initial_members.setdefault(name, frozenset(members))
                    sample["membership"] = dict(
                        initial=sorted(initial),
                        present=sorted(members),
                        added=sorted(members - initial),
                        removed=sorted(initial - members),
                    )
                    sample["engine_generation"] = (
                        None  # Mock HTTP does not provide restart identity.
                    )
                    sample["targets"] = self.params["targets"]
                    sample["data"] = data
                    missing = set(self.params["targets"]) - members
                    if missing:
                        sample["status"] = "partial"
                        raise DebugUnavailable(
                            f"requested engines missing: {sorted(missing)}"
                        )
            except Exception as error:
                if sample["status"] == "ok":
                    sample["status"] = "error"
                sample["errors"].append(
                    dict(type=type(error).__name__, message=str(error))
                )
            sample.update(
                finished_mono=self.ctx.clock(),
                finished_epoch_ms=int(time.time() * 1000),
            )
            sources[name] = sample
        payload = dict(
            observation_snapshot_schema_version=1,
            env_epoch=self.epoch,
            started_mono=started,
            finished_mono=self.ctx.clock(),
            sources=sources,
            status=(
                "ok"
                if all(s["status"] == "ok" for s in sources.values())
                else "partial"
            ),
        )
        encoded = json.dumps(payload, allow_nan=False)
        if len(encoded.encode()) > MAX_DATASET_BYTES:
            raise DebugUnavailable("snapshot exceeds dataset byte budget")
        return FrozenSnapshot(encoded)
