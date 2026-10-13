"""Epoch-pinned Master, Mock HTTP and client-record observation sources."""
import json
import time
import urllib.request

from schema_contract import matches_schema
from adapters.debug_client import DebugClient, DebugUnavailable
from runtime.observation import FrozenSnapshot

SOURCES = frozenset(
    {"master_debug", "engine_snapshot", "engine_requests", "client_records"}
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
        self.master_port = self.env.master_http_port
        self.mock_port = self.env.mock_http_port
        self.records = None
        for kind in ("requests", "flow"):
            if kind in params:
                self.records = ctx.resource(params[kind], kind)
                if not callable(getattr(self.records, "snapshot_records", None)):
                    raise DebugUnavailable(
                        f"{kind} does not implement snapshot_records"
                    )
        self.master_instance = None
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
                if name == "master_debug":
                    result = DebugClient(
                        f"http://127.0.0.1:{self.master_port}",
                        timeout_s=min(5.0, deadline.remaining()),
                    ).snapshot(
                        include=",".join(self.params["master_include"]),
                        limit=self.params["limit"],
                        scan_limit=self.params["scan_limit"],
                        endpoint_limit=self.params["endpoint_limit"],
                    )
                    sample["data"] = result.payload
                    instance = result.payload["instanceId"]
                    sample["master_instance_id"] = instance
                    if self.master_instance is None:
                        self.master_instance = instance
                    elif self.master_instance != instance:
                        sample["status"] = "epoch_changed"
                        raise DebugUnavailable(
                            "Master instance changed; previous baseline is no longer comparable"
                        )
                    if result.payload["status"] != "ok" or any(
                        page["status"] != "ok" or page["truncated"]
                        for page in result.payload["components"].values()
                    ):
                        sample["status"] = "partial"
                        raise DebugUnavailable(
                            "Master snapshot has partial/busy/truncated components"
                        )
                elif name == "client_records":
                    records = self.records.snapshot_records()
                    if not isinstance(records, list) or any(
                        not isinstance(r, dict)
                        or not matches_schema(r, "client_record_schema_version", 1)
                        or r.get("env_epoch") != self.epoch
                        or type(r.get("wire_request_id")) is not int
                        or type(r.get("attempt")) is not int
                        for r in records
                    ):
                        raise DebugUnavailable("invalid or stale client_records v1")
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
