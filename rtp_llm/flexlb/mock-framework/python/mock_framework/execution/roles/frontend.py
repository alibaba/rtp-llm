"""Frontend JVMs and the one current playback; Java frontend0 owns traffic coordination."""

import asyncio
from dataclasses import dataclass
from typing import Annotated

from ...config.validation import Bounds, schema, validate
from .. import ExecutionError
from ..processes import ProcessIdentity, start_instance
from ..transport import invoke
from . import register_ready_instance


@dataclass(frozen=True)
class FrontendResult:
    done: bool
    stopped: bool
    produced: Annotated[int, Bounds(minimum=0)]
    issued: Annotated[int, Bounds(minimum=0)]
    finished: Annotated[int, Bounds(minimum=0)]
    unissued: Annotated[int, Bounds(minimum=0)]
    queued: Annotated[int, Bounds(minimum=0)]
    inflight: Annotated[int, Bounds(minimum=0)]
    outcomes: dict[str, Annotated[int, Bounds(minimum=0)]]
    errors: list[str]


class FrontendRole:
    def __init__(
        self,
        *,
        role_config,
        launch_config,
        supervisor,
        topology,
        records,
        monitoring,
    ):
        self.role_config, self.launch_config = role_config, launch_config
        self.supervisor, self.topology, self.records, self.monitoring = (
            supervisor,
            topology,
            records,
            monitoring,
        )
        self.pending, self.stop_timeout = False, 30

    async def start(self):
        if self.role_config is None:
            return
        for index in range(self.role_config["deployment"]["replicas"]):
            identity = ProcessIdentity("frontend", index)
            _, control, endpoint = await start_instance(
                self.supervisor,
                self.records,
                self.role_config,
                identity,
                self.launch_config,
            )
            if endpoint is None:
                raise ExecutionError(f"{identity.instance_id}: missing endpoints.json")
            register_ready_instance(
                self.topology,
                self.monitoring,
                self.records,
                identity,
                control,
                endpoint,
            )

    def _coordinator(self):
        nodes = [
            node
            for node in self.topology.list_active_endpoints("frontend")
            if node["id"] == "frontend-0"
        ]
        if len(nodes) != 1:
            raise ExecutionError("Expected exactly one frontend-0 coordinator")
        return nodes[0]["control"]

    async def _control(self, operation, args, timeout):
        return await invoke(self._coordinator(), operation, args, timeout)

    async def start_playback(self, body, timeout):
        frontends = self.topology.list_active_endpoints("frontend")
        if not frontends or body["total_concurrency"] < len(frontends):
            raise ExecutionError("Need at least one concurrency slot per Frontend")
        available = {
            node["rpc"] for node in self.topology.list_active_endpoints("master")
        }
        if not available or "masters" in body and set(body["masters"]) - available:
            raise ExecutionError("Unknown Master address")
        if self.pending:
            raise ExecutionError(
                "Collect previous playback with frontend/stop/playback before starting again"
            )
        # A lost acknowledgement may still leave Java sending; teardown must attempt collection.
        self.pending = True
        self.stop_timeout = body["drain_timeout_ms"] / 1000 + 15 + len(frontends) * 6
        result = await self._control("frontend/start", body, timeout)
        if result != {"accepted": True}:
            raise ExecutionError("Invalid Frontend start acknowledgement")
        return True

    async def stop_playback(self, timeout=None):
        if timeout is None:
            if not self.pending:
                return {}
            async with asyncio.timeout(self.stop_timeout):
                return await self._stop_playback(self.stop_timeout)
        return await self._stop_playback(timeout)

    async def _stop_playback(self, timeout):
        result = await self._control("frontend/stop", {}, timeout)
        self.records.emit(event="frontend_result", result=result)
        if not isinstance(result, dict) or set(result) != {"frontends", "errors"}:
            raise ExecutionError("Invalid Frontend result")
        validate(
            result["frontends"],
            schema(dict[str, FrontendResult]),
            "frontend/stop.result",
        )
        validate(result["errors"], schema(list[str]), "frontend/stop.errors")
        if not result["frontends"] and not result["errors"] and not self.pending:
            return {}
        frontends = self.topology.list_active_endpoints("frontend")
        if (
            result["errors"]
            or set(result["frontends"]) != {node["control"] for node in frontends}
            or any(
                not value["done"]
                or value["errors"]
                or value["inflight"]
                or value["queued"]
                or value["issued"] != value["finished"]
                or sum(value["outcomes"].values()) != value["finished"]
                or set(value["outcomes"])
                != {"success", "error", "timeout", "cancelled"}
                for value in result["frontends"].values()
            )
        ):
            raise ExecutionError(
                f"Frontend playback failed or results incomplete: {result}"
            )
        self.pending = False
        by_address = {node["control"]: node["rpc"] for node in frontends}
        return {by_address[url]: value for url, value in result["frontends"].items()}

    async def close(self):
        process_names = [
            process.process_name
            for process in self.supervisor.process_identities("frontend")
        ]
        return await self.supervisor.stop(process_names, 3)
