"""Gang worker for multi-machine AOne execution.

One worker process runs per machine inside a Bazel gang shard
(``perf_test_gang_shard``). The worker registers with the rendezvous
broker, keeps a heartbeat flowing while it waits, derives its rank layout
from the gang plan, starts the host's local rank group, and then either
executes the unified plan (the master shard) or waits for the master to
finish (every other shard). Every exit path casts exactly one terminal
vote to the broker and returns a process exit code; a broker-reported
gang failure is always a nonzero exit, never an endless retry.

The broker URL and run ID are declared in ``gang_plan.json``. Only the token
arrives through ``PERF_TEST_BROKER_TOKEN``. The token is
only ever sent in the authorization header; it never reaches worker logs
or persisted artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple

import requests

from rtp_llm.test.perf_test.aone.broker import (
    STATE_COMPLETED,
    STATE_READY,
    BrokerMember,
)
from rtp_llm.test.perf_test.aone.config_schema import parse_execution
from rtp_llm.test.perf_test.aone.package import GANG_PLAN_SCHEMA_VERSION
from rtp_llm.test.perf_test.config_resolver import canonical_json
from rtp_llm.test.perf_test.config_schema import parse_config
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.execution.devices import (
    GPU_LOCKED_DEVICES_ENV,
    resolve_allocated_devices,
)
from rtp_llm.test.perf_test.execution.distributed.rank_group import (
    build_gang_config_string,
)
from rtp_llm.test.perf_test.plan import ExecutionPlan
from rtp_llm.test.perf_test.run_context import read_run_context, write_run_context
from rtp_llm.test.perf_test.storage import atomic_write_json

if TYPE_CHECKING:  # pragma: no cover - typing only, heavy at runtime
    from rtp_llm.test.perf_test.execution.distributed.rank_group import ServerHandle

BROKER_TOKEN_ENV = "PERF_TEST_BROKER_TOKEN"

SHARD_INDEX_ENV = "TEST_SHARD_INDEX"
TOTAL_SHARDS_ENV = "TEST_TOTAL_SHARDS"

# Client-side request bounds. The broker serves ``members`` as a blocking
# call until the gang completes or the register deadline (3600 s default)
# passes, so the client uses a short timeout and re-polls while heartbeats
# keep flowing between calls.
REGISTER_TIMEOUT_S = 30.0
HEARTBEAT_TIMEOUT_S = 30.0
REPORT_TIMEOUT_S = 30.0
MEMBERS_POLL_TIMEOUT_S = 30.0
FORMATION_RETRY_DELAY_S = 1.0
WAITER_POLL_INTERVAL_S = 5.0
# The broker expires members after HEARTBEAT_EXPIRY_S (120 s); the worker
# interval keeps several beats of margin even under transient delays.
HEARTBEAT_INTERVAL_S = 30.0
HEARTBEAT_JOIN_TIMEOUT_S = 5.0

EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 3
EXIT_RENDEZVOUS_ERROR = 4
EXIT_EXECUTION_ERROR = 5

_LOG = logging.getLogger(__name__)
_LOG.addHandler(logging.NullHandler())

_sleep = time.sleep


class GangWorkerError(RuntimeError):
    """Worker failure carrying the process exit code."""

    exit_code = EXIT_RENDEZVOUS_ERROR


class WorkerConfigError(GangWorkerError):
    """Invalid environment, gang plan, or runfile inputs."""

    exit_code = EXIT_CONFIG_ERROR


class GangRendezvousError(GangWorkerError):
    """The rendezvous could not be completed."""


class GangBrokerFailure(GangRendezvousError):
    """The broker reported that the gang run failed."""


class BrokerRejectedError(GangRendezvousError):
    """The broker rejected a request with an unrecoverable status."""


class BrokerUnavailableError(GangRendezvousError):
    """The broker could not be reached (transient transport error)."""


class GangExecutionError(GangWorkerError):
    """The local rank group or the master plan failed."""

    exit_code = EXIT_EXECUTION_ERROR


@dataclass(frozen=True)
class BrokerCoordinates:
    """Broker location from the gang plan, with an inherited token."""

    url: str
    token: str
    run_id: str


def broker_coordinates(
    gang_plan: Mapping[str, Any], environ: Optional[Mapping[str, str]] = None
) -> BrokerCoordinates:
    """Read the connection from the plan and authentication from the environment."""
    run_id = gang_plan.get("broker_run_id")
    url = gang_plan.get("broker_url")
    if not isinstance(url, str) or not url.strip():
        raise WorkerConfigError("gang plan broker_url must be a non-empty string")
    if not isinstance(run_id, str) or not run_id.strip():
        raise WorkerConfigError("gang plan broker_run_id must be a non-empty string")
    environ = os.environ if environ is None else environ
    if not str(environ.get(BROKER_TOKEN_ENV, "")).strip():
        raise WorkerConfigError(
            f"gang worker requires the environment variable: {BROKER_TOKEN_ENV}"
        )
    return BrokerCoordinates(
        url=url.strip(),
        token=environ[BROKER_TOKEN_ENV].strip(),
        run_id=run_id,
    )


def _no_proxy_request(method: str, url: str, **kwargs: Any) -> Any:
    """Issue a broker request that ignores ambient proxy environment variables.

    Workers may inherit proxy settings that cannot reach the submit host.
    Keep broker connections direct within the execution network.
    """
    with requests.Session() as session:
        session.trust_env = False
        return session.request(method, url, **kwargs)


class BrokerClient:
    """Minimal authenticated HTTP client for the gang rendezvous broker.

    ``gang_state`` exposes the state observed by the latest ``members`` or
    ``register`` call; the broker serves both ready and completed gangs as
    HTTP 200, so the state is the only way to tell them apart.
    """

    _FAILURE_CODES = frozenset(
        {"gang_failed", "run_closed", "broker_stopped", "register_deadline_exceeded"}
    )

    def __init__(self, url: str, token: str, run_id: str, *, request=None):
        if request is None:
            request = _no_proxy_request
        self._base = url.rstrip("/")
        self._token = token
        self._run_id = run_id
        self._request = request
        self._gang_state: Optional[str] = None

    @property
    def gang_state(self) -> Optional[str]:
        return self._gang_state

    def register(
        self, shard_index: int, hostname: str, port: int
    ) -> List[BrokerMember]:
        """Register this shard; returns the members when this call completes the gang."""
        body = self._call(
            "POST",
            "register",
            {"shard_index": shard_index, "hostname": hostname, "port": port},
            timeout_s=REGISTER_TIMEOUT_S,
        )
        self._gang_state = body.get("gang_state")
        if self._gang_state == STATE_READY:
            return self.members(REGISTER_TIMEOUT_S)
        return []

    def members(self, timeout_s: float) -> List[BrokerMember]:
        """Poll the member list; a client timeout is transient and re-polled."""
        try:
            body = self._call("GET", "members", None, timeout_s=timeout_s)
        except BrokerUnavailableError as error:
            _LOG.debug("members poll was transient: %s", error)
            return []
        self._gang_state = body.get("gang_state")
        return [BrokerMember(**item) for item in body.get("members", [])]

    def heartbeat(self, shard_index: int) -> None:
        self._call(
            "POST",
            "heartbeat",
            {"shard_index": shard_index},
            timeout_s=HEARTBEAT_TIMEOUT_S,
        )

    def report(
        self, shard_index: int, state: str, detail: Optional[str] = None
    ) -> None:
        self._call(
            "POST",
            "result",
            {"shard_index": shard_index, "state": state, "detail": detail},
            timeout_s=REPORT_TIMEOUT_S,
        )

    def cleanup(self) -> None:
        """Close the rendezvous run. Only the submit side calls this."""
        self._call("POST", "cleanup", None, timeout_s=REPORT_TIMEOUT_S)

    def _call(
        self,
        method: str,
        op: str,
        payload: Optional[Dict[str, Any]],
        *,
        timeout_s: float,
    ) -> Dict[str, Any]:
        url = f"{self._base}/v1/runs/{self._run_id}/{op}"
        try:
            response = self._request(
                method,
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=timeout_s,
            )
        except requests.RequestException as error:
            raise BrokerUnavailableError(
                f"broker request {op} failed: {error}"
            ) from error
        status = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = {}
        if status == 200:
            return body if isinstance(body, dict) else {}
        code = (
            body.get("error", f"http_{status}")
            if isinstance(body, dict)
            else f"http_{status}"
        )
        if status == 504 or code in self._FAILURE_CODES:
            raise GangBrokerFailure(f"broker reported gang failure during {op}: {code}")
        raise BrokerRejectedError(f"broker rejected {op}: HTTP {status} {code}")


class _HeartbeatLoop:
    """Background heartbeat thread; the broker expires silent members.

    The loop beats from registration onward (it starts before the first
    ``members`` poll), survives transient broker errors, and records a
    broker-reported gang failure in ``failed`` so waiting workers can stop
    early. ``stop`` joins the thread, so no heartbeat outlives the worker.
    """

    def __init__(self, broker: BrokerClient, shard_index: int):
        self._broker = broker
        self._shard_index = shard_index
        self._stop = threading.Event()
        self.failed = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop,
            name=f"gang-worker-heartbeat-{self._shard_index}",
            daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._broker.heartbeat(self._shard_index)
            except GangBrokerFailure as error:
                _LOG.warning("heartbeat rejected; gang failed: %s", error)
                self.failed.set()
                return
            except GangWorkerError as error:
                # Transient transport errors and unexpected rejections are
                # retried on the next beat; the main flow surfaces real
                # failures through its own broker calls.
                _LOG.warning("heartbeat attempt failed, will retry: %s", error)
            if self._stop.wait(HEARTBEAT_INTERVAL_S):
                return

    def check_failed(self) -> None:
        if self.failed.is_set():
            raise GangBrokerFailure("heartbeat rejected by the broker; gang failed")

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=HEARTBEAT_JOIN_TIMEOUT_S)


@dataclass(frozen=True)
class GangWorkerContext:
    """Everything a host needs from the rendezvous, frozen at formation."""

    gang_plan: Dict[str, Any]
    members: Tuple[BrokerMember, ...]
    shard_index: int
    world_rank: int
    local_world_size: int
    gang_config_string: str
    cuda_visible_devices: str


@dataclass(frozen=True)
class _EngineMember:
    """Engine-facing ``name/ip/port`` member identity."""

    name: str
    ip: str
    server_port: int


def load_gang_plan(path: Path) -> Dict[str, Any]:
    """Read and validate one ``gang_plan.json`` runfile."""
    try:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerConfigError(f"cannot read gang plan {path}: {error}") from error
    if not isinstance(plan, dict):
        raise WorkerConfigError(f"gang plan {path} must contain an object")
    if plan.get("schema_version") != GANG_PLAN_SCHEMA_VERSION:
        raise WorkerConfigError(
            f"unsupported gang plan schema_version: {plan.get('schema_version')!r}"
        )
    for field in (
        "run_id",
        "machine_count",
        "local_world_size",
        "master_shard_index",
        "port_range",
        "topology",
    ):
        if field not in plan:
            raise WorkerConfigError(f"gang plan is missing required field {field!r}")
    if not isinstance(plan["run_id"], str) or not plan["run_id"]:
        raise WorkerConfigError("gang plan run_id must be a non-empty string")
    machine_count = plan["machine_count"]
    local_world_size = plan["local_world_size"]
    if type(machine_count) is not int or machine_count < 1:
        raise WorkerConfigError("gang plan machine_count must be a positive integer")
    if type(local_world_size) is not int or local_world_size < 1:
        raise WorkerConfigError("gang plan local_world_size must be a positive integer")
    master_shard_index = plan["master_shard_index"]
    if (
        type(master_shard_index) is not int
        or not 0 <= master_shard_index < machine_count
    ):
        raise WorkerConfigError(
            f"gang plan master_shard_index {master_shard_index!r} is outside the gang"
        )
    if master_shard_index != 0:
        raise WorkerConfigError(
            "gang plan master_shard_index must be 0 (the gang generator pins "
            f"shard 0 as the master); got {master_shard_index!r}"
        )
    port_range = plan["port_range"]
    if (
        not isinstance(port_range, dict)
        or type(port_range.get("start")) is not int
        or not 0 < port_range["start"] < 65536
        or port_range.get("count") != local_world_size
    ):
        raise WorkerConfigError(
            "gang plan port_range must be {start: valid port, count: local_world_size}"
        )
    topology = plan["topology"]
    if not isinstance(topology, dict) or any(
        type(topology.get(name)) is not int or topology.get(name) < 1
        for name in ("tp", "dp", "pp", "ep")
    ):
        raise WorkerConfigError("gang plan topology must supply positive tp/dp/pp/ep")
    return plan


def shard_index_from_env(
    gang_plan: Dict[str, Any], environ: Optional[Mapping[str, str]] = None
) -> int:
    """Derive this worker's shard index from the Bazel sharding env."""
    environ = os.environ if environ is None else environ
    raw_index = environ.get(SHARD_INDEX_ENV)
    raw_total = environ.get(TOTAL_SHARDS_ENV)
    if raw_index is None or raw_total is None:
        raise WorkerConfigError(
            f"gang worker requires the Bazel sharding env {SHARD_INDEX_ENV}/{TOTAL_SHARDS_ENV}"
        )
    try:
        shard_index = int(raw_index)
        total_shards = int(raw_total)
    except ValueError as error:
        raise WorkerConfigError(
            f"invalid Bazel sharding env {raw_index!r}/{raw_total!r}"
        ) from error
    machine_count = gang_plan["machine_count"]
    if total_shards != machine_count or not 0 <= shard_index < machine_count:
        raise WorkerConfigError(
            f"test shard {shard_index}/{total_shards} does not match the gang's "
            f"machine_count {machine_count}"
        )
    return shard_index


def build_worker_context(
    gang_plan: Dict[str, Any], members: Tuple[BrokerMember, ...], shard_index: int
) -> GangWorkerContext:
    """Freeze the rank layout: world rank, gang config, and GPU visibility."""
    local_world_size = int(gang_plan["local_world_size"])
    engine_members = tuple(
        _EngineMember(
            name=f"gang_part{member.shard_index}",
            ip=member.hostname,
            server_port=int(member.port),
        )
        for member in sorted(members, key=lambda member: member.shard_index)
    )
    gang_config_string = build_gang_config_string(engine_members, local_world_size)
    return GangWorkerContext(
        gang_plan=gang_plan,
        members=tuple(members),
        shard_index=shard_index,
        world_rank=shard_index * local_world_size,
        local_world_size=local_world_size,
        gang_config_string=gang_config_string,
        cuda_visible_devices=",".join(
            str(device) for device in resolve_allocated_devices(range(local_world_size))
        ),
    )


def _apply_environment(context: GangWorkerContext) -> None:
    """Export the rendezvous-derived environment for the local rank group."""
    os.environ["CUDA_VISIBLE_DEVICES"] = context.cuda_visible_devices
    if GPU_LOCKED_DEVICES_ENV in os.environ and "HIP_VISIBLE_DEVICES" in os.environ:
        os.environ["HIP_VISIBLE_DEVICES"] = context.cuda_visible_devices
    os.environ["GANG_CONFIG_STRING"] = context.gang_config_string


def _canonical_local_hostname() -> str:
    """Choose a stable claimed hostname, falling back to the plain name.

    The broker records this claim and its own best-effort reverse lookup for
    diagnostics; registration tolerates a mismatch between the two.
    """
    name = socket.gethostname()
    try:
        address = socket.gethostbyname(name)
        return socket.gethostbyaddr(address)[0]
    except OSError:
        return name


def _await_formation(
    broker: BrokerClient,
    gang_plan: Dict[str, Any],
    *,
    clock: Callable[[], float],
    check_cancelled: Callable[[], None] = lambda: None,
) -> Tuple[BrokerMember, ...]:
    """Poll until the broker reports the complete member list."""
    machine_count = int(gang_plan["machine_count"])
    started = clock()
    while True:
        check_cancelled()
        members = broker.members(MEMBERS_POLL_TIMEOUT_S)
        if (
            broker.gang_state in (STATE_READY, STATE_COMPLETED)
            and len(members) == machine_count
        ):
            _LOG.info(
                "gang formed after %.1f s with %d members",
                clock() - started,
                machine_count,
            )
            return tuple(members)
        _sleep(FORMATION_RETRY_DELAY_S)


def _run_master(
    broker: BrokerClient,
    shard_index: int,
    context: GangWorkerContext,
    handle: Any,
    *,
    resolved_config_path: Path,
    outputs: Path,
    master_executor: Optional[Callable[..., int]],
    check_cancelled: Callable[[], None] = lambda: None,
) -> int:
    """Execute the unified plan on the master shard and cast its vote."""
    executor = _default_master_executor if master_executor is None else master_executor
    kwargs = dict(
        context=context,
        config_path=str(resolved_config_path),
        result_directory=str(outputs / "run"),
    )
    if executor in (_default_master_executor, _default_prefill_master_executor):
        kwargs["check_cancelled"] = check_cancelled
    try:
        check_cancelled()
        exit_code = int(executor(handle, **kwargs))
        check_cancelled()
    except BaseException as error:
        _report_failure_vote(
            broker,
            shard_index,
            f"master executor failed: {type(error).__name__}: {error}",
        )
        if not isinstance(error, Exception):
            raise
        raise GangExecutionError(
            f"master executor failed: {type(error).__name__}: {error}"
        ) from error
    if exit_code != 0:
        _report_failure_vote(
            broker, shard_index, f"master executor exited with code {exit_code}"
        )
        return exit_code
    # A success vote that the broker rejects (the gang already failed while
    # the master was executing) is a nonzero exit, not a silent success.
    broker.report(shard_index, "completed", "master plan completed")
    return EXIT_SUCCESS


def _run_waiter(
    broker: BrokerClient,
    shard_index: int,
    heartbeats: _HeartbeatLoop,
    detail: str = "rank group up; awaiting master completion",
) -> int:
    """Wait for the master's outcome and mirror it in this shard's exit code.

    A non-master shard's own duty ends once its rank group is up, so it
    casts its success vote now — the gang can only complete after the
    master reports, because the master always reports last. The exit code
    then follows the gang outcome: completed is success, any broker-
    reported failure is an immediate nonzero exit.
    """
    broker.report(shard_index, "completed", detail)
    while True:
        heartbeats.check_failed()
        broker.members(MEMBERS_POLL_TIMEOUT_S)
        if broker.gang_state == STATE_COMPLETED:
            return EXIT_SUCCESS
        _sleep(WAITER_POLL_INTERVAL_S)


def _report_failure_vote(broker: BrokerClient, shard_index: int, detail: str) -> None:
    """Cast the failure vote; a broker that already failed the gang wins."""
    try:
        broker.report(shard_index, "failed", detail)
    except GangWorkerError as report_error:
        _LOG.warning(
            "failure vote was rejected (gang already terminal): %s", report_error
        )


def _run_worker(
    resolved_config_path: Path,
    gang_plan: Dict[str, Any],
    shard_index: int,
    *,
    broker: BrokerClient,
    server_runner: Callable[[GangWorkerContext], Any],
    outputs: Path,
    master_executor: Optional[Callable[..., int]],
    clock: Callable[[], float],
    record: Dict[str, Any],
) -> int:
    """Full worker lifecycle: register, form, start, execute, report."""
    hostname = _canonical_local_hostname()
    port = int(gang_plan["port_range"]["start"])
    broker.register(shard_index, hostname, port)
    heartbeats = _HeartbeatLoop(broker, shard_index)
    heartbeats.start()  # beats from registration onward, before any members poll
    handle: Any = None
    try:
        try:
            members = _await_formation(
                broker, gang_plan, clock=clock, check_cancelled=heartbeats.check_failed
            )
            context = build_worker_context(gang_plan, members, shard_index)
            record["context"] = context
            if _is_distributed_decode(resolved_config_path, context):
                from rtp_llm.test.perf_test.execution.distributed.decode_sync import (
                    validate_decode_layout,
                )

                validate_decode_layout(
                    context.local_world_size,
                    int(context.gang_plan["topology"]["tp"]),
                    int(context.gang_plan["topology"]["dp"]),
                    tuple(
                        index * context.local_world_size
                        for index in range(len(context.members))
                    ),
                )
            _apply_environment(context)
            handle = server_runner(context)
            heartbeats.check_failed()
            if not handle.start_server():
                handle.print_process_log()
                raise GangExecutionError("local rank-group server failed to start")
        except BaseException as error:
            # Failures before dispatch have not voted yet; every dispatched
            # path owns exactly one vote of its own. Non-GangWorkerError
            # startup failures (import, planning, key errors) vote too: the
            # broker rejects a second result with 409 and the vote helper
            # swallows that, so this cannot double-vote.
            _report_failure_vote(
                broker, shard_index, f"{type(error).__name__}: {error}"
            )
            raise
        if shard_index == int(gang_plan["master_shard_index"]):
            return _run_master(
                broker,
                shard_index,
                context,
                handle,
                resolved_config_path=resolved_config_path,
                outputs=outputs,
                master_executor=master_executor,
                check_cancelled=heartbeats.check_failed,
            )
        if _is_distributed_decode(resolved_config_path, context):
            try:
                _default_decode_worker(
                    handle,
                    context=context,
                    config_path=str(resolved_config_path),
                    check_cancelled=heartbeats.check_failed,
                )
            except BaseException as error:
                _report_failure_vote(
                    broker,
                    shard_index,
                    f"decode worker failed: {type(error).__name__}: {error}",
                )
                if not isinstance(error, Exception):
                    raise
                raise GangExecutionError(
                    f"decode worker failed: {type(error).__name__}: {error}"
                ) from error
            return _run_waiter(
                broker, shard_index, heartbeats, "decode measurements completed"
            )
        return _run_waiter(broker, shard_index, heartbeats)
    finally:
        try:
            if handle is not None:
                handle.stop_server()
        finally:
            heartbeats.stop()


def run_gang_worker(
    resolved_config_path: Path,
    gang_plan_path: Path,
    *,
    broker: BrokerClient,
    server_runner: Callable[[GangWorkerContext], Any],
    test_outputs_dir: Path,
    master_executor: Optional[Callable[..., int]] = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run one gang worker to its terminal state and return the exit code.

    ``server_runner`` starts (but does not launch) the host's local rank
    group and returns the owning server handle; ``master_executor`` runs
    the unified plan on the master shard and defaults to the external
    decode or prefill executor chosen by the merged config's workload
    kind. Partial ``run/`` artifacts always remain on disk.
    """
    # The broker client already owns its credential. Engine subprocesses must
    # never inherit it, including during imports, startup, and shutdown.
    token = os.environ.pop(BROKER_TOKEN_ENV, None)
    outputs = Path(test_outputs_dir)
    outcome: Dict[str, Any] = {
        "state": "completed",
        "exit_code": EXIT_SUCCESS,
        "detail": None,
    }
    gang_plan: Optional[Dict[str, Any]] = None
    record: Dict[str, Any] = {}
    try:
        outputs.mkdir(parents=True, exist_ok=True)
        shard_status = os.environ.get("TEST_SHARD_STATUS_FILE")
        if shard_status:
            Path(shard_status).touch()
        gang_plan = load_gang_plan(gang_plan_path)
        resolved = Path(resolved_config_path)
        if not resolved.is_file():
            raise WorkerConfigError(f"resolved config {resolved} does not exist")
        shard_index = shard_index_from_env(gang_plan)
        exit_code = _run_worker(
            resolved,
            gang_plan,
            shard_index,
            broker=broker,
            server_runner=server_runner,
            outputs=outputs,
            master_executor=master_executor,
            clock=clock,
            record=record,
        )
        outcome.update(
            state="completed" if exit_code == EXIT_SUCCESS else "failed",
            exit_code=exit_code,
        )
        return exit_code
    except GangWorkerError as error:
        outcome.update(state="failed", exit_code=error.exit_code, detail=str(error))
        return error.exit_code
    except Exception as error:  # defensive: unexpected failures still exit nonzero
        outcome.update(
            state="failed",
            exit_code=EXIT_EXECUTION_ERROR,
            detail=f"{type(error).__name__}: {error}",
        )
        return EXIT_EXECUTION_ERROR
    except BaseException as error:
        outcome.update(
            state="failed",
            exit_code=EXIT_EXECUTION_ERROR,
            detail=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        try:
            _write_worker_summary(outputs, gang_plan, record.get("context"), outcome)
        finally:
            if token is not None:
                os.environ[BROKER_TOKEN_ENV] = token


def _write_worker_summary(
    outputs: Path,
    gang_plan: Optional[Dict[str, Any]],
    context: Optional[GangWorkerContext],
    outcome: Mapping[str, Any],
) -> None:
    """Persist a token-free worker summary under the undeclared outputs."""
    summary = {
        "schema_version": 1,
        "run_id": gang_plan.get("run_id") if gang_plan else None,
        "machine_count": gang_plan.get("machine_count") if gang_plan else None,
        "shard_index": context.shard_index if context else None,
        "world_rank": context.world_rank if context else None,
        "local_world_size": context.local_world_size if context else None,
        "gang_config_string": context.gang_config_string if context else None,
        "members": [member.to_dict() for member in context.members] if context else [],
        "state": outcome["state"],
        "exit_code": outcome["exit_code"],
        "detail": outcome["detail"],
    }
    try:
        atomic_write_json(outputs / "gang_worker.json", summary)
    except OSError as error:
        _LOG.warning("cannot write worker summary: %s", error)


def _load_rank_group():
    from rtp_llm.test.perf_test.execution.distributed import rank_group

    return rank_group


def _build_rank_group_plan(
    config_path: str, context: GangWorkerContext, outputs: Path
) -> Tuple[Any, Mapping[str, Any]]:
    """Build the plan before starting any local engine ranks."""
    return _build_gang_execution_plan(config_path, context, str(outputs / "run"))


def _build_gang_execution_plan(config_path, context, result_directory):
    """Consume the controller's v2 plan; never replan cases on a worker."""
    from rtp_llm.test.perf_test.plan import bind_member_plan

    try:
        plan = ExecutionPlan.from_dict(context.gang_plan["execution_plan"])
        source = json.loads(Path(config_path).read_text(encoding="utf-8"))
        workload = json.loads(
            Path(config_path).with_suffix(".frozen.json").read_text(encoding="utf-8")
        )
    except (KeyError, OSError, ValueError) as error:
        raise PlanningError(
            f"gang requires a complete frozen execution plan: {error}"
        ) from error
    if plan.backend != "aone":
        raise PlanningError("gang execution plan requires backend=aone")
    topology = plan.resolved_config["model"]["topology"]
    if dict(topology) != context.gang_plan["topology"]:
        raise PlanningError("runtime topology does not match frozen config")
    parsed = asdict(parse_config(source, execution_parser=parse_execution))
    parsed.pop("workload")

    def fingerprint(value):
        return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

    if fingerprint(parsed) != plan.config_fingerprint:
        raise PlanningError("gang frozen config_fingerprint mismatch")
    if fingerprint(workload) != plan.workload_fingerprint:
        raise PlanningError("gang frozen workload_fingerprint mismatch")
    for name in ("config_fingerprint", "workload_fingerprint"):
        if getattr(plan, name) != context.gang_plan[name]:
            raise PlanningError(f"gang frozen {name} mismatch")
    return bind_member_plan(plan, result_directory=str(result_directory)), source


def _default_server_runner(context: GangWorkerContext, *, config_path: str):
    """Start this host's rank group through the extracted legacy startup."""
    from rtp_llm.config.server_config_setup import set_parallelism_config
    from rtp_llm.server.server_args.server_args import setup_args

    runner = _load_rank_group()
    plan = context.gang_plan
    topology = plan["topology"]
    engine_environment = {
        "GANG_CONFIG_STRING": context.gang_config_string,
        "CUDA_VISIBLE_DEVICES": context.cuda_visible_devices,
        "WORLD_SIZE": str(plan["world_size"]),
        "WORLD_RANK": str(context.world_rank),
        "LOCAL_WORLD_SIZE": str(context.local_world_size),
        "TP_SIZE": str(topology["tp"]),
        "DP_SIZE": str(topology["dp"]),
        "EP_SIZE": str(topology["ep"]),
        "PP_SIZE": str(topology["pp"]),
        # Match the legacy decode runner's engine switches.
        "USE_BATCH_DECODE_SCHEDULER": "1",
        "FAKE_BALANCE_EXPERT": "1",
    }
    if int(plan["machine_count"]) > 1:
        # Symmetric-memory custom all-reduce passes file descriptors between
        # TP ranks, which cannot cross machines; a multi-machine gang spans
        # its TP group across hosts, so the custom AR must be disabled.
        engine_environment["FT_DISABLE_CUSTOM_AR"] = "1"
    for key, value in engine_environment.items():
        os.environ[key] = value

    outputs = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()))
    log_dir = outputs / f"TEST_OUTPUT_{plan['run_id']}_{context.world_rank}"
    # The plan builder follows the gang mode: decode gangs keep the decode
    # plan, cache_grid (prefill) gangs build the prefill plan.
    execution_plan, _source = _build_rank_group_plan(config_path, context, outputs)
    model = _resolve_model_paths(config_path, plan)
    os.environ.setdefault("MODEL_TYPE", str(model["type"]))
    os.environ.setdefault("CHECKPOINT_PATH", str(model["checkpoint"]))
    os.environ.setdefault("TOKENIZER_PATH", str(model["tokenizer"]))
    # Engine argv comes from the plan, never from this process's sys.argv
    # (the worker's own --resolved-config flags would be rejected by the
    # engine parser), and it must carry the RESOLVED model paths: hub repo
    # ids in the argv would make the engine read config.json from a
    # nonexistent "repo-id/config.json" path instead of the snapshot.
    from rtp_llm.test.perf_test.execution.launch_boundary import (
        configure_perf_environment,
    )
    from rtp_llm.test.perf_test.execution.runtime import build_engine_launch_spec
    from rtp_llm.test.perf_test.plan import thaw

    resolved_config = thaw(execution_plan.resolved_config)
    resolved_model = dict(resolved_config["model"])
    resolved_model["checkpoint"] = model["checkpoint"]
    resolved_model["tokenizer"] = model["tokenizer"]
    resolved_config["model"] = resolved_model
    launch = build_engine_launch_spec(
        _replace_resolved_config(execution_plan, resolved_config)
    )
    for name in engine_environment.keys() & launch.environment.keys():
        if engine_environment[name] != launch.environment[name]:
            raise WorkerConfigError(
                f"engine.env.{name} conflicts with the required gang runtime value "
                f"{engine_environment[name]!r}"
            )
    launch_environment = {**launch.environment, **engine_environment}
    configure_perf_environment(launch_environment)
    # Engine argv/environment reach the server subprocess through the rank
    # group startup; without them per-run switches (reuse_cache, topology
    # sizes) never reach the engine and prefix-cache reuse silently
    # disappears.
    configs = setup_args(list(launch.argv))
    set_parallelism_config(
        configs.parallelism_config, py_prefill_cp_config=configs.prefill_cp_config
    )
    rank_context = runner.LocalRankGroupContext(
        port=int(plan["port_range"]["start"]),
        world_rank=context.world_rank,
        log_dir_path=str(log_dir),
        profiler_directory=log_dir,
        plan=execution_plan,
    )
    return runner.start_local_rank_group(
        rank_context,
        configs,
        model["tokenizer"],
        engine_argv=launch.argv,
        engine_environment=launch_environment,
    )


def _replace_resolved_config(plan: Any, resolved_config: Dict[str, Any]) -> Any:
    """Return a copy of ``plan`` with the resolved config replaced.

    Real plans are frozen dataclasses; test doubles are plain namespaces.
    """
    import dataclasses
    from types import SimpleNamespace

    if dataclasses.is_dataclass(plan) and not isinstance(plan, type):
        return dataclasses.replace(plan, resolved_config=resolved_config)
    fields = dict(vars(plan))
    fields["resolved_config"] = resolved_config
    return SimpleNamespace(**fields)


def _resolve_model_paths(
    config_path: str, plan: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Resolve the model section (type, checkpoint, tokenizer) locally.

    Hub repo ids are downloaded through the shared remote resolver; the
    engine subprocess only sees environment variables (MODEL_TYPE,
    CHECKPOINT_PATH, TOKENIZER_PATH), so the worker must export the
    resolved values before the rank-group server starts.
    """
    from rtp_llm.test.perf_test.aone.worker import (
        RemoteInputs,
        resolve_remote_runtime_inputs,
    )

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    resolved = resolve_remote_runtime_inputs(
        RemoteInputs(
            run_id=str(plan["run_id"]),
            shard_id=f"gang_part{plan['master_shard_index']}",
            config_fingerprint=str(plan.get("config_fingerprint", "")),
            workload_fingerprint=str(plan.get("workload_fingerprint", "")),
            config=config,
            owned_case_ids=(),
            owned_cases=(),
        )
    )
    return resolved.config["model"]


def _is_distributed_decode(config_path: Path, context: GangWorkerContext) -> bool:
    from rtp_llm.test.perf_test.decode.scheduler import requires_distributed_decode

    if int(context.gang_plan["machine_count"]) < 2:
        return False
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    workload = config.get("workload", {})
    return workload.get("kind") in (
        "grid",
        "distribution",
    ) and requires_distributed_decode(
        context.gang_plan["topology"], context.local_world_size
    )


def _decode_control_port(context: GangWorkerContext) -> int:
    start = int(context.gang_plan["port_range"]["start"])
    port = start + 1024
    if not 0 < port < 65536:
        raise WorkerConfigError(
            "derived decode control port must be between 1 and 65535"
        )
    return port


def _decode_member(handle, *, context, config_path, result_directory, check_cancelled):
    from rtp_llm.test.perf_test.execution.distributed.decode import (
        execute_decode_member,
    )

    plan, source = _build_gang_execution_plan(config_path, context, result_directory)
    tokenizer_path = (
        handle.tokenizer_path
        or _resolve_model_paths(config_path, context.gang_plan)["tokenizer"]
    )
    if not tokenizer_path:
        raise WorkerConfigError("decode member tokenizer path is missing")
    return execute_decode_member(
        plan,
        source_config=source,
        endpoint=f"http://127.0.0.1:{int(context.gang_plan['port_range']['start'])}",
        tokenizer_path=tokenizer_path,
        server_log_path=handle.log_file_path,
        profiler_directory=Path(handle.log_dir),
        world_rank=context.world_rank,
        local_world_size=context.local_world_size,
        member_hosts=tuple(member.hostname for member in context.members),
        control_port=_decode_control_port(context),
        run_context=context.gang_plan,
        check_cancelled=lambda: _check_handle_health(handle, check_cancelled),
    )


def _check_handle_health(handle, check_cancelled):
    from rtp_llm.test.perf_test.execution.distributed.member import check_member_health

    check_member_health(handle, check_cancelled)


def _default_decode_worker(
    handle: Any,
    *,
    context: GangWorkerContext,
    config_path: str,
    check_cancelled: Callable[[], None] = lambda: None,
) -> None:
    _decode_member(
        handle,
        context=context,
        config_path=config_path,
        result_directory=str(Path(handle.log_dir) / "unused-run"),
        check_cancelled=check_cancelled,
    )


def _default_master_executor(
    handle: Any,
    *,
    context: GangWorkerContext,
    config_path: str,
    result_directory: str,
    check_cancelled: Callable[[], None] = lambda: None,
) -> int:
    result = _decode_member(
        handle,
        context=context,
        config_path=config_path,
        result_directory=result_directory,
        check_cancelled=check_cancelled,
    )
    state = getattr(result, "state", None)
    return EXIT_SUCCESS if getattr(state, "value", None) == "completed" else 1


def _default_prefill_master_executor(
    handle: Any,
    *,
    context: GangWorkerContext,
    config_path: str,
    result_directory: str,
    check_cancelled: Callable[[], None] = lambda: None,
) -> int:
    """Prepare gang inputs; the shared executor owns measurement and artifacts."""
    from rtp_llm.test.perf_test.execution.distributed.prefill import (
        execute_master_prefill,
    )

    plan, source = _build_gang_execution_plan(config_path, context, result_directory)
    model = _resolve_model_paths(config_path, context.gang_plan)
    result = execute_master_prefill(
        plan,
        source_config=source,
        endpoint=f"http://127.0.0.1:{int(context.gang_plan['port_range']['start'])}",
        tokenizer_path=model["tokenizer"],
        model_paths=model,
        server_log_path=handle.log_file_path,
        profiler_directory=Path(handle.log_dir),
        run_context=context.gang_plan,
        health_check=lambda: _check_handle_health(handle, check_cancelled),
    )
    return EXIT_SUCCESS if result.state.value == "completed" else 1


def merged_config_mode(config_path: Path) -> str:
    """Derive the master execution mode from the merged resolved config."""
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        kind = config["workload"]["kind"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise WorkerConfigError(
            f"cannot derive the gang mode from the merged config: {error}"
        ) from error
    if kind == "cache_grid":
        return "prefill"
    if kind in ("grid", "distribution"):
        return "decode"
    raise WorkerConfigError(f"unsupported gang workload kind: {kind!r}")


def _prefill_workload_source(workload: Dict[str, Any]) -> Dict[str, Any]:
    """Restore source shape from a frozen gang workload, before strict parsing.

    Dataclass serialization includes mutually exclusive workload
    shapes. Named grids also wrap their child workload, unlike source JSON.
    Keep this conversion at the runfile boundary, never in source validation.
    """
    generated = {
        "max_input_len",
        "compute_steps",
        "batch_sizes",
        "min_compute_len",
        "min_cache_len",
        "max_cache_len",
        "cache_lengths",
        "random_cache_count",
        "seed",
    }
    source = {key: value for key, value in workload.items() if value is not None}
    source.pop("traffic_sampling", None)
    grids = source.pop("grids", [])
    batches = source.pop("request_batches", [])
    if grids:
        source = {key: value for key, value in source.items() if key not in generated}
        source["grids"] = [
            dict(
                name=grid["name"],
                **{
                    key: value
                    for key, value in _prefill_workload_source(grid["workload"]).items()
                    if key
                    in generated
                    | {
                        "request_batches",
                        "random_request_batches",
                        "logarithmic_request_batches",
                    }
                },
            )
            for grid in grids
        ]
    elif (
        batches
        or source.get("random_request_batches")
        or source.get("logarithmic_request_batches")
    ):
        source = {
            key: value
            for key, value in source.items()
            if key not in generated - {"seed"}
        }
        if batches:
            source["request_batches"] = batches
    return source


def _merge_workload(
    resolved_config: str, resolved_workload: str, outputs: Path
) -> Path:
    """Combine the package's config and workload runfiles for plan building.

    Keep a frozen workload companion next to the parser-facing config.
    Both rank startup and master execution consume this pair; sampling
    metadata and inline distributions never pass through source resolution.
    """
    try:
        config = json.loads(Path(resolved_config).read_text(encoding="utf-8"))
        workload = json.loads(Path(resolved_workload).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerConfigError(f"cannot read gang runfiles: {error}") from error
    if not isinstance(config, dict) or not isinstance(workload, dict):
        raise WorkerConfigError("gang resolved config and workload must be objects")
    config["workload"] = (
        _prefill_workload_source(workload)
        if workload.get("kind") == "cache_grid"
        else workload
    )
    merged = outputs / "resolved_config_with_workload.json"
    if workload.get("kind") == "distribution":
        # The planner consumes the frozen inline distribution, never this
        # source-schema placeholder or the original submit-host filename.
        config["workload"] = {
            key: value for key, value in workload.items() if key != "distribution"
        }
        config["workload"]["distribution_file"] = "frozen-distribution"
    atomic_write_json(merged.with_suffix(".frozen.json"), workload)
    atomic_write_json(merged, config)
    return merged


def main(
    argv: Optional[List[str]] = None,
    *,
    broker_factory: Optional[Callable[..., BrokerClient]] = None,
    server_runner: Optional[Callable[[GangWorkerContext], Any]] = None,
    master_executor: Optional[Callable[..., int]] = None,
) -> int:
    """Bazel entry point: parse runfile args and drive one gang worker."""
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Run one multi-machine gang worker shard."
    )
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--resolved-workload", required=True)
    parser.add_argument("--gang-plan", required=True)
    args = parser.parse_args(argv)

    outputs = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
    try:
        outputs.mkdir(parents=True, exist_ok=True)
        gang_plan = load_gang_plan(Path(args.gang_plan))
        coordinates = broker_coordinates(gang_plan)
        merged_config = _merge_workload(
            args.resolved_config, args.resolved_workload, outputs
        )
        # Resolve the mode here so a bad workload kind fails setup with the
        # intended message and exit code instead of escaping as a traceback.
        mode = (
            None if master_executor is not None else merged_config_mode(merged_config)
        )
        # The descriptor is local to this worker; controller paths are never inherited.
        context_path = write_run_context(
            outputs / "run_context.json",
            config_path=str(merged_config.resolve()),
            run_id=gang_plan["run_id"],
            repository_revisions=gang_plan.get("repository_revisions", {}),
            **({"mode": mode} if mode is not None else {}),
        )
        run_context = read_run_context(context_path, required=True)
    except (GangWorkerError, ValueError, OSError) as error:
        print(f"gang worker setup failed: {error}", file=sys.stderr)
        return getattr(error, "exit_code", EXIT_CONFIG_ERROR)
    factory = BrokerClient if broker_factory is None else broker_factory
    broker = factory(coordinates.url, coordinates.token, coordinates.run_id)
    # The master path follows the resolved config's mode: decode keeps the
    # existing decode master wiring, cache_grid runs the external prefill
    # session against the gang's own server. An explicit executor wins.
    executor = master_executor
    if executor is None:
        executor = (
            _default_master_executor
            if mode == "decode"
            else _default_prefill_master_executor
        )
    return run_gang_worker(
        merged_config,
        Path(args.gang_plan),
        broker=broker,
        server_runner=(
            (
                lambda context: _default_server_runner(
                    context, config_path=run_context["config_path"]
                )
            )
            if server_runner is None
            else server_runner
        ),
        test_outputs_dir=outputs,
        master_executor=executor,
    )


if __name__ == "__main__":
    raise SystemExit(main())
