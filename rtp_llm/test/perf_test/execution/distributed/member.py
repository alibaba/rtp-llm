"""Execute one frozen distributed member, independent of SSH or AOne scheduling."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from rtp_llm.test.perf_test.execution.devices import resolve_allocated_devices
from rtp_llm.test.perf_test.plan import bind_member_plan
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.storage import RunStore


def check_member_health(handle, check_cancelled):
    from rtp_llm.test.perf_test.errors import EngineLifecycleError

    try:
        check_cancelled()
    except Exception as error:
        raise EngineLifecycleError(f"distributed member cancelled: {error}") from error
    if handle.manager.is_alive() is False:
        raise EngineLifecycleError(
            f"distributed engine exited with code {handle.manager.exit_code}; log: {handle.log_file_path}"
        )


def begin_master_run(root, plan, *, source_config, run_context=None):
    """Initialize a master run directory and flip its manifest into running state."""
    run_context = run_context or {}
    revisions = run_context.get("repository_revisions", {})
    run_id = run_context.get("run_id", root.name)
    initialize_run_directory(
        root,
        source_config=source_config,
        plan=plan,
        run_id=run_id,
        code_version=revisions.get("public", "unknown"),
        repository_revisions=revisions,
    )
    store = RunStore(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["state"] = "running"
    store.write_manifest(manifest)
    return run_id, store, manifest


class CompletionGroup:
    """An explicit run-bound completion signal for non-load-driving members.

    The control listener starts before engine startup. Peers send their frozen
    run/rank identity, then keep their engines alive until the master broadcasts
    a terminal outcome. A closed or stale control connection is an error, never
    evidence that measurement succeeded.
    """

    def __init__(
        self,
        *,
        run_id,
        world_rank,
        member_ranks,
        master_host,
        port,
        timeout_seconds=86400,
        check_cancelled=lambda: None,
    ):
        self.run_id = run_id
        self.world_rank = world_rank
        self.expected = set(member_ranks) - {0}
        self.check_cancelled = check_cancelled
        self.deadline = time.monotonic() + timeout_seconds
        self.peers = {}
        self.listener = None
        self.connection = None
        if world_rank == 0:
            self.listener = socket.socket()
            self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.listener.bind(("0.0.0.0", port))
            self.listener.listen(max(1, len(self.expected)))
            self.listener.settimeout(0.5)
        else:
            while self.connection is None:
                self._check_deadline()
                try:
                    self.connection = socket.create_connection(
                        (master_host, port), timeout=0.5
                    )
                except (ConnectionRefusedError, TimeoutError, OSError):
                    # Bounded startup backoff; the next attempt also checks cancellation.
                    threading.Event().wait(0.1)
            self.connection.sendall(
                (
                    json.dumps({"run_id": run_id, "world_rank": world_rank}) + "\n"
                ).encode()
            )

    def _check_deadline(self):
        self.check_cancelled()
        if time.monotonic() >= self.deadline:
            raise TimeoutError("distributed completion control timed out")

    def _read(self, connection):
        connection.settimeout(0.5)
        buffer = bytearray()
        while b"\n" not in buffer:
            self._check_deadline()
            try:
                chunk = connection.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                raise RuntimeError("distributed completion peer disconnected")
            buffer.extend(chunk)
            if len(buffer) > 4096:
                raise RuntimeError("invalid distributed completion message")
        return json.loads(bytes(buffer).split(b"\n", 1)[0])

    def ready(self):
        if self.listener is None:
            return
        while set(self.peers) != self.expected:
            self._check_deadline()
            try:
                connection, _address = self.listener.accept()
            except socket.timeout:
                continue
            try:
                message = self._read(connection)
                rank = message.get("world_rank")
                if (
                    message.get("run_id") != self.run_id
                    or rank not in self.expected
                    or rank in self.peers
                ):
                    raise RuntimeError(
                        "completion member identity does not match the frozen gang"
                    )
                self.peers[rank] = connection
            except BaseException:
                connection.close()
                raise

    def wait(self):
        message = self._read(self.connection)
        if (
            message.get("run_id") != self.run_id
            or type(message.get("exit_code")) is not int
        ):
            raise RuntimeError("completion result does not match this run")
        return message["exit_code"]

    def finish(self, exit_code):
        if self.listener is not None:
            message = (
                json.dumps({"run_id": self.run_id, "exit_code": exit_code}) + "\n"
            ).encode()
            for connection in self.peers.values():
                try:
                    connection.sendall(message)
                except OSError:
                    pass
        self.close()

    def close(self):
        for connection in [*self.peers.values(), self.connection, self.listener]:
            if connection is not None:
                connection.close()
        self.peers.clear()


def run_member(plan, *, source_config, descriptor, check_cancelled=lambda: None):
    """Own this host's server while consuming the controller's frozen plan."""
    topology = plan.resolved_config["model"]["topology"]
    local_world_size = descriptor["local_world_size"]
    world_rank = descriptor["world_rank"]
    members = descriptor["members"]
    devices = descriptor["devices"]
    world_size = topology["tp"] * topology["dp"] * topology["pp"]
    if (
        type(local_world_size) is not int
        or local_world_size <= 0
        or world_size != len(members) * local_world_size
        or type(world_rank) is not int
        or world_rank % local_world_size
        or not 0 <= world_rank < world_size
        or len(devices) != local_world_size
        or len(set(devices)) != len(devices)
    ):
        raise ValueError("member runtime layout does not match the frozen topology")
    allocated_devices = resolve_allocated_devices(devices)
    port = descriptor["start_port"]
    if not 0 < port <= 65535 - 1025:
        raise ValueError("member start_port cannot fit distributed control ports")
    from rtp_llm.config.server_config_setup import set_parallelism_config
    from rtp_llm.server.server_args.server_args import setup_args
    from rtp_llm.test.perf_test.execution.distributed.decode import (
        execute_decode_member,
    )
    from rtp_llm.test.perf_test.execution.distributed.prefill import (
        execute_master_prefill,
    )
    from rtp_llm.test.perf_test.execution.distributed.rank_group import (
        LocalRankGroupContext,
        build_gang_config_string,
        start_local_rank_group,
    )
    from rtp_llm.test.perf_test.execution.launch_boundary import (
        configure_perf_environment,
    )
    from rtp_llm.test.perf_test.execution.runtime import build_engine_launch_spec

    plan = bind_member_plan(plan, result_directory=str(descriptor["result_directory"]))
    run_id = descriptor["run_id"]
    log_dir = Path(descriptor["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    member_records = [
        SimpleNamespace(
            name=f"test_part{index}", ip=member["hostname"], server_port=member["port"]
        )
        for index, member in enumerate(members)
    ]
    runtime = {
        "GANG_CONFIG_STRING": build_gang_config_string(
            member_records, local_world_size
        ),
        "WORLD_SIZE": str(world_size),
        "WORLD_RANK": str(world_rank),
        "LOCAL_WORLD_SIZE": str(local_world_size),
        "TP_SIZE": str(topology["tp"]),
        "DP_SIZE": str(topology["dp"]),
        "PP_SIZE": str(topology["pp"]),
        "EP_SIZE": str(topology["ep"]),
        "START_PORT": str(port),
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, allocated_devices)),
        "HIP_VISIBLE_DEVICES": ",".join(map(str, allocated_devices)),
    }
    launch = build_engine_launch_spec(plan)
    environment = {**launch.environment, **runtime}
    if topology["tp"] > local_world_size:
        environment["FT_DISABLE_CUSTOM_AR"] = "1"
    configure_perf_environment(environment)
    configs = setup_args(list(launch.argv))
    set_parallelism_config(
        configs.parallelism_config, py_prefill_cp_config=configs.prefill_cp_config
    )
    context = LocalRankGroupContext(
        port=port,
        world_rank=world_rank,
        log_dir_path=str(log_dir),
        profiler_directory=log_dir,
        plan=plan,
    )
    completion = (
        CompletionGroup(
            run_id=run_id,
            world_rank=world_rank,
            member_ranks=tuple(
                index * local_world_size for index in range(len(members))
            ),
            master_host=members[0]["hostname"],
            port=port + 1025,
            check_cancelled=check_cancelled,
        )
        if len(members) > 1
        else None
    )
    handle = None
    exit_code = 1
    try:
        check_cancelled()
        handle = start_local_rank_group(
            context,
            configs,
            configs.model_args.tokenizer_path,
            engine_argv=launch.argv,
            engine_environment=environment,
        )
        if not handle.start_server():
            handle.print_process_log()
            raise RuntimeError("member engine startup failed")
        if world_rank == 0 and completion is not None:
            completion.ready()
        result = None
        if plan.mode == "decode":
            tokenizer = handle.tokenizer_path or configs.model_args.tokenizer_path
            result = execute_decode_member(
                plan,
                source_config=source_config,
                endpoint=f"http://127.0.0.1:{port}",
                tokenizer_path=tokenizer,
                server_log_path=handle.log_file_path,
                profiler_directory=log_dir,
                world_rank=world_rank,
                local_world_size=local_world_size,
                member_hosts=tuple(member["hostname"] for member in members),
                control_port=port + 1024,
                run_context=descriptor,
                check_cancelled=lambda: check_member_health(handle, check_cancelled),
            )
        elif world_rank == 0:
            result = execute_master_prefill(
                plan,
                source_config=source_config,
                endpoint=f"http://127.0.0.1:{port}",
                tokenizer_path=handle.tokenizer_path,
                server_log_path=handle.log_file_path,
                profiler_directory=log_dir,
                run_context=descriptor,
                health_check=lambda: check_member_health(handle, check_cancelled),
            )
        if world_rank:
            exit_code = completion.wait() if completion is not None else 0
        else:
            exit_code = 0 if result.state.value == "completed" else 1
        check_cancelled()
        return exit_code
    finally:
        if completion is not None:
            if world_rank == 0:
                completion.finish(exit_code)
            else:
                completion.close()
        if handle is not None:
            handle.stop_server()
