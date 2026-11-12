"""Own OS processes and interrupt operations when one exits unexpectedly."""

import asyncio
import json
import os
import signal
from dataclasses import dataclass

from ..config.validation import duration
from . import ExecutionError
from .step_trace import write_json


@dataclass(frozen=True)
class ProcessIdentity:
    role: str
    index: int | None = None
    generation: int = 0

    @property
    def instance_id(self):
        return self.role if self.index is None else f"{self.role}-{self.index}"

    @property
    def process_name(self):
        if self.generation:
            return f"{self.instance_id}-restart-{self.generation}"
        return self.instance_id

    def directory(self, run_directory):
        directory = run_directory / "components" / self.instance_id
        return (
            directory / f"restart-{self.generation}" if self.generation else directory
        )

    def event_fields(self):
        return {
            "role": self.role,
            "instance_id": self.instance_id,
            "generation": self.generation,
        }


class ProcessSupervisor:
    def __init__(self, records):
        self.records = records
        self._children, self._monitors = {}, []
        self._expected, self._zero_exit = set(), set()
        self.failure = asyncio.get_running_loop().create_future()
        self.closing = False

    @property
    def children(self):
        return tuple(
            (process_name, child) for process_name, (_, child) in self._children.items()
        )

    def child(self, process_name):
        return self._children[process_name][1]

    def process_identities(self, *roles):
        return tuple(
            identity
            for identity, _ in self._children.values()
            if not roles or identity.role in roles
        )

    def fail(self, message):
        if not self.closing and not self.failure.done():
            self.failure.set_result(message)

    async def _monitor(self, process_name, child):
        code = await child.wait()
        if process_name not in self._expected:
            self.fail(f"Component {process_name} exited unexpectedly with code {code}")

    async def guard(self, operation, timeout):
        task = asyncio.create_task(operation)
        try:
            async with asyncio.timeout(timeout):
                done, _ = await asyncio.wait(
                    (task, self.failure), return_when=asyncio.FIRST_COMPLETED
                )
                if self.failure in done:
                    raise ExecutionError(self.failure.result())
                return task.result()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _process_event(self, event, identity, **fields):
        self.records.emit(
            event=event, node=identity.process_name, **identity.event_fields(), **fields
        )

    async def spawn(
        self,
        identity: ProcessIdentity,
        command,
        cwd,
        env,
        directory,
        *,
        zero_exit=False,
    ):
        process_name = identity.process_name
        if process_name in self._children or process_name in self._expected:
            raise ExecutionError(
                f"Process identity has already been used: {process_name}"
            )

        def retain(child):
            self._children[process_name] = (identity, child)
            if zero_exit:
                self._zero_exit.add(process_name)
            self._monitors.append(
                asyncio.create_task(self._monitor(process_name, child))
            )
            self._process_event("process_started", identity, pid=child.pid)

        with (directory / "stdout.log").open("wb") as out, (
            directory / "stderr.log"
        ).open("wb") as err:
            spawning = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *command,
                    cwd=cwd,
                    env=env,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
            )
            try:
                child = await asyncio.shield(spawning)
            except asyncio.CancelledError:
                retain(await spawning)
                raise
        retain(child)
        return child

    async def stop_intentionally(self, process_name, *, mode="terminate", grace=3):
        identity, child = self._children[process_name]
        if child.returncode is not None:
            raise ExecutionError(
                f"{process_name} already exited before intentional {mode}"
            )
        if mode not in ("terminate", "kill"):
            raise ExecutionError(f"Unknown intentional stop mode: {mode}")
        if mode == "terminate":
            return await self.stop([process_name], grace)
        self._expected.add(process_name)
        errors = []
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except OSError as error:
            errors.append(f"{process_name}: cannot kill process group: {error}")
        try:
            await asyncio.wait_for(child.wait(), timeout=grace)
        except TimeoutError:
            errors.append(
                f"{process_name}: process did not exit after intentional SIGKILL"
            )
        if child.returncode is not None:
            if child.returncode not in (-signal.SIGKILL, 128 + signal.SIGKILL):
                errors.append(
                    f"{process_name}: unexpected exit status during kill: {child.returncode}"
                )
            del self._children[process_name]
            self._record_stop(identity, child.returncode, errors)
        return errors

    async def stop(self, process_names, grace):
        children = [
            (process_name, *self._children[process_name])
            for process_name in process_names
            if process_name in self._children
        ]
        errors = [
            f"{process_name}: exited unexpectedly before stop with code {child.returncode}"
            for process_name, _, child in children
            if child.returncode is not None and process_name not in self._expected
        ]
        self._expected.update(process_name for process_name, _, _ in children)
        # Kill only owned process groups, including grandchildren after their leader exits.
        for process_name, _, child in reversed(children):
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f"{process_name}: {error}")
        deadline = asyncio.get_running_loop().time() + grace
        while (
            any(child.returncode is None for _, _, child in children)
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.02)
        for process_name, identity, child in reversed(children):
            killed = False
            try:
                os.killpg(child.pid, signal.SIGKILL)
                killed = child.returncode is None
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f"{process_name}: {error}")
            try:
                await asyncio.wait_for(child.wait(), timeout=3)
            except TimeoutError:
                errors.append(f"{process_name}: process did not exit after SIGKILL")
            if killed:
                errors.append(
                    f"{process_name}: did not stop within the grace period; SIGKILL was required"
                )
            if child.returncode not in (
                0,
                -signal.SIGTERM,
                128 + signal.SIGTERM,
            ):
                errors.append(
                    f"{process_name}: unexpected exit status during stop: {child.returncode}"
                )
            if process_name in self._zero_exit and (killed or child.returncode != 0):
                errors.append(
                    f"{process_name}: did not stop gracefully; retained data may be incomplete"
                )
            if child.returncode is not None:
                del self._children[process_name]
            self._record_stop(identity, child.returncode, errors)
        return errors

    def _record_stop(self, identity, code, errors):
        try:
            self._process_event("process_stopped", identity, code=code)
        except OSError as error:
            errors.append(f"{identity.process_name}: record process_stopped: {error}")

    async def close(self):
        self.closing = True
        errors = await self.stop(list(self._children), 3)
        for monitor in self._monitors:
            if not monitor.done():
                monitor.cancel()
        await asyncio.gather(*self._monitors, return_exceptions=True)
        return errors


async def start_instance(
    supervisor, records, role_config, identity, launch_config, *, configure=None
):
    instance_id = identity.instance_id
    directory = identity.directory(records.directory)
    directory.mkdir(parents=True)
    config, ready = directory / "config.json", directory / "ready.json"
    write_json(
        config,
        {
            "component_role": identity.role,
            "component": role_config,
            "id": instance_id,
        },
    )
    env = {} if configure is None else configure(directory)
    child = await supervisor.spawn(
        identity,
        launch_config["command"],
        launch_config["cwd"],
        {
            **os.environ,
            **launch_config["env"],
            **env,
            "MOCK_TOPOLOGY_FILE": str(records.directory / "topology.json"),
            "MOCK_NODE_ID": instance_id,
            "MOCK_CONFIG_FILE": str(config),
            "MOCK_READY_FILE": str(ready),
            "MOCK_OUTPUT_DIR": str(directory),
        },
        directory,
    )

    async def await_ready():
        while not ready.exists():
            await asyncio.sleep(0.02)
        value = json.loads(ready.read_text())
        if set(value) != {"url"} or not isinstance(value["url"], str):
            raise ExecutionError(f"Invalid readiness file for {instance_id}")
        return value["url"]

    control = await supervisor.guard(
        await_ready(), float(duration(launch_config["startup_timeout"]))
    )
    records.emit(
        event="process_ready",
        node=identity.process_name,
        url=control,
        **identity.event_fields(),
    )
    path = directory / "endpoints.json"
    endpoint = json.loads(path.read_text()) if path.is_file() else None
    return child, control, endpoint
