"""Explicit SSH lifecycle for one immutable remote performance-test plan."""

from __future__ import annotations

import hashlib
import json
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rtp_llm.test.perf_test.errors import (
    CaseExecutionError,
    CollectionError,
    PerfTestError,
)
from rtp_llm.test.perf_test.execution.artifacts import (
    collect_master_artifacts,
    copy_member_traces,
)
from rtp_llm.test.perf_test.execution.gang import reconcile_terminal_state, record_stage
from rtp_llm.test.perf_test.plan import thaw
from rtp_llm.test.perf_test.run_directory import RunDirectoryLock, inspect_run_directory
from rtp_llm.test.perf_test.ssh.remote import RemoteHost
from rtp_llm.test.perf_test.ssh.source import build_identity
from rtp_llm.test.perf_test.state import RunOutcome, RunState
from rtp_llm.test.perf_test.storage import RunStore, atomic_write_json

_WORKER = "rtp_llm/test/perf_test/ssh/worker.py"
_BOOTSTRAP = """import json,os,pathlib,subprocess,sys
p=json.load(sys.stdin)
if os.getuid()==0: raise RuntimeError("SSH workers must not run as root")
root=pathlib.Path(p["work_dir"] or pathlib.Path.home()/"rtp_llm_perf_test").resolve()
if root==pathlib.Path("/") or root==pathlib.Path.home(): raise RuntimeError("unsafe work_dir")
if root.exists() and root.stat().st_uid!=os.getuid(): raise RuntimeError("work_dir is not owned by current user")
marker=root/".rtp-perf-owned"
if marker.is_symlink(): raise RuntimeError("invalid ownership marker")
if not marker.exists() and any((root/n).exists() for n in ("mirrors","sources","builds","runs")):
    raise RuntimeError("work_dir contains unmanaged performance directories")
root.mkdir(parents=True,exist_ok=True); marker.touch()
mirror=root/"mirrors"/"public.git"; mirror.parent.mkdir(exist_ok=True)
if not mirror.exists(): subprocess.run(["git","init","--bare",str(mirror)],check=True,stdout=subprocess.DEVNULL)
print(json.dumps({"root":str(root),"mirror":str(mirror)}))
"""
_CHECKOUT = """import fcntl,json,pathlib,subprocess,sys
p=json.load(sys.stdin); target=pathlib.Path(p["source_dir"]); target.parent.mkdir(parents=True,exist_ok=True)
with (target.parent/(target.name+".lock")).open("a") as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if not target.exists():
        subprocess.run(["git","clone","--no-checkout",p["mirror"],str(target)],check=True,stdout=subprocess.DEVNULL)
        subprocess.run(["git","-C",str(target),"checkout","--detach",p["revision"]],check=True,stdout=subprocess.DEVNULL)
    actual=subprocess.check_output(["git","-C",str(target),"rev-parse","HEAD"],text=True).strip()
    if actual!=p["revision"]: raise RuntimeError("cached checkout revision mismatch")
    if subprocess.run(["git","-C",str(target),"diff","--quiet","HEAD","--"]).returncode:
        raise RuntimeError("cached checkout contains uncommitted tracked changes")
    if not (target/p["worker_relative"]).is_file(): raise RuntimeError("selected revision has no SSH worker")
print(json.dumps({"source_sha":actual}))
"""


@dataclass(frozen=True)
class SshExecutionOutcome:
    state: RunState
    failure: Optional[PerfTestError] = None
    report: Optional[dict] = None
    signal_number: Optional[int] = None

    @property
    def exit_code(self):
        return RunOutcome(self.state, self.failure, self.signal_number).exit_code


def _parallel(members, operation, on_failure=None, on_interrupt=None):
    results, errors = {}, {}
    pool = ThreadPoolExecutor(max_workers=min(8, len(members) or 1))
    try:
        futures = {pool.submit(operation, member): member for member in members}
        for future in as_completed(futures):
            member = futures[future]
            key = str(member["member_id"])
            try:
                results[key] = future.result()
            except Exception as error:
                errors[key] = str(error)
                if on_failure is not None:
                    callback, on_failure = on_failure, None
                    callback()
    except BaseException:
        if on_interrupt is not None:
            on_interrupt()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return results, errors


def _host(member, execution, command_log=None):
    return RemoteHost(
        member["ssh_target"],
        user=execution.get("run_user"),
        port=execution.get("ssh_port"),
        python_bin=execution["python_bin"],
        command_log=command_log,
    )


def _terminal(root, store, manifest, state):
    manifest["state"] = state.value
    store.write_manifest(manifest)
    for relative in ("results/result.json", "results/summary.json"):
        path = root / relative
        if path.exists():
            payload = json.loads(path.read_text())
            payload["status"] = state.value
            atomic_write_json(path, payload)


class SshExecutor:
    def __init__(self, *, host_factory=_host):
        self.host_factory = host_factory

    def __call__(self, plan):
        return self.execute(plan)

    def execute(self, plan):
        if plan.backend != "ssh" or plan.action not in {"run", "report"}:
            raise ValueError("SSH executor requires an SSH run or offline report plan")
        root = Path(plan.result_directory)
        if plan.action == "report":
            from rtp_llm.test.perf_test.prefill.report import generate_prefill_report

            return SshExecutionOutcome(
                RunState.COMPLETED, report=generate_prefill_report(root)
            )
        with RunDirectoryLock(root, action="ssh-run"):
            snapshot = inspect_run_directory(root)
            if snapshot.execution_plan != plan:
                raise ValueError("persisted execution plan differs from SSH plan")
            return self._execute(plan, snapshot.source_config, snapshot.manifest)

    def _execute(self, plan, source_config, manifest):
        root = Path(plan.result_directory)
        execution = thaw(plan.resolved_config["execution"])
        source = execution["source"]
        store = RunStore(root)
        home = root / "execution" / "ssh"
        home.mkdir(parents=True, exist_ok=True)
        run_id = manifest["run_id"]
        bundle = {
            "config/execution_plan.json": plan.to_dict(),
            "config/source_config.json": source_config,
        }
        bundle_bytes = {
            name: json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode()
            for name, value in bundle.items()
        }
        bundle_hashes = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in bundle_bytes.items()
        }
        plan_hash = bundle_hashes["config/execution_plan.json"]
        members = [
            dict(member_id=str(index), member_index=index, **host)
            for index, host in enumerate(execution["hosts"])
        ]
        prepared = []
        failure = None
        interrupted = False
        state = RunState.FAILED
        collection = None
        report = None

        host_cache = {}
        host_lock = threading.Lock()
        transports_cancelled = False

        def host(member):
            with host_lock:
                key = member["member_id"]
                if key not in host_cache:
                    host_cache[key] = self.host_factory(
                        member, execution, home / "commands.jsonl"
                    )
                    if transports_cancelled:
                        host_cache[key].cancel_pending()
                return host_cache[key]

        def abort_pending():
            nonlocal transports_cancelled
            with host_lock:
                transports_cancelled = True
                remotes = tuple(host_cache.values())
            for remote in remotes:
                remote.cancel_pending()

        def reset_transports():
            nonlocal transports_cancelled
            with host_lock:
                for remote in tuple(host_cache.values()):
                    remote.reset_cancelled()
                transports_cancelled = False

        def persist_members():
            atomic_write_json(
                home / "members.json",
                {
                    "schema_version": 1,
                    "run_id": run_id,
                    "controller_plan_hash": plan_hash,
                    "execution": execution,
                    "members": prepared,
                },
            )
            atomic_write_json(home / "controller_plan.json", plan.to_dict())

        def prepare(member):
            remote = host(member)
            paths = json.loads(
                remote.python(_BOOTSTRAP, {"work_dir": execution.get("work_dir")})
            )
            remote.push_git(source["worktree"], source["ref"], paths["mirror"], run_id)
            identity = build_identity(source)
            source_dir = str(Path(paths["root"]) / "sources" / identity)
            remote.python(
                _CHECKOUT,
                {
                    "source_dir": source_dir,
                    "mirror": paths["mirror"],
                    "revision": source["ref"],
                    "worker_relative": _WORKER,
                },
            )
            run_dir = str(Path(paths["root"]) / "runs" / run_id / member["member_id"])
            descriptor = {
                "schema_version": 1,
                "run_id": run_id,
                "member_id": member["member_id"],
                "member_index": member["member_index"],
                "controller_plan_hash": plan_hash,
                "bundle_hashes": bundle_hashes,
                "run_dir": run_dir,
                "source": source,
                "source_dir": source_dir,
                "build_dir": str(Path(paths["root"]) / "builds" / identity),
                "worker_path": str(Path(source_dir) / _WORKER),
                "members": [
                    {
                        "hostname": item["advertise_host"],
                        "port": execution["start_port"],
                    }
                    for item in members
                ],
                "world_rank": member["member_index"]
                * plan.resource_plan.local_world_size,
                "local_world_size": plan.resource_plan.local_world_size,
                "devices": list(range(plan.resource_plan.local_world_size)),
                "start_port": execution["start_port"],
                "log_dir": str(Path(run_dir) / "logs"),
                "result_directory": str(Path(run_dir) / "run"),
            }
            for relative, value in bundle.items():
                remote.write_json(str(Path(run_dir) / relative), value)
            descriptor_path = str(Path(run_dir) / "descriptor.json")
            remote.write_json(descriptor_path, descriptor)
            local = home / "members" / member["member_id"]
            local.mkdir(parents=True, exist_ok=True)
            atomic_write_json(local / "descriptor.json", descriptor)
            complete = {
                **member,
                "descriptor": descriptor,
                "descriptor_path": descriptor_path,
                "worker_path": descriptor["worker_path"],
            }
            remote.worker(complete["worker_path"], descriptor_path, "preflight")
            return complete

        def worker(member, operation, timeout=120):
            return host(member).worker(
                member["worker_path"],
                member["descriptor_path"],
                operation,
                timeout=timeout,
            )

        active_stage = "rendezvous"
        try:
            record_stage(
                store,
                manifest,
                stage="rendezvous",
                status="running",
                state=RunState.RUNNING,
            )
            results, errors = _parallel(members, prepare, on_interrupt=abort_pending)
            prepared = sorted(results.values(), key=lambda item: item["member_index"])
            persist_members()
            if errors:
                raise CaseExecutionError(f"SSH preparation failed: {errors}")
            record_stage(store, manifest, stage="rendezvous", status="completed")
            active_stage = "build"
            record_stage(store, manifest, stage="build", status="running")
            _, errors = _parallel(
                prepared,
                lambda member: worker(member, "build", 14400),
                on_interrupt=abort_pending,
            )
            if errors:
                raise CaseExecutionError(f"SSH build failed: {errors}")
            record_stage(store, manifest, stage="build", status="completed")
            active_stage = "submit"
            record_stage(store, manifest, stage="submit", status="running")
            _, errors = _parallel(
                prepared,
                lambda member: worker(member, "start"),
                on_interrupt=abort_pending,
            )
            if errors:
                raise CaseExecutionError(f"SSH start failed: {errors}")

            def wait_member(member):
                value = worker(member, "wait", 86430)
                if value.get("state") != "completed" or value.get("exit_code") != 0:
                    raise CaseExecutionError(
                        f'worker {member["member_id"]} did not complete: {value}'
                    )
                return value

            statuses, errors = _parallel(
                prepared,
                wait_member,
                on_interrupt=abort_pending,
                on_failure=lambda: _parallel(
                    prepared, lambda member: worker(member, "cancel", 30)
                ),
            )
            for member_id, value in statuses.items():
                atomic_write_json(
                    home / "members" / member_id / "remote_status.json", value
                )
                if value.get("state") != "completed" or value.get("exit_code") != 0:
                    errors[member_id] = str(value)
            if errors:
                raise CaseExecutionError(f"SSH workers did not complete: {errors}")
            record_stage(store, manifest, stage="submit", status="completed")
        except KeyboardInterrupt:
            interrupted = True
            record_stage(
                store,
                manifest,
                stage=active_stage,
                status="failed",
                details={"error": "interrupted"},
            )
        except Exception as error:
            failure = (
                error
                if isinstance(error, PerfTestError)
                else CaseExecutionError(str(error))
            )
            manifest.setdefault("execution_errors", []).append(str(error))
            record_stage(
                store,
                manifest,
                stage=active_stage,
                status="failed",
                details={"error": str(error)},
            )
        finally:
            try:
                for stage in ("build", "submit"):
                    if stage not in manifest.get("stages", {}):
                        record_stage(store, manifest, stage=stage, status="skipped")
                if interrupted:
                    reset_transports()
                if failure is not None or interrupted:
                    _, cancel_errors = _parallel(
                        prepared, lambda member: worker(member, "cancel", 30)
                    )
                    if cancel_errors:
                        manifest["cancel_errors"] = cancel_errors
                record_stage(
                    store,
                    manifest,
                    stage="collect",
                    status="running",
                    state=RunState.COLLECTING,
                )

                def collect(member):
                    destination = home / "members" / member["member_id"] / "collected"
                    host(member).fetch_tree(
                        member["descriptor"]["run_dir"], destination
                    )
                    return destination

                trees, errors = _parallel(prepared, collect)
                if errors:
                    manifest["collection_errors"] = errors
                try:
                    if "0" not in trees:
                        raise CollectionError(
                            "SSH master did not produce a collected run tree"
                        )
                    collection = collect_master_artifacts(
                        root, trees["0"] / "run", plan=plan, run_id=run_id
                    )
                    for member_id, directory in trees.items():
                        if member_id == "0":
                            continue
                        copy_member_traces(
                            directory / "logs", root / "traces", member_id
                        )
                    if len(trees) != len(members):
                        raise CollectionError("SSH member artifacts are missing")
                    record_stage(store, manifest, stage="collect", status="completed")
                except Exception as error:
                    if failure is None:
                        failure = (
                            error
                            if isinstance(error, PerfTestError)
                            else CollectionError(str(error))
                        )
                    record_stage(
                        store,
                        manifest,
                        stage="collect",
                        status="failed",
                        details={"error": str(error)},
                    )
                state = (
                    RunState.INTERRUPTED
                    if interrupted
                    else reconcile_terminal_state(
                        collection.state if collection else RunState.FAILED,
                        submit_failure=failure,
                    )
                )
                _terminal(root, store, manifest, state)
                record_stage(
                    store,
                    manifest,
                    stage="report",
                    status="running",
                    state=RunState.REPORTING,
                )
                try:
                    if collection is None:
                        raise CollectionError(
                            "cannot report without a validated master result"
                        )
                    if plan.mode == "prefill":
                        from rtp_llm.test.perf_test.prefill.report import (
                            generate_prefill_report,
                        )

                        report = generate_prefill_report(root, acquire_lock=False)
                    else:
                        report = json.loads(
                            (root / "reports" / "decode_report.json").read_text()
                        )
                    record_stage(store, manifest, stage="report", status="completed")
                except Exception as error:
                    if failure is None and not interrupted:
                        failure = CollectionError(f"SSH report failed: {error}")
                        state = RunState.FAILED
                    record_stage(
                        store,
                        manifest,
                        stage="report",
                        status="failed",
                        details={"error": str(error)},
                    )
                _terminal(root, store, manifest, state)
            except KeyboardInterrupt:
                interrupted = True
                state = RunState.INTERRUPTED
            except BaseException:
                state = RunState.FAILED
                raise
            finally:
                # Collection/reporting can themselves be interrupted. Persist
                # a terminal state even when they cannot finish collecting.
                _terminal(root, store, manifest, state)
        return SshExecutionOutcome(
            state,
            None if interrupted else failure,
            report,
            signal.SIGINT if interrupted else None,
        )


def create_default_ssh_executor(plan):
    return SshExecutor()


# ---------------------------------------------------------------------------
# Maintenance of one saved SSH deployment (merged from ssh/maintenance.py);
# never rebuild or resume a model.
# ---------------------------------------------------------------------------

from rtp_llm.test.perf_test.ssh.remote import RemoteHost


def read_deployment(run_dir):
    root = Path(run_dir).resolve()
    home = root / "execution" / "ssh"
    deployment = json.loads((home / "members.json").read_text())
    plan = json.loads((home / "controller_plan.json").read_text())
    payload = json.dumps(
        plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    if (
        deployment.get("schema_version") != 1
        or hashlib.sha256(payload).hexdigest() != deployment["controller_plan_hash"]
    ):
        raise ValueError("SSH deployment plan identity mismatch")
    if deployment["execution"] != plan["resolved_config"]["execution"]:
        raise ValueError("SSH deployment execution differs from the frozen plan")
    if len(deployment["members"]) > plan["resource_plan"]["machine_count"]:
        raise ValueError("SSH deployment has extra members")
    if plan["backend"] != "ssh":
        raise ValueError("maintenance requires an SSH deployment")
    seen = set()
    for member in deployment["members"]:
        index = member["member_index"]
        if (
            type(index) is not int
            or not 0 <= index < plan["resource_plan"]["machine_count"]
            or index in seen
            or member.get("member_id") != str(index)
        ):
            raise ValueError("invalid or duplicate SSH member index")
        seen.add(index)
        expected = plan["resolved_config"]["execution"]["hosts"][index]
        if (
            member["ssh_target"] != expected["ssh_target"]
            or member["advertise_host"] != expected["advertise_host"]
        ):
            raise ValueError("SSH deployment host differs from the frozen plan")
        descriptor = member["descriptor"]
        if (
            member.get("worker_path") != descriptor.get("worker_path")
            or member.get("worker_path")
            != str(
                Path(descriptor["source_dir"]) / "rtp_llm/test/perf_test/ssh/worker.py"
            )
            or member.get("descriptor_path")
            != str(Path(descriptor["run_dir"]) / "descriptor.json")
        ):
            raise ValueError("SSH deployment worker/descriptor path mismatch")
        if (
            descriptor.get("controller_plan_hash") != deployment["controller_plan_hash"]
            or descriptor.get("run_id") != deployment["run_id"]
            or descriptor.get("member_id") != member["member_id"]
        ):
            raise ValueError("SSH member descriptor identity mismatch")
    return home, deployment


def maintain(run_dir, operation, dry_run=False):
    if operation not in {"status", "cancel", "collect", "clean"}:
        raise ValueError("unsupported SSH maintenance operation")
    home, deployment = read_deployment(run_dir)
    execution = deployment["execution"]
    result = {
        "operation": operation,
        "run_id": deployment["run_id"],
        "members": {},
        "errors": {},
    }
    for member in deployment["members"]:
        member_id = member["member_id"]
        remote = RemoteHost(
            member["ssh_target"],
            user=execution.get("run_user"),
            port=execution.get("ssh_port"),
            python_bin=execution["python_bin"],
            command_log=home / "commands.jsonl",
        )
        if dry_run:
            result["members"][member_id] = {
                "argv": remote.command(
                    [
                        execution["python_bin"],
                        member["worker_path"],
                        "--descriptor",
                        member["descriptor_path"],
                        "--operation",
                        operation,
                    ]
                )
            }
            continue
        try:
            if operation == "collect":
                # A distinct destination preserves previous evidence and terminal state.
                import uuid

                destination = home / "maintenance" / uuid.uuid4().hex / member_id
                remote.fetch_tree(member["descriptor"]["run_dir"], destination)
                result["members"][member_id] = {"collected": str(destination)}
            else:
                result["members"][member_id] = remote.worker(
                    member["worker_path"],
                    member["descriptor_path"],
                    operation,
                    timeout=45,
                )
        except Exception as error:
            result["errors"][member_id] = str(error)
    result["exit_code"] = 1 if result["errors"] else 0
    return result
