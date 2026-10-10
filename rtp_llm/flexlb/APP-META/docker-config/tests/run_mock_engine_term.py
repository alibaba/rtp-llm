"""Run a packaged FlexLB via whale_start.sh, send real gRPC traffic, TERM its entry.

Build flexlb-api and flexlb-mock-engine first; see README.md. This is a local
single-master integration test, not a container-runtime or ZK handoff test.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

CONFIG = Path(__file__).resolve().parents[1]
FLEXLB = CONFIG.parents[1]
sys.path.insert(0, str(FLEXLB / "tools/online_eval"))

from flexlb_ft.harness import (  # noqa: E402
    API_JAR,
    MOCK_JAR,
    JAVA_MODULE_OPTS,
    EnvManager,
    EnvSpec,
    flexlb_config_for_profile,
    http_get_json,
    http_post_json,
    resolve_java21,
    wait_for,
)
from flexlb_ft.engine_ops import EngineOps  # noqa: E402
from google.protobuf.json_format import MessageToDict  # noqa: E402


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario", choices=("queued", "engine", "late"), default="queued"
    )
    parser.add_argument(
        "--quiet-period-ms",
        type=int,
        default=5000,
        help="Write grpcServer.shutdownQuietPeriodMs into FLEXLB_CONFIG",
    )
    args = parser.parse_args()
    if args.quiet_period_ms < 2000:
        parser.error("quiet period must be at least 2000ms for this test")
    quiet_seconds = args.quiet_period_ms / 1000
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ["FLEXLB_EVAL_PROTO_OUT"] = str(root / "proto")
    timeline = []

    def event(name, **fields):
        record = {"epoch_ms": time.time_ns() // 1_000_000, "event": name, **fields}
        timeline.append(record)
        with (root / "client-events.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def save(name, data):
        (root / name).write_text(json.dumps(data, indent=2))

    manager = EnvManager(root / "engine", keep=True)
    proc = None
    ops = None
    clients = ThreadPoolExecutor(max_workers=3)
    pid_file = root / "service.pid"
    summary = {
        "passed": False,
        "scenario": args.scenario,
        "quiet_period_ms": args.quiet_period_ms,
        "scope": "single master, full packaged application, real gRPC, Java mock P/D engines",
    }
    try:
        for jar in (API_JAR, MOCK_JAR):
            check(jar.is_file(), f"Build missing jar first: {jar}")
        save(
            "artifacts.json",
            {
                str(jar): hashlib.sha256(jar.read_bytes()).hexdigest()
                for jar in (API_JAR, MOCK_JAR)
            },
        )
        env = manager.ensure(
            EnvSpec(
                label="term",
                n_prefill=1,
                n_decode=1,
                master_profile="none",
                mock_heap="512m",
                event_loop_threads=2,
                completion_threads=2,
            )
        )
        config = json.loads(flexlb_config_for_profile("single-batch"))
        # A real long decode occupies the sole decode slot. Subsequent Schedule
        # RPCs remain queued inside the real scheduler when TERM arrives.
        config["router"]["roles"]["decode"]["availability"]["maxEngineRequests"] = 1
        config.setdefault("grpcServer", {})[
            "shutdownQuietPeriodMs"
        ] = args.quiet_period_ms
        save("flexlb-config.json", config)
        launch_env = dict(os.environ, **manager._master_env(env))
        launch_env["FLEXLB_CONFIG"] = json.dumps(config)
        launch_env["FLEXLB_SYNC_CONSISTENCY_CONFIG"] = '{"needConsistency":false}'
        launch_env["APP_NAME"] = "FlexLB"
        bin_dir = root / "bin"
        bin_dir.mkdir()
        (root / "target").mkdir()
        (root / "target/FlexLB.tgz").touch()
        (root / "logs").mkdir()
        for name in ("appctl.sh", "hook.sh"):
            shutil.copyfile(CONFIG / "environment/common/bin" / name, bin_dir / name)
        (bin_dir / "setenv.sh").write_text(
            f"APP_NAME=FlexLB\nSERVICE_PID={shlex.quote(str(pid_file))}\n"
            "NGINXCTL=true\nENABLE_XAGENT=false\n"
            f"CATALINA_PID={shlex.quote(str(root / 'absent-tomcat.pid'))}\n"
        )
        java_args = [
            resolve_java21(),
            "-Xms256m",
            "-Xmx1g",
            *JAVA_MODULE_OPTS,
            "-jar",
            str(API_JAR),
            f"--server.port={env.master_http_port}",
            f"--management.server.port={env.master_management_port}",
            "--spring.profiles.active=default",
            f"--flexlb.log.path={root / 'logs'}",
        ]
        save("java-command.json", java_args)
        startup = root / "start.sh"
        startup.write_text(
            "#!/bin/bash\n"
            + shlex.join(java_args)
            + f" > {shlex.quote(str(root / 'java-console.log'))} 2>&1 &\n"
            + f"echo $! > {shlex.quote(str(pid_file))}\n"
        )
        startup.chmod(0o755)
        stop = root / "stop.sh"
        stop.write_text(
            f"#!/bin/bash\nexec /bin/bash {shlex.quote(str(bin_dir / 'appctl.sh'))} stop\n"
        )
        source = (CONFIG / "environment/common/bin/startx.sh").read_text()
        source = source.replace(
            "STARTUP=/home/admin/start.sh", f"STARTUP={shlex.quote(str(startup))}"
        )
        source = source.replace(
            "STOP=/home/admin/stop.sh", f"STOP={shlex.quote(str(stop))}"
        )
        # Installation ownership/chmod is unrelated to stop; no /home/admin on macOS.
        source = source.replace(
            "\nlisten_signal\ndo_start", "\nprepare() { :; }\nlisten_signal\ndo_start"
        )
        (bin_dir / "startx.sh").write_text(source)
        wrapper = root / "whale_start.sh"
        wrapper.write_text(
            (CONFIG / "environment/common/bin/whale_start.sh")
            .read_text()
            .replace(
                "/home/admin/${APP_NAME}/bin/startx.sh", str(bin_dir / "startx.sh")
            )
        )
        if not shutil.which("setsid"):
            shim = bin_dir / "setsid"
            shim.write_text(
                f"#!{sys.executable}\nimport os,sys\nos.setsid()\nos.execvp(sys.argv[1],sys.argv[1:])\n"
            )
            shim.chmod(0o755)
        launch_env["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
        with (root / "entry.log").open("w") as output:
            proc = subprocess.Popen(
                ["sh", str(wrapper)],
                env=launch_env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        event("entry-start", pid=proc.pid, command=["sh", str(wrapper)])
        check(
            wait_for(
                lambda: bool(
                    (
                        http_post_json(env.master_http("/rtp_llm/master/info"), {})[1]
                        or {}
                    ).get("ready")
                ),
                90,
                0.5,
            ),
            "full FlexLB failed to start",
        )
        ops = EngineOps(
            "127.0.0.1",
            env.master_http_port,
            env.mock_http_port,
            env.master_management_port,
        )
        check(
            wait_for(lambda: ops.verify_recovery()[0], 30, 1), "warmup request failed"
        )
        save(
            "master-info-before.json",
            http_post_json(env.master_http("/rtp_llm/master/info"), {})[1],
        )
        save(
            "processes-before.txt.json",
            subprocess.check_output(
                [
                    "ps",
                    "-o",
                    "pid,ppid,pgid,command",
                    "-p",
                    f"{proc.pid},{pid_file.read_text().strip()}",
                ],
                text=True,
            ),
        )
        request_base = 917_210_000
        records = []

        def schedule(rid, length):
            event("schedule-send", request_id=rid, output_len=length)
            response = ops.schedule(
                rid, output_len=length, input_len=1024, timeout_s=45
            )
            event(
                "schedule-return",
                request_id=rid,
                code=response.code,
                success=response.success,
            )
            check(
                response.success and response.code == 200, f"Schedule {rid}: {response}"
            )
            check(
                response.enqueued_by_master,
                "test requires actual EnqueueBatch dispatch",
            )
            save(
                f"schedule-{rid}.json",
                MessageToDict(response, preserving_proto_field_name=True),
            )
            stream = ops.start_stream(response, rid)
            records.append((rid, length, stream))
            return response

        schedule(request_base, 10 if args.scenario == "late" else 2000)
        if args.scenario == "late":
            check(records[0][2].wait_end(5), "initial request did not complete")
            time.sleep(quiet_seconds + 0.5)
            check(proc.poll() is None, "normal idle service exited without TERM")
            check(
                "keep serving until no new requests"
                not in (root / "logs/flexlb.log").read_text(),
                "drain started before TERM",
            )
            event("normal-idle-longer-than-quiet-period", pid=proc.pid)
        else:
            check(
                wait_for(
                    lambda: any(
                        e.get("running", 0) > 0 and e.get("role") == "decode"
                        for e in ops.snapshot().get("engines", [])
                    ),
                    5,
                    0.1,
                ),
                "long request never entered decode",
            )
        pending_count = 3 if args.scenario == "queued" else 0
        pending = [
            clients.submit(schedule, request_base + n, 10)
            for n in range(1, pending_count + 1)
        ]

        def all_admitted():
            data = http_get_json(env.master_http("/rtp_llm/inflight_status")) or {}
            return data.get("scheduler_inflight", 0) >= pending_count + 1

        if args.scenario != "late":
            check(wait_for(all_admitted, 5, 0.05), "requests not admitted before TERM")
        check(
            all(not item.done() for item in pending),
            "Schedule was not pending before TERM",
        )
        save(
            "master-inflight-at-term.json",
            http_get_json(env.master_http("/rtp_llm/inflight_status")),
        )
        save("engine-at-term.json", ops.snapshot())
        event(
            "send-TERM-to-entry",
            pid=proc.pid,
            java_pid=int(pid_file.read_text()),
            pending_schedule=pending_count,
        )
        proc.send_signal(signal.SIGTERM)
        if pending:
            time.sleep(6)
            check(
                proc.poll() is None,
                "entry exited before the pending Schedule requests drained",
            )
            check(
                all(not item.done() for item in pending),
                "pending RPC did not survive six seconds",
            )
            event("entry-and-three-schedule-RPCs-alive-after-6s", pid=proc.pid)
        elif args.scenario == "late":
            check(
                wait_for(
                    lambda: "keep serving until no new requests"
                    in (root / "logs/flexlb.log").read_text(),
                    5,
                    0.05,
                ),
                "TERM did not enter idle drain",
            )
            for n in range(1, 5):
                time.sleep(quiet_seconds * 0.3)
                check(proc.poll() is None, "entry exited while late arrivals continued")
                schedule(request_base + n, 10)
                check(records[-1][2].wait_end(5), "late engine stream did not finish")
            event("late-arrivals-stopped", requests=4)
            time.sleep(quiet_seconds * 0.6)
            check(
                proc.poll() is None, "quiet window was not restarted by late arrivals"
            )
            event(
                "entry-alive-after-last-arrival",
                pid=proc.pid,
                wait_seconds=quiet_seconds * 0.6,
            )
        else:
            check(proc.wait(timeout=15) == 0, "entry did not exit successfully")
            event(
                "entry-exit-while-engine-running",
                pid=proc.pid,
                exit_code=proc.returncode,
            )
            check(
                not records[0][2].snap.terminated,
                "long engine request ended before entry exit",
            )
            save("engine-at-entry-exit.json", ops.snapshot())
        for item in pending:
            item.result(timeout=40)
        for rid, length, stream in records:
            check(stream.wait_end(40), f"stream timeout: {rid}")
            snap = stream.snap
            save(
                f"stream-{rid}.json",
                [
                    MessageToDict(frame, preserving_proto_field_name=True)
                    for frame in snap.outputs
                ],
            )
            check(
                snap.completed and not snap.error and snap.stream_error_code is None,
                f"stream failure {rid}: {snap.error}, {snap.stream_error_code}",
            )
            terminal = snap.outputs[-1]
            check(terminal.request_id == rid, f"wrong response request ID for {rid}")
            check(
                terminal.flatten_output.aux_info[0].output_len == length,
                f"truncated output for {rid}",
            )
            event(
                "stream-complete",
                request_id=rid,
                expected_output_tokens=length,
                frames=len(snap.outputs),
            )
        check(proc.wait(timeout=40) == 0, "entry did not exit successfully")
        event("entry-exit", pid=proc.pid, exit_code=proc.returncode)
        check(not pid_file.exists(), "appctl did not observe Java exit")
        output = (root / "entry.log").read_text()
        check("exited; service stopped." in output, "missing Java exit confirmation")
        check(
            "application stopped; exiting container supervisor" in output,
            "missing supervisor exit confirmation",
        )
        shutdown_lines = (root / "logs/flexlb.log").read_text().splitlines()

        def log_time(fragment):
            matches = [line for line in shutdown_lines if fragment in line]
            check(
                len(matches) == 1,
                f"expected one shutdown event: {fragment}, got {matches}",
            )
            return int(
                datetime.strptime(matches[0][:23], "%Y-%m-%d %H:%M:%S.%f").timestamp()
                * 1000
            )

        started = log_time("keep serving until no new requests")
        quiet = log_time("Schedule quiet period elapsed")
        drained = log_time("All accepted gRPC requests completed")
        last_arrival = max(
            e["epoch_ms"] for e in timeline if e["event"] == "schedule-send"
        )
        term = next(
            e["epoch_ms"] for e in timeline if e["event"] == "send-TERM-to-entry"
        )
        check(started >= term, "quiet timer started before TERM")
        check(
            quiet - max(started, last_arrival) >= args.quiet_period_ms - 2,
            "gRPC stopped before a full quiet period since TERM/latest arrival",
        )
        if pending:
            check(
                drained - quiet > 5000,
                "test must exercise accepted RPC drain beyond quiet period",
            )
        summary["shutdown_timing_ms"] = dict(
            term=term,
            drain_started=started,
            last_schedule_sent=last_arrival,
            quiet_elapsed=quiet,
            accepted_rpcs_drained=drained,
        )
        save("engine-after.json", ops.snapshot())
        engine_requests = http_get_json(
            f"http://127.0.0.1:{env.mock_http_port}/requests"
        )
        save("engine-requests.json", engine_requests)
        for rid, _, _ in records:
            for role in ("prefill-0", "decode-0"):
                check(
                    engine_requests[role][str(rid)]["end_state"] == "completed",
                    f"{role} did not complete request {rid}",
                )
        summary.update(
            passed=True,
            requests=len(records),
            request_ids=[r[0] for r in records],
            entry_exit=proc.returncode,
        )
    except Exception as failure:
        summary["failure"] = repr(failure)
        traceback.print_exc()
    finally:
        save("result.json", summary)
        if ops:
            ops.close()
        if proc and proc.poll() is None:
            # Only this test's known process group; cleanup is not a tested stop.
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        clients.shutdown(wait=True)
        manager.teardown()
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
