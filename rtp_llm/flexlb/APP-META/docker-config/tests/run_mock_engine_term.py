"""Verify packaged FlexLB TERM using the current Mock Engine and Client artifacts."""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from urllib.error import URLError
from urllib.request import Request, urlopen

CONFIG = Path(__file__).resolve().parents[1]
FLEXLB = CONFIG.parents[1]
FRAMEWORK = FLEXLB / "mock-framework"
sys.path.insert(0, str(FRAMEWORK / "python"))
from mock_framework.input import pb, write_inputs
from mock_framework.evidence import read_requests
from mock_framework.execution.master import master_environment


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def http(url, value=None):
    request = Request(url, data=None if value is None else json.dumps(value).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=70) as response:
        return json.load(response)


def wait_for(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except (OSError, URLError):
            pass
        time.sleep(.05)
    raise AssertionError("Timed out waiting for " + predicate.__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=("queued", "engine", "late"), default="queued")
    parser.add_argument("--quiet-period-ms", type=int, default=5000)
    args = parser.parse_args()
    if not 2000 <= args.quiet_period_ms <= 10000:
        parser.error("quiet period must be between 2000 and 10000ms for this bounded test")
    java = str(Path(os.environ["JAVA_HOME"]) / "bin/java")
    api_jar = FLEXLB / "flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar"
    engine_jar = FRAMEWORK / "java/host/target/mock-engine-service-1.0-SNAPSHOT-all.jar"
    client_jar = FRAMEWORK / "java/client/target/mock-client-1.0-SNAPSHOT-all.jar"
    for jar in (api_jar, engine_jar, client_jar):
        check(jar.is_file(), f"Build missing artifact first: {jar}")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    children, timeline = [], []
    proc = None
    pid_file = root / "service.pid"
    quiet_seconds = args.quiet_period_ms / 1000
    summary = {"passed": False, "scenario": args.scenario,
               "scope": "packaged application, real shell entry and gRPC, current Mock P/D/Client"}

    def save(name, value):
        (root / name).write_text(json.dumps(value, indent=2) + "\n")

    def event(name, **fields):
        value = dict(event=name, epoch_ms=time.time_ns() // 1_000_000, **fields)
        timeline.append(value)
        with (root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)

    def start_node(name, jar, component):
        directory = root / name
        directory.mkdir()
        config_file = directory / "config.json"
        config_file.write_text(json.dumps(dict(id=name, component_role=name.split("-")[0], component=component)))
        env = {k: v for k, v in os.environ.items() if not k.startswith("MOCK_")}
        env.update(MOCK_CONFIG_FILE=str(config_file), MOCK_OUTPUT_DIR=str(directory),
                   MOCK_READY_FILE=str(directory / "ready.json"), MOCK_NODE_ID=name,
                   MOCK_TOPOLOGY_FILE=str(root / "topology.json"))
        command = [java, "-XX:ActiveProcessorCount=2", "-jar", str(jar)]
        with (directory / "process.log").open("w") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
        children.append(child)
        save(name + "-command.json", command)
        def ready():
            check(child.poll() is None, f"{name} exited: {(directory / 'process.log').read_text()}")
            return (directory / "ready.json").exists()
        wait_for(ready)
        return json.loads((directory / "endpoints.json").read_text())

    try:
        save("artifacts.json", {str(jar): hashlib.sha256(jar.read_bytes()).hexdigest()
                                for jar in (api_jar, engine_jar, client_jar)})
        engines = []
        for role in ("prefill", "decode"):
            engines.append(start_node(role + "-0", engine_jar, {"engine": "rtp", "config": {
                "seq_size_per_block": 16, "test_block_num": 8192, "reserve_block_ratio": 0,
                "max_seq_len": 4096, "compute": {"forward": {"expression": "10"}}}}))
        # Reserve the HTTP/gRPC pair together while selecting a local port.
        while True:
            with socket.socket() as first, socket.socket() as second:
                first.bind(("127.0.0.1", 0))
                master_port = first.getsockname()[1]
                if master_port > 65533:
                    continue
                try:
                    second.bind(("127.0.0.1", master_port + 2))
                except OSError:
                    continue
                break
        with socket.socket() as management:
            management.bind(("127.0.0.1", 0))
            management_port = management.getsockname()[1]
        native = {"scheduler": {"type": "QUEUE", "decision": {"type": "FIXED_WINDOW", "maxRequests": 4,
                   "maxCollectionWaitMs": 10}}, "dispatcher": {"type": "BATCH"},
                  "router": {"roles": {"decode": {"availability": {"maxEngineRequests": 1}}}},
                  "requestLifecycle": {"request": {"timeoutMs": 60000}},
                  "grpcServer": {"shutdownQuietPeriodMs": args.quiet_period_ms}}
        config = {"FLEXLB_CONFIG": json.dumps(native), "MODEL_SERVICE_CONFIG": json.dumps({
            "service_id": "aigc.text-generation.generation.mock", "role_endpoints": [{"group": "group",
            "prefill_endpoint": {"address": "p-domain", "protocol": "grpc"},
            "decode_endpoint": {"address": "d-domain", "protocol": "grpc"}}]})}
        launch_env = dict(os.environ, **master_environment(config, engines, root))
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
            java,
            "-Xms256m",
            "-Xmx1g",
            "-jar",
            str(api_jar),
            f"--server.port={master_port}",
            f"--management.server.port={management_port}",
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
        event("entry-start", pid=proc.pid)
        master_url = f"http://127.0.0.1:{master_port}"
        def membership():
            check(proc.poll() is None, "packaged application exited during startup")
            state = http(master_url + "/rtp_llm/inflight_status")
            return bool(state["prefill_endpoints"] and state["decode_endpoints"])
        wait_for(membership, 90)
        client = start_node("client-0", client_jar, {"config": {}})
        save("topology.json", [client, {"id": "master-0", "role": "master",
            "http": master_url, "rpc": f"127.0.0.1:{master_port+2}"}])
        sequence = 0
        expected = {}
        def invoke(operation, arguments):
            return http(client["control"] + "/invoke", {"interface": operation, "args": arguments,
                "context": {"run": "foreground", "query_time": "2026-01-01T00:00:00Z", "timeout_ms": 60000,
                            "observation": {}}})["result"]
        def start(lengths):
            nonlocal sequence
            sequence += 1
            source = root / f"input-{sequence}.binpb.zst"
            write_inputs(source, (pb.Request(token_ids=list(range(32)), max_new_tokens=n) for n in lengths))
            event("schedule-plan", sequence=sequence, lengths=lengths)
            invoke("client/start", dict(masters=[f"127.0.0.1:{master_port+2}"], requests_file=str(source), qps=20,
                   concurrency=8, queue_capacity=8, request_timeout_ms=60000, drain_timeout_ms=60000))
            expected[sequence] = lengths
        def state():
            return invoke("worker/state", {"sequence": sequence})
        def collect():
            wait_for(lambda: state()["done"], 65)
            result = invoke("client/stop", {})
            check(not result["errors"], str(result))
            records = sorted(read_requests(root / "client-0/playback" / str(sequence)), key=lambda record: record.request_id)
            check(len(records) == len(expected[sequence]), "missing Client terminal evidence")
            for record, length in zip(records, expected[sequence]):
                check(record.outcome == "success" and record.output_tokens == length, str(record))
                check(len(record.attempts) == 1 and record.attempts[0].response_success, str(record))
            save(f"requests-{sequence}.json", [asdict(record) for record in records])
            event("plan-complete", sequence=sequence)
            return records
        def snapshot():
            value = http(master_url + "/rtp_llm/inflight_status")
            save("master-inflight-at-term.json", value)
            return value
        def decode_running():
            with urlopen(engines[1]["metrics"], timeout=2) as response:
                metrics = response.read().decode()
            return any(float(line.rsplit(" ", 1)[1]) > 0 for line in metrics.splitlines()
                       if line.startswith("rtp_llm_running_stream_size{"))

        start([10])
        collect()
        if args.scenario == "late":
            time.sleep(quiet_seconds + .5)
            check(proc.poll() is None, "idle service exited without TERM")
            check("keep serving until no new requests" not in (root / "logs/flexlb.log").read_text(),
                  "drain started before TERM")
        else:
            start([2000] + ([10] * 3 if args.scenario == "queued" else []))
            wait_for(decode_running, 5)
            if args.scenario == "queued":
                wait_for(lambda: state()["issued"] == 4 and snapshot()["scheduler_inflight"] >= 3, 5)
            snapshot()
        event("send-TERM-to-entry", pid=proc.pid, java_pid=int(pid_file.read_text()))
        proc.send_signal(signal.SIGTERM)
        if args.scenario == "queued":
            time.sleep(6)
            check(proc.poll() is None and state()["finished"] == 0,
                  "entry or pending requests did not survive six seconds")
            records = collect()
        elif args.scenario == "engine":
            check(proc.wait(timeout=15) == 0, "entry did not exit successfully")
            check(decode_running() and state()["finished"] == 0,
                  "long Engine request ended before entry exit")
            records = collect()
        else:
            wait_for(lambda: "keep serving until no new requests" in (root / "logs/flexlb.log").read_text(), 5)
            for _ in range(4):
                time.sleep(quiet_seconds * .3)
                check(proc.poll() is None, "entry exited during late arrivals")
                start([10])
                records = collect()
            time.sleep(quiet_seconds * .6)
            check(proc.poll() is None, "late arrivals did not extend the quiet window")
        check(proc.wait(timeout=40) == 0, "entry did not exit successfully")
        check(not pid_file.exists(), "appctl did not observe Java exit")
        output = (root / "entry.log").read_text()
        check("exited; service stopped." in output, "missing Java exit confirmation")
        check("application stopped; exiting container supervisor" in output, "missing supervisor exit confirmation")
        lines = (root / "logs/flexlb.log").read_text().splitlines()
        def log_time(fragment):
            matches = [line for line in lines if fragment in line]
            check(len(matches) == 1, f"expected one shutdown event: {fragment}")
            return int(datetime.strptime(matches[0][:23], "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000)
        started = log_time("keep serving until no new requests")
        quiet = log_time("Schedule quiet period elapsed")
        drained = log_time("All accepted gRPC requests completed")
        term = next(v["epoch_ms"] for v in timeline if v["event"] == "send-TERM-to-entry")
        last_arrival = max(record.attempts[0].started_at_ms for record in records)
        check(started >= term, "quiet timer started before TERM")
        check(quiet - max(started, last_arrival) >= args.quiet_period_ms - 2,
              "gRPC stopped before a full quiet period")
        if args.scenario == "queued":
            check(drained - quiet > 5000, "accepted RPC drain did not extend beyond quiet expiry")
        summary.update(passed=True, shutdown_timing_ms=dict(term=term, started=started, quiet=quiet, drained=drained))
    except Exception as failure:
        summary["failure"] = repr(failure)
        traceback.print_exc()
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        for child in reversed(children):
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        save("result.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
