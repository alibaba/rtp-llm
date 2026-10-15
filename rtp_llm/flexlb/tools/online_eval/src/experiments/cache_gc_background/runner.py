#!/usr/bin/env python3
"""稳定拓扑下校准真实请求与缓存同步背景；输出保留在独占运行目录。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
SUPPORT = ROOT / "config/diagnostics/cache_gc/support"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/diagnostics/cache_gc/before.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master-jar", type=Path, required=True)
    parser.add_argument("--master-source", required=True)
    parser.add_argument("--mock-jar", type=Path, required=True)
    parser.add_argument("--trace-file", type=Path)
    parser.add_argument("--container-readonly-root", type=Path, required=True,
                        help="Container-visible root containing Java, repository and dependency inputs")
    parser.add_argument("--docker-image", required=True, help="Linux image providing the required runtime libraries")
    parser.add_argument("--support-dir", type=Path, default=SUPPORT,
                        help="Read-only diagnostic and Mock source patches to record with the run")
    return parser.parse_args()


def fetch(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def controls(base, engine, operation, **values):
    request = urllib.request.Request(base + "/cache_fixture",
        data=json.dumps(dict(engine=engine, operation=operation, **values)).encode(),
        headers={"Content-Type": "application/json"})
    result = json.loads(urllib.request.urlopen(request, timeout=15).read())
    if result.get("status") != "ok" or result.get("engine") != engine:
        raise ValueError("cache fixture identity not acknowledged")
    return result["fixture"]


def parse_metrics(text):
    result = []
    pattern = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([-+a-zA-Z0-9.eE]+)')
    for line in text.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        name, labels, raw = match.groups()
        if not name.startswith(("flexlb_app_", "jvm_", "process_cpu_")):
            continue
        value = float(raw)
        if math.isfinite(value):
            result.append(dict(name=name, labels=dict(re.findall(r'(\w+)="([^"]*)"', labels or "")), value=value))
    return result


def observed(meters, fragment, *, labels=None, reducer=sum):
    values = [m["value"] for m in meters if fragment in m["name"]
              and all(m["labels"].get(k) == v for k, v in (labels or {}).items())]
    return reducer(values) if values else None


def container_launch(original, output, resources, registry, cache_config, readonly_root, docker_image):
    from runtime.process import ManagedProcess

    class DockerProcess(ManagedProcess):
        @property
        def pid(self):
            return self.java_pid

        def terminate(self, timeout_s=10):
            subprocess.run(["docker", "stop", "-t", str(min(10, int(timeout_s))), self.container],
                           capture_output=True, timeout=20, check=True)
            self.wait(20)

        def kill9(self):
            self.terminate()

    def start(argv, env, log_file, cwd=None, **kwargs):
        if "org.flexlb.mockengine.JavaLoadClient" in argv:
            kind = "client"
        elif "-jar" in argv:
            kind = "mock" if "mock" in argv[argv.index("-jar") + 1] else "master"
        else:
            return original(argv, env, log_file, cwd, **kwargs)
        if kind in {"master", "mock"}:
            selected = Path(argv[argv.index("-jar") + 1]).resolve()
            wanted = Path(os.environ["FLEXLB_FT_MASTER_JAR"]).resolve() if kind == "master" else __import__("runtime.paths", fromlist=["MOCK_JAR"]).MOCK_JAR.resolve()
            if selected != wanted:
                raise AssertionError(f"artifact mismatch: {kind}: {selected} != {wanted}")
            (output / (kind + "-launched-artifact.json")).write_text(json.dumps(dict(path=str(selected), sha256=hashlib.sha256(selected.read_bytes()).hexdigest())))
        budget = resources[kind]
        env = dict(env)
        env["JAVA_TOOL_OPTIONS"] = env.get("JAVA_TOOL_OPTIONS", "") + " -XX:ActiveProcessorCount=" + str(budget["cpu"])
        if kind == "mock":
            env["JAVA_TOOL_OPTIONS"] += " -Dflexlb.mock.cache.omit-unchanged=" + str(cache_config["omit_unchanged_keys"]).lower()
            env["JAVA_TOOL_OPTIONS"] += " -Dflexlb.mock.cache.churn-interval-ms=" + str(cache_config["churn_interval_ms"])
        name = "gc-bg-" + output.name + "-" + kind
        command = ["docker", "run", "--name", name, "--cpus", str(budget["cpu"]),
                   "--memory", str(budget["memory_mib"]) + "m",
                   "--memory-swap", str(budget["memory_mib"]) + "m", "--network=host",
                   "-v", str(readonly_root) + ":" + str(readonly_root) + ":ro",
                   "-v", str(output.parent) + ":" + str(output.parent) + ":rw"]
        for key, value in env.items():
            if key.startswith(("FLEXLB", "RTP", "MOCK", "SPRING")) or key in {
                "JAVA_TOOL_OPTIONS", "MODEL_SERVICE_CONFIG", "HIPPO_ROLE", "TMPDIR",
                "TRACE_FILE", "GRPC_TARGET", "FLOW_CONTROL_DIR", "FLOW_RUN_ID", "FLOW_GROUP_ID",
                "FLOW_PHASE_ID", "OUTPUT_DIR", "COLLECTION_PROFILE", "LIVE_CLIENT_EVENTS",
                "CLIENT_MONITORING", "FETCH_OUTPUT_STREAM", "REPLAY_UNIQUE_PREFIX", "DURATION_S",
                "MAX_CONCURRENCY", "TIMEOUT_MS", "RESPONSE_TIMEOUT", "N_CHANNELS",
                "EVENT_LOOP_THREADS", "SEND_MODE", "SEND_MODE_QPS", "RAMP_UP_SECONDS",
                "MAX_LAPS", "LAP_IDENTITY", "REPLAY_SPEED", "PRIORITY", "SKIP_SERVER_LATENCY"}:
                command += ["-e", key + "=" + value]
        command += ["--entrypoint", argv[0],
                    docker_image, *argv[1:]]
        proc = original(command, env, log_file, cwd, **kwargs)
        proc.__class__ = DockerProcess
        proc.container = name
        proc.java_pid = proc.proc.pid
        registry.append(proc)
        for _ in range(80):
            inspection = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
            if inspection.returncode == 0:
                detail = json.loads(inspection.stdout)[0]
                if detail["State"]["Pid"] > 0:
                    actual = detail["HostConfig"]
                    if (actual["NanoCpus"] != budget["cpu"] * 1_000_000_000
                            or actual["Memory"] != budget["memory_mib"] * 1024**2):
                        raise ValueError("actual cgroup budget differs from declaration")
                    proc.java_pid = detail["State"]["Pid"]
                    (output / (kind + "-container.json")).write_text(json.dumps(detail, indent=2))
                    return proc
            time.sleep(.25)
        raise RuntimeError("owned container did not start: " + name)
    return start


def main():
    args = parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt("received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    config = json.loads(args.config.read_text())
    if config.get("schema_version") != 1 or re.fullmatch(r"[0-9a-f]{40}", args.master_source) is None:
        raise ValueError("require config schema 1 and full master source SHA")
    if not args.container_readonly_root.is_dir():
        raise ValueError("container read-only root must be an existing directory")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifacts = output / "artifacts"
    artifacts.mkdir()
    shutil.copy2(__file__, artifacts / "experiment.py")
    shutil.copy2(args.config, artifacts / "configuration.json")
    for name in ("diagnostic-master.patch", "mock-runtime.patch"):
        shutil.copy2(args.support_dir / name, artifacts / name)
    master_jar = artifacts / "master.jar"
    mock_jar = artifacts / "mock-all.jar"
    shutil.copy2(args.master_jar, master_jar)
    shutil.copy2(args.mock_jar, mock_jar)
    os.environ["FLEXLB_FT_MASTER_JAR"] = str(master_jar)
    import runtime.paths
    runtime.paths.API_JAR = master_jar
    runtime.paths.MOCK_JAR = mock_jar
    cache = config["cache"]
    windows = config["windows"]
    traffic = config["traffic"]
    os.environ["FLEXLB_FT_MASTER_JAR"] = str(master_jar)
    os.environ["FLEXLB_FT_MASTER_SOURCE_COMMIT"] = args.master_source
    from adapters.environment import EnvManager
    from adapters.environment_config import EnvSpec
    from adapters.java_client import ClientOps
    from adapters.java_flow import JavaFlowGroup
    from adapters.master_control import master_url
    from adapters.perf_presets import load_preset
    from flexlb_cfg import ConfigOverride, render_env
    from runtime.deadline import Deadline
    from runtime.process import ProcessOps
    from traffic.traffic_source import materialize

    perf, _ = load_preset("deepseek_v4_flash_l20c")
    perf["decode"].pop("eos", None)  # The frozen request plan owns output-length sampling.
    raw = json.loads(render_env("single-nonbatch", ConfigOverride(
        ordering="FIFO", max_inflight_per_prefill_worker=64,
        cache_min_refresh_ms=cache["poll_interval_ms"], cache_max_refresh_ms=cache["poll_interval_ms"],
        status_rpc_ms=5000, status_stale_after_ms=10000)))
    raw["workerRegistry"]["health"]["statusPollIntervalMs"] = config.get("status_poll_interval_ms", 20)
    raw["observability"] = {"cacheHit": {"recentKeyWindow": {
        "writeEnabled": True, "durationMs": 1800000, "maxKeyOccurrences": cache["recent_key_occurrences"]}}}
    raw["grpcServer"] = {"executorCoreSize": 64, "executorMaxSize": 64, "executorQueueSize": 1000}
    duration = windows["traffic_warmup_s"] + windows["measurement_s"] + 20
    trace = output / "traffic.jsonl"
    source = dict(kind="trace", model="prefix_lineage", parameters=dict(
        path=traffic["source"], count=traffic["source_count"], output_tokens=8192,
        output_distribution=dict(kind="geometric", mean_tokens=traffic["output_mean_tokens"], seed=20260922),
        priority=50, max_input_tokens=traffic["max_input_tokens"]))
    if args.trace_file is None:
        materialize(trace, source, output.name, ROOT,
                    max_requests=math.ceil(duration * traffic["qps"] * 1.02))
    else:
        shutil.copy2(args.trace_file, trace)
        source_manifest = args.trace_file.with_suffix(".manifest.json")
        if source_manifest.exists():
            shutil.copy2(source_manifest, trace.with_suffix(".manifest.json"))
    manifest = dict(config=config, source=args.master_source,
                    master_sha256=hashlib.sha256(master_jar.read_bytes()).hexdigest(),
                    mock_sha256=hashlib.sha256(mock_jar.read_bytes()).hexdigest(),
                    request_plan_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                    instrumentation="allocation counter, read-only poll health and terminal fingerprints; existing 114 diff counters",
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    diagnostic_patch_sha256=hashlib.sha256((args.support_dir / "diagnostic-master.patch").read_bytes()).hexdigest(),
                    mock_patch_sha256=hashlib.sha256((args.support_dir / "mock-runtime.patch").read_bytes()).hexdigest(),
                    topology_changes="NONE_INJECTED", calibration="OBSERVATION_ONLY")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    original = ProcessOps.start
    owned = []
    ProcessOps.start = staticmethod(container_launch(original, output, config["resources"], owned, cache,
                                                          args.container_readonly_root.resolve(), args.docker_image))
    manager = EnvManager(output, verbose=True)
    events, frames, errors = [], [], []
    flow = None
    cleanup = []
    deadline = Deadline(time.monotonic() + 1800)

    def event(name):
        events.append(dict(name=name, epoch_s=time.time()))
        (output / "events.json").write_text(json.dumps(events))
        print("EVENT", name, flush=True)

    try:
        jfr = output / "master.jfr"
        spec = EnvSpec(label="gc_background", run_dir=output / "env",
            n_prefill=config["topology"]["prefill"], n_decode=config["topology"]["decode"],
            mock_heap=config["resources"]["mock"]["heap"], perf=perf,
            event_loop_threads=config["mock_runtime"]["event_loop_threads"],
            completion_threads=config["mock_runtime"]["completion_threads"],
            mock_extra_args=["--rpc-queue-capacity", str(config["mock_runtime"]["rpc_queue_capacity"])],
            master_profile="single-nonbatch", raw_config=json.dumps(raw), discovery="discovery_file",
            prefill_cache_blocks=21553, decode_cache_blocks=221484,
            master_jvm_heap=config["resources"]["master"]["heap"],
            master_jvm_args=["-Xms1g", "-Dflexlb.diagnostic.allocation-counter=true", "-XX:MaxDirectMemorySize=8g", "-XX:ActiveProcessorCount=32",
                "-XX:+UseG1GC", "-XX:FlightRecorderOptions=repository=" + str(output / "jfr-repository"),
                "-XX:StartFlightRecording=filename=" + str(jfr) + ",settings=profile,dumponexit=true,maxsize=512m",
                "-Xlog:gc*,safepoint:file=" + str(output / "gc.log") + ":time,uptime,level,tags"],
            master_extra_args=["--flexlb.monitor.per-engine-tags=true",
                "--flexlb.monitor.metric-whitelist=flexlb_app_engine_health_check_visitor_success_qps,flexlb_app_engine_health_check_visitor_rt,flexlb_app_engine_health_check_fail,flexlb_app_engine_health_check_success_period,flexlb_app_grpc_call_duration,flexlb_app_cache_status_check_visitor_success_qps,flexlb_app_engine_health_check_finished_task_list_size,flexlb_app_cache_global_total_count,flexlb_app_cache_recent_key_total_count,flexlb_app_cache_diff_,flexlb_app_engine_health_check_running_task_info_size,flexlb_app_engine_balancing_thread_pool_info,flexlb_app_engine_balancing_event_loop_group_info,flexlb_app_grpc_call_count,flexlb_app_grpc_response_size,flexlb_grpc_", "--flexlb.log.path=" + str(output / "logs")],
            master_pv_log=False, diagnostic_events=False)
        if not config.get("diagnostic_monitoring", True):
            spec.master_jvm_args = [x for x in spec.master_jvm_args if not x.startswith(("-XX:FlightRecorderOptions=", "-XX:StartFlightRecording="))]
            spec.master_extra_args += ["--flexlb.monitor.enabled=false"]
        env = manager.ensure(spec, deadline=deadline)
        resource_groups = {}
        for kind, proc in (("master", env.master), ("mock", env.mock)):
            relative = next(line.split("::", 1)[1] for line in
                            Path(f"/proc/{proc.pid}/cgroup").read_text().splitlines() if line.startswith("0::"))
            resource_groups[kind] = (proc.pid, Path("/sys/fs/cgroup") / relative.lstrip("/"))
        base = "http://127.0.0.1:" + str(env.mock_http_port)
        topology = json.loads(fetch(base + "/snapshot?view=topology"))["engines"]
        engines = sorted([e for e in topology if e["role"] == "prefill"],
                         key=lambda e: int(e["name"].rsplit("-", 1)[1]))
        health_url = master_url("127.0.0.1", env.master_http_port, "inflight")
        metrics_url = "http://127.0.0.1:" + str(env.master_management_port) + "/prometheus"

        def fixture_stats():
            stats = [controls(base, e["name"], "inspect") for e in engines]
            return {key: sum(x[key] for x in stats)
                    for key in ("responses", "full_responses", "unchanged_responses", "entries", "key_payload_bytes", "changed_responses", "added_keys", "removed_keys")}

        def sample(phase):
            row = dict(epoch_s=time.time(), phase=phase)
            for kind, (pid, cgroup) in resource_groups.items():
                cpu = dict((key, int(value)) for key,value in
                           (line.split() for line in (cgroup/"cpu.stat").read_text().splitlines()))
                rss = next(int(line.split()[1])*1024 for line in
                           Path(f"/proc/{pid}/status").read_text().splitlines() if line.startswith("VmRSS:"))
                resource = dict(epoch_s=row["epoch_s"], cpu=cpu, rss_bytes=rss,
                                memory_bytes=int((cgroup/"memory.current").read_text()),
                                cpu_max=(cgroup/"cpu.max").read_text().strip())
                filename = "resources.jsonl" if kind == "master" else "mock-resources.jsonl"
                with (output / filename).open("a") as dest:
                    dest.write(json.dumps(resource) + "\n")
            try:
                health = json.loads(fetch(health_url))
                meters = parse_metrics(fetch(metrics_url).decode()) if config.get("diagnostic_monitoring", True) else []
                ps = health["prefill_endpoints"]
                ds = health["decode_endpoints"]
                row.update(scheduler_inflight=health["scheduler_inflight"],
                    process_allocated_bytes=health.get("process_allocated_bytes"),
                    p_workers=len(ps), d_workers=len(ds),
                    max_status_age_ms=max((x["status_age_ms"] for x in ps + ds), default=-1),
                    stale_workers=sum(x["status_age_ms"] > 10000 for x in ps + ds),
                    indexed_p=sum(x["cache_indexed"] for x in ps),
                    p_running_mean=observed(meters, "engine_health_check_running_task_info_size", labels={"role": "PREFILL"}, reducer=lambda v: sum(v)/len(v)),
                    d_running_mean=observed(meters, "engine_health_check_running_task_info_size", labels={"role": "DECODE"}, reducer=lambda v: sum(v)/len(v)),
                    global_keys=observed(meters, "cache_global_total_count"),
                    request_input_tokens_sample=observed(meters, "cache_recent_key_total_count"),
                    heap_bytes=observed(meters, "jvm_memory_used_bytes", labels={"area": "heap"}),
                    direct_bytes=observed(meters, "jvm_buffer_memory_used_bytes", labels={"id": "direct"}),
                    meters=meters)
                row["reported_dead"] = sum(not x["reported_alive"] for x in ps + ds)
                row["healthy_generations"] = {x["ip_port"]: x["generation"] for x in ps + ds}
                if flow is not None:
                    row["client"] = flow.control_status()
            except Exception as exc:
                row["error"] = repr(exc)
                errors.append(dict(epoch_s=row["epoch_s"], error=repr(exc)))
            frames.append(row)
            with (output / "observations.jsonl").open("a") as dest:
                dest.write(json.dumps(row) + "\n")
            if len(frames) % 15 == 0:
                print("SAMPLE", phase, {k: v for k, v in row.items()
                      if k not in {"meters", "healthy_generations", "client"}}, flush=True)
            return row

        event("cache_seed_started")
        for i, engine in enumerate(engines):
            controls(base, engine["name"], "seed", worker=i, workers=len(engines),
                     unique=cache["private_keys"], shared=cache["shared_keys"])
            if i + 1 < len(engines):
                time.sleep(cache["seed_interval_s"])
        until = time.monotonic() + 300
        converged = 0
        while True:
            row = sample("cache_warmup")
            if row.get("indexed_p") == len(engines) and row.get("p_workers") == len(engines) and row.get("d_workers") == config["topology"]["decode"] and row.get("max_status_age_ms", 1e9) < 5000:
                converged += 1
            else:
                converged = 0
            if converged >= 5:
                break
            if time.monotonic() > until:
                raise TimeoutError("cache baseline did not converge")
            time.sleep(windows["sample_s"])
        event("cache_baseline_ready")
        if cache.get("churn_during_cache_only", False):
            for engine in engines:
                controls(base, engine["name"], "churn", delta=cache["delta_keys"], every=cache["change_every_responses"])
            event("small_diff_started")

        intensity = {}
        for phase, seconds in [("cache_only", windows["cache_only_s"]),
                               ("traffic_warmup", windows["traffic_warmup_s"]),
                               ("measurement", windows["measurement_s"])]:
            if phase == "traffic_warmup":
                if not cache.get("churn_during_cache_only", False):
                    for engine in engines:
                        controls(base, engine["name"], "churn", delta=cache["delta_keys"], every=cache["change_every_responses"])
                    event("small_diff_started")
                flow = JavaFlowGroup(ClientOps(manager, "2g", config["resources"]["client"]["heap"]),
                    output / "flow", run_id=str(output), group_id="business", phase_id="stable",
                    poll_s=.2, collection_profile="aggregate")
                client = dict(playback=dict(duration_s=duration, mode="uniform", qps=traffic["qps"],
                                            ramp_up_seconds=30, max_laps=1, identity="structural-relabel"),
                    limits=dict(max_concurrency=traffic["max_concurrency"],
                                request_timeout_s=traffic["request_timeout_s"], drain_timeout_s=traffic["drain_timeout_s"]),
                    transport=dict(channels=8, event_loop_threads=8))
                flow.start(trace, client, deadline, target="127.0.0.1:" + str(env.master_http_port + 2))
            before = fixture_stats()
            event(phase + "_started")
            started = time.monotonic()
            until = started + seconds
            while time.monotonic() < until:
                deadline.check()
                if (output / "STOP").exists():
                    raise InterruptedError("operator requested calibration stop")
                sample(phase)
                time.sleep(min(windows["sample_s"], max(0, until-time.monotonic())))
            after = fixture_stats()
            elapsed = time.monotonic() - started
            intensity[phase] = dict(duration_s=elapsed,
                **{k: (after[k] - before[k]) / elapsed for k in before})
            (output / "intensity.json").write_text(json.dumps(intensity, indent=2))
        (output / "intensity.json").write_text(json.dumps(intensity, indent=2))
        event("measurement_finished")
        if flow is not None:
            flow.stop_sending(deadline)
            event("sending_stopped")
            flow.drain(deadline)
            event("drain_finished")
            (output / "client-evidence.json").write_text(json.dumps(flow.evidence_snapshot()))
        event("performance_observation_finished")
        event("consistency_started")
        for engine in engines:
            controls(base, engine["name"], "churn", delta=0, every=0)
        expected = {engine["http_addr"]: controls(base, engine["name"], "inspect") for engine in engines}
        until = time.monotonic() + 120
        while True:
            health = json.loads(fetch(health_url))
            actual = {x["ip_port"]: x for x in health["prefill_endpoints"]}
            if all(address in actual and actual[address]["cache_indexed_version"] == item["version"]
                   and actual[address]["cache_key_size"] == item["keys"] for address,item in expected.items()):
                break
            if time.monotonic() > until:
                raise TimeoutError("frozen cache versions did not converge")
            time.sleep(2)
        (output / "cache-expected.json").write_text(json.dumps(expected, indent=2))
        verified = json.loads(fetch(health_url.rsplit("/", 1)[0] + "/cache_fingerprint", timeout=180))
        (output / "cache-verified-raw.json").write_text(json.dumps(verified))
        checks = []
        for worker in verified["prefill_endpoints"]:
            want = expected.get(worker["ip_port"])
            got = worker.get("fingerprint", {})
            matches = want is not None and got.get("indexed_version") == want["version"]
            matches = matches and got.get("indexed_keys") == want["keys"] and got.get("status_keys") == want["keys"]
            matches = matches and got.get("sum") == want["sum"] and got.get("xor") == want["xor"]
            matches = matches and got.get("missing_status") == 0 and got.get("missing_global") == 0
            checks.append(dict(worker=worker["ip_port"], matched=matches, expected=want, actual=got))
        mappings = sum(item["keys"] for item in expected.values())
        global_keys = len(engines) * (cache["private_keys"] + cache["shared_keys"])
        consistent = len(checks) == len(engines) and all(c["matched"] for c in checks)
        consistent = consistent and verified["global_mappings"] == mappings and verified["global_unique_keys"] == global_keys
        receipt = dict(consistent=consistent, workers=checks, global_mappings=verified["global_mappings"],
                       global_unique_keys=verified["global_unique_keys"], expected_mappings=mappings,
                       expected_global_keys=global_keys, scope="terminal frozen fingerprints plus exhaustive status/local/global membership and counts")
        (output / "cache-consistency.json").write_text(json.dumps(receipt, indent=2))
        event("consistency_finished")
        if not consistent:
            raise AssertionError("cache consistency verification failed")
        event("observation_finished")
        (output / "intensity.json").write_text(json.dumps(intensity, indent=2))
    except BaseException as exc:
        errors.append(dict(fatal=repr(exc)))
        raise
    finally:
        if flow is not None and flow.proc is not None and flow.proc.alive():
            try:
                flow.stop_sending(Deadline(time.monotonic() + 10))
                flow.drain(Deadline(time.monotonic() + traffic["drain_timeout_s"] + 15))
            except Exception as exc:
                errors.append(dict(cleanup=str(exc)))
        manager.teardown()
        for proc in owned:
            inspect = subprocess.run(["docker", "inspect", proc.container], capture_output=True, text=True)
            if inspect.returncode == 0:
                detail = json.loads(inspect.stdout)[0]
                cleanup.append(dict(container=proc.container, state=detail["State"]))
                if detail["State"]["Running"]:
                    subprocess.run(["docker", "stop", "-t", "5", proc.container], check=True, timeout=15)
            subprocess.run(["docker", "rm", proc.container], capture_output=True, timeout=10)
        ProcessOps.start = original
        (output / "cleanup.json").write_text(json.dumps(cleanup))
        (output / "errors.json").write_text(json.dumps(errors, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
