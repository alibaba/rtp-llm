"""Monitored stress run using the same Java environment and client runtime as cases."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from artifacts.archive import create_archive
from flexlb_cfg import parse_overrides, render_env
from mode_profiles import master_mode_for_profile, resolve_mode
from monitoring.session import write_monitor_report
from runtime.harness import (API_JAR, FLEXLB_DIR, MOCK_JAR, TOOL_DIR, ClientOps,
                             EnvManager, EnvSpec, ProcessOps, PROBE_BIND_HOST,
                             http_get_json, http_post_json, port_in_use,
                             resolve_java21, wait_for)
from runtime.load_client import LOAD_CLIENT_ENV_VARS
from traffic.catalog import catalog, verify_model
from traffic.traffic_source import materialize

# Producer-side whitelist from the retired stress launcher. Prometheus stores
# all series exposed by these families; no log or API samples become curves.
MASTER_METRIC_WHITELIST = (
    "flexlb_app_cache_", "flexlb_app_flexlb_batcher_queue_size",
    "flexlb_app_flexlb_inflight_max_age_ms", "flexlb_app_flexlb_inflight_ttl",
    "flexlb_app_engine_balancing_master_dispatch_reason_total",
    "flexlb_app_engine_balancing_master_batch_size", "flexlb_auto_tpm_request_count",
    "flexlb_app_engine_balancing_master_all_qps",
    "flexlb_app_flexlb_scheduler_inflight_size",
    "flexlb_app_flexlb_inflight_batch_count",
    "flexlb_app_flexlb_inflight_request_count",
    "flexlb_auto_tpm_decode_reserved_count", "flexlb_auto_tpm_decode_running_count",
)


class ClientExitError(RuntimeError):
    def __init__(self, returncode):
        self.returncode = 128 - returncode if returncode < 0 else returncode
        super().__init__(f"load client exited nonzero: {returncode}")


def shard_concurrency(total: int, workers: int) -> int:
    if total < 1 or workers < 1:
        raise ValueError("concurrency and workers must be positive")
    return (total + workers - 1) // workers


def jfr_option(path: Path, duration: str) -> str:
    return (f"-XX:StartFlightRecording=filename={path},settings=profile,"
            f"duration={duration},disk=true,maxsize=256m,dumponexit=true")


def _number(value: float) -> str:
    """Keep the old snapshot's integer spelling for whole-number CLI values."""
    return str(int(value)) if float(value).is_integer() else str(value)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, default=TOOL_DIR / "run")
    p.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--master-mode", choices=("sb", "sn", "wb", "wn"))
    p.add_argument("--profile", help="explicit full config profile (must agree with mode)")
    p.add_argument("--config-override", default="")
    p.add_argument("--n-prefill", type=int, default=12)
    p.add_argument("--n-decode", type=int, default=40)
    p.add_argument("--mock-base-grpc-port", type=int, default=61000)
    p.add_argument("--master-http-port", type=int, default=7001)
    p.add_argument("--master-management-port", type=int, default=7002)
    p.add_argument("--performance", type=Path,
                   default=Path(os.environ["PERFORMANCE_FILE"]) if os.environ.get("PERFORMANCE_FILE")
                   else TOOL_DIR / "data/performance/dsv4_flash_performance.fast_ab.json")
    traffic = p.add_mutually_exclusive_group()
    traffic.add_argument("--traffic-source-spec", type=Path)
    traffic.add_argument("--traffic-model", choices=tuple(catalog()["models"]),
                         help="versioned trace model registered in data/catalog.json")
    p.add_argument("--traffic-output-tokens", type=int, default=420)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--send-mode", choices=("replay", "uniform"))
    p.add_argument("--send-mode-qps", type=float, default=650)
    p.add_argument("--replay-speed", type=float, default=10)
    p.add_argument("--duration-s", type=float, default=120)
    p.add_argument("--warmup-s", type=float, default=10)
    p.add_argument("--ramp-up-s", type=float, default=30)
    p.add_argument("--client-start-delay-s", type=float, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-concurrency", type=int, default=999999999)
    p.add_argument("--timeout-ms", type=int, default=3600000)
    p.add_argument("--sla-ttft-ms", type=int, default=500)
    p.add_argument("--fetch-output-stream", choices=(0, 1), type=int, default=1)
    p.add_argument("--force-priority", type=int, default=50)
    p.add_argument("--client-option", action="append", default=[], metavar="KEY=VALUE",
                   help="advanced JavaLoadClient option; key must be in config/load_client_env.txt")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--collection-profile", choices=("aggregate", "request", "diagnostic"), default="aggregate")
    p.add_argument("--monitor-interval-s", type=float, default=1)
    p.add_argument("--jfr-duration", default="300s")
    p.add_argument("--master-heap", default="32g")
    p.add_argument("--master-pv-log", action="store_true", help="keep per-request Master pv.log")
    p.add_argument("--mock-heap", default="32g")
    p.add_argument("--client-heap", default="16g")
    p.add_argument("--event-loop-threads", type=int, default=32)
    p.add_argument("--completion-threads", type=int, default=16)
    p.add_argument("--mock-stats-interval-ms", type=int, default=1000)
    p.add_argument("--decode-max-concurrency", type=int, default=128)
    p.add_argument("--prefill-cache-blocks", type=int, default=6000)
    p.add_argument("--decode-cache-blocks", type=int, default=3000)
    p.add_argument("--batch-drain-s", type=float, default=0)
    p.add_argument("--pacing-lag-p99-limit-ms", type=float, default=100)
    p.add_argument("--archive", type=Path)
    p.add_argument("--dry-run", action="store_true", help="validate and print the resolved plan without starting services")
    a = p.parse_args(argv)
    if os.environ.get("TRACE_FILE"):
        p.error("TRACE_FILE is generated per run; use --traffic-source-spec")
    if os.environ.get("LOAD_CLIENT_IMPL") == "python" or os.environ.get("MOCK_ENGINE_IMPL") == "python":
        p.error("Python mock/client implementations were removed")
    for name in ("n_prefill", "n_decode", "workers", "duration_s", "replay_speed", "send_mode_qps"):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if a.limit < 0 or a.limit > 2147483647:
        p.error("--limit must be a nonnegative Java int")
    if a.traffic_source_spec:
        spec = json.loads(a.traffic_source_spec.read_text())
        kind = spec["kind"]
    else:
        kind = "trace"
    a.send_mode = a.send_mode or ("uniform" if kind == "synthetic" else "replay")
    if kind == "synthetic" and a.send_mode == "replay":
        p.error("synthetic traffic has ordinal timestamps; select uniform pacing and --send-mode-qps")
    if a.traffic_source_spec is None and kind != "trace":
        p.error("invalid traffic source")
    profile_given = a.profile is not None
    if a.master_mode is None:
        a.master_mode = master_mode_for_profile(a.profile) if a.profile else "wb"
    plan = resolve_mode("stress", a.master_mode)
    if a.profile and a.profile != plan["master_profile"]:
        p.error("--profile disagrees with --master-mode")
    a.profile = a.profile or plan["master_profile"]
    plan["master_profile"] = a.profile
    plan["profile_source"] = "explicit" if profile_given else "runtime_default"
    a.mode_plan = plan
    advanced = {}
    protected = {"TRACE_FILE", "OUTPUT_DIR", "NUM_SHARDS", "SHARD_INDEX", "START_AT_EPOCH_MS",
                 "COLLECTION_PROFILE", "CLIENT_MONITORING", "SKIP_SERVER_LATENCY", "ENABLE_FALLBACK",
                 "TARGET_ADDR", "DURATION_S", "MAX_CONCURRENCY", "REPLAY_SPEED", "LOAD_CLIENT_WORKERS",
                 "LIMIT", "TIMEOUT_MS", "SLA_TTFT_MS", "FETCH_OUTPUT_STREAM", "FORCE_PRIORITY",
                 "LOOP", "SEND_MODE", "SEND_MODE_QPS", "RAMP_UP_SECONDS"}
    for item in a.client_option:
        key, sep, value = item.partition("=")
        if not sep or key not in LOAD_CLIENT_ENV_VARS or key in protected:
            p.error(f"unsupported --client-option {item!r}")
        advanced[key] = value
    a.advanced_client_options = advanced
    a.run_dir = (a.run_dir or a.run_root / a.run_id).resolve()
    return a


def _traffic(a, output: Path):
    if a.traffic_source_spec:
        spec = json.loads(a.traffic_source_spec.read_text())
        base = a.traffic_source_spec.resolve().parent
    else:
        selected = a.traffic_model or catalog()["default_trace"]
        entry, paths, manifest, _ = verify_model(selected)
        model = paths["model"]
        spec = dict(kind="trace", model=entry["codec"], version=str(entry["codec_version"]), parameters=dict(
            path=model.name, sha256=manifest["sha256"], count=manifest["count"],
            output_tokens=a.traffic_output_tokens, priority=50))
        base = model.parent
    materialize(output, spec, "stress-" + a.run_id, base,
                max_requests=a.limit if a.limit else None)


def _preflight(a):
    if not shutil.which(os.environ.get("PROMETHEUS_BIN", "prometheus")):
        raise RuntimeError("Prometheus required (set PROMETHEUS_BIN)")
    resolve_java21()
    for module, artifact in (("flexlb-mock-engine", MOCK_JAR), ("flexlb-api", API_JAR)):
        if not artifact.is_file():
            print(f"building {module} because {artifact} is missing", flush=True)
            subprocess.run([str(FLEXLB_DIR / "mvnw"), "-Popensource,!internal", "-pl", module,
                            "-am", "package", "-DskipTests"], cwd=FLEXLB_DIR, check=True)
        if not artifact.is_file():
            raise RuntimeError(f"build did not produce {artifact}")
    ports = required_ports(a)
    if len(ports) != len(set(ports)):
        raise RuntimeError("mock and master port windows overlap")
    busy = [p for p in ports if port_in_use(p, PROBE_BIND_HOST)]
    if busy:
        raise RuntimeError(f"required ports busy: {busy}")
    if shutil.which("pgrep"):
        matches = subprocess.run(["pgrep", "-af", "flexlb-api-[^ ]*\\.jar|flexlb-mock-engine-[^ ]*\\.jar"],
                                 capture_output=True, text=True).stdout.strip()
        if matches:
            raise RuntimeError("Concurrent FlexLB processes detected:\n" + matches)


def required_ports(a) -> list[int]:
    """Reserve mock control + contiguous P/D window and all Master ports."""
    ports = [a.mock_base_grpc_port - 1,
             *range(a.mock_base_grpc_port, a.mock_base_grpc_port + a.n_prefill + a.n_decode),
             a.master_http_port, a.master_management_port, a.master_http_port + 2]
    if any(not 1 <= port <= 65535 for port in ports):
        raise ValueError("port window exceeds TCP range")
    return ports


def _config(a):
    overrides = parse_overrides(a.config_override) if a.config_override.strip() else None
    doc = json.loads(render_env(a.profile, overrides))
    doc.setdefault("grpcServer", {}).update(executorCoreSize=128, executorMaxSize=128)
    return json.dumps(doc, separators=(",", ":"))


def _env_spec(a):
    whitelist = ",".join(MASTER_METRIC_WHITELIST)
    return EnvSpec(
        label="stress", run_dir=a.run_dir, runtime_mode="stress", diagnostic_events=a.collection_profile == "diagnostic",
        n_prefill=a.n_prefill, n_decode=a.n_decode, mock_heap=a.mock_heap,
        perf=json.loads(a.performance.read_text()), master_profile=a.profile,
        raw_config=_config(a), master_jvm_heap=a.master_heap,
        master_env={"HIPPO_ROLE": "test", "FLEXLB_MONITOR_ENABLED": "true",
                    "FLEXLB_MONITOR_METRIC_WHITELIST": whitelist,
                    "FLEXLB_GRPC_EXECUTOR_CORE_SIZE": "128", "FLEXLB_GRPC_EXECUTOR_MAX_SIZE": "128"},
        master_jvm_args=[jfr_option(a.run_dir / "flexlb_profile.jfr", a.jfr_duration),
                         f"-Xms{a.master_heap}", f"-Xmx{a.master_heap}",
                         "-Dreactor.schedulers.defaultBoundedElasticSize=64"],
        master_log_name="flexlb.log",
        master_pv_log=a.master_pv_log,
        master_extra_args=[f"--flexlb.log.path={a.run_dir / 'flexlb_logs'}",
                           f"--flexlb.monitor.metric-whitelist={whitelist}"],
        spring_profile="default", discovery="file",
        mock_extra_args=["--stats-interval-ms", str(a.mock_stats_interval_ms),
                         "--decode-max-concurrency", str(a.decode_max_concurrency)],
        event_loop_threads=a.event_loop_threads, completion_threads=a.completion_threads,
        mock_auto_fetch=a.fetch_output_stream == 0,
        prefill_cache_blocks=a.prefill_cache_blocks, decode_cache_blocks=a.decode_cache_blocks,
        master_stable_window_s=3,
    )


def _monitor_start(a, env):
    target = [f"mock=http://127.0.0.1:{env.mock_http_port}/metrics?per_engine=true",
              f"master-single=http://127.0.0.1:{env.master_management_port}/prometheus"]
    argv = [sys.executable, "-m", "monitoring.session", "serve", "--run-dir", str(a.run_dir),
            "--interval", str(a.monitor_interval_s), "--clients", str(a.workers)]
    for t in target:
        argv += ["--target", t]
    menv = dict(os.environ)
    menv["PYTHONPATH"] = os.pathsep.join((str(TOOL_DIR / "src"), str(TOOL_DIR), menv.get("PYTHONPATH", "")))
    proc = ProcessOps.start(argv, menv, a.run_dir / "monitor.log")
    if not wait_for(lambda: (a.run_dir / "monitor-ready").exists() or not proc.alive(), 30, .1):
        proc.terminate()
        raise RuntimeError("monitor startup timed out")
    if not proc.alive():
        raise RuntimeError("monitor failed: " + proc.tail_log())
    return proc


def _monitor_stop(a, proc):
    if proc is None:
        return
    (a.run_dir / "monitor-stop").touch()
    if not proc.wait(90):
        proc.terminate()
    if proc.proc.returncode != 0 or not (a.run_dir / "monitor-complete").exists():
        raise RuntimeError("Prometheus collection/archive failed: " + proc.tail_log())


def _client_base(a, traffic: Path, epoch_ms: int) -> dict[str, str]:
    values = dict(TRACE_FILE=str(traffic), TARGET_ADDR=f"127.0.0.1:{a.master_http_port}",
                  GRPC_TARGET="", DURATION_S=_number(a.duration_s), MAX_CONCURRENCY=str(a.max_concurrency),
                  REPLAY_SPEED=_number(a.replay_speed), LOAD_CLIENT_WORKERS=str(a.workers),
                  LIMIT=str(a.limit), TIMEOUT_MS=str(a.timeout_ms), SLA_TTFT_MS=str(a.sla_ttft_ms),
                  FETCH_OUTPUT_STREAM=str(a.fetch_output_stream), FORCE_PRIORITY=str(a.force_priority),
                  LOOP="1" if a.loop else "0", SEND_MODE=a.send_mode,
                  SEND_MODE_QPS=_number(a.send_mode_qps), RAMP_UP_SECONDS=_number(a.ramp_up_s),
                  START_AT_EPOCH_MS=str(epoch_ms), COLLECTION_PROFILE=a.collection_profile,
                  CLIENT_MONITORING="true", SKIP_SERVER_LATENCY="false" if a.collection_profile == "diagnostic" and a.workers == 1 else "true",
                  ENABLE_FALLBACK="0", CLIENT_PACING_LAG_P99_LIMIT_MS=_number(a.pacing_lag_p99_limit_ms),
                  GRADIENT="0", GRADIENT_START_SPEED="10", GRADIENT_MAX_SPEED="1000",
                  MAX_INPUT_LEN="0", MAX_OUTPUT_LEN="0", PUSHGATEWAY_URL="",
                  ENDPOINTS_FILE="", N_CHANNELS="", EVENT_LOOP_THREADS="",
                  RESPONSE_TIMEOUT="", MODEL="", API_KEY="", DRY_RUN="0")
    values.update(a.advanced_client_options)
    return values


def _run_clients(a, manager, traffic):
    epoch_ms = int((time.time() + a.client_start_delay_s) * 1000)
    base = _client_base(a, traffic, epoch_ms)
    first_out = a.run_dir / "load_client" / ("shard_0" if a.workers > 1 else "")
    a.run_dir.joinpath("client_env.json").write_text(json.dumps({**base, "PRIORITY": "",
        "MAX_CONCURRENCY": str(shard_concurrency(a.max_concurrency, a.workers)),
        "LOAD_CLIENT_WORKERS": str(a.workers), "NUM_SHARDS": str(a.workers),
        "SHARD_INDEX": "0", "OUTPUT_DIR": str(first_out)}, indent=2, sort_keys=True) + "\n")
    client = ClientOps(manager, jvm_xms=a.client_heap, jvm_xmx=a.client_heap)
    count = a.workers
    processes = []
    for shard in range(count):
        out = a.run_dir / "load_client" / (f"shard_{shard}" if count > 1 else "")
        log = a.run_dir / (f"client_shard_{shard}.stdout" if count > 1 else "client.stdout")
        vals = {**base, "MAX_CONCURRENCY": str(shard_concurrency(a.max_concurrency, count)),
                "NUM_SHARDS": str(count), "SHARD_INDEX": str(shard)}
        proc, _ = client.run_async(vals, out, log)
        processes.append(proc)
    failures = []
    for proc in processes:
        rc = proc.proc.wait()
        if rc:
            failures.append(rc)
    if failures:
        raise ClientExitError(failures[-1])


def _check_services(a, env):
    if not env.mock.alive() or not env.master.alive():
        raise RuntimeError("mock or master exited during load")
    status, info = http_post_json(env.master_http("/rtp_llm/master/info"), {})
    if status != 200 or not info or not info.get("ready"):
        raise RuntimeError("master endpoints not ready")
    summary = info.get("worker_summary") or {}
    for role, expected in (("PREFILL", a.n_prefill), ("DECODE", a.n_decode)):
        entry = summary.get(role) or {}
        try:
            discovered = int(entry.get("discovered", -1))
            alive = int(entry.get("alive", -1))
        except (ValueError, TypeError):
            raise RuntimeError(f"{role} endpoint counts malformed: {entry}")
        if discovered < expected or alive < expected:
            raise RuntimeError(f"{role} endpoints not ready: {entry}")
    log = a.run_dir / "mock_engine.log"
    if log.is_file():
        with log.open(errors="replace") as stream:
            if any("OutOfMemoryError" in line for line in stream):
                raise RuntimeError("mock engine encountered OutOfMemoryError")


def run(a):
    if a.dry_run:
        print(json.dumps({"run_dir": str(a.run_dir), "mode": a.mode_plan,
            "send_mode": a.send_mode, "workers": a.workers,
            "shard_max_concurrency": shard_concurrency(a.max_concurrency, a.workers)}, indent=2))
        return 0
    _preflight(a)
    if (a.run_dir / "monitor-ready").exists() or (a.run_dir / "aggregate.json").exists():
        raise RuntimeError(f"run directory already contains evidence: {a.run_dir}")
    a.run_dir.mkdir(parents=True, exist_ok=True)
    (a.run_dir / "flexlb_logs").mkdir(exist_ok=True)
    (a.run_dir / "mode_plan.json").write_text(json.dumps(a.mode_plan, indent=2) + "\n")
    traffic = a.run_dir / "traffic-plan.jsonl"
    _traffic(a, traffic)
    os.environ["FLEXLB_FT_MOCK_BASE_GRPC_PORT"] = str(a.mock_base_grpc_port)
    os.environ["FLEXLB_FT_MASTER_HTTP_PORT"] = str(a.master_http_port)
    os.environ["FLEXLB_FT_MASTER_MANAGEMENT_PORT"] = str(a.master_management_port)
    manager = EnvManager(a.run_dir, verbose=True)
    monitor = None
    status = 1
    try:
        env = manager.ensure(_env_spec(a))
        if a.warmup_s:
            time.sleep(a.warmup_s)
        _check_services(a, env)
        if a.collection_profile == "diagnostic":
            _, info = http_post_json(env.master_http("/rtp_llm/master/info"), {})
            (a.run_dir / "master_info_before.json").write_text(json.dumps(info))
        monitor = _monitor_start(a, env)
        if a.collection_profile == "diagnostic" and a.workers > 1:
            http_post_json(env.master_http("/rtp_llm/server_latency/reset"), {})
        try:
            _run_clients(a, manager, traffic)
        finally:
            if a.collection_profile == "diagnostic":
                latency = http_get_json(env.master_http("/rtp_llm/server_latency"))
                if latency is not None:
                    (a.run_dir / "load_client").mkdir(exist_ok=True)
                    (a.run_dir / "load_client/server_latency.json").write_text(json.dumps(latency))
                else:
                    print("WARNING: terminal server_latency fetch failed", file=sys.stderr)
        _monitor_stop(a, monitor)
        monitor = None
        if a.batch_drain_s:
            time.sleep(a.batch_drain_s)
        _check_services(a, env)
        if a.collection_profile == "diagnostic":
            _, info = http_post_json(env.master_http("/rtp_llm/master/info"), {})
            (a.run_dir / "master_info_after.json").write_text(json.dumps(info))
        if not write_monitor_report(a.run_dir):
            raise RuntimeError("monitor report could not be written")
        aggregate = a.run_dir / "aggregate.json"
        if aggregate.exists():
            try:
                valid = json.loads(aggregate.read_text())["summary"]["test_valid"]
            except (KeyError, ValueError, TypeError):
                print("WARNING: aggregate verdict cannot be parsed", file=sys.stderr)
            else:
                if valid is False:
                    raise RuntimeError("INVALID PERFORMANCE RUN: summary.test_valid=false")
        else:
            print("WARNING: aggregate.json missing", file=sys.stderr)
        status = 0
        print(f"aggregate={aggregate}\nmonitoring={a.run_dir / 'telemetry/0'}")
        return 0
    finally:
        if monitor is not None:
            try:
                _monitor_stop(a, monitor)
            except Exception as exc:
                print(f"WARNING: failed to stop monitor: {exc}", file=sys.stderr)
        manager.teardown()
        if a.archive and a.run_dir.is_dir():
            try:
                create_archive(a.archive, {"run": a.run_dir}, kind="stress",
                               status="complete" if status == 0 and (a.run_dir / "aggregate.json").is_file() else "incomplete")
            except Exception as exc:
                print(f"WARNING: archive failed: {exc}", file=sys.stderr)


def main(argv=None):
    try:
        return run(parse_args(argv))
    except ClientExitError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.returncode
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
