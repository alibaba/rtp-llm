"""Supervise one master and a local mock cluster as separate JVMs in a test Pod."""

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml
from deployment import resolve
from master_compat import legacy_discovery, mock_formula_config

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "online_eval"))
sys.path.insert(0, str(ROOT.parent / "online_eval" / "src"))
from adapters.mock_launch import MockLaunchSpec
from flexlb_cfg import render_env, render_process_config
from mode_profiles import load_mode_tables, resolve_address_plan, resolve_mode


def normalize_heap(value, name):
    """Accept JVM size strings or positive integer byte counts."""
    if type(value) is int and value > 0:
        return str(value)
    if isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*[kKmMgG]?", value):
        return value
    raise ValueError(f"{name} must be a positive byte count or JVM heap size")


def master_discovery_environment(endpoints, legacy):
    """Embedded masters always consume the changing file, never a static snapshot."""
    return legacy_discovery(endpoints, file_discovery=True) if legacy else endpoints["env"]


def vit_configuration(cfg, env, mock_port, legacy):
    enabled = env.get("MOCK_VIT_ENABLED", "0")
    if enabled not in {"0", "1"}:
        raise ValueError("MOCK_VIT_ENABLED must be 0 or 1")
    if enabled == "0":
        return None
    if legacy:
        raise ValueError("modern Mock ViT requires the modern Master protocol")
    values = {"port": int(env.get("MOCK_VIT_PORT", mock_port + cfg["prefill"] + cfg["decode"]))}
    if not mock_port + cfg["prefill"] + cfg["decode"] <= values["port"] <= 65535:
        raise ValueError("Mock ViT port overlaps P/D or exceeds 65535")
    return values


def publish_vit_discovery(source, destination, vit_address):
    # P/D owns the source file and can rewrite it during dynamic scale changes.
    # Publish a separate atomic combined view rather than racing its writer.
    hosts = json.loads(source.read_text())
    hosts["mock.vit.hosts.address"] = [vit_address]
    payload = json.dumps(hosts, sort_keys=True)
    if not destination.exists() or destination.read_text() != payload:
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(payload)
        temporary.replace(destination)


def run():
    if os.environ.get("RTP_LLM_MOCK_BUNDLE") != "1":
        raise ValueError("test-only bundle requires RTP_LLM_MOCK_BUNDLE=1")
    plan = resolve(os.environ, "bundle")
    cfg_path = Path(os.environ.get("MOCK_BUNDLE_CONFIG_PATH", ROOT / "bundle.yaml"))
    cfg = yaml.safe_load(cfg_path.read_text())
    overrides = yaml.safe_load(os.environ.get("MOCK_BUNDLE_OVERRIDES_YAML", "{}"))
    numeric_overrides = {"prefill", "decode"}
    if not isinstance(overrides, dict) or set(overrides) - numeric_overrides - {"mock_heap", "master_heap"}:
        raise ValueError("unsupported bundle override")
    for role in numeric_overrides:
        if role in overrides and (type(overrides[role]) is not int or overrides[role] <= 0):
            raise ValueError(f"{role} must be a positive integer")
    for role in ("prefill", "decode"):
        key = "MOCK_" + role.upper() + "_COUNT"
        if key in os.environ:
            value = int(os.environ[key])
            if value <= 0:
                raise ValueError(f"{key} must be positive for embedded deployment")
            if role in overrides and overrides[role] != value:
                raise ValueError(f"{key} conflicts with MOCK_BUNDLE_OVERRIDES_YAML")
            overrides[role] = value
    cfg.update(overrides)
    for name in ("mock_heap", "master_heap"):
        cfg[name] = normalize_heap(cfg[name], name)
    mode = resolve_mode("whale_embedded", cfg["master_mode"]) if "master_mode" in cfg else None
    profile = mode["master_profile"] if mode else cfg["profile"]
    observation = mode["observation"] if mode else load_mode_tables()["runtime_modes"]["whale_embedded"]["observation"]
    cfg.setdefault("kmonitor", observation["kmonitor"])
    if "profile" in cfg and cfg["profile"] != profile:
        raise ValueError("master_mode and profile disagree")
    fetch_output_stream = os.environ.get("FETCH_OUTPUT_STREAM", "0")
    if fetch_output_stream not in {"0", "1"}:
        raise ValueError("FETCH_OUTPUT_STREAM must be 0 or 1")
    event_log_enabled = os.environ.get("MOCK_EVENT_LOG_ENABLED", "1" if observation["jsonl"] else "0")
    if event_log_enabled not in {"0", "1"}:
        raise ValueError("MOCK_EVENT_LOG_ENABLED must be 0 or 1")
    http_port = int(os.environ.get("START_PORT", "7001"))
    mock_port = http_port + cfg["mock_port_offset"]
    runtime = Path(os.environ.get("MOCK_BUNDLE_RUN_DIR", "/home/admin/ai-whale/mock"))
    runtime.mkdir(parents=True, exist_ok=False)
    plan.record(runtime, prefill=cfg["prefill"], decode=cfg["decode"])
    jars = Path(os.environ.get("MOCK_BUNDLE_JAR_DIR", ROOT / "jars"))
    raw = os.environ.get("FLEXLB_CONFIG") or render_env(profile)
    legacy = os.environ.get("MOCK_BUNDLE_LEGACY_MASTER") == "1"
    vit_cfg = vit_configuration(cfg, os.environ, mock_port, legacy)
    mock_raw = mock_formula_config(raw, legacy, os.environ.get("MOCK_PREFILL_EXECUTION_FORMULA"))
    (runtime / "master-source-config.json").write_text(raw)
    dispatcher = json.loads(raw).get("dispatcher", {}).get("type", "BATCH")
    (runtime / "master-config.json").write_text(render_process_config(profile, jvm_heap=cfg["master_heap"], raw_config=mock_raw))
    performance_path = (cfg_path.parent / cfg["performance"]).resolve()
    from runtime_configs import calibration_input, engine_input

    engine = engine_input(os.environ, cfg_path.parent / cfg["engine_config"])
    (runtime / "engine-config.json").write_text(json.dumps(engine))
    eos_json = os.environ.get("MOCK_EOS_CONFIG_JSON")
    performance = calibration_input(os.environ, performance_path)
    if eos_json is not None:
        eos = json.loads(eos_json)
        if not isinstance(eos, dict):
            raise ValueError("MOCK_EOS_CONFIG_JSON must be a JSON object")
        performance.setdefault("decode", {})["eos"] = eos
    performance_path = runtime / "performance.json"
    performance_path.write_text(json.dumps(performance))
    children, logs = [], []
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def start(name, command, env):
        log = (runtime / (name + ".log")).open("w")
        logs.append(log)
        p = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append(p)
        return p

    def ready(port, process, host="127.0.0.1"):
        deadline = time.monotonic() + cfg["startup_timeout_s"]
        while time.monotonic() < deadline and not stopping:
            if process.poll() is not None:
                raise RuntimeError("JVM exited before ready")
            try:
                with socket.create_connection((host, port), timeout=1):
                    return
            except OSError:
                time.sleep(0.2)
        raise TimeoutError("startup deadline exceeded")

    try:
        env = os.environ.copy()
        java_home = env.get("JAVA_HOME")
        java = shutil.which("java", path=env.get("PATH"))
        if java is None:
            candidates = [Path(java_home) / "bin" / "java"] if java_home else []
            candidates.append(Path("/opt/taobao/java/bin/java"))
            java = next((str(path) for path in candidates if path.is_file()), "java")
        # FetchResponse originates in a different Pod, so its advertised engine
        # address must be reachable from the frontend. Ports identify engines.
        pod_ip = os.environ.get("POD_IP") or socket.gethostbyname(socket.gethostname())
        if pod_ip.startswith("127.") or pod_ip == "0.0.0.0":
            raise ValueError("bundle requires an advertised Pod IP")
        address_plan = resolve_address_plan(
            "whale_embedded", external_engine_clients=(dispatcher == "NON_BATCH" or fetch_output_stream == "1"), pod_ip=pod_ip
        )
        # With output fetching enabled the frontend owns continuation and completion.
        mock = start(
            "mock",
            MockLaunchSpec(
                java=java,
                jar=jars / "mock.jar",
                jvm_options=("-Xmx" + cfg["mock_heap"],),
                whale=True,
                whale_bundle=True,
                kmonitor=cfg["kmonitor"],
                n_prefill=cfg["prefill"],
                n_decode=cfg["decode"],
                base_grpc_port=mock_port,
                event_loop_threads=cfg["event_loop_threads"],
                completion_threads=cfg["completion_threads"],
                host=pod_ip,
                unique_engine_ips=address_plan["unique_engine_ips"],
                auto_fetch=fetch_output_stream == "0",
                endpoint_file=runtime / "endpoints.json",
                discovery_file=runtime / "discovery.json",
                calibration=performance_path,
                engine_config=runtime / "engine-config.json",
                events_file=runtime / "engine-events.jsonl" if event_log_enabled == "1" else None,
            ).argv(),
            env,
        )
        ready(mock_port - 1, mock, pod_ip)
        endpoints = json.loads((runtime / "endpoints.json").read_text())
        env.update(master_discovery_environment(endpoints, legacy))

        if vit_cfg is not None:
            vit = start(
                "vit",
                [
                    java,
                    "-Xms64m",
                    "-Xmx512m",
                    "-Djava.awt.headless=true",
                    "-cp",
                    str(jars / "mock.jar"),
                    "org.flexlb.mockengine.MockVitServer",
                    "--engine-config",
                    str(runtime / "engine-config.json"),
                    "--calibration",
                    str(performance_path),
                    *[part for key, value in vit_cfg.items() for part in ("--" + key.replace("_", "-"), str(value))],
                    *(["--model-path", os.environ["MOCK_VIT_MODEL_PATH"]] if os.environ.get("MOCK_VIT_MODEL_PATH") else []),
                ],
                env,
            )
            ready(vit_cfg["port"], vit, pod_ip)
            service = json.loads(env["MODEL_SERVICE_CONFIG"])
            pd_discovery = Path(service["discovery_file"])
            combined_discovery = runtime / "vit-discovery.json"
            vit_address = f"{pod_ip}:{vit_cfg['port'] - 1}"
            publish_vit_discovery(pd_discovery, combined_discovery, vit_address)
            service["discovery_file"] = str(combined_discovery)
            for group in service["role_endpoints"]:
                group["vit_endpoint"] = {"address": "mock.vit.hosts.address", "protocol": "http", "path": "/"}
            env["MODEL_SERVICE_CONFIG"] = json.dumps(service)
            (runtime / "model-service.json").write_text(env["MODEL_SERVICE_CONFIG"])
        env.update(FLEXLB_CONFIG=raw, SERVER_PORT=str(http_port), MANAGEMENT_SERVER_PORT=str(http_port + 1))
        master = start(
            "master",
            [
                java,
                "-Xmx" + cfg["master_heap"],
                "-Dreactor.schedulers.defaultBoundedElasticSize=64",
                "-jar",
                str(jars / ("legacy-master.jar" if legacy else "master.jar")),
                "--server.port=" + str(http_port),
                "--management.server.port=" + str(http_port + 1),
                "--flexlb.log.path=" + str(runtime / "master-logs"),
            ],
            env,
        )
        ready(http_port + 2, master)
        # This bundle replaces appctl, so it also owns the normal local online hook.
        deadline = time.monotonic() + cfg["startup_timeout_s"]
        while not stopping:
            if master.poll() is not None:
                raise RuntimeError("master exited before application readiness")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{http_port}/hook/process_ok", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("master application readiness deadline exceeded") from None
                time.sleep(0.2)
        if stopping:
            return
        with urllib.request.urlopen(f"http://127.0.0.1:{http_port}/hook/after_start", timeout=cfg["startup_timeout_s"]) as response:
            response.read()
        with urllib.request.urlopen(f"http://127.0.0.1:{http_port}/health", timeout=2) as response:
            response.read()
        (runtime / "identity.json").write_text(
            json.dumps(
                {
                    "mode": ("WHALE_BUNDLE_FETCH" if fetch_output_stream == "1" else "WHALE_BUNDLE_SCHEDULE_ONLY"),
                    "deployment": plan.mode,
                    "master_discovery": plan.master_discovery,
                    "fetch_output_stream": fetch_output_stream == "1",
                    "master_pid": master.pid,
                    "mock_pid": mock.pid,
                    "vit": vit_cfg,
                    "master_http_port": http_port,
                    "mock_http_port": mock_port - 1,
                    "configuration": cfg,
                    "source": os.environ.get("BUILD_SOURCE_SHA", "unknown"),
                    "legacy_master": legacy,
                    "dispatcher": dispatcher,
                },
                indent=2,
            )
        )
        while not stopping:
            if vit_cfg is not None:
                publish_vit_discovery(pd_discovery, combined_discovery, vit_address)
            if any(p.poll() is not None for p in children):
                raise RuntimeError("bundle child exited; terminate its peer")
            time.sleep(0.2)
    finally:
        for p in reversed(children):
            if p.poll() is None:
                p.terminate()
        deadline = time.monotonic() + cfg["shutdown_timeout_s"]
        for p in reversed(children):
            try:
                p.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    run()
