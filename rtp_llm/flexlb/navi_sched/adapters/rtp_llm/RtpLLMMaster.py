import json
import logging
import os
import signal
import threading
from dataclasses import replace

from ...master_process import (
    MasterProcessConfig,
    load_master_config,
    make_runtime_config,
)
from ...master import NaviSchedMaster
from .RtpLLMAdapterUtils import make_kmon_options, split_address
from .RtpLLMUnifiedConfig import load_navi_sched_config
from .ServiceDiscovery import (
    DEFAULT_MASTER_BIZ_NAME,
    RtpLLMServiceRoute,
    agent_grpc_port_diff,
    make_discovery_runtime,
    master_grpc_port_diff,
    prefill_biz_name,
)

DEFAULT_MASTER_GRPC_PORT = 7003
DEFAULT_START_PORT = 7001
DEFAULT_MASTER_OPTIONS = {
    "scheduler_name": "cost",
    "scheduler_config": {
        "lambda": 0.0,
        "alpha": 512.0,
        "alpha_decay": 1.0,
        "min_alpha": 0.0,
        "max_loop_count": 10,
        "parallelism": 0,
    },
    "max_wait_schedule_count": 30,
    "max_wait_schedule_time_ms": 30,
    "runtime_config": {"thread_num": 64},
}


def make_rtp_llm_master_options(config, environ):
    unified_config = load_navi_sched_config(environ)
    master_config = unified_config.get("master", {})
    route = RtpLLMServiceRoute.from_env(environ)
    if route.prefill_endpoint is None:
        raise ValueError("MODEL_SERVICE_CONFIG has no prefill endpoint")

    master_biz_name = config.master_biz_name or DEFAULT_MASTER_BIZ_NAME
    start_port = int(
        environ.get("START_PORT", DEFAULT_START_PORT) or DEFAULT_START_PORT
    )
    http_port = int(master_config.get("http_port", config.http_port))
    arpc_port = int(master_config.get("arpc_port", config.arpc_port))
    grpc_port = int(master_config.get("grpc_port", config.grpc_port))
    if http_port == 0:
        http_port = start_port
    if arpc_port == 0:
        arpc_port = start_port + 1
    if grpc_port == 0:
        if "START_PORT" in environ:
            grpc_port = start_port + 2
        elif route.use_local and route.master_endpoint is not None:
            _, advertised_port = split_address(
                route.master_endpoint.address.split(",", 1)[0]
            )
            grpc_port = advertised_port + master_grpc_port_diff(
                route.master_endpoint
            )
        else:
            grpc_port = DEFAULT_MASTER_GRPC_PORT
    log_path = str(unified_config.get("log_path", "")).strip()
    config = replace(
        config,
        master_biz_name=master_biz_name,
        http_port=http_port,
        arpc_port=arpc_port,
        grpc_port=grpc_port,
        config_json=(
            config.config_json
            or ("{}" if not config.config_file else "")
        ),
        log_file_name=(
            os.path.join(log_path, "navi_sched.log")
            if log_path else config.log_file_name
        ),
        snapshot_dir="navi_sched",
        enable_kmon=True,
    )

    role_options = load_master_config(config.config_file, config.config_json)
    for name, value in DEFAULT_MASTER_OPTIONS.items():
        role_options.setdefault(name, value)
    for name, value in master_config.items():
        if name in {"http_port", "arpc_port", "grpc_port", "enable_kmon"}:
            continue
        if name == "scheduler_config" and isinstance(value, dict):
            value = json.dumps(value, separators=(",", ":"), sort_keys=True)
        role_options[name] = value
    if isinstance(role_options.get("scheduler_config"), dict):
        role_options["scheduler_config"] = json.dumps(
            role_options["scheduler_config"],
            separators=(",", ":"),
            sort_keys=True,
        )
    runtime_config = role_options.get("runtime_config", {})
    if not isinstance(runtime_config, dict):
        raise ValueError("runtime_config must be a JSON object")
    runtime_config = dict(runtime_config)
    port_diff = agent_grpc_port_diff(route.prefill_endpoint)
    agent_biz_name = prefill_biz_name(
        environ,
        route.prefill_endpoint,
        unified_config.get("client", {}).get("biz_name", ""),
    )
    discovery = make_discovery_runtime(
        route.prefill_endpoint,
        agent_biz_name,
        route.use_local,
        port_diff,
        environ,
    )
    for node in discovery.get("local_nodes", []):
        node["support_heartbeat"] = True
    runtime_config.pop("local_nodes", None)
    runtime_config.pop("vip_subscription", None)
    runtime_config.update(discovery)
    role_options = dict(role_options)
    role_options["runtime_config"] = runtime_config
    return config, role_options


def _run_master_once(config, role_options, stopped, environ):
    config.validate()
    options = dict(role_options)
    options.pop("biz_name", None)
    runtime_config = make_runtime_config(options)
    with NaviSchedMaster(
        config.master_biz_name,
        name=config.name,
        gig={
            "http_port": config.http_port,
            "arpc_port": config.arpc_port,
            "grpc_port": config.grpc_port,
            "snapshot_log_dir": config.snapshot_dir,
        },
        log_file_name=config.log_file_name,
        snapshot_dir=config.snapshot_dir,
        wait_no_query=False,
        runtime_config=runtime_config,
        **make_kmon_options("master", environ),
        **options,
    ) as master:
        master.init()
        logging.info(
            "navi_sched_effective_master_config=%s",
            json.dumps(
                {
                    "process": {
                        "master_biz_name": config.master_biz_name,
                        "name": config.name,
                        "http_port": config.http_port,
                        "arpc_port": config.arpc_port,
                        "grpc_port": config.grpc_port,
                        "log_file_name": config.log_file_name,
                        "snapshot_dir": config.snapshot_dir,
                        "enable_kmon": config.enable_kmon,
                    },
                    "role_options": role_options,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        stopped.wait()


def run_rtp_llm_master(config, environ):
    stopped = threading.Event()

    def stop(signum, frame):
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    while not stopped.is_set():
        try:
            config, role_options = make_rtp_llm_master_options(config, environ)
        except Exception:
            logging.exception("failed to resolve initial NaviSched Prefill endpoints")
            stopped.wait(2.0)
            continue
        try:
            _run_master_once(config, role_options, stopped, environ)
        except Exception:
            logging.exception("NaviSched Master role failed; retrying")
            stopped.wait(2.0)


def main():
    config = MasterProcessConfig.from_env()
    run_rtp_llm_master(config, os.environ)


if __name__ == "__main__":
    main()
