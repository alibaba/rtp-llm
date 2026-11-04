import argparse
import json
import math
import os
import signal
import sys
import threading
import time

from navi_sched import (
    NaviSchedAgent,
    NaviSchedClient,
)
from navi_sched._navi_sched_core import (
    NaviSchedSimulatedExecutionConfig,
    NaviSchedTrafficReplayConfig,
    init_simulated_agent,
    init_simulated_traffic_replay,
)
from navi_sched.util import (
    DEFAULT_AGENT_RUNTIME_NAME,
    DEFAULT_CLIENT_RUNTIME_NAME,
)


def _parse_port(value: str) -> int:
    port = int(value)
    if port < 0 or port > 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return port


def _parse_node(value: str):
    try:
        biz_name, ip, grpc_port = value.rsplit(",", 2)
        grpc_port = _parse_port(grpc_port)
    except ValueError as error:
        raise argparse.ArgumentTypeError("node must be BIZ,IP,GRPC_PORT") from error
    if not biz_name or not ip or grpc_port <= 0:
        raise argparse.ArgumentTypeError("invalid node")
    return (biz_name, ip, grpc_port)


def _parse_options(role: str, argv=None):
    parser = argparse.ArgumentParser(prog=f"navi_sched_simulate_{role}")
    parser.add_argument("--biz", required=True)
    parser.add_argument("--grpc-port", type=_parse_port, default=0)
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=_parse_port, default=0)
    parser.add_argument("--runtime-dir", default="")
    parser.add_argument("--thread-num", type=int, default=64)
    parser.add_argument("--time-compression", type=float, default=1.0)
    parser.add_argument("--node", type=_parse_node, action="append", default=[])
    if role == "agent":
        parser.add_argument(
            "--cache-capacity-bytes", type=int, default=300 * 1024**3)
        parser.add_argument(
            "--token-cache-bytes", type=int, default=10 * 1024)
        parser.add_argument("--block-size", type=int, default=256)
        parser.add_argument("--formula-parameters-json", default="")
        parser.add_argument("--latency-model-state-file", default="")
        parser.add_argument("--latency-fit-batch-size", type=int, default=100000)
        parser.add_argument("--latency-fit-interval", type=int, default=0)
        parser.add_argument("--latency-learning-rate", type=float, default=0.05)
        parser.add_argument("--latency-warmup-sample-count", type=int, default=10)
        parser.add_argument("--noise-stddev-ratio", type=float, default=0.01)
        parser.add_argument("--random-seed", type=int, default=1)
        parser.add_argument("--queue-trace-output", default="")
    elif role == "client":
        parser.add_argument("--master-biz", required=True)
        parser.add_argument("--traffic-log", default="")
        parser.add_argument("--traffic-biz", default="")
        parser.add_argument("--traffic-service-name", default="")
        parser.add_argument("--traffic-client-count", type=int, default=1)
        parser.add_argument("--traffic-client-index", type=int, default=0)
        parser.add_argument("--traffic-timeout-ms", type=int, default=5000)
        parser.add_argument("--traffic-start-delay-ms", type=int, default=2000)
        parser.add_argument("--block-size", type=int, default=256)
        parser.add_argument(
            "--traffic-collect-replay-sequence",
            type=int,
            action="append",
            default=[])
        parser.add_argument("--traffic-replay-output", default="")
        parser.add_argument("--traffic-latency-output", default="")
    options = parser.parse_args(argv)
    if options.thread_num <= 0:
        parser.error("--thread-num must be positive")
    if options.time_compression <= 0:
        parser.error("--time-compression must be positive")
    if role == "agent":
        if options.cache_capacity_bytes <= 0:
            parser.error("cache capacity must be positive")
        if (options.token_cache_bytes <= 0 or options.block_size <= 0 or
                options.latency_fit_batch_size <= 0 or
                options.latency_fit_interval < 0 or
                not math.isfinite(options.latency_learning_rate) or
                options.latency_learning_rate <= 0 or
                options.latency_warmup_sample_count < 0 or
                options.noise_stddev_ratio < 0):
            parser.error("agent latency and cache options are invalid")
        if options.formula_parameters_json:
            try:
                parameters = json.loads(options.formula_parameters_json)
            except json.JSONDecodeError as error:
                parser.error(f"invalid formula parameters JSON: {error}")
            if (not isinstance(parameters, list) or len(parameters) != 11 or
                    not all(isinstance(value, (int, float)) for value in parameters)):
                parser.error("formula parameters must contain eleven numbers")
            options.formula_parameters = [float(value) for value in parameters]
        else:
            options.formula_parameters = []
        if (options.latency_model_state_file and
                not os.path.isfile(options.latency_model_state_file)):
            parser.error("latency model state file does not exist")
    if role == "client" and options.traffic_log:
        if not options.traffic_biz:
            parser.error("--traffic-biz is required with --traffic-log")
        if not options.traffic_service_name:
            parser.error("--traffic-service-name is required with --traffic-log")
        if options.traffic_client_count <= 0:
            parser.error("--traffic-client-count must be positive")
        if not 0 <= options.traffic_client_index < options.traffic_client_count:
            parser.error("--traffic-client-index must be in client count range")
        if options.traffic_timeout_ms <= 0:
            parser.error("--traffic-timeout-ms must be positive")
        if options.traffic_start_delay_ms <= 0 or options.block_size <= 0:
            parser.error("--traffic-start-delay-ms must be positive")
        if any(sequence < 0 for sequence in options.traffic_collect_replay_sequence):
            parser.error("--traffic-collect-replay-sequence must not be negative")
    return options


def _change_runtime_directory(runtime_dir: str) -> str:
    if not runtime_dir:
        return ""
    os.chdir(runtime_dir)
    return os.getcwd()


def _service_options(role: str, options, runtime_dir: str):
    result = {
        "enable_kmon": False,
        "enable_snapshot": False,
        "gig": {"grpc_port": options.grpc_port},
        "wait_no_query": False,
    }
    if runtime_dir:
        # The process has already changed into runtime_dir. Keep every
        # runtime artifact relative so Gig does not prepend the cwd twice.
        result["log_file_name"] = "navi_sched.log"
        result["gig"].update(
            log_prefix="gig",
            snapshot_log_dir="gig_snapshot",
        )
    return result


def _runtime_config(options):
    nodes = []
    for biz_name, ip, grpc_port in options.node:
        nodes.append({
            "biz_name": biz_name,
            "ip": ip,
            "grpc_port": grpc_port,
        })
    return {"thread_num": options.thread_num, "local_nodes": nodes}


def _wait_for_stop():
    stopped = threading.Event()

    def stop(signum, frame):
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    stopped.wait()


def run_agent(argv=None):
    options = _parse_options("agent", argv)
    runtime_dir = _change_runtime_directory(options.runtime_dir)
    agent = NaviSchedAgent(
        options.biz,
        server_port=options.server_port,
        server_ip=options.server_ip,
        total_blocks=(options.cache_capacity_bytes //
                      (options.token_cache_bytes * options.block_size)),
        block_size=options.block_size,
        linear_step=1,
        hybrid_tail_required=False,
        name=DEFAULT_AGENT_RUNTIME_NAME,
        runtime_config=_runtime_config(options),
        batch_timeout_ms=max(
            1, int(round(10 / options.time_compression))),
        latency_fit_batch_size=options.latency_fit_batch_size,
        latency_fit_interval=options.latency_fit_interval,
        latency_learning_rate=options.latency_learning_rate,
        latency_warmup_sample_count=options.latency_warmup_sample_count,
        latency_validation_enabled=True,
        latency_persist_interval=100000,
        latency_model_state_file=options.latency_model_state_file,
        time_compression=options.time_compression,
        **_service_options("agent", options, runtime_dir),
    )
    simulate_handle = None
    try:
        simulate_config = NaviSchedSimulatedExecutionConfig()
        simulate_config.cache_capacity_bytes = options.cache_capacity_bytes
        simulate_config.token_cache_bytes = options.token_cache_bytes
        simulate_config.block_size = options.block_size
        simulate_config.formula_parameters = options.formula_parameters
        simulate_config.noise_stddev_ratio = options.noise_stddev_ratio
        simulate_config.random_seed = options.random_seed
        simulate_config.time_compression = options.time_compression
        simulate_config.queue_trace_output_path = options.queue_trace_output
        simulate_handle = init_simulated_agent(agent, simulate_config)
        if not simulate_handle:
            raise RuntimeError("failed to initialize simulated agent")
        agent.report_cache_status(
            options.cache_capacity_bytes //
            (options.token_cache_bytes * options.block_size),
            options.block_size,
            1,
            False,
        )
        _wait_for_stop()
    finally:
        if simulate_handle:
            simulate_handle.stop()
        agent.stop()


def run_client(argv=None):
    options = _parse_options("client", argv)
    traffic_log = os.path.abspath(options.traffic_log) if options.traffic_log else ""
    runtime_dir = _change_runtime_directory(options.runtime_dir)
    client = NaviSchedClient(
        options.biz,
        options.master_biz,
        runtime_config=_runtime_config(options),
        enable_agent_forward=True,
        name=DEFAULT_CLIENT_RUNTIME_NAME,
        **_service_options("client", options, runtime_dir),
    )
    traffic_handle = None
    try:
        if traffic_log:
            traffic = NaviSchedTrafficReplayConfig()
            traffic.client_biz_name = options.biz
            traffic.traffic_log = traffic_log
            traffic.target_biz_name = options.traffic_biz
            traffic.master_biz_name = options.master_biz
            traffic.service_name = options.traffic_service_name
            traffic.client_count = options.traffic_client_count
            traffic.client_index = options.traffic_client_index
            traffic.time_compression = options.time_compression
            traffic.request_timeout_ms = options.traffic_timeout_ms
            traffic.start_delay_ms = options.traffic_start_delay_ms
            traffic.block_size = options.block_size
            traffic.collect_replay_sequences = (
                ",".join(
                    str(sequence)
                    for sequence in options.traffic_collect_replay_sequence))
            if options.traffic_collect_replay_sequence:
                traffic.replay_output_path = (
                    options.traffic_replay_output or
                    os.path.join(runtime_dir, "traffic_schedule_replay.jsonl"))
            traffic.latency_output_path = (
                options.traffic_latency_output or
                os.path.join(runtime_dir, "traffic_request_latency.jsonl"))
            deadline = time.monotonic() + 30
            while not traffic_handle and time.monotonic() < deadline:
                traffic_handle = init_simulated_traffic_replay(client, traffic)
                if not traffic_handle:
                    time.sleep(0.1)
            configured = bool(traffic_handle)
        else:
            configured = client.init()
        if not configured:
            raise RuntimeError("failed to initialize simulated client")
        _wait_for_stop()
    finally:
        if traffic_handle:
            traffic_handle.stop()
        client.stop()


def main(role: str):
    try:
        if role == "agent":
            run_agent()
        else:
            run_client()
    except (RuntimeError, ValueError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0
