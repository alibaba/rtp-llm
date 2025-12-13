import argparse
import logging
import multiprocessing
import os
import signal
import sys
import time
import traceback

import requests

from rtp_llm.config.py_config_modules import ServerConfig, StaticConfig
from rtp_llm.distribute.gang_info import get_gang_info
from rtp_llm.ops import ProfilingDebugLoggingConfig, RoleType
from rtp_llm.tools.api.hf_model_helper import get_hf_model_info

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.distribute.worker_info import WorkerInfo, g_parallel_info
from rtp_llm.server.server_args.server_args import EnvArgumentParser, setup_args
from rtp_llm.utils.concurrency_controller import init_controller
from rtp_llm.utils.process_manager import ProcessManager


def check_server_health(server_port):
    try:
        response = requests.get(f"http://localhost:{server_port}/health", timeout=60)
        logging.info(
            f"response status_code = {response.status_code}, text = {response.text}, len = {len(response.text)}"
        )
        if response.status_code == 200 and response.text.strip() == '"ok"':
            return True
        else:
            logging.info(f"health check is not ready")
            return False
    except BaseException as e:
        logging.debug("health check is not ready, %s", str(e))
        return False


def start_backend_server_impl(global_controller, process_manager=None):
    from rtp_llm.start_backend_server import start_backend_server

    profiling_debug_config = ProfilingDebugLoggingConfig()
    profiling_debug_config.update_from_env()
    # only for debug
    if profiling_debug_config.debug_load_server:
        start_backend_server(global_controller)
        os._exit(-1)
    backend_process = multiprocessing.Process(
        target=start_backend_server, args=(global_controller,), name="backend_server"
    )
    backend_process.start()

    retry_interval_seconds = 5
    server_config = ServerConfig()
    server_config.update_from_env()
    start_port = server_config.start_port
    backend_server_port = WorkerInfo.backend_server_port_offset(0, start_port)
    while True:
        if not backend_process.is_alive():
            logging.error("Backend server is not alive")
            raise Exception("backend server is not alive")

        try:
            if check_server_health(backend_server_port):
                logging.info(f"backend server is ready")
                break
            else:
                time.sleep(retry_interval_seconds)
        except Exception as e:
            logging.info(f"backend server is not ready")
            time.sleep(retry_interval_seconds)

    return backend_process


def start_frontend_server_impl(
    global_controller, backend_process, process_manager=None
):
    from rtp_llm.start_frontend_server import start_frontend_server

    server_config = ServerConfig()
    server_config.update_from_env()
    frontend_server_count = server_config.frontend_server_count
    if frontend_server_count < 1:
        logging.info(
            "frontend server's count is {frontend_server_count}, this may be a mistake"
        )

    frontend_processes = []

    # tmp code
    local_world_size = g_parallel_info.world_size
    if "LOCAL_WORLD_SIZE" in os.environ:
        logging.info(
            f"multi rank starts with local world size specified in env: {os.environ['LOCAL_WORLD_SIZE']}"
        )
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        logging.info(
            f"multi rank starts with default local world size: {local_world_size}, world size = {g_parallel_info.world_size}"
        )

    for rank in range(local_world_size):
        for i in range(frontend_server_count):
            process = multiprocessing.Process(
                target=start_frontend_server,
                args=(rank, i, global_controller),
                name=f"frontend_server_{i}",
            )
            frontend_processes.append(process)
            process.start()

    retry_interval_seconds = 5
    start_port = server_config.start_port

    while True:
        if not all(proc.is_alive() for proc in frontend_processes):
            logging.error("Frontend server is not alive")
            raise Exception("frontend server is not alive")

        try:
            check_server_health(start_port)
            logging.info(f"frontend server is ready")
            break
        except Exception as e:
            # 如果连接失败，等待一段时间后重试
            time.sleep(retry_interval_seconds)

    return frontend_processes


def should_auto_configure_deepep(args: argparse.Namespace) -> bool:
    """
    Check if DeepEP should be auto-configured.
    Returns True if all DeepEP arguments are None (not set), meaning user hasn't manually configured.
    Returns False if user has manually set any of the DeepEP arguments.
    """
    use_deepep_moe = getattr(args, "use_deepep_moe", None)
    use_deepep_internode = getattr(args, "use_deepep_internode", None)
    use_deepep_low_latency = getattr(args, "use_deepep_low_latency", None)

    # If all are None, user hasn't manually configured, so we should auto-configure
    # If any is not None, user has manually configured, so we shouldn't auto-configure
    return (
        use_deepep_moe is None
        and use_deepep_internode is None
        and use_deepep_low_latency is None
    )


def auto_configure_deepep(args: argparse.Namespace):
    """
    Automatically configure DeepEP settings based on deployment scenario.

    Note: USE_ALL_GATHER should be enabled for pure TP scenarios (ep_size == tp_size).
    When USE_ALL_GATHER is enabled, DeepEP should not be used.

    Configuration rules (for 8-GPU machine):
    - Non-PD separation + Inference node + Single GPU (1TP): 0, 0, 0
    - Non-PD separation + Inference node + Single-node multi-GPU (>1TP): 1, 0, 0
    - Non-PD separation + Inference node + Multi-node multi-GPU: 1, 0, 1
    - PD separation + Prefill node + Single-node single-GPU (1TP): 0, 0, 0
    - PD separation + Decode node + Single-node single-GPU (1TP): 0, 0, 0
    - PD separation + Prefill node + Single-node multi-GPU (2-8 GPUs): 1, 0, 0
    - PD separation + Decode node + Single-node multi-GPU (2-8 GPUs): 1, 1, 0
    - PD separation + Prefill node + Multi-node multi-GPU (>=9 GPUs): 1, 0, 1
    - PD separation + Decode node + Multi-node multi-GPU (>=9 GPUs): 1, 1, 1
    """
    logging.info("auto configure deepep work")
    # Get parallelism info for use_all_gather calculation
    world_size = g_parallel_info.world_size
    tp_size = g_parallel_info.tp_size
    ep_size = g_parallel_info.ep_size
    logging.info(f"world_size: {world_size}, tp_size: {tp_size}, ep_size: {ep_size}")
    # If USE_ALL_GATHER is enabled (for pure TP scenarios), disable all DeepEP settings
    # Calculate use_all_gather: (USE_ALL_GATHER env is True) and (ep_size == tp_size)
    use_all_gather_env = StaticConfig.parallelism_distributed_config.use_all_gather
    use_all_gather = use_all_gather_env and (ep_size == tp_size)

    if use_all_gather:
        logging.info("use all gather in `auto_configure_deepep`")
        os.environ["USE_DEEPEP_MOE"] = "0"
        os.environ["USE_DEEPEP_LOW_LATENCY"] = "0"
        os.environ["USE_DEEPEP_INTERNODE"] = "0"
        logging.info(
            f"USE_ALL_GATHER is enabled (use_all_gather={use_all_gather}), "
            f"all DeepEP settings are disabled (0, 0, 0)"
        )
        return

    # Get deployment information from StaticConfig
    role_type_enum = StaticConfig.role_config.role_type
    role_type = (
        role_type_enum.name if hasattr(role_type_enum, "name") else str(role_type_enum)
    )

    # Get number of nodes
    try:
        gang_info = get_gang_info()
        num_nodes = gang_info.num_nodes
    except Exception:
        # If get_gang_info fails, estimate from world_size
        # Assuming 8 GPUs per node
        num_nodes = (world_size + 7) // 8
        logging.info(
            f"Failed to get gang_info, estimated num_nodes={num_nodes} from world_size={world_size}"
        )

    # Determine if PD separation is enabled
    is_pd_separation = role_type_enum in [RoleType.PREFILL, RoleType.DECODE]
    is_inference = role_type_enum == RoleType.PDFUSION
    is_decode = role_type_enum == RoleType.DECODE

    # Determine GPU configuration
    is_single_gpu = tp_size == 1
    is_multi_gpu = tp_size > 1
    is_multi_node = num_nodes > 1 or world_size >= 9

    # Apply configuration rules
    use_deepep_moe = False
    use_deepep_low_latency = False
    use_deepep_internode = False

    if is_inference:
        # Non-PD separation + Inference node
        if is_single_gpu:
            # Single GPU (1TP): 0, 0, 0
            use_deepep_moe = False
            use_deepep_low_latency = False
            use_deepep_internode = False
        elif is_multi_gpu and not is_multi_node:
            # Single-node multi-GPU (>1TP): 1, 0, 0
            use_deepep_moe = True
            use_deepep_low_latency = False
            use_deepep_internode = False
        elif is_multi_node:
            # Multi-node multi-GPU: 1, 0, 1
            use_deepep_moe = True
            use_deepep_low_latency = False
            use_deepep_internode = True
    elif is_pd_separation:
        # PD separation
        if is_single_gpu:
            # Single-node single-GPU: 0, 0, 0
            use_deepep_moe = False
            use_deepep_low_latency = False
            use_deepep_internode = False
        elif is_multi_gpu and not is_multi_node:
            # Single-node multi-GPU (2-8 GPUs)
            use_deepep_moe = True
            if is_decode:
                use_deepep_low_latency = True
        elif is_multi_node:
            # Multi-node multi-GPU (>=9 GPUs)
            use_deepep_moe = True
            use_deepep_internode = True
            if is_decode:
                use_deepep_low_latency = True

    # Set environment variables
    os.environ["USE_DEEPEP_MOE"] = "1" if use_deepep_moe else "0"
    os.environ["USE_DEEPEP_LOW_LATENCY"] = "1" if use_deepep_low_latency else "0"
    os.environ["USE_DEEPEP_INTERNODE"] = "1" if use_deepep_internode else "0"

    logging.info(
        f"Auto-configured DeepEP settings based on deployment scenario:\n"
        f"  Role Type: {role_type}\n"
        f"  TP Size: {tp_size}\n"
        f"  World Size: {world_size}\n"
        f"  Num Nodes: {num_nodes}\n"
        f"  PD Separation: {is_pd_separation}\n"
        f"  USE_DEEPEP_MOE: {use_deepep_moe}\n"
        f"  USE_DEEPEP_LOW_LATENCY: {use_deepep_low_latency}\n"
        f"  USE_DEEPEP_INTERNODE: {use_deepep_internode}"
    )


def get_model_type_and_update_env(parser: EnvArgumentParser, args: argparse.Namespace):
    if (
        hasattr(args, "checkpoint_path")
        and args.checkpoint_path is not None
        and args.checkpoint_path != ""
    ):
        model_path = args.checkpoint_path
        current_model_type = os.environ.get(
            "MODEL_TYPE", StaticConfig.model_config.model_type
        )
        if not current_model_type or current_model_type == "":
            if (
                hasattr(args, "model_type")
                and args.model_type is not None
                and args.model_type != ""
            ):
                config_model_type = args.model_type
            else:
                model_info = get_hf_model_info(model_path)
                config_model_type = model_info.ft_model_type
                setattr(args, "model_type", config_model_type)
            if config_model_type is not None and config_model_type != "":
                EnvArgumentParser.update_env_from_args(parser, "model_type", args)
    StaticConfig.update_from_env()


def main():
    parser, args = setup_args()

    start_server(parser, args)


def start_server(parser: EnvArgumentParser, args: argparse.Namespace):
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError as e:
        logging.warning(str(e))

    global_controller = init_controller()
    process_manager = ProcessManager()
    get_model_type_and_update_env(parser, args)

    # Auto-configure DeepEP settings based on deployment scenario
    # Check from args to see if user has manually configured
    if should_auto_configure_deepep(args):
        auto_configure_deepep(args)
    else:
        logging.info(
            "DeepEP configuration already set manually, skipping auto-configuration"
        )

    try:
        backend_process = None
        if os.environ.get("ROLE_TYPE", "") != "FRONTEND":
            logging.info("start backend server")
            backend_process = start_backend_server_impl(
                global_controller, process_manager
            )
            process_manager.add_process(backend_process)
            logging.info(f"backend server process = {backend_process}")

        logging.info("start frontend server")
        frontend_process = start_frontend_server_impl(
            global_controller, backend_process, process_manager
        )
        process_manager.add_processes(frontend_process)
        logging.info(f"frontend server process = {frontend_process}")

        logging.info(f"后端RPC 服务监听的ip为 0.0.0.0，ip/ip段可自定义为所需范围")
    except Exception as e:
        logging.error(f"start failed, trace: {traceback.format_exc()}")
        # Trigger graceful shutdown on any exception
        process_manager.graceful_shutdown()
    finally:
        process_manager.monitor_and_release_processes()


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    main()
