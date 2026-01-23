import logging
import multiprocessing
import os
import sys
import time
import traceback

import requests
import torch

from rtp_llm.distribute.distributed_server import get_world_info
from rtp_llm.utils.time_util import timer_wrapper

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.distribute.worker_info import WorkerInfo, g_parallel_info
from rtp_llm.ops import RoleType, VitSeparation
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.concurrency_controller import init_controller
from rtp_llm.utils.process_manager import ProcessManager

setup_logging()


def check_server_health(server_port):
    try:
        response = requests.get(f"http://localhost:{server_port}/health", timeout=60)
        if response.status_code == 200 and response.json().get("status", "") == "ok":
            logging.info(
                f"{server_port}/health, response status_code = {response.status_code}, text = {response.text}, len = {len(response.text)}"
            )
            return True
        else:
            return False
    except BaseException as e:
        return False


def _check_all_vit_workers_ready(worker_addresses):
    """
    检查所有 VIT worker 进程是否就绪（通过 gRPC GetWorkerStatus）

    Args:
        worker_addresses: worker 地址列表，格式如 ['127.0.0.1:9202', '127.0.0.1:9203']

    Returns:
        True 如果所有 worker 都就绪，False 如果有任何 worker 不就绪
    """
    import grpc

    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import StatusVersionPB
    from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
        MultimodalRpcServiceStub,
    )

    if not worker_addresses:
        return True

    request = StatusVersionPB()
    healthy_count = 0

    for worker_address in worker_addresses:
        try:
            # 创建临时连接检查 worker 状态
            channel = grpc.insecure_channel(
                worker_address,
                options=[
                    ("grpc.max_send_message_length", 1024 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
                ],
            )
            stub = MultimodalRpcServiceStub(channel)
            worker_status_response = stub.GetWorkerStatus(request, timeout=2)
            channel.close()

            if worker_status_response.alive:
                healthy_count += 1
                logging.debug(f"[VIT_WORKER_CHECK] Worker {worker_address} is ready")
            else:
                logging.warning(
                    f"[VIT_WORKER_CHECK] Worker {worker_address} is not alive"
                )
        except grpc.RpcError as e:
            logging.debug(
                f"[VIT_WORKER_CHECK] Worker {worker_address} not ready yet: {e.code()} - {e.details()}"
            )
        except Exception as e:
            logging.debug(
                f"[VIT_WORKER_CHECK] Worker {worker_address} not ready yet: {e}"
            )

    all_ready = healthy_count == len(worker_addresses)
    if all_ready:
        logging.info(
            f"[VIT_WORKER_CHECK] All {len(worker_addresses)} workers are ready"
        )
    else:
        logging.info(
            f"[VIT_WORKER_CHECK] Worker readiness: {healthy_count}/{len(worker_addresses)} workers are ready"
        )
    return all_ready


@timer_wrapper(description="start backend server")
def start_backend_server_impl(
    global_controller,
    py_env_configs: PyEnvConfigs,
    process_manager: ProcessManager = None,
):
    from rtp_llm.start_backend_server import start_backend_server

    # only for debug
    if py_env_configs.profiling_debug_logging_config.debug_load_server:
        start_backend_server(global_controller, py_env_configs, None)
        os._exit(-1)

    # Create pipe for subprocess startup status communication
    pipe_reader, pipe_writer = torch.multiprocessing.Pipe(duplex=False)
    logging.info(f"[PROCESS_SPAWN]Start backend server process outer")

    backend_process = torch.multiprocessing.Process(
        target=start_backend_server,
        args=(global_controller, py_env_configs, pipe_writer),
        name="backend_manager",
    )
    backend_process.start()
    pipe_writer.close()  # Parent process closes write end

    # Create check_ready_fn for pipe-based health check
    max_wait_seconds = 60 * 60
    startup_status = {"ready": False, "error": None}

    def check_backend_ready():
        """Check if backend server is ready via pipe communication"""
        if startup_status["ready"]:
            return True

        if startup_status["error"]:
            raise Exception(startup_status["error"])

        # Non-blocking check if data is available
        if pipe_reader.poll(timeout=0):
            try:
                status_msg = pipe_reader.recv()
                if status_msg.get("status") == "success":
                    logging.info(
                        f"Backend server started successfully: {status_msg.get('message', '')}"
                    )
                    startup_status["ready"] = True
                    pipe_reader.close()
                    return True
                else:
                    # Startup failed
                    error_msg = status_msg.get("message", "Unknown error")
                    traceback_info = status_msg.get("traceback", "")
                    if traceback_info:
                        logging.error(f"Traceback: {traceback_info}")

                    error = f"Backend server start failed: {error_msg}"
                    startup_status["error"] = error
                    pipe_reader.close()
                    raise Exception(error)
            except EOFError:
                error = "Backend server pipe closed unexpectedly"
                startup_status["error"] = error
                pipe_reader.close()
                raise Exception(error)

        return False

    # Register health check with ProcessManager using custom check_ready_fn
    if process_manager:
        process_manager.register_health_check(
            processes=[backend_process],
            process_name="backend_server",
            check_ready_fn=check_backend_ready,
            retry_interval_seconds=1,
        )

    return backend_process


@timer_wrapper(description="start vit server")
def start_vit_server_impl(
    py_env_configs: PyEnvConfigs,
    process_manager: ProcessManager = None,
):
    """
    启动 VIT 服务器

    Args:
        py_env_configs: 配置对象
        process_manager: 进程管理器
    """
    from rtp_llm.distribute.worker_info import WorkerInfo
    from rtp_llm.multimodal.vit_proxy_start_server import vit_proxy_start_server
    from rtp_llm.multimodal.vit_start_server import vit_start_server

    server_config = py_env_configs.server_config
    start_port = server_config.start_port
    vit_server_count = server_config.vit_server_count

    if vit_server_count > 1:
        logging.info(
            f"[VIT_SERVER] Starting in PROXY mode: 1 proxy + {vit_server_count} workers "
            f"(role_type=VIT)"
        )

        worker_processes = []
        worker_addresses = []

        base_grpc_port = WorkerInfo.rpc_server_port_offset(0, start_port)

        for i in range(vit_server_count):
            internal_grpc_port = base_grpc_port + i + 1
            worker_addresses.append(f"127.0.0.1:{internal_grpc_port}")

            logging.info(
                f"[PROCESS_SPAWN] Start vit worker process worker_{i} "
                f"(internal grpc_port={internal_grpc_port})"
            )
            process = torch.multiprocessing.Process(
                target=vit_start_server,
                args=(
                    i,
                    py_env_configs,
                    internal_grpc_port,  # grpc_port
                    None,  # http_port (工作进程不需要 HTTP，None 表示工作进程模式)
                    True,  # is_proxy_mode (proxy 模式下的 worker 进程)
                ),
                name=f"vit_worker_{i}",
            )
            worker_processes.append(process)
            process.start()

        external_grpc_port = base_grpc_port  # 主进程使用基础 gRPC 端口
        external_http_port = WorkerInfo.server_port_offset(0, start_port)

        # 2. 启动主进程（代理服务器）
        logging.info(
            f"[PROCESS_SPAWN] Start vit proxy process "
            f"(external grpc_port={external_grpc_port}, http_port={external_http_port})"
        )
        proxy_process = torch.multiprocessing.Process(
            target=vit_proxy_start_server,
            args=(
                py_env_configs,
                worker_addresses,
                external_grpc_port,  # grpc_port
                external_http_port,  # http_port
            ),
            name="vit_proxy",
        )
        proxy_process.start()

        vit_processes = [proxy_process] + worker_processes

        # 健康检查：检查代理服务器的端口（代理模式只在 VIT 角色时启用）
        vit_server_port = WorkerInfo.server_port_offset(0, start_port)
        # 保存 worker 地址列表，用于健康检查

    else:
        grpc_port = WorkerInfo.rpc_server_port_offset(0, start_port)
        http_port = WorkerInfo.server_port_offset(0, start_port)
        vit_server_port = http_port
        logging.info(
            f"[PROCESS_SPAWN] Start vit server process "
            f"(grpc_port={grpc_port}, http_port={http_port})"
        )
        process = torch.multiprocessing.Process(
            target=vit_start_server,
            args=(
                0,
                py_env_configs,
                grpc_port,  # grpc_port
                http_port,  # http_port
                False,  # is_proxy_mode (standalone 模式，需要记录 QPS)
            ),
            name="vit_server",
        )
        process.start()
        vit_processes = [process]

    if process_manager and vit_processes:
        logging.info(
            f"[VIT_SERVER] Registering health check for {len(vit_processes)} VIT processes, "
            f"current_managed_processes={len(process_manager.processes)}"
        )

        def check_vit_ready():
            return check_server_health(vit_server_port)

        process_manager.register_health_check(
            processes=vit_processes,
            process_name="vit_server",
            check_ready_fn=check_vit_ready,
            retry_interval_seconds=1,
        )
        logging.info(
            f"[VIT_SERVER] Health check registered, after_registration: "
            f"managed_processes={len(process_manager.processes)}, "
            f"health_check_processes={len(process_manager.health_check_processes)}"
        )

    return vit_processes


@timer_wrapper(description="start frontend server")
def start_frontend_server_impl(
    global_controller,
    py_env_configs: PyEnvConfigs,
    process_manager=None,
):
    from rtp_llm.start_frontend_server import start_frontend_server

    frontend_server_count = py_env_configs.server_config.frontend_server_count
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

    # To reduce the number of frontend servers, we only start those with tp_rank=0;
    # however, since k8s needs to check machine heartbeat, rank 0 on each machine also needs to be started.
    for rank in range(local_world_size):
        for i in range(frontend_server_count):
            if (
                rank == 0
                or (g_parallel_info.world_rank + rank) % g_parallel_info.tp_size == 0
            ):
                logging.info(
                    f"[PROCESS_SPAWN]Start frontend server process rank_{rank}_server_{i} outer"
                )
                process = multiprocessing.Process(
                    target=start_frontend_server,
                    args=(rank, i, global_controller, py_env_configs),
                    name=f"frontend_server_{i}",
                )
                frontend_processes.append(process)
                process.start()
            else:
                logging.info(
                    f"rank {g_parallel_info.world_rank + rank} skipping frontend startup"
                )

    if process_manager and frontend_processes:

        def check_frontend_ready():
            return check_server_health(py_env_configs.server_config.start_port)

        process_manager.register_health_check(
            processes=frontend_processes,
            process_name="frontend_server",
            check_ready_fn=check_frontend_ready,
            retry_interval_seconds=1,
        )

    return frontend_processes


def main():
    py_env_configs: PyEnvConfigs = setup_args()
    setup_and_configure_server(py_env_configs)
    start_server(py_env_configs)


def start_server(py_env_configs: PyEnvConfigs):
    logging.info(f"[PROCESS_START]Start server")
    start_time = time.time()
    try:
        multiprocessing.set_start_method("spawn", force=True)
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError as e:
        logging.warning(str(e))

    global_controller = init_controller(
        py_env_configs.concurrency_config, dp_size=g_parallel_info.dp_size
    )

    # Create process manager with config values
    process_manager = ProcessManager(
        shutdown_timeout=py_env_configs.server_config.shutdown_timeout,
        monitor_interval=py_env_configs.server_config.monitor_interval,
    )

    # Initialize backend_process to None in case role_type is FRONTEND
    backend_process = None
    # Get number of nodes
    try:
        world_info = get_world_info(
            py_env_configs.server_config, py_env_configs.distribute_config
        )
        num_nodes = world_info.num_nodes
    except Exception:
        # If get_world_info fails, estimate from world_size
        # Assuming 8 GPUs per node
        num_nodes = (g_parallel_info.world_size + 7) // 8
        logging.info(
            f"Failed to get world_info, estimated num_nodes={num_nodes} from world_size={g_parallel_info.world_size}"
        )
    try:
        if py_env_configs.role_config.role_type == RoleType.VIT:
            vit_processes = start_vit_server_impl(py_env_configs, process_manager)
            process_manager.add_processes(vit_processes)

        if (
            py_env_configs.role_config.role_type != RoleType.FRONTEND
            and py_env_configs.role_config.role_type != RoleType.VIT
        ):
            # For backend server, vit_process_engine is None when vit is separated
            backend_process = start_backend_server_impl(
                global_controller, py_env_configs, process_manager
            )
            process_manager.add_process(backend_process)

        if py_env_configs.role_config.role_type != RoleType.VIT:
            # vit has its own frontend server
            frontend_process = start_frontend_server_impl(
                global_controller, py_env_configs, process_manager
            )
            process_manager.add_processes(frontend_process)

        if not process_manager.run_health_checks():
            logging.error("[START_SERVER] Health checks failed")
            raise Exception("Health checks failed")

    except Exception as e:
        logging.error(f"start failed, trace: {traceback.format_exc()}")
        process_manager.graceful_shutdown()
    finally:
        process_manager.monitor_and_release_processes()


if __name__ == "__main__":
    main()
