import json
import logging
import os
import pathlib
import shlex
import socket
import sys
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence

current_file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(current_file_path.parent.parent.parent.absolute()))
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import (
    set_parallelism_config,
    setup_and_configure_server,
)
from rtp_llm.distribute.distributed_server import members_from_test_env
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.test.perf_test.decode.scheduler import requires_distributed_decode
from rtp_llm.test.perf_test.decode.transport import DecodeHttpTransport
from rtp_llm.test.perf_test.execution.runtime import build_engine_launch_spec
from rtp_llm.test.perf_test.multi_node.decode_runtime import (
    build_external_decode_plan,
    execute_master_decode,
    run_external_rank,
)
from rtp_llm.test.perf_test.multi_node.decode_sync import (
    ClusterDecodeTransport,
    serve_decode_worker,
    validate_decode_layout,
)
from rtp_llm.test.perf_test.multi_node.gang_config import build_gang_config_string
from rtp_llm.test.perf_test.multi_node.prefill_runtime import (
    build_external_prefill_plan,
    execute_master_prefill,
)
from rtp_llm.test.perf_test.plan import ExecutionPlan
from rtp_llm.test.perf_test.test_util import create_query
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.utils.fuser import fetch_remote_file_to_local

# from uvicorn.loops.uvloop import uvloop_setup
# uvloop_setup()


def wait_master_done(
    env_dict: Optional[Dict[str, str]] = None, world_rank: int = 0
) -> None:
    # Get gang_config_string from environment variable or env_dict
    dist_config_str = (env_dict or {}).get("GANG_CONFIG_STRING") or os.environ.get(
        "GANG_CONFIG_STRING", ""
    )
    if not dist_config_str:
        raise RuntimeError("no gang config string, unexpected!")
    dist_members = members_from_test_env(dist_config_str)
    master_member = dist_members[0]
    master_host = master_member.ip
    master_port = master_member.server_port
    while True:
        logging.info(
            f"rank [{world_rank}] waiting for master {master_host}:{master_port} done"
        )
        time.sleep(10)

        # 添加3次重试机制
        retry_count = 0
        max_retries = 5
        connection_failed = True

        while retry_count < max_retries and connection_failed:
            try:
                sock = socket.create_connection(
                    (master_host, master_port), timeout=1000
                )
                sock.close()
                connection_failed = False
            except (socket.error, ConnectionRefusedError) as e:
                retry_count += 1
                if retry_count < max_retries:
                    logging.info(
                        f"rank [{world_rank}] connection attempt {retry_count} failed, retrying... Error: {e}"
                    )
                    time.sleep(5)  # 重试前等待2秒
                else:
                    logging.info(
                        f"rank [{world_rank}] all {max_retries} connection attempts failed, master is done"
                    )
                    break

        if connection_failed:
            break

    logging.info(
        f"rank [{world_rank}] master {master_host}:{master_port} done, this worker exit!"
    )
    return


def try_upload_log(log_dir_path: str, upload_path: str):
    import shutil

    if not os.path.exists(log_dir_path) or not os.path.isdir(log_dir_path):
        logging.info(f"{log_dir_path} not exist, skip upload")
        return
    if not os.listdir(log_dir_path):
        logging.info(f"{log_dir_path} is empty, skip upload")
        return

    zip_path = f"{log_dir_path}.zip"
    shutil.make_archive(log_dir_path, "zip", log_dir_path)
    logging.info(f"zip {log_dir_path} to {zip_path}")

    logging.info(f"upload {zip_path} ...")
    os.system(f"osscmd put {zip_path} {upload_path}/{zip_path}")

    # shutil.rmtree(log_dir_path)
    os.remove(zip_path)


@dataclass
class LocalRankGroupContext:
    """Per-host context needed to start one local rank group."""

    port: int
    world_rank: int
    log_dir_path: str
    profiler_directory: pathlib.Path
    plan: ExecutionPlan


@dataclass
class ServerHandle:
    """Owning handle for one host's local rank-group server.

    Wraps the MagaServerManager together with the startup byproducts the
    caller needs: the locally resolved tokenizer path (rank zero only),
    whether this process became its own process group leader during
    startup, and the member's log dir (where the engine drops raw chrome
    traces through TORCH_CUDA_PROFILER_DIR). The caller launches and stops
    the server through the handle so the legacy cleanup, upload, and exit
    semantics stay at the call site.
    """

    manager: MagaServerManager
    tokenizer_path: Optional[str]
    pgrp_set: bool
    log_dir: str

    @property
    def log_file_path(self) -> Optional[str]:
        return self.manager.log_file_path

    def start_server(self) -> bool:
        return self.manager.start_server()

    def print_process_log(self) -> None:
        self.manager.print_process_log()

    def stop_server(self) -> None:
        self.manager.stop_server()


def _host_engine_environment(environment: Dict[str, str]) -> Dict[str, str]:
    """Keep launcher-owned host assignments out of custom engine overrides."""
    runtime_keys = {
        "WORLD_RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "TP_SIZE",
        "DP_SIZE",
        "PP_SIZE",
        "EP_SIZE",
        "GANG_CONFIG_STRING",
        "START_PORT",
        "CUDA_VISIBLE_DEVICES",
        "TORCH_CUDA_PROFILER_DIR",
        "TEST_UNDECLARED_OUTPUTS_DIR",
        "MAGA_SERVER_WORK_DIR",
    }
    return {
        **{key: value for key, value in environment.items() if key not in runtime_keys},
        **{key: os.environ[key] for key in runtime_keys if key in os.environ},
    }


def start_local_rank_group(
    context: LocalRankGroupContext,
    config: PyEnvConfigs,
    model_paths: Optional[str],
    *,
    engine_argv: Optional[Sequence[str]] = None,
    engine_environment: Optional[Dict[str, str]] = None,
) -> ServerHandle:
    """Start this host's local rank group.

    Performs the legacy per-host startup in the legacy order: model setup
    and download, engine environment, output directories, process-group
    isolation, rank-zero tokenizer resolution, then construction of the
    maga server manager that owns every local rank. ``config`` is the
    parsed ``PyEnvConfigs`` and ``model_paths`` is the tokenizer location
    used as a fallback when model setup leaves ``model_args.tokenizer_path``
    empty. The tokenizer otherwise uses the same resolved path as the engine
    (gang workers pass their locally resolved path). The server
    subprocess itself is launched by the caller through the returned
    handle so legacy failure, upload, and exit semantics are unchanged.

    ``engine_argv``/``engine_environment`` are the unified launch-spec
    projection of the resolved config (gang workers always supply them):
    without ``engine_argv`` the server subprocess starts with no engine
    arguments, so per-run switches such as ``reuse_cache`` would be lost.
    Both SSH startup and gang workers supply this projection.
    """
    if engine_environment is not None:
        engine_environment = _host_engine_environment(engine_environment)
        os.environ.update(engine_environment)
    setup_and_configure_server(config)
    engine = context.plan.resolved_config["engine"]
    os.environ["MAX_SEQ_LEN"] = str(engine["max_seq_len"])
    os.environ["MAX_BATCH_TOKENS_SIZE"] = str(engine["max_batch_tokens"])
    os.environ["CONCURRENCY_LIMIT"] = str(engine["concurrency_limit"])
    os.makedirs(context.log_dir_path, exist_ok=True)
    os.environ["TORCH_CUDA_PROFILER_DIR"] = str(context.profiler_directory)
    # for maga server files
    os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = context.log_dir_path
    os.environ["MAGA_SERVER_WORK_DIR"] = os.getcwd()

    pgrp_set = False
    try:
        os.setpgrp()
        pgrp_set = True
    except Exception as e:
        logging.info(f"setpgrp error: {e}")

    tokenizer_path = None
    if context.world_rank == 0:
        # Use the same resolved tokenizer as the engine after model setup.
        model_paths = config.model_args.tokenizer_path or model_paths
        if not model_paths:
            raise RuntimeError(
                "tokenizer_path is missing; set model_args.tokenizer_path or ckpt_path"
            )
        tokenizer_path = fetch_remote_file_to_local(
            os.path.expanduser(str(model_paths).strip())
        )

    manager_kwargs: Dict[str, Any] = {"port": str(context.port)}
    if engine_argv is not None:
        # Configuration downloads model files on this host. Explicit argv
        # must use those paths rather than overriding them with remote IDs.
        model_options = {
            "checkpoint_path": config.model_args.ckpt_path,
            "tokenizer_path": config.model_args.tokenizer_path,
            "model_type": config.model_args.model_type,
        }
        local_argv = [
            arg
            for arg in engine_argv
            if arg.split("=", 1)[0] not in {"--" + key for key in model_options}
        ]
        local_argv.extend(
            f"--{key}={value}" for key, value in model_options.items() if value
        )
        manager_kwargs["smoke_args_str"] = shlex.join(local_argv)
    if engine_environment:
        # Output paths are assigned after the initial projection; refresh them
        # so manager overrides cannot restore stale or custom directories.
        manager_kwargs["env_args"] = _host_engine_environment(engine_environment)
    server = MagaServerManager(**manager_kwargs)
    return ServerHandle(
        manager=server,
        tokenizer_path=tokenizer_path,
        pgrp_set=pgrp_set,
        log_dir=str(context.log_dir_path),
    )


if __name__ == "__main__":
    config_path = os.environ.get("PERF_TEST_CONFIG_PATH")
    if not config_path:
        raise RuntimeError("PERF_TEST_CONFIG_PATH is required for multi-node perf")

    mode = os.environ.get("PERF_TEST_MODE", "decode")
    if mode not in ("decode", "prefill"):
        raise RuntimeError(f"unsupported PERF_TEST_MODE: {mode}")

    os.environ["USE_BATCH_DECODE_SCHEDULER"] = "1"
    os.environ["FAKE_BALANCE_EXPERT"] = "1"

    py_env_configs: PyEnvConfigs = setup_args()
    set_parallelism_config(
        py_env_configs.parallelism_config,
        py_prefill_cp_config=py_env_configs.prefill_cp_config,
    )
    port = py_env_configs.server_config.start_port
    world_rank = py_env_configs.parallelism_config.world_rank
    run_id = os.environ.setdefault("PERF_TEST_RUN_ID", f"{mode}-{time.time_ns()}")
    log_dir_name = f"TEST_OUTPUT_{run_id}_{world_rank}"
    log_dir_path = os.path.abspath(log_dir_name)
    runtime_profiler_directory = pathlib.Path(log_dir_path)
    parallelism = py_env_configs.parallelism_config
    build_plan = (
        build_external_prefill_plan if mode == "prefill" else build_external_decode_plan
    )
    plan, source = build_plan(
        config_path,
        **(
            {"resolved_workload_path": os.environ["PERF_TEST_RESOLVED_WORKLOAD_PATH"]}
            if mode == "prefill" and os.environ.get("PERF_TEST_RESOLVED_WORKLOAD_PATH")
            else {}
        ),
        result_directory=str(pathlib.Path(log_dir_path) / "run"),
        actual_topology={
            "tp": parallelism.tp_size,
            "dp": parallelism.dp_size,
            "pp": parallelism.pp_size,
            "ep": parallelism.ep_size,
        },
    )
    # Host model options are assertions about the frozen plan, not overrides.
    # Compare sources before downloads; resolved local paths are projected only
    # after setup_and_configure_server, without changing the plan's identity.
    host_model = {
        "checkpoint": py_env_configs.model_args.ckpt_path,
        "tokenizer": py_env_configs.model_args.tokenizer_path,
        "type": py_env_configs.model_args.model_type,
    }
    for key, value in host_model.items():
        if not value:
            continue
        host_value = str(value).strip()
        planned_value = str(plan.resolved_config["model"][key]).strip()
        if key != "type":
            if "://" not in host_value:
                host_value = str(pathlib.Path(host_value).expanduser().resolve())
            if "://" not in planned_value:
                planned_value = str(pathlib.Path(planned_value).expanduser().resolve())
        if host_value != planned_value:
            raise RuntimeError(
                f"host model {key} conflicts with the frozen perf config; "
                "set the model in PERF_TEST_CONFIG_PATH and remove the host override"
            )
    launch = build_engine_launch_spec(plan)
    launch = replace(launch, environment=_host_engine_environment(launch.environment))
    os.environ.update(launch.environment)
    py_env_configs = setup_args(list(launch.argv))
    set_parallelism_config(
        py_env_configs.parallelism_config,
        py_prefill_cp_config=py_env_configs.prefill_cp_config,
    )
    # Normalize the inherited gang config through the shared builder so the
    # legacy flow and gang workers construct identical GANG_CONFIG_STRINGs.
    gang_config_string = py_env_configs.distribute_config.gang_config_string
    if gang_config_string:
        os.environ["GANG_CONFIG_STRING"] = build_gang_config_string(
            members_from_test_env(gang_config_string),
            parallelism.local_world_size,
        )
    input_len_list = (
        sorted(
            {length for case in plan.cases for length in case.identity["input_lengths"]}
        )
        if mode == "decode"
        else []
    )
    context = LocalRankGroupContext(
        port=port,
        world_rank=world_rank,
        log_dir_path=log_dir_path,
        profiler_directory=runtime_profiler_directory,
        plan=plan,
    )
    server_handle = start_local_rank_group(
        context,
        py_env_configs,
        py_env_configs.model_args.tokenizer_path,
        engine_argv=launch.argv,
        engine_environment=launch.environment,
    )
    distributed_decode = mode == "decode" and requires_distributed_decode(
        {
            "tp": parallelism.tp_size,
            "dp": parallelism.dp_size,
            "pp": parallelism.pp_size,
        },
        parallelism.local_world_size,
    )
    members = ()
    if distributed_decode:
        members = members_from_test_env(
            py_env_configs.distribute_config.gang_config_string
        )
        validate_decode_layout(
            parallelism.local_world_size,
            parallelism.tp_size,
            parallelism.dp_size,
            tuple(
                index * parallelism.local_world_size for index in range(len(members))
            ),
        )
    input_query_dict = {}
    if mode == "decode" and (world_rank == 0 or distributed_decode):
        tokenizer_path = server_handle.tokenizer_path
        if tokenizer_path is None:
            tokenizer_path = fetch_remote_file_to_local(
                os.path.expanduser(
                    str(py_env_configs.model_args.tokenizer_path).strip()
                )
            )
        input_query_dict = create_query(
            tokenizer_path=tokenizer_path,
            input_len_list=input_len_list,
        )
    cluster = None
    try:
        if not server_handle.start_server():
            server_handle.print_process_log()
            raise Exception("server start failed")
        execute_worker = None
        if distributed_decode:
            local_transport = DecodeHttpTransport(
                endpoint=f"http://127.0.0.1:{port}",
                prompt_for_length=lambda length: input_query_dict[length],
            )
            control_port = int(
                os.environ.get("PERF_TEST_DECODE_CONTROL_PORT", port + 1024)
            )
            if not 0 < control_port < 65536:
                raise ValueError(
                    "PERF_TEST_DECODE_CONTROL_PORT must be a valid TCP port"
                )
            if world_rank == 0:
                cluster = ClusterDecodeTransport(
                    local_transport,
                    bind_host="0.0.0.0",
                    control_port=control_port,
                    local_world_size=parallelism.local_world_size,
                    tp_size=parallelism.tp_size,
                    dp_size=parallelism.dp_size,
                    member_world_ranks=tuple(
                        index * parallelism.local_world_size
                        for index in range(len(members))
                    ),
                )
                cluster.connect_workers()
            else:
                execute_worker = lambda: serve_decode_worker(
                    members[0].ip,
                    control_port,
                    world_rank=world_rank,
                    local_world_size=parallelism.local_world_size,
                    tp_size=parallelism.tp_size,
                    dp_size=parallelism.dp_size,
                    local_transport=local_transport,
                )
        execute_master = (
            (
                lambda: execute_master_prefill(
                    plan,
                    source_config=source,
                    endpoint=f"http://127.0.0.1:{port}",
                    tokenizer_path=server_handle.tokenizer_path,
                    server_log_path=server_handle.log_file_path,
                    profiler_directory=runtime_profiler_directory,
                )
            )
            if mode == "prefill"
            else lambda: execute_master_decode(
                plan,
                source_config=source,
                endpoint=f"http://127.0.0.1:{port}",
                prompt_for_length=lambda length: input_query_dict[length],
                server_log_path=server_handle.log_file_path,
                profiler_directory=runtime_profiler_directory,
                transport=cluster,
            )
        )
        result = run_external_rank(
            world_rank=world_rank,
            wait_for_master=lambda: wait_master_done(world_rank=world_rank),
            execute_master=execute_master,
            execute_worker=execute_worker,
        )
        if result is not None:
            print(
                f"PERF_TEST_{mode.upper()}_RESULT "
                + json.dumps(
                    {
                        "state": result.state.value,
                        "run_id": run_id,
                        "world_rank": world_rank,
                        "config_fingerprint": plan.config_fingerprint,
                        "workload_fingerprint": plan.workload_fingerprint,
                    }
                )
            )
            sys.stdout.flush()
            if result.state.value != "completed":
                raise RuntimeError(f"{mode} run is {result.state.value}")
    finally:
        if cluster is not None:
            cluster.close()
        server_handle.stop_server()
        upload_path = os.environ.get("UPLOAD_OSS_PATH", "")
        if upload_path != "":
            logging.info(f"upload log to {upload_path}")
            try_upload_log(log_dir_path, upload_path)
