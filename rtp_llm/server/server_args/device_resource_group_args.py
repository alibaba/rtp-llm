import argparse

from rtp_llm.server.server_args.util import str2bool


def init_device_resource_group_args(parser, runtime_config):
    ##############################################################################################################
    # 设备和资源管理
    ##############################################################################################################
    device_resource_group = parser.add_argument_group("设备和资源管理")

    # Runtime memory and GPU-arch selection (bound to runtime_config).
    device_resource_group.add_argument(
        "--reserver_runtime_mem_mb",
        env_name="RESERVER_RUNTIME_MEM_MB",
        bind_to=(
            runtime_config,
            "reserve_runtime_mem_mb",
        ),  # Note: spelling difference (reserver -> reserve)
        type=int,
        default=1024,
        help="设备保留的运行时显存大小",
    )
    device_resource_group.add_argument(
        "--specify_gpu_arch",
        env_name="SPECIFY_GPU_ARCH",
        bind_to=(runtime_config, "specify_gpu_arch"),
        type=str,
        default="",
        help="测试时使用的指定GPU架构",
    )
