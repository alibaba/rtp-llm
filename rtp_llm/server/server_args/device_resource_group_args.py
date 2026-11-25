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
        "--torch_cuda_pool_prealloc_mb",
        env_name="TORCH_CUDA_POOL_PREALLOC_MB",
        bind_to=(runtime_config, "torch_cuda_pool_prealloc_mb"),
        type=int,
        default=-1,
        help="Prefill CUDA pool preallocation limit in MiB (-1 for available memory, 0 to disable).",
    )
    device_resource_group.add_argument(
        "--torch_cuda_pool_prealloc_safety_mb",
        env_name="TORCH_CUDA_POOL_PREALLOC_SAFETY_MB",
        bind_to=(runtime_config, "torch_cuda_pool_prealloc_safety_mb"),
        type=int,
        default=8192,
        help="Free GPU memory in MiB to leave outside the preallocated Prefill pool.",
    )
    device_resource_group.add_argument(
        "--specify_gpu_arch",
        env_name="SPECIFY_GPU_ARCH",
        bind_to=(runtime_config, "specify_gpu_arch"),
        type=str,
        default="",
        help="测试时使用的指定GPU架构",
    )
