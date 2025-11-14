from rtp_llm.server.server_args.util import str2bool


def init_device_resource_group_args(parser, device_resource_config, runtime_config):
    ##############################################################################################################
    # 设备和资源管理
    ##############################################################################################################
    device_resource_group = parser.add_argument_group("设备和资源管理")

    device_resource_group.add_argument(
        "--device_reserve_memory_bytes",
        env_name="DEVICE_RESERVE_MEMORY_BYTES",
        bind_to=(device_resource_config, 'device_reserve_memory_bytes'),
        type=int,
        default=-1024 * 1024 * 1024,  # -1GB, prevent oom
        help="指定在GPU设备上预留的内存量（单位：字节）。此内存不会被常规操作使用，可用于应对突发需求或特定驱动/内核开销。",
    )

    device_resource_group.add_argument(
        "--host_reserve_memory_bytes",
        env_name="HOST_RESERVE_MEMORY_BYTES",
        bind_to=(device_resource_config, 'host_reserve_memory_bytes'),
        type=int,
        default=4 * 1024 * 1024 * 1024,  # 4GB
        help="指定在主机（CPU）上预留的内存量（单位：字节）。此内存不会被常规操作使用。默认为 4GB。",
    )

    device_resource_group.add_argument(
        "--overlap_math_sm_count",
        env_name="OVERLAP_MATH_SM_COUNT",
        bind_to=(device_resource_config, 'overlap_math_sm_count'),
        type=int,
        default=0,
        help="指定用于计算与通信重叠优化的 SM 数量。",
    )

    device_resource_group.add_argument(
        "--overlap_comm_type",
        env_name="OVERLAP_COMM_TYPE",
        bind_to=(device_resource_config, 'overlap_comm_type'),
        type=int,
        default=0,
        help="指定计算与通信重叠的策略类型。0: 禁止重叠，串行执行；1: 轻量级重叠，平衡性能与复杂度；2: 深度优化，通过多流和事件管理实现更大吞吐量。",
    )

    device_resource_group.add_argument(
        "--m_split",
        env_name="M_SPLIT",
        bind_to=(device_resource_config, 'm_split'),
        type=int,
        default=0,
        help="为特定设备操作设置 M_SPLIT 参数值。`0` 通常表示使用默认或不拆分。",
    )

    device_resource_group.add_argument(
        "--enable_comm_overlap",
        env_name="ENABLE_COMM_OVERLAP",
        bind_to=(device_resource_config, 'enable_comm_overlap'),
        type=str2bool,
        default=None,
        help="设置为 `True` 以启用计算与通信之间的重叠执行，旨在提高设备利用率和吞吐量。",
    )

    device_resource_group.add_argument(
        "--enable_layer_micro_batch",
        env_name="ENABLE_LAYER_MICRO_BATCH",
        bind_to=(device_resource_config, 'enable_layer_micro_batch'),
        type=int,
        default=0,
        help="控制是否启用层级的 micro-batching。",
    )

    device_resource_group.add_argument(
        "--not_use_default_stream",
        env_name="NOT_USE_DEFAULT_STREAM",
        bind_to=(device_resource_config, 'not_use_default_stream'),
        type=str2bool,
        default=False,
        help="控制 PyTorch 操作不使用标准的默认 CUDA 流。",
    )

    # Fields merged from PyDeviceResourceConfig to RuntimeConfig
    device_resource_group.add_argument(
        "--reserver_runtime_mem_mb",
        env_name="RESERVER_RUNTIME_MEM_MB",
        bind_to=(runtime_config, 'reserve_runtime_mem_mb'),  # Note: spelling difference (reserver -> reserve)
        type=int,
        default=1024,
        help="设备保留的运行时显存大小",
    )
    device_resource_group.add_argument(
        "--specify_gpu_arch",
        env_name="SPECIFY_GPU_ARCH",
        bind_to=(runtime_config, 'specify_gpu_arch'),
        type=str,
        default="",
        help="测试时使用的指定GPU架构",
    )
    device_resource_group.add_argument(
        "--acext_gemm_config_dir",
        env_name="ACEXT_GEMM_CONFIG_DIR",
        bind_to=(runtime_config, 'acext_gemm_config_dir'),
        type=str,
        default="",
        help="ACEXT GEMM配置目录",
    )
