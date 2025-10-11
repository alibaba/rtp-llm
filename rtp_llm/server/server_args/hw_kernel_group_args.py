from rtp_llm.server.server_args.util import str2bool


def init_hw_kernel_group_args(parser):
    ##############################################################################################################
    # 硬件/Kernel 特定优化
    ##############################################################################################################
    hw_kernel_group = parser.add_argument_group("硬件/Kernel 特定优化")

    hw_kernel_group.add_argument(
        "--enable_cuda_graph",
        env_name="ENABLE_CUDA_GRAPH",
        type=str2bool,
        default=False,
        help="系统是否允许使用Cuda Graph",
    )

    hw_kernel_group.add_argument(
        "--enable_cuda_graph_debug_mode",
        env_name="ENABLE_CUDA_GRAPH_DEBUG_MODE",
        type=str2bool,
        default=False,
        help="系统是否允许使用Cuda Graph开启Debug模式来生成可视化文件",
    )

    hw_kernel_group.add_argument(
        "--enable_native_cuda_graph",
        env_name="ENABLE_NATIVE_CUDA_GRAPH",
        type=str2bool,
        default=False,
        help="系统是否允许在C++后端使用Cuda Graph",
    )

    hw_kernel_group.add_argument(
        "--num_native_cuda_graph",
        env_name="NUM_NATIVE_CUDA_GRAPH",
        type=int,
        default=200,
        help="C++后端缓存Cuda Graph数量",
    )

    hw_kernel_group.add_argument(
        "--deep_gemm_num_sm",
        env_name="DEEP_GEMM_NUM_SM",
        type=int,
        default=None,
        help="指定 DeepGEMM 使用的 SM (Streaming Multiprocessor) 数量。如果设置，此值将覆盖自动检测的数量。",
    )

    hw_kernel_group.add_argument(
        "--arm_gemm_use_kai",
        env_name="ARM_GEMM_USE_KAI",
        type=str2bool,
        default=False,
        help="设置为 `True` 时，为 ARM GEMM 操作启用 KleidiAI 支持。这可能影响权重处理和计算性能。",
    )

    hw_kernel_group.add_argument(
        "--enable_stable_scatter_add",
        env_name="ENABLE_STABLE_SCATTER_ADD",
        type=str2bool,
        default=False,
        help="控制是否启用稳定的 scatter add 操作。",
    )

    hw_kernel_group.add_argument(
        "--enable_multi_block_mode",
        env_name="ENABLE_MULTI_BLOCK_MODE",
        type=str2bool,
        default=True,
        help="控制是否为 Multi-Head Attention (MMHA) 启用 multi-block 模式。设置为 'ON' 启用，'OFF' 禁用。",
    )

    hw_kernel_group.add_argument(
        "--rocm_hipblaslt_config",
        env_name="ROCM_HIPBLASLT_CONFIG",
        type=str,
        default="gemm_config.csv",
        help="指定 hipBLASLt GEMM 配置文件的路径。此文件用于优化 ROCm平台上的 GEMM 操作。",
    )

    hw_kernel_group.add_argument(
        "--ft_disable_custom_ar",
        env_name="FT_DISABLE_CUSTOM_AR",
        type=str2bool,
        default=None,
        help="设置为 `True` 时，禁用自定义的 AllReduce (AR) 实现，可能回退到标准库（如 NCCL）的 AllReduce。",
    )

    hw_kernel_group.add_argument(
        "--use_aiter_pa",
        env_name="USE_AITER_PA",
        type=str2bool,
        default=True,
        help="Rocm是否使用AITER Attention",
    )

    hw_kernel_group.add_argument(
        "--use_asm_pa",
        env_name="USE_ASM_PA",
        type=str2bool,
        default=True,
        help="Rocm是否使用AITER ASM Attention",
    )

    hw_kernel_group.add_argument(
        "--use_swizzleA",
        env_name="USE_SWIZZLEA",
        type=str2bool,
        default=False,
        help="hipBLASLt GEMM 是否使用 swizzle",
    )
