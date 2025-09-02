from rtp_llm.server.server_args.util import str2bool


def init_fmha_group_args(parser):
    ##############################################################################################################
    # FMHA
    ##############################################################################################################
    fmha_group = parser.add_argument_group("FMHA")
    fmha_group.add_argument(
        "--enable_fmha",
        env_name="ENABLE_FMHA",
        type=str2bool,
        default=True,
        help="控制是否启用Fused Multi-Head Attention (FMHA) 功能。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_trt_fmha",
        env_name="ENABLE_TRT_FMHA",
        type=str2bool,
        default=True,
        help="控制是否启用经TensorRT(V2版本)优化的FMHA功能。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_paged_trt_fmha",
        env_name="ENABLE_PAGED_TRT_FMHA",
        type=str2bool,
        default=True,
        help="控制是否启用Paged TensorRT FMHA功能。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_open_source_fmha",
        env_name="ENABLE_OPENSOURCE_FMHA",
        type=str2bool,
        default=True,
        help="控制是否启用开源版本的FMHA实现。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_paged_open_source_fmha",
        env_name="ENABLE_PAGED_OPEN_SOURCE_FMHA",
        type=str2bool,
        default=True,
        help="控制是否启用Paged开源版本的FMHA实现。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_trtv1_fmha",
        env_name="ENABLE_TRTV1_FMHA",
        type=str2bool,
        default=True,
        help="控制是否启用TRTv1风格的FMHA功能。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--fmha_perf_instrument",
        env_name="FMHA_PERF_INSTRUMENT",
        type=str2bool,
        default=False,
        help="控制是否为FMHA启用NVTX性能分析。设置为 True 启用, False 禁用。",
    )
    fmha_group.add_argument(
        "--fmha_show_params",
        env_name="FMHA_SHOW_PARAMS",
        type=str2bool,
        default=False,
        help="控制是否显示FMHA的参数信息。设置为 True 启用, False 禁用。",
    )
    fmha_group.add_argument(
        "--disable_flash_infer",
        env_name="DISABLE_FLASH_INFER",
        type=str2bool,
        default=False,
        help="控制是否禁用FlashInfer Attention机制。设置为 True 启用, False 禁用。",
    )
    fmha_group.add_argument(
        "--enable_xqa",
        env_name="ENABLE_XQA",
        type=str2bool,
        default=True,
        help="控制是否开启 xqa 的功能，此功能需要 SM 90 (Hopper) 或更新的 GPU 架构。可选值: True (启用), False (禁用)。",
    )
