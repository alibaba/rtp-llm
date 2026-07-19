from rtp_llm.server.server_args.util import str2bool


def init_fmha_group_args(parser, fmha_config):
    ##############################################################################################################
    # FMHA
    ##############################################################################################################
    fmha_group = parser.add_argument_group("FMHA")
    fmha_group.add_argument(
        "--enable_fmha",
        env_name="ENABLE_FMHA",
        bind_to=(fmha_config, "enable_fmha"),
        type=str2bool,
        default=True,
        help="控制是否启用Fused Multi-Head Attention (FMHA) 功能。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_flashinfer_trtllm_gen",
        env_name="ENABLE_FLASHINFER_TRTLLM_GEN",
        bind_to=(fmha_config, "enable_flashinfer_trtllm_gen"),
        type=str2bool,
        default=True,
        help="控制是否启用FlashInfer TRT-LLM Gen Attention实现。支持SM100。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_flashinfer_trt_fmha_v2",
        env_name="ENABLE_FLASHINFER_TRT_FMHA_V2",
        bind_to=(fmha_config, "enable_flashinfer_trt_fmha_v2"),
        type=str2bool,
        default=True,
        help="控制是否启用FlashInfer TRT-LLM FMHA v2连续Prefill。支持SM90和SM12x。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_paged_flashinfer_trt_fmha_v2",
        env_name="ENABLE_PAGED_FLASHINFER_TRT_FMHA_V2",
        bind_to=(fmha_config, "enable_paged_flashinfer_trt_fmha_v2"),
        type=str2bool,
        default=True,
        help="控制是否启用FlashInfer TRT-LLM FMHA v2 Paged Prefill。支持SM90和SM12x。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_open_source_fmha",
        env_name="ENABLE_OPENSOURCE_FMHA",
        bind_to=(fmha_config, "enable_open_source_fmha"),
        type=str2bool,
        default=True,
        help="控制是否启用开源版本的FMHA实现。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_paged_open_source_fmha",
        env_name="ENABLE_PAGED_OPEN_SOURCE_FMHA",
        bind_to=(fmha_config, "enable_paged_open_source_fmha"),
        type=str2bool,
        default=True,
        help="控制是否启用Paged开源版本的FMHA实现。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--disable_flashinfer_native",
        env_name="DISABLE_FLASHINFER_NATIVE",
        bind_to=(fmha_config, "disable_flashinfer_native"),
        type=str2bool,
        default=False,
        help="控制是否禁用FlashInfer Native Attention实现。True表示禁用，False表示启用。",
    )
    fmha_group.add_argument(
        "--enable_xqa",
        env_name="ENABLE_XQA",
        bind_to=(fmha_config, "enable_xqa"),
        type=str2bool,
        default=True,
        help="控制是否开启 xqa 的功能，此功能需要 SM 90 (Hopper) 或更新的 GPU 架构。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--use_aiter_pa",
        env_name="USE_AITER_PA",
        bind_to=(fmha_config, "use_aiter_pa"),
        type=str2bool,
        default=True,
        help="Rocm是否使用AITER Attention",
    )
    fmha_group.add_argument(
        "--use_asm_pa",
        env_name="USE_ASM_PA",
        bind_to=(fmha_config, "use_asm_pa"),
        type=str2bool,
        default=True,
        help="Rocm是否使用AITER ASM Attention",
    )
    fmha_group.add_argument(
        "--use_triton_pa",
        env_name="USE_TRITON_PA",
        bind_to=(fmha_config, "use_triton_pa"),
        type=str2bool,
        default=False,
        help="Rocm decode阶段是否使用Triton PA",
    )
    fmha_group.add_argument(
        "--absorb_opt_len",
        env_name="RTP_LLM_ABSORB_OPT_LEN",
        bind_to=(fmha_config, "absorb_opt_len"),
        type=int,
        default=1024,
        help="控制命中reuse cache后，走absorb attn的最大q_len",
    )
