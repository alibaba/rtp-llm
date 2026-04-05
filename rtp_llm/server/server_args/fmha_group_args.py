from rtp_llm.server.server_args.util import str2bool


def init_fmha_group_args(parser, fmha_config):
    ##############################################################################################################
    # FMHA
    ##############################################################################################################
    fmha_group = parser.add_argument_group("FMHA")

    # New string-based attention backend selection
    fmha_group.add_argument(
        "--attn_backend",
        env_name="ATTN_BACKEND",
        bind_to=(fmha_config, "attn_backend"),
        type=str,
        default="auto",
        help="Attention backend selection. 'auto' = priority-ordered auto-selection (default). "
             "'none' = disable all attention. Or a specific backend name: "
             "trt, trt_paged, flashinfer, py_flashinfer, py_flashinfer_paged, xqa, "
             "trtllm_gen, trtllm_spec, headwise, headwise_fp8, "
             "aiter_asm, aiter, aiter_triton, aiter_paged, cp_flashinfer, "
             "flashinfer_mla, sparse_mla.",
    )
    fmha_group.add_argument(
        "--prefill_attn_backend",
        env_name="PREFILL_ATTN_BACKEND",
        bind_to=(fmha_config, "prefill_attn_backend"),
        type=str,
        default="",
        help="Override attention backend for prefill stage (including chunked prefill "
             "and speculative verify). Empty = use --attn_backend.",
    )
    fmha_group.add_argument(
        "--decode_attn_backend",
        env_name="DECODE_ATTN_BACKEND",
        bind_to=(fmha_config, "decode_attn_backend"),
        type=str,
        default="",
        help="Override attention backend for decode stage. Empty = use --attn_backend.",
    )
    fmha_group.add_argument(
        "--disable_attn_backends",
        env_name="DISABLE_ATTN_BACKENDS",
        bind_to=(fmha_config, "disable_attn_backends"),
        type=str,
        default="",
        help="Comma-separated list of attention backend NAMEs to disable. "
             "Example: --disable_attn_backends=flashinfer,xqa",
    )

    # Legacy boolean flags (kept for backward compatibility)
    fmha_group.add_argument(
        "--enable_fmha",
        env_name="ENABLE_FMHA",
        bind_to=(fmha_config, "enable_fmha"),
        type=str2bool,
        default=True,
        help="控制是否启用Fused Multi-Head Attention (FMHA) 功能。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_trt_fmha",
        env_name="ENABLE_TRT_FMHA",
        bind_to=(fmha_config, "enable_trt_fmha"),
        type=str2bool,
        default=True,
        help="控制是否启用经TensorRT(V2版本)优化的FMHA功能。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_paged_trt_fmha",
        env_name="ENABLE_PAGED_TRT_FMHA",
        bind_to=(fmha_config, "enable_paged_trt_fmha"),
        type=str2bool,
        default=True,
        help="控制是否启用Paged TensorRT FMHA功能。可选值: True (启用), False (禁用)。",
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
        "--disable_flash_infer",
        env_name="DISABLE_FLASH_INFER",
        bind_to=(fmha_config, "disable_flash_infer"),
        type=str2bool,
        default=False,
        help="控制是否禁用FlashInfer Attention机制。设置为 True 启用, False 禁用。",
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
        "--absorb_opt_len",
        env_name="RTP_LLM_ABSORB_OPT_LEN",
        bind_to=(fmha_config, "absorb_opt_len"),
        type=int,
        default=1024,
        help="控制命中reuse cache后，走absorb attn的最大q_len",
    )
