from rtp_llm.server.server_args.util import str2bool


def init_dsv4_warmup_group_args(parser, dsv4_warmup_config):
    dsv4_warmup_group = parser.add_argument_group("DeepSeek V4 Warmup Configuration")
    dsv4_warmup_group.add_argument(
        "--dsv4_startup_real_warmup",
        env_name="DSV4_STARTUP_REAL_WARMUP",
        bind_to=dsv4_warmup_config.startup_real_warmup,
        type=str,
        default="auto",
        help="DeepSeek V4 real request warmup policy: auto, force, or off.",
    )
    dsv4_warmup_group.add_argument(
        "--dsv4_startup_real_warmup_timeout_s",
        env_name="DSV4_STARTUP_REAL_WARMUP_TIMEOUT_S",
        bind_to=dsv4_warmup_config.startup_real_warmup_timeout_s,
        type=float,
        default=600.0,
        help="Timeout in seconds for each DeepSeek V4 startup real warmup request.",
    )
    dsv4_warmup_group.add_argument(
        "--dsv4_startup_real_warmup_token_lens",
        env_name="DSV4_STARTUP_REAL_WARMUP_TOKEN_LENS",
        bind_to=dsv4_warmup_config.startup_real_warmup_token_lens,
        type=str,
        default="",
        help=(
            "Comma separated token lengths for DeepSeek V4 startup real warmup. "
            "Empty keeps the default powers-of-two sweep through max_seq_len."
        ),
    )
    dsv4_warmup_group.add_argument(
        "--dsv4_prewarm_flash_mla_swa",
        env_name="DSV4_PREWARM_FLASH_MLA_SWA",
        bind_to=dsv4_warmup_config.prewarm_flash_mla_swa,
        type=str2bool,
        default=True,
        help="Prewarm the DeepSeek V4 FlashMLA SWA kv_full path before CUDA graph capture.",
    )
    dsv4_warmup_group.add_argument(
        "--dsv4_prewarm_mega_moe",
        env_name="DSV4_PREWARM_MEGA_MOE",
        bind_to=dsv4_warmup_config.prewarm_mega_moe,
        type=str2bool,
        default=True,
        help="Prewarm the DeepSeek V4 MegaMoE path before CUDA graph capture.",
    )
    dsv4_warmup_group.add_argument(
        "--dsv4_mega_moe_jit_warmup",
        env_name="DSV4_MEGA_MOE_JIT_WARMUP",
        bind_to=dsv4_warmup_config.mega_moe_jit_warmup,
        type=str2bool,
        default=True,
        help="Run DeepSeek V4 MegaMoE JIT warmup during model initialization.",
    )
    dsv4_warmup_group.add_argument(
        "--dsv4_mega_moe_jit_warmup_tokens",
        env_name="DSV4_MEGA_MOE_JIT_WARMUP_TOKENS",
        bind_to=dsv4_warmup_config.mega_moe_jit_warmup_tokens,
        type=str,
        default="",
        help="Comma separated MegaMoE token buckets to JIT warm up. Empty keeps auto buckets.",
    )
    dsv4_warmup_group.add_argument(
        "--dsv4_kernel_jit_warmup",
        env_name="DSV4_KERNEL_JIT_WARMUP",
        bind_to=dsv4_warmup_config.kernel_jit_warmup,
        type=str2bool,
        default=True,
        help="Run DeepSeek V4 startup JIT warmup for branch, DenseGEMM, MHC and logits kernels.",
    )
    dsv4_warmup_group.add_argument(
        "--dsv4_dense_gemm_warmup_max_m",
        env_name="DSV4_DENSE_GEMM_WARMUP_MAX_M",
        bind_to=dsv4_warmup_config.dense_gemm_warmup_max_m,
        type=int,
        default=0,
        help="Optional positive cap for DeepSeek V4 DenseGEMM warmup M. 0 keeps the runtime-derived bound.",
    )
