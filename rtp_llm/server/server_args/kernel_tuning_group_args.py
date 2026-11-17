from rtp_llm.server.server_args.util import str2bool


def init_kernel_tuning_group_args(parser, kernel_tuning_config):
    ##############################################################################################################
    # Kernel Tuning Configuration
    ##############################################################################################################
    kernel_tuning_group = parser.add_argument_group("Kernel Tuning Configuration")
    kernel_tuning_group.add_argument(
        "--rocm_fp8_moe_deterministic_reduce",
        env_name="RTP_LLM_ROCM_FP8_MOE_DETERMINISTIC_REDUCE",
        bind_to=(kernel_tuning_config, "rocm_fp8_moe_deterministic_reduce"),
        type=str2bool,
        default=False,
        help="开启确定性 ROCm FP8-per-channel MoE 路径（gfx942/CU80 限定形状）：固定顺序 FP32 归约替代 aiter 原子散射；"
        "权重形状静态匹配而运行时不满足要求时在模型加载期失败关闭；"
        "ft-style 权重需用当前版本 weights_convert 重新导出（index 记录 preshuffle 标记）",
    )
    kernel_tuning_group.add_argument(
        "--rocm_fp8_moe_deterministic_max_tokens",
        env_name="RTP_LLM_ROCM_FP8_MOE_DETERMINISTIC_MAX_TOKENS",
        bind_to=(kernel_tuning_config, "rocm_fp8_moe_deterministic_max_tokens"),
        type=int,
        default=128,
        help="确定性 ROCm FP8 MoE 路径单次前向允许的最大 token 数，超过该值的请求回退原 aiter 路径",
    )
    kernel_tuning_group.add_argument(
        "--aiter_fmoe_config_paths",
        env_name="AITER_CONFIG_FMOE",
        bind_to=(kernel_tuning_config, "aiter_fmoe_config_paths"),
        type=str,
        default="",
        help="显式指定 aiter FMoE tuning 配置文件列表（os.pathsep 分隔，须包含 RTP overlay 且不含重复/冲突行）；"
        "校验通过后 RTP 仍会写入 AITER_CONFIG_FMOE 传给 aiter pip 包（该包固定只认此环境变量）；"
        "留空时自动组合 aiter 自带配置与 RTP overlay",
    )
