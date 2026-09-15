from rtp_llm.server.server_args.util import nonnegative_int, str2bool


def init_load_group_args(parser, load_config, model_args):
    ##############################################################################################################
    # Load Configuration
    ##############################################################################################################
    load_group = parser.add_argument_group("Load Configuration")
    load_group.add_argument(
        "--load_method",
        env_name="LOAD_METHOD",
        bind_to=(load_config, "load_method"),
        type=str,
        default="auto",
        help="模型权重加载方法",
    )
    load_group.add_argument(
        "--fastsafetensors_reserve_mb",
        env_name="RTP_FASTSAFETENSORS_RESERVE_MB",
        bind_to=(load_config, "fastsafetensors_reserve_mb"),
        type=nonnegative_int,
        default=2048,
        help="AUTO 选择 FastSafeTensors 时额外预留的显存（MiB，非负整数，默认 2048）；0 关闭此额外预留，不影响显式加载模式",
    )
    load_group.add_argument(
        "--force_cpu_load_weights",
        env_name="FORCE_CPU_LOAD_WEIGHTS",
        bind_to=(load_config, "force_cpu_load_weights"),
        type=str2bool,
        default=False,
        help="强制在CPU上加载权重，用于显存不足的场景",
    )
