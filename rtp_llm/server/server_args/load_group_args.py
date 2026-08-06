from rtp_llm.server.server_args.util import str2bool


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
        "--force_cpu_load_weights",
        env_name="FORCE_CPU_LOAD_WEIGHTS",
        bind_to=(load_config, "force_cpu_load_weights"),
        type=str2bool,
        default=False,
        help="强制在CPU上加载权重，用于显存不足的场景",
    )
    load_group.add_argument(
        "--loader_recycle_handles",
        env_name="LOADER_RECYCLE_HANDLES",
        bind_to=(load_config, "loader_recycle_handles"),
        type=str2bool,
        default=True,
        help="ROCm + safetensors 主模型加载时回收已读 shard handle；置假关闭",
    )
    load_group.add_argument(
        "--moe_pure_tp_preshard",
        env_name="MOE_PURE_TP_PRESHARD",
        bind_to=(load_config, "moe_pure_tp_preshard"),
        type=str2bool,
        default=True,
        help="pure TP 下预切分已支持的 MoE 权重；置假回滚为全量读取",
    )
