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
        "--keep_mla_checkpoint_weights",
        env_name="KEEP_MLA_CHECKPOINT_WEIGHTS",
        bind_to=(load_config, "keep_mla_checkpoint_weights"),
        type=str2bool,
        default=False,
        help=(
            "仅对 DeepSeek MLA newloader 生效：保留已转换为运行时布局的 "
            "checkpoint 权重。会增加显存占用并减少 KV cache 可用块，仅用于调试"
        ),
    )
