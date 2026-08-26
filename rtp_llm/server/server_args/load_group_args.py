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
        default=False,
        help="默认关闭；设为 true 后在 pure TP 下预切分已支持的 MoE 权重；不支持的来源或布局回退为全量读取",
    )
    load_group.add_argument(
        "--use_new_loader",
        env_name="USE_NEW_LOADER",
        bind_to=(model_args, "use_new_loader"),
        type=str2bool,
        default=None,
        help=(
            "默认按模型注册表和加载能力选择：已适配且当前配置受支持时使用 "
            "NewLoader，否则回退 legacy loader；显式置真强制 NewLoader，置假强制 "
            "legacy loader"
        ),
    )
    load_group.add_argument(
        "--require_weight_update",
        env_name="REQUIRE_WEIGHT_UPDATE",
        bind_to=(model_args, "require_weight_update"),
        type=str2bool,
        default=False,
        help=(
            "声明部署需要在线 UpdateWeights RPC；自动选路时会使用支持权重热更的 "
            "legacy loader。与 --use_new_loader true 同时设置会在启动期报错"
        ),
    )
    load_group.add_argument(
        "--keep_mla_checkpoint_weights",
        env_name="KEEP_MLA_CHECKPOINT_WEIGHTS",
        bind_to=(load_config, "keep_mla_checkpoint_weights"),
        type=str2bool,
        default=False,
        help=(
            "需先启用 newloader；对 DeepSeek/Kimi MLA 生效：保留已转换为运行时 "
            "布局的 checkpoint 权重。会增加显存占用并减少 KV cache 可用块，"
            "仅用于调试"
        ),
    )
