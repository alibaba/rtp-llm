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
        "--loader_drop_page_cache",
        env_name="LOADER_DROP_PAGE_CACHE",
        bind_to=(load_config, "loader_drop_page_cache"),
        type=str2bool,
        default=True,
        help="权重加载后按阈值释放 shard page cache（POSIX_FADV_DONTNEED）；生产默认开，"
        "作为共享 worker 显存压力下的 OOM 逃生门可关",
    )
    load_group.add_argument(
        "--loader_drop_page_cache_threshold_mb",
        env_name="LOADER_DROP_PAGE_CACHE_THRESHOLD_MB",
        bind_to=(load_config, "loader_drop_page_cache_threshold_mb"),
        type=int,
        default=2048,
        help="单 shard 累计读取多少 MB 后执行一次 page cache 释放",
    )
    load_group.add_argument(
        "--enable_resident_weights",
        env_name="ENABLE_RESIDENT_WEIGHTS",
        bind_to=(load_config.resident_weights_config, "enable"),
        type=str2bool,
        default=False,
        help="常驻加载并变换后的 GPU 权重，加速兼容环境下的重启；影响权重或布局的代码、依赖、checkpoint 变更需先 drop/stop 旧记录，配置指纹不检测所有变化",
    )
    load_group.add_argument(
        "--verify_resident_weights",
        env_name="VERIFY_RESIDENT_WEIGHTS",
        bind_to=(load_config.resident_weights_config, "verify"),
        type=str2bool,
        default=False,
        help="配合 --enable_resident_weights：本次照常从 ckpt 加载，逐个权重与已常驻的校验和比对并报告差异，然后替换常驻记录",
    )
