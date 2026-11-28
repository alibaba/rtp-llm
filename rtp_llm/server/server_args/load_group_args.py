from rtp_llm.server.server_args.util import str2bool


def init_load_group_args(parser, load_config, model_args):
    ##############################################################################################################
    # Load Configuration
    ##############################################################################################################
    load_group = parser.add_argument_group("Load Configuration")
    load_group.add_argument(
        "--fastsafetensors_config_json",
        env_name="FASTSAFETENSORS_CONFIG_JSON",
        bind_to=(load_config, "fastsafetensors_config_json"),
        type=str,
        default=None,
        help="FastSafeTensors JSON configuration; takes precedence over config path",
    )
    load_group.add_argument(
        "--fastsafetensors_config",
        env_name="FASTSAFETENSORS_CONFIG",
        bind_to=(load_config, "fastsafetensors_config"),
        type=str,
        default=None,
        help="FastSafeTensors configuration file path",
    )
    load_group.add_argument(
        "--load_method",
        env_name="LOAD_METHOD",
        bind_to=(load_config, "load_method"),
        type=str,
        default="auto",
        help=(
            "NewLoader 权重读取：auto 在加速卡上优先 fastsafetensors，"
            "前置条件不满足时选择 scratch；显式 fastsafetensors 不回退。"
        ),
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
        help="旧 ROCm safetensors 读者提前关闭已读 shard；新 scratch 读者始终逐 shard 关闭，FST 不受影响",
    )
    load_group.add_argument(
        "--moe_pure_tp_preshard",
        env_name="MOE_PURE_TP_PRESHARD",
        bind_to=(load_config, "moe_pure_tp_preshard"),
        type=str2bool,
        default=False,
        help="默认关闭；pure TP 下 Qwen3-Next/Qwen3.5 MoE scratch 可只读本 rank 专家切片；其他布局沿用普通读取，FST 不受影响",
    )
    load_group.add_argument(
        "--loader_drop_page_cache",
        env_name="LOADER_DROP_PAGE_CACHE",
        bind_to=(load_config, "loader_drop_page_cache"),
        type=str2bool,
        default=True,
        help="scratch 映射 safetensors 时建议 Linux 回收已消费的文件缓存；不保证回收，也不释放显存，不控制 FST",
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
        "--resident_weights_mode",
        env_name="RESIDENT_WEIGHTS_MODE",
        bind_to=(load_config.resident_weights_config, "mode"),
        type=str,
        choices=["none", "dev", "prod"],
        default="none",
        help="常驻加载并变换后的权重，加速兼容环境下的重启。none：关闭，不使用也不连接 store；"
        "dev：store 独立于引擎存活，任何退出都保留，用户执行 store drop/stop 或随容器回收；"
        "prod：使用本实例专属 store（rtp_llm_resident_weights.<uid>.<START_PORT>，或 RTP_LLM_RESIDENT_STORE），"
        "start_server 正常退出时停掉它，失败退出时保留。改成 none 不会释放已有记录。"
        "影响权重或布局的代码、依赖、checkpoint 变更需先 drop/stop 旧记录，配置指纹不检测所有变化",
    )
    load_group.add_argument(
        "--resident_weights_offload_after_s",
        env_name="RESIDENT_WEIGHTS_OFFLOAD_AFTER_S",
        bind_to=(load_config.resident_weights_config, "offload_after_s"),
        type=float,
        default=None,
        help="常驻记录无人使用（没有引擎持有）多少秒后，把其中的显存权重搬到主机内存，下次 attach 时再搬回显存。"
        "未设置时取 mode 默认值（dev 600，prod -1），-1 表示关闭（不搬），其他 <= 0 非法。只在本次启动 store 时生效",
    )
    load_group.add_argument(
        "--resident_weights_idle_ttl_s",
        env_name="RESIDENT_WEIGHTS_IDLE_TTL_S",
        bind_to=(load_config.resident_weights_config, "idle_ttl_s"),
        type=float,
        default=None,
        help="常驻记录卸载到主机内存后（或关闭 offload 时在显存上）无人使用多少秒后自动丢弃。"
        "未设置时取 mode 默认值（dev 3600，prod -1），-1 表示关闭（永不丢弃），其他 <= 0 非法。只在本次启动 store 时生效",
    )
    load_group.add_argument(
        "--resident_weights_offload_host_reserve_ratio",
        env_name="RESIDENT_WEIGHTS_OFFLOAD_HOST_RESERVE_RATIO",
        bind_to=(load_config.resident_weights_config, "offload_host_reserve_ratio"),
        type=float,
        default=0.2,
        help="空闲 offload 到主机内存后，至少保留整机/容器总主机内存的比例（[0, 1)，默认 0.2 即 20%%）；"
        "与 --resident_weights_offload_host_reserve_min_gb 取较大值。只在本次启动 store 时生效",
    )
    load_group.add_argument(
        "--resident_weights_offload_host_reserve_min_gb",
        env_name="RESIDENT_WEIGHTS_OFFLOAD_HOST_RESERVE_MIN_GB",
        bind_to=(load_config.resident_weights_config, "offload_host_reserve_min_gb"),
        type=float,
        default=0.0,
        help="空闲 offload 到主机内存后，至少保留的主机内存绝对下限（GiB，>= 0，默认 0 表示纯按比例）；"
        "与 --resident_weights_offload_host_reserve_ratio 取较大值。只在本次启动 store 时生效",
    )
    load_group.add_argument(
        "--verify_resident_weights",
        env_name="VERIFY_RESIDENT_WEIGHTS",
        bind_to=(load_config.resident_weights_config, "verify"),
        type=str2bool,
        default=False,
        help="配合 --resident_weights_mode dev/prod：本次照常从 ckpt 加载，逐个权重与已常驻的校验和比对并报告差异，然后替换常驻记录",
    )
