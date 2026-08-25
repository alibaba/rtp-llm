from rtp_llm.server.server_args.util import str2bool


def init_kv_cache_group_args(parser, kv_cache_config):
    ##############################################################################################################
    # KV Cache 相关配置
    ##############################################################################################################
    kv_cache_group = parser.add_argument_group("KVCache")
    kv_cache_group.add_argument(
        "--reuse_cache",
        env_name="REUSE_CACHE",
        bind_to=(kv_cache_config, "reuse_cache"),
        type=str2bool,
        default=False,
        help="控制是否激活KV Cache的重用机制, 默认开启显存重用, 其他cache重用需手动开启。设置为 True 启用 , False 关闭",
    )
    kv_cache_group.add_argument(
        "--enable_device_cache",
        env_name="ENABLE_DEVICE_CACHE",
        bind_to=(kv_cache_config, "enable_device_cache"),
        type=str2bool,
        default=True,
        help="控制是否启用显存Cache的重用机制, 默认开启。设置为 True 启用 , False 关闭",
    )
    kv_cache_group.add_argument(
        "--reserve_block_ratio",
        env_name="RESERVE_BLOCK_RATIO",
        bind_to=(kv_cache_config, "reserve_block_ratio"),
        type=int,
        default=5,
        help="KV cache 预留 block 的百分比（仅对首次分配/空 batch_kv_resource 生效），用于保护正在运行的 stream 后续增量申请。",
    )
    kv_cache_group.add_argument(
        "--enable_remote_cache",
        env_name="ENABLE_REMOTE_CACHE",
        bind_to=(kv_cache_config, "enable_remote_cache"),
        type=str2bool,
        default=False,
        help="控制是否启用Remote Cache的机制。设置为 True 启用 , False 关闭",
    )
    kv_cache_group.add_argument(
        "--multi_task_prompt",
        env_name="MULTI_TASK_PROMPT",
        bind_to=(kv_cache_config, "multi_task_prompt"),
        type=str,
        default=None,
        help="指定一个多任务提示（multi-task prompt），为一个路径，系统会读取路径指定的多任务json文件。默认为空",
    )
    kv_cache_group.add_argument(
        "--multi_task_prompt_str",
        env_name="MULTI_TASK_PROMPT_STR",
        bind_to=(kv_cache_config, "multi_task_prompt_str"),
        type=str,
        default=None,
        help="指定一个多任务提示字符串（multi-task prompt string），为多任务纯json字符串，类似于系统提示词。默认为空 ",
    )
    kv_cache_group.add_argument(
        "--fp8_kv_cache",
        env_name="FP8_KV_CACHE",
        bind_to=(kv_cache_config, "fp8_kv_cache"),
        type=int,
        help="是否开启FP8的KV_CACHE",
    )
    # compatible with old version
    kv_cache_group.add_argument(
        "--blockwise_use_fp8_kv_cache",
        env_name="BLOCKWISE_USE_FP8_KV_CACHE",
        bind_to=(kv_cache_config, "fp8_kv_cache"),
        type=int,
        help="是否开启FP8的KV_CACHE",
    )
    kv_cache_group.add_argument(
        "--kv_cache_mem_mb",
        env_name="KV_CACHE_MEM_MB",
        bind_to=(kv_cache_config, "kv_cache_mem_mb"),
        type=int,
        default=-1,
        help="KV_CACHE的大小",
    )
    kv_cache_group.add_argument(
        "--seq_size_per_block",
        env_name="SEQ_SIZE_PER_BLOCK",
        bind_to=(kv_cache_config, "seq_size_per_block"),
        type=int,
        default=0,
        help="单独一个KV_CACHE的Block里面token的数量, 0表示使用平台默认值(CUDA:64, PPU:256, ROCm:16)",
    )
    kv_cache_group.add_argument(
        "--kernel_seq_size_per_block",
        env_name="KERNEL_SEQ_SIZE_PER_BLOCK",
        bind_to=(kv_cache_config, "kernel_seq_size_per_block"),
        type=int,
        default=0,
        help="Attention算子使用的kernel block大小（token数量）。0表示与seq_size_per_block相同。",
    )
    kv_cache_group.add_argument(
        "--linear_step",
        env_name="LINEAR_STEP",
        bind_to=(kv_cache_config, "linear_step"),
        type=int,
        default=1,
        help="线性注意力（Linear Attention）缓存重用的步长：每隔 linear_step 个 block 额外保留一个 block（>=1）。",
    )
    kv_cache_group.add_argument(
        "--ssm_state_dtype",
        env_name="SSM_STATE_DTYPE",
        bind_to=(kv_cache_config, "ssm_state_dtype"),
        type=str,
        choices=["bf16", "fp32"],
        default="bf16",
        help="线性注意力 SSM state 的数据类型。默认 bf16，可选 fp32 和 bf16。",
    )
    kv_cache_group.add_argument(
        "--test_block_num",
        env_name="TEST_BLOCK_NUM",
        bind_to=(kv_cache_config, "test_block_num"),
        type=int,
        default=0,
        help="在测试时强制指定BLOCK的数量",
    )
    kv_cache_group.add_argument(
        "--enable_host_cache",
        "--enable_memory_cache",
        env_name="ENABLE_HOST_CACHE",
        env_aliases=("ENABLE_MEMORY_CACHE",),
        bind_to=(kv_cache_config, "enable_host_cache"),
        type=str2bool,
        default=False,
        help="Host KVCache 开关。开启时必须通过 HOST_CACHE_SIZE_MB 设置容量",
    )
    kv_cache_group.add_argument(
        "--enable_host_cache_pinned",
        env_name="ENABLE_HOST_CACHE_PINNED",
        bind_to=(kv_cache_config, "enable_host_cache_pinned"),
        type=str2bool,
        default=True,
        help="Host KVCache 是否使用 pinned memory。",
    )
    kv_cache_group.add_argument(
        "--enable_disk_cache",
        "--enable_memory_cache_disk",
        env_name="ENABLE_DISK_CACHE",
        env_aliases=("ENABLE_MEMORY_CACHE_DISK",),
        bind_to=(kv_cache_config, "enable_disk_cache"),
        type=str2bool,
        default=False,
        help="磁盘 KVCache (L3) 开关。开启时必须配置 DISK_CACHE_SIZE_MB 和 "
        "DISK_CACHE_PATHS。",
    )
    kv_cache_group.add_argument(
        "--host_cache_size_mb",
        "--memory_cache_size_mb",
        env_name="HOST_CACHE_SIZE_MB",
        env_aliases=("MEMORY_CACHE_SIZE_MB",),
        bind_to=(kv_cache_config, "host_cache_size_mb"),
        type=int,
        default=0,
        help="单个 rank 的 Host KVCache 容量，单位 MB",
    )
    kv_cache_group.add_argument(
        "--host_cache_sync_timeout_ms",
        "--memory_cache_sync_timeout_ms",
        env_name="HOST_CACHE_SYNC_TIMEOUT_MS",
        env_aliases=("MEMORY_CACHE_SYNC_TIMEOUT_MS",),
        bind_to=(kv_cache_config, "host_cache_sync_timeout_ms"),
        type=int,
        default=10000,
        help="Host KVCache 多 TP 同步超时，单位毫秒",
    )
    kv_cache_group.add_argument(
        "--disk_cache_paths",
        "--memory_cache_disk_paths",
        env_name="DISK_CACHE_PATHS",
        env_aliases=("MEMORY_CACHE_DISK_PATHS",),
        bind_to=(kv_cache_config, "disk_cache_paths"),
        type=str,
        default="",
        help="磁盘 KV cache 路径；多个路径的格式由磁盘 cache 实现解析。",
    )
    kv_cache_group.add_argument(
        "--disk_cache_size_mb",
        "--memory_cache_disk_size_mb",
        env_name="DISK_CACHE_SIZE_MB",
        env_aliases=("MEMORY_CACHE_DISK_SIZE_MB",),
        bind_to=(kv_cache_config, "disk_cache_size_mb"),
        type=int,
        default=0,
        help="单个 rank 的磁盘 KV cache 容量，单位 MB。",
    )
    kv_cache_group.add_argument(
        "--disk_cache_buffered_io",
        "--memory_cache_disk_buffered_io",
        env_name="DISK_CACHE_BUFFERED_IO",
        env_aliases=("MEMORY_CACHE_DISK_BUFFERED_IO",),
        bind_to=(kv_cache_config, "disk_cache_buffered_io"),
        type=str2bool,
        default=True,
        help="磁盘 KV cache 是否使用 buffered I/O。默认开启。",
    )
    kv_cache_group.add_argument(
        "--disk_cache_sync_timeout_ms",
        "--memory_cache_disk_sync_timeout_ms",
        env_name="DISK_CACHE_SYNC_TIMEOUT_MS",
        env_aliases=("MEMORY_CACHE_DISK_SYNC_TIMEOUT_MS",),
        bind_to=(kv_cache_config, "disk_cache_sync_timeout_ms"),
        type=int,
        default=30000,
        help="磁盘 KV cache 同步超时，单位毫秒。",
    )
    kv_cache_group.add_argument(
        "--disk_cache_staging_block_count",
        env_name="DISK_CACHE_STAGING_BLOCK_COUNT",
        bind_to=(kv_cache_config, "disk_cache_staging_block_count"),
        type=int,
        default=4,
        help="单个 rank Device<->Disk 直传的 Host staging block 数，决定直传并发容量。",
    )
    kv_cache_group.add_argument(
        "--memory_cache_max_descriptors_per_transfer_batch",
        env_name="MEMORY_CACHE_MAX_DESCRIPTORS_PER_TRANSFER_BATCH",
        bind_to=(kv_cache_config, "memory_cache_max_descriptors_per_transfer_batch"),
        type=int,
        default=8,
        help="Device<->Host 单次底层批调用包含的最大 descriptor 数；其他方向默认逐条执行。",
    )
    kv_cache_group.add_argument(
        "--write_cache_sync",
        env_name="WRITE_CACHE_SYNC",
        type=str2bool,
        default=None,
        help="Deprecated compatibility option. BlockTree cache settlement is always coordinated internally.",
    )

    kv_cache_group.add_argument(
        "--block_tree_full_prefix_scan_interval_ms",
        env_name="BLOCK_TREE_FULL_PREFIX_SCAN_INTERVAL_MS",
        bind_to=(kv_cache_config, "block_tree_full_prefix_scan_interval_ms"),
        type=int,
        default=0,
        help="BlockTreeCache FULL 前缀异常扫描周期，单位毫秒；0 表示关闭且不创建线程，非 0 时必须不小于 1000。"
        "单轮扫描节点数和单 cycle 明细上限由 scanner 内部常量固定，不可配置。",
    )

    # Remote connector configuration arguments
    kv_cache_group.add_argument(
        "--reco_enable_vipserver",
        env_name="RECO_ENABLE_VIPSERVER",
        bind_to=(kv_cache_config, "reco_enable_vipserver"),
        type=str2bool,
        default=False,
        help="是否启用kvcm的VIPServer",
    )
    kv_cache_group.add_argument(
        "--reco_vipserver_domain",
        env_name="RECO_VIPSERVER_DOMAIN",
        bind_to=(kv_cache_config, "reco_vipserver_domain"),
        type=str,
        default="",
        help="kvcm VIPServer域名",
    )
    kv_cache_group.add_argument(
        "--reco_server_address",
        env_name="RECO_SERVER_ADDRESS",
        bind_to=(kv_cache_config, "reco_server_address"),
        type=str,
        default="",
        help="kvcm server地址",
    )
    kv_cache_group.add_argument(
        "--reco_instance_group",
        env_name="RECO_INSTANCE_GROUP",
        bind_to=(kv_cache_config, "reco_instance_group"),
        type=str,
        default="default",
        help="instance_group名称",
    )
    kv_cache_group.add_argument(
        "--reco_meta_channel_retry_time",
        env_name="RECO_META_CHANNEL_RETRY_TIME",
        bind_to=(kv_cache_config, "reco_meta_channel_retry_time"),
        type=int,
        default=3,
        help="grpc重试次数",
    )
    kv_cache_group.add_argument(
        "--reco_meta_channel_connection_timeout",
        env_name="RECO_META_CHANNEL_CONNECTION_TIMEOUT",
        bind_to=(kv_cache_config, "reco_meta_channel_connection_timeout"),
        type=int,
        default=6000,
        help="超时时间",
    )
    kv_cache_group.add_argument(
        "--reco_meta_channel_call_timeout",
        env_name="RECO_META_CHANNEL_CALL_TIMEOUT",
        bind_to=(kv_cache_config, "reco_meta_channel_call_timeout"),
        type=int,
        default=1500,
        help="超时时间",
    )
    kv_cache_group.add_argument(
        "--reco_storage_thread_num",
        env_name="RECO_STORAGE_THREAD_NUM",
        bind_to=(kv_cache_config, "reco_storage_thread_num"),
        type=int,
        default=4,
        help="kvcm SdkWrapper中任务处理线程数量",
    )
    kv_cache_group.add_argument(
        "--reco_storage_queue_size",
        env_name="RECO_STORAGE_QUEUE_SIZE",
        bind_to=(kv_cache_config, "reco_storage_queue_size"),
        type=int,
        default=2000,
        help="kvcm SdkWrapper中线程池队列大小",
    )
    kv_cache_group.add_argument(
        "--reco_put_timeout_ms",
        env_name="RECO_PUT_TIMEOUT_MS",
        bind_to=(kv_cache_config, "reco_put_timeout_ms"),
        type=int,
        default=12000,
        help="PUT操作超时时间（毫秒）",
    )
    kv_cache_group.add_argument(
        "--reco_get_timeout_ms",
        env_name="RECO_GET_TIMEOUT_MS",
        bind_to=(kv_cache_config, "reco_get_timeout_ms"),
        type=int,
        default=12000,
        help="GET操作超时时间（毫秒）",
    )
    kv_cache_group.add_argument(
        "--reco_model_sdk_config",
        env_name="RECO_MODEL_SDK_CONFIG",
        bind_to=(kv_cache_config, "reco_model_sdk_config"),
        type=str,
        default='[{"type":"local","sdk_log_level":"DEBUG"}]',
        help="SDK 配置",
    )
    kv_cache_group.add_argument(
        "--reco_model_user_data",
        env_name="RECO_MODEL_USER_DATA",
        bind_to=(kv_cache_config, "reco_model_user_data"),
        type=str,
        default="",
        help="模型用户数据",
    )
    kv_cache_group.add_argument(
        "--reco_model_extra_info",
        env_name="RECO_MODEL_EXTRA_INFO",
        bind_to=(kv_cache_config, "reco_model_extra_info"),
        type=str,
        default="",
        help="模型额外信息",
    )
    kv_cache_group.add_argument(
        "--reco_instance_id_salt",
        env_name="RECO_INSTANCE_ID_SALT",
        bind_to=(kv_cache_config, "reco_instance_id_salt"),
        type=str,
        default="",
        help="实例 ID salt值",
    )
    kv_cache_group.add_argument(
        "--reco_asyncwrapper_thread_num",
        env_name="RECO_ASYNCWRAPPER_THREAD_NUM",
        bind_to=(kv_cache_config, "reco_asyncwrapper_thread_num"),
        type=int,
        default=16,
        help="异步包装器线程数量",
    )
    kv_cache_group.add_argument(
        "--reco_asyncwrapper_queue_size",
        env_name="RECO_ASYNCWRAPPER_QUEUE_SIZE",
        bind_to=(kv_cache_config, "reco_asyncwrapper_queue_size"),
        type=int,
        default=1000,
        help="异步包装器队列大小",
    )
    kv_cache_group.add_argument(
        "--reco_get_broadcast_timeout",
        env_name="RECO_GET_BROADCAST_TIMEOUT",
        bind_to=(kv_cache_config, "reco_get_broadcast_timeout"),
        type=int,
        default=15000,
        help="GET广播超时时间（毫秒）",
    )
    kv_cache_group.add_argument(
        "--reco_put_broadcast_timeout",
        env_name="RECO_PUT_BROADCAST_TIMEOUT",
        bind_to=(kv_cache_config, "reco_put_broadcast_timeout"),
        type=int,
        default=15000,
        help="PUT广播超时时间（毫秒）",
    )
    kv_cache_group.add_argument(
        "--reco_client_config",
        env_name="RECO_CLIENT_CONFIG",
        bind_to=(kv_cache_config, "reco_client_config"),
        type=str,
        default="",
    )
    kv_cache_group.add_argument(
        "--device_cache_min_free_blocks",
        env_name="DEVICE_CACHE_MIN_FREE_BLOCKS",
        bind_to=(kv_cache_config, "device_cache_min_free_blocks"),
        type=int,
        default=0,
        help="Device KVCache 全局最小空闲 block 数；独立 pool 按各自 block 容量占比分摊。0 表示按调度容量自动计算。",
    )
    kv_cache_group.add_argument(
        "--dsv4_fixed_pool_blocks",
        env_name="DSV4_FIXED_POOL_BLOCKS",
        bind_to=(kv_cache_config, "dsv4_fixed_pool_blocks"),
        type=int,
        default=0,
        help="DSV4 固定池 block 数。>0 时用于 INDEXER_STATE/CSA_STATE/HCA_STATE/SWA_KV 四个 pool；"
        "不配置或配置为 0 时，这四个 pool 按 linear_step 派生 block 数，并保持一致。",
    )
    kv_cache_group.add_argument(
        "--dsv4_hca_state_pool_blocks",
        env_name="DSV4_HCA_STATE_POOL_BLOCKS",
        bind_to=(kv_cache_config, "dsv4_hca_state_pool_blocks"),
        type=int,
        default=0,
        help="DSV4 HCA_STATE pool 单独 block 数。>0 时仅覆盖 HCA_STATE；"
        "不配置或配置为 0 时，HCA_STATE 跟随 DSV4_FIXED_POOL_BLOCKS 或 linear_step 派生 block 数。",
    )
    kv_cache_group.add_argument(
        "--dsv4_fixed_pool_use_memory",
        env_name="DSV4_FIXED_POOL_USE_MEMORY",
        bind_to=(kv_cache_config, "dsv4_fixed_pool_use_memory"),
        type=str2bool,
        default=False,
        help="DSV4 固定池（INDEXER_STATE/CSA_STATE/HCA_STATE/SWA_KV）是否使用 pinned CPU memory。False 表示继续使用 GPU memory。",
    )
