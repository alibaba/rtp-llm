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
        "--int8_kv_cache",
        env_name="INT8_KV_CACHE",
        bind_to=(kv_cache_config, "int8_kv_cache"),
        type=int,
        default=0,
        help="是否开启INT8的KV_CACHE",
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
        "--enable_memory_cache",
        env_name="ENABLE_MEMORY_CACHE",
        bind_to=(kv_cache_config, "enable_memory_cache"),
        type=str2bool,
        default=False,
        help="内存 KVCache 开关. 当开启时, 需要显示通过 MEMORY_CACHE_SIZE_MB 设置内存大小",
    )
    kv_cache_group.add_argument(
        "--enable_memory_cache_sm_copy",
        env_name="ENABLE_MEMORY_CACHE_SM_COPY",
        bind_to=(kv_cache_config, "enable_memory_cache_sm_copy"),
        type=str2bool,
        default=False,
        help="内存 Cache 拷贝是否启用 split-KV SM scatter/gather（CUDA 上满足布局条件时）。默认 False；True 时满足条件可走 SM copy。",
    )
    kv_cache_group.add_argument(
        "--memory_cache_size_mb",
        env_name="MEMORY_CACHE_SIZE_MB",
        bind_to=(kv_cache_config, "memory_cache_size_mb"),
        type=int,
        default=0,
        help="单个RANK Memory Cache 的大小, 单位为MB",
    )
    kv_cache_group.add_argument(
        "--memory_cache_sync_timeout_ms",
        env_name="MEMORY_CACHE_SYNC_TIMEOUT_MS",
        bind_to=(kv_cache_config, "memory_cache_sync_timeout_ms"),
        type=int,
        default=10000,
        help="Memory Cache 多TP同步的超时时间, 单位为毫秒",
    )
    kv_cache_group.add_argument(
        "--enable_memory_cache_disk",
        env_name="ENABLE_MEMORY_CACHE_DISK",
        bind_to=(kv_cache_config, "enable_memory_cache_disk"),
        type=str2bool,
        default=False,
        help="Memory connector 内部 disk KV cache 开关。开启时必须同时开启 enable_memory_cache。",
    )
    kv_cache_group.add_argument(
        "--memory_cache_disk_paths",
        env_name="MEMORY_CACHE_DISK_PATHS",
        bind_to=(kv_cache_config, "memory_cache_disk_paths"),
        type=str,
        default="",
        help="逗号分隔的本机 disk KV cache 挂载点列表，数量必须等于本机 local_world_size。",
    )
    kv_cache_group.add_argument(
        "--memory_cache_disk_size_mb",
        env_name="MEMORY_CACHE_DISK_SIZE_MB",
        bind_to=(kv_cache_config, "memory_cache_disk_size_mb"),
        type=int,
        default=0,
        help="每个 GPU rank 可用的 disk KV cache 文件大小，单位 MB。",
    )
    kv_cache_group.add_argument(
        "--memory_cache_disk_buffered_io",
        env_name="MEMORY_CACHE_DISK_BUFFERED_IO",
        bind_to=(kv_cache_config, "memory_cache_disk_buffered_io"),
        type=str2bool,
        default=True,
        help="disk KV cache 是否使用 buffered IO。False 时使用 O_DIRECT。",
    )
    kv_cache_group.add_argument(
        "--memory_cache_disk_sync_timeout_ms",
        env_name="MEMORY_CACHE_DISK_SYNC_TIMEOUT_MS",
        bind_to=(kv_cache_config, "memory_cache_disk_sync_timeout_ms"),
        type=int,
        default=30000,
        help="包含 disk backing 的 memory cache copy plan 同步超时时间，单位毫秒。",
    )
    kv_cache_group.add_argument(
        "--write_cache_sync",
        env_name="WRITE_CACHE_SYNC",
        bind_to=(kv_cache_config, "write_cache_sync"),
        type=str2bool,
        default=False,
        help="KVCache 同步写入开关. 当开启时, 在写入 Cache 时会等待写入完成. 默认关闭(即异步写入), Smoke 测试时需开启",
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
        "--enable_tiered_memory_cache",
        env_name="ENABLE_TIERED_MEMORY_CACHE",
        bind_to=(kv_cache_config, "enable_tiered_memory_cache"),
        type=str2bool,
        default=False,
        help="分层 cache 开关。开启后，stream 释放时只全量写 remote，再按 GPU 空闲 block 阈值将冷 block 淘汰到 memory。",
    )
    kv_cache_group.add_argument(
        "--device_cache_min_free_blocks",
        env_name="DEVICE_CACHE_MIN_FREE_BLOCKS",
        bind_to=(kv_cache_config, "device_cache_min_free_blocks"),
        type=int,
        default=0,
        help="分层 cache 模式下 GPU 侧至少保留的空闲 block 数；当空闲 block 低于该阈值时，会把冷 block 从 GPU 淘汰到 memory。"
        "不填或填 0 时自动计算为 min(max_context_batch_size * max_seq_len, max_batch_tokens_size) / seq_size_per_block。",
    )
    kv_cache_group.add_argument(
        "--non_full_addition_kvcache_blocks",
        env_name="NON_FULL_ADDITION_KVCACHE_BLOCKS",
        bind_to=(kv_cache_config, "non_full_addition_kvcache_blocks"),
        type=int,
        default=256,
        help="对每个非-FULL group（SWA / LINEAR）与 step>1 时的 memory incomplete pool，"
        "在规则分配的 block_num 之外额外追加的固定 block 数，"
        "作为并发请求 tail block 与 prefix cache 命中场景的 headroom。"
        "每个非-FULL pool 实际容量 = 规则 block_num + 本参数。"
        "device 侧无论 linear_step 都生效；host incomplete pool 只在 linear_step>1 时生效。设为 0 关闭。",
    )
    kv_cache_group.add_argument(
        "--state_pool_memory_mb",
        env_name="STATE_POOL_MEMORY_MB",
        bind_to=(kv_cache_config, "state_pool_memory_mb"),
        type=int,
        default=0,
        help="DSV4 STATE pools (INDEXER_STATE/CSA_STATE/HCA_STATE) CPU pinned BlockPool 的"
        "每-rank 内存预算，单位 MiB（按 1024*1024 折算，与 KV_CACHE_MEM_MB 一致）。"
        "0 表示不显式指定，STATE pool block_num 沿用 KV pool block_num（向后兼容）。"
        "设置为 >0 时，STATE pool block_num = floor(state_pool_memory_mb * 1024 * 1024 / "
        "Σstate_block_size_bytes)，与 HBM 派生的 block_num 解耦。"
        "**Production PREFILL deployments MUST set this value explicitly** —"
        "B-fix (XR2_B1/B2/B3) routes PREFILL STATE to pinned-CPU; the "
        "fallback path mirrors HBM block_num to pinned-CPU bytes, which can "
        "exceed host RLIMIT_MEMLOCK on dense multi-rank-per-host fleets. "
        "Startup emits WARNING when fallback est. > 8 GiB/pool. Hard cap "
        "is pending (canary §1.0 item 10).",
    )
    kv_cache_group.add_argument(
        "--dsv4_unified_block_count",
        env_name="DSV4_UNIFIED_BLOCKS",
        bind_to=(kv_cache_config, "dsv4_unified_block_count"),
        type=int,
        default=-1,
        help="F02 DSV4 unified super-block layout opt-in (tri-state): "
        "-1=auto (follow compiled default; legacy OFF semantics), 0=force OFF (legacy per-group path), "
        "1=force ON (CacheConfig.super_block_layout.enabled=true, bps≡1). "
        "Phase 5 default flip is GATED PENDING prerequisites: createSpConfig MTP "
        "unified branch (R02 H-1), validatePeerHandshake wiring + hash_salt_version "
        "computation (R17 F1/F2), ConfigModules.h C++ default mirror (R10 C2), "
        "pinned-CPU oversubscribe guard (R02 H-2), canary metric registration "
        "(R10 C6), and smoke baseline env pin (R10 C1). Until those land, default "
        "stays -1 (auto = legacy OFF) and operators opt in per-process via "
        "DSV4_UNIFIED_BLOCKS=1. "
        "See docs/dsv4/kvcache-unify-final/canary/PHASE5_CANARY_PROCEDURE.md "
        "for the re-flip pre-flight checklist.",
    )
    kv_cache_group.add_argument(
        "--dsv4_state_entries_per_block",
        env_name="DSV4_STATE_ENTRIES_PER_BLOCK",
        bind_to=(kv_cache_config, "dsv4_state_entries_per_block"),
        type=int,
        default=0,
        help="F01 phase-2 K_state hook for the 3 DSV4 STATE pools "
        "(INDEXER_STATE / CSA_STATE / HCA_STATE): collapses each state "
        "pool's entries_per_block from the kernel-block size (256) down to "
        "this value, reducing per-block bytes by 256/K_state. "
        "0 = OFF (default; state pools keep 256 entries/block — production "
        "state-on-CPU goldens are byte-identical to today). "
        ">0 = K_state (typical opus F01 §2 setting K_state=4 → 9.9x state "
        "pool density / 64x reduction per state pool per block, with kernel-"
        "side compressor/decode_attn_metadata changes landing in F01-PR2 and "
        "HBM mem accounting smoke in F01-PR3). Only the per-block byte "
        "sizing flips in PR-1 — peer hash_salt K_state bit (bit3) and the "
        "kernel-side metadata both stay wired through their F01-PR2 producer.",
    )
