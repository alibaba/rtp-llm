from rtp_llm.server.server_args.util import str2bool


def init_profile_debug_logging_group_args(parser):
    ##############################################################################################################
    # Profiling、Debugging、Logging
    ##############################################################################################################
    profile_debug_logging_group = parser.add_argument_group(
        "Profiling、Debugging、Logging"
    )
    profile_debug_logging_group.add_argument(
        "--trace_memory",
        env_name="RTP_LLM_TRACE_MEMORY",
        type=str2bool,
        default=False,
        help="控制是否在BufferManager中启用内存追踪功能。可选值: True (启用), False (禁用)。默认为 False",
    )
    profile_debug_logging_group.add_argument(
        "--trace_malloc_stack",
        env_name="RTP_LLM_TRACE_MALLOC_STACK",
        type=str2bool,
        default=False,
        help="是否启用 malloc stack 追踪,与RTP_LLM_TRACE_MEMORY结合使用",
    )
    profile_debug_logging_group.add_argument(
        "--enable_device_perf",
        env_name="ENABLE_DEVICE_PERF",
        type=str2bool,
        default=False,
        help="控制是否在DeviceBase中启用设备性能指标的收集和报告。可选值: True (启用), False (禁用)。",
    )
    profile_debug_logging_group.add_argument(
        "--ft_core_dump_on_exception",
        env_name="FT_CORE_DUMP_ON_EXCEPTION",
        type=str2bool,
        default=False,
        help="控制在发生特定异常或断言失败时是否强制执行core dump (程序中止并生成核心转储文件)。可选值: True (启用), False (禁用)。",
    )
    profile_debug_logging_group.add_argument(
        "--ft_alog_conf_path",
        env_name="FT_ALOG_CONF_PATH",
        type=str,
        default=None,
        help="设置日志配置文件路径。",
    )
    profile_debug_logging_group.add_argument(
        "--log_level",
        env_name="LOG_LEVEL",
        type=str,
        default="INFO",
        help="设置日志记录级别。可选级别包括: ERROR, WARN, INFO, DEBUG。默认为 INFO",
    )
    profile_debug_logging_group.add_argument(
        "--gen_timeline_sync",
        env_name="GEN_TIMELINE_SYNC",
        type=str2bool,
        default=False,
        help="是否开启收集Timeline信息用于性能分析",
    )
    profile_debug_logging_group.add_argument(
        "--torch_cuda_profiler_dir",
        env_name="TORCH_CUDA_PROFILER_DIR",
        type=str,
        default="",
        help="指定开启Torch的Profile时对应的生成目录",
    )

    profile_debug_logging_group.add_argument(
        "--log_path", env_name="LOG_PATH", type=str, default="logs", help="日志路径"
    )
    profile_debug_logging_group.add_argument(
        "--log_file_backup_count",
        env_name="LOG_FILE_BACKUP_COUNT",
        type=int,
        default=16,
        help="日志文件备份数量",
    )

    profile_debug_logging_group.add_argument(
        "--nccl_debug_file",
        env_name="NCCL_DEBUG_FILE",
        type=str,
        default=None,
        help="NCCL调试文件路径",
    )
    profile_debug_logging_group.add_argument(
        "--debug_load_server",
        env_name="DEBUG_LOAD_SERVER",
        type=str2bool,
        default=False,
        help="开启加载服务的调试模式",
    )
    profile_debug_logging_group.add_argument(
        "--hack_layer_num",
        env_name="HACK_LAYER_NUM",
        type=int,
        default=0,
        help="截断使用的模型层数",
    )
    profile_debug_logging_group.add_argument(
        "--debug_start_fake_process",
        env_name="DEBUG_START_FAKE_PROCESS",
        type=str2bool,
        default=None,
        help="开启启动Fake进程的Debug模式",
    )
    profile_debug_logging_group.add_argument(
        "--dg_print_reg_reuse",
        env_name="DG_PRINT_REG_REUSE",
        type=str2bool,
        default=None,
        help="控制是否打印 DeepGEMM 中的寄存器重用信息。",
    )
    profile_debug_logging_group.add_argument(
        "--qwen_agent_debug",
        env_name="QWEN_AGENT_DEBUG",
        type=int,
        default=0,
        help="控制是 Qwen Agent 的调试模式。0: Info, 其他： Debug。",
    )
    profile_debug_logging_group.add_argument(
        "--disable_dpc_random",
        env_name="DISABLE_DPC_RANDOM",
        type=str2bool,
        default=None,
        help="控制是否禁用 DPC 的随机性",
    )
    profile_debug_logging_group.add_argument(
        "--enable_detail_log",
        env_name="ENABLE_DETAIL_LOG",
        type=str2bool,
        default=None,
        help="控制是否打印细节日志，为了排查使用",
    )
    profile_debug_logging_group.add_argument(
        "--check_nan",
        env_name="CHECK_NAN",
        type=str2bool,
        default=False,
        help="控制是否check nan, 为了排查。可选值: True (启用), False (禁用)。默认为 False",
    )
