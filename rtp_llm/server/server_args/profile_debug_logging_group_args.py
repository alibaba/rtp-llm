from rtp_llm.server.server_args.util import str2bool


def init_profile_debug_logging_group_args(parser, profiling_debug_config):
    ##############################################################################################################
    # Profiling、Debugging、Logging
    ##############################################################################################################
    profile_debug_logging_group = parser.add_argument_group(
        "Profiling、Debugging、Logging"
    )
    profile_debug_logging_group.add_argument(
        "--trace_memory",
        env_name="RTP_LLM_TRACE_MEMORY",
        bind_to=(profiling_debug_config, "trace_memory"),
        type=str2bool,
        default=False,
        help="控制是否在BufferManager中启用内存追踪功能。可选值: True (启用), False (禁用)。默认为 False",
    )
    profile_debug_logging_group.add_argument(
        "--trace_malloc_stack",
        env_name="RTP_LLM_TRACE_MALLOC_STACK",
        bind_to=(profiling_debug_config, "trace_malloc_stack"),
        type=str2bool,
        default=False,
        help="是否启用 malloc stack 追踪,与RTP_LLM_TRACE_MEMORY结合使用",
    )
    profile_debug_logging_group.add_argument(
        "--enable_device_perf",
        env_name="ENABLE_DEVICE_PERF",
        bind_to=(profiling_debug_config, "enable_device_perf"),
        type=str2bool,
        default=False,
        help="控制是否在DeviceBase中启用设备性能指标的收集和报告。可选值: True (启用), False (禁用)。",
    )
    profile_debug_logging_group.add_argument(
        "--ft_core_dump_on_exception",
        env_name="FT_CORE_DUMP_ON_EXCEPTION",
        bind_to=(profiling_debug_config, "ft_core_dump_on_exception"),
        type=str2bool,
        default=False,
        help="控制在发生特定异常或断言失败时是否强制执行core dump (程序中止并生成核心转储文件)。可选值: True (启用), False (禁用)。",
    )
    profile_debug_logging_group.add_argument(
        "--ft_alog_conf_path",
        env_name="FT_ALOG_CONF_PATH",
        bind_to=(profiling_debug_config, "ft_alog_conf_path"),
        type=str,
        default=None,
        help="设置日志配置文件路径。",
    )
    profile_debug_logging_group.add_argument(
        "--gen_timeline_sync",
        env_name="GEN_TIMELINE_SYNC",
        bind_to=(profiling_debug_config, "gen_timeline_sync"),
        type=str2bool,
        default=False,
        help="是否开启收集Timeline信息用于性能分析",
    )
    profile_debug_logging_group.add_argument(
        "--torch_cuda_profiler_dir",
        env_name="TORCH_CUDA_PROFILER_DIR",
        bind_to=(profiling_debug_config, "torch_cuda_profiler_dir"),
        type=str,
        default="",
        help="指定开启Torch的Profile时对应的生成目录",
    )
    profile_debug_logging_group.add_argument(
        "--debug_load_server",
        env_name="DEBUG_LOAD_SERVER",
        bind_to=(profiling_debug_config, "debug_load_server"),
        type=str2bool,
        default=False,
        help="开启加载服务的调试模式",
    )
    profile_debug_logging_group.add_argument(
        "--hack_layer_num",
        env_name="HACK_LAYER_NUM",
        bind_to=(profiling_debug_config, "hack_layer_num"),
        type=int,
        default=0,
        help="截断使用的模型层数",
    )
    profile_debug_logging_group.add_argument(
        "--debug_start_fake_process",
        env_name="DEBUG_START_FAKE_PROCESS",
        bind_to=(profiling_debug_config, "debug_start_fake_process"),
        type=str2bool,
        default=None,
        help="开启启动Fake进程的Debug模式",
    )
    profile_debug_logging_group.add_argument(
        "--enable_detail_log",
        env_name="ENABLE_DETAIL_LOG",
        bind_to=(profiling_debug_config, "enable_detail_log"),
        type=str2bool,
        default=None,
        help="控制是否打印细节日志，为了排查使用",
    )
    profile_debug_logging_group.add_argument(
        "--check_nan",
        env_name="CHECK_NAN",
        bind_to=(profiling_debug_config, "check_nan"),
        type=str2bool,
        default=False,
        help="控制是否check nan, 为了排查。可选值: True (启用), False (禁用)。默认为 False",
    )
    profile_debug_logging_group.add_argument(
        "--enable_torch_alloc_profile",
        env_name="ENABLE_TORCH_ALLOC_PROFILE",
        bind_to=(profiling_debug_config, "enable_torch_alloc_profile"),
        type=str2bool,
        default=False,
        help="控制是否在TorchCudaAllocator中启用Python堆栈追踪,用于调试Torch buffer分配。可选值: True (启用), False (禁用)。默认为 False",
    )
