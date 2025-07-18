import argparse
import logging
import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from rtp_llm.config.py_config_modules import StaticConfig

_T = TypeVar("_T")


class EnvArgumentGroup:
    def __init__(self, group: argparse._ArgumentGroup, parser: "EnvArgumentParser"):
        self._group = group
        self._parser = parser

    def add_argument(
        self, *args, env_name: Optional[str] = None, **kwargs
    ) -> argparse.Action:
        if "metavar" not in kwargs and "type" in kwargs:
            type_ = kwargs["type"]
            if isinstance(type_, type) and issubclass(type_, bool):
                kwargs["metavar"] = "BOOL"
            elif isinstance(type_, type) and issubclass(type_, int):
                kwargs["metavar"] = "INT"
            elif isinstance(type_, type) and issubclass(type_, str):
                kwargs["metavar"] = "STR"
        action = self._group.add_argument(*args, **kwargs)
        self._parser._register_env_mapping(action, args, env_name)
        return action

    def __getattr__(self, name):
        return getattr(self._group, name)


class EnvArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, env_prefix: str = "", **kwargs):
        self.env_prefix = env_prefix.upper()
        self._env_mappings: Dict[str, str] = {}
        self._groups: Dict[str, EnvArgumentGroup] = {}

        super().__init__(*args, **kwargs)

        self._default_group = EnvArgumentGroup(self._positionals, self)
        self._optional_group = EnvArgumentGroup(self._optionals, self)

    def add_argument_group(self, *args, **kwargs) -> EnvArgumentGroup:
        group = super().add_argument_group(*args, **kwargs)
        env_group = EnvArgumentGroup(group, self)

        if hasattr(group, "title") and group.title:
            self._groups[group.title] = env_group

        return env_group

    def add_mutually_exclusive_group(self, **kwargs) -> EnvArgumentGroup:
        group = super().add_mutually_exclusive_group(**kwargs)
        return EnvArgumentGroup(group, self)

    def add_argument(
        self, *args, env_name: Optional[str] = None, **kwargs
    ) -> argparse.Action:
        if args and isinstance(args[0], str) and not args[0].startswith("-"):
            action = self._positionals.add_argument(*args, **kwargs)
        else:
            action = self._optionals.add_argument(*args, **kwargs)

        self._register_env_mapping(action, args, env_name)
        return action

    def _register_env_mapping(
        self,
        action: argparse.Action,
        args: Sequence[Any],
        env_name: Optional[str] = None,
    ) -> None:
        effective_env_name = env_name
        if effective_env_name is None:
            for arg_name_or_flag in args:
                if isinstance(arg_name_or_flag, str) and arg_name_or_flag.startswith(
                    "--"
                ):
                    effective_env_name = arg_name_or_flag[2:].upper().replace("-", "_")
                    break
            else:
                effective_env_name = action.dest.upper().replace("-", "_")
        else:
            effective_env_name = effective_env_name.upper().replace("-", "_")

        if self.env_prefix:
            full_env_name = f"{self.env_prefix}_{effective_env_name}"
        else:
            full_env_name = effective_env_name

        self._env_mappings[action.dest] = full_env_name

    def parse_args(
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> argparse.Namespace:
        logging.info("Parsing arguments and setting environment variables...")
        parsed_args = super().parse_args(args, namespace)

        for dest, env_name in self._env_mappings.items():
            value = getattr(parsed_args, dest, None)

            if value is None:
                continue

            if env_name in os.environ and value == self.get_default(dest):
                continue

            env_value: str
            if isinstance(value, bool):
                env_value = "1" if value else "0"
            elif isinstance(value, list):
                env_value = ",".join(map(str, value))
            else:
                env_value = str(value)
            logging.info(f"{env_name} = {env_value}")
            os.environ[env_name] = env_value

        return parsed_args

    def print_env_mappings(self, group_name: Optional[str] = None) -> None:
        logging.info("Argument -> Environment Variable Mappings:")
        logging.info("-" * 50)

        if group_name:
            if group_name in self._groups:
                group = self._groups[group_name]._group
                for action in group._group_actions:
                    if action.dest in self._env_mappings:
                        logging.info(
                            f"{action.dest:<20} -> {self._env_mappings[action.dest]}"
                        )
            else:
                logging.info(f"Group '{group_name}' not found.")
        else:
            for dest, env_name in self._env_mappings.items():
                logging.info(f"{dest:<20} -> {env_name}")

        logging.info("-" * 50)

    def get_env_mappings(self, group_name: Optional[str] = None) -> Dict[str, str]:
        if group_name and group_name in self._groups:
            group = self._groups[group_name]._group
            mappings = {}
            for action in group._group_actions:
                if action.dest in self._env_mappings:
                    mappings[action.dest] = self._env_mappings[action.dest]
            return mappings
        else:
            return self._env_mappings.copy()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "on"):
        return True
    if v.lower() in ("no", "false", "f", "0", "off"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_args():
    parser = EnvArgumentParser(description="RTP LLM")

    ##############################################################################################################
    # Parallelism and Distributed Setup Configuration
    ##############################################################################################################
    parallel_group = parser.add_argument_group(
        "Parallelism and Distributed Setup Configuration"
    )
    parallel_group.add_argument(
        "--tp_size",
        env_name="TP_SIZE",
        type=int,
        default=None,
        help="指定用于张量并行度。",
    )
    parallel_group.add_argument(
        "--ep_size",
        env_name="EP_SIZE",
        type=int,
        default=None,
        help="定义用于专家并行（Expert Parallelism）的模型（专家）实例数量。",
    )
    parallel_group.add_argument(
        "--dp_size",
        env_name="DP_SIZE",
        type=int,
        default=None,
        help="设置数据并行（Data Parallelism）的副本数量或组大小。",
    )
    parallel_group.add_argument(
        "--world_size",
        env_name="WORLD_SIZE",
        type=int,
        default=None,
        help="分布式设置中使用的GPU总数。通常情况下，`WORLD_SIZE = TP_SIZE * DP_SIZE`",
    )
    parallel_group.add_argument(
        "--world_rank",
        env_name="WORLD_RANK",
        type=int,
        default=None,
        help="当前进程/GPU在分布式系统中的全局唯一编号（从0到 `WORLD_SIZE - 1`）。",
    )
    parallel_group.add_argument(
        "--local_world_size",
        env_name="LOCAL_WORLD_SIZE",
        type=int,
        default=None,
        help="在多节点分布式设置中，当前节点（Node）上使用的GPU设备数量。",
    )
    parallel_group.add_argument(
        "--ffn_sp_size",
        env_name="FFN_SP_SIZE",
        type=int,
        default=1,
        help="FFN层序列并行大小。",
    )

    ##############################################################################################################
    # Concurrency 控制
    ##############################################################################################################
    concurrent_group = parser.add_argument_group("Concurrent")
    concurrent_group.add_argument(
        "--concurrency_with_block",
        env_name="CONCURRENCY_WITH_BLOCK",
        type=str2bool,
        default=False,
        help="控制并发请求的阻塞行为。通常设置为 '1' (启用阻塞) 或 '0' (禁用阻塞)。",
    )
    concurrent_group.add_argument(
        "--concurrency_limit",
        env_name="CONCURRENCY_LIMIT",
        type=int,
        default=32,
        help="设置系统允许的最大并发请求数量。",
    )

    ##############################################################################################################
    # FMHA
    ##############################################################################################################
    fmha_group = parser.add_argument_group("FMHA")
    fmha_group.add_argument(
        "--enable_fmha",
        env_name="ENABLE_FMHA",
        type=str2bool,
        default=True,
        help="控制是否启用Fused Multi-Head Attention (FMHA) 功能。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_trt_fmha",
        env_name="ENABLE_TRT_FMHA",
        type=str2bool,
        default=True,
        help="控制是否启用经TensorRT(V2版本)优化的FMHA功能。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_paged_trt_fmha",
        env_name="ENABLE_PAGED_TRT_FMHA",
        type=str2bool,
        default=True,
        help="控制是否启用Paged TensorRT FMHA功能。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_open_source_fmha",
        env_name="ENABLE_OPENSOURCE_FMHA",
        type=str2bool,
        default=True,
        help="控制是否启用开源版本的FMHA实现。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_paged_open_source_fmha",
        env_name="ENABLE_PAGED_OPEN_SOURCE_FMHA",
        type=str2bool,
        default=True,
        help="控制是否启用Paged开源版本的FMHA实现。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--enable_trtv1_fmha",
        env_name="ENABLE_TRTV1_FMHA",
        type=str2bool,
        default=True,
        help="控制是否启用TRTv1风格的FMHA功能。可选值: True (启用), False (禁用)。",
    )
    fmha_group.add_argument(
        "--fmha_perf_instrument",
        env_name="FMHA_PERF_INSTRUMENT",
        type=str2bool,
        default=False,
        help="控制是否为FMHA启用NVTX性能分析。设置为 True 启用, False 禁用。",
    )
    fmha_group.add_argument(
        "--fmha_show_params",
        env_name="FMHA_SHOW_PARAMS",
        type=str2bool,
        default=False,
        help="控制是否显示FMHA的参数信息。设置为 True 启用, False 禁用。",
    )
    fmha_group.add_argument(
        "--disable_flash_infer",
        env_name="DISABLE_FLASH_INFER",
        type=str2bool,
        default=False,
        help="控制是否禁用FlashInfer Attention机制。设置为 True 启用, False 禁用。",
    )
    fmha_group.add_argument(
        "--enable_xqa",
        env_name="ENABLE_XQA",
        type=str2bool,
        default=True,
        help="控制是否开启 xqa 的功能，此功能需要 SM 90 (Hopper) 或更新的 GPU 架构。可选值: True (启用), False (禁用)。",
    )

    ##############################################################################################################
    # KV Cache 相关配置
    ##############################################################################################################
    kv_cache_group = parser.add_argument_group("KV_Cache")
    kv_cache_group.add_argument(
        "--reuse_cache",
        env_name="REUSE_CACHE",
        type=str2bool,
        default=False,
        help="控制是否激活KV Cache的重用机制。设置为 True 启用 , False 关闭",
    )
    kv_cache_group.add_argument(
        "--multi_task_prompt",
        env_name="MULTI_TASK_PROMPT",
        type=str,
        default=None,
        help="指定一个多任务提示（multi-task prompt），为一个路径，系统会读取路径指定的多任务json文件。默认为空",
    )
    kv_cache_group.add_argument(
        "--multi_task_prompt_str",
        env_name="MULTI_TASK_PROMPT_STR",
        type=str,
        default=None,
        help="指定一个多任务提示字符串（multi-task prompt string），为多任务纯json字符串，类似于系统提示词。默认为空 ",
    )
    kv_cache_group.add_argument(
        "--int8_kv_cache",
        env_name="INT8_KV_CACHE",
        type=int,
        default=0,
        help="是否开启INT8的KV_CACHE",
    )
    kv_cache_group.add_argument(
        "--kv_cache_mem_mb",
        env_name="KV_CACHE_MEM_MB",
        type=int,
        default=-1,
        help="KV_CACHE的大小",
    )
    kv_cache_group.add_argument(
        "--seq_size_per_block",
        env_name="SEQ_SIZE_PER_BLOCK",
        type=str,
        default=None,
        help="单独一个KV_CACHE的Block里面token的数量",
    )
    kv_cache_group.add_argument(
        "--test_block_num",
        env_name="TEST_BLOCK_NUM",
        type=int,
        default=0,
        help="在测试时强制指定BLOCK的数量",
    )

    ##############################################################################################################
    # Profiling、Debugging、Logging
    ##############################################################################################################
    profile_debug_logging_group = parser.add_argument_group(
        "Profiling、Debugging、Logging"
    )
    profile_debug_logging_group.add_argument(
        "--ft_nvtx",
        env_name="FT_NVTX",
        type=str2bool,
        default=False,
        help="控制是否启用NVTX性能分析。可选值: True (启用), False (禁用)。默认为 False",
    )
    profile_debug_logging_group.add_argument(
        "--py_inference_log_response",
        env_name="PY_INFERENCE_LOG_RESPONSE",
        type=str2bool,
        default=False,
        help="控制是否在Python推理的access log中记录响应内容。可选值: `True` (记录), `False` (不记录)。默认为 `False`",
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
        "--test_layer_num",
        env_name="TEST_LAYER_NUM",
        type=int,
        default=None,
        help="测试用的模型层数数量",
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

    ##############################################################################################################
    # 硬件/Kernel 特定优化
    ##############################################################################################################
    hw_kernel_group = parser.add_argument_group("硬件/Kernel 特定优化")

    hw_kernel_group.add_argument(
        "--deep_gemm_num_sm",
        env_name="DEEP_GEMM_NUM_SM",
        type=int,
        default=None,
        help="指定 DeepGEMM 使用的 SM (Streaming Multiprocessor) 数量。如果设置，此值将覆盖自动检测的数量。",
    )

    hw_kernel_group.add_argument(
        "--arm_gemm_use_kai",
        env_name="ARM_GEMM_USE_KAI",
        type=str2bool,
        default=False,
        help="设置为 `True` 时，为 ARM GEMM 操作启用 KleidiAI 支持。这可能影响权重处理和计算性能。",
    )

    hw_kernel_group.add_argument(
        "--enable_stable_scatter_add",
        env_name="ENABLE_STABLE_SCATTER_ADD",
        type=str2bool,
        default=False,
        help="控制是否启用稳定的 scatter add 操作。",
    )

    hw_kernel_group.add_argument(
        "--enable_multi_block_mode",
        env_name="ENABLE_MULTI_BLOCK_MODE",
        type=str2bool,
        default=True,
        help="控制是否为 Multi-Head Attention (MMHA) 启用 multi-block 模式。设置为 'ON' 启用，'OFF' 禁用。",
    )

    hw_kernel_group.add_argument(
        "--rocm_hipblaslt_config",
        env_name="ROCM_HIPBLASLT_CONFIG",
        type=str,
        default="gemm_config.csv",
        help="指定 hipBLASLt GEMM 配置文件的路径。此文件用于优化 ROCm平台上的 GEMM 操作。",
    )

    hw_kernel_group.add_argument(
        "--ft_disable_custom_ar",
        env_name="FT_DISABLE_CUSTOM_AR",
        type=str2bool,
        default=True,
        help="设置为 `True` 时，禁用自定义的 AllReduce (AR) 实现，可能回退到标准库（如 NCCL）的 AllReduce。",
    )
    hw_kernel_group.add_argument(
        "--enable_merge_w13",
        env_name="ENABLE_MERGE_W13",
        type=str2bool,
        default=False,
        help="设置为 `True` 时，启用FFN W13 的合并操作。",
    )

    ##############################################################################################################
    # 采样
    ##############################################################################################################
    sampling_group = parser.add_argument_group("采样")

    sampling_group.add_argument(
        "--max_batch_size",
        env_name="MAX_BATCH_SIZE",
        type=int,
        default=0,
        help="覆盖系统自动计算的最大 batch size。",
    )
    sampling_group.add_argument(
        "--enable_flashinfer_sample_kernel",
        env_name="ENABLE_FLASHINFER_SAMPLE_KERNEL",
        type=str2bool,
        default=True,
        help="控制是否启用FlashInfer的采样 kernel。可选值: True (启用), False (禁用)。",
    )

    ##############################################################################################################
    # 设备和资源管理
    ##############################################################################################################
    device_resource_group = parser.add_argument_group("设备和资源管理")

    device_resource_group.add_argument(
        "--device_reserve_memory_bytes",
        env_name="DEVICE_RESERVE_MEMORY_BYTES",
        type=int,
        default=0,
        help="指定在GPU设备上预留的内存量（单位：字节）。此内存不会被常规操作使用，可用于应对突发需求或特定驱动/内核开销。",
    )

    device_resource_group.add_argument(
        "--host_reserve_memory_bytes",
        env_name="HOST_RESERVE_MEMORY_BYTES",
        type=int,
        default=4 * 1024 * 1024 * 1024,  # 4GB
        help="指定在主机（CPU）上预留的内存量（单位：字节）。此内存不会被常规操作使用。默认为 4GB。",
    )

    device_resource_group.add_argument(
        "--overlap_math_sm_count",
        env_name="OVERLAP_MATH_SM_COUNT",
        type=int,
        default=0,
        help="指定用于计算与通信重叠优化的 SM 数量。",
    )

    device_resource_group.add_argument(
        "--overlap_comm_type",
        env_name="OVERLAP_COMM_TYPE",
        type=int,
        default=0,
        help="指定计算与通信重叠的策略类型。0: 禁止重叠，串行执行；1: 轻量级重叠，平衡性能与复杂度；2: 深度优化，通过多流和事件管理实现更大吞吐量。",
    )

    device_resource_group.add_argument(
        "--m_split",
        env_name="M_SPLIT",
        type=int,
        default=0,
        help="为特定设备操作设置 M_SPLIT 参数值。`0` 通常表示使用默认或不拆分。",
    )

    device_resource_group.add_argument(
        "--enable_comm_overlap",
        env_name="ENABLE_COMM_OVERLAP",
        type=str2bool,
        default=True,
        help="设置为 `True` 以启用计算与通信之间的重叠执行，旨在提高设备利用率和吞吐量。",
    )

    device_resource_group.add_argument(
        "--enable_layer_micro_batch",
        env_name="ENABLE_LAYER_MICRO_BATCH",
        type=int,
        default=0,
        help="控制是否启用层级的 micro-batching。",
    )

    device_resource_group.add_argument(
        "--not_use_default_stream",
        env_name="NOT_USE_DEFAULT_STREAM",
        type=str2bool,
        default=False,
        help="控制 PyTorch 操作不使用标准的默认 CUDA 流。",
    )

    device_resource_group.add_argument(
        "--reserver_runtime_mem_mb",
        env_name="RESERVER_RUNTIME_MEM_MB",
        type=int,
        default=1024,
        help="设备保留的运行时显存大小",
    )
    device_resource_group.add_argument(
        "--specify_gpu_arch",
        env_name="SPECIFY_GPU_ARCH",
        type=str,
        default="",
        help="测试时使用的指定GPU架构",
    )
    device_resource_group.add_argument(
        "--acext_gemm_config_dir",
        env_name="ACEXT_GEMM_CONFIG_DIR",
        type=str,
        default=None,
        help="ACEXT GEMM配置目录",
    )

    ##############################################################################################################
    # MOE 特性
    ##############################################################################################################
    moe_group = parser.add_argument_group("MOE 专家并行")
    moe_group.add_argument(
        "--use_deepep_moe",
        env_name="USE_DEEPEP_MOE",
        type=str2bool,
        default=False,
        help="设置为 `True` 以启用 DeepEP 来处理 MoE 模型的 expert 部分。",
    )

    moe_group.add_argument(
        "--use_deepep_internode",
        env_name="USE_DEEPEP_INTERNODE",
        type=str2bool,
        default=False,
        help="设置为 `True` 以启用 DeepEP 来优化跨节点 (inter-node) 通信。",
    )

    moe_group.add_argument(
        "--use_deepep_low_latency",
        env_name="USE_DEEPEP_LOW_LATENCY",
        type=str2bool,
        default=True,
        help="设置为 `True` 以启用 DeepEP 的低延迟模式。",
    )

    moe_group.add_argument(
        "--use_deepep_p2p_low_latency",
        env_name="USE_DEEPEP_P2P_LOW_LATENCY",
        type=str2bool,
        default=False,
        help="设置为 `True` 以启用 DeepEP 的点对点 (P2P) 低延迟模式。",
    )

    moe_group.add_argument(
        "--deep_ep_num_sm",
        env_name="DEEP_EP_NUM_SM",
        type=int,
        default=0,
        help="为 DeepEPBuffer 设置 SM (Streaming Multiprocessor) 数量。设置为 `0` 将使用系统默认配置。",
    )

    moe_group.add_argument(
        "--fake_balance_expert",
        env_name="FAKE_BALANCE_EXPERT",
        type=str2bool,
        default=False,
        help="设置为 `True` 时，为 MoE 模型中的 expert 启用伪均衡 (fake balancing) 机制。用于测试或模拟特定均衡行为。",
    )

    moe_group.add_argument(
        "--eplb_control_step",
        env_name="EPLB_CONTROL_STEP",
        type=int,
        default=100,
        help="为 EPLB (Expert Placement Load Balancing) 控制器指定控制周期或步骤参数。这可能影响专家的负载均衡调整的频率或粒度。",
    )

    moe_group.add_argument(
        "--eplb_test_mode",
        env_name="EPLB_TEST_MODE",
        type=str2bool,
        default=False,
        help="设置为 `True` 时，为 ExpertBalancer 组件启用测试模式。用于调试或特定的测试场景。",
    )

    moe_group.add_argument(
        "--eplb_balance_layer_per_step",
        env_name="EPLB_BALANCE_LAYER_PER_STEP",
        type=int,
        default=1,
        help="设置 eplb 每次更新的层数。",
    )

    moe_group.add_argument(
        "--eplb_mode",
        env_name="EPLB_MODE",
        type=str,
        default="NONE",
        help="专家并行的负载均衡模式",
    )
    moe_group.add_argument(
        "--eplb_update_time",
        env_name="EPLB_UPDATE_TIME",
        type=int,
        default=5000,
        help="专家并行复杂均衡的更新时间",
    )
    moe_group.add_argument(
        "--redundant_expert",
        env_name="REDUNDANT_EXPERT",
        type=int,
        default=0,
        help="冗余专家个数",
    )
    moe_group.add_argument(
        "--hack_ep_single_entry",
        env_name="HACK_EP_SINGLE_ENTRY",
        type=int,
        default=0,
        help="HACK_EP_SINGLE_ENTRY",
    )
    moe_group.add_argument(
        "--balance_method",
        env_name="BALANCE_METHOD",
        type=str,
        default="mix",
        help="负载均衡的方法",
    )
    moe_group.add_argument(
        "--eplb_force_repack",
        env_name="EPLB_FORCE_REPACK",
        type=int,
        default=0,
        help="EPLB_FORCE_REPACK",
    )
    moe_group.add_argument(
        "--eplb_stats_window_size",
        env_name="EPLB_STATS_WINDOW_SIZE",
        type=int,
        default=10,
        help="负载均衡的统计窗口大小",
    )

    ##############################################################################################################
    # 模型特定配置
    ##############################################################################################################
    model_specific_group = parser.add_argument_group("模型特定配置")

    model_specific_group.add_argument(
        "--max_lora_model_size",
        env_name="MAX_LORA_MODEL_SIZE",
        type=int,
        default=-1,
        help="指定 LoRA 模型的最大允许大小。",
    )

    ##############################################################################################################
    # 投机采样配置
    ##############################################################################################################
    speculative_decoding_group = parser.add_argument_group("投机采样")
    speculative_decoding_group.add_argument(
        "--sp_model_type",
        env_name="SP_MODEL_TYPE",
        type=str,
        default="",
        help='指定 speculative decoding 的草稿模型类型。例如："mixtbstars-mtp", "deepseek-v3-mtp"。',
    )

    speculative_decoding_group.add_argument(
        "--sp_type",
        env_name="SP_TYPE",
        type=str,
        default="",
        help='控制是否启用 speculative decoding 。"vanilla" 不启用，"mtp" 启用 ',
    )

    speculative_decoding_group.add_argument(
        "--sp_min_token_match",
        env_name="SP_MIN_TOKEN_MATCH",
        type=int,
        default=2,
        help="为 speculative decoding 设置最小 token 匹配长度。",
    )

    speculative_decoding_group.add_argument(
        "--sp_max_token_match",
        env_name="SP_MAX_TOKEN_MATCH",
        type=int,
        default=2,
        help="为 speculative decoding 设置最大 token 匹配长度。",
    )

    speculative_decoding_group.add_argument(
        "--tree_decode_config",
        env_name="TREE_DECODE_CONFIG",
        type=str,
        default="",
        help="Tree decode的配置文件名，定义了从前缀词到候选Token的映射。",
    )
    speculative_decoding_group.add_argument(
        "--sp_act_type",
        env_name="SP_ACT_TYPE",
        type=str,
        default=None,
        help="小模型的计算使用的类型",
    )
    speculative_decoding_group.add_argument(
        "--sp_quantization", env_name="SP_QUANTIZATION", type=str, default=None, help=""
    )
    speculative_decoding_group.add_argument(
        "--sp_checkpoint_path",
        env_name="SP_CHECKPOINT_PATH",
        type=str,
        default=None,
        help="",
    )

    speculative_decoding_group.add_argument(
        "--gen_num_per_cycle",
        env_name="GEN_NUM_PER_CIRCLE",
        type=int,
        default=1,
        help="每一轮 speculative execution（推测式生成）中，最多生成多少个 token。",
    )

    speculative_decoding_group.add_argument(
        "--force_stream_sample",
        env_name="FORCE_STREAM_SAMPLE",
        type=str2bool,
        default=False,
        help="投机采样强制使用流式采样",
    )

    speculative_decoding_group.add_argument(
        "--force_score_context_attention",
        env_name="FORCE_SCORE_CONTEXT_ATTENTION",
        type=str2bool,
        default=True,
        help="投机采样强制score阶段使用context attention",
    )

    ##############################################################################################################
    # RPC 与服务发现配置
    ##############################################################################################################
    rpc_discovery_group = parser.add_argument_group("RPC 与服务发现")

    rpc_discovery_group.add_argument(
        "--use_local",
        env_name="USE_LOCAL",
        type=str2bool,
        default=False,
        help="设置为 `True` 时，系统将使用本地配置进行 decode 和 ViT 服务发现，而不是依赖外部服务注册与发现机制 (如 CM2)。适用于本地测试或特定部署。",
    )

    rpc_discovery_group.add_argument(
        "--remote_rpc_server_ip",
        env_name="REMOTE_RPC_SERVER_IP",
        type=str,
        default=None,
        help="指定远程 RPC 服务器的 IP 地址和可选端口 (格式: `ip:port` 或 `ip`)。主要用于 prefill server 的本地测试和调试。",
    )

    rpc_discovery_group.add_argument(
        "--decode_cm2_config",
        env_name="RTP_LLM_DECODE_CM2_CONFIG",
        type=str,
        default=None,
        help="为 decode cluster 提供服务发现 (如 CM2) 的 JSON 配置字符串。用于在集群环境中定位 decode 服务。",
    )

    rpc_discovery_group.add_argument(
        "--remote_vit_server_ip",
        env_name="REMOTE_VIT_SERVER_IP",
        type=str,
        default=None,
        help="指定远程 ViT (Visual Transformer) 服务器的 IP 地址和可选端口。主要用于多模态模型的本地测试和调试。",
    )

    rpc_discovery_group.add_argument(
        "--multimodal_part_cm2_config",
        env_name="RTP_LLM_MULTIMODAL_PART_CM2_CONFIG",
        type=str,
        default=None,
        help="为多模态 (ViT) 服务部分提供服务发现 (如 CM2) 的 JSON 配置字符串。用于在集群环境中定位多模态处理服务。",
    )

    ##############################################################################################################
    # Cache Store 配置
    ##############################################################################################################
    cache_store_group = parser.add_argument_group("Cache Store")
    cache_store_group.add_argument(
        "--cache_store_rdma_mode",
        env_name="CACHE_STORE_RDMA_MODE",
        type=str2bool,
        default=True,
        help="控制 cache store 是否使用 RDMA 模式。",
    )

    cache_store_group.add_argument(
        "--wrr_available_ratio",
        env_name="WRR_AVAILABLE_RATIO",
        type=int,
        default=80,
        help="为 WRR (Weighted Round Robin) 负载均衡器设置的可用性阈值百分比 (0-100)，数值越低越容易启用动态权重分配，但可能降低全局负载均衡准确性。",
    )

    cache_store_group.add_argument(
        "--rank_factor",
        env_name="RANK_FACTOR",
        type=int,
        default=0,
        choices=[0, 1],
        help="指定 WRR 负载均衡器使用的排序因子。`0` 表示基于 KV_CACHE 使用情况排序，`1` 表示基于正在处理的请求数 (ONFLIGHT_REQUESTS) 排序。",
    )
    cache_store_group.add_argument(
        "--cache_store_thread_count",
        env_name="CACHE_STORE_THREAD_COUNT",
        type=int,
        default=16,
        help="为 cache store 线程池设置线程数量。",
    )

    cache_store_group.add_argument(
        "--cache_store_rdma_connect_timeout_ms",
        env_name="CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS",
        type=int,
        default=250,
        help="为 cache store RDMA 连接设置超时时间，单位为毫秒。",
    )

    cache_store_group.add_argument(
        "--cache_store_rdma_qp_count_per_connection",
        env_name="CACHE_STORE_RDMA_QP_COUNT_PER_CONNECTION",
        type=int,
        default=2,
        help="为 cache store RDMA 连接设置每个连接的底层QP数量。",
    )

    ##############################################################################################################
    # 调度器配置
    ##############################################################################################################
    scheduler_group = parser.add_argument_group("Scheduler")
    scheduler_group.add_argument(
        "--use_batch_decode_scheduler",
        env_name="USE_BATCH_DECODE_SCHEDULER",
        type=str2bool,
        default=False,
        help="若为 True，则启用一个专门为decode阶段优化的特化调度器。此调度器在 decode 期间以固定大小的 batch 处理请求。若为 False，系统将使用一个 FIFO-based的默认调度器，默认调度器采用continuous batching。",
    )

    ##############################################################################################################
    # FIFO 调度器配置
    ##############################################################################################################
    fifo_scheduler_group = parser.add_argument_group("FIFO Scheduler")
    fifo_scheduler_group.add_argument(
        "--max_context_batch_size",
        env_name="MAX_CONTEXT_BATCH_SIZE",
        type=int,
        default=1,
        help="max_context_batch_size * max_seq_len 表示限制prefill请求的凑批大小。",
    )
    fifo_scheduler_group.add_argument(
        "--scheduler_reserve_resource_ratio",
        env_name="SCHEDULER_RESERVE_RESOURCE_RATIO",
        type=int,
        default=5,
        help="默认调度器将尝试保留的 KV cache blocks 的最小百分比。这有助于应对突发请求模式，为高优先级请求预留空间，或防止系统性能颠簸。",
    )

    fifo_scheduler_group.add_argument(
        "--enable_fast_gen",
        env_name="ENABLE_FAST_GEN",
        type=str2bool,
        default=False,
        help="若为 True，长请求会被拆分为chunks并分步处理。这主要用于提高长序列或流式输入的处理效率，并能改善并发场景下其他请求的交互性。注意：仅在使用默认调度器时有效。",
    )

    fifo_scheduler_group.add_argument(
        "--fast_gen_context_budget",
        env_name="FAST_GEN_MAX_CONTEXT_LEN",  # 和参数名不一致
        type=int,
        help="当 ENABLE_FAST_GEN 启用时，拆分成的chunk的大小。注意：仅当 ENABLE_FAST_GEN 为 True 且使用默认调度器时有效。",
    )
    fifo_scheduler_group.add_argument(
        "--enable_partial_fallback",
        env_name="ENABLE_PARTIAL_FALLBACK",
        type=str2bool,
        default=False,
        help="若为 True，则允许默认调度器在系统内存不足以满足活动请求时，从某些请求中回收一部分 KV cache blocks。这可以在高负载下提高系统利用率，但可能会影响那些资源被回收的请求的公平性。注意：在使用默认调度器时有效。",
    )

    ##############################################################################################################
    # BatchDecode 调度器配置
    ##############################################################################################################
    batch_decode_scheduler_group = parser.add_argument_group("BatchDecode Scheduler")
    batch_decode_scheduler_group.add_argument(
        "--batch_decode_scheduler_batch_size",
        env_name="BATCH_DECODE_SCHEDULER_BATCH_SIZE",
        type=int,
        default=1,
        help="当 USE_BATCH_DECODE_SCHEDULER 为 True 时，decode 阶段单次处理迭代中将组合在一起的请求数量。增加此值可以提高系统的整体 throughput，但代价是单个请求的 latency 可能会增加。减小此值可以降低 latency，但可能无法充分利用硬件资源。约束：整数 > 0。仅当 USE_BATCH_DECODE_SCHEDULER 为 True 时有效。",
    )

    ##############################################################################################################
    # Gang Configuration
    ##############################################################################################################
    gang_group = parser.add_argument_group("Gang Configuration")
    gang_group.add_argument(
        "--fake_gang_env",
        env_name="FAKE_GANG_ENV",
        type=str2bool,
        default=False,
        help="在多机启动时的fake行为",
    )
    gang_group.add_argument(
        "--gang_annocation_path",
        env_name="GANG_ANNOCATION_PATH",
        type=str,
        default="/etc/podinfo/annotations",
        help="GANG信息的路径",
    )
    gang_group.add_argument(
        "--gang_config_string",
        env_name="GANG_CONFIG_STRING",
        type=str,
        default=None,
        help="GAG信息的字符串表达",
    )
    gang_group.add_argument(
        "--zone_name", env_name="ZONE_NAME", type=str, default="", help="角色名"
    )
    gang_group.add_argument(
        "--distribute_config_file",
        env_name="DISTRIBUTE_CONFIG_FILE",
        type=str,
        default=None,
        help="分布式的配置文件路径",
    )
    gang_group.add_argument(
        "--dist_barrier_timeout",
        env_name="DIST_BARRIER_TIMEOUT",
        type=int,
        default=45,
        help="心跳检测的超时时间",
    )
    gang_group.add_argument(
        "--gang_sleep_time",
        env_name="GANG_SLEEP_TIME",
        type=int,
        default=10,
        help="心跳检测的间隔时间",
    )
    gang_group.add_argument(
        "--gang_timeout_min",
        env_name="GANG_TIMEOUT_MIN",
        type=int,
        default=30,
        help="心跳超时的最小时间",
    )

    ##############################################################################################################
    # Vit Configuration
    ##############################################################################################################
    vit_group = parser.add_argument_group("Vit Configuration")
    vit_group.add_argument(
        "--vit_separation",
        env_name="VIT_SEPARATION",
        type=int,
        default=0,
        help="VIT是否和主进程进行分离",
    )
    vit_group.add_argument(
        "--vit_trt", env_name="VIT_TRT", type=int, default=0, help="VIT是否使用TRT库"
    )
    vit_group.add_argument(
        "--trt_cache_enabled",
        env_name="TRT_CACHE_ENABLED",
        type=int,
        default=0,
        help="是否使用TRT_CACHE",
    )
    vit_group.add_argument(
        "--trt_cache_path",
        env_name="TRT_CACHE_PATH",
        type=str,
        default=os.path.join(os.getcwd(), "trt_cache"),
        help="TRT_CACHE路径",
    )
    vit_group.add_argument(
        "--download_headers",
        env_name="DOWNLOAD_HEADERS",
        type=str,
        default="",
        help="是否需要下载headers",
    )
    vit_group.add_argument(
        "--mm_cache_item_num",
        env_name="MM_CACHE_ITEM_NUM",
        type=int,
        default=10,
        help="多模态开启的Cache的大小",
    )
    vit_group.add_argument(
        "--url_cache_item_num",
        env_name="URL_CACHE_ITEM_NUM",
        type=int,
        default=100,
        help="多模态开启的用于URL的Cache的大小",
    )

    ##############################################################################################################
    # Server Configuration
    ##############################################################################################################
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument(
        "--frontend_server_count",
        env_name="FRONTEND_SERVER_COUNT",
        type=int,
        default=4,
        help="前端服务器启动进程数量",
    )
    server_group.add_argument(
        "--start_port",
        env_name="START_PORT",
        type=int,
        default=8088,
        help="服务启动端口",
    )
    server_group.add_argument(
        "--timeout_keep_alive",
        env_name="TIMEOUT_KEEP_ALIVE",
        type=int,
        default=5,
        help="健康检查的超时时间",
    )
    server_group.add_argument(
        "--frontend_server_id",
        env_name="FRONTEND_SERVER_ID",
        type=int,
        default=0,
        help="前端服务器序号",
    )

    ##############################################################################################################
    # Generate Configuration
    ##############################################################################################################
    generate_group = parser.add_argument_group("Generate Configuration")
    generate_group.add_argument(
        "--think_end_tag",
        env_name="THINK_END_TAG",
        type=str,
        default="</think>\n\n",
        help="深度思考模式的结束标签",
    )
    generate_group.add_argument(
        "--think_end_token_id",
        env_name="THINK_END_TOKEN_ID",
        type=int,
        default=-1,
        help="深度思考模式的结束标签的 TOKEN_ID",
    )
    generate_group.add_argument(
        "--think_mode",
        env_name="THINK_MODE",
        type=int,
        default=0,
        help="深度思考模式是否开启",
    )
    generate_group.add_argument(
        "--force_stop_words",
        env_name="FORCE_STOP_WORDS",
        type=str2bool,
        default=False,
        help="是否开启使用环境变量强制指定模型的STOP WORDS",
    )
    generate_group.add_argument(
        "--stop_words_list",
        env_name="STOP_WORDS_LIST",
        type=str,
        default=None,
        help="STOP_WORDS的TokenID列表",
    )
    generate_group.add_argument(
        "--stop_words_str",
        env_name="STOP_WORDS_STR",
        type=str,
        default=None,
        help="STOP_WORDS的string明文",
    )
    generate_group.add_argument(
        "--think_start_tag",
        env_name="THINK_START_TAG",
        type=str,
        default="<think>\\n",
        help="深度思考模式的起始标签",
    )
    generate_group.add_argument(
        "--generation_config_path",
        env_name="GENERATION_CONFIG_PATH",
        type=str,
        default=None,
        help="生成配置路径",
    )

    ##############################################################################################################
    # Quantization Configuration
    ##############################################################################################################
    quantization_group = parser.add_argument_group("Quantization Configuration")
    quantization_group.add_argument(
        "--int8_mode",
        env_name="INT8_MODE",
        type=int,
        default=0,
        help="权重类型是否使用int8模式",
    )
    quantization_group.add_argument(
        "--quantization", env_name="QUANTIZATION", type=str, default=None, help=""
    )

    ##############################################################################################################
    # PD Separation Configuration
    ##############################################################################################################
    # todo

    ##############################################################################################################
    # Sparse Configuration
    ##############################################################################################################
    sparse_group = parser.add_argument_group("Sparse Configuration")
    sparse_group.add_argument(
        "--sparse_config_file",
        env_name="SPARSE_CONFIG_FILE",
        type=str,
        default=None,
        help="稀疏模型的配置文件路径",
    )

    ##############################################################################################################
    # Engine Configuration
    ##############################################################################################################
    engine_group = parser.add_argument_group("Engine Configuration")
    engine_group.add_argument(
        "--warm_up",
        env_name="WARM_UP",
        type=int,
        default=1,
        help="在服务启动时是否开启预热",
    )
    engine_group.add_argument(
        "--warm_up_with_loss",
        env_name="WARM_UP_WITH_LOSS",
        type=int,
        default=0,
        help="在服务启动时是否开启损失去预热",
    )
    engine_group.add_argument(
        "--max_seq_len",
        env_name="MAX_SEQ_LEN",
        type=int,
        default=0,
        help="输入输出的最大长度",
    )

    ##############################################################################################################
    # Embedding Configuration
    ##############################################################################################################
    embedding_group = parser.add_argument_group("Embedding Configuration")
    embedding_group.add_argument(
        "--embedding_model",
        env_name="EMBEDDING_MODEL",
        type=int,
        default=0,
        help="嵌入模型的具体类型",
    )
    embedding_group.add_argument(
        "--extra_input_in_mm_embedding",
        env_name="EXTRA_INPUT_IN_MM_EMBEDDING",
        type=str,
        default=None,
        help='在多模态嵌入中使用额外的输入，可选值"INDEX"',
    )

    ##############################################################################################################
    # Worker Configuration
    ##############################################################################################################
    worker_group = parser.add_argument_group("Worker Configuration")
    worker_group.add_argument(
        "--worker_info_port_num",
        env_name="WORKER_INFO_PORT_NUM",
        type=int,
        default=7,
        help="worker的总的端口的数量",
    )

    ##############################################################################################################
    # Model Configuration
    ##############################################################################################################
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--extra_data_path",
        env_name="EXTRA_DATA_PATH",
        type=str,
        default=None,
        help="额外的数据路径",
    )
    model_group.add_argument(
        "--local_extra_data_path",
        env_name="LOCAL_EXTRA_DATA_PATH",
        type=str,
        default=None,
        help="本地额外数据路径",
    )
    model_group.add_argument(
        "--tokenizer_path",
        env_name="TOKENIZER_PATH",
        type=str,
        default=None,
        help="分词器的路径",
    )
    model_group.add_argument(
        "--act_type",
        env_name="ACT_TYPE",
        type=str,
        default="FP16",
        help="计算使用的数据类型",
    )
    model_group.add_argument(
        "--use_float32",
        env_name="USE_FLOAT32",
        type=str2bool,
        default=False,
        help="是否使用FP32",
    )
    model_group.add_argument(
        "--original_checkpoint_path",
        env_name="ORIGINAL_CHECKPOINT_PATH",
        type=str,
        default=None,
        help="原始的checkpoint的路径",
    )
    model_group.add_argument(
        "--mla_ops_type",
        env_name="MLA_OPS_TYPE",
        type=str,
        default="AUTO",
        help="Multi Latent Attention的操作类型",
    )
    model_group.add_argument(
        "--parallel_batch",
        env_name="PARALLEL_BATCH",
        type=int,
        default=0,
        help="Batch推理时采用串行还是并行",
    )
    model_group.add_argument(
        "--ft_plugin_path",
        env_name="FT_PLUGIN_PATH",
        type=str,
        default=None,
        help="插件路径",
    )
    model_group.add_argument(
        "--weight_type",
        env_name="WEIGHT_TYPE",
        type=str,
        default=None,
        help="模型权重类型",
    )
    model_group.add_argument(
        "--task_type", env_name="TASK_TYPE", type=str, default=None, help="任务类型"
    )
    model_group.add_argument(
        "--model_type", env_name="MODEL_TYPE", type=str, default=None, help="模型类型"
    )
    model_group.add_argument(
        "--checkpoint_path",
        env_name="CHECKPOINT_PATH",
        type=str,
        default=None,
        help="Checkpoint路径",
    )
    model_group.add_argument(
        "--oss_endpoint",
        env_name="OSS_ENDPOINT",
        type=str,
        default=None,
        help="OSS端点",
    )
    model_group.add_argument(
        "--ptuning_path",
        env_name="PTUNING_PATH",
        type=str,
        default=None,
        help="PTuning路径",
    )
    model_group.add_argument(
        "--dashscope_api_key",
        env_name="DASHSCOPE_API_KEY",
        type=str,
        default="EMPTY",
        help="Dashscope API Key",
    )
    model_group.add_argument(
        "--dashscope_http_url",
        env_name="DASHSCOPE_HTTP_URL",
        type=str,
        default=None,
        help="Dashscope HTTP URL",
    )
    model_group.add_argument(
        "--dashscope_websocket_url",
        env_name="DASHSCOPE_WEBSOCKET_URL",
        type=str,
        default=None,
        help="Dashscope WebSocket URL",
    )
    model_group.add_argument(
        "--openai_api_key",
        env_name="OPENAI_API_KEY",
        type=str,
        default="EMPTY",
        help="OpenAI API Key",
    )

    ##############################################################################################################
    # Lora Configuration
    ##############################################################################################################
    lora_group = parser.add_argument_group("Lora Configuration")
    lora_group.add_argument(
        "--lora_info", env_name="LORA_INFO", type=str, default="{}", help="Lora的信息"
    )
    lora_group.add_argument(
        "--merge_lora",
        env_name="MERGE_LORA",
        type=str2bool,
        default=True,
        help="Lora合并",
    )

    ##############################################################################################################
    # Load Configuration
    ##############################################################################################################
    load_group = parser.add_argument_group("Load Configuration")
    load_group.add_argument(
        "--phy2log_path",
        env_name="PHY2LOG_PATH",
        type=str,
        default="",
        help="python日志输出路径",
    )
    load_group.add_argument(
        "--converter_num_per_gpu",
        env_name="CONVERTER_NUM_PER_GPU",
        type=int,
        default=4,
        help="每个GPU做多少个转化",
    )
    load_group.add_argument(
        "--tokenizers_parallelism",
        env_name="TOKENIZERS_PARALLELISM",
        type=str2bool,
        default=False,
        help="分词器并行度",
    )
    load_group.add_argument(
        "--load_ckpt_num_process",
        env_name="LOAD_CKPT_NUM_PROCESS",
        type=int,
        default=0,
        help="加载Checkpoint的进程数量",
    )

    ##############################################################################################################
    # Render Configuration
    ##############################################################################################################
    render_group = parser.add_argument_group("Render Configuration")
    render_group.add_argument(
        "--model_template_type",
        env_name="MODEL_TEMPLATE_TYPE",
        type=str,
        default=None,
        help="模型的模版类型",
    )
    render_group.add_argument(
        "--default_chat_template_key",
        env_name="DEFAULT_CHAT_TEMPLATE_KEY",
        type=str,
        default="default",
        help="OpenAI的chat模型键",
    )
    render_group.add_argument(
        "--default_tool_use_template_key",
        env_name="DEFAULT_TOOL_USE_TEMPLATE_KEY",
        type=str,
        default="tool_use",
        help="默认工具使用的模板的key",
    )
    render_group.add_argument(
        "--llava_chat_template",
        env_name="LLAVA_CHAT_TEMPLATE",
        type=str,
        default="",
        help="LLava模型的会话模板",
    )

    ##############################################################################################################
    # Miscellaneous 配置
    ##############################################################################################################
    misc_group = parser.add_argument_group("Miscellaneous")

    misc_group.add_argument(
        "--load_balance",
        env_name="LOAD_BALANCE",
        type=str2bool,
        default=False,
        help="当设置为true时，启用基于吞吐量和延迟的动态并发控制；否则使用固定并发数。",
    )

    misc_group.add_argument(
        "--step_records_time_range",
        env_name="STEP_RECORDS_TIME_RANGE",
        type=int,
        default=60 * 1000 * 1000,
        help="性能记录 (step records) 的保留时间窗口，单位为微秒。例如，默认值 `60000000` 表示保留最近1分钟的记录。",
    )

    misc_group.add_argument(
        "--step_records_max_size",
        env_name="STEP_RECORDS_MAX_SIZE",
        type=int,
        default=1000,
        help="保留的性能记录 (step records) 的最大条数。与 `STEP_RECORDS_TIME_RANGE` 共同决定记录的保留策略。",
    )

    misc_group.add_argument(
        "--disable_pdl",
        env_name="DISABLE_PDL",
        type=str2bool,
        default=False,
        help="是否禁用PDL",
    )

    ##############################################################################################################
    # 3FS 配置
    ##############################################################################################################
    threefs_group = parser.add_argument_group("3FS")
    threefs_group.add_argument(
        "--enable_3fs",
        env_name="ENABLE_3FS",
        type=str2bool,
        default=False,
        help="是否启用 3FS 管理 KVCache",
    )

    ##############################################################################################################
    #  Role配置
    ##############################################################################################################
    role_group = parser.add_argument_group("Role")
    role_group.add_argument(
        "--role_type",
        env_name="ROLE_TYPE",
        type=str,
        default="PDFUSION",
        help="role的类型: 包含PDFUSION / PREFILL / DECODE / VIT / FRONTEND 几种类型",
    )

    parser.parse_args()

    parser.print_env_mappings()
    StaticConfig.update_from_env()
