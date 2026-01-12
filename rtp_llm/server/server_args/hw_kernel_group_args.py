import logging
from typing import List

from rtp_llm.server.server_args.util import str2bool


def init_hw_kernel_group_args(parser, hw_kernel_config):
    ##############################################################################################################
    # 硬件/Kernel 特定优化
    ##############################################################################################################
    hw_kernel_group = parser.add_argument_group("硬件/Kernel 特定优化")

    hw_kernel_group.add_argument(
        "--enable_cuda_graph",
        env_name="ENABLE_CUDA_GRAPH",
        bind_to=(hw_kernel_config, 'enable_cuda_graph'),
        type=str2bool,
        default=False,
        help="系统是否允许使用Cuda Graph",
    )

    hw_kernel_group.add_argument(
        "--enable_cuda_graph_debug_mode",
        env_name="ENABLE_CUDA_GRAPH_DEBUG_MODE",
        bind_to=(hw_kernel_config, 'enable_cuda_graph_debug_mode'),
        type=str2bool,
        default=False,
        help="系统是否允许使用Cuda Graph开启Debug模式来生成可视化文件",
    )

    hw_kernel_group.add_argument(
        "--enable_native_cuda_graph",
        env_name="ENABLE_NATIVE_CUDA_GRAPH",
        bind_to=(hw_kernel_config, 'enable_native_cuda_graph'),
        type=str2bool,
        default=False,
        help="系统是否允许在C++后端使用Cuda Graph",
    )

    hw_kernel_group.add_argument(
        "--num_native_cuda_graph",
        env_name="NUM_NATIVE_CUDA_GRAPH",
        bind_to=(hw_kernel_config, 'num_native_cuda_graph'),
        type=int,
        default=200,
        help="C++后端缓存Cuda Graph数量",
    )

    hw_kernel_group.add_argument(
        "--deep_gemm_num_sm",
        env_name="DEEP_GEMM_NUM_SM",
        bind_to=(hw_kernel_config, 'deep_gemm_num_sm'),
        type=int,
        default=None,
        help="指定 DeepGEMM 使用的 SM (Streaming Multiprocessor) 数量。如果设置，此值将覆盖自动检测的数量。",
    )

    hw_kernel_group.add_argument(
        "--arm_gemm_use_kai",
        env_name="ARM_GEMM_USE_KAI",
        bind_to=(hw_kernel_config, 'arm_gemm_use_kai'),
        type=str2bool,
        default=False,
        help="设置为 `True` 时，为 ARM GEMM 操作启用 KleidiAI 支持。这可能影响权重处理和计算性能。",
    )

    hw_kernel_group.add_argument(
        "--enable_stable_scatter_add",
        env_name="ENABLE_STABLE_SCATTER_ADD",
        bind_to=(hw_kernel_config, 'enable_stable_scatter_add'),
        type=str2bool,
        default=False,
        help="控制是否启用稳定的 scatter add 操作。",
    )

    hw_kernel_group.add_argument(
        "--enable_multi_block_mode",
        env_name="ENABLE_MULTI_BLOCK_MODE",
        bind_to=(hw_kernel_config, 'enable_multi_block_mode'),
        type=str2bool,
        default=True,
        help="控制是否为 Multi-Head Attention (MMHA) 启用 multi-block 模式。设置为 'ON' 启用，'OFF' 禁用。",
    )

    hw_kernel_group.add_argument(
        "--rocm_hipblaslt_config",
        env_name="ROCM_HIPBLASLT_CONFIG",
        bind_to=(hw_kernel_config, 'rocm_hipblaslt_config'),
        type=str,
        default="gemm_config.csv",
        help="指定 hipBLASLt GEMM 配置文件的路径。此文件用于优化 ROCm平台上的 GEMM 操作。",
    )

    hw_kernel_group.add_argument(
        "--ft_disable_custom_ar",
        env_name="FT_DISABLE_CUSTOM_AR",
        bind_to=(hw_kernel_config, 'ft_disable_custom_ar'),
        type=str2bool,
        default=None,
        help="设置为 `True` 时，禁用自定义的 AllReduce (AR) 实现，可能回退到标准库（如 NCCL）的 AllReduce。",
    )

    hw_kernel_group.add_argument(
        "--use_swizzleA",
        env_name="USE_SWIZZLEA",
        bind_to=(hw_kernel_config, 'use_swizzleA'),
        type=str2bool,
        default=True,
        help="hipBLASLt GEMM 是否使用 swizzle",
    )

    hw_kernel_group.add_argument(
        "--prefill_capture_config",
        env_name="PREFILL_CAPTURE_CONFIG",
        type=_parse_prefill_capture_config,
        default="240:3",
        bind_to=(hw_kernel_config, 'prefill_capture_seq_lens'),
        help=(
            "Prefill CUDA Graph capture sequence lengths configuration. "
            "Supports three formats:\n"
            "  1. File path: starts with 'file://' or '/', e.g., 'file:///path/to/seq_lens.txt' or '/path/to/seq_lens.txt'\n"
            "  2. Comma-separated list: e.g., '10,100,500,1000,2000'\n"
            "  3. Range: format 'max:step', e.g., '16384:128' (generates [128, 256, ..., 16384])"
        ),
    )

    hw_kernel_group.add_argument(
        "--decode_capture_config",
        env_name="DECODE_CAPTURE_CONFIG",
        type=_parse_decode_capture_config,
        default="",
        bind_to=(hw_kernel_config, 'decode_capture_batch_sizes'),
        help=(
            "Decode CUDA Graph capture batch sizes configuration. "
            "Supports comma-separated list format, e.g., '1,2,4,8,16,32'. "
            "If not set, default logic will be used to generate batch sizes."
        ),
    )

    hw_kernel_group.add_argument(
        "--disable_dpc_random",
        env_name="DISABLE_DPC_RANDOM",
        bind_to=(hw_kernel_config, 'disable_dpc_random'),
        type=str2bool,
        default=None,
        help="控制是否禁用 DPC 的随机性",
    )

    hw_kernel_group.add_argument(
        "--rocm_disable_custom_ag",
        env_name="ROCM_DISABLE_CUSTOM_AG",
        bind_to=(hw_kernel_config, 'rocm_disable_custom_ag'),
        type=str2bool,
        default=None,
        help="设置为 `True` 时，禁用ROCm平台自定义的 AllGather (AG) 实现，可能回退到标准库（如 RCCL）的 AllGather。",
    )

def _parse_comma_separated_ints(
    config: str, config_name: str, item_name: str, raise_on_empty: bool = True
) -> List[int]:
    """
    Parse comma-separated list of positive integers from config string.

    Args:
        config: Configuration string containing comma-separated integers
        config_name: Name of the configuration (for error messages)
        item_name: Name of the items being parsed (e.g., "sequence lengths", "batch sizes")
        raise_on_empty: If True, raise ValueError when no valid items found; if False, return empty list

    Returns:
        List of positive integers parsed from the config string

    Raises:
        ValueError: If raise_on_empty=True and no valid items found, or if parsing fails
    """
    try:
        values = [int(x.strip()) for x in config.split(",") if x.strip()]
        values = [x for x in values if x > 0]
        if values:
            logging.info(
                f"Using {item_name} from comma-separated list: {len(values)} items"
            )
            return values
        else:
            # Extract base item name (last 2 words, e.g., "sequence lengths" from "prefill capture sequence lengths")
            words = item_name.split()
            base_item_name = (
                " ".join(words[-2:])
                if len(words) >= 2
                else words[-1] if words else item_name
            )
            error_msg = f"{config_name} contains no valid {base_item_name}"
            if raise_on_empty:
                raise ValueError(error_msg)
            else:
                logging.warning(f"{error_msg}, using default logic")
                return []
    except ValueError as e:
        # Check if the exception is from our own code (contains config_name)
        if config_name in str(e):
            # Re-raise our own exceptions as-is
            if raise_on_empty:
                raise
            else:
                logging.warning(f"{e}, using default logic")
                return []
        else:
            # For parsing errors (e.g., invalid int), use simpler message
            words = item_name.split()
            base_item_name = (
                " ".join(words[-2:])
                if len(words) >= 2
                else words[-1] if words else item_name
            )
            error_msg = f"{config_name} contains no valid {base_item_name}"
            if raise_on_empty:
                raise ValueError(error_msg)
            else:
                logging.warning(f"{error_msg}, using default logic")
                return []


def _parse_prefill_capture_config(config: str) -> List[int]:
    """
    Parse prefill capture sequence lengths configuration string.
    Supports three formats:
    1. File path: starts with 'file://' or '/', e.g., 'file:///path/to/seq_lens.txt' or '/path/to/seq_lens.txt'
    2. Comma-separated list: e.g., '10,100,500,1000,2000'
    3. Range: format 'max:step', e.g., '16384:128' (generates [128, 256, ..., 16384])

    This function MUST return a non-empty list. If no configuration is provided,
    it will raise an error.

    Args:
        config: Configuration string

    Returns:
        List of positive integers (sequence lengths)

    Raises:
        argparse.ArgumentTypeError: If configuration is invalid or empty
    """
    import argparse

    if not config:
        raise argparse.ArgumentTypeError(
            "prefill_capture_config must be set. Supported formats:\n"
            "  1. File path: 'file:///path/to/seq_lens.txt' or '/path/to/seq_lens.txt'\n"
            "  2. Comma-separated list: '10,100,500,1000,2000'\n"
            "  3. Range: '16384:128' (generates [128, 256, ..., 16384])"
        )

    config = config.strip()

    # Mode 1: File path (starts with 'file://' or '/')
    if config.startswith("file://") or config.startswith("/"):
        file_path = config[7:] if config.startswith("file://") else config
        try:
            seq_lens = []
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        try:
                            seq_len = int(line)
                            if seq_len > 0:
                                seq_lens.append(seq_len)
                        except ValueError:
                            logging.warning(
                                f"Invalid sequence length in file: {line}"
                            )
            if seq_lens:
                logging.info(
                    f"Loaded {len(seq_lens)} sequence lengths from {file_path}"
                )
                return seq_lens
            else:
                raise argparse.ArgumentTypeError(
                    f"No valid sequence lengths found in file: {file_path}"
                )
        except FileNotFoundError:
            raise argparse.ArgumentTypeError(f"Prefill capture file not found: {file_path}")
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Error reading prefill capture file: {e}")

    # Mode 3: Range format (max:step)
    if ":" in config:
        try:
            parts = config.split(":")
            if len(parts) != 2:
                raise ValueError("Range format must be 'max:step'")
            max_seq_len = int(parts[0].strip())
            step = int(parts[1].strip())
            if max_seq_len <= 0 or step <= 0:
                raise ValueError("max_seq_len and step must be positive integers")
            seq_lens = list(range(step, max_seq_len + 1, step))
            if max_seq_len not in seq_lens:
                seq_lens.append(max_seq_len)
            if seq_lens:
                logging.info(
                    f"Generated {len(seq_lens)} sequence lengths from range (step={step}, max={max_seq_len})"
                )
                return seq_lens
            else:
                raise ValueError(
                    f"Invalid range parameters: max_seq_len={max_seq_len}, step={step}"
                )
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid range format '{config}': {e}")

    # Mode 2: Comma-separated list (default)
    try:
        return _parse_comma_separated_ints(
            config,
            "prefill_capture_config",
            "prefill capture sequence lengths",
            raise_on_empty=True,
        )
    except ValueError as e:
        # Convert ValueError to ArgumentTypeError for argparse
        raise argparse.ArgumentTypeError(str(e))


def _parse_decode_capture_config(config: str) -> List[int]:
    """
    Parse decode capture batch sizes configuration string.
    Only supports comma-separated list format, e.g., '1,2,4,8,16,32'

    Returns empty list if no configuration is provided (will use default logic).

    Args:
        config: Configuration string

    Returns:
        List of positive integers (batch sizes), or empty list if config is empty
    """
    import argparse

    if not config:
        # Return empty list to use default logic
        return []

    config = config.strip()
    if not config:
        return []

    # Only support comma-separated list format
    try:
        return _parse_comma_separated_ints(
            config,
            "decode_capture_config",
            "decode capture batch sizes",
            raise_on_empty=False,
        )
    except ValueError as e:
        # Convert ValueError to ArgumentTypeError for argparse
        raise argparse.ArgumentTypeError(str(e))
