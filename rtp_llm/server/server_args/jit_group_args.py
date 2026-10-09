import argparse
import math
from contextlib import suppress

from rtp_llm.server.server_args.util import str2bool


def _positive_or_unlimited(value: str) -> int:
    with suppress(ValueError):
        if (timeout := int(value)) == -1 or timeout > 0:
            return timeout
    raise argparse.ArgumentTypeError(f"must be a positive integer or -1, got {value!r}")


def _positive_float(value: str) -> float:
    with suppress(ValueError):
        if (result := float(value)) > 0 and math.isfinite(result):
            return result
    raise argparse.ArgumentTypeError(f"must be a positive number, got {value!r}")


def _positive_int(value: str) -> int:
    with suppress(ValueError):
        if (result := int(value)) > 0:
            return result
    raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")


def init_jit_group_args(parser, jit_config):
    ##############################################################################################################
    # JIT Configuration
    ##############################################################################################################
    jit_group = parser.add_argument_group("JIT Configuration")
    jit_group.add_argument(
        "--remote_jit_dir",
        env_name="REMOTE_JIT_DIR",
        bind_to=(jit_config, "remote_jit_dir"),
        type=str,
        default="",
        help="JIT远程v1快照根（可信绝对路径或FUSE URI）；为空只关远端、仍用统一本地缓存/tmp/rtp-llm/.jit_cache；"
        "完全关闭见--manage_jit_cache；预设组件cache环境变量可退出该组件的jit产物重定向",
    )
    jit_group.add_argument(
        "--jit_cache_setup_timeout_s",
        env_name="JIT_CACHE_SETUP_TIMEOUT_S",
        bind_to=(jit_config, "jit_cache_setup_timeout_s"),
        type=_positive_or_unlimited,
        default=180,
        help="JIT快照恢复等待秒数，正数或-1（不限时）；不含scope探测时间",
    )
    jit_group.add_argument(
        "--manage_jit_cache",
        env_name="MANAGE_JIT_CACHE",
        bind_to=(jit_config, "manage_jit_cache"),
        type=str2bool,
        default=True,
        help="默认启用JIT cache统一管理；传0则完全退出该特性",
    )
    jit_group.add_argument(
        "--startup_real_warmup_timeout_s",
        env_name="STARTUP_REAL_WARMUP_TIMEOUT_S",
        bind_to=(jit_config, "startup_real_warmup_timeout_s"),
        type=_positive_float,
        default=None,
        help="Prefill 启动 gRPC 预热单请求超时秒数；未设置时 DSV4 兼容旧环境变量，默认 600 秒",
    )
    jit_group.add_argument(
        "--startup_real_warmup_max_len",
        env_name="STARTUP_REAL_WARMUP_MAX_LEN",
        bind_to=(jit_config, "startup_real_warmup_max_len"),
        type=_positive_int,
        default=None,
        help="显式限制 Prefill 启动 gRPC 预热输入长度，不超过模型 max_seq_len",
    )
