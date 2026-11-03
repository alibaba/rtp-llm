import argparse
import os
from contextlib import suppress
from pathlib import Path

from rtp_llm.server.server_args.util import str2bool


def _positive_or_unlimited(value: str) -> int:
    with suppress(ValueError):
        if (timeout := int(value)) == -1 or timeout > 0:
            return timeout
    raise argparse.ArgumentTypeError(f"must be a positive integer or -1, got {value!r}")


def _env_expanded_path(value: str) -> str:
    # auto 哨兵是唯一自动解析形态：CI 传纯字母穿透命令装配层，运行端从
    # HIPPO_APP_WORKDIR 解析；其余值一律直接作为路径使用。
    if value == "auto":
        workdir = os.environ.get("HIPPO_APP_WORKDIR", "").strip()
        if not workdir:
            raise argparse.ArgumentTypeError(
                "REMOTE_JIT_DIR=auto requires HIPPO_APP_WORKDIR in the environment"
            )
        return str(Path(workdir) / "main" / "jit_cache")
    return value


def init_jit_group_args(parser, jit_config):
    ##############################################################################################################
    # JIT Configuration
    ##############################################################################################################
    jit_group = parser.add_argument_group("JIT Configuration")
    jit_group.add_argument(
        "--local_jit_dir",
        env_name="LOCAL_JIT_DIR",
        bind_to=(jit_config, "local_jit_dir"),
        type=str,
        default="",
        help="统一JIT本地cache根目录（自动追加版本子目录）；为空时使用/tmp/rtp-llm/.jit_cache。"
        "SCR可配置为/dev/shm/rtp-llm/.jit_cache，须可写且允许加载动态库",
    )
    jit_group.add_argument(
        "--remote_jit_dir",
        env_name="REMOTE_JIT_DIR",
        bind_to=(jit_config, "remote_jit_dir"),
        type=_env_expanded_path,
        default="",
        help="JIT远程v1快照根（绝对路径直接使用），或auto哨兵"
        "（由HIPPO_APP_WORKDIR解析出$HIPPO_APP_WORKDIR/main/jit_cache，纯字母可穿透CI命令装配层，"
        "CI 的标准形态）；为空只关远端、仍用统一本地缓存/tmp/rtp-llm/.jit_cache；"
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
