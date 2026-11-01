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
    # 远端缓存根允许 $VAR/${VAR} 引用（例如 REMOTE_JIT_DIR=$HIPPO_APP_WORKDIR/main/jit_cache，
    # 由运行端自己解析，CI/客户端只传字面量）。os.path.expandvars 对未定义变量会原样保留 $，
    # 那样得到的是永远不存在的垃圾路径且会被 resolve_remote 的 fail-open 静默吞掉，
    # 所以引用了变量却残留 $ 时按配置错误报错。
    # REMOTE_JIT_DIR=auto 是 CI 安全形态：纯字母能穿透 CI 命令装配的每一层引用
    # （$VAR 引用会在 runner 上被展开成空）。gpu_lock wrapper 正常会先把它解析成
    # 绝对路径再传给子进程；这里兜底 wrapper 未运行的直接启动场景。
    if value == "auto":
        workdir = os.environ.get("HIPPO_APP_WORKDIR", "").strip()
        if not workdir:
            raise argparse.ArgumentTypeError(
                "REMOTE_JIT_DIR=auto requires HIPPO_APP_WORKDIR in the environment"
            )
        value = str(Path(workdir) / "main" / "jit_cache")
    expanded = os.path.expandvars(value)
    if "$" in value and "$" in expanded:
        raise argparse.ArgumentTypeError(
            f"unresolved environment variable reference: {value!r} -> {expanded!r}"
        )
    return expanded


def init_jit_group_args(parser, jit_config):
    ##############################################################################################################
    # JIT Configuration
    ##############################################################################################################
    jit_group = parser.add_argument_group("JIT Configuration")
    jit_group.add_argument(
        "--remote_jit_dir",
        env_name="REMOTE_JIT_DIR",
        bind_to=(jit_config, "remote_jit_dir"),
        type=_env_expanded_path,
        default="",
        help="JIT远程v1快照根（可信绝对路径或FUSE URI），支持$VAR/${VAR}环境变量引用"
        "（如$HIPPO_APP_WORKDIR/main/jit_cache，在解析端展开；引用未定义变量报错）或"
        "auto哨兵（由HIPPO_APP_WORKDIR解析出$HIPPO_APP_WORKDIR/main/jit_cache，纯字母可穿透CI命令装配层）；"
        "为空只关远端、仍用统一本地缓存/tmp/rtp-llm/.jit_cache；"
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
