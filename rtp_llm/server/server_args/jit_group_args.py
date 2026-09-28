import argparse
from contextlib import suppress

from rtp_llm.server.server_args.util import str2bool


def _positive_or_unlimited(value: str) -> int:
    with suppress(ValueError):
        if (timeout := int(value)) == -1 or timeout > 0:
            return timeout
    raise argparse.ArgumentTypeError(f"must be a positive integer or -1, got {value!r}")


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
        help=(
            "JIT 远程 v1 snapshot 根目录（可信绝对路径或 FUSE URI）；为空时只使用"
            "统一本地 cache /tmp/rtp-llm/.jit_cache"
        ),
    )
    jit_group.add_argument(
        "--jit_cache_setup_timeout_s",
        env_name="JIT_CACHE_SETUP_TIMEOUT_S",
        bind_to=(jit_config, "jit_cache_setup_timeout_s"),
        type=_positive_or_unlimited,
        default=180,
        help="JIT snapshot 恢复等待秒数，正数或 -1（不限时）",
    )
    jit_group.add_argument(
        "--manage_jit_cache",
        env_name="MANAGE_JIT_CACHE",
        bind_to=(jit_config, "manage_jit_cache"),
        type=str2bool,
        default=True,
        help="是否启用统一 JIT cache 管理；传 0 完全关闭",
    )
