import argparse
import glob
import logging
import os
from argparse import Namespace
from typing import Any, Dict, Optional, Sequence, Tuple, TypeVar

from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.server.server_args.batch_decode_scheduler_group_args import (
    init_batch_decode_scheduler_group_args,
)
from rtp_llm.server.server_args.cache_store_group_args import (
    init_cache_store_group_args,
)
from rtp_llm.server.server_args.concurrent_group_args import init_concurrent_group_args
from rtp_llm.server.server_args.device_resource_group_args import (
    init_device_resource_group_args,
)
from rtp_llm.server.server_args.embedding_group_args import init_embedding_group_args
from rtp_llm.server.server_args.engine_group_args import init_engine_group_args
from rtp_llm.server.server_args.fifo_scheduler_group_args import (
    init_fifo_scheduler_group_args,
)
from rtp_llm.server.server_args.fmha_group_args import init_fmha_group_args
from rtp_llm.server.server_args.gang_group_args import init_gang_group_args
from rtp_llm.server.server_args.generate_group_args import init_generate_group_args
from rtp_llm.server.server_args.grpc_group_args import init_grpc_group_args
from rtp_llm.server.server_args.hw_kernel_group_args import init_hw_kernel_group_args
from rtp_llm.server.server_args.jit_group_args import init_jit_group_args
from rtp_llm.server.server_args.kv_cache_group_args import init_kv_cache_group_args
from rtp_llm.server.server_args.load_group_args import init_load_group_args
from rtp_llm.server.server_args.lora_group_args import init_lora_group_args
from rtp_llm.server.server_args.misc_group_args import init_misc_group_args
from rtp_llm.server.server_args.model_group_args import init_model_group_args
from rtp_llm.server.server_args.model_specific_group_args import (
    init_model_specific_group_args,
)
from rtp_llm.server.server_args.moe_group_args import init_moe_group_args
from rtp_llm.server.server_args.parallel_group_args import init_parallel_group_args
from rtp_llm.server.server_args.pd_separation_group_args import (
    init_pd_separation_group_args,
)
from rtp_llm.server.server_args.profile_debug_logging_group_args import (
    init_profile_debug_logging_group_args,
)
from rtp_llm.server.server_args.quantization_group_args import (
    init_quantization_group_args,
)
from rtp_llm.server.server_args.render_group_args import init_render_group_args
from rtp_llm.server.server_args.role_group_args import init_role_group_args
from rtp_llm.server.server_args.rpc_discovery_group_args import (
    init_rpc_discovery_group_args,
)
from rtp_llm.server.server_args.sampling_group_args import init_sampling_group_args
from rtp_llm.server.server_args.scheduler_group_args import init_scheduler_group_args
from rtp_llm.server.server_args.server_group_args import init_server_group_args
from rtp_llm.server.server_args.sparse_group_args import init_sparse_group_args
from rtp_llm.server.server_args.speculative_decoding_group_args import (
    init_speculative_decoding_group_args,
)
from rtp_llm.server.server_args.vit_group_args import init_vit_group_args
from rtp_llm.server.server_args.worker_group_args import init_worker_group_args

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
    _env_mappings: Dict[str, str] = {}

    def __init__(self, *args, env_prefix: str = "", **kwargs):
        self.env_prefix = env_prefix.upper()
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

        EnvArgumentParser._env_mappings[action.dest] = full_env_name

    def parse_args(
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> argparse.Namespace:
        logging.info("Parsing arguments and setting environment variables...")
        parsed_args = super().parse_args(args, namespace)

        for dest, env_name in EnvArgumentParser._env_mappings.items():
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

    @staticmethod
    def update_env_from_args(
        parser: argparse.ArgumentParser,
        args_name: str,
        namespace: argparse.Namespace,
    ) -> None:
        env_name = EnvArgumentParser._env_mappings[args_name]
        value = getattr(namespace, args_name, None)
        if value is None:
            return None

        # 如果环境变量已存在且值等于默认值，则跳过
        if env_name in os.environ and value == parser.get_default(args_name):
            return None

        # 转换值为字符串形式
        env_value: str
        if isinstance(value, bool):
            env_value = "1" if value else "0"
        elif isinstance(value, list):
            env_value = ",".join(map(str, value))
        else:
            env_value = str(value)

        # 更新环境变量
        os.environ[env_name] = env_value
        logging.info(f"Updated environment variable: {env_name} = {env_value}")

    def print_env_mappings(self, group_name: Optional[str] = None) -> None:
        logging.info("Argument -> Environment Variable Mappings:")
        logging.info("-" * 50)

        if group_name:
            if group_name in self._groups:
                group = self._groups[group_name]._group
                for action in group._group_actions:
                    if action.dest in EnvArgumentParser._env_mappings:
                        logging.info(
                            f"{action.dest:<20} -> {EnvArgumentParser._env_mappings[action.dest]}"
                        )
            else:
                logging.info(f"Group '{group_name}' not found.")
        else:
            for dest, env_name in EnvArgumentParser._env_mappings.items():
                logging.info(f"{dest:<20} -> {env_name}")

        logging.info("-" * 50)

    def get_env_mappings(self, group_name: Optional[str] = None) -> Dict[str, str]:
        if group_name and group_name in self._groups:
            group = self._groups[group_name]._group
            mappings = {}
            for action in group._group_actions:
                if action.dest in EnvArgumentParser._env_mappings:
                    mappings[action.dest] = EnvArgumentParser._env_mappings[action.dest]
            return mappings
        else:
            return EnvArgumentParser._env_mappings.copy()


def init_all_group_args(parser: EnvArgumentParser) -> None:
    """
    初始化所有参数组到解析器中

    Args:
        parser: EnvArgumentParser实例
    """
    init_batch_decode_scheduler_group_args(parser)
    init_cache_store_group_args(parser)
    init_concurrent_group_args(parser)
    init_device_resource_group_args(parser)
    init_embedding_group_args(parser)
    init_engine_group_args(parser)
    init_fifo_scheduler_group_args(parser)
    init_fmha_group_args(parser)
    init_gang_group_args(parser)
    init_generate_group_args(parser)
    init_hw_kernel_group_args(parser)
    init_kv_cache_group_args(parser)
    # init_threefs_group_args(parser)
    init_load_group_args(parser)
    init_lora_group_args(parser)
    init_misc_group_args(parser)
    init_model_group_args(parser)
    init_model_specific_group_args(parser)
    init_moe_group_args(parser)
    init_parallel_group_args(parser)
    init_profile_debug_logging_group_args(parser)
    init_quantization_group_args(parser)
    init_render_group_args(parser)
    init_role_group_args(parser)
    init_rpc_discovery_group_args(parser)
    init_sampling_group_args(parser)
    init_scheduler_group_args(parser)
    init_server_group_args(parser)
    init_sparse_group_args(parser)
    init_speculative_decoding_group_args(parser)
    init_vit_group_args(parser)
    init_worker_group_args(parser)
    init_jit_group_args(parser)
    init_pd_separation_group_args(parser)
    init_grpc_group_args(parser)


def setup_args() -> tuple[EnvArgumentParser, Namespace]:
    parser = EnvArgumentParser(description="RTP LLM")

    # 使用统一的函数初始化所有参数组
    init_all_group_args(parser)

    args = parser.parse_args()

    # add rocm env config, if using default value, change it to optimize version
    if os.path.exists("/dev/kfd") and os.getenv("FT_DISABLE_CUSTOM_AR") is None:
        os.environ["FT_DISABLE_CUSTOM_AR"] = "0"
        logging.info(
            "[MI308X] enable FT_DISABLE_CUSTOM_AR by default, as amd has own implementation."
        )

    if os.path.exists("/dev/kfd") and os.getenv("SEQ_SIZE_PER_BLOCK") is None:
        os.environ["SEQ_SIZE_PER_BLOCK"] = "16"
        logging.info(
            "[MI308X] set SEQ_SIZE_PER_BLOCK 16 by default, as it just support 16 now."
        )

    if os.path.exists("/dev/kfd") and os.getenv("ENABLE_COMM_OVERLAP") is None:
        os.environ["ENABLE_COMM_OVERLAP"] = "0"
        logging.info("[MI308X] disable ENABLE_COMM_OVERLAP by default.")

    parser.print_env_mappings()
    StaticConfig.update_from_env()
    return parser, args
