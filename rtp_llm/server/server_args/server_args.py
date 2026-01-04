import argparse
import logging
import os
import sys
from argparse import Namespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from rtp_llm.config.py_config_modules import PyEnvConfigs
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
from rtp_llm.server.server_args.speculative_decoding_group_args import (
    init_speculative_decoding_group_args,
)
from rtp_llm.server.server_args.threefs_group_args import init_threefs_group_args
from rtp_llm.server.server_args.vit_group_args import init_vit_group_args

_T = TypeVar("_T")


class ConfigBinding:
    """配置绑定描述符，用于将解析的参数值绑定到配置对象"""

    def __init__(
        self,
        action: argparse.Action,
        bind_to: Union[Tuple[Any, str], str, List[Union[Tuple[Any, str], str]]],
    ):
        """
        Args:
            action: argparse.Action 对象
            bind_to: 绑定目标，可以是 (config_obj, 'attr_name') 元组、'path.to.attr' 字符串，或这些的列表
        """
        self.action = action
        self.dest = action.dest
        self.bind_to = bind_to
        self._resolved_bind_to: Optional[List[Tuple[Any, str]]] = None

    def resolve_bind_to(self, root_config: Any) -> List[Tuple[Any, str]]:
        """解析绑定目标，返回 (config_obj, attr_name) 元组列表"""
        if self._resolved_bind_to is not None:
            return self._resolved_bind_to

        resolved = []

        # Handle list of bindings
        bind_to_list = (
            self.bind_to if isinstance(self.bind_to, list) else [self.bind_to]
        )

        for bind_target in bind_to_list:
            if isinstance(bind_target, tuple) and len(bind_target) == 2:
                # 直接是 (config_obj, 'attr_name') 形式
                config_obj, attr_name = bind_target
                resolved.append((config_obj, attr_name))
            elif isinstance(bind_target, str):
                # 字符串路径形式，如 'server_config.frontend_server_count'
                parts = bind_target.split(".")
                config_obj = root_config
                for part in parts[:-1]:
                    config_obj = getattr(config_obj, part)
                attr_name = parts[-1]
                resolved.append((config_obj, attr_name))
            else:
                raise ValueError(f"Invalid bind_to format: {bind_target}")

        self._resolved_bind_to = resolved
        return resolved

    def apply(self, value: Any, root_config: Any) -> None:
        """应用绑定：将值设置到配置对象"""
        bindings = self.resolve_bind_to(root_config)
        for config_obj, attr_name in bindings:
            setattr(config_obj, attr_name, value)


class EnvArgumentGroup:
    def __init__(self, group: argparse._ArgumentGroup, parser: "EnvArgumentParser"):
        self._group = group
        self._parser = parser

    def add_argument(
        self,
        *args,
        env_name: Optional[str] = None,
        bind_to: Optional[Union[Tuple[Any, str], str]] = None,
        **kwargs,
    ) -> argparse.Action:
        """
        Add an argument to the group.

        Args:
            *args: 标准 argparse add_argument 参数
            env_name: 环境变量名称（保留用于兼容，但不再自动更新到 os.environ）
            bind_to: 配置绑定目标，可以是 (config_obj, 'attr_name') 或 'path.to.attr' 字符串
            **kwargs: 其他 argparse add_argument 参数
        """
        if "metavar" not in kwargs and "type" in kwargs:
            type_ = kwargs["type"]
            if isinstance(type_, type) and issubclass(type_, bool):
                kwargs["metavar"] = "BOOL"
            elif isinstance(type_, type) and issubclass(type_, int):
                kwargs["metavar"] = "INT"
            elif isinstance(type_, type) and issubclass(type_, str):
                kwargs["metavar"] = "STR"
        action = self._group.add_argument(*args, **kwargs)

        # 注册配置绑定
        if bind_to is not None:
            self._parser._register_config_binding(action, bind_to)

        # 保留 env 映射（用于兼容和日志）
        self._parser._register_env_mapping(action, args, env_name)
        return action

    def __getattr__(self, name):
        return getattr(self._group, name)


class EnvArgumentParser(argparse.ArgumentParser):
    _env_mappings: Dict[str, str] = {}

    def __init__(self, *args, env_prefix: str = "", **kwargs):
        self.env_prefix = env_prefix.upper()
        self._groups: Dict[str, EnvArgumentGroup] = {}
        self._config_bindings: List[ConfigBinding] = []  # 配置绑定列表
        self._root_config: Optional[Any] = None  # 根配置对象（PyEnvConfigs）

        super().__init__(*args, **kwargs)

        self._default_group = EnvArgumentGroup(self._positionals, self)
        self._optional_group = EnvArgumentGroup(self._optionals, self)

    def set_root_config(self, root_config: Any) -> None:
        """设置根配置对象，用于解析字符串路径形式的 bind_to"""
        self._root_config = root_config

    def _register_config_binding(
        self, action: argparse.Action, bind_to: Union[Tuple[Any, str], str]
    ) -> None:
        """注册参数到配置对象的绑定关系"""
        binding = ConfigBinding(action, bind_to)
        self._config_bindings.append(binding)

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
        logging.info("Parsing arguments and applying config bindings...")

        # If args is None, check if we should read from environment variables
        # argparse will use sys.argv when args is None, so we need to check sys.argv first
        has_cmd_args = args is not None or (len(sys.argv) > 1)

        if args is None:
            # Check if there are command line arguments (more than just program name)
            if not has_cmd_args:
                # No command line arguments, read from environment variables and construct args list
                args = []
                # Read values from environment variables for all registered arguments
                for dest, env_name in self._env_mappings.items():
                    env_value = os.environ.get(env_name)
                    if env_value is not None:
                        # Find the action for this dest
                        action = None
                        for action_item in self._actions:
                            if (
                                hasattr(action_item, "dest")
                                and action_item.dest == dest
                            ):
                                action = action_item
                                break

                        if action is not None:
                            # Get the option string (e.g., "--model_type")
                            option_string = None
                            for option in action.option_strings:
                                if option.startswith("--"):
                                    option_string = option
                                    break

                            if option_string:
                                args.extend([option_string, env_value])
            # If has_cmd_args is True, args remains None and argparse will use sys.argv

        parsed_args = super().parse_args(args, namespace)

        # After parsing, if there were command line arguments, fill in missing values from environment variables
        # This allows mixing command line arguments and environment variables
        if has_cmd_args:
            # Build a set of argument names that were provided via command line
            provided_args = set()
            if args is not None:
                # If args was provided, check which arguments are in the args list
                i = 0
                while i < len(args):
                    arg = args[i]
                    if arg.startswith("--"):
                        # Find the action for this option
                        for action_item in self._actions:
                            if arg in action_item.option_strings:
                                provided_args.add(action_item.dest)
                                # Check if this action requires a value
                                if action_item.nargs in (None, "?", 1):
                                    # Skip the value if present
                                    if i + 1 < len(args) and not args[i + 1].startswith(
                                        "-"
                                    ):
                                        i += 1
                                break
                    i += 1
            else:
                # If args is None, argparse used sys.argv, so check sys.argv
                i = 1  # Skip program name
                while i < len(sys.argv):
                    arg = sys.argv[i]
                    if arg.startswith("--"):
                        # Find the action for this option
                        for action_item in self._actions:
                            if arg in action_item.option_strings:
                                provided_args.add(action_item.dest)
                                # Check if this action requires a value
                                if action_item.nargs in (None, "?", 1):
                                    # Skip the value if present
                                    if i + 1 < len(sys.argv) and not sys.argv[
                                        i + 1
                                    ].startswith("-"):
                                        i += 1
                                break
                    i += 1

            # Now fill in missing values from environment variables
            for dest, env_name in self._env_mappings.items():
                # Only set from environment if the value wasn't provided via command line
                if dest not in provided_args:
                    env_value = os.environ.get(env_name)
                    if env_value is not None:
                        # Find the action to get the type converter
                        action = None
                        for action_item in self._actions:
                            if (
                                hasattr(action_item, "dest")
                                and action_item.dest == dest
                            ):
                                action = action_item
                                break

                        if action is not None:
                            # Convert the value using the action's type
                            if action.type is not None:
                                try:
                                    converted_value = action.type(env_value)
                                    setattr(parsed_args, dest, converted_value)
                                except (ValueError, TypeError):
                                    # If conversion fails, skip this value
                                    pass
                            else:
                                # No type converter, use as string
                                setattr(parsed_args, dest, env_value)

        # 应用所有配置绑定
        if self._root_config is not None:
            self._apply_config_bindings(parsed_args)

        # 不再自动更新 os.environ，但保留日志记录（用于调试）
        for dest, env_name in self._env_mappings.items():
            value = getattr(parsed_args, dest, None)
            if value is not None:
                env_value: str
                if isinstance(value, bool):
                    env_value = "1" if value else "0"
                elif isinstance(value, list):
                    env_value = ",".join(map(str, value))
                else:
                    env_value = str(value)
                logging.debug(f"[EnvMapping] {env_name} = {env_value}")

        return parsed_args

    def _apply_config_bindings(self, parsed_args: argparse.Namespace) -> None:
        """应用所有配置绑定，将解析的参数值设置到配置对象"""
        for binding in self._config_bindings:
            value = getattr(parsed_args, binding.dest, None)
            if value is not None:
                try:
                    binding.apply(value, self._root_config)
                    logging.debug(
                        f"[ConfigBinding] {binding.dest} -> {binding.bind_to} = {value}"
                    )
                except Exception as e:
                    logging.warning(
                        f"Failed to apply config binding for {binding.dest}: {e}"
                    )

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


def init_all_group_args(
    parser: EnvArgumentParser, py_env_configs: PyEnvConfigs
) -> None:
    """
    初始化所有参数组到解析器中，并绑定到配置对象

    Args:
        parser: EnvArgumentParser实例
        py_env_configs: PyEnvConfigs配置对象，用于绑定参数
    """
    init_batch_decode_scheduler_group_args(
        parser, py_env_configs.runtime_config.batch_decode_scheduler_config
    )
    init_cache_store_group_args(parser, py_env_configs.cache_store_config)
    init_concurrent_group_args(parser, py_env_configs.concurrency_config)
    init_device_resource_group_args(
        parser, py_env_configs.device_resource_config, py_env_configs.runtime_config
    )
    init_embedding_group_args(parser, py_env_configs.embedding_config)
    init_engine_group_args(parser, py_env_configs.runtime_config)
    init_fifo_scheduler_group_args(
        parser, py_env_configs.runtime_config.fifo_scheduler_config
    )
    init_fmha_group_args(parser, py_env_configs.fmha_config)
    init_gang_group_args(parser, py_env_configs.distribute_config)
    init_generate_group_args(parser, py_env_configs.generate_env_config)
    init_hw_kernel_group_args(parser, py_env_configs.py_hw_kernel_config)
    init_kv_cache_group_args(parser, py_env_configs.kv_cache_config)
    init_threefs_group_args(parser, py_env_configs.kv_cache_config)
    init_load_group_args(parser, py_env_configs.load_config, py_env_configs.model_args)
    init_lora_group_args(parser, py_env_configs.lora_config)
    init_misc_group_args(parser, py_env_configs.misc_config)
    init_model_group_args(parser, py_env_configs.model_args)
    init_model_specific_group_args(parser, py_env_configs.model_specific_config)
    init_moe_group_args(
        parser,
        py_env_configs.moe_config,
        py_env_configs.eplb_config,
        py_env_configs.deep_ep_config,
    )
    init_parallel_group_args(
        parser,
        py_env_configs.parallelism_config,
        py_env_configs.ffn_disaggregate_config,
    )
    init_profile_debug_logging_group_args(
        parser, py_env_configs.profiling_debug_logging_config
    )
    init_quantization_group_args(parser, py_env_configs.quantization_config)
    init_render_group_args(parser, py_env_configs.render_config)
    init_role_group_args(parser, py_env_configs.role_config)
    init_rpc_discovery_group_args(parser)
    init_sampling_group_args(parser, py_env_configs.sampler_config)
    init_scheduler_group_args(parser, py_env_configs.runtime_config)
    init_server_group_args(parser, py_env_configs.server_config)
    init_speculative_decoding_group_args(parser, py_env_configs.sp_config)
    init_vit_group_args(parser, py_env_configs.vit_config)
    init_jit_group_args(parser, py_env_configs.jit_config)
    init_pd_separation_group_args(parser, py_env_configs.pd_separation_config)
    init_grpc_group_args(parser, py_env_configs.grpc_config)


def setup_args() -> PyEnvConfigs:
    parser = EnvArgumentParser(description="RTP LLM")

    # 先创建配置对象
    py_env_configs = PyEnvConfigs()

    # 设置根配置对象，用于解析字符串路径形式的 bind_to
    parser.set_root_config(py_env_configs)

    # 使用统一的函数初始化所有参数组，并绑定到配置对象
    init_all_group_args(parser, py_env_configs)

    # 解析参数（会自动应用所有配置绑定）
    parsed_args = parser.parse_args()

    return py_env_configs
