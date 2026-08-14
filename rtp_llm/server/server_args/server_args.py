import argparse
import logging
import os
import sys
from argparse import Namespace
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

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
from rtp_llm.server.server_args.grammar_group_args import init_grammar_group_args
from rtp_llm.server.server_args.grpc_group_args import (
    init_dash_sc_grpc_group_args,
    init_model_grpc_group_args,
)
from rtp_llm.server.server_args.hw_kernel_group_args import init_hw_kernel_group_args
from rtp_llm.server.server_args.jit_group_args import init_jit_group_args
from rtp_llm.server.server_args.kv_cache_group_args import init_kv_cache_group_args
from rtp_llm.server.server_args.load_group_args import init_load_group_args
from rtp_llm.server.server_args.lora_group_args import init_lora_group_args
from rtp_llm.server.server_args.master_group_args import init_master_group_args
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
from rtp_llm.server.server_args.repetition_detection_group_args import (
    init_repetition_detection_group_args,
)
from rtp_llm.server.server_args.role_group_args import init_role_group_args
from rtp_llm.server.server_args.rpc_discovery_group_args import (
    init_rpc_discovery_group_args,
)
from rtp_llm.server.server_args.scheduler_group_args import init_scheduler_group_args
from rtp_llm.server.server_args.server_group_args import init_server_group_args
from rtp_llm.server.server_args.speculative_decoding_group_args import (
    init_speculative_decoding_group_args,
)
from rtp_llm.server.server_args.util import argument_metavar
from rtp_llm.server.server_args.vit_group_args import init_vit_group_args

# ``argparse`` invokes selected subparsers through ``parse_known_args``. Keep
# one parse-local list so ``parse_args`` can defer every parser's external
# config side effects until the complete parser tree has rejected unknown
# arguments. ContextVar preserves correctness if independent parsers are used
# concurrently in different threads or async contexts.
_DEFERRED_PARSE_FINALIZERS: ContextVar[Optional[List[Tuple[Any, Namespace]]]] = (
    ContextVar("deferred_parse_finalizers", default=None)
)

_T = TypeVar("_T")


def _apply_default_metavar(kwargs: Dict[str, Any]) -> None:
    """Use bounded-type help text without hiding explicit choice lists."""

    if "metavar" in kwargs or "choices" in kwargs or "type" not in kwargs:
        return
    metavar = argument_metavar(kwargs["type"])
    if metavar is not None:
        kwargs["metavar"] = metavar


@dataclass(frozen=True)
class EnvArgSemantics:
    empty_as_unset: bool = False
    strict_choice: bool = False
    emit_string_from_env: bool = False
    strict_config_binding: bool = False


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
        bind_to: Optional[
            Union[Tuple[Any, str], str, List[Union[Tuple[Any, str], str]]]
        ] = None,
        empty_env_as_unset: bool = False,
        strict_env_choice: bool = False,
        emit_string_from_env: bool = False,
        strict_config_binding: bool = False,
        **kwargs,
    ) -> argparse.Action:
        """
        Add an argument to the group.

        Args:
            *args: 标准 argparse add_argument 参数
            env_name: 环境变量名称（保留用于兼容，但不再自动更新到 os.environ）
            bind_to: 配置绑定目标，可以是 (config_obj, 'attr_name')、
                'path.to.attr' 字符串或这些目标的列表
            empty_env_as_unset: 空环境变量是否按未设置处理
            strict_env_choice: pure-env 与 mixed CLI + env 两条路径是否严格校验
                choices，并在错误信息中标注来源环境变量
            emit_string_from_env: 是否允许部署参数生成器输出显式设置的
                字符串环境变量
            strict_config_binding: 配置绑定失败时是否终止解析；仅用于不能
                安全回退到旧配置的新增参数
            **kwargs: 其他 argparse add_argument 参数
        """
        _apply_default_metavar(kwargs)
        action = self._group.add_argument(*args, **kwargs)
        self._parser._register_env_semantics(
            action,
            empty_env_as_unset=empty_env_as_unset,
            strict_env_choice=strict_env_choice,
            emit_string_from_env=emit_string_from_env,
            strict_config_binding=strict_config_binding,
        )

        # 注册配置绑定
        if bind_to is not None:
            self._parser._register_config_binding(action, bind_to)

        # 保留 env 映射（用于兼容和日志）
        self._parser._register_env_mapping(action, args, env_name)
        return action

    def __getattr__(self, name):
        return getattr(self._group, name)


class EnvArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, env_prefix: str = "", **kwargs):
        self.env_prefix = env_prefix.upper()
        # Environment mappings belong to one parser. Keeping this mutable map
        # at class scope leaked arguments and prefixes across parser instances,
        # which could make a small parser inspect unrelated process env vars.
        self._env_mappings: Dict[str, str] = {}
        self._env_semantics: Dict[str, EnvArgSemantics] = {}
        self._groups: Dict[str, EnvArgumentGroup] = {}
        self._config_bindings: List[ConfigBinding] = []  # 配置绑定列表
        self._post_parse_validators: List[Callable[[], None]] = []
        self._root_config: Optional[Any] = None  # 根配置对象（PyEnvConfigs）

        super().__init__(*args, **kwargs)

        self._default_group = EnvArgumentGroup(self._positionals, self)
        self._optional_group = EnvArgumentGroup(self._optionals, self)

    def set_root_config(self, root_config: Any) -> None:
        """设置根配置对象，用于解析字符串路径形式的 bind_to"""
        self._root_config = root_config

    def register_post_parse_validator(self, validator: Callable[[], None]) -> None:
        """Register validation that runs after all config bindings are applied."""

        self._post_parse_validators.append(validator)

    def _register_config_binding(
        self,
        action: argparse.Action,
        bind_to: Union[Tuple[Any, str], str, List[Union[Tuple[Any, str], str]]],
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
        self,
        *args,
        env_name: Optional[str] = None,
        empty_env_as_unset: bool = False,
        strict_env_choice: bool = False,
        emit_string_from_env: bool = False,
        strict_config_binding: bool = False,
        **kwargs,
    ) -> argparse.Action:
        _apply_default_metavar(kwargs)
        if args and isinstance(args[0], str) and not args[0].startswith("-"):
            action = self._positionals.add_argument(*args, **kwargs)
        else:
            action = self._optionals.add_argument(*args, **kwargs)

        self._register_env_semantics(
            action,
            empty_env_as_unset=empty_env_as_unset,
            strict_env_choice=strict_env_choice,
            emit_string_from_env=emit_string_from_env,
            strict_config_binding=strict_config_binding,
        )
        self._register_env_mapping(action, args, env_name)
        return action

    def _register_env_semantics(
        self,
        action: argparse.Action,
        *,
        empty_env_as_unset: bool,
        strict_env_choice: bool,
        emit_string_from_env: bool,
        strict_config_binding: bool,
    ) -> None:
        self._env_semantics[action.dest] = EnvArgSemantics(
            empty_as_unset=empty_env_as_unset,
            strict_choice=strict_env_choice,
            emit_string_from_env=emit_string_from_env,
            strict_config_binding=strict_config_binding,
        )

    def _find_action(self, dest: str) -> Optional[argparse.Action]:
        return next((action for action in self._actions if action.dest == dest), None)

    @staticmethod
    def _display_option(action: argparse.Action) -> str:
        return action.option_strings[0] if action.option_strings else action.dest

    def _provided_option_dests(self, argv: Sequence[str]) -> Set[str]:
        """Return options explicitly present in argv using argparse's syntax."""

        # Mirror argparse's own O/A/- token pattern so options that consume
        # REMAINDER, PARSER, optional, or variable-length values hide exactly
        # the same following tokens here as they do during the real parse.
        argument_pattern: List[str] = []
        after_separator = False
        for token in argv:
            if after_separator:
                argument_pattern.append("A")
            elif token == "--":
                argument_pattern.append("-")
                after_separator = True
            else:
                argument_pattern.append(
                    "O" if self._parse_optional(token) is not None else "A"
                )

        provided: Set[str] = set()
        index = 0
        while index < len(argv):
            token = argv[index]
            if token == "--":
                break
            if not token.startswith("-"):
                index += 1
                continue
            parsed = self._parse_optional(token)
            if parsed is not None and parsed[0] is not None:
                # RTP-LLM currently supports Python 3.10. These two private
                # helpers are the only argparse recognizers on that runtime
                # that preserve abbreviations, combined short options, and
                # every nargs spelling. The focused contract test below pins
                # the tuple fields this scanner uses.
                action = parsed[0]
                option_string = parsed[1]
                explicit_arg = parsed[-1]
                while action is not None:
                    provided.add(action.dest)
                    if explicit_arg is None:
                        consumed = self._match_argument(
                            action, "".join(argument_pattern[index + 1 :])
                        )
                        index += consumed
                        break

                    explicit_count = self._match_argument(action, "A")
                    if (
                        explicit_count == 0
                        and len(option_string) > 1
                        and option_string[1] not in self.prefix_chars
                        and explicit_arg
                    ):
                        # argparse expands -abc into -a -b -c only while each
                        # preceding short action consumes zero arguments.
                        option_string = option_string[0] + explicit_arg[0]
                        explicit_arg = explicit_arg[1:] or None
                        action = self._option_string_actions.get(option_string)
                        continue
                    break
            index += 1
        return provided

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

    def _env_value_provided(
        self, action: argparse.Action, env_value: Optional[str]
    ) -> bool:
        if env_value is None:
            return False
        semantics = self._env_semantics.get(action.dest, EnvArgSemantics())
        if env_value == "" and semantics.empty_as_unset:
            return False
        return True

    def _validate_env_choice(
        self, action: argparse.Action, value: Any, env_name: str
    ) -> None:
        if action.choices is None or value in action.choices:
            return

        option = self._display_option(action)
        choices = ", ".join(repr(choice) for choice in action.choices)
        if not self._env_semantics.get(action.dest, EnvArgSemantics()).strict_choice:
            # Preserve the legacy tolerance for pre-existing arguments: the
            # invalid value is still bound (as before this validation existed)
            # and only surfaced via the ERROR log above.
            logging.error(
                "Invalid value %r for environment variable %s (argument %s); "
                "allowed values are %s; the value is used as-is for backward "
                "compatibility",
                value,
                env_name,
                option,
                choices,
            )
            return
        logging.error(
            "Invalid value %r for environment variable %s (argument %s); "
            "allowed values are %s",
            value,
            env_name,
            option,
            choices,
        )
        self.error(
            f"argument {option}: invalid choice: {value!r} from environment "
            f"variable {env_name} (choose from {choices})"
        )

    def parse_args(
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> argparse.Namespace:
        deferred_finalizers: List[Tuple[Any, Namespace]] = []
        token = _DEFERRED_PARSE_FINALIZERS.set(deferred_finalizers)
        try:
            parsed_args, unknown_args = self.parse_known_args(args, namespace)
            if unknown_args:
                self.error("unrecognized arguments: %s" % " ".join(unknown_args))
        finally:
            _DEFERRED_PARSE_FINALIZERS.reset(token)

        # Subparsers finish parsing before their parents, matching argparse's
        # normal lifecycle. Commit bindings and validators in that same order
        # only after the entire command line is known to be valid.
        for parser, parser_args in deferred_finalizers:
            parser._finalize_parsed_args(parser_args)
        return parsed_args

    def parse_known_args(
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> Tuple[argparse.Namespace, List[str]]:
        """Parse, bind, and validate for direct and subparser callers alike.

        ``argparse`` invokes a selected subparser through ``parse_known_args``.
        Keeping the complete environment/config lifecycle here prevents CLI
        subcommands from bypassing bindings and post-parse validation.
        """
        logging.info("Parsing arguments and applying config bindings...")

        # If args is None, check if we should read from environment variables
        # argparse will use sys.argv when args is None, so we need to check sys.argv first
        has_cmd_args = args is not None or (len(sys.argv) > 1)
        deferred_env_values: Dict[str, Any] = {}

        if args is None:
            # Check if there are command line arguments (more than just program name)
            if not has_cmd_args:
                # No command line arguments, read from environment variables and construct args list
                args = []
                # Read values from environment variables for all registered arguments
                for dest, env_name in self._env_mappings.items():
                    env_value = os.environ.get(env_name)
                    # Find the action for this dest before applying its
                    # declaration-local environment semantics.
                    action = self._find_action(dest)
                    if action is not None and self._env_value_provided(
                        action, env_value
                    ):
                        # Get the option string (e.g., "--model_type")
                        option_string = None
                        for option in action.option_strings:
                            if option.startswith("--"):
                                option_string = option
                                break

                        if option_string:
                            # Pre-convert every synthetic environment value so
                            # the pure-env path has exactly the mixed CLI+env
                            # compatibility policy. Explicit converter
                            # rejections stay fatal; ordinary legacy conversion
                            # failures fall back to the parser default.
                            try:
                                converted_value = (
                                    action.type(env_value)
                                    if action.type is not None
                                    else env_value
                                )
                            except argparse.ArgumentTypeError:
                                self.error(
                                    f"argument {self._display_option(action)}: "
                                    f"invalid value {env_value!r} from environment "
                                    f"variable {env_name}"
                                )
                            except (ValueError, TypeError):
                                logging.warning(
                                    "Ignoring environment variable %s because it "
                                    "cannot be converted for argument %s; using "
                                    "the default value %r",
                                    env_name,
                                    self._display_option(action),
                                    action.default,
                                )
                                continue

                            if (
                                action.choices is not None
                                and converted_value not in action.choices
                            ):
                                self._validate_env_choice(
                                    action, converted_value, env_name
                                )
                                # Non-strict legacy choices are deliberately
                                # bound as-is in the mixed path. Keep argparse
                                # from rejecting the synthetic token, then
                                # apply the same converted value after parsing.
                                deferred_env_values[action.dest] = converted_value
                                continue
                            args.extend([option_string, env_value])
            # If has_cmd_args is True, args remains None and argparse will use sys.argv

        parsed_args, unknown_args = super().parse_known_args(args, namespace)

        for dest, value in deferred_env_values.items():
            setattr(parsed_args, dest, value)

        # After parsing, if there were command line arguments, fill in missing values from environment variables
        # This allows mixing command line arguments and environment variables
        if has_cmd_args:
            # Reuse argparse's own optional-token recognition so explicit CLI
            # values always win for every supported spelling, including
            # --name=value, abbreviated long options, and short options.
            command_line_args = args if args is not None else sys.argv[1:]
            provided_args = self._provided_option_dests(command_line_args)

            # Now fill in missing values from environment variables
            for dest, env_name in self._env_mappings.items():
                # Only set from environment if the value wasn't provided via command line
                if dest not in provided_args:
                    env_value = os.environ.get(env_name)
                    # Find the action to get the type converter and its
                    # declaration-local environment semantics.
                    action = self._find_action(dest)
                    if action is not None and self._env_value_provided(
                        action, env_value
                    ):
                        # Convert the value using the action's type
                        if action.type is not None:
                            try:
                                converted_value = action.type(env_value)
                            except argparse.ArgumentTypeError:
                                # Values explicitly rejected by the converter
                                # (e.g. str2bool) fail fast so the mixed
                                # CLI+env path matches the pure-env path,
                                # where argparse raises the same error.
                                option = self._display_option(action)
                                logging.error(
                                    "Invalid value for environment variable %s (argument %s)",
                                    env_name,
                                    option,
                                )
                                self.error(
                                    f"argument {option}: invalid value "
                                    f"{env_value!r} from environment "
                                    f"variable {env_name}"
                                )
                            except (ValueError, TypeError):
                                # Preserve the legacy fallback-to-default behavior
                                # for plain conversion failures, but make the
                                # ignored configuration discoverable.
                                logging.warning(
                                    "Ignoring environment variable %s because it "
                                    "cannot be converted for argument %s; using "
                                    "the default value %r",
                                    env_name,
                                    self._display_option(action),
                                    action.default,
                                )
                            else:
                                self._validate_env_choice(
                                    action, converted_value, env_name
                                )
                                setattr(parsed_args, dest, converted_value)
                        else:
                            # No type converter, use as string
                            self._validate_env_choice(action, env_value, env_name)
                            setattr(parsed_args, dest, env_value)

        deferred_finalizers = _DEFERRED_PARSE_FINALIZERS.get()
        if deferred_finalizers is not None:
            deferred_finalizers.append((self, parsed_args))
        else:
            self._finalize_parsed_args(parsed_args)

        return parsed_args, unknown_args

    def _finalize_parsed_args(self, parsed_args: argparse.Namespace) -> None:
        """Commit config bindings, validators, and resolved-value logging."""

        # 应用所有配置绑定
        self._apply_config_bindings(parsed_args)
        for validator in self._post_parse_validators:
            validator()

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

    def _apply_config_bindings(self, parsed_args: argparse.Namespace) -> None:
        """应用所有配置绑定，将解析的参数值设置到配置对象"""
        for binding in self._config_bindings:
            value = getattr(parsed_args, binding.dest, None)
            if value is not None:
                strict_binding = self._env_semantics.get(
                    binding.dest, EnvArgSemantics()
                ).strict_config_binding
                # Tuple targets already contain their owning object. A root
                # config is required only for dotted-string targets. Preserve
                # the historical warning for legacy declarations, while new
                # strict declarations fail instead of running validators
                # against stale defaults.
                targets = (
                    binding.bind_to
                    if isinstance(binding.bind_to, list)
                    else [binding.bind_to]
                )
                if self._root_config is None and any(
                    isinstance(target, str) for target in targets
                ):
                    message = (
                        f"Failed to apply config binding for {binding.dest}: "
                        "set_root_config() is required for string config bindings"
                    )
                    if strict_binding:
                        self.error(message)
                    logging.warning(message)
                    continue
                try:
                    binding.apply(value, self._root_config)
                except Exception as error:
                    message = (
                        f"failed to bind argument {binding.dest} to "
                        f"{binding.bind_to!r}: {error}"
                    )
                    if strict_binding:
                        self.error(message)
                    logging.warning(message)
                    continue
                logging.debug(
                    f"[ConfigBinding] {binding.dest} -> {binding.bind_to} = {value}"
                )

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

    def get_env_semantics(self, dest: str) -> EnvArgSemantics:
        """Return the immutable environment semantics declared for an argument."""

        return self._env_semantics.get(dest, EnvArgSemantics())

    def get_argument_default(self, dest: str) -> Any:
        """Return an argument's parser default through a stable query API."""

        action = self._find_action(dest)
        if action is None:
            raise KeyError(dest)
        return action.default


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
    init_grammar_group_args(parser, py_env_configs.grammar_config)
    init_hw_kernel_group_args(parser, py_env_configs.py_hw_kernel_config)
    init_kv_cache_group_args(parser, py_env_configs.kv_cache_config)
    init_load_group_args(parser, py_env_configs.load_config, py_env_configs.model_args)
    init_lora_group_args(parser, py_env_configs.lora_config)
    init_master_group_args(parser, py_env_configs.master_config)
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
        py_env_configs.prefill_cp_config,
    )
    init_profile_debug_logging_group_args(
        parser, py_env_configs.profiling_debug_logging_config
    )
    init_quantization_group_args(parser, py_env_configs.quantization_config)
    init_render_group_args(parser, py_env_configs.render_config)
    init_repetition_detection_group_args(
        parser, py_env_configs.repetition_detection_config
    )
    init_role_group_args(parser, py_env_configs.role_config)
    init_rpc_discovery_group_args(parser)
    init_scheduler_group_args(parser, py_env_configs.runtime_config)
    init_server_group_args(
        parser,
        py_env_configs.server_config,
        py_env_configs.distribute_config,
    )
    init_speculative_decoding_group_args(parser, py_env_configs.sp_config)
    init_vit_group_args(parser, py_env_configs.vit_config)
    init_jit_group_args(parser, py_env_configs.jit_config)
    init_pd_separation_group_args(parser, py_env_configs.pd_separation_config)
    init_model_grpc_group_args(parser, py_env_configs.grpc_config)
    init_dash_sc_grpc_group_args(parser, py_env_configs.dash_sc_grpc_config)


def setup_args(args: Optional[Sequence[str]] = None) -> PyEnvConfigs:
    """Parse engine arguments into the canonical ``PyEnvConfigs`` object.

    ``args=None`` preserves the server entry point behavior (parse ``sys.argv``).
    Supplying an explicit sequence lets in-process tools, such as the offline
    capacity estimator, reuse the exact same parser and config bindings without
    temporarily replacing global process arguments.
    """
    parser = EnvArgumentParser(description="RTP LLM")

    # 先创建配置对象
    py_env_configs = PyEnvConfigs()

    # 设置根配置对象，用于解析字符串路径形式的 bind_to
    parser.set_root_config(py_env_configs)

    # 使用统一的函数初始化所有参数组，并绑定到配置对象
    init_all_group_args(parser, py_env_configs)

    # 解析参数（会自动应用所有配置绑定）
    parser.parse_args(args)
    return py_env_configs
