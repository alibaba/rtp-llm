#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从环境变量读取参数值并生成--xxx格式的参数列表
"""

import argparse
import datetime
import os
from dataclasses import dataclass
from typing import Any, List

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.server.server_args.server_args import EnvArgumentParser
from rtp_llm.server.server_args.util import argument_base_type, str2bool

_UNSAFE_GENERATED_STRING_CHARS = frozenset("'\"\\`$;&|<>(){}!*?[]#~")


@dataclass(frozen=True)
class GeneratedArgumentSpec:
    long_option: str
    env_name: str
    default_value: Any
    converter: Any
    empty_as_unset: bool
    emit_string: bool


def get_all_arguments_from_parser(
    parser: EnvArgumentParser,
) -> List[GeneratedArgumentSpec]:
    """
    从解析器中获取所有参数信息
    """
    all_args: List[GeneratedArgumentSpec] = []

    # 获取所有环境变量映射
    env_mappings = parser.get_env_mappings()

    # 遍历所有action来获取参数信息
    for action in parser._actions:
        if hasattr(action, "dest") and action.dest in env_mappings:
            arg_name = action.dest
            env_name = env_mappings[arg_name]
            default_value = action.default
            arg_type = action.type if action.type else str
            semantics = parser.get_env_semantics(arg_name)

            # 获取参数的长选项名
            long_option = None
            for option_string in action.option_strings:
                if option_string.startswith("--"):
                    long_option = option_string
                    break

            if long_option:
                all_args.append(
                    GeneratedArgumentSpec(
                        long_option=long_option,
                        env_name=env_name,
                        default_value=default_value,
                        converter=arg_type,
                        empty_as_unset=semantics.empty_as_unset,
                        emit_string=semantics.emit_string_from_env,
                    )
                )

    return all_args


def _convert_env_value(
    env_name: str, long_option: str, raw_value: str, converter: Any
) -> Any:
    """Convert one explicit environment value with source-aware errors."""

    try:
        if converter == str2bool:
            return str2bool(raw_value)
        base_type = argument_base_type(converter)
        if base_type == bool:
            return str2bool(raw_value)
        if base_type in (int, float):
            return converter(raw_value)
        return str(raw_value)
    except argparse.ArgumentTypeError as error:
        raise argparse.ArgumentTypeError(
            f"{env_name} ({long_option}): {error}"
        ) from error


def read_env_value(
    env_name: str,
    default_value: Any,
    arg_type: Any,
    long_option: str = "<generated argument>",
) -> Any:
    """
    从环境变量读取值，如果环境变量不存在则返回默认值
    """
    if default_value is None:
        return None

    env_value = os.getenv(env_name)
    if env_value is None:
        return default_value

    try:
        return _convert_env_value(env_name, long_option, env_value, arg_type)
    except (ValueError, TypeError):
        # Preserve the legacy fallback for converters that raise ordinary
        # conversion errors. Explicit argparse rejections are source-aware and
        # fail fast instead of silently changing deployment intent.
        return default_value


def format_argument_value(value: Any) -> str:
    """
    格式化参数值为字符串
    """
    if value is None:
        return ""
    elif isinstance(value, bool):
        return "1" if value else "0"
    else:
        return str(value)


def format_argument_pair(
    long_option: str, value: Any, *, emit_string: bool = False
) -> List[str]:
    """
    格式化参数对为列表，支持 --xx xx 格式
    """
    # String arguments were deliberately excluded from this deployment tool
    # because its final output is a shell command fragment. Opt in only at the
    # argument declaration, where the producer and validation contract are
    # visible together.
    if isinstance(value, str) and not emit_string:
        return []

    formatted_value = format_argument_value(value)
    if formatted_value:
        return [long_option, formatted_value]
    else:
        return [long_option]


def append_generated_argument(
    args_list: List[str], argument: GeneratedArgumentSpec, value: Any
) -> None:
    """Append one argv pair if its declaration permits deployment emission."""

    if isinstance(value, str):
        if value == "":
            if argument.emit_string and not argument.empty_as_unset:
                raise argparse.ArgumentTypeError(
                    f"{argument.env_name} ({argument.long_option}): an explicit "
                    "empty string cannot be represented by the generated argument hand-off"
                )
            return
        if argument.emit_string and any(char.isspace() for char in value):
            # The deployment hand-off serializes this argv list as a
            # whitespace-delimited string. Reject an unrepresentable value
            # instead of silently splitting one identity into multiple args.
            raise argparse.ArgumentTypeError(
                f"{argument.env_name} ({argument.long_option}): "
                "whitespace is not supported by the generated argument hand-off"
            )
        if argument.emit_string and (
            value.startswith("-")
            or any(char in _UNSAFE_GENERATED_STRING_CHARS for char in value)
        ):
            # The final hand-off is deliberately a command fragment rather
            # than a structured argv encoding. Reject tokens that can become
            # options, glob patterns, substitutions, redirections, comments,
            # or control operators when a deployment consumes or pastes it.
            raise argparse.ArgumentTypeError(
                f"{argument.env_name} ({argument.long_option}): "
                "shell metacharacters and leading '-' are not supported by "
                "the generated argument hand-off"
            )
    args_list.extend(
        format_argument_pair(
            argument.long_option,
            value,
            emit_string=argument.emit_string,
        )
    )


def generate_args_list(only_env_vars: bool = False) -> List[str]:
    """
    生成从环境变量读取的参数列表

    Args:
        only_env_vars: 如果为True，只输出环境变量中存在的参数
    """

    # 创建解析器并设置所有参数
    parser = EnvArgumentParser(description="RTP LLM")
    # 使用统一的函数初始化所有参数组
    from rtp_llm.server.server_args.server_args import init_all_group_args

    py_env_configs = PyEnvConfigs()
    init_all_group_args(parser, py_env_configs)

    # 获取所有参数信息
    all_args = get_all_arguments_from_parser(parser)

    args_list = []

    for argument in all_args:
        # 过滤掉argparse的内部参数
        if (
            argument.long_option == "--help"
            or argument.default_value == argparse.SUPPRESS
        ):
            continue

        # 检查环境变量是否存在
        raw_env_value = os.getenv(argument.env_name)
        env_var_exists = raw_env_value is not None and not (
            argument.empty_as_unset and raw_env_value == ""
        )
        # 如果default_value为None，只要环境变量存在就读取
        if argument.default_value is None:
            if env_var_exists:
                env_value_str = raw_env_value
                if env_value_str is not None:
                    try:
                        env_value = _convert_env_value(
                            argument.env_name,
                            argument.long_option,
                            env_value_str,
                            argument.converter,
                        )
                        append_generated_argument(args_list, argument, env_value)
                    except (ValueError, TypeError):
                        # 如果转换失败，跳过这个参数
                        continue
        else:
            # 从环境变量读取值
            env_value = (
                read_env_value(
                    argument.env_name,
                    argument.default_value,
                    argument.converter,
                    argument.long_option,
                )
                if env_var_exists
                else argument.default_value
            )

            # 历史上生成器不会输出隐式的字符串默认值。保留这个
            # 行为，但不能丢弃调用方显式设置的字符串环境变量。
            if isinstance(env_value, str) and not env_var_exists:
                continue

            # 根据only_env_vars参数决定是否只输出环境变量中存在的参数
            if only_env_vars:
                if env_var_exists:
                    append_generated_argument(args_list, argument, env_value)
            else:
                # 总是添加参数，不管环境变量是否存在
                append_generated_argument(args_list, argument, env_value)

    return args_list


def main() -> int:
    """
    主函数：生成并打印参数列表
    """
    parser = argparse.ArgumentParser(description="从环境变量生成RTP LLM参数列表")
    parser.add_argument(
        "--only-env-vars", action="store_true", help="只输出环境变量中存在的参数"
    )
    parser.add_argument("--output-file", type=str, help="输出文件路径（可选）")
    parser.add_argument("--quiet", action="store_true", help="静默模式，只输出参数列表")
    parser.add_argument(
        "--set-env", action="store_true", help="将参数列表保存到环境变量 env_args"
    )
    parser.add_argument(
        "--export-env",
        action="store_true",
        help="输出环境变量设置命令（用于在shell中执行）",
    )

    args = parser.parse_args()

    if not args.quiet:
        print("正在从环境变量读取参数...")

    try:
        args_list = generate_args_list(only_env_vars=args.only_env_vars)

        if not args.quiet:
            print(f"\n找到 {len(args_list)} 个参数:")
            print("=" * 50)

            for i, arg in enumerate(args_list, 1):
                print(f"{i:3d}. {arg}")

            print("=" * 50)
            print(f"\n完整的参数列表:")

        # 输出参数列表
        print(" ".join(args_list))

        # 保存到环境变量 env_args（如果用户指定了 --set-env 选项）
        env_args_value = " ".join(args_list)
        if args.set_env:
            os.environ["env_args"] = env_args_value
            if not args.quiet:
                print(f"\n参数列表已保存到环境变量 env_args")

        # 输出环境变量设置命令（如果用户指定了 --export-env 选项）
        if args.export_env:
            # 转义单引号，使用双引号包围整个值
            escaped_value = env_args_value.replace("'", "'\"'\"'")
            print(f"\nexport env_args='{escaped_value}'")

        # 保存到文件
        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(env_args_value)
            if not args.quiet:
                print(f"参数列表已保存到: {args.output_file}")
        return 0

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # measure the initialization time
    current_time = (
        datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")
    )
    print(f"[PROCESS_START]{current_time} Start generate args")
    raise SystemExit(main())
