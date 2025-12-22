#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从环境变量读取参数值并生成--xxx格式的参数列表
"""

import argparse
import os
from ast import arg
from typing import Any, List, Tuple

from rtp_llm.server.server_args.server_args import EnvArgumentParser
from rtp_llm.server.server_args.util import str2bool
from rtp_llm.config.py_config_modules import PyEnvConfigs


def get_all_arguments_from_parser(
    parser: EnvArgumentParser,
) -> List[Tuple[str, str, Any, str]]:
    """
    从解析器中获取所有参数信息
    返回: [(arg_name, env_name, default_value, arg_type), ...]
    """
    all_args = []

    # 获取所有环境变量映射
    env_mappings = parser.get_env_mappings()

    # 遍历所有action来获取参数信息
    for action in parser._actions:
        if hasattr(action, "dest") and action.dest in env_mappings:
            arg_name = action.dest
            env_name = env_mappings[arg_name]
            default_value = action.default
            arg_type = action.type if action.type else str

            # 获取参数的长选项名
            long_option = None
            for option_string in action.option_strings:
                if option_string.startswith("--"):
                    long_option = option_string
                    break

            if long_option:
                all_args.append((long_option, env_name, default_value, arg_type))

    return all_args


def read_env_value(env_name: str, default_value: Any, arg_type: type) -> Any:
    """
    从环境变量读取值，如果环境变量不存在则返回默认值
    """
    if default_value is None:
        return None

    env_value = os.getenv(env_name)
    if env_value is None:
        return default_value

    try:
        # 根据类型转换环境变量值
        if arg_type == bool:
            return env_value.lower() in ("true", "1", "yes", "on")
        elif arg_type == int:
            return int(env_value)
        elif arg_type == float:
            return float(env_value)
        else:
            return str(env_value)
    except (ValueError, TypeError):
        # 如果转换失败，返回默认值
        return default_value


def format_argument_value(value: Any) -> str:
    """
    格式化参数值为字符串
    只处理非字符串类型，字符串类型由 format_argument_pair 跳过
    """
    if value is None:
        return ""
    elif isinstance(value, bool):
        return "1" if value else "0"
    else:
        return str(value)


def format_argument_pair(long_option: str, value: Any) -> List[str]:
    """
    格式化参数对为列表，支持 --xx xx 格式
    跳过字符串类型的参数
    """
    # 跳过字符串类型的参数
    if isinstance(value, str):
        return []

    formatted_value = format_argument_value(value)
    if formatted_value:
        return [long_option, formatted_value]
    else:
        return [long_option]


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

    for long_option, env_name, default_value, arg_type in all_args:
        # 过滤掉argparse的内部参数
        if long_option == "--help" or default_value == "==SUPPRESS==":
            continue

        # 检查环境变量是否存在
        env_var_exists = os.getenv(env_name) is not None
        # 如果default_value为None，只要环境变量存在就读取
        if default_value is None:
            if env_var_exists:
                env_value_str = os.getenv(env_name)
                if env_value_str is not None:
                    try:
                        # 根据类型转换环境变量值
                        if arg_type == bool:
                            env_value = env_value_str.lower() in (
                                "true",
                                "1",
                                "yes",
                                "on",
                            )
                        elif arg_type == int:
                            env_value = int(env_value_str)
                        elif arg_type == float:
                            env_value = float(env_value_str)
                        elif arg_type == str2bool:
                            env_value = str2bool(env_value_str)
                        else:
                            env_value = str(env_value_str)

                        # 跳过空字符串参数
                        if isinstance(env_value, str) and env_value == "":
                            continue

                        args_list.extend(format_argument_pair(long_option, env_value))
                    except (ValueError, TypeError):
                        # 如果转换失败，跳过这个参数
                        continue
        else:
            # 从环境变量读取值
            env_value = read_env_value(env_name, default_value, arg_type)

            # 跳过空字符串参数
            if isinstance(env_value, str) and env_value == "":
                continue

            # 根据only_env_vars参数决定是否只输出环境变量中存在的参数
            if only_env_vars:
                if env_var_exists:
                    args_list.extend(format_argument_pair(long_option, env_value))
            else:
                # 总是添加参数，不管环境变量是否存在
                args_list.extend(format_argument_pair(long_option, env_value))

    return args_list


def main():
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

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
