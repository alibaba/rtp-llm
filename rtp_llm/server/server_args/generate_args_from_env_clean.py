#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从环境变量读取参数值并生成--xxx格式的参数列表
"""

import argparse
import os
import sys
from typing import Any, List, Tuple

# 添加项目路径到sys.path
sys.path.insert(0, "/home/tanboyu.tby/RTP-LLM")

from rtp_llm.server.server_args.server_args import EnvArgumentParser


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
    """
    if value is None:
        return ""
    elif isinstance(value, bool):
        return "1" if value else "0"
    elif isinstance(value, str):
        # 如果是空字符串，显示为 ''
        if value == "":
            return "''"

        # 处理字符串中的特殊字符，将换行符等转换为转义序列
        processed_value = (
            value.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        )

        # 如果字符串包含空格、换行符或特殊字符，需要加引号
        if (
            " " in processed_value
            or "\\" in processed_value
            or any(char in processed_value for char in ['"', "'"])
        ):
            return f'"{processed_value}"'
        return processed_value
    else:
        return str(value)


def format_argument_pair(long_option: str, value: Any) -> List[str]:
    """
    格式化参数对为列表，支持 --xx xx 格式
    """
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

    # 手动调用所有的init函数
    from rtp_llm.server.server_args.batch_decode_scheduler_group_args import (
        init_batch_decode_scheduler_group_args,
    )
    from rtp_llm.server.server_args.cache_store_group_args import (
        init_cache_store_group_args,
    )
    from rtp_llm.server.server_args.concurrent_group_args import (
        init_concurrent_group_args,
    )
    from rtp_llm.server.server_args.device_resource_group_args import (
        init_device_resource_group_args,
    )
    from rtp_llm.server.server_args.embedding_group_args import (
        init_embedding_group_args,
    )
    from rtp_llm.server.server_args.engine_group_args import init_engine_group_args
    from rtp_llm.server.server_args.fifo_scheduler_group_args import (
        init_fifo_scheduler_group_args,
    )
    from rtp_llm.server.server_args.fmha_group_args import init_fmha_group_args
    from rtp_llm.server.server_args.gang_group_args import init_gang_group_args
    from rtp_llm.server.server_args.generate_group_args import init_generate_group_args
    from rtp_llm.server.server_args.hw_kernel_group_args import (
        init_hw_kernel_group_args,
    )
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
    from rtp_llm.server.server_args.scheduler_group_args import (
        init_scheduler_group_args,
    )
    from rtp_llm.server.server_args.server_group_args import init_server_group_args
    from rtp_llm.server.server_args.sparse_group_args import init_sparse_group_args
    from rtp_llm.server.server_args.speculative_decoding_group_args import (
        init_speculative_decoding_group_args,
    )
    from rtp_llm.server.server_args.vit_group_args import init_vit_group_args
    from rtp_llm.server.server_args.worker_group_args import init_worker_group_args

    # 初始化所有参数组
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

    # 获取所有参数信息
    all_args = get_all_arguments_from_parser(parser)

    args_list = []

    for long_option, env_name, default_value, arg_type in all_args:
        # 过滤掉argparse的内部参数
        if long_option == "--help" or default_value == "==SUPPRESS==":
            continue

        # 只处理default不为None的参数
        if default_value is not None:
            # 从环境变量读取值
            env_value = read_env_value(env_name, default_value, arg_type)

            # 根据only_env_vars参数决定是否只输出环境变量中存在的参数
            if only_env_vars:
                if os.getenv(env_name) is not None:
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
            print(f"\nexport env_args='{env_args_value}'")

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
