import argparse
from typing import List

from rtp_llm.server.server_args.util import str2bool


def parse_external_model_packages(value: str) -> List[str]:
    package_names = []
    for item in value.split(","):
        package_name = item.strip()
        if not package_name:
            continue
        if not all(part.isidentifier() for part in package_name.split(".")):
            raise argparse.ArgumentTypeError(
                f"invalid external model package path: {package_name!r}"
            )
        if package_name not in package_names:
            package_names.append(package_name)
    return package_names


def init_model_group_args(parser, model_args):
    ##############################################################################################################
    # Model Configuration
    ##############################################################################################################
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--tokenizer_path",
        env_name="TOKENIZER_PATH",
        bind_to=(model_args, "tokenizer_path"),
        type=str,
        default="",
        help="分词器的路径",
    )
    model_group.add_argument(
        "--act_type",
        env_name="ACT_TYPE",
        bind_to=(model_args, "act_type"),
        type=str,
        default=None,
        help="计算使用的数据类型",
    )
    model_group.add_argument(
        "--mla_ops_type",
        env_name="MLA_OPS_TYPE",
        bind_to=(model_args, "mla_ops_type"),
        type=str,
        default=None,
        help="Multi Latent Attention的操作类型（将自动转换为枚举）",
    )
    model_group.add_argument(
        "--task_type",
        env_name="TASK_TYPE",
        bind_to=(model_args, "task_type"),
        type=str,
        default=None,
        help="任务类型（将自动转换为枚举）",
    )
    model_group.add_argument(
        "--model_type",
        env_name="MODEL_TYPE",
        bind_to=(model_args, "model_type"),
        type=str,
        default=None,
        help="模型类型",
    )
    model_group.add_argument(
        "--checkpoint_path",
        env_name="CHECKPOINT_PATH",
        bind_to=(model_args, "ckpt_path"),
        type=str,
        help="Checkpoint路径",
    )
    model_group.add_argument(
        "--ptuning_path",
        env_name="PTUNING_PATH",
        bind_to=(model_args, "ptuning_path"),
        type=str,
        default="",
        help="PTuning路径",
    )
    model_group.add_argument(
        "--json_model_override_args",
        env_name="JSON_MODEL_OVERRIDE_ARGS",
        bind_to=(model_args, "json_model_override_args"),
        type=str,
        default="{}",
        help="A dictionary in JSON string format used to override default model configurations.",
    )
    model_group.add_argument(
        "--external_model_packages",
        enable_env=False,
        bind_to=(model_args, "external_model_packages"),
        type=parse_external_model_packages,
        default=None,
        help=(
            "逗号分隔的受信任外部模型模块路径（如 "
            "atom.plugin.rtpllm.models）；仅支持命令行配置"
        ),
    )
    model_group.add_argument(
        "--max_seq_len",
        env_name="MAX_SEQ_LEN",
        bind_to=(model_args, "max_seq_len"),
        type=int,
        default=None,
        help="最大序列长度",
    )
    model_group.add_argument(
        "--enable_fp32_lm_head",
        env_name="ENABLE_FP32_LM_HEAD",
        bind_to=(model_args, "enable_fp32_lm_head"),
        type=str2bool,
        default=None,
        help="是否将lm_head权重加载为fp32精度，默认为true",
    )
    model_group.add_argument(
        "--enable_output_vocab_pruning",
        env_name="ENABLE_OUTPUT_VOCAB_PRUNING",
        bind_to=(model_args, "enable_output_vocab_pruning"),
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Use output_tokens.json from the checkpoint directory to prune the "
            "LM head. Flat or grouped token strings/IDs form one static set."
        ),
    )
