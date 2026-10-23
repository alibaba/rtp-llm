import argparse
import math

from rtp_llm.server.server_args.util import str2bool


def _confidence_temperature(value):
    try:
        temperature = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "expected a positive finite temperature"
        ) from error
    if not math.isfinite(temperature) or temperature <= 0:
        raise argparse.ArgumentTypeError("expected a positive finite temperature")
    return temperature


def _verify_budget(value):
    try:
        budget = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("expected an integer budget >= -1") from error
    if budget < -1:
        raise argparse.ArgumentTypeError("expected an integer budget >= -1")
    return budget


def init_speculative_decoding_group_args(parser, sp_config):
    ##############################################################################################################
    # 投机采样配置
    ##############################################################################################################
    speculative_decoding_group = parser.add_argument_group("投机采样")
    speculative_decoding_group.add_argument(
        "--sp_model_type",
        env_name="SP_MODEL_TYPE",
        bind_to=(sp_config, "model_type"),
        type=str,
        default="",
        help='指定 speculative decoding 的草稿模型类型。例如："mixtbstars-mtp", "deepseek-v3-mtp"。',
    )

    speculative_decoding_group.add_argument(
        "--sp_type",
        env_name="SP_TYPE",
        bind_to=(sp_config, "type"),
        type=str,
        default="",
        help='控制 speculative decoding 类型，例如 "mtp"、"eagle" 或 "dspark"。',
    )

    speculative_decoding_group.add_argument(
        "--sp_min_token_match",
        env_name="SP_MIN_TOKEN_MATCH",
        bind_to=(sp_config, "sp_min_token_match"),
        type=int,
        default=2,
        help="为 speculative decoding 设置最小 token 匹配长度。",
    )

    speculative_decoding_group.add_argument(
        "--sp_max_token_match",
        env_name="SP_MAX_TOKEN_MATCH",
        bind_to=(sp_config, "sp_max_token_match"),
        type=int,
        default=2,
        help="为 speculative decoding 设置最大 token 匹配长度。",
    )

    speculative_decoding_group.add_argument(
        "--tree_decode_config",
        env_name="TREE_DECODE_CONFIG",
        bind_to=(sp_config, "tree_decode_config"),
        type=str,
        default="",
        help="Tree decode的配置文件名，定义了从前缀词到候选Token的映射。",
    )
    speculative_decoding_group.add_argument(
        "--sp_act_type",
        env_name="SP_ACT_TYPE",
        type=str,
        default=None,
        help="小模型的计算使用的类型",
    )
    speculative_decoding_group.add_argument(
        "--sp_quantization",
        env_name="SP_QUANTIZATION",
        bind_to=(sp_config, "quantization"),
        type=str,
        default=None,
        help="",
    )
    speculative_decoding_group.add_argument(
        "--sp_checkpoint_path",
        env_name="SP_CHECKPOINT_PATH",
        bind_to=(sp_config, "checkpoint_path"),
        type=str,
        default=None,
        help="",
    )

    speculative_decoding_group.add_argument(
        "--gen_num_per_cycle",
        env_name="GEN_NUM_PER_CIRCLE",
        bind_to=(sp_config, "gen_num_per_cycle"),
        type=int,
        default=1,
        help="每轮生成的草稿 token 数 gamma；不含 target 额外生成的 token。",
    )

    speculative_decoding_group.add_argument(
        "--force_stream_sample",
        env_name="FORCE_STREAM_SAMPLE",
        bind_to=(sp_config, "force_stream_sample"),
        type=str2bool,
        default=False,
        help="投机采样强制使用流式采样",
    )

    speculative_decoding_group.add_argument(
        "--sp_dspark_confidence_mode",
        env_name="SP_DSPARK_CONFIDENCE_MODE",
        bind_to=(sp_config, "sp_dspark_confidence_mode"),
        type=str,
        choices=["off", "observe", "bounded"],
        default="off",
        help=(
            "DeepSeek V4.1 confidence: off disables the head; observe records "
            "confidence without changing sampling; bounded allocates verification "
            "budget across request prefixes."
        ),
    )
    speculative_decoding_group.add_argument(
        "--sp_dspark_confidence_temperature",
        env_name="SP_DSPARK_CONFIDENCE_TEMPERATURE",
        bind_to=(sp_config, "sp_dspark_confidence_temperature"),
        type=_confidence_temperature,
        default=1.0,
        help="Positive confidence sigmoid temperature, independent of token sampling temperature.",
    )
    speculative_decoding_group.add_argument(
        "--sp_dspark_verify_budget",
        env_name="SP_DSPARK_VERIFY_BUDGET",
        bind_to=(sp_config, "sp_dspark_verify_budget"),
        type=_verify_budget,
        default=-1,
        help=(
            "Integer draft verification budget b in [0, gamma]: each decode rank "
            "selects b * local_batch_size positions across request prefixes, "
            "excluding one extra target anchor row per request. -1 keeps all "
            "gamma positions; off/observe require -1."
        ),
    )
    speculative_decoding_group.add_argument(
        "--sp_dspark_verify_mode",
        env_name="SP_DSPARK_VERIFY_MODE",
        bind_to=(sp_config, "sp_dspark_verify_mode"),
        type=str,
        choices=["dense", "packed"],
        default="dense",
        help=(
            "dense computes all gamma+1 target rows per request even with a "
            "bounded budget; packed computes selected prefixes plus anchors. "
            "Packed requires DeepSeek V4.1 DECODE with FP8 KV cache, TP=1, "
            "bounded confidence mode and a nonnegative budget."
        ),
    )

    speculative_decoding_group.add_argument(
        "--force_score_context_attention",
        env_name="FORCE_SCORE_CONTEXT_ATTENTION",
        bind_to=(sp_config, "force_score_context_attention"),
        type=str2bool,
        default=True,
        help="投机采样强制score阶段使用context attention",
    )
