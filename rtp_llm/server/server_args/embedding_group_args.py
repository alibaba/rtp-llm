import argparse

from rtp_llm.server.server_args.util import str2bool, str2finite_float

QWEN3_RERANKER_INPUT_FORMATS = ("template", "raw")


def str2qwen3_reranker_input_format(value):
    if value not in QWEN3_RERANKER_INPUT_FORMATS:
        raise argparse.ArgumentTypeError(
            f"invalid qwen3_reranker_input_format {value!r}, "
            f"expected one of {list(QWEN3_RERANKER_INPUT_FORMATS)}."
        )
    return value


def init_embedding_group_args(parser, embedding_config):
    ##############################################################################################################
    # Embedding Configuration
    ##############################################################################################################
    embedding_group = parser.add_argument_group("Embedding Configuration")
    embedding_group.add_argument(
        "--embedding_model",
        env_name="EMBEDDING_MODEL",
        bind_to=(embedding_config, "embedding_model"),
        type=int,
        default=0,
        help="嵌入模型的具体类型",
    )
    embedding_group.add_argument(
        "--embedding_share_grpc_channel",
        env_name="RTP_LLM_EMBEDDING_SHARE_CHANNEL",
        bind_to=(embedding_config, "embedding_share_grpc_channel"),
        type=str2bool,
        default=True,
        help="Embedding/Reranker frontend 在每个 event loop 内复用同一个到 backend 的 gRPC channel，"
        "而不是每个请求新建 channel",
    )
    embedding_group.add_argument(
        "--embedding_grpc_keepalive",
        env_name="RTP_LLM_EMBEDDING_GRPC_KEEPALIVE",
        bind_to=(embedding_config, "embedding_grpc_keepalive"),
        type=str2bool,
        default=False,
        help="Opt in to gRPC keepalive (10s ping interval, 5s ACK timeout). "
        "Backend pauses exceeding the ACK timeout can fail active RPCs; "
        "enable only when the server ping policy and pause budget allow it.",
    )
    embedding_group.add_argument(
        "--embedding_rpc_timeout_s",
        env_name="RTP_LLM_EMBEDDING_RPC_TIMEOUT_S",
        bind_to=(embedding_config, "embedding_rpc_timeout_s"),
        type=str2finite_float,
        default=None,
        help="Embedding/Reranker frontend 到 backend 的 gRPC deadline（秒），默认不设 deadline；"
        "<=0 同样表示不设。注意 backend 目前不处理 gRPC 取消，deadline 到期后 backend 仍会算完该请求",
    )
    embedding_group.add_argument(
        "--qwen3_reranker_fixed_head",
        env_name="RTP_LLM_QWEN3_RERANKER_FIXED_HEAD",
        bind_to=(embedding_config, "qwen3_reranker_fixed_head"),
        type=str2bool,
        default=True,
        help="Qwen3 Reranker 只使用 lm_head 的 no/yes 两行、无 bias、fp32 计算 sigmoid(yes - no)",
    )
    embedding_group.add_argument(
        "--qwen3_reranker_input_format",
        env_name="RTP_LLM_QWEN3_RERANKER_INPUT_FORMAT",
        bind_to=(embedding_config, "qwen3_reranker_input_format"),
        type=str2qwen3_reranker_input_format,
        default="template",
        choices=QWEN3_RERANKER_INPUT_FORMATS,
        help="Qwen3 Reranker 输入格式：template 使用官方 Qwen3-Reranker prompt 模板；"
        "raw 直接拼接 query 和 document token（与 vLLM 无 chat template 时一致）",
    )
