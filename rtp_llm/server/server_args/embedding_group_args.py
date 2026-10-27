from rtp_llm.server.server_args.util import str2bool, str2finite_float


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
