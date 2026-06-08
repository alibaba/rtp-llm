from rtp_llm.server.server_args.util import str2bool


def init_embedding_group_args(parser, embedding_config):
    ##############################################################################################################
    # Embedding Configuration
    ##############################################################################################################
    embedding_group = parser.add_argument_group("Embedding Configuration")
    embedding_group.add_argument(
        "--embedding_model",
        env_name="EMBEDDING_MODEL",
        bind_to=(embedding_config, 'embedding_model'),
        type=int,
        default=0,
        help="嵌入模型的具体类型",
    )

    embedding_group.add_argument(
        "--extra_input_in_mm_embedding",
        env_name="EXTRA_INPUT_IN_MM_EMBEDDING",
        bind_to=(embedding_config, 'extra_input_in_mm_embedding'),
        type=str,
        default=None,
        help='在多模态嵌入中使用额外的输入，可选值"INDEX"',
    )

    embedding_group.add_argument(
        "--embedding_arpc_rdma_mode",
        env_name="EMBEDDING_ARPC_RDMA_MODE",
        bind_to=(embedding_config, 'embedding_arpc_rdma_mode'),
        type=str2bool,
        default=False,
        help="控制 embedding ARPC 是否使用 RDMA 模式。",
    )
