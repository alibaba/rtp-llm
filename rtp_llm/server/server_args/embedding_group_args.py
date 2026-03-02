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
