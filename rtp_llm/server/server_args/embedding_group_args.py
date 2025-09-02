def init_embedding_group_args(parser):
    ##############################################################################################################
    # Embedding Configuration
    ##############################################################################################################
    embedding_group = parser.add_argument_group("Embedding Configuration")
    embedding_group.add_argument(
        "--embedding_model",
        env_name="EMBEDDING_MODEL",
        type=int,
        default=0,
        help="嵌入模型的具体类型",
    )

    embedding_group.add_argument(
        "--extra_input_in_mm_embedding",
        env_name="EXTRA_INPUT_IN_MM_EMBEDDING",
        type=str,
        default=None,
        help='在多模态嵌入中使用额外的输入，可选值"INDEX"',
    )
