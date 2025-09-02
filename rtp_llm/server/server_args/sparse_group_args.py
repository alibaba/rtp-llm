def init_sparse_group_args(parser):
    ##############################################################################################################
    # Sparse Configuration
    ##############################################################################################################
    sparse_group = parser.add_argument_group("Sparse Configuration")
    sparse_group.add_argument(
        "--sparse_config_file",
        env_name="SPARSE_CONFIG_FILE",
        type=str,
        default=None,
        help="稀疏模型的配置文件路径",
    )
