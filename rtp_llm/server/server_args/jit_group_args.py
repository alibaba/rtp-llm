def init_jit_group_args(parser):
    ##############################################################################################################
    # JIT Configuration
    ##############################################################################################################
    jit_group = parser.add_argument_group("JIT Configuration")
    jit_group.add_argument(
        "--remote_jit_dir",
        env_name="REMOTE_JIT_DIR",
        type=str,
        default="/mnt/nas1",
        help="JIT远程cache目录",
    )
