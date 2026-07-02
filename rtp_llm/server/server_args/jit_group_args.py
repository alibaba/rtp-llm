def init_jit_group_args(parser, jit_config):
    ##############################################################################################################
    # JIT Configuration
    ##############################################################################################################
    jit_group = parser.add_argument_group("JIT Configuration")
    jit_group.add_argument(
        "--local_jit_dir",
        env_name="LOCAL_JIT_DIR",
        bind_to=(jit_config, "local_jit_dir"),
        type=str,
        default="./.jit_cache",
        help="JIT本地cache目录",
    )
    jit_group.add_argument(
        "--remote_jit_dir",
        env_name="REMOTE_JIT_DIR",
        bind_to=(jit_config, "remote_jit_dir"),
        type=str,
        default="",
        help="JIT远程cache目录，支持已存在的本地绝对路径或可通过FUSE/NFS挂载的远端URI",
    )
    jit_group.add_argument(
        "--remote_sync_timeout_s",
        env_name="REMOTE_SYNC_ONCE_TIMEOUT_S",
        bind_to=(jit_config, "remote_sync_timeout_s"),
        type=float,
        default=60,
        help="周期sync jit cache到远端的超时时间，单位秒",
    )
