def init_jit_group_args(parser, jit_config):
    ##############################################################################################################
    # JIT Configuration
    ##############################################################################################################
    jit_group = parser.add_argument_group("JIT Configuration")
    jit_group.add_argument(
        "--remote_jit_dir",
        env_name="REMOTE_JIT_DIR",
        bind_to=(jit_config, "remote_jit_dir"),
        type=str,
        default="",
        help="JIT远程cache目录，必须是已存在的本地绝对路径或可通过FUSE/NFS挂载的远端URI；若本地JIT路径非默认值，则禁用远程cache",
    )
    jit_group.add_argument(
        "--remote_jit_read_dir",
        env_name="REMOTE_JIT_READ_DIR",
        bind_to=(jit_config, "remote_jit_read_dir"),
        type=str,
        default="",
        help="JIT 远程只读 cache 目录：以 RO 方式 fuse，把内容拷贝到本机 JIT cache 后读取",
    )
    jit_group.add_argument(
        "--warm_up_jit_and_write_remote",
        env_name="WARM_UP_JIT_AND_WRITE_REMOTE",
        bind_to=(jit_config, "warm_up_jit_and_write_remote"),
        type=str,
        default="",
        help=(
            "服务启动成功并完成 warmup 后，以 RW 方式 fuse 该远程目录，"
            "并把本机 JIT cache 产物拷贝进去"
        ),
    )
