from rtp_llm.server.server_args.util import str2bool


def init_sampling_group_args(parser, sampler_config):
    ##############################################################################################################
    # 采样
    ##############################################################################################################
    sampling_group = parser.add_argument_group("采样")

    sampling_group.add_argument(
        "--max_batch_size",
        env_name="MAX_BATCH_SIZE",
        bind_to=(sampler_config, 'max_batch_size'),
        type=int,
        default=0,
        help="覆盖系统自动计算的最大 batch size。",
    )
    sampling_group.add_argument(
        "--enable_flashinfer_sample_kernel",
        env_name="ENABLE_FLASHINFER_SAMPLE_KERNEL",
        bind_to=(sampler_config, 'enable_flashinfer_sample_kernel'),
        type=str2bool,
        default=True,
        help="控制是否启用FlashInfer的采样 kernel。可选值: True (启用), False (禁用)。",
    )
