from rtp_llm.server.server_args.util import str2bool


def init_misc_group_args(parser):
    ##############################################################################################################
    # Miscellaneous 配置
    ##############################################################################################################
    misc_group = parser.add_argument_group("Miscellaneous")

    misc_group.add_argument(
        "--load_balance",
        env_name="LOAD_BALANCE",
        type=str2bool,
        default=False,
        help="当设置为true时，启用基于吞吐量和延迟的动态并发控制；否则使用固定并发数。",
    )

    misc_group.add_argument(
        "--step_records_time_range",
        env_name="STEP_RECORDS_TIME_RANGE",
        type=int,
        default=60 * 1000 * 1000,
        help="性能记录 (step records) 的保留时间窗口，单位为微秒。例如，默认值 `60000000` 表示保留最近1分钟的记录。",
    )

    misc_group.add_argument(
        "--step_records_max_size",
        env_name="STEP_RECORDS_MAX_SIZE",
        type=int,
        default=1000,
        help="保留的性能记录 (step records) 的最大条数。与 `STEP_RECORDS_TIME_RANGE` 共同决定记录的保留策略。",
    )

    misc_group.add_argument(
        "--disable_pdl",
        env_name="DISABLE_PDL",
        type=str2bool,
        default=True,
        help="是否禁用PDL",
    )

    misc_group.add_argument(
        "--aux_string",
        env_name="AUX_STRING",
        type=str,
        default="",
        help="管控环境变量字符串",
    )
