def init_role_group_args(parser, role_config):
    ##############################################################################################################
    #  Role配置
    ##############################################################################################################
    role_group = parser.add_argument_group("Role")
    role_group.add_argument(
        "--role_type",
        env_name="ROLE_TYPE",
        bind_to=(role_config, 'role_type'),
        type=str,
        default="PDFUSION",
        help="role的类型: 包含PDFUSION / PREFILL / DECODE / VIT / FRONTEND 几种类型",
    )
