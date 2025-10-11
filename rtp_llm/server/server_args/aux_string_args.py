def init_aux_string_group_args(parser):
    ##############################################################################################################
    # control and management args   管控参数
    aux_string_group = parser.add_argument_group("aux_string")
    aux_string_group.add_argument(
        "--aux_string",
        env_name="AUX_STRING",
        type=str,
        default='{"DEPLOYMENT_NAME":"DEFAULT_DEPLOYMENT_NAME"}',
        help="管控传输的部署参数, 将会透传给aux info",
    )
