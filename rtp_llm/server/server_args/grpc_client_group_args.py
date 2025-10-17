def init_grpc_client_group_args(parser):
    ##############################################################################################################
    # Grpc Client
    ##############################################################################################################
    grpc_client_group = parser.add_argument_group("Grpc Client")
    grpc_client_group.add_argument(
        "--grpc_client_max_receive_message_length",
        env_name="GRPC_CLIENT_MAX_RECEIVE_MESSAGE_LENGTH",
        type=int,
        default=1024 * 1024 * 1024,
        help="grpc client 单次接收消息的最大长度",
    )
    grpc_client_group.add_argument(
        "--grpc_client_max_send_message_length",
        env_name="GRPC_CLIENT_MAX_SEND_MESSAGE_LENGTH",
        type=int,
        default=1024 * 1024 * 1024,
        help="grpc client 单次发送消息的最大长度",
    )
    grpc_client_group.add_argument(
        "--grpc_client_max_metadata_size",
        env_name="GRPC_CLIENT_MAX_METADATA_SIZE",
        type=int,
        default=1024 * 1024 * 1024,
        help="grpc client 元数据（如 headers）的最大总大小",
    )
    grpc_client_group.add_argument(
        "--grpc_client_need_compression",
        env_name="GRPC_CLIENT_NEED_COMPRESSION",
        type=bool,
        default=True,
        help="grpc client 传输数据是否压缩",
    )
