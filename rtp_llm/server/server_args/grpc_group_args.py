import argparse
import logging


def _grpc_config_from_json(grpc_config):
    """创建类型转换函数，将 JSON 字符串转换为 GrpcConfig
    
    这个函数直接修改传入的 grpc_config 对象，而不是创建新对象。
    返回 grpc_config 对象本身（虽然不会被使用，因为 bind_to=None）。
    """
    def converter(json_str):
        if json_str is None or json_str == "":
            return grpc_config
        try:
            grpc_config.from_json(json_str)
            logging.debug(f"Initialized gRPC config from JSON: {json_str}")
            return grpc_config
        except Exception as e:
            logging.warning(f"Failed to parse gRPC config JSON: {e}")
            raise argparse.ArgumentTypeError(f"Invalid gRPC config JSON: {e}")
    return converter


def init_grpc_group_args(parser, grpc_config):
    ##############################################################################################################
    # gRPC Configuration
    ##############################################################################################################
    grpc_group = parser.add_argument_group("gRPC Configuration")
    
    default_json = '{"client_config": {"grpc.max_receive_message_length": 1073741824, "grpc.max_metadata_size": 1073741824}, "server_config": {"grpc.max_metadata_size": 1073741824,"grpc.max_concurrent_streams": 100000, "grpc.max_connection_idle_ms": 600000, "grpc.http2.min_recv_ping_interval_without_data_ms": 1000, "grpc.http2.max_ping_strikes": 1000}}'
    
    # 初始化默认值
    grpc_config.from_json(default_json)

    grpc_group.add_argument(
        "--grpc_config_json",
        env_name="GRPC_CONFIG_JSON",
        bind_to=None,  # 不需要绑定，type 函数已经直接修改了 grpc_config
        type=_grpc_config_from_json(grpc_config),
        default=default_json,
        help="gRPC configuration as JSON string. Format: {\"client_config\": {...}, \"server_config\": {...}}",
    )
