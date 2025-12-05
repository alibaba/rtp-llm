import json
import logging
import os


def init_grpc_group_args(parser):
    ##############################################################################################################
    # Grpc Group
    ##############################################################################################################
    grpc_group = parser.add_argument_group("Grpc group")

    grpc_group.add_argument(
        "--grpc_config_json",
        env_name="GRPC_CONFIG_JSON",
        type=str,
        default='{"client_config": {"grpc.max_receive_message_length": 1073741824, "grpc.max_metadata_size": 1073741824}, "server_config": {"grpc.max_metadata_size": 1073741824,"grpc.max_concurrent_streams": 100000, "grpc.max_connection_idle_ms": 600000, "grpc.http2.min_recv_ping_interval_without_data_ms": 1000, "grpc.http2.max_ping_strikes": 1000}}',
        help="gRPC configuration as JSON string.",
    )
