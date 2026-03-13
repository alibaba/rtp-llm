import argparse
import json
import logging

DEFAULT_GRPC_MAX_SERVER_POLLERS = 4
DEFAULT_BAILIAN_GRPC_MAX_SERVER_WORKERS = 4

# Model RPC: receive / metadata limits (C++ GenerateStreamCall path).
_MODEL_RPC_GRPC_RECV_AND_METADATA_BYTES = 1024 * 1024 * 1024


def _default_model_grpc_config_json() -> str:
    b = _MODEL_RPC_GRPC_RECV_AND_METADATA_BYTES
    return json.dumps(
        {
            "client_config": {
                "grpc.max_receive_message_length": b,
                "grpc.max_metadata_size": b,
            },
            "server_config": {
                "grpc.max_metadata_size": b,
                "grpc.max_concurrent_streams": 100000,
                "grpc.max_connection_idle_ms": 600000,
                "grpc.http2.min_recv_ping_interval_without_data_ms": 1000,
                "grpc.http2.max_ping_strikes": 1000,
            },
        },
        separators=(",", ":"),
    )


def default_model_grpc_config_json() -> str:
    """Model RPC maps from ``_default_model_grpc_config_json`` plus ``max_server_pollers``."""
    obj = json.loads(_default_model_grpc_config_json())
    obj["max_server_pollers"] = DEFAULT_GRPC_MAX_SERVER_POLLERS
    return json.dumps(obj, separators=(",", ":"))


def default_bailian_grpc_config_json() -> str:
    """Same as ``default_model_grpc_config_json``, plus Bailian ``max_server_workers``."""
    obj = json.loads(default_model_grpc_config_json())
    obj["max_server_workers"] = DEFAULT_BAILIAN_GRPC_MAX_SERVER_WORKERS
    return json.dumps(obj, separators=(",", ":"))


def _grpc_config_from_json(grpc_config):
    """创建类型转换函数，将 JSON 字符串写入传入的 config 对象。"""

    def converter(json_str):
        if json_str is None or json_str == "":
            return grpc_config
        try:
            grpc_config.from_json(json_str)
            logging.debug("Initialized gRPC config from JSON: %s", json_str)
            return grpc_config
        except Exception as e:
            logging.warning("Failed to parse gRPC config JSON: %s", e)
            raise argparse.ArgumentTypeError(f"Invalid gRPC config JSON: {e}") from e

    return converter


def init_model_grpc_group_args(parser, grpc_config):
    """Model RPC（C++ GenerateStreamCall）用的 gRPC channel / server 参数。"""
    model_grpc_group = parser.add_argument_group("Model RPC gRPC configuration")

    default_json = default_model_grpc_config_json()

    grpc_config.from_json(default_json)

    model_grpc_group.add_argument(
        "--grpc_config_json",
        env_name="GRPC_CONFIG_JSON",
        bind_to=None,
        type=_grpc_config_from_json(grpc_config),
        default=default_json,
        help=(
            "Model RPC gRPC JSON: "
            '{"client_config": {...}, "server_config": {...}, '
            '"max_server_pollers": <int, default 4>}. '
            "max_server_pollers>0 sets C++ sync server SetSyncServerOption(MAX_POLLERS, value) "
            "(per completion queue)."
        ),
    )


def init_bailian_grpc_group_args(parser, bailian_grpc_config):
    """Bailian gRPC（predict_v2.proto ModelStreamInfer）Python client / server。"""
    bailian_group = parser.add_argument_group("Bailian gRPC configuration")

    default_json = default_bailian_grpc_config_json()

    bailian_grpc_config.from_json(default_json)

    bailian_group.add_argument(
        "--bailian_grpc_config_json",
        env_name="BAILIAN_GRPC_CONFIG_JSON",
        bind_to=None,
        type=_grpc_config_from_json(bailian_grpc_config),
        default=default_json,
        help=(
            "Bailian gRPC JSON: "
            '{"client_config": {...}, "server_config": {...}, "max_server_workers": <int>}. '
            "max_server_workers is ThreadPoolExecutor size for grpc.server."
        ),
    )
