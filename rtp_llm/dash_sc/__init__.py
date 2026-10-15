"""DashSc gRPC (wire: predict_v2.proto) server and client.

Public API (P8-level explicit surface):

- ``DashScApp``: process-level gRPC server lifecycle (start/stop + signal handling).
- ``SamplingParams`` / ``DashScRequestControls``: strongly-typed request parameter objects.
- ``build_model_infer_request``: canonical client-side request builder.
- ``dash_sc_grpc_client_channel_options``: gRPC channel options (keepalive etc.)
  derived from ``DashScGrpcConfig``.
- ``decode_finish_reason``: parse the ``finish_reason`` output tensor (dtype-aware).

Everything else in this package is internal and subject to change without notice.
External callers must import via this package boundary, not via submodules.
"""


def __getattr__(name):
    # Keep pure thinking helpers usable without loading the server's native ops.
    from importlib import import_module

    if name == "DashScApp":
        module = "rtp_llm.dash_sc.app"
    elif name in {"SamplingParams", "DashScRequestControls"}:
        module = "rtp_llm.dash_sc.codec"
    elif name in {
        "build_model_infer_request",
        "dash_sc_grpc_client_channel_options",
        "decode_finish_reason",
    }:
        module = "rtp_llm.dash_sc.client"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


__all__ = [
    "DashScApp",
    "SamplingParams",
    "DashScRequestControls",
    "build_model_infer_request",
    "dash_sc_grpc_client_channel_options",
    "decode_finish_reason",
]
