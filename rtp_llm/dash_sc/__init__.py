"""DashSc gRPC (wire: predict_v2.proto) server and client.

Public API (P8-level explicit surface):

- ``DashScApp``: process-level gRPC server lifecycle (start/stop + signal handling).
- ``SamplingParams`` / ``OtherParams``: strongly-typed request parameter objects.
- ``build_model_infer_request``: canonical client-side request builder.
- ``dash_sc_grpc_client_channel_options``: gRPC channel options (keepalive etc.)
  derived from ``DashScGrpcConfig``.
- ``decode_finish_reason``: parse the ``finish_reason`` output tensor (dtype-aware).

Everything else in this package is internal and subject to change without notice.
External callers must import via this package boundary, not via submodules.
"""

from rtp_llm.dash_sc.client import (
    build_model_infer_request,
    dash_sc_grpc_client_channel_options,
    decode_finish_reason,
)
from rtp_llm.dash_sc.codec import OtherParams, SamplingParams


def __getattr__(name):
    if name == "DashScApp":
        from rtp_llm.dash_sc.app import DashScApp

        return DashScApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DashScApp",
    "SamplingParams",
    "OtherParams",
    "build_model_infer_request",
    "dash_sc_grpc_client_channel_options",
    "decode_finish_reason",
]
