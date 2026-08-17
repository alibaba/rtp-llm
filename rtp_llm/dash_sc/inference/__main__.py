"""Standalone fake-inference entry: ``python -m rtp_llm.dash_sc.inference``.

Binds the dash_sc gRPC server with a ``DashScInferenceServicer`` running in
fake mode (``backend_visitor=None``) — useful for grpcurl/dashllm_client
smoke tests without needing the real engine. Real inference in production
is wired through :class:`rtp_llm.dash_sc.app.DashScApp`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from rtp_llm.dash_sc.inference.servicer import DashScInferenceServicer
from rtp_llm.dash_sc.server import DashScGrpcServer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DashSc gRPC server (predict_v2.proto, fake inference mode)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="gRPC port (default: 8000)"
    )
    parser.add_argument(
        "--dash_sc_grpc_config_json",
        type=str,
        default="",
        help="Optional JSON for DashScGrpcConfig.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dash_sc_cfg = None
    if args.dash_sc_grpc_config_json.strip():
        from rtp_llm.ops import DashScGrpcConfig

        dash_sc_cfg = DashScGrpcConfig()
        dash_sc_cfg.from_json(args.dash_sc_grpc_config_json.strip())

    async def _run() -> None:
        servicer = DashScInferenceServicer(backend_visitor=None)
        grpc_server = DashScGrpcServer(dash_sc_grpc_config=dash_sc_cfg)
        server = await grpc_server.start(args.port, servicer=servicer)
        await server.wait_for_termination()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
