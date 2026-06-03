"""Standalone reverse-proxy entry: ``python -m rtp_llm.dash_sc.proxy``.

Binds the dash_sc gRPC server with a ``DashScProxyServicer`` that forwards
every ``ModelStreamInfer`` call to the backend(s) configured via
``SERVICE_ROUTE`` — useful for running a standalone proxy process in tests /
canary deployments without instantiating the backend visitor.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from rtp_llm.dash_sc.proxy.servicer import DashScProxyServicer
from rtp_llm.dash_sc.server import DashScGrpcServer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DashSc gRPC reverse-proxy (predict_v2.proto)"
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
        servicer = DashScProxyServicer()
        grpc_server = DashScGrpcServer(dash_sc_grpc_config=dash_sc_cfg)
        server = await grpc_server.start(args.port, servicer=servicer)
        await server.wait_for_termination()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
