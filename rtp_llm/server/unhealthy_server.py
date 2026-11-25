"""HTTP health probes on the original frontend ports of an excluded host."""

import logging
import socket
from contextlib import ExitStack
from copy import copy

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from uvicorn import Config, Server

from rtp_llm.config.py_config_modules import PyEnvConfigs


def serve_unhealthy_host(configs: PyEnvConfigs):
    app = FastAPI(openapi_url=None, redirect_slashes=False)

    @app.get("/liveness")
    @app.post("/liveness")
    async def liveness():
        return JSONResponse("ok")

    @app.exception_handler(404)
    @app.exception_handler(405)
    async def unavailable(request, exc):
        # Preserve the path-based probe contract for other HTTP methods too.
        if request.url.path == "/liveness":
            return await liveness()
        return JSONResponse(
            {"error": "Host excluded from the inference world"}, status_code=500
        )

    pc = configs.parallelism_config
    sc = copy(configs.server_config)
    with ExitStack() as listeners:
        sockets = []
        if sc.frontend_server_count > 0:
            for slot in range(0, pc.local_world_size, pc.tp_size):
                sc.set_local_rank(slot)
                sockets.append(
                    listeners.enter_context(
                        socket.create_server(("0.0.0.0", sc.server_port))
                    )
                )
        logging.info(
            "Excluded host listening on HTTP ports %s",
            [s.getsockname()[1] for s in sockets],
        )
        server = Server(
            Config(app, access_log=False, timeout_keep_alive=sc.timeout_keep_alive)
        )
        server.run(sockets=sockets)
