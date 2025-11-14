import argparse
import glob
import importlib
import logging
import os

import uvicorn
from anyio import CapacityLimiter
from anyio.lowlevel import RunVar
from fastapi import FastAPI
from fastapi.routing import APIRouter

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.uvicorn_config import get_uvicorn_logging_config

MAX_INCOMPLETE_EVENT_SIZE = 1024 * 1024


class ModelAssistantServer(object):
    def __init__(self, server_port):
        self._server_port = server_port

    def start(self):
        app = self.create_app()
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=self._server_port,
            log_config=get_uvicorn_logging_config(),
            h11_max_incomplete_event_size=MAX_INCOMPLETE_EVENT_SIZE,
        )

    def create_app(self):
        app = FastAPI()

        @app.on_event("startup")
        async def startup():
            RunVar("_default_thread_limiter").set(CapacityLimiter(40))

        api_router = ModelAssistantServer.include_api_from_subpath("api")

        # add APIRouter to FastAPI app
        app.include_router(api_router)

        return app

    @staticmethod
    def include_api_from_subpath(sub_path: str):
        router = APIRouter()

        # find all submodule
        current_path = os.path.dirname(os.path.abspath(__file__))
        py_files = glob.glob(
            os.path.join(current_path, sub_path, "*.py"), recursive=True
        )
        pyc_files = glob.glob(
            os.path.join(current_path, sub_path, "*.pyc"), recursive=True
        )
        all_files = py_files + pyc_files
        for filename in all_files:
            if os.path.isfile(filename):
                # convert path to module name
                module_name = "rtp_llm.tools." + filename.replace(
                    current_path, ""
                ).replace("/", ".")[:-3].strip(".")
                # add module
                module = importlib.import_module(module_name)
                if hasattr(module, "router"):
                    # add router
                    logging.info(f"register module:{module_name}, {filename}")
                    router.include_router(module.router)

        return router


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", "-p", type=int, required=False, default=8088, help="service port"
    )
    args = parser.parse_args()

    server = ModelAssistantServer(args.port)
    server.start()


if __name__ == "__main__":
    setup_logging()
    main()
