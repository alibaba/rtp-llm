import os
import json
from typing import Dict, Any, Union, List
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.requests import Request
from threading import Thread
import uvicorn

import logging
logging.basicConfig(level="INFO",
                format="[process-%(process)d][%(name)s][%(asctime)s][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
                datefmt='%m/%d/%Y %H:%M:%S')

UVICORN_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',  # noqa: E501
        },
    },
    "handlers": {
        "access": {
            "formatter": "access",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "uvicorn_access.log",
            "maxBytes": 256 * 1024,
            "backupCount": 4,
        },
    },
    "loggers": {
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}

class FakeServer:
    def __init__(self, running_task_list: List[Any]):
        self.app = FastAPI(debug=True)
        self.server_thread = None

        @self.app.get("/worker_status")
        async def worker_status():
            resposne = {
                "available_concurrency": 10,
                "available_kv_cache": 1000,
                "total_kv_cache": 1000,
                "step_latency_ms": 10,
                "step_per_minute": 100,
                "iterate_count" : 100,
                "alive": True,
                "running_task_list": running_task_list,
                "finished_task_list": [],
                "last_schedule_time": 100000,
                "machine_info": "FAKE_GPU_TP1_PP1_EP1_W16A16"
            }
            return ORJSONResponse(resposne)

        @self.app.post("/tokenize")
        async def tokenize(req: Union[Dict[str, str], str]):
            if isinstance(req, str):
                req = json.loads(req)
            prompt = req['prompt']
            return ORJSONResponse({"token_ids": [i for i in range(len(prompt))]})

    def start(self, port: int):
        logging.info(f"server log in {os.path.join(os.getcwd(), 'uvicorn_access.log')}")
        uvicorn.run(self.app, host="127.0.0.1", port=port, log_config=UVICORN_LOGGING_CONFIG)

if __name__ == '__main__':
    server = FakeServer([])
    server.start(8088) 