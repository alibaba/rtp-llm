import json
from typing import Dict, Any, Union
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.requests import Request
from threading import Thread
import uvicorn

import logging
logging.basicConfig(level="INFO",
                format="[process-%(process)d][%(name)s][%(asctime)s][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
                datefmt='%m/%d/%Y %H:%M:%S')

class FakeServer:
    def __init__(self):
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
                "onflight_requests": 0,
                "alive": True,
                "running_task_list": [{"prefix_length": 5, "input_length": 5, "task_id": "100"}],
                "finished_task_list": [{"prefix_length": 5, "input_length": 5, "task_id": "100"}],
                "last_schedule_time": 100000,
                "machine_info": "FAKE_GPU_TP1"
            }
            return ORJSONResponse(resposne)

        @self.app.post("/tokenize")
        async def tokenize(req: Union[Dict[str, str], str]):
            # body = await request_str.body()
            print("body:", req)
            request_str = json.loads(req)['prompt']
            return ORJSONResponse({"token_ids": [i for i in range(len(request_str))]})

    def start(self, port: int):
        uvicorn.run(self.app, host="127.0.0.1", port=port)

if __name__ == '__main__':
    server = FakeServer()
    server.start(8088) 

