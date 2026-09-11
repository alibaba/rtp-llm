"""Opt-in test endpoint: acknowledge scheduling without consuming engine output."""

import time

from fastapi import HTTPException
from fastapi.responses import JSONResponse


def register_mock_schedule(app, frontend, track):
    @app.post("/internal/mock/schedule")
    async def schedule(body: dict):
        async def call():
            import torch
            from rtp_llm.config.generate_config import GenerateConfig
            from rtp_llm.frontend.request_id_generator import generate_request_id
            from rtp_llm.utils.base_model_datatypes import GenerateInput

            if set(body) - {
                "prompt",
                "input_ids",
                "max_new_tokens",
                "priority",
                "timeout_ms",
            }:
                raise HTTPException(422, "unsupported schedule-only field")
            if ("prompt" in body) == ("input_ids" in body):
                raise HTTPException(422, "provide exactly one of prompt or input_ids")
            worker = frontend._frontend_worker
            tokens = body.get("input_ids")
            if tokens is None:
                if not isinstance(body["prompt"], str):
                    raise HTTPException(422, "prompt must be text")
                tokens = worker.pipeline.encode(body["prompt"])
            output = body.get("max_new_tokens", 16)
            priority = body.get("priority", 50)
            timeout = body.get("timeout_ms", 30000)
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(type(t) is not int or not 0 <= t <= 2147483647 for t in tokens)
                or type(output) is not int
                or not 1 <= output <= 4096
                or type(priority) is not int
                or not 1 <= priority <= 100
                or type(timeout) is not int
                or not 1 <= timeout <= 60000
            ):
                raise HTTPException(422, "invalid token plan or schedule budget")
            if len(tokens) + output > worker.backend_rpc_server_visitor.max_seq_len:
                raise HTTPException(422, "token plan exceeds max_seq_len")
            sequence = frontend._global_controller.increment()
            try:
                config = frontend.py_env_configs.server_config
                rid = generate_request_id(
                    config.ip,
                    config.server_port,
                    frontend.server_id,
                    sequence % 4096,
                )
                request = GenerateInput(
                    request_id=rid,
                    token_ids=torch.tensor(tokens, dtype=torch.int32),
                    mm_inputs=[],
                    generate_config=GenerateConfig(
                        max_new_tokens=output, timeout_ms=timeout, qos_priority=priority
                    ),
                    headers={"x-dashscope-inner-qos-level": str(priority)},
                )
                visitor = worker.backend_rpc_server_visitor
                visitor.fill_request_info(request)
                start = time.monotonic()
                # Use the production token hash / master client path, with no domain fallback,
                # engine RPC, stream reader, retries or generated completion response.
                failed = await visitor.get_master_route_addrs(request)
                if failed is not None:
                    raise HTTPException(503, "master unavailable")
                if not request.enqueued_by_master:
                    raise HTTPException(409, "schedule-only requires BATCH enqueue")
                return JSONResponse(
                    status_code=202,
                    content={
                        "status": "accepted",
                        "request_id": str(rid),
                        "enqueued_by_master": True,
                        "fetch_output_stream": False,
                        "inference_completed": False,
                        "schedule_ms": (time.monotonic() - start) * 1000,
                    },
                )
            finally:
                frontend._global_controller.decrement()

        return await track(call)
