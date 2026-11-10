"""AWI evaluator contract endpoints (evaluator-protocol.md).

Mounted by FrontendApp.create_app. These endpoints expose:

- POST /awi/generate      pretokenized benchmark generation. Propagates the
                          X-AWI-Request-ID header (request_headers allowlist
                          -> RequestInfoPB.request_id) so scheduler-side
                          observations carry the benchmark's own id.
- POST /awi/barrier       arms an operation-scoped admission/residency probe
                          on every rank (AwiCommand gRPC broadcast).
- GET  /awi/observations  aggregates per-rank telemetry into the contract
                          document: build source pair, configuration,
                          startup network fetch count, the whole-checkpoint
                          weight placement census (rank documents written by
                          model_loader.awi_recorder) and raw engine events
                          drained from every rank's AwiRankStore.

Nothing here synthesizes counters: weights come from rank census documents
derived from live tensors, events come from engine instrumentation, and the
network-fetch count from the StartupNetworkGuard installed at process start.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict

from fastapi import Request
from fastapi.responses import ORJSONResponse
from rtp_llm.utils.awi_telemetry import (
    StartupNetworkGuard,
    build_source,
    read_rank_documents,
    startup_network_fetch_count,
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "on", "yes")


def _configuration() -> Dict[str, Any]:
    import torch

    graph_backend = "hip" if torch.version.hip is not None else "cuda"
    return {
        "context_tokens": int(os.environ.get("MAX_SEQ_LEN", "0")),
        "memory_kv_reuse": _env_flag("REUSE_CACHE") and _env_flag("ENABLE_MEMORY_CACHE"),
        "graph_backend": graph_backend,
    }


def _weights_census() -> Dict[str, Any]:
    documents = read_rank_documents("weights")
    rows = []
    ranks = []
    coverage = []
    for document in documents:
        ranks.append(document.get("world_rank"))
        rows.extend(document.get("rows", []))
        coverage.append(
            {
                "world_rank": document.get("world_rank"),
                **(document.get("coverage") or {}),
            }
        )
    return {"rows": rows, "ranks": sorted(r for r in ranks if r is not None), "coverage": coverage}


async def _awi_command(frontend_app: Any, op: str, **kwargs: Any) -> Dict[str, Any]:
    payload = {
        "command_id": uuid.uuid4().hex,
        "op": op,
        "operation_id": kwargs.get("operation_id", ""),
        "request_ids": kwargs.get("request_ids", []),
        "context_tokens": kwargs.get("context_tokens", 0),
    }
    return await frontend_app.grpc_client.post_request("awi_command", payload)


def register_awi_routes(app: Any, frontend_app: Any) -> None:
    @app.post("/awi/generate")
    async def awi_generate(request: Request):
        """Pretokenized benchmark generation with OpenAI-compatible output."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return ORJSONResponse(status_code=400, content={"error": "invalid JSON body"})
        input_ids = body.get("input_ids")
        if not isinstance(input_ids, list) or not all(
            isinstance(t, int) for t in input_ids
        ):
            return ORJSONResponse(
                status_code=400, content={"error": "input_ids must be a list of ints"}
            )
        max_tokens = int(body.get("max_tokens", 1))
        extra = body.get("extra_configs") or {}

        generate_config: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "aux_info": True,
        }
        if "min_new_tokens" in extra:
            generate_config["min_new_tokens"] = int(extra["min_new_tokens"])
        if "ignore_eos" in extra:
            generate_config["ignore_eos"] = bool(extra["ignore_eos"])
        for key in ("temperature", "top_p", "top_k"):
            if key in extra:
                generate_config[key] = extra[key]
            elif key in body:
                generate_config[key] = body[key]

        awi_request_id = request.headers.get("x-awi-request-id", "")
        legacy_req = {
            "prompt_tokens": input_ids,
            "generate_config": generate_config,
        }
        start_ns = time.time_ns()
        rep = await frontend_app.frontend_server.inference(legacy_req, request)
        end_ns = time.time_ns()

        status_code = getattr(rep, "status_code", 200)
        try:
            payload = json.loads(bytes(rep.body).decode())  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return ORJSONResponse(
                status_code=502,
                content={"error": "backend produced a non-JSON response"},
            )
        if payload.get("error_code") not in (None, 0):
            return ORJSONResponse(
                status_code=status_code if status_code >= 400 else 500,
                content=payload,
            )
        aux = payload.get("aux_info") or {}
        text = payload.get("response") or ""
        prompt_tokens = int(aux.get("input_len") or len(input_ids))
        completion_tokens = int(aux.get("output_len") or 0)
        return ORJSONResponse(
            content={
                "id": awi_request_id or str(payload.get("request_id", "")),
                "object": "chat.completion",
                "created": int(time.time()),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop" if payload.get("finished") else "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "aux_info": aux,
                "awi": {
                    "request_id": awi_request_id,
                    "frontend_start_ns": start_ns,
                    "frontend_end_ns": end_ns,
                },
            }
        )

    @app.post("/awi/barrier")
    async def awi_barrier(request: Request):
        """Arm an operation-scoped admission/residency probe on every rank."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return ORJSONResponse(status_code=400, content={"error": "invalid JSON body"})
        operation_id = str(body.get("operation_id") or "")
        request_ids = [str(r) for r in (body.get("request_ids") or [])]
        context_tokens = int(body.get("context_tokens") or 0)
        if not operation_id:
            return ORJSONResponse(
                status_code=400, content={"error": "operation_id is required"}
            )
        result = await _awi_command(
            frontend_app,
            "barrier",
            operation_id=operation_id,
            request_ids=request_ids,
            context_tokens=context_tokens,
        )
        status_code = 200 if result.get("status") == "ok" else 502
        return ORJSONResponse(
            status_code=status_code,
            content={
                "operation_id": operation_id,
                "armed_ranks": [
                    r["world_rank"] for r in result.get("results", []) if r.get("success")
                ],
                "errors": result.get("errors", []),
            },
        )

    @app.get("/awi/observations")
    async def awi_observations(request: Request):
        operation_id = request.query_params.get("operation_id", "")
        events_result = await _awi_command(
            frontend_app, "collect", operation_id=operation_id
        )
        events = []
        for rank_result in events_result.get("results", []):
            payload = rank_result.get("payload") or {}
            for event in payload.get("events", []):
                events.append(event)
        events.sort(key=lambda e: (e.get("host_ns") or e.get("end_ns") or 0))
        census = _weights_census()
        # Publish this (frontend) process's own count so the aggregate
        # includes it alongside every backend rank's published document.
        StartupNetworkGuard.publish()
        return ORJSONResponse(
            content={
                "build": {"source": build_source()},
                "configuration": _configuration(),
                "startup_network_fetches": startup_network_fetch_count(),
                "weights": census["rows"],
                "weights_coverage": {
                    "ranks": census["ranks"],
                    "per_rank": census["coverage"],
                },
                "events": events,
                "event_collection": {
                    "status": events_result.get("status"),
                    "errors": events_result.get("errors", []),
                },
            }
        )

    logging.info("[awi] evaluator endpoints registered")
