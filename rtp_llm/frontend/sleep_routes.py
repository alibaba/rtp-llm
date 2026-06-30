from typing import Any, Dict, Optional

from fastapi import Body, FastAPI
from fastapi.responses import ORJSONResponse

from rtp_llm.frontend.sleep_validation import unsupported_lifecycle_control_field


def sleep_error_status(response: Dict[str, Any]) -> int:
    grpc_status = response.get("grpc_status")
    if grpc_status == "UNIMPLEMENTED":
        return 501
    if grpc_status == "INVALID_ARGUMENT":
        return 400
    if grpc_status == "FAILED_PRECONDITION":
        return 409
    return 500


def register_sleep_routes(app: FastAPI, grpc_client: Any) -> None:
    # request format (all fields optional):
    #   {"level": 1, "mode": "wait"|"abort", "timeout_ms": 30000, "reason": "...", "tags": []}
    # level=0 is a defined state-preserving sleep level, but is currently unimplemented.
    @app.post("/sleep")
    async def sleep(req: Optional[Dict[Any, Any]] = Body(None)):
        req = req or {}
        unsupported_field = unsupported_lifecycle_control_field(req)
        if unsupported_field:
            return ORJSONResponse(
                status_code=400,
                content={"error": f"sleep {unsupported_field} is unsupported"},
            )
        try:
            level = int(req.get("level", 1))
            if "timeout_ms" in req:
                int(req["timeout_ms"])
        except (TypeError, ValueError):
            return ORJSONResponse(
                status_code=400,
                content={"error": "sleep level and timeout_ms must be integers"},
            )
        if level not in (0, 1):
            return ORJSONResponse(
                status_code=400,
                content={"error": "sleep level must be 0 or 1"},
            )
        mode = req.get("mode", "wait")
        if mode not in ("wait", "abort"):
            return ORJSONResponse(
                status_code=400,
                content={"error": 'sleep mode must be "wait" or "abort"'},
            )
        tags = req.get("tags", [])
        if tags is None:
            tags = []
        if not isinstance(tags, list):
            return ORJSONResponse(
                status_code=400,
                content={"error": "sleep tags must be a list"},
            )
        if any(not isinstance(tag, str) or not tag for tag in tags):
            return ORJSONResponse(
                status_code=400,
                content={"error": "sleep tags must be non-empty strings"},
            )
        response = await grpc_client.post_request("sleep", req)
        if "error" in response:
            return ORJSONResponse(
                status_code=sleep_error_status(response), content=response
            )
        return response

    @app.post("/wake_up")
    async def wake_up(req: Optional[Dict[Any, Any]] = Body(None)):
        req = req or {}
        unsupported_field = unsupported_lifecycle_control_field(req)
        if unsupported_field:
            return ORJSONResponse(
                status_code=400,
                content={"error": f"wake_up {unsupported_field} is unsupported"},
            )
        response = await grpc_client.post_request("wake_up", req)
        if "error" in response:
            return ORJSONResponse(
                status_code=sleep_error_status(response), content=response
            )
        return response

    @app.get("/is_sleeping")
    async def is_sleeping():
        response = await grpc_client.post_request("is_sleeping", {})
        if "error" in response:
            return ORJSONResponse(status_code=500, content=response)
        return response

    @app.get("/sleep_status")
    async def sleep_status():
        response = await grpc_client.post_request("sleep_status", {})
        if "error" in response:
            return ORJSONResponse(status_code=500, content=response)
        return response
