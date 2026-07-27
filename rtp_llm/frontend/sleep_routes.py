import hmac
import os
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, Request
from fastapi.responses import ORJSONResponse

from rtp_llm.utils.grpc_client_wrapper import (
    normalize_sleep_request,
    normalize_wake_request,
)

# Opt-in admin token for the lifecycle control routes. When this env var is set
# to a non-empty value, /sleep, /wake_up, /is_sleeping and /sleep_status require
# a matching credential (see _sleep_admin_auth_error).
SLEEP_ADMIN_TOKEN_ENV = "RTP_LLM_SLEEP_ADMIN_TOKEN"


def _configured_sleep_admin_token() -> str:
    return os.environ.get(SLEEP_ADMIN_TOKEN_ENV, "").strip()


def _presented_sleep_admin_token(request: Request) -> Optional[str]:
    # Accept either a dedicated admin header or a standard bearer token.
    header_token = request.headers.get("X-Sleep-Admin-Token")
    if header_token:
        return header_token.strip()
    authorization = request.headers.get("Authorization", "")
    bearer_prefix = "Bearer "
    if authorization.startswith(bearer_prefix):
        return authorization[len(bearer_prefix) :].strip()
    return None


def _sleep_admin_auth_error(request: Request) -> Optional[ORJSONResponse]:
    """Gate the lifecycle control routes behind an optional admin token.

    SECURITY TRADEOFF: when ``RTP_LLM_SLEEP_ADMIN_TOKEN`` is unset (the default),
    these endpoints stay unauthenticated so existing L1/L2 deployments that
    already drive unauthenticated /sleep|/wake_up keep working unchanged. Any
    deployment whose frontend port is reachable beyond a trusted admin network
    should set the env var: a single unauthenticated /sleep triggers a
    whole-group CUDA process checkpoint (instant service outage). When set, a
    matching ``Authorization: Bearer <token>`` or ``X-Sleep-Admin-Token: <token>``
    header is required; a missing credential returns 401 and a wrong one 403.
    """
    expected = _configured_sleep_admin_token()
    if not expected:
        return None
    presented = _presented_sleep_admin_token(request)
    if presented is None:
        return ORJSONResponse(
            status_code=401,
            content={"error": "sleep admin authentication required"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Constant-time compare to avoid leaking the token via timing.
    if not hmac.compare_digest(presented, expected):
        return ORJSONResponse(
            status_code=403,
            content={"error": "sleep admin authentication failed"},
        )
    return None


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
    # Levels 1/2/3 are accepted here; the configured backend level remains the
    # authority and rejects mismatches.
    @app.post("/sleep")
    async def sleep(request: Request, req: Optional[Dict[Any, Any]] = Body(None)):
        auth_error = _sleep_admin_auth_error(request)
        if auth_error is not None:
            return auth_error
        configured_level = getattr(grpc_client, "configured_sleep_level", 1)
        if not isinstance(configured_level, int):
            configured_level = 1
        req, validation_error = normalize_sleep_request(req, configured_level)
        if validation_error is not None:
            return ORJSONResponse(status_code=400, content=validation_error)
        assert req is not None
        response = await grpc_client.post_request("sleep", req)
        if "error" in response:
            return ORJSONResponse(
                status_code=sleep_error_status(response), content=response
            )
        return response

    @app.post("/wake_up")
    async def wake_up(request: Request, req: Optional[Dict[Any, Any]] = Body(None)):
        auth_error = _sleep_admin_auth_error(request)
        if auth_error is not None:
            return auth_error
        req, validation_error = normalize_wake_request(req)
        if validation_error is not None:
            return ORJSONResponse(status_code=400, content=validation_error)
        assert req is not None
        response = await grpc_client.post_request("wake_up", req)
        if "error" in response:
            return ORJSONResponse(
                status_code=sleep_error_status(response), content=response
            )
        return response

    @app.get("/is_sleeping")
    async def is_sleeping(request: Request):
        auth_error = _sleep_admin_auth_error(request)
        if auth_error is not None:
            return auth_error
        response = await grpc_client.post_request("is_sleeping", {})
        if "error" in response:
            return ORJSONResponse(status_code=500, content=response)
        return response

    @app.get("/sleep_status")
    async def sleep_status(request: Request):
        auth_error = _sleep_admin_auth_error(request)
        if auth_error is not None:
            return auth_error
        response = await grpc_client.post_request("sleep_status", {})
        if "error" in response:
            return ORJSONResponse(status_code=500, content=response)
        return response
