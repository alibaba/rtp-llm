"""Pure aggregation helpers for multi-rank lifecycle status."""

from typing import Any, Dict, List


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def error_details(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    details = []
    for result in results:
        if "error" not in result:
            continue
        detail = {
            "address": result.get("address", ""),
            "error": result.get("error", ""),
        }
        if "grpc_status" in result:
            detail["grpc_status"] = result["grpc_status"]
        details.append(detail)
    return details


def recovery_required(
    operation: str, reason: str, statuses: List[Dict[str, Any]]
) -> Dict[str, Any]:
    details = []
    for status in statuses:
        detail = {"address": status.get("address", "")}
        if "error" in status:
            detail["error"] = status["error"]
            if "grpc_status" in status:
                detail["grpc_status"] = status["grpc_status"]
        else:
            detail["state"] = status.get("state", "")
        details.append(detail)
    return {
        "error": f"RECOVERY_REQUIRED: {operation} {reason}",
        "grpc_status": "FAILED_PRECONDITION",
        "recovery_required": True,
        "details": details,
    }


def aggregate(
    results: List[Dict[str, Any]], coverage_error: str = ""
) -> Dict[str, Any]:
    successes = [result for result in results if "error" not in result]
    failures = [result for result in results if "error" in result]
    if not successes:
        return {
            "error": "Failed to get sleep status from all control ranks",
            "grpc_status": (
                failures[0].get("grpc_status", "UNAVAILABLE")
                if failures
                else "UNAVAILABLE"
            ),
            "details": error_details(results),
        }
    if failures:
        return {
            "error": "Failed to get sleep status from some control ranks",
            "grpc_status": failures[0].get("grpc_status", "UNAVAILABLE"),
            "details": error_details(results),
        }

    fields = (
        "state",
        "sleep_mode_enabled",
        "effective",
        "gpu_resource_state",
        "kv_memory_state",
        "supported_levels",
        "supported_modes",
    )
    if any(
        len({str(result.get(field, "")) for result in successes}) != 1
        for field in fields
    ):
        return {
            "error": "Sleep status did not converge across control ranks",
            "grpc_status": "FAILED_PRECONDITION",
        }

    aggregate_status = dict(successes[0])
    aggregate_status["sleep_epoch"] = max(
        _as_int(result.get("sleep_epoch", 0)) for result in successes
    )
    aggregate_status["active_request_count"] = sum(
        _as_int(result.get("active_request_count", 0)) for result in successes
    )
    aggregate_status["active_cache_transfer_count"] = sum(
        _as_int(result.get("active_cache_transfer_count", 0)) for result in successes
    )
    aggregate_status["device_kv_cache_valid"] = all(
        bool(result.get("device_kv_cache_valid", False)) for result in successes
    )
    aggregate_status.pop("address", None)
    aggregate_status.pop("status", None)
    if coverage_error:
        aggregate_status["effective"] = False
        aggregate_status["supported_levels"] = []
        aggregate_status["supported_modes"] = []
        aggregate_status["disabled_reason"] = coverage_error
    return aggregate_status
