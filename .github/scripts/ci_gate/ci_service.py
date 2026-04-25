"""Internal CI service API helpers and status parsing."""

import json
import time
from typing import Any, Dict, List, Tuple

from .common import CI_STATUS_URL, PIPELINE_ID, PROJECT_ID, GateError, log


def ci_service_request(payload, security, context, url=CI_STATUS_URL):
    # type: (Dict[str, Any], str, str, str) -> Any
    from .common import http_json

    status, body, raw_body = http_json(
        url,
        headers={"Authorization": "Basic %s" % security},
        payload=payload,
        context=context,
    )
    log("Response (HTTP %d): %s" % (status, raw_body))
    if status < 200 or status >= 300:
        raise GateError("::error::%s failed with HTTP %d" % (context, status))
    return body


def retrieve_task_status(commit_id, security, repo):
    # type: (str, str, str) -> Dict[str, Any]
    body = ci_service_request(
        {
            "type": "RETRIEVE-TASK-STATUS",
            "aone": {"projectId": PROJECT_ID, "pipelineId": PIPELINE_ID},
            "commitId": commit_id,
            "repositoryUrl": repo,
        },
        security,
        "querying CI status",
    )
    if not isinstance(body, dict):
        raise GateError("::error::CI status response is not a JSON object")
    return body


def get_branch_info(branch_name, repo, commit_id, security):
    # type: (str, str, str, str) -> Dict[str, Any]
    end_time = time.time() + 120
    while time.time() < end_time:
        body = ci_service_request(
            {
                "type": "RETRIEVE-BRANCH-INFO",
                "commitId": commit_id,
                "repositoryUrl": repo,
                "aone": {"projectId": PROJECT_ID},
                "branchName": branch_name,
                "clearCache": "false",
            },
            security,
            "retrieving branch info for %s" % branch_name,
        )
        if not isinstance(body, dict):
            raise GateError("::error::Branch info response is not a JSON object")
        if body.get("success") is not True:
            error_code = body.get("errorCode")
            error_msg = "Branch not found" if error_code == "SYSTEM_NOT_FOUND_ERROR" else body.get("errorMsg", "unknown")
            raise GateError("Error: Failed to query branch info - %s" % error_msg)
        info_raw = body.get("internal_branch_info")
        if info_raw and info_raw != "UNKNOWN":
            if isinstance(info_raw, str):
                return json.loads(info_raw)
            if isinstance(info_raw, dict):
                return info_raw
            raise GateError("::error::Unexpected internal_branch_info type")
        log("Waiting for branch info...")
        time.sleep(5)
    raise GateError("Timeout: Could not retrieve valid branch info within 2 minutes")


def branch_commit_id(branch_info):
    # type: (Dict[str, Any]) -> str
    return str(((branch_info.get("commit") or {}).get("id")) or "")


def _is_count_map(d):
    # type: (dict) -> bool
    """True when d looks like {"SUCCESS":2, "RUNNING":7} -- all values are ints."""
    return bool(d) and all(isinstance(v, int) for v in d.values())


def collect_status_tokens(value, allow_plain=True, status_map=False):
    # type: (Any, bool, bool) -> List[str]
    status_keys = {"status", "state", "result", "conclusion"}
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if allow_plain else []
    if isinstance(value, dict):
        if status_map and _is_count_map(value):
            return [str(k) for k, v in value.items() if v > 0]
        tokens = []  # type: List[str]
        has_status_key = any(str(key).lower() in status_keys for key in value)
        for key, nested in value.items():
            key_is_status = str(key).lower() in status_keys
            if status_map and not has_status_key:
                tokens.extend(collect_status_tokens(nested, allow_plain=True, status_map=True))
            elif key_is_status:
                tokens.extend(collect_status_tokens(nested, allow_plain=True, status_map=True))
            elif isinstance(nested, (dict, list)):
                tokens.extend(collect_status_tokens(nested, allow_plain=False, status_map=False))
        return tokens
    if isinstance(value, list):
        tokens = []
        for nested in value:
            tokens.extend(collect_status_tokens(nested, allow_plain=allow_plain, status_map=status_map))
        return tokens
    return [str(value)] if allow_plain else []


def normalize_status_payload(status):
    # type: (Any) -> Any
    if isinstance(status, str):
        stripped = status.strip()
        if stripped and stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return status
    return status


def parse_ci_status(response):
    # type: (Dict[str, Any]) -> Tuple[str, str]
    status_payload = normalize_status_payload(response.get("status"))
    if status_payload is None:
        return "UNKNOWN", "null"

    token_root = status_payload
    if isinstance(status_payload, dict) and "status" in status_payload:
        token_root = status_payload["status"]

    if isinstance(token_root, (dict, list)):
        status_summary = json.dumps(token_root, separators=(",", ":"))
    else:
        status_summary = str(token_root)

    tokens = [token.upper() for token in collect_status_tokens(token_root, status_map=True) if str(token).strip()]
    if not tokens:
        return "UNKNOWN", status_summary

    joined = " ".join(tokens)
    if any(word in joined for word in ("FAILED", "ERROR", "TIMEOUT", "CANCELED", "CANCELLED")):
        return "FAILED", status_summary
    if "RUNNING" in joined:
        return "RUNNING", status_summary
    if "PENDING" in joined:
        return "PENDING", status_summary

    real_success = {"SUCCESS", "SUCCEEDED", "DONE", "PASS", "PASSED"}
    ignorable = {"NOT_RUN", "SKIPPED"}
    all_acceptable = real_success | ignorable
    if all(token in all_acceptable for token in tokens):
        if any(token in real_success for token in tokens):
            return "DONE", status_summary
        log("::error::All CI jobs were %s — treating as FAILED" % "/".join(sorted(set(tokens))))
        return "FAILED", status_summary
    if len(tokens) == 1 and tokens[0] in {"UNKNOWN", "NULL"}:
        return "UNKNOWN", status_summary
    return "FAILED", status_summary
