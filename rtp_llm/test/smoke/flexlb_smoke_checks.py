import copy
import json
import logging
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM
from rtp_llm.utils.util import str_to_bool

FLEXLB_QUEUE_FULL_CODE = 8502
FLEXLB_QUEUE_TIMEOUT_CODE = 8503

# Qwen3 Next cold-start through frontend->FlexLB can exceed 30s on CI.
# Keep functional probes aligned with existing PD load-cache smoke timeouts.
FLEXLB_SMOKE_DEFAULT_TIMEOUT_MS = "120000"
FLEXLB_SMOKE_DEFAULT_MASTER_SESSION_TIMEOUT_S = "10"


def make_worker_dp_leader_addrs(
    base_port: int,
    dp_size: int,
    tp_size: int = 1,
    world_size: Optional[int] = None,
    host: str = "127.0.0.1",
) -> List[str]:
    """Return worker HTTP addrs that can accept frontend/FlexLB traffic.

    start_server.py only starts frontend/http workers for rank 0 and TP-group
    leaders (`rank % tp_size == 0`). FlexLB must discover those DP leaders,
    not every rank inside the TP group, otherwise it can route to a non-leader
    rank and the request hangs until the RPC timeout.
    """

    base_port = int(base_port)
    dp_size = max(1, int(dp_size or 1))
    tp_size = max(1, int(tp_size or 1))
    world_size = max(1, int(world_size or dp_size * tp_size))

    addrs = []
    for dp_rank in range(dp_size):
        rank = dp_rank * tp_size
        if rank >= world_size:
            break
        addrs.append(f"{host}:{base_port + rank * MIN_WORKER_INFO_PORT_NUM}")
    return addrs or [f"{host}:{base_port}"]


def flexlb_check_enabled(
    flexlb_envs: Dict[str, str],
    name: str,
    default: bool = False,
) -> bool:
    raw_value = flexlb_smoke_env(flexlb_envs, name)
    if raw_value is None:
        return default
    return str_to_bool(str(raw_value))


def flexlb_smoke_env(
    flexlb_envs: Dict[str, str],
    name: str,
    default: Optional[str] = None,
) -> Optional[str]:
    return flexlb_envs.get(name, os.environ.get(name, default))


def flexlb_smoke_positive_int_list(
    flexlb_envs: Dict[str, str],
    name: str,
    default: str,
) -> List[int]:
    raw_value = flexlb_smoke_env(flexlb_envs, name, default) or default
    values: List[int] = []
    for raw_item in str(raw_value).split(","):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        values.append(max(1, int(raw_item)))
    return values


def apply_flexlb_frontend_timeouts(frontend_envs: Dict[str, str]) -> None:
    frontend_envs.setdefault(
        "MASTER_DEFAULT_TIMEOUT_MS",
        os.environ.get(
            "FLEXLB_SMOKE_MASTER_TIMEOUT_MS",
            FLEXLB_SMOKE_DEFAULT_TIMEOUT_MS,
        ),
    )
    frontend_envs.setdefault(
        "MASTER_SESSION_TIMEOUT_S",
        os.environ.get(
            "FLEXLB_SMOKE_MASTER_SESSION_TIMEOUT_S",
            FLEXLB_SMOKE_DEFAULT_MASTER_SESSION_TIMEOUT_S,
        ),
    )


def plain_json(value: Any) -> Any:
    # Smoke probes cross HTTP, Pydantic, dataclass, and generated enum objects;
    # normalize that boundary data before inspecting aux_info/debug fields.
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return plain_json(value.model_dump())
    if is_dataclass(value):
        return plain_json(asdict(value))
    if isinstance(value, dict):
        return {k: plain_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_json(v) for v in value]
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        return value.value
    return value


def _role_name(role: Any) -> str:
    # Role values may arrive as Python enums, Java-style enum strings, or raw
    # JSON strings depending on whether the source is frontend aux_info or
    # FlexLB /schedule debug output.
    if hasattr(role, "name"):
        return str(role.name).upper()
    role_str = str(role).upper()
    if "." in role_str:
        role_str = role_str.rsplit(".", 1)[-1]
    return role_str


def _response_aux_info(response: Any) -> Any:
    data = plain_json(response)
    if not isinstance(data, dict):
        return None
    aux_info = data.get("aux_info")
    if aux_info is not None:
        return aux_info
    response_batch = data.get("response_batch")
    if isinstance(response_batch, list) and response_batch:
        first_response = response_batch[0]
        if isinstance(first_response, dict):
            return first_response.get("aux_info")
    return None


def iter_aux_infos(response: Any) -> List[Dict[str, Any]]:
    aux_info = _response_aux_info(response)
    if isinstance(aux_info, dict):
        return [aux_info]
    if isinstance(aux_info, list):
        return [x for x in aux_info if isinstance(x, dict)]
    return []


def find_role_addr(response: Any, role: str) -> Optional[Dict[str, Any]]:
    expected = role.upper()
    for aux_info in iter_aux_infos(response):
        for addr in aux_info.get("role_addrs", []) or []:
            if isinstance(addr, dict) and _role_name(addr.get("role")) == expected:
                return addr
    data = plain_json(response)
    if isinstance(data, dict):
        for status in data.get("server_status", []) or []:
            if isinstance(status, dict) and _role_name(status.get("role")) == expected:
                return status
    return None


def max_reuse_len(response: Any) -> int:
    fields = (
        "reuse_len",
        "local_reuse_len",
        "memory_reuse_len",
        "prefill_total_reuse_len",
        "prefill_local_reuse_len",
        "prefill_memory_reuse_len",
    )
    max_reuse = 0
    for aux_info in iter_aux_infos(response):
        for field in fields:
            try:
                max_reuse = max(max_reuse, int(aux_info.get(field) or 0))
            except (TypeError, ValueError):
                continue
    return max_reuse


def decode_frontend_json_response(response: Any) -> Tuple[bool, Any]:
    if isinstance(response, list):
        response = list(filter(None, response))
        if not response:
            return False, "empty streaming response"
        last = response[-1]
        if isinstance(last, bytes):
            last = last.decode("utf-8")
        if isinstance(last, str) and last.startswith("data: "):
            last = last[6:]
        response = last
    try:
        return True, json.loads(response)
    except Exception as e:  # noqa: BLE001
        return False, f"failed to parse frontend response: {e}, raw={response}"


def visit_frontend_url_json(
    url: str,
    query: Dict[str, Any],
    retry_times: int,
    timeout: Optional[float],
) -> Tuple[bool, Any]:
    last_err: Any = None
    for _ in range(retry_times):
        try:
            logging.info("curl %s -d '%s'", url, json.dumps(query))
            response = requests.post(url, json=query, timeout=timeout)
            if response.status_code != 200:
                last_err = f"status={response.status_code} body={response.text}"
                logging.warning("frontend POST %s failed: %s", url, last_err)
                time.sleep(1)
                continue

            if response.headers.get("Transfer-Encoding", None) == "chunked":
                return decode_frontend_json_response([x for x in response.iter_lines()])
            return decode_frontend_json_response(response.text)
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            logging.warning("frontend POST %s error: %s", url, last_err)
    return False, last_err


def visit_frontend_url_json_process(
    idx: int,
    url: str,
    query: Dict[str, Any],
    retry_times: int,
    timeout: Optional[float],
    result_queue: Any,
) -> None:
    try:
        ok, response = visit_frontend_url_json(url, query, retry_times, timeout)
    except Exception as e:  # noqa: BLE001
        ok, response = False, f"frontend request {idx} raised: {e}"
    result_queue.put((idx, ok, response))


def tokenize_frontend_url(
    url: str,
    query: Dict[str, Any],
    timeout: Optional[float],
) -> Tuple[bool, Any]:
    try:
        logging.info("tokenize %s -d '%s'", url, json.dumps(query))
        response = requests.post(url, json=query, timeout=timeout)
        if response.status_code != 200:
            return False, f"status={response.status_code} body={response.text}"
        body = response.json()
        token_ids = body.get("token_ids") if isinstance(body, dict) else None
        if not isinstance(token_ids, list) or not all(
            isinstance(x, int) for x in token_ids
        ):
            return False, f"tokenize response missing integer token_ids: {body}"
        return True, token_ids
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def wait_for_flexlb_roles(
    flexlb_manager: Any,
    roles: List[str],
    role_name: str,
    timeout_seconds: float = 30.0,
) -> Tuple[bool, str]:
    deadline = time.time() + timeout_seconds
    last_result: Any = None
    while time.time() < deadline:
        request_id = int(time.time() * 1000000)
        query = {
            "request_id": request_id,
            "block_cache_keys": [
                request_id * 1000,
                request_id * 1000 + 1,
                request_id * 1000 + 2,
            ],
            "seq_len": 1,
            "generate_timeout": 1000,
            "request_time_ms": int(time.time() * 1000),
        }
        status_code, body = flexlb_manager.post_schedule_once(
            query,
            timeout_seconds=2.0,
        )
        last_result = (status_code, body)
        if (
            status_code == 200
            and isinstance(body, dict)
            and body.get("success") is True
            and all(find_role_addr(body, role) for role in roles)
        ):
            logging.info("flexlb synced %s roles: %s", role_name, body)
            return True, "ok"
        time.sleep(0.2)
    return (
        False,
        f"flexlb did not sync {role_name} roles {roles} within "
        f"{timeout_seconds}s, last_result={last_result}",
    )


def queue_snapshot_count(snapshot: Any) -> int:
    if not isinstance(snapshot, dict):
        return 0
    try:
        return int(snapshot.get("count") or 0)
    except (TypeError, ValueError):
        return 0


def role_hit_cache_len(response: Any, role: str) -> Optional[int]:
    addr = find_role_addr(response, role)
    if not isinstance(addr, dict):
        return None
    debug_info = addr.get("debug_info")
    if not isinstance(debug_info, dict):
        return None
    for key in ("hit_cache_len", "hitCacheLen"):
        value = debug_info.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def load_flexlb_config(flexlb_envs: Dict[str, str]) -> Dict[str, Any]:
    raw_config = flexlb_envs.get("FLEXLB_CONFIG") or os.environ.get("FLEXLB_CONFIG")
    if not raw_config:
        return {}
    try:
        parsed = json.loads(raw_config)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:  # noqa: BLE001
        logging.warning("failed to parse FLEXLB_CONFIG in smoke: %s", e)
        return {}


def _rewrite_prompt_for_probe(
    query: Dict[str, Any],
    suffix: str,
    prompt_repeat: int,
) -> None:
    prompt_repeat = max(1, prompt_repeat)
    if "messages" in query and isinstance(query["messages"], list):
        for message in reversed(query["messages"]):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = ((content + "\n") * prompt_repeat).strip()
                if suffix:
                    message["content"] += suffix
                return
    if isinstance(query.get("prompt"), str):
        query["prompt"] = ((query["prompt"] + "\n") * prompt_repeat).strip()
        if suffix:
            query["prompt"] += suffix


def make_fast_probe_qr(
    source_qr: Dict[str, Any],
    suffix: str = "",
    prompt_repeat: int = 1,
    max_tokens: Optional[int] = None,
    timeout_ms: Optional[int] = None,
) -> Dict[str, Any]:
    qr = copy.deepcopy(source_qr)
    query = qr["query"]
    if max_tokens is None:
        max_tokens = int(os.environ.get("FLEXLB_SMOKE_MAX_TOKENS", "1"))
    if timeout_ms is None:
        timeout_ms = int(
            os.environ.get(
                "FLEXLB_SMOKE_GENERATE_TIMEOUT_MS",
                FLEXLB_SMOKE_DEFAULT_TIMEOUT_MS,
            )
        )
    if "messages" in query:
        query["max_tokens"] = max_tokens
        query["temperature"] = 0.0
        query["top_p"] = 0
        query["top_k"] = 1
        query["stream"] = False
        query["aux_info"] = True
        extra_configs = query.get("extra_configs")
        if not isinstance(extra_configs, dict):
            extra_configs = {}
            query["extra_configs"] = extra_configs
        extra_configs["max_new_tokens"] = max_tokens
        extra_configs["temperature"] = 0.0
        extra_configs["top_p"] = 0
        extra_configs["top_k"] = 1
        extra_configs["timeout_ms"] = timeout_ms
        extra_configs["ttft_timeout_ms"] = timeout_ms
        extra_configs["aux_info"] = True
    else:
        query["yield_generator"] = False
        generate_config = query.setdefault("generate_config", {})
        generate_config["max_new_tokens"] = max_tokens
        generate_config["temperature"] = 0.0
        generate_config["top_p"] = 1.0
        generate_config["top_k"] = 1
        generate_config["timeout_ms"] = timeout_ms
        generate_config["ttft_timeout_ms"] = timeout_ms
        generate_config["aux_info"] = True
    _rewrite_prompt_for_probe(query, suffix, prompt_repeat)
    return qr
