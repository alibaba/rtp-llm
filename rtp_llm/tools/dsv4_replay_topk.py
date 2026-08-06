#!/opt/conda310/bin/python
"""Replay captured DSV4 requests and validate per-query MegaMoE top-k dumps."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import torch

MAIN_MODEL = "DeepSeekV4Model"
MTP_MODEL = "DeepSeekV4MtpModel"
TOP_K = 6
NUM_EXPERTS = 256
COLLECTION_GENERATION_TOKEN_CAP = 2_000
MTP_TOKENS_PER_DECODE_STEP = 4
MAX_DECODE_STEPS = 512
DUMP_QUIESCENCE_TIMEOUT_SECONDS = 30.0
DUMP_QUIESCENCE_STABLE_SECONDS = 2.0
DUMP_QUIESCENCE_POLL_SECONDS = 0.25


class NonRetryableRequestError(RuntimeError):
    """The local endpoint rejected an unchanged request deterministically."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--prefill-dir", type=Path, required=True)
    parser.add_argument("--decode-dir", type=Path, required=True)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        help="Recoverable destination for incomplete query dumps",
    )
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--case",
        help="Replay only this input filename, stem, or trace_id",
    )
    return parser.parse_args()


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "query"


def load_cases(input_dir: Path) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(input_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        request = data.get("request")
        if not isinstance(request, dict):
            raise ValueError(f"{path}: request is not an object")
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}: request.messages must be a non-empty list")
        request_parameters = request.get("parameters") or {}
        if not isinstance(request_parameters, dict):
            raise ValueError(f"{path}: request.parameters is not an object")
        backend_parameters = data.get("params")
        if not isinstance(backend_parameters, dict):
            raise ValueError(f"{path}: params is not an object")
        trace_id = str(data.get("trace_id") or path.stem)
        query_id = safe_component(trace_id)
        expected_tokens = int((data.get("usage") or {}).get("input_tokens") or 0)
        cases.append(
            {
                "path": path,
                "query_id": query_id,
                "trace_id": trace_id,
                "service": str(data.get("service") or ""),
                "messages": messages,
                "request_parameters": request_parameters,
                "backend_parameters": backend_parameters,
                "transport": request.get("transport") or {},
                "expected_tokens": expected_tokens,
            }
        )
    if len({case["query_id"] for case in cases}) != len(cases):
        raise ValueError("query ids are not unique")
    return cases


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_openai_payload(case: dict[str, Any]) -> dict[str, Any]:
    """Adapt the captured frontend request without inventing sampling values."""
    payload = copy.deepcopy(case["request_parameters"])
    payload["messages"] = copy.deepcopy(case["messages"])
    payload.setdefault("model", case["service"] or None)
    payload["trace_id"] = case["trace_id"]

    # ``params`` is the captured backend sampling configuration. Preserve its
    # supported values through OpenAI's ``extra_configs`` field; aliases below
    # only translate production field names to GenerateConfig field names.
    backend = case["backend_parameters"]
    aliases = {
        "eos_token_id": "eos_token_id",
        "max_new_tokens": "max_new_tokens",
        "max_new_think_tokens": "max_thinking_tokens",
        "min_length": "min_new_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "do_sample": "do_sample",
        "num_beams": "num_beams",
        "response_format": "response_format",
    }
    extra_configs = {
        target: backend[source]
        for source, target in aliases.items()
        if source in backend
    }
    captured_extra = payload.get("extra_configs")
    if captured_extra is not None:
        if not isinstance(captured_extra, dict):
            raise ValueError("request parameter extra_configs is not an object")
        extra_configs.update(captured_extra)
    if extra_configs:
        payload["extra_configs"] = extra_configs

    # These are top-level OpenAI request fields. Only fill absent fields from
    # the captured backend parameters; explicit captured request parameters win.
    top_level_aliases = {
        "logprobs": "logprobs",
        "n": "n",
        "seed": "seed",
        "temperature": "temperature",
        "top_logprobs": "top_logprobs",
        "top_p": "top_p",
    }
    for source, target in top_level_aliases.items():
        if source in backend:
            payload.setdefault(target, backend[source])
    if not any(
        key in payload
        for key in ("max_tokens", "max_completion_tokens", "max_new_tokens")
    ):
        payload["max_tokens"] = backend["max_new_tokens"]

    transport = case["transport"]
    if "stream" not in payload and isinstance(transport, dict):
        stream_mode = str(transport.get("streammode") or "NONE").upper()
        payload["stream"] = stream_mode != "NONE"
    return payload


def adapt_legacy_openai_schema(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Add only fields required by this endpoint's stricter OpenAI schema.

    Captured production gateways accepted text content parts without an
    explicit type and function tools without the two otherwise optional schema
    fields.  Adding these fields preserves the captured semantic content.  The
    caller records every addition and hashes both the before and after payloads.
    """
    adaptations: list[dict[str, Any]] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part_index, part in enumerate(content):
                if not isinstance(part, dict) or "type" in part:
                    continue
                base_path = f"/messages/{message_index}/content/{part_index}"
                legacy_field = next(
                    (field for field in ("text", "image", "video") if field in part),
                    None,
                )
                if legacy_field is None:
                    continue
                content_type = {
                    "text": "text",
                    "image": "image_url",
                    "video": "video_url",
                }[legacy_field]
                part["type"] = content_type
                adaptations.append(
                    {
                        "operation": "add",
                        "path": f"{base_path}/type",
                        "value": content_type,
                    }
                )
                if legacy_field in ("image", "video"):
                    standardized_field = f"{legacy_field}_url"
                    standardized_value = {"url": part[legacy_field]}
                    part[standardized_field] = standardized_value
                    adaptations.append(
                        {
                            "operation": "add",
                            "path": f"{base_path}/{standardized_field}",
                            "source_path": f"{base_path}/{legacy_field}",
                            "value_sha256": _json_sha256(standardized_value),
                        }
                    )

            # The endpoint schema accepts ContentPart objects, but the current
            # DeepSeek-V4 text encoding module still concatenates message
            # content as a string.  Project ordered text/media values to that
            # string without changing, dropping, or synthesizing any value.
            projected_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError(
                        f"message {message_index}: non-object content part {part!r}"
                    )
                if "text" in part:
                    value = part["text"]
                elif "image" in part:
                    value = part["image"]
                elif "video" in part:
                    value = part["video"]
                elif isinstance(part.get("image_url"), dict):
                    value = part["image_url"].get("url")
                elif isinstance(part.get("video_url"), dict):
                    value = part["video_url"].get("url")
                else:
                    raise ValueError(
                        f"message {message_index}: unsupported content part {part!r}"
                    )
                if not isinstance(value, str):
                    raise ValueError(
                        f"message {message_index}: content value is not a string"
                    )
                projected_parts.append(value)
            projected_content = "".join(projected_parts)
            before_sha256 = _json_sha256(content)
            message["content"] = projected_content
            adaptations.append(
                {
                    "after_sha256": _json_sha256(projected_content),
                    "before_sha256": before_sha256,
                    "operation": "replace",
                    "path": f"/messages/{message_index}/content",
                    "projection": "ordered_text_and_media_url_concatenation",
                    "source_part_count": len(content),
                }
            )

    tools = payload.get("tools")
    if isinstance(tools, list):
        for tool_index, tool in enumerate(tools):
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict):
                continue
            if "description" not in function:
                function["description"] = ""
                adaptations.append(
                    {
                        "operation": "add",
                        "path": f"/tools/{tool_index}/function/description",
                        "value": "",
                    }
                )
            if "parameters" not in function:
                empty_parameters = {"type": "object", "properties": {}}
                function["parameters"] = empty_parameters
                adaptations.append(
                    {
                        "operation": "add",
                        "path": f"/tools/{tool_index}/function/parameters",
                        "value": empty_parameters,
                    }
                )
    extra_configs = payload.get("extra_configs")
    if (
        isinstance(extra_configs, dict)
        and isinstance(extra_configs.get("max_thinking_tokens"), int)
        and extra_configs["max_thinking_tokens"] < 0
    ):
        original = int(extra_configs.pop("max_thinking_tokens"))
        adaptations.append(
            {
                "operation": "remove",
                "original_value": original,
                "path": "/extra_configs/max_thinking_tokens",
                "production_semantics": (
                    "negative max_new_think_tokens means no explicit thinking limit"
                ),
            }
        )
    return adaptations


def post_real_request(
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    with session.post(
        url,
        json=payload,
        timeout=timeout,
        stream=bool(payload.get("stream")),
    ) as response:
        if response.status_code != 200:
            response_text = response.text
            message = f"HTTP {response.status_code}: {response_text[:1000]}"
            if 400 <= response.status_code < 500 and response.status_code not in (
                408,
                429,
            ):
                raise NonRetryableRequestError(message)
            if response.status_code == 500 and (
                "Error rendering" in response_text
                or "507_ERROR_INPUT_FORMAT_ERROR" in response_text
            ):
                raise NonRetryableRequestError(message)
            raise RuntimeError(message)
        if not payload.get("stream"):
            result = response.json()
            if result.get("error_code"):
                raise RuntimeError(str(result))
            return result

        last_chunk: dict[str, Any] = {}
        usage: dict[str, Any] | None = None
        aux_info: Any = None
        chunks = 0
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if not isinstance(chunk, dict):
                raise ValueError(f"stream chunk is not an object: {chunk!r}")
            chunks += 1
            last_chunk = chunk
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            if chunk.get("aux_info") is not None:
                aux_info = chunk["aux_info"]
        if not chunks:
            raise ValueError("stream response contained no JSON chunks")
        if usage is not None:
            last_chunk["usage"] = usage
        if aux_info is not None:
            last_chunk["aux_info"] = aux_info
        return last_chunk


def status_state(status_file: Path) -> tuple[set[str], set[str]]:
    completed: set[str] = set()
    latest: dict[str, str] = {}
    if not status_file.exists():
        return completed, set()
    for line_number, line in enumerate(
        status_file.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            print(
                f"warning: ignoring malformed status line {line_number}: {error}",
                flush=True,
            )
            continue
        query_id = str(record.get("query_id") or "")
        if not query_id or query_id == "startup":
            continue
        latest[query_id] = str(record.get("status") or "")
        if record.get("status") == "completed":
            completed.add(query_id)
    failed = {
        query_id
        for query_id, status in latest.items()
        if status == "failed" and query_id not in completed
    }
    return completed, failed


def write_active_query(query_file: Path, query_id: str) -> None:
    query_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = query_file.with_name(f".{query_file.name}.{os.getpid()}.tmp")
    tmp.write_text(query_id, encoding="utf-8")
    os.replace(tmp, query_file)


def append_status(status_file: Path, record: dict[str, Any]) -> None:
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with status_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def model_layers() -> list[tuple[str, int]]:
    return [(MAIN_MODEL, layer) for layer in range(43)] + [(MTP_MODEL, 0)]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_check(path: Path, *, expected_rows: int | None) -> dict[str, Any]:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError(f"{path}: expected a 2D tensor")
    rows, columns = map(int, tensor.shape)
    if columns != TOP_K:
        raise ValueError(f"{path}: expected top-k width {TOP_K}, got {columns}")
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} token rows, got {rows}")
    if rows <= 0:
        raise ValueError(f"{path}: empty topk_idx tensor")
    if tensor.dtype != torch.int64:
        raise ValueError(f"{path}: expected torch.int64, got {tensor.dtype}")
    minimum = int(tensor.min().item())
    maximum = int(tensor.max().item())
    if minimum < 0 or maximum >= NUM_EXPERTS:
        raise ValueError(
            f"{path}: expert index range [{minimum}, {maximum}] is outside "
            f"[0, {NUM_EXPERTS - 1}]"
        )
    ordered = tensor.sort(dim=1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any().item()):
        raise ValueError(f"{path}: duplicate expert ids found within a top-k row")
    return {
        "columns": columns,
        "dtype": str(tensor.dtype),
        "maximum": maximum,
        "minimum": minimum,
        "rows": rows,
        "sha256": file_sha256(path),
    }


def _manifest_sha256(
    root: Path, paths: list[Path], *, expected_rows: int | None
) -> str:
    digest = hashlib.sha256()
    for path in paths:
        metadata = load_and_check(path, expected_rows=expected_rows)
        entry = {
            "metadata": metadata,
            "path": path.relative_to(root).as_posix(),
        }
        digest.update(
            json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def validate_prefill(root: Path, query_id: str, input_len: int) -> tuple[int, str]:
    query_dir = root / f"query_{query_id}"
    expected = []
    for model, layer in model_layers():
        expected.append(
            query_dir / "step_000" / model / f"layer_{layer:03d}" / "all_tokens.pt"
        )
    actual = sorted(query_dir.rglob("*.pt")) if query_dir.exists() else []
    if set(actual) != set(expected):
        missing = sorted(str(path) for path in set(expected) - set(actual))
        extra = sorted(str(path) for path in set(actual) - set(expected))
        raise ValueError(
            f"prefill {query_id}: expected 44 files, got {len(actual)}; "
            f"missing={missing[:3]} extra={extra[:3]}"
        )
    manifest_sha256 = _manifest_sha256(query_dir, expected, expected_rows=input_len)
    return len(actual), manifest_sha256


def decode_step_numbers(
    root: Path, query_id: str, model: str = MAIN_MODEL
) -> list[int]:
    """Return the contiguous per-model decode counter values.

    The base model and MTP model maintain independent dump counters.  With
    ``gen_num_per_cycle=3`` the MTP model can therefore have three times as
    many step directories as the base model until either counter reaches the
    configured dump cap.
    """
    query_dir = root / f"query_{query_id}"
    steps: list[int] = []
    if query_dir.exists():
        for path in query_dir.glob("step_*"):
            if not path.is_dir():
                continue
            match = re.fullmatch(r"step_(\d+)", path.name)
            if match is None:
                raise ValueError(f"decode {query_id}: invalid step directory {path}")
            if (path / model).is_dir():
                steps.append(int(match.group(1)))
    steps.sort()
    if steps != list(range(len(steps))):
        raise ValueError(f"decode {query_id}: non-contiguous steps {steps[:20]}")
    if len(steps) > MAX_DECODE_STEPS:
        raise ValueError(
            f"decode {query_id}: expected at most {MAX_DECODE_STEPS} steps, "
            f"got {len(steps)}"
        )
    return steps


def validate_decode(
    root: Path, query_id: str
) -> tuple[int, int | None, str, dict[str, Any]]:
    query_dir = root / f"query_{query_id}"
    actual = sorted(query_dir.rglob("*.pt")) if query_dir.exists() else []
    main_steps = decode_step_numbers(root, query_id, MAIN_MODEL)
    mtp_steps = decode_step_numbers(root, query_id, MTP_MODEL)
    if actual and (not main_steps or not mtp_steps):
        raise ValueError(
            f"decode {query_id}: incomplete model coverage "
            f"main_steps={len(main_steps)} mtp_steps={len(mtp_steps)}"
        )
    expected_layer_dirs = []
    for step in main_steps:
        expected_layer_dirs.extend(
            query_dir / f"step_{step:03d}" / MAIN_MODEL / f"layer_{layer:03d}"
            for layer in range(43)
        )
    for step in mtp_steps:
        expected_layer_dirs.append(
            query_dir / f"step_{step:03d}" / MTP_MODEL / "layer_000"
        )
    if len(actual) != len(expected_layer_dirs):
        raise ValueError(
            f"decode {query_id}: expected {len(expected_layer_dirs)} files "
            f"(43 x {len(main_steps)} base steps + "
            f"1 x {len(mtp_steps)} MTP steps), got {len(actual)}"
        )
    ranks: set[int] = set()
    expected_files: set[Path] = set()
    for layer_dir in expected_layer_dirs:
        files = list(layer_dir.glob("rank_*.pt")) if layer_dir.exists() else []
        if len(files) != 1:
            raise ValueError(
                f"{layer_dir}: expected one real DP-rank file, got {files}"
            )
        match = re.fullmatch(r"rank_(\d+)\.pt", files[0].name)
        if match is None:
            raise ValueError(f"{files[0]}: invalid decode rank filename")
        ranks.add(int(match.group(1)))
        expected_files.add(files[0])
    if set(actual) != expected_files:
        raise ValueError(f"decode {query_id}: unexpected topk_idx files found")
    if actual and len(ranks) != 1:
        raise ValueError(
            f"decode {query_id}: real query is incomplete across DP ranks {sorted(ranks)}"
        )
    manifest_sha256 = _manifest_sha256(
        query_dir, sorted(expected_files), expected_rows=None
    )
    layout = {
        "decode_step_count": len(main_steps),
        "decode_steps": main_steps,
        "decode_mtp_step_count": len(mtp_steps),
        "decode_mtp_steps": mtp_steps,
    }
    return len(actual), next(iter(ranks)) if ranks else None, manifest_sha256, layout


def response_input_len(response: dict[str, Any]) -> int:
    usage = response.get("usage")
    if isinstance(usage, dict):
        value = usage.get("prompt_tokens", usage.get("input_tokens"))
        if value is not None:
            return int(value)
    aux = response.get("aux_info")
    if isinstance(aux, list):
        if len(aux) != 1:
            raise ValueError(f"expected one aux_info entry, got {len(aux)}")
        aux = aux[0]
    if not isinstance(aux, dict) or not aux.get("input_len"):
        raise ValueError(f"response missing aux_info.input_len: {response}")
    return int(aux["input_len"])


def response_output_len(response: dict[str, Any]) -> int | None:
    usage = response.get("usage")
    if isinstance(usage, dict):
        value = usage.get("completion_tokens", usage.get("output_tokens"))
        if value is not None:
            return int(value)
    aux = response.get("aux_info")
    if isinstance(aux, list) and len(aux) == 1:
        aux = aux[0]
    if isinstance(aux, dict) and aux.get("output_len") is not None:
        return int(aux["output_len"])
    return None


def query_dump_counts(
    prefill_root: Path, decode_root: Path, query_id: str
) -> tuple[int, int]:
    name = f"query_{query_id}"
    prefill_dir = prefill_root / name
    decode_dir = decode_root / name
    return (
        sum(1 for _ in prefill_dir.rglob("*.pt")) if prefill_dir.exists() else 0,
        sum(1 for _ in decode_dir.rglob("*.pt")) if decode_dir.exists() else 0,
    )


def query_dump_activity_signature(
    prefill_root: Path, decode_root: Path, query_id: str
) -> tuple[int, int, int, int]:
    """Return a cheap signature that changes while async dump writes continue."""
    file_counts = []
    entry_count = 0
    newest_mtime_ns = 0
    name = f"query_{query_id}"
    for root in (prefill_root / name, decode_root / name):
        file_count = 0
        if root.exists():
            for current, directories, files in os.walk(root):
                current_path = Path(current)
                entry_count += 1 + len(directories) + len(files)
                try:
                    newest_mtime_ns = max(
                        newest_mtime_ns, current_path.stat().st_mtime_ns
                    )
                except FileNotFoundError:
                    pass
                for filename in files:
                    path = current_path / filename
                    if path.suffix == ".pt":
                        file_count += 1
                    try:
                        newest_mtime_ns = max(newest_mtime_ns, path.stat().st_mtime_ns)
                    except FileNotFoundError:
                        pass
        file_counts.append(file_count)
    return file_counts[0], file_counts[1], entry_count, newest_mtime_ns


def wait_for_query_dump_quiescence(
    prefill_root: Path,
    decode_root: Path,
    query_id: str,
    *,
    timeout_seconds: float = DUMP_QUIESCENCE_TIMEOUT_SECONDS,
    stable_seconds: float = DUMP_QUIESCENCE_STABLE_SECONDS,
    poll_seconds: float = DUMP_QUIESCENCE_POLL_SECONDS,
) -> tuple[int, int, int, int]:
    """Wait until the per-query dump tree stops changing before validation.

    The HTTP response can finish slightly before asynchronous tensor writes do.
    Validating immediately can observe a newly created step with only some of
    its layer files and then incorrectly retry the whole request.  Requiring a
    stable file/directory/mtime signature avoids mixing multiple attempts in a
    single query directory.
    """
    deadline = time.monotonic() + timeout_seconds
    previous: tuple[int, int, int, int] | None = None
    stable_since: float | None = None
    while True:
        now = time.monotonic()
        signature = query_dump_activity_signature(prefill_root, decode_root, query_id)
        if signature == previous:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= stable_seconds:
                return signature
        else:
            previous = signature
            stable_since = now
        if now >= deadline:
            return signature
        time.sleep(poll_seconds)


def completed_outputs_present(
    prefill_root: Path, decode_root: Path, query_id: str
) -> bool:
    prefill_count, decode_count = query_dump_counts(prefill_root, decode_root, query_id)
    if prefill_count != 44:
        return False
    if decode_count > MAX_DECODE_STEPS * 44:
        return False
    try:
        validate_decode(decode_root, query_id)
    except Exception:
        return False
    return True


def validate_complete_query_dump(
    prefill_root: Path,
    decode_root: Path,
    query_id: str,
    input_len: int | None,
) -> dict[str, Any] | None:
    """Strictly validate a capped variable-step dump after a transport error."""
    prefill_count, decode_count = query_dump_counts(prefill_root, decode_root, query_id)
    if prefill_count != 44 or decode_count <= 0 or decode_count > MAX_DECODE_STEPS * 44:
        return None
    if input_len is None:
        probe = (
            prefill_root
            / f"query_{query_id}"
            / "step_000"
            / MAIN_MODEL
            / "layer_000"
            / "all_tokens.pt"
        )
        input_len = int(load_and_check(probe, expected_rows=None)["rows"])
    prefill_files, prefill_sha256 = validate_prefill(prefill_root, query_id, input_len)
    decode_files, decode_rank, decode_sha256, decode_layout = validate_decode(
        decode_root, query_id
    )
    dump_manifest_sha256 = hashlib.sha256(
        (f"prefill:{prefill_sha256}\n" f"decode:{decode_sha256}\n").encode("ascii")
    ).hexdigest()
    return {
        "decode_files": decode_files,
        "decode_rank": decode_rank,
        "decode_sha256": decode_sha256,
        **decode_layout,
        "dump_manifest_sha256": dump_manifest_sha256,
        "input_len": input_len,
        "prefill_files": prefill_files,
        "prefill_sha256": prefill_sha256,
    }


def restore_quarantined_query(
    quarantine_root: Path,
    prefill_root: Path,
    decode_root: Path,
    query_id: str,
) -> str | None:
    """Restore the newest partial dump so live per-rank counters can continue."""
    target_name = f"query_{query_id}"
    targets = {
        "prefill": prefill_root / target_name,
        "decode": decode_root / target_name,
    }
    if any(path.exists() for path in targets.values()):
        return None

    candidates: list[Path] = []
    legacy = quarantine_root / query_id
    if legacy.is_dir():
        candidates.append(legacy)
    query_root = quarantine_root / target_name
    if query_root.is_dir():
        candidates.extend(path for path in query_root.iterdir() if path.is_dir())
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for candidate in candidates:
        if not any((candidate / role).is_dir() for role in targets):
            continue
        for role, target in targets.items():
            source = candidate / role
            if source.is_dir():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
        return str(candidate)
    return None


def quarantine_query(
    quarantine_root: Path,
    prefill_root: Path,
    decode_root: Path,
    query_id: str,
    metadata: dict[str, Any],
) -> str:
    target_name = f"query_{query_id}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = quarantine_root / target_name / f"run_{timestamp}_{time.time_ns()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for role, root in (("prefill", prefill_root), ("decode", decode_root)):
        source = root / target_name
        if source.exists():
            os.replace(source, run_dir / role)
    metadata_path = run_dir / "failure.json"
    tmp = metadata_path.with_suffix(f".pid{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, metadata_path)
    return str(run_dir)


def request_evidence(
    case: dict[str, Any],
    payload: dict[str, Any],
    *,
    pre_schema_payload_sha256: str,
    schema_adaptations: list[dict[str, Any]],
) -> dict[str, Any]:
    request_parameters = case["request_parameters"]
    transport = case["transport"]
    if "stream" in request_parameters:
        stream_source = "request.parameters.stream"
    else:
        stream_source = "request.transport.streammode"
    return {
        "adapted_payload_sha256": _json_sha256(payload),
        "backend_parameters_sha256": _json_sha256(case["backend_parameters"]),
        "input_file": case["path"].name,
        "input_file_sha256": file_sha256(case["path"]),
        "messages_sha256": _json_sha256(case["messages"]),
        "pre_schema_payload_sha256": pre_schema_payload_sha256,
        "request_parameters_sha256": _json_sha256(request_parameters),
        "schema_adaptation_count": len(schema_adaptations),
        "schema_adaptations": schema_adaptations,
        "stream": bool(payload.get("stream")),
        "stream_source": stream_source,
        "transport_sha256": _json_sha256(transport),
    }


def _generation_limit_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    extra_configs = payload.get("extra_configs")
    if extra_configs is not None and not isinstance(extra_configs, dict):
        raise ValueError("generation-limit extra_configs is not an object")
    extra_configs = extra_configs or {}
    return {
        "max_completion_tokens": payload.get("max_completion_tokens"),
        "max_new_tokens": payload.get("max_new_tokens"),
        "max_tokens": payload.get("max_tokens"),
        "extra_configs.max_new_tokens": extra_configs.get("max_new_tokens"),
        "extra_configs.min_new_tokens": extra_configs.get("min_new_tokens"),
    }


def requested_max_new_tokens(case: dict[str, Any]) -> int:
    """Return the captured request's effective positive generation limit."""
    request_parameters = case["request_parameters"]
    backend_parameters = case["backend_parameters"]
    value: Any = None
    source = ""
    for key in ("max_completion_tokens", "max_tokens", "max_new_tokens"):
        candidate = request_parameters.get(key)
        if isinstance(candidate, int):
            value = candidate
            source = f"request.parameters.{key}"
            break
    if value is None:
        value = backend_parameters.get("max_new_tokens")
        source = "params.max_new_tokens"
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{source or 'generation limit'} must be a positive integer")
    return int(value)


def normalize_generation_limits(
    adapted_payload: dict[str, Any], case: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cap the captured max-new-token value at 2,000 without raising its floor."""
    normalized = copy.deepcopy(adapted_payload)
    original_transport_limits = _generation_limit_snapshot(adapted_payload)
    extra_configs = normalized.setdefault("extra_configs", {})
    if not isinstance(extra_configs, dict):
        raise ValueError("generation normalization extra_configs is not an object")

    original_generation_tokens = requested_max_new_tokens(case)
    transport_generation_tokens = min(
        original_generation_tokens, COLLECTION_GENERATION_TOKEN_CAP
    )
    normalized["max_tokens"] = transport_generation_tokens
    normalized["max_completion_tokens"] = transport_generation_tokens
    if "max_new_tokens" in normalized:
        normalized["max_new_tokens"] = transport_generation_tokens
    extra_configs["max_new_tokens"] = transport_generation_tokens
    captured_minimum = extra_configs.get("min_new_tokens")
    if (
        isinstance(captured_minimum, int)
        and captured_minimum > transport_generation_tokens
    ):
        extra_configs["min_new_tokens"] = transport_generation_tokens
    normalized_transport_limits = _generation_limit_snapshot(normalized)

    changes: list[dict[str, Any]] = []
    paths = {
        "max_tokens": "/max_tokens",
        "max_completion_tokens": "/max_completion_tokens",
        "max_new_tokens": "/max_new_tokens",
        "extra_configs.max_new_tokens": "/extra_configs/max_new_tokens",
        "extra_configs.min_new_tokens": "/extra_configs/min_new_tokens",
    }
    for field, path in paths.items():
        before = original_transport_limits[field]
        after = normalized_transport_limits[field]
        if before != after:
            changes.append(
                {
                    "new_value": after,
                    "operation": "add" if before is None else "replace",
                    "original_value": before,
                    "path": path,
                }
            )

    request_parameters = case["request_parameters"]
    backend_parameters = case["backend_parameters"]
    audit = {
        "changes": changes,
        "gen_num_per_cycle": 3,
        "mtp_tokens_per_decode_step": MTP_TOKENS_PER_DECODE_STEP,
        "max_decode_steps": MAX_DECODE_STEPS,
        "normalized_transport_limits": normalized_transport_limits,
        "original_generation_tokens": original_generation_tokens,
        "original_generation_limits": {
            "backend_parameters": {
                key: backend_parameters.get(key)
                for key in ("max_new_tokens", "min_length")
            },
            "request_parameters": {
                key: request_parameters.get(key)
                for key in (
                    "max_tokens",
                    "max_completion_tokens",
                    "max_new_tokens",
                )
            },
            "transport_before_normalization": original_transport_limits,
        },
        "reason": "cap_real_max_new_tokens_at_2000",
        "transport_generation_tokens": transport_generation_tokens,
        "validation_basis": (
            "keep the captured positive max-new-token value when it is at most "
            "2000; otherwise cap it at 2000; preserve the captured minimum"
        ),
    }
    return normalized, audit


def format_progress(
    *,
    total: int,
    completed: set[str],
    failed: set[str],
    newly_completed: int,
    started: float,
) -> str:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate_per_hour = newly_completed * 3600.0 / elapsed
    pending = total - len(completed) - len(failed)
    remaining = total - len(completed)
    if rate_per_hour > 0:
        eta_seconds = remaining * 3600.0 / rate_per_hour
        eta = f"{eta_seconds / 3600.0:.2f}h"
    else:
        eta = "unknown"
    return (
        f"completed={len(completed)}/{total} failed={len(failed)} "
        f"pending={pending} throughput={rate_per_hour:.2f}q/h ETA={eta}"
    )


def main() -> None:
    args = parse_args()
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")
    quarantine_dir = args.quarantine_dir or (
        args.status_file.parent / "incomplete_dumps"
    )
    cases = load_cases(args.input_dir)
    all_query_ids = {case["query_id"] for case in cases}
    total = len(cases)
    if args.case:
        case_name = args.case
        cases = [
            case
            for case in cases
            if case["path"].name == case_name
            or case["path"].stem == case_name
            or case["trace_id"] == case_name
        ]
        if not cases:
            raise ValueError(f"case not found in {args.input_dir}: {case_name}")
    if args.limit is not None:
        cases = cases[: args.limit]
    completed, failed = status_state(args.status_file)
    completed &= all_query_ids
    failed &= all_query_ids
    invalid_completed = {
        query_id
        for query_id in completed
        if not completed_outputs_present(args.prefill_dir, args.decode_dir, query_id)
    }
    for query_id in sorted(invalid_completed):
        completed.remove(query_id)
        failed.add(query_id)
        append_status(
            args.status_file,
            {
                "error": "completed ledger record has incomplete output files",
                "query_id": query_id,
                "status": "failed",
            },
        )
    pending = [case for case in cases if case["query_id"] not in completed]
    print(
        f"loaded={total} selected={len(cases)} "
        f"selected_pending={len(pending)} "
        f"completed={len(completed)} failed={len(failed)} "
        f"pending={total - len(completed) - len(failed)}",
        flush=True,
    )

    session = requests.Session()
    # This tool replays directly to a local smoke server.  Do not let a
    # workstation HTTP_PROXY redirect 127.0.0.1 requests away from that server.
    session.trust_env = False
    url = args.server_url.rstrip("/") + "/v1/chat/completions"
    run_started = time.monotonic()
    newly_completed = 0
    for index, case in enumerate(pending, 1):
        query_id = case["query_id"]
        failed.discard(query_id)
        adapted_payload = build_openai_payload(case)
        pre_schema_payload_sha256 = _json_sha256(adapted_payload)
        schema_adaptations = adapt_legacy_openai_schema(adapted_payload)
        evidence = request_evidence(
            case,
            adapted_payload,
            pre_schema_payload_sha256=pre_schema_payload_sha256,
            schema_adaptations=schema_adaptations,
        )
        payload, generation_normalization = normalize_generation_limits(
            adapted_payload, case
        )
        payload_sha256 = _json_sha256(payload)
        evidence.update(
            {
                "generation_normalization": generation_normalization,
                "generation_normalized_payload_sha256": payload_sha256,
            }
        )
        restored_from = restore_quarantined_query(
            quarantine_dir,
            args.prefill_dir,
            args.decode_dir,
            query_id,
        )
        write_active_query(args.query_file, query_id)
        query_started = time.monotonic()
        attempt_errors: list[str] = []
        input_len: int | None = None
        output_len: int | None = None
        prefill_result: tuple[int, str] | None = None
        completed_record: dict[str, Any] | None = None
        primary_attempts = 0
        try:
            for attempt in range(1, args.max_attempts + 1):
                primary_attempts = attempt
                if _json_sha256(payload) != payload_sha256:
                    raise RuntimeError("request payload mutated between retries")
                try:
                    response_json = post_real_request(
                        session, url, payload, timeout=args.timeout
                    )
                    attempt_input_len = response_input_len(response_json)
                    output_len = response_output_len(response_json)
                    generation_limit = int(
                        generation_normalization["transport_generation_tokens"]
                    )
                    if output_len is None:
                        raise ValueError("response is missing output token count")
                    if output_len > generation_limit:
                        raise ValueError(
                            f"output_len {output_len} exceeds capped generation "
                            f"limit {generation_limit}"
                        )
                    if input_len is not None and attempt_input_len != input_len:
                        raise ValueError(
                            f"identical request changed input_len from {input_len} "
                            f"to {attempt_input_len}"
                        )
                    input_len = attempt_input_len
                    wait_for_query_dump_quiescence(
                        args.prefill_dir, args.decode_dir, query_id
                    )
                    if prefill_result is None:
                        prefill_result = validate_prefill(
                            args.prefill_dir, query_id, input_len
                        )
                    (
                        decode_files,
                        decode_rank,
                        decode_sha256,
                        decode_layout,
                    ) = validate_decode(args.decode_dir, query_id)
                    decode_steps = decode_layout["decode_steps"]
                    if output_len > 1 and not decode_steps:
                        raise ValueError(
                            f"decode {query_id}: output_len={output_len} but no "
                            "decode steps were dumped"
                        )
                    prefill_files, prefill_sha256 = prefill_result
                    dump_manifest_sha256 = hashlib.sha256(
                        (
                            f"prefill:{prefill_sha256}\n" f"decode:{decode_sha256}\n"
                        ).encode("ascii")
                    ).hexdigest()
                    elapsed = time.monotonic() - query_started
                    completed_record = {
                        **evidence,
                        "attempts": attempt,
                        "collection_mode": "transport_generation_normalized",
                        "collection_phase": "normalized_transport",
                        "completion_basis": "validated_response_and_dump",
                        "decode_files": decode_files,
                        "decode_rank": decode_rank,
                        "decode_sha256": decode_sha256,
                        **decode_layout,
                        "dump_manifest_sha256": dump_manifest_sha256,
                        "elapsed_seconds": round(elapsed, 3),
                        "historical_input_len": int(case["expected_tokens"]),
                        "input_len": input_len,
                        "output_len": output_len,
                        "primary_attempts": attempt,
                        "primary_output_len": output_len,
                        "primary_payload_sha256": evidence["adapted_payload_sha256"],
                        "prefill_files": prefill_files,
                        "prefill_sha256": prefill_sha256,
                        "query_id": query_id,
                        "restored_from": restored_from,
                        "service": case["service"],
                        "status": "completed",
                    }
                    break
                except Exception as error:
                    prefill_count, decode_count = query_dump_counts(
                        args.prefill_dir, args.decode_dir, query_id
                    )
                    if prefill_count or decode_count:
                        wait_for_query_dump_quiescence(
                            args.prefill_dir, args.decode_dir, query_id
                        )
                        prefill_count, decode_count = query_dump_counts(
                            args.prefill_dir, args.decode_dir, query_id
                        )
                    error_text = repr(error)
                    attempt_errors.append(error_text)
                    dump_validation_error: str | None = None
                    validated_dump: dict[str, Any] | None = None
                    if (
                        prefill_count == 44
                        and 0 < decode_count <= MAX_DECODE_STEPS * 44
                    ):
                        try:
                            validated_dump = validate_complete_query_dump(
                                args.prefill_dir,
                                args.decode_dir,
                                query_id,
                                input_len,
                            )
                        except Exception as validation_error:
                            dump_validation_error = repr(validation_error)
                    if validated_dump is not None:
                        input_len = int(validated_dump["input_len"])
                        elapsed = time.monotonic() - query_started
                        completed_record = {
                            **evidence,
                            **validated_dump,
                            "attempts": attempt,
                            "collection_mode": ("transport_generation_normalized"),
                            "collection_phase": "normalized_transport",
                            "completion_basis": (
                                "validated_dump_after_transport_error"
                            ),
                            "elapsed_seconds": round(elapsed, 3),
                            "historical_input_len": int(case["expected_tokens"]),
                            "output_len": output_len,
                            "output_source": (
                                "transport_error_response_not_used_for_completion"
                            ),
                            "primary_attempts": attempt,
                            "primary_output_len": output_len,
                            "primary_payload_sha256": evidence[
                                "adapted_payload_sha256"
                            ],
                            "query_id": query_id,
                            "restored_from": restored_from,
                            "service": case["service"],
                            "status": "completed",
                            "transport_errors": list(attempt_errors),
                        }
                        break
                    can_retry = attempt < args.max_attempts and not isinstance(
                        error, NonRetryableRequestError
                    )
                    append_status(
                        args.status_file,
                        {
                            **evidence,
                            "attempt": attempt,
                            "collection_phase": "normalized_transport",
                            "decode_files_observed": decode_count,
                            "dump_validation_error": dump_validation_error,
                            "error": error_text,
                            "input_len": input_len,
                            "output_len": output_len,
                            "prefill_files_observed": prefill_count,
                            "query_id": query_id,
                            "retrying": can_retry,
                            "status": "retrying",
                        },
                    )
                    print(
                        f"[{index}/{len(pending)}] query={query_id} "
                        f"attempt={attempt}/{args.max_attempts} failed "
                        f"prefill={prefill_count}/44 decode={decode_count} "
                        f"(max={MAX_DECODE_STEPS * 44}) "
                        f"error={error_text}",
                        flush=True,
                    )
                    if can_retry:
                        time.sleep(args.retry_delay)
                    else:
                        break
        finally:
            try:
                args.query_file.unlink()
            except FileNotFoundError:
                pass

        if completed_record is not None:
            append_status(args.status_file, completed_record)
            completed.add(query_id)
            newly_completed += 1
            failed.discard(query_id)
            print(
                f"[{index}/{len(pending)}] query={query_id} "
                f"input_len={completed_record['input_len']} "
                f"output_len={completed_record['output_len']} "
                f"prefill=44 decode={completed_record['decode_files']} "
                f"decode_steps={completed_record['decode_step_count']} "
                f"dp_rank={completed_record['decode_rank']} "
                f"attempts={completed_record['attempts']} "
                f"elapsed={completed_record['elapsed_seconds']:.1f}s",
                flush=True,
            )
        else:
            failure_metadata = {
                **evidence,
                "attempt_errors": attempt_errors,
                "query_id": query_id,
            }
            quarantine_path = quarantine_query(
                quarantine_dir,
                args.prefill_dir,
                args.decode_dir,
                query_id,
                failure_metadata,
            )
            failed.add(query_id)
            append_status(
                args.status_file,
                {
                    **failure_metadata,
                    "error": attempt_errors[-1] if attempt_errors else "unknown",
                    "quarantine_path": quarantine_path,
                    "status": "failed",
                },
            )
            print(
                f"[{index}/{len(pending)}] query={query_id} failed permanently "
                f"for this pass; quarantined={quarantine_path}",
                flush=True,
            )

        if index % args.progress_interval == 0 or index == len(pending):
            print(
                "progress "
                + format_progress(
                    total=total,
                    completed=completed,
                    failed=failed,
                    newly_completed=newly_completed,
                    started=run_started,
                ),
                flush=True,
            )

    print(
        "finished-pass "
        + format_progress(
            total=total,
            completed=completed,
            failed=failed,
            newly_completed=newly_completed,
            started=run_started,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
