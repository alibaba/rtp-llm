"""Bounded, opt-in prefill batch tracing for online precision debugging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict

import torch

_LOCK = threading.Lock()
_BATCH_SEQ = 0
_HIDDEN_FILE_COUNT = 0
_WARNED_MISSING_DIR = False


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "off", "no")


def _env_nonnegative_int(name: str, default: int) -> int:
    try:
        return max(int(os.environ.get(name, str(default))), 0)
    except ValueError:
        logging.warning("[PREFILL_TRACE] invalid %s; using %d", name, default)
        return default


def _cpu_i64(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().reshape(-1).to(device="cpu", dtype=torch.int64)


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _write_hidden_file(trace_dir: Path, filename: str, payload: Dict[str, Any]) -> None:
    destination = trace_dir / filename
    temporary = trace_dir / f".{filename}.tmp.{os.getpid()}.{time.time_ns()}"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def record_prefill(
    hidden_states: torch.Tensor,
    request_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    prefix_lengths: torch.Tensor,
    hidden_is_last_rows: bool,
    model_id: int,
    tp_rank: int,
) -> None:
    """Record one prefill execution batch; C++ catches all failures."""
    global _BATCH_SEQ, _HIDDEN_FILE_COUNT, _WARNED_MISSING_DIR

    if not _env_bool("RTP_LLM_PREFILL_TRACE", False) or int(tp_rank) != 0:
        return
    trace_dir_value = os.environ.get("RTP_LLM_PREFILL_TRACE_DIR", "").strip()
    if not trace_dir_value:
        if not _WARNED_MISSING_DIR:
            logging.warning(
                "[PREFILL_TRACE] RTP_LLM_PREFILL_TRACE_DIR is empty; tracing disabled"
            )
            _WARNED_MISSING_DIR = True
        return

    request_ids_cpu = _cpu_i64(request_ids)
    executed = _cpu_i64(input_lengths)
    reused = _cpu_i64(prefix_lengths)
    batch_size = int(request_ids_cpu.numel())
    if batch_size == 0:
        return
    if executed.numel() != batch_size or reused.numel() != batch_size:
        raise ValueError(
            "prefill trace metadata size mismatch: "
            f"request_ids={batch_size}, input_lengths={executed.numel()}, "
            f"prefix_lengths={reused.numel()}"
        )

    max_batches = _env_nonnegative_int("RTP_LLM_PREFILL_TRACE_MAX_BATCHES", 10000)
    short_max = _env_nonnegative_int("RTP_LLM_PREFILL_TRACE_SHORT_MAX_INPUT_LEN", 8192)
    save_hidden = _env_bool("RTP_LLM_PREFILL_TRACE_SAVE_HIDDEN", True)
    max_hidden_files = _env_nonnegative_int(
        "RTP_LLM_PREFILL_TRACE_MAX_HIDDEN_FILES", 2000
    )

    with _LOCK:
        if max_batches > 0 and _BATCH_SEQ >= max_batches:
            return
        batch_seq = _BATCH_SEQ
        _BATCH_SEQ += 1
        trace_dir = Path(trace_dir_value)
        trace_dir.mkdir(parents=True, exist_ok=True)

        totals = executed + reused
        short_rows = torch.nonzero(totals <= short_max, as_tuple=False).reshape(-1)
        requests = []
        for row in range(batch_size):
            total = int(totals[row])
            reuse_len = int(reused[row])
            requests.append(
                {
                    "request_id": int(request_ids_cpu[row]),
                    "input_len": total,
                    "exec_len": int(executed[row]),
                    "reuse_len": reuse_len,
                    "reuse_ratio": float(reuse_len) / total if total else 0.0,
                    "is_short": total <= short_max,
                }
            )

        now_ns = time.time_ns()
        record: Dict[str, Any] = {
            "schema": "rtp_llm_prefill_trace_v1",
            "timestamp_ns": now_ns,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "model_id": int(model_id),
            "tp_rank": int(tp_rank),
            "batch_seq": batch_seq,
            "batch_size": batch_size,
            "batch_exec_tokens": int(executed.sum()),
            "batch_reuse_tokens": int(reused.sum()),
            "batch_input_tokens": int(totals.sum()),
            "short_max_input_len": short_max,
            "requests": requests,
        }

        if short_rows.numel() > 0 and save_hidden:
            if max_hidden_files == 0 or _HIDDEN_FILE_COUNT < max_hidden_files:
                selected = None
                if hidden_states.dim() != 2:
                    record["hidden_error"] = (
                        f"expected 2-D hidden_states, got {tuple(hidden_states.shape)}"
                    )
                elif hidden_is_last_rows:
                    if hidden_states.size(0) != batch_size:
                        record["hidden_error"] = (
                            "last-hidden row mismatch: "
                            f"hidden={hidden_states.size(0)}, batch={batch_size}"
                        )
                    else:
                        selected = hidden_states.index_select(
                            0, short_rows.to(device=hidden_states.device)
                        )
                else:
                    last_rows = (
                        torch.cumsum(executed.to(device=hidden_states.device), dim=0)
                        - 1
                    )
                    if int(last_rows[-1]) >= hidden_states.size(0):
                        record["hidden_error"] = (
                            "full-hidden row mismatch: "
                            f"last_index={int(last_rows[-1])}, "
                            f"hidden={hidden_states.size(0)}"
                        )
                    else:
                        selected = hidden_states.index_select(
                            0,
                            last_rows.index_select(
                                0, short_rows.to(device=last_rows.device)
                            ),
                        )

                if selected is not None:
                    selected_cpu = selected.detach().to(device="cpu").contiguous()
                    filename = (
                        f"prefill_hidden_b{batch_seq:08d}_m{int(model_id)}_"
                        f"p{os.getpid()}_{now_ns}.pt"
                    )
                    _write_hidden_file(
                        trace_dir,
                        filename,
                        {
                            "schema": "rtp_llm_prefill_hidden_v1",
                            "batch_seq": batch_seq,
                            "model_id": int(model_id),
                            "request_ids": request_ids_cpu.index_select(0, short_rows),
                            "input_lengths": totals.index_select(0, short_rows),
                            "exec_lengths": executed.index_select(0, short_rows),
                            "reuse_lengths": reused.index_select(0, short_rows),
                            "last_hidden_states": selected_cpu,
                        },
                    )
                    record["hidden_file"] = filename
                    record["hidden_dtype"] = str(selected_cpu.dtype)
                    record["hidden_shape"] = list(selected_cpu.shape)
                    for row, digest in zip(
                        short_rows.tolist(), [_sha256(row) for row in selected_cpu]
                    ):
                        requests[row]["last_hidden_sha256"] = digest
                    _HIDDEN_FILE_COUNT += 1
            else:
                record["hidden_skipped"] = "max_hidden_files_reached"

        metadata_path = (
            trace_dir / f"prefill_batches_m{int(model_id)}_p{os.getpid()}.jsonl"
        )
        with metadata_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")

        if _env_bool("RTP_LLM_PREFILL_TRACE_LOG", True):
            logging.info(
                "[PREFILL_TRACE] batch_seq=%d exec_tokens=%d reuse_tokens=%d "
                "requests=%s hidden_file=%s",
                batch_seq,
                record["batch_exec_tokens"],
                record["batch_reuse_tokens"],
                [
                    (r["request_id"], r["input_len"], r["exec_len"], r["reuse_len"])
                    for r in requests
                ],
                record.get("hidden_file", ""),
            )


def _reset_for_test() -> None:
    global _BATCH_SEQ, _HIDDEN_FILE_COUNT, _WARNED_MISSING_DIR
    with _LOCK:
        _BATCH_SEQ = 0
        _HIDDEN_FILE_COUNT = 0
        _WARNED_MISSING_DIR = False
