"""Opt-in Qwen3.5 Full-Attention/KV diagnostics."""
from __future__ import annotations
import json, os, threading
from pathlib import Path
from typing import Any, Optional
import torch

_LOCK = threading.Lock()
_RESET: Optional[str] = None
_STEP = 0
_COUNT = 0

def _ints(value: str) -> set[int]:
    out: set[int] = set()
    for x in value.split(","):
        try:
            if x.strip():
                out.add(int(x.strip()))
        except ValueError:
            pass
    return out

def _pos_ok(pos: int) -> bool:
    for x in os.environ.get("RTP_QWEN35_DIAG_POSITIONS", "120:136,248:264").split(","):
        try:
            x = x.strip()
            if ":" in x:
                a, b = (int(y) for y in x.split(":", 1))
                if a <= pos <= b:
                    return True
            elif x and pos == int(x):
                return True
        except ValueError:
            pass
    return False

def _meta(x: Any) -> Optional[dict[str, Any]]:
    if not isinstance(x, torch.Tensor):
        return None
    return {"shape": list(x.shape), "stride": list(x.stride()),
            "dtype": str(x.dtype), "device": str(x.device)}

def _vals(x: Any) -> list[Any]:
    return x.detach().cpu().reshape(-1).tolist() if isinstance(x, torch.Tensor) else []

def _save(d: Path, name: str, x: torch.Tensor) -> str:
    p = d / (name + ".pt")
    torch.save(x.detach().cpu(), p)
    return str(p)

def _kv(ai: Any, cache: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("kv_cache_block_id_host", "kv_cache_kernel_block_id_host",
                 "kv_cache_kernel_block_id_device", "sequence_lengths",
                 "sequence_lengths_plus_1_d", "combo_position_ids"):
        x = getattr(ai, name, None)
        if isinstance(x, torch.Tensor):
            result[name] = {"meta": _meta(x), "values": _vals(x)}
    if cache is not None:
        result["kv_cache_base"] = _meta(getattr(cache, "kv_cache_base", None))
        result["kv_scale_base"] = _meta(getattr(cache, "kv_scale_base", None))
        result["seq_size_per_block"] = getattr(cache, "seq_size_per_block", None)
        result["layer_id"] = getattr(cache, "layer_id", None)
    return result

def capture_layer(*, layer_idx: int, hidden_in: torch.Tensor,
                  hidden_out: torch.Tensor, residual: torch.Tensor,
                  attention_inputs: Any, kv_cache: Any,
                  input_ids: Optional[torch.Tensor]) -> None:
    global _RESET, _STEP, _COUNT
    if os.environ.get("RTP_QWEN35_DIAG_ENABLE") != "1":
        return
    if getattr(attention_inputs, "is_cuda_graph", False):
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if layer_idx not in _ints(os.environ.get("RTP_QWEN35_DIAG_LAYERS", "3")):
        return
    ranks = _ints(os.environ.get("RTP_QWEN35_DIAG_RANKS", ""))
    if ranks and rank not in ranks:
        return
    with _LOCK:
        reset = os.environ.get("RTP_QWEN35_DIAG_RESET", "0")
        if reset != _RESET:
            _RESET, _STEP, _COUNT = reset, 0, 0
        step = _STEP
        _STEP += 1
        if _COUNT >= int(os.environ.get("RTP_QWEN35_DIAG_MAX_STEPS", "1")):
            return
        _COUNT += 1
    pos = getattr(attention_inputs, "combo_position_ids", None)
    if not isinstance(pos, torch.Tensor):
        pos = torch.arange(hidden_in.shape[0], device=hidden_in.device)
    position_values = _vals(pos)
    rows = [i for i, x in enumerate(position_values) if _pos_ok(int(x))]
    if not rows:
        return
    index = torch.tensor(rows, device=hidden_in.device, dtype=torch.long)
    directory = Path(os.environ.get("RTP_QWEN35_DIAG_DIR", "/tmp/rtp_qwen35_diag"))
    directory = directory / ("rank%d" % rank) / ("step_%06d_layer_%d" % (step, layer_idx))
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "step_id": step, "layer_id": layer_idx, "rank": rank,
        "positions": [position_values[i] for i in rows],
        "consumed_input_token_ids": _vals(input_ids)[:len(position_values)],
        "hidden_in": _meta(hidden_in), "hidden_out": _meta(hidden_out),
        "residual": _meta(residual), "kv": _kv(attention_inputs, kv_cache),
        "generated_token_ids": "collect from API response; input_ids are consumed tokens",
    }
    record["hidden_in_file"] = _save(directory, "hidden_in", hidden_in.index_select(0, index))
    record["hidden_out_file"] = _save(directory, "hidden_out", hidden_out.index_select(0, index))
    record["residual_file"] = _save(directory, "residual", residual.index_select(0, index))
    (directory / "metadata.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

__all__ = ["capture_layer"]
