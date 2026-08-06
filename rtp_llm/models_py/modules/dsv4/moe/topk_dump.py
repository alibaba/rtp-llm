"""Opt-in recorder for the actual ``fp8_fp4_mega_moe`` top-k input."""

from __future__ import annotations

import contextlib
import os
import re
import threading
from dataclasses import dataclass
from typing import Iterator

import torch

_DUMP_DIR_ENV = "DSV4_MEGA_MOE_TOPK_DUMP_DIR"
_PREFILL_DIR_ENV = "DSV4_MEGA_MOE_TOPK_DUMP_PREFILL_DIR"
_DECODE_DIR_ENV = "DSV4_MEGA_MOE_TOPK_DUMP_DECODE_DIR"
_QUERY_FILE_ENV = "DSV4_MEGA_MOE_TOPK_DUMP_QUERY_FILE"
_DECODE_STEPS_ENV = "DSV4_MEGA_MOE_TOPK_DUMP_STEPS"
_PREFILL_STEPS_ENV = "DSV4_MEGA_MOE_TOPK_DUMP_PREFILL_STEPS"
_ARM_FILE_ENV = "DSV4_MEGA_MOE_TOPK_DUMP_ARM_FILE"
_DEFAULT_DECODE_STEPS = 3
_DEFAULT_PREFILL_STEPS = 1


@dataclass(frozen=True)
class _ForwardContext:
    is_decode_role: bool
    is_fake_stream: bool
    model_name: str
    dump_dir: str | None
    query_name: str | None
    max_steps: int


_local = threading.local()
_save_lock = threading.Lock()


def configured() -> bool:
    return any(
        os.environ.get(name, "").strip()
        for name in (_DUMP_DIR_ENV, _PREFILL_DIR_ENV, _DECODE_DIR_ENV)
    )


def _arm_file(dump_dir: str) -> str:
    return os.environ.get(_ARM_FILE_ENV, os.path.join(dump_dir, ".armed"))


def _role_config(is_decode_role: bool) -> tuple[str | None, int]:
    if is_decode_role:
        dump_dir = os.environ.get(_DECODE_DIR_ENV, "").strip()
        if not dump_dir:
            dump_dir = os.environ.get(_DUMP_DIR_ENV, "").strip()
        steps = int(os.environ.get(_DECODE_STEPS_ENV, str(_DEFAULT_DECODE_STEPS)))
    else:
        dump_dir = os.environ.get(_PREFILL_DIR_ENV, "").strip()
        steps = int(os.environ.get(_PREFILL_STEPS_ENV, str(_DEFAULT_PREFILL_STEPS)))
    return (dump_dir or None), steps


def _active_query(dump_dir: str | None) -> str | None:
    query_file = os.environ.get(_QUERY_FILE_ENV, "").strip()
    if query_file:
        try:
            with open(query_file, encoding="utf-8") as handle:
                query_name = handle.read().strip()
        except OSError:
            return None
        return _safe_component(query_name) if query_name else None

    if dump_dir and os.path.isfile(_arm_file(dump_dir)):
        return ""
    return None


@contextlib.contextmanager
def forward_context(
    *, is_decode_role: bool, is_fake_stream: bool, model_name: str
) -> Iterator[None]:
    """Expose request metadata to nested MoE strategies for one forward."""
    dump_dir, max_steps = _role_config(bool(is_decode_role))
    query_name = _active_query(dump_dir)
    previous = getattr(_local, "forward_context", None)
    _local.forward_context = _ForwardContext(
        is_decode_role=bool(is_decode_role),
        is_fake_stream=bool(is_fake_stream),
        model_name=model_name,
        dump_dir=dump_dir,
        query_name=query_name,
        max_steps=max_steps,
    )
    try:
        yield
    finally:
        _local.forward_context = previous


@contextlib.contextmanager
def cp_context(cp_ctx: object | None) -> Iterator[None]:
    """Expose the prefill CP layout while the transformer layers run."""
    previous = getattr(_local, "cp_context", None)
    _local.cp_context = cp_ctx
    try:
        yield
    finally:
        _local.cp_context = previous


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return int(os.environ.get("RANK", os.environ.get("WORLD_RANK", "0")))


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def maybe_dump(strategy: object, topk_idx: torch.Tensor, token_count: int) -> None:
    """Save a complete prefill layer or the real decode DP-rank tensor."""
    context = getattr(_local, "forward_context", None)
    if (
        context is None
        or context.is_fake_stream
        or not context.dump_dir
        or context.query_name is None
    ):
        return

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "MegaMoE topk dumping requires eager execution; CUDA graph must be disabled"
        )

    if context.max_steps <= 0:
        return

    with _save_lock:
        steps_by_query = getattr(strategy, "_topk_dump_steps_by_query", None)
        if steps_by_query is None:
            steps_by_query = {}
            setattr(strategy, "_topk_dump_steps_by_query", steps_by_query)
        query_key = (context.dump_dir, context.query_name)
        step = int(steps_by_query.get(query_key, 0))
        if step >= context.max_steps:
            return

        layer_id = int(getattr(getattr(strategy, "cfg"), "layer_id"))
        rank = _rank()
        model_name = _safe_component(context.model_name)
        path_parts = [context.dump_dir]
        if context.query_name:
            path_parts.append(f"query_{context.query_name}")
        path_parts.extend([f"step_{step:03d}", model_name, f"layer_{layer_id:03d}"])
        out_dir = os.path.join(*path_parts)
        local_topk = topk_idx[: int(token_count)].detach().contiguous()
        cp_ctx = None
        if not context.is_decode_role:
            cp_ctx = getattr(_local, "cp_context", None)
        if cp_ctx is not None and int(getattr(cp_ctx, "cp_size", 1)) > 1:
            expected_local_rows = int(getattr(cp_ctx, "chunk_length"))
            if int(local_topk.size(0)) != expected_local_rows:
                # Long MoE forwards are split into fixed-size kernel chunks.
                # Preserve their actual packed top-k inputs in call order and
                # gather only after this layer's complete CP-local slice exists.
                pending_by_query = getattr(
                    strategy, "_topk_dump_pending_by_query", None
                )
                if pending_by_query is None:
                    pending_by_query = {}
                    setattr(strategy, "_topk_dump_pending_by_query", pending_by_query)
                pending = pending_by_query.setdefault(query_key, [])
                pending.append(local_topk.clone())
                pending_rows = sum(int(chunk.size(0)) for chunk in pending)
                if pending_rows < expected_local_rows:
                    return
                if pending_rows > expected_local_rows:
                    raise ValueError(
                        f"MegaMoE topk chunks have {pending_rows} rows, expected "
                        f"CP chunk_length {expected_local_rows}"
                    )
                local_topk = torch.cat(pending, dim=0)
                del pending_by_query[query_key]

            # Every CP rank must enter this collective. cp_all_gather_full uses
            # the framework's restore indices to undo zigzag rank partitioning
            # and strip padding, yielding the original global token order.
            from rtp_llm.models_py.modules.dsv4.cp import cp_all_gather_full

            full_topk = cp_all_gather_full(
                local_topk,
                cp_ctx,
                profile_name="dsv4.topk_dump.cp_all_gather",
            )
            steps_by_query[query_key] = step + 1
            if int(getattr(cp_ctx, "cp_rank")) != 0:
                return
            snapshot = full_topk.cpu().clone()
            filename = "all_tokens.pt"
        else:
            snapshot = local_topk.cpu().clone()
            filename = (
                f"rank_{rank:03d}.pt" if context.is_decode_role else "all_tokens.pt"
            )

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        tmp_path = f"{out_path}.pid{os.getpid()}.tmp"
        torch.save(snapshot, tmp_path)
        os.replace(tmp_path, out_path)
        steps_by_query[query_key] = step + 1
        print(
            "[DSV4 MegaMoE topk dump] "
            f"query={context.query_name or 'legacy'} step={step} "
            f"model={model_name} layer={layer_id} rank={rank} "
            f"shape={tuple(snapshot.shape)} path={out_path}",
            flush=True,
        )
