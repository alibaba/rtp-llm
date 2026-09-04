"""Small, fail-open helpers for integrating RTP-LLM with Epsilon/sCR.

The Epsilon API is deliberately kept at the rank boundary. A rank registers
the CUDA-backed KV-cache tensors and, when enabled, arrives at Epsilon's
process-side snapshot barrier. The complete dump/restore lifecycle remains
owned by the external control plane; this module never invokes
``scr_controller`` or performs a dump/restore operation itself.

The helpers are inert unless ``RTPLLM_ENABLE_SCR`` (the historical spelling
``RTP_LLM_ENABLE_SCR`` is accepted as an alias) is enabled.  This is the
single RTP-LLM participation switch: before Epsilon is imported it also
supplies the compatibility shim's ``SCR_ENABLE=1`` when no legacy override is
present.  Checkpoint versus restore is control-plane state and remains owned
by the controller/platform through ``SCR_PHASE``; this module never chooses it.
"""

from __future__ import annotations

import importlib
import logging
import os
import platform
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional


LOGGER = logging.getLogger(__name__)

# ``RTPLLM_ENABLE_SCR`` is the one public switch.  ``SCR_ENABLE`` is a
# compatibility variable populated from it before the first Epsilon import.
# ``SCR_PHASE`` is deliberately not derived here: checkpoint/restore is chosen
# by the sCR controller or restore hook, not by the RTP-LLM feature gate.
RTPLLM_ENABLE_SCR_ENV = "RTPLLM_ENABLE_SCR"
SCR_ENABLE_ENV = RTPLLM_ENABLE_SCR_ENV
SCR_ENABLE_ALIAS_ENV = "RTP_LLM_ENABLE_SCR"
SCR_SHIM_ENABLE_ENV = "SCR_ENABLE"
SCR_PHASE_ENV = "SCR_PHASE"
SCR_EPSILON_DIR = "/etc/scr/epsilon"

SCR_PHASE_CHECKPOINT = "checkpoint"
SCR_PHASE_RESTORE = "restore"
SCR_PHASE_NORMAL = "normal"

def _flag(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _unified_scr_value() -> Optional[str]:
    """Return the canonical gate value, honoring the legacy alias."""

    value = os.environ.get(RTPLLM_ENABLE_SCR_ENV)
    if value is None:
        value = os.environ.get(SCR_ENABLE_ALIAS_ENV)
    return value.strip().lower() if value is not None else None


def configure_scr_environment() -> bool:
    """Normalize the unified SCR switch before importing Epsilon.

    ``RTPLLM_ENABLE_SCR=1`` enables the RTP-LLM integration and selects the
    SCR compatibility shim unless a legacy ``SCR_ENABLE`` override is already
    present.  It does not set ``SCR_PHASE``.  The controller/platform owns that
    state and may inject ``checkpoint`` or ``restore`` for the current
    lifecycle.
    """

    value = _unified_scr_value()
    if not _flag(value):
        return False

    # Preserve explicit compatibility overrides.  A unified-only launcher
    # therefore gets the expected external shim automatically.
    os.environ.setdefault(SCR_SHIM_ENABLE_ENV, "1")
    return True


def is_scr_enabled() -> bool:
    """Return whether the RTP-LLM sCR integration was explicitly requested.

    The default is false.  This function does not import ``epsilon`` and is
    therefore safe to call before CUDA/PyTorch initialization.
    """

    return configure_scr_environment()


def epsilon_backend_mode() -> str:
    """Return the Epsilon implementation selected on the next import.

    The unified switch is normalized before this diagnostic.  The wheel
    selects the external SCR shim only when its directory exists,
    ``SCR_ENABLE=1`` is set, and the kernel release does not contain the
    wheel's ``kangaroo`` marker.
    """

    if not is_scr_enabled():
        return "disabled"
    if (
        _flag(os.environ.get(SCR_SHIM_ENABLE_ENV))
        and os.path.isdir(SCR_EPSILON_DIR)
        and "kangaroo" not in platform.release()
    ):
        return "external-shim"
    return "wheel-native"


def _load_epsilon() -> Any | None:
    """Import Epsilon only when the feature gate is enabled.

    Import failures are intentionally fail-open: a normal RTP-LLM startup must
    still work when the optional package or SCR agent is absent.  The warning
    includes enough context to diagnose a bad image without exposing a stack
    trace during the normal path.
    """

    if not is_scr_enabled():
        return None
    try:
        epsilon = importlib.import_module("epsilon")
        external_dir = str(getattr(epsilon, "_EXTERNAL_DIR", "") or "")
        effective_impl = (
            os.path.join(external_dir, "__init__.py")
            if external_dir
            else getattr(epsilon, "__file__", "")
        )
        LOGGER.info(
            "sCR Epsilon loaded wrapper=%s effective_impl=%s",
            getattr(epsilon, "__file__", ""),
            effective_impl,
        )
        return epsilon
    except Exception as exc:  # pragma: no cover - exact import errors vary by image
        LOGGER.warning("sCR requested but epsilon import failed; continuing: %s", exc)
        return None


def _epsilon_is_active(epsilon: Any) -> bool:
    try:
        return bool(epsilon.is_snapstart_enable())
    except Exception as exc:
        LOGGER.warning("epsilon.is_snapstart_enable() failed; continuing: %s", exc)
        return False


def _is_tensor(value: Any) -> bool:
    # Import torch lazily so the feature gate remains cheap and does not touch
    # CUDA merely because this utility module was imported.
    try:
        import torch

        return isinstance(value, torch.Tensor)
    except Exception:
        return False


def _iter_tensors(value: Any) -> Iterator[Any]:
    """Yield tensors from the nested forms accepted by Epsilon.

    RTP-LLM's bound ``KVCache`` exposes both a flat per-layer view and a
    per-layer/per-region view.  The latter is preferred by callers, but this
    walker is also useful for plain lists/dicts in tests and future models.
    """

    if value is None:
        return
    if _is_tensor(value):
        # Undefined/empty placeholders are emitted by the typed region layout;
        # Epsilon should only receive real allocations.
        try:
            if value.numel() > 0 and value.data_ptr() != 0:
                yield value
        except Exception:
            return
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
        return


def _tensor_key(tensor: Any) -> tuple[Any, ...]:
    """Return a stable key for de-duplicating tensor storage ranges."""

    try:
        device = str(tensor.device)
    except Exception:
        device = ""
    try:
        return (device, int(tensor.data_ptr()), int(tensor.nbytes))
    except Exception:
        # A tensor-like test double may not expose all metadata.  Object id is
        # still sufficient to avoid duplicate references in one registration.
        return (id(tensor),)


def _kv_cache_tensors(kv_cache: Any) -> tuple[Any, ...]:
    """Extract all non-empty base/region/scale cache tensors from KVCache."""

    if kv_cache is None:
        return ()

    # Collect every exposed view.  A pybind vector is often present but empty
    # for ordinary MHA models, so choosing a source solely by ``is not None``
    # would accidentally discard the populated legacy view.  The pointer
    # de-duplication below makes it safe to visit aliases more than once.
    sources: list[Any] = []
    for name in (
        "kv_cache_base_by_layer_region",
        "kv_cache_base_by_layer_region_flat",
        "kv_cache_base_by_layer",
        "kv_scale_base_by_layer_region",
        "kv_scale_base_by_layer_region_flat",
        "kv_scale_base_by_layer",
    ):
        try:
            value = getattr(kv_cache, name, None)
        except Exception:
            value = None
        if value is not None:
            sources.append(value)

    result: list[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for source in sources:
        for tensor in _iter_tensors(source):
            key = _tensor_key(tensor)
            if key in seen:
                continue
            seen.add(key)
            result.append(tensor)
    return tuple(result)


@dataclass(frozen=True)
class ScrRegistration:
    """Diagnostic result retained by the caller after registration."""

    epsilon: Any
    tensors: tuple[Any, ...]
    cache_result: int | None
    hook_result: int | None
    ok: bool


_registration_lock = threading.Lock()
_registrations: dict[int, ScrRegistration] = {}


def _call_result(function: Callable[..., Any], *args: Any, **kwargs: Any) -> int | None:
    result = function(*args, **kwargs)
    if result is None:
        return None
    try:
        return int(result)
    except (TypeError, ValueError):
        return None


def _cuda_synchronize() -> None:
    """Before-checkpoint callback used by default for every active rank."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        # A callback exception must not prevent the agent from attempting its
        # own barrier; the Epsilon/SCR implementation logs the eventual error.
        LOGGER.exception("default sCR CUDA synchronize callback failed")


def _capture_cuda_device() -> Any | None:
    """Capture this rank's current CUDA device for the native callback thread."""

    # Do not import torch here.  Registration is also used by CPU/fake test
    # environments, and a first CUDA import can initialize the driver or NVML
    # for many seconds.  Backend ranks have already imported torch by the time
    # their model/KV cache exists, so consulting sys.modules captures the real
    # device without introducing a new startup side effect.
    torch = sys.modules.get("torch")
    try:
        # ``is_available`` may initialize NVML/driver on some builds.  Only
        # query it after CUDA has already been initialized by the backend.
        if (
            torch is not None
            and bool(getattr(torch.cuda, "_initialized", False))
            and torch.cuda.is_available()
        ):
            return torch.cuda.current_device()
    except Exception:
        LOGGER.exception("unable to capture CUDA device for sCR callback")
    return None


def _make_cuda_synchronize(device: Any | None) -> Callable[[], None]:
    """Build a callback that synchronizes the captured rank device."""

    if device is None:
        return _cuda_synchronize

    def _synchronize_captured_device() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize(device=device)
        except Exception:
            LOGGER.exception("captured sCR CUDA synchronize callback failed")

    return _synchronize_captured_device


def register_for_scr(
    engine: Any,
    *,
    model_name: str = "",
    instance: int = 0,
    rank: int | None = None,
    local_rank: int | None = None,
    after_restore: Callable[..., Any] | None = None,
) -> bool:
    """Register one rank's KV cache and runtime hooks with Epsilon.

    The bound C++ engine has already populated ``py_model.kv_cache`` by this
    point. Epsilon only needs the CUDA-backed cache storage for snapshot and
    restore; the model pointer is deliberately not registered.
    """

    del model_name, instance, rank  # metadata is optional in SCR shim
    if not is_scr_enabled():
        return False

    if engine is None:
        return False

    epsilon = _load_epsilon()
    if epsilon is None or not _epsilon_is_active(epsilon):
        return False

    # Registration may be retried when the cache is initialized lazily. Avoid
    # registering the Epsilon callback twice for the same engine object.
    engine_key = id(engine)
    with _registration_lock:
        previous = _registrations.get(engine_key)
        if previous is not None and previous.ok:
            return True

    engine_model = getattr(engine, "model", None)
    cache_owner = (
        getattr(engine_model, "py_model", None)
        or engine_model
        or getattr(engine, "py_model", None)
        or engine
    )
    kv_cache = getattr(cache_owner, "kv_cache", None)
    tensors = _kv_cache_tensors(kv_cache)

    cache_result: int | None = None
    hook_result: int | None = None
    ok = True

    try:
        if tensors and hasattr(epsilon, "register_kv_caches"):
            cache_result = _call_result(epsilon.register_kv_caches, list(tensors))
            ok = ok and cache_result in (None, 0)
        elif not tensors:
            LOGGER.warning("sCR active but no non-empty KV-cache tensors were found")
            ok = False
        else:
            LOGGER.warning("sCR active but Epsilon KV-cache registration is unavailable")
            ok = False

        if hasattr(epsilon, "register_before_checkpoint_func"):
            captured_device = _capture_cuda_device()
            # If CUDA has not exposed a current device yet, local_rank is the
            # launcher-provided device index and is safer than allowing the
            # native callback thread to default to device 0.
            if captured_device is None and local_rank is not None:
                captured_device = int(local_rank)
            before_callback = _make_cuda_synchronize(captured_device)
            hook_result = _call_result(
                epsilon.register_before_checkpoint_func, before_callback
            )
            ok = ok and hook_result in (None, 0)
        else:
            LOGGER.warning("sCR active but Epsilon before-checkpoint hook is unavailable")
            ok = False
        if after_restore is not None and hasattr(epsilon, "register_after_restore_func"):
            _call_result(epsilon.register_after_restore_func, after_restore)
    except Exception:
        # Registration is an optimization hint; generic sCR dump remains a
        # valid fallback when registration is unavailable.
        LOGGER.exception("sCR registration failed; continuing without hint")
        ok = False

    registration = ScrRegistration(
        epsilon=epsilon,
        tensors=tensors,
        cache_result=cache_result,
        hook_result=hook_result,
        ok=ok,
    )
    with _registration_lock:
        _registrations[engine_key] = registration
    return ok


def arrive_scr_checkpoint_barrier(
    *, worker_id: int, worker_num: int
) -> int | None:
    """Arrive at Epsilon's rank-local snapshot barrier.

    This is the one active-looking call that remains in RTP-LLM. It is not a
    controller operation: ``scr_controller`` still initiates ``check`` /
    ``block`` / ``dump`` / ``restore`` from the control plane. The native
    Epsilon call lets each CUDA rank announce that its registered state is at
    a safe point and then wait for the controller-driven snapshot lifecycle.

    Every participating rank must call this once per snapshot generation with
    a unique ``worker_id`` in ``[0, worker_num)`` and the same ``worker_num``.
    Calls from different processes are expected to happen concurrently.
    """

    if not is_scr_enabled():
        return None

    try:
        worker_id = int(worker_id)
        worker_num = int(worker_num)
    except (TypeError, ValueError):
        LOGGER.error(
            "invalid sCR worker mapping (worker_id=%r worker_num=%r)",
            worker_id,
            worker_num,
        )
        return None
    if worker_num <= 0 or not 0 <= worker_id < worker_num:
        LOGGER.error(
            "invalid sCR worker mapping (worker_id=%d worker_num=%d)",
            worker_id,
            worker_num,
        )
        return None

    epsilon = _load_epsilon()
    if epsilon is None or not _epsilon_is_active(epsilon):
        return None
    checkpoint = getattr(epsilon, "snapstart_checkpoint", None)
    if checkpoint is None:
        LOGGER.warning("sCR active but Epsilon snapshot barrier is unavailable")
        return None

    try:
        return _call_result(
            checkpoint,
            wait_mode=1,
            worker_id=worker_id,
            worker_num=worker_num,
        )
    except Exception:
        # The barrier is optional. A timeout or an unavailable sidecar must
        # not take down a serving rank; the control plane can use a fallback.
        LOGGER.exception(
            "sCR snapshot barrier arrival failed (worker_id=%d worker_num=%d)",
            worker_id,
            worker_num,
        )
        return None


def start_scr_checkpoint_arrival_thread(
    *, worker_id: int, worker_num: int, name: str = "scr-checkpoint-arrival"
) -> threading.Thread | None:
    """Start one daemon thread for this rank's Epsilon barrier arrival.

    The thread is intentionally rank-local and unjoined. A blocking native
    wait therefore cannot delay the rank's startup/serving loop, while the
    external controller remains responsible for the snapshot action.
    """

    if not is_scr_enabled():
        return None

    def _arrive() -> None:
        try:
            arrive_scr_checkpoint_barrier(
                worker_id=worker_id,
                worker_num=worker_num,
            )
        except BaseException:
            LOGGER.exception("sCR snapshot barrier arrival thread failed")

    thread = threading.Thread(target=_arrive, name=name, daemon=True)
    thread.start()
    return thread


def _reset_for_test() -> None:
    """Clear process-local registration state for unit tests."""

    with _registration_lock:
        _registrations.clear()


__all__ = [
    "RTPLLM_ENABLE_SCR_ENV",
    "SCR_ENABLE_ALIAS_ENV",
    "SCR_ENABLE_ENV",
    "SCR_PHASE_CHECKPOINT",
    "SCR_PHASE_ENV",
    "SCR_PHASE_NORMAL",
    "SCR_PHASE_RESTORE",
    "SCR_SHIM_ENABLE_ENV",
    "ScrRegistration",
    "arrive_scr_checkpoint_barrier",
    "epsilon_backend_mode",
    "configure_scr_environment",
    "is_scr_enabled",
    "register_for_scr",
    "start_scr_checkpoint_arrival_thread",
]
