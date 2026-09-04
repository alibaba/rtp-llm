"""Small, fail-open helpers for integrating RTP-LLM with Epsilon/sCR.

The Epsilon API is deliberately kept at the rank boundary.  A rank registers
the CUDA-backed KV-cache tensors, then enters Epsilon's
steady-point barrier.  ``scr_controller`` remains an out-of-process control
client; this module does not invoke it from every rank.

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
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping as TypingMapping, Optional


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

SCR_WORKER_ID_ENV = "RTP_LLM_SCR_WORKER_ID"
SCR_WORKER_NUM_ENV = "RTP_LLM_SCR_WORKER_NUM"
SCR_WORKER_OFFSET_ENV = "RTP_LLM_SCR_WORKER_OFFSET"
SCR_TIMEOUT_ENV = "RTP_LLM_SCR_TIMEOUT"
SCR_INACTIVITY_TIMEOUT_ENV = "RTP_LLM_SCR_INACTIVITY_TIMEOUT"
SCR_TRIGGER_FILE_ENV = "RTP_LLM_SCR_TRIGGER_FILE"

DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_INACTIVITY_TIMEOUT_SECONDS = 10


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


@dataclass(frozen=True)
class ScrParticipantManifest:
    """Frozen process-wide Epsilon quorum membership.

    Epsilon's ``wait_mode=1`` barrier is positional: every participant must
    use one unique ID in ``[0, worker_num)``.  A local ``multiprocessing``
    barrier cannot represent the complete process tree (the launcher owns
    direct children while the backend manager owns GPU-rank children), so the
    launcher computes this immutable mapping once and passes it to every
    process.  Keys are stable role/instance strings and do not contain PIDs.
    """

    worker_num: int
    participant_ids: TypingMapping[str, int]

    def worker_id(self, role: str, instance: Any = "0") -> int:
        key = f"{role}:{instance}"
        try:
            return int(self.participant_ids[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError(
                f"sCR participant {key!r} is not present in the frozen manifest"
            ) from exc

    def validate(self) -> None:
        ids = sorted(int(value) for value in self.participant_ids.values())
        if self.worker_num <= 0 or ids != list(range(self.worker_num)):
            raise ValueError(
                "invalid sCR participant manifest: "
                f"worker_num={self.worker_num}, ids={ids}"
            )


def build_scr_participant_manifest(
    participants: list[tuple[str, Any]],
) -> ScrParticipantManifest:
    """Assign contiguous IDs to an ordered ``(role, instance)`` sequence."""

    mapping: dict[str, int] = {}
    for worker_id, (role, instance) in enumerate(participants):
        key = f"{role}:{instance}"
        if key in mapping:
            raise ValueError(f"duplicate sCR participant {key!r}")
        mapping[key] = worker_id
    manifest = ScrParticipantManifest(len(mapping), mapping)
    manifest.validate()
    return manifest


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

    # Startup can install both a formal coordinator and a hot hook.  Avoid
    # registering the callback twice when they happen to observe the same
    # engine object.
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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        LOGGER.warning("invalid %s=%r; using %d", name, raw, default)
        return default


def _default_local_rank() -> int:
    # LOCAL_RANK is the canonical rank-local identity.  RANK is only a
    # fallback for single-node launchers that do not export LOCAL_RANK.
    return _env_int("LOCAL_RANK", _env_int("RANK", 0))


def _resolve_worker_id(worker_id: int | None) -> int:
    # Explicit manifest IDs are authoritative.  The environment override is
    # retained for legacy/direct callers (for example PD roles sharing one
    # scheduler) that do not receive a manifest.
    if worker_id is not None:
        return int(worker_id)
    if os.environ.get(SCR_WORKER_ID_ENV) is not None:
        return _env_int(SCR_WORKER_ID_ENV, 0)
    return _env_int(SCR_WORKER_ID_ENV, _default_local_rank())


def _resolve_worker_num(worker_num: int | None) -> int:
    if worker_num is not None and int(worker_num) > 0:
        return int(worker_num)
    # Epsilon's wait_mode=1 is scoped to workers in this Pod.  LOCAL_WORLD_SIZE
    # is therefore preferred over global WORLD_SIZE.
    value = _env_int(SCR_WORKER_NUM_ENV, 0)
    if value > 0:
        return value
    value = _env_int("LOCAL_WORLD_SIZE", 0)
    return value if value > 0 else 1


def start_scr_checkpoint(
    *,
    worker_id: int | None = None,
    worker_num: int | None = None,
    timeout: int | None = None,
    inactivity_timeout: int | None = None,
) -> int | None:
    """Enter Epsilon's process-wide steady-point barrier.

    The launcher calls this once in every process that belongs to the frozen
    restore unit (parent, backend manager/ranks, frontend, and DashSc).  It
    does not execute ``scr_controller``; the controller/coordinator separately
    performs ``check``/``dump``/``restore`` over the scheduler UDS.
    ``None`` means the optional integration was inactive or unavailable.
    """

    if not is_scr_enabled():
        return None
    epsilon = _load_epsilon()
    if epsilon is None or not _epsilon_is_active(epsilon):
        return None

    resolved_timeout = (
        int(timeout)
        if timeout is not None
        else _env_int(SCR_TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS)
    )
    resolved_inactivity = (
        int(inactivity_timeout)
        if inactivity_timeout is not None
        else _env_int(
            SCR_INACTIVITY_TIMEOUT_ENV, DEFAULT_INACTIVITY_TIMEOUT_SECONDS
        )
    )
    resolved_timeout = max(1, resolved_timeout)
    resolved_inactivity = max(0, resolved_inactivity)
    resolved_worker_num = _resolve_worker_num(worker_num)
    resolved_worker_id = _resolve_worker_id(worker_id)
    # The scheduler quorum is positional: IDs must be unique and cover
    # [0, worker_num).  Refuse an invalid mapping instead of allowing a rank
    # to poison the shared scope or wait forever for a missing worker.
    if resolved_worker_num <= 0 or not 0 <= resolved_worker_id < resolved_worker_num:
        LOGGER.error(
            "invalid sCR worker mapping (worker_id=%d worker_num=%d); "
            "falling back to normal startup",
            resolved_worker_id,
            resolved_worker_num,
        )
        return None

    try:
        return _call_result(
            epsilon.snapstart_checkpoint,
            wait_mode=1,
            worker_id=resolved_worker_id,
            worker_num=resolved_worker_num,
            timeout=resolved_timeout,
            inactivity_timeout=resolved_inactivity,
        )
    except Exception:
        # The caller may still serve traffic or let the external controller
        # choose a cold-start fallback.
        LOGGER.exception(
            "sCR snapstart_checkpoint failed (worker_id=%d worker_num=%d)",
            resolved_worker_id,
            resolved_worker_num,
        )
        return None


def _shutdown_requested(manager: Any) -> bool:
    try:
        if bool(getattr(manager, "shutdown_requested", False)):
            return True
    except Exception:
        pass
    event = getattr(manager, "_shutdown_requested", None) if manager else None
    try:
        return bool(event is not None and event.is_set())
    except Exception:
        return False


def start_scr_checkpoint_thread(
    manager: Any = None,
    *,
    engine: Any = None,
    worker_id: int | None = None,
    worker_num: int | None = None,
    timeout: int | None = None,
    inactivity_timeout: int | None = None,
    trigger_file: str | None = None,
    idle_grace_seconds: float = 0.0,
    poll_interval_seconds: float = 0.25,
    name: str = "scr-checkpoint-waiter",
) -> threading.Thread | None:
    """Start a fail-open daemon waiter for one process's checkpoint call.

    ``trigger_file`` is optional and intended for tests/controlled template
    production.  With no trigger, the thread enters Epsilon immediately after
    the caller has registered model/cache state.  A manager with a
    ``_shutdown_requested`` event can stop the wait before the trigger arrives.
    """

    if not is_scr_enabled():
        return None

    if trigger_file is None:
        trigger_file = os.environ.get(SCR_TRIGGER_FILE_ENV)

    def _wait_and_checkpoint() -> None:
        try:
            if trigger_file:
                while not os.path.exists(trigger_file):
                    if _shutdown_requested(manager):
                        return
                    time.sleep(max(0.01, poll_interval_seconds))
            if idle_grace_seconds > 0:
                deadline = time.monotonic() + idle_grace_seconds
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    if _shutdown_requested(manager):
                        return
                    time.sleep(min(max(0.01, poll_interval_seconds), remaining))
            # Re-run registration here as well as at startup.  This supports
            # launchers that switch SCR_PHASE from ``normal`` to ``checkpoint``
            # immediately before touching the trigger file.
            if engine is not None:
                if not register_for_scr(engine, local_rank=worker_id):
                    # Registration is an optimization for CUDA KV-cache
                    # restore, not membership.  Every process in the frozen
                    # manifest must still arrive at the common barrier;
                    # otherwise CPU participants would wait forever for a GPU
                    # rank that failed to expose a cache hint.
                    LOGGER.warning(
                        "sCR registration unavailable; entering common checkpoint "
                        "barrier without KV-cache hint"
                    )
            start_scr_checkpoint(
                worker_id=worker_id,
                worker_num=worker_num,
                timeout=timeout,
                inactivity_timeout=inactivity_timeout,
            )
        except BaseException:
            # Daemon-thread failures must never turn into a backend startup
            # failure.  Log the traceback and leave controller fallback intact.
            LOGGER.exception("sCR checkpoint waiter terminated unexpectedly")

    thread = threading.Thread(target=_wait_and_checkpoint, name=name, daemon=True)
    thread.start()
    return thread


def _reset_for_test() -> None:
    """Clear process-local registration state for unit tests."""

    with _registration_lock:
        _registrations.clear()


__all__ = [
    "DEFAULT_INACTIVITY_TIMEOUT_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "RTPLLM_ENABLE_SCR_ENV",
    "SCR_ENABLE_ALIAS_ENV",
    "SCR_ENABLE_ENV",
    "SCR_PHASE_CHECKPOINT",
    "SCR_PHASE_ENV",
    "SCR_PHASE_NORMAL",
    "SCR_PHASE_RESTORE",
    "SCR_INACTIVITY_TIMEOUT_ENV",
    "SCR_SHIM_ENABLE_ENV",
    "SCR_TIMEOUT_ENV",
    "SCR_TRIGGER_FILE_ENV",
    "SCR_WORKER_ID_ENV",
    "SCR_WORKER_NUM_ENV",
    "SCR_WORKER_OFFSET_ENV",
    "ScrParticipantManifest",
    "ScrRegistration",
    "build_scr_participant_manifest",
    "epsilon_backend_mode",
    "configure_scr_environment",
    "is_scr_enabled",
    "register_for_scr",
    "start_scr_checkpoint",
    "start_scr_checkpoint_thread",
]
