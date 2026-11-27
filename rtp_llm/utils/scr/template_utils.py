"""Small, fail-open helpers for integrating RTP-LLM with Epsilon/sCR.

The Epsilon API is deliberately kept at the rank boundary. A rank registers
the CUDA-backed KV-cache tensors and, when enabled, arrives at Epsilon's
process-side snapshot barrier. The complete dump/restore lifecycle remains
owned by the external control plane; this module never invokes
``scr_controller`` or performs a dump/restore operation itself.

The helpers are inert unless ``RTPLLM_ENABLE_SCR`` is enabled.  This is the
RTP-LLM participation switch.  The SCR runtime's separate ``SCR_ENABLE`` and
``SCR_PHASE`` inputs are supplied by the external control plane/container
environment; this module only reads them and never changes them.
Checkpoint versus restore is control-plane state and remains owned by the
controller/platform through ``SCR_PHASE``; this module never chooses it.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable
from typing import Mapping as TypingMapping

from rtp_llm.utils.scr.lifecycle import (
    RTPLLM_ENABLE_SCR_ENV,
    SCR_PHASE_CHECKPOINT,
    SCR_PHASE_ENV,
    SCR_PHASE_NORMAL,
    SCR_PHASE_RESTORE,
    CallbackHook,
    current_phase,
    get_template_lifecycle,
    is_scr_enabled,
    template_phase_active,
)
from rtp_llm.utils.scr.restore_context import RestoreContext
from rtp_llm.utils.scr.runtime_fixup import fixup_runtime_after_restore
from rtp_llm.utils.signal_guard import ignore_sigpipe

LOGGER = logging.getLogger(__name__)


def _start_native_kmonitor() -> None:
    # Do not import CUDA in CPU/frontend processes just to start metrics.
    extension = sys.modules.get("libth_transformer")
    if extension is None:
        return
    start = getattr(extension, "resume_kmonitor_after_scr", None)
    # Identity fixup remains mandatory. A monitoring sink failure must
    # have the same availability impact as it does during ordinary startup.
    try:
        if start is not None and start():
            return
    except Exception:
        LOGGER.exception("native Kmonitor failed after SCR identity fixup")
        return
    LOGGER.error("native Kmonitor did not start after SCR identity fixup")


get_template_lifecycle().register(
    "native-kmonitor", CallbackHook(release=_start_native_kmonitor)
)


class _ServerConfigTemplateHook:
    """Re-derive ``ServerConfig.ip`` from the current Pod on every restore.

    ``ServerConfig.ip`` is a plain attribute copied into consumers at startup
    (frontend request-id machine bits, log lines). CRIU restores it with the
    seed value, so refresh it after the unified identity fixup and before the
    process releases into serving.
    """

    def __init__(self, py_env_configs: Any) -> None:
        self.configs = py_env_configs

    def restore_fixup(self, context: RestoreContext) -> None:
        self.configs.server_config.ip = context.pod_ip


def register_server_config_template_hook(py_env_configs: Any) -> None:
    get_template_lifecycle().register_fixup(
        f"server-config:{id(py_env_configs)}",
        _ServerConfigTemplateHook(py_env_configs),
        replace=True,
    )


# ``RTPLLM_ENABLE_SCR`` is RTP-LLM's own participation switch.  ``SCR_ENABLE``
# and ``SCR_PHASE`` are external control-plane inputs and are never derived or
# mutated here. Checkpoint versus restore is chosen by the controller/platform.


def epsilon_backend_mode(epsilon: Any | None = None) -> str:
    """Describe a loaded provider without importing it or guessing its path.

    Epsilon owns provider selection. Before import its implementation is
    unknown; filesystem layout and kernel names are not an RTP-LLM contract.
    """

    if not is_scr_enabled():
        return "disabled"
    if epsilon is None:
        epsilon = sys.modules.get("epsilon")
    if epsilon is None:
        return "not-loaded"
    if getattr(epsilon, "_EXTERNAL_DIR", ""):
        return "external-shim"
    return "wheel-native"


def _external_shim_phase_active() -> bool:
    """Apply the controller phase after Epsilon identifies its provider.

    RTP-LLM does not add a provider-selection environment variable. Before
    import, the provider is unknown and the call remains fail-open; once the
    external shim is loaded, only the controller-supplied phase can activate
    the SCR operation.
    """

    mode = epsilon_backend_mode()
    if mode == "external-shim":
        return template_phase_active()
    return True


def _load_epsilon() -> Any | None:
    """Import Epsilon only when the feature gate is enabled.

    Import failures are intentionally fail-open: a normal RTP-LLM startup must
    still work when the optional package or SCR agent is absent.  The warning
    includes enough context to diagnose a bad image without exposing a stack
    trace during the normal path.
    """

    if not is_scr_enabled():
        return None
    # Provider import may load native libraries. Do not probe a
    # deployment-specific directory or kernel name; Epsilon owns provider
    # selection and activation semantics.
    if not _external_shim_phase_active():
        LOGGER.info(
            "sCR external phase=%s inactive; skipping Epsilon import",
            os.environ.get(SCR_PHASE_ENV, "<unset>"),
        )
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
            "sCR Epsilon loaded wrapper=%s effective_impl=%s backend_mode=%s",
            getattr(epsilon, "__file__", ""),
            effective_impl,
            epsilon_backend_mode(epsilon),
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


def _resolve_epsilon() -> Any | None:
    """Load the provider and require it to be active for the current phase."""
    epsilon = _load_epsilon()
    if epsilon is None or not _epsilon_is_active(epsilon):
        return None
    return epsilon


def _is_tensor(value: Any) -> bool:
    # Import torch lazily so the feature gate remains cheap and does not touch
    # CUDA merely because this utility module was imported.
    try:
        import torch

        return isinstance(value, torch.Tensor)
    except Exception:
        return False


def _engine_kv_cache_tensors(engine: Any) -> tuple[Any, ...]:
    """Read complete owning allocations from the native KV allocator.

    Do not infer storage from Python model views: those can omit tiered MLA,
    draft layers or entire pools. An invalid export rejects the whole hint.
    """
    result: list[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for tensor in engine.gpu_cache_tensors():
        if not _is_tensor(tensor):
            raise ValueError("native KV cache export contains a non-tensor")
        if (
            str(tensor.device).split(":", 1)[0] != "cuda"
            or not tensor.is_contiguous()
            or tensor.numel() <= 0
            or tensor.data_ptr() == 0
            or tensor.nbytes <= 0
        ):
            raise ValueError(
                "native KV cache export must contain non-empty contiguous CUDA tensors"
            )
        key = (str(tensor.device), int(tensor.data_ptr()), int(tensor.nbytes))
        if key not in seen:
            seen.add(key)
            result.append(tensor)
    return tuple(result)


@dataclass(frozen=True)
class ScrRegistration:
    """Diagnostic result retained by the caller after registration."""

    tensors: tuple[Any, ...]
    ok: bool
    registration_duration_ms: float = 0.0


class EpsilonProtocolError(RuntimeError):
    """Raised when an Epsilon response cannot be classified safely."""


class ScrArrivalError(RuntimeError):
    """Raised when an active template participant cannot reach its barrier."""


@dataclass(frozen=True)
class ScrParticipantManifest:
    """Stable full-process membership for one Epsilon scheduler scope.

    The manifest only assigns IDs.  It never invokes a controller or performs
    dump/restore; each process uses its ID for the passive Epsilon arrival.
    """

    worker_num: int
    participant_ids: TypingMapping[str, int]

    def worker_id(self, role: str, instance: Any = "0") -> int:
        key = f"{role}:{instance}"
        try:
            return int(self.participant_ids[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError(f"sCR participant {key!r} is not present") from exc

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
    """Assign contiguous IDs to an ordered process-role sequence."""

    mapping: dict[str, int] = {}
    for worker_id, (role, instance) in enumerate(participants):
        key = f"{role}:{instance}"
        if key in mapping:
            raise ValueError(f"duplicate sCR participant {key!r}")
        mapping[key] = worker_id
    manifest = ScrParticipantManifest(len(mapping), mapping)
    manifest.validate()
    return manifest


@dataclass
class _EngineRegistration:
    """One engine's SCR registration state.

    ``engine`` pins the object so ``id(engine)`` cannot be reused while a
    record still refers to it.
    """

    engine: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    record: ScrRegistration | None = None
    before_checkpoint_hook: bool = False


_registration_lock = threading.Lock()
_engine_registrations: dict[int, _EngineRegistration] = {}
_scr_prepare_local = threading.local()


def _call_result(function: Callable[..., Any], *args: Any, **kwargs: Any) -> int | None:
    result = function(*args, **kwargs)
    if result is None:
        return None
    if isinstance(result, bool):
        raise EpsilonProtocolError(
            f"Epsilon returned boolean result {result!r}; expected integer status"
        )
    if isinstance(result, Mapping):
        for key in ("errno", "code", "result", "status"):
            if key in result:
                try:
                    return int(result[key])
                except (TypeError, ValueError) as exc:
                    raise EpsilonProtocolError(
                        f"Epsilon returned non-integer {key}={result[key]!r}"
                    ) from exc
        raise EpsilonProtocolError(
            f"Epsilon returned an unclassifiable mapping: {result!r}"
        )
    try:
        return int(result)
    except (TypeError, ValueError):
        raise EpsilonProtocolError(
            f"Epsilon returned an unclassifiable result: {result!r}"
        )


def _synchronize_cuda(device: Any | None = None) -> None:
    """Synchronize the rank's CUDA work, raising on a failed preparation."""

    # Do not import/initialize CUDA merely because a SCR-enabled CPU process
    # reaches the optional arrival helper.  GPU backend ranks have already
    # initialized torch.cuda before KV registration; a registered CUDA cache
    # therefore cannot be skipped by this guard.
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return

    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize(device=device)


def _prepare_cuda_for_arrival(device: Any | None = None) -> None:
    """Quiesce initialized CUDA work even if there are no KV tensors to register."""

    with _registration_lock:
        gpu_state_registered = any(
            entry.record is not None and bool(entry.record.tensors)
            for entry in _engine_registrations.values()
        )
    torch = sys.modules.get("torch")
    cuda_initialized = bool(
        torch is not None and torch.cuda.is_initialized()
    )
    if gpu_state_registered and not cuda_initialized:
        raise RuntimeError(
            "GPU KV state is registered but torch CUDA is not initialized"
        )
    if not cuda_initialized:
        LOGGER.debug("sCR arrival CUDA prepare skipped for CPU-only participant")
        return
    _synchronize_cuda(device)
    torch.cuda.empty_cache()


def _make_cuda_synchronize(device: Any | None) -> Callable[[], None]:
    """Build the before-checkpoint callback for the captured rank device."""

    def callback() -> None:
        # External SCR invokes this callback synchronously from
        # snapstart_checkpoint. Avoid doing the same CUDA synchronization twice
        # after the arrival path's result-bearing prepare step; a provider
        # callback running on another thread still synchronizes.
        if getattr(_scr_prepare_local, "skip_next", False):
            _scr_prepare_local.skip_next = False
            return
        try:
            _synchronize_cuda(device)
        except Exception:
            LOGGER.exception("sCR CUDA synchronize callback failed")

    return callback


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
            and torch.cuda.is_initialized()
            and torch.cuda.is_available()
        ):
            return torch.cuda.current_device()
    except Exception:
        LOGGER.exception("unable to capture CUDA device for sCR callback")
    return None


def register_for_scr(
    engine: Any,
    *,
    local_rank: int | None = None,
) -> bool:
    """Serialize lazy registration retries for one engine identity."""

    if engine is None or not is_scr_enabled():
        return False
    engine_key = id(engine)
    with _registration_lock:
        entry = _engine_registrations.get(engine_key)
        if entry is None:
            entry = _EngineRegistration(engine=engine)
            _engine_registrations[engine_key] = entry
    with entry.lock:
        return _register_for_scr_once(
            entry,
            local_rank=local_rank,
        )


def _register_for_scr_once(
    entry: _EngineRegistration,
    *,
    local_rank: int | None = None,
) -> bool:
    """Register one rank's KV cache and runtime hooks with Epsilon.

    The bound C++ engine exports the allocator's complete GPU cache buffers.
    Epsilon receives the original allocations without copying or freeing them.
    """

    started = time.monotonic()
    if not is_scr_enabled():
        return False

    engine = entry.engine
    if engine is None:
        return False

    epsilon = _resolve_epsilon()
    if epsilon is None:
        return False

    # Registration may be retried when the cache is initialized lazily. Avoid
    # registering the Epsilon callback twice for the same engine object.
    with _registration_lock:
        previous = entry.record
        if previous is not None and previous.ok:
            return True

    tensors: tuple[Any, ...] = ()
    kv_bytes = 0

    cache_result: int | None = None
    hook_result: int | None = None
    ok = True

    try:
        tensors = _engine_kv_cache_tensors(engine)
        kv_bytes = sum(int(tensor.nbytes) for tensor in tensors)
        if tensors and callable(getattr(epsilon, "register_kv_caches", None)):
            cache_result = _call_result(epsilon.register_kv_caches, list(tensors))
            ok = ok and cache_result in (None, 0)
        elif not tensors:
            LOGGER.warning("sCR active but no non-empty KV-cache tensors were found")
            ok = False
        else:
            LOGGER.warning(
                "sCR active but Epsilon KV-cache registration is unavailable"
            )
            ok = False

        with _registration_lock:
            hook_registered = entry.before_checkpoint_hook
        if hook_registered:
            # A lazy KV-cache retry must not append another callback to the
            # external shim's process-global callback list.
            hook_result = 0
        elif hasattr(epsilon, "register_before_checkpoint_func"):
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
            if hook_result in (None, 0):
                with _registration_lock:
                    entry.before_checkpoint_hook = True
            ok = ok and hook_result in (None, 0)
        else:
            LOGGER.warning(
                "sCR active but Epsilon before-checkpoint hook is unavailable"
            )
            ok = False
    except Exception:
        # Registration is an optimization hint; generic sCR dump remains a
        # valid fallback when registration is unavailable.
        LOGGER.exception("sCR registration failed; continuing without hint")
        ok = False

    registration = ScrRegistration(
        tensors=tensors,
        ok=ok,
        registration_duration_ms=(time.monotonic() - started) * 1000.0,
    )
    with _registration_lock:
        entry.record = registration
    LOGGER.info(
        "sCR registration completed phase=%s tensors=%d cache_result=%s "
        "hook_result=%s ok=%s elapsed_ms=%.3f kv_bytes=%d",
        os.environ.get(SCR_PHASE_ENV, "<unset>"),
        len(tensors),
        cache_result,
        hook_result,
        ok,
        registration.registration_duration_ms,
        kv_bytes,
    )
    return ok


_scr_barrier_return_time: float | None = None


def get_scr_barrier_return_time() -> float | None:
    """Process-local monotonic timestamp after a successful arrival, before fixup.

    This is not the platform's restore start; it excludes checkpoint waiting and
    native restore work inside Epsilon. It is None for processes that did not
    arrive at a template barrier.
    """
    return _scr_barrier_return_time


def arrive_scr_checkpoint_barrier(
    *,
    worker_id: int,
    worker_num: int,
    fail_closed: bool = False,
) -> int | None:
    """Arrive at Epsilon's rank-local snapshot barrier.

    This is the one active-looking call that remains in RTP-LLM. It is not a
    controller operation: ``scr_controller`` still initiates ``check`` /
    ``block`` / ``dump`` / ``restore`` from the control plane. The native
    Epsilon call lets each template participant announce that its process
    state is at a safe point and then wait for the controller-driven snapshot
    lifecycle. GPU-only KV registration happens separately.

    Every participating rank must call this once per checkpoint with
    a unique ``worker_id`` in ``[0, worker_num)`` and the same ``worker_num``.
    Calls from different processes are expected to happen concurrently.
    """

    def _raise_if_strict(message: str, cause: BaseException | None = None):
        if fail_closed:
            if cause is not None:
                raise ScrArrivalError(message) from cause
            raise ScrArrivalError(message)
        return None

    if not is_scr_enabled():
        return None

    global _scr_barrier_return_time
    _scr_barrier_return_time = None

    started = time.monotonic()
    try:
        worker_id = int(worker_id)
        worker_num = int(worker_num)
    except (TypeError, ValueError):
        LOGGER.error(
            "invalid sCR worker mapping (worker_id=%r worker_num=%r)",
            worker_id,
            worker_num,
        )
        return _raise_if_strict(
            f"invalid sCR worker mapping worker_id={worker_id!r} worker_num={worker_num!r}"
        )
    if worker_num <= 0 or not 0 <= worker_id < worker_num:
        LOGGER.error(
            "invalid sCR worker mapping (worker_id=%d worker_num=%d)",
            worker_id,
            worker_num,
        )
        return _raise_if_strict(
            f"invalid sCR worker mapping worker_id={worker_id} worker_num={worker_num}"
        )

    epsilon = _resolve_epsilon()
    if epsilon is None:
        return _raise_if_strict("sCR Epsilon provider is unavailable or inactive")
    checkpoint = getattr(epsilon, "snapstart_checkpoint", None)
    if not callable(checkpoint):
        LOGGER.warning("sCR active but Epsilon snapshot barrier is unavailable")
        return _raise_if_strict("sCR snapshot barrier API is unavailable")

    try:
        LOGGER.info(
            "sCR snapshot arrival started phase=%s worker_id=%d " "worker_num=%d",
            os.environ.get(SCR_PHASE_ENV, "<unset>"),
            worker_id,
            worker_num,
        )
        prepare_device = _capture_cuda_device()
        try:
            _prepare_cuda_for_arrival(prepare_device)
        except Exception:
            LOGGER.exception(
                "sCR CUDA prepare failed; refusing snapshot arrival "
                "worker_id=%d worker_num=%d",
                worker_id,
                worker_num,
            )
            return _raise_if_strict("sCR CUDA preparation failed")
        # The external shim invokes its callback synchronously in this same
        # thread. Mark the successful preparation so the compatibility callback
        # does not perform an unnecessary second synchronize.
        _scr_prepare_local.skip_next = True
        try:
            # Provider discovery above may load native modules that set signals.
            ignore_sigpipe("before Epsilon snapshot arrival")
            # One blocking call. The provider/control plane owns its deadline
            # and cancellation policy; do not negotiate or override them here.
            result = _call_result(
                checkpoint, wait_mode=1, worker_id=worker_id, worker_num=worker_num
            )
        finally:
            # Do not let an exception or a provider that invokes the callback
            # asynchronously suppress synchronization on the next arrival.
            _scr_prepare_local.skip_next = False
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if result not in (None, 0):
            LOGGER.error(
                "sCR snapshot arrival returned non-zero phase=%s "
                "worker_id=%d worker_num=%d result=%s elapsed_ms=%.3f",
                os.environ.get(SCR_PHASE_ENV, "<unset>"),
                worker_id,
                worker_num,
                result,
                elapsed_ms,
            )
            if fail_closed:
                raise ScrArrivalError(
                    f"sCR snapshot arrival returned status {result!r}"
                )
        else:
            # Use a fresh clock origin for post-restore startup metrics.
            # Waiting belongs to the controller/provider; elapsed time is
            # diagnostic only.
            _scr_barrier_return_time = time.monotonic()
            LOGGER.info(
                "sCR snapstart checkpoint reached pid=%d phase=%s "
                "worker_id=%d worker_num=%d result=%s elapsed_ms=%.3f",
                os.getpid(),
                os.environ.get(SCR_PHASE_ENV, "<unset>"),
                worker_id,
                worker_num,
                result,
                elapsed_ms,
            )
        return result
    except Exception as exc:
        # Preserve provider failures. Active template participants fail closed
        # instead of releasing listeners after a failed checkpoint arrival.
        LOGGER.exception(
            "sCR snapshot barrier arrival failed phase=%s "
            "worker_id=%d worker_num=%d elapsed_ms=%.3f",
            os.environ.get(SCR_PHASE_ENV, "<unset>"),
            worker_id,
            worker_num,
            (time.monotonic() - started) * 1000.0,
        )
        if fail_closed:
            raise ScrArrivalError(
                "sCR snapshot barrier arrival failed "
                f"worker_id={worker_id} worker_num={worker_num}: {exc}"
            ) from exc
        return None


def arrive_scr_template_barrier(
    *,
    worker_id: int,
    worker_num: int,
    fail_closed: bool = False,
) -> int | None:
    """Run lifecycle hooks and Epsilon arrival as one template barrier."""

    # Helper processes can close their pipes while the dump is running; keep
    # SIGPIPE ignored so those writes fail with EPIPE instead of killing a rank
    # that is being captured.
    ignore_sigpipe("before template arrival")

    if not template_phase_active():
        return arrive_scr_checkpoint_barrier(
            worker_id=worker_id,
            worker_num=worker_num,
            fail_closed=fail_closed,
        )

    phase = current_phase()
    lifecycle = get_template_lifecycle()
    lifecycle.prepare_for_template(phase)
    try:
        result = arrive_scr_checkpoint_barrier(
            worker_id=worker_id,
            worker_num=worker_num,
            # A prepared template must never release after a failed arrival.
            # Legacy Epsilon APIs return None on success, so enforce errors at
            # the arrival boundary instead of interpreting None here.
            fail_closed=True,
        )
        fixup_runtime_after_restore(lifecycle)
        lifecycle.release_template()
        return result
    except BaseException:
        lifecycle.abort_template()
        raise


def _reset_for_test() -> None:
    """Clear process-local registration state for unit tests."""

    global _scr_barrier_return_time
    _scr_barrier_return_time = None
    with _registration_lock:
        _engine_registrations.clear()
    if hasattr(_scr_prepare_local, "skip_next"):
        del _scr_prepare_local.skip_next

