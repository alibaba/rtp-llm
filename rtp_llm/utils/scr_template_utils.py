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
import inspect
import logging
import os
import socket
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping as TypingMapping, Optional

from rtp_llm.utils.scr_template_lifecycle import get_template_lifecycle


LOGGER = logging.getLogger(__name__)


class _NativeKmonitorTemplateHook:
    """Pause native kmonitor without importing the CUDA extension eagerly."""

    def __init__(self) -> None:
        self._paused = False

    @staticmethod
    def _extension():
        # The backend imports libth_transformer during engine construction. Do
        # not import it in frontend/CPU-only processes just to register a hook.
        return sys.modules.get("libth_transformer")

    def prepare_for_template(self, generation: str) -> None:
        extension = self._extension()
        pause = getattr(extension, "pause_kmonitor_for_scr", None) if extension else None
        if extension is not None and pause is None:
            # A loaded extension must expose the native hook.  Silently
            # continuing would capture an active metrics sink and can leave
            # the restored process with duplicate/stale reporters.  Frontend
            # and CPU-only processes simply have no extension and remain
            # optional participants.
            raise RuntimeError("loaded native library lacks SCR Kmonitor hooks")
        self._paused = bool(pause()) if pause is not None else False
        if self._paused:
            LOGGER.info("native Kmonitor paused for SCR generation=%s", generation)

    def restore_fixup(self, generation: str) -> None:
        return None

    def release_template(self, generation: str) -> None:
        self._resume(generation)

    def abort_template(self, generation: str) -> None:
        self._resume(generation)

    def _resume(self, generation: str) -> None:
        if not self._paused:
            return
        extension = self._extension()
        resume = getattr(extension, "resume_kmonitor_after_scr", None) if extension else None
        # CRIU preserves seed environment values. Resolve the current namespace
        # identity before native Kmonitor rebuilds its configuration.
        if os.environ.get("HIPPO_ROLE"):
            try:
                os.environ["RequestedIP"] = socket.gethostbyname(socket.gethostname())
            except OSError:
                LOGGER.warning(
                    "Cannot resolve current container IP for native Kmonitor",
                    exc_info=True,
                )
        if resume is None or not resume():
            raise RuntimeError("native Kmonitor did not resume after SCR")
        self._paused = False
        LOGGER.info("native Kmonitor released for SCR generation=%s", generation)


class _PythonKmonitorTemplateHook:
    """Quiesce the already-loaded Python reporter during a template barrier."""

    @staticmethod
    def _worker() -> Any | None:
        module = sys.modules.get(
            "rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker"
        )
        return getattr(module, "report_worker", None)

    def __init__(self) -> None:
        self._worker_ref: Any | None = None
        self._was_started = False

    def prepare_for_template(self, generation: str) -> None:
        del generation
        worker = self._worker()
        self._worker_ref = worker
        self._was_started = False
        if worker is not None:
            try:
                self._was_started = bool(worker.pause_for_checkpoint())
            except BaseException:
                self._worker_ref = None
                self._was_started = False
                raise

    def restore_fixup(self, generation: str) -> None:
        del generation

    def release_template(self, generation: str) -> None:
        del generation
        worker = self._worker_ref
        was_started = self._was_started
        self._worker_ref = None
        self._was_started = False
        if worker is not None:
            worker.resume_after_checkpoint(was_started)

    def abort_template(self, generation: str) -> None:
        self.release_template(generation)


_NATIVE_KMONITOR_HOOK = _NativeKmonitorTemplateHook()
_PYTHON_KMONITOR_HOOK = _PythonKmonitorTemplateHook()
get_template_lifecycle().register("native-kmonitor", _NATIVE_KMONITOR_HOOK)
get_template_lifecycle().register("python-kmonitor", _PYTHON_KMONITOR_HOOK)


class _BackendVisitorTemplateHook:
    def __init__(self, visitor: Any, py_env_configs: Any) -> None:
        self.visitor = visitor
        self.configs = py_env_configs

    def prepare_for_template(self, generation: str) -> None:
        return None

    def restore_fixup(self, generation: str) -> None:
        from rtp_llm.distribute.distributed_server import (
            get_dp_addrs_from_world_info,
            get_world_info,
        )
        from rtp_llm.utils.scr_endpoint_provider import resolve_world_info

        current = get_world_info(
            self.configs.server_config,
            self.configs.distribute_config,
            self.configs.parallelism_config,
        )
        role = self.configs.role_config.role_type
        role_name = str(getattr(role, "name", role)).lower()
        world_info = resolve_world_info(
            current,
            generation=generation,
            require_manifest=os.environ.get("SCR_PHASE", "").strip().lower() == "restore"
            and (current.num_nodes > 1 or role_name in {"prefill", "decode", "role_type.prefill", "role_type.decode"}),
            require_transport=os.environ.get("SCR_PHASE", "").strip().lower() == "restore"
            and (current.num_nodes > 1 or role_name in {"prefill", "decode", "role_type.prefill", "role_type.decode"}),
        )
        self.visitor.update_addresses(
            get_dp_addrs_from_world_info(world_info, self.configs.parallelism_config)
        )

    def release_template(self, generation: str) -> None:
        return None

    def abort_template(self, generation: str) -> None:
        return None


def register_backend_visitor_template_hook(visitor: Any, py_env_configs: Any) -> None:
    get_template_lifecycle().register(
        f"backend-visitor:{id(visitor)}",
        _BackendVisitorTemplateHook(visitor, py_env_configs),
    )

# ``RTPLLM_ENABLE_SCR`` is RTP-LLM's own participation switch.  ``SCR_ENABLE``
# and ``SCR_PHASE`` are external control-plane inputs and are never derived or
# mutated here. Checkpoint versus restore is chosen by the controller/platform.
RTPLLM_ENABLE_SCR_ENV = "RTPLLM_ENABLE_SCR"
# Kept for callers that historically used this constant to refer to the
# RTP-LLM gate; it is intentionally the same single switch, not a second gate.
SCR_ENABLE_ENV = RTPLLM_ENABLE_SCR_ENV
SCR_PHASE_ENV = "SCR_PHASE"

SCR_PHASE_CHECKPOINT = "checkpoint"
SCR_PHASE_RESTORE = "restore"
SCR_PHASE_NORMAL = "normal"

# Epsilon's wait_mode=1 IDs are scoped by the scheduler, not by Python's
# process tree. Keep the mapping explicit so a deployment that puts prefill
# and decode ranks in one scheduler can give the roles disjoint ranges. The
# default remains one local Pod (LOCAL_WORLD_SIZE) and local_rank.
SCR_WORKER_ID_ENV = "RTP_LLM_SCR_WORKER_ID"
SCR_WORKER_NUM_ENV = "RTP_LLM_SCR_WORKER_NUM"
SCR_WORKER_OFFSET_ENV = "RTP_LLM_SCR_WORKER_OFFSET"
# These values are control-plane inputs only. RTP-LLM never invokes a
# controller or performs dump/restore; it passes the timeout budget to
# Epsilon and includes the controller generation in diagnostics. A finite
# default prevents a missing sidecar/quorum from blocking forever.
SCR_TIMEOUT_ENV = "RTP_LLM_SCR_TIMEOUT"
SCR_INACTIVITY_TIMEOUT_ENV = "RTP_LLM_SCR_INACTIVITY_TIMEOUT"
SCR_GENERATION_ENV = "RTP_LLM_SCR_GENERATION"
SCR_GENERATION_ALIAS_ENV = "SCR_GENERATION"
SCR_RESTORE_START_TIME_ENV = "RTP_LLM_SCR_RESTORE_START_EPOCH_MS"
SCR_TIMEOUT_ALIASES = (
    "RTPLLM_SCR_CHECKPOINT_TIMEOUT_S",
    "SCR_TIMEOUT",
    "EPSILON_CR_TIMEOUT",
)
SCR_INACTIVITY_TIMEOUT_ALIASES = (
    "RTPLLM_SCR_INACTIVITY_TIMEOUT_S",
    "SCR_INACTIVITY_TIMEOUT",
)
DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_INACTIVITY_TIMEOUT_SECONDS = 10
# Compatibility names used by deployment/contract tests.
DEFAULT_SCR_TIMEOUT_S = DEFAULT_TIMEOUT_SECONDS
DEFAULT_SCR_INACTIVITY_TIMEOUT_S = DEFAULT_INACTIVITY_TIMEOUT_SECONDS

def _flag(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _scr_generation() -> str:
    """Return the controller-provided generation for structured diagnostics."""

    for name in (SCR_GENERATION_ENV, SCR_GENERATION_ALIAS_ENV, "SCR_GENERATION_ID"):
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return "<unset>"


def _unified_scr_value() -> Optional[str]:
    """Return the sole RTP-LLM feature-gate value."""

    value = os.environ.get(RTPLLM_ENABLE_SCR_ENV)
    return value.strip().lower() if value is not None else None


def configure_scr_environment() -> bool:
    """Read the RTP-LLM switch before importing Epsilon.

    ``RTPLLM_ENABLE_SCR=1`` enables the RTP-LLM integration. ``SCR_ENABLE`` and
    ``SCR_PHASE`` must already have been provided by the controller/container;
    this function deliberately does not set or normalize either variable.
    """

    value = _unified_scr_value()
    if not _flag(value):
        return False

    return True


def is_scr_enabled() -> bool:
    """Return whether the RTP-LLM sCR integration was explicitly requested.

    The default is false.  This function does not import ``epsilon`` and is
    therefore safe to call before CUDA/PyTorch initialization.
    """

    return configure_scr_environment()


def is_scr_template_phase_active() -> bool:
    """Return whether external configuration selected a template lifecycle.

    The RTP-LLM integration switch alone must not delay normal startup.  Only
    the externally supplied checkpoint/restore phase enables the synchronous
    pre-service barrier and deferred listener path.
    """

    if not is_scr_enabled():
        return False
    return os.environ.get(SCR_PHASE_ENV, "").strip().lower() in {
        SCR_PHASE_CHECKPOINT,
        SCR_PHASE_RESTORE,
    }


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
        return is_scr_template_phase_active()
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
            device = getattr(value, "device", None)
            if device is not None:
                device_type = getattr(device, "type", str(device).split(":", 1)[0])
                if device_type != "cuda":
                    LOGGER.error(
                        "refusing non-CUDA KV tensor registration device=%s",
                        device,
                    )
                    return
            is_contiguous = getattr(value, "is_contiguous", None)
            if callable(is_contiguous) and not is_contiguous():
                LOGGER.error("refusing non-contiguous KV tensor registration")
                return
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


def _engine_kv_cache_tensors(engine: Any) -> tuple[Any, ...]:
    """Collect cache storage owned by the target and optional draft engines.

    Speculative decoding can construct a second C++ ``PyWrappedModel`` for an
    MTP/Eagle/DSpARK draft model. Its Python model is reachable through
    ``engine.propose_model.model`` and may own a distinct KV allocation. A
    snapshot that registers only ``engine.model`` would then restore the main
    cache while leaving the draft cache outside the optimized registration.
    Pointer de-duplication keeps shared/aliased allocations safe.
    """

    candidates: list[Any] = []
    seen_candidates: set[int] = set()

    def _append(value: Any) -> None:
        if value is None:
            return
        value_id = id(value)
        if value_id in seen_candidates:
            return
        seen_candidates.add(value_id)
        candidates.append(value)

    _append(getattr(engine, "model", None))
    _append(getattr(engine, "py_model", None))
    propose_model = getattr(engine, "propose_model", None)
    _append(propose_model)
    _append(getattr(propose_model, "model", None))

    result: list[Any] = []
    seen_tensors: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        for owner in (getattr(candidate, "py_model", None), candidate):
            kv_cache = getattr(owner, "kv_cache", None)
            for tensor in _kv_cache_tensors(kv_cache):
                key = _tensor_key(tensor)
                if key in seen_tensors:
                    continue
                seen_tensors.add(key)
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
    after_restore_result: int | None = None
    registration_duration_ms: float = 0.0


class EpsilonProtocolError(RuntimeError):
    """Raised when an Epsilon response cannot be classified safely."""


class ScrArrivalError(RuntimeError):
    """Raised when an active template participant cannot reach its barrier."""


@dataclass(frozen=True)
class EpsilonCapabilities:
    """Capabilities discovered from one loaded Epsilon implementation."""

    api_version: str
    supports_timeout: bool
    supports_inactivity_timeout: bool
    supports_kv_registration: bool
    supports_after_restore: bool
    signature_source: str


def _callable_accepts(function: Any, parameter: str) -> bool | None:
    """Inspect a Python/pybind callable without invoking it.

    ``None`` means that the implementation does not expose an inspectable
    signature.  In that case callers must use explicit capability metadata or
    choose the conservative legacy path; they must not probe by making a
    second barrier call after a TypeError.
    """

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return None
    parameters = signature.parameters.values()
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters):
        return True
    return parameter in signature.parameters


def _epsilon_capabilities(epsilon: Any) -> EpsilonCapabilities:
    """Discover Epsilon capabilities without side effects or trial calls."""

    explicit: Mapping[str, Any] = {}
    raw_capabilities = getattr(epsilon, "capabilities", None)
    try:
        if callable(raw_capabilities):
            raw_capabilities = raw_capabilities()
        if isinstance(raw_capabilities, Mapping):
            explicit = raw_capabilities
    except Exception as exc:
        LOGGER.warning("sCR Epsilon capability query failed: %s", exc)

    version = str(
        explicit.get("api_version")
        or getattr(epsilon, "API_VERSION", None)
        or getattr(epsilon, "__version__", None)
        or "unknown"
    )
    checkpoint = getattr(epsilon, "snapstart_checkpoint", None)
    timeout = _callable_accepts(checkpoint, "timeout")
    inactivity_timeout = _callable_accepts(checkpoint, "inactivity_timeout")
    if timeout is None:
        timeout = bool(explicit.get("supports_timeout", getattr(epsilon, "SUPPORTS_TIMEOUT", False)))
    if inactivity_timeout is None:
        inactivity_timeout = bool(
            explicit.get(
                "supports_inactivity_timeout",
                getattr(epsilon, "SUPPORTS_INACTIVITY_TIMEOUT", False),
            )
        )

    register_kv = getattr(epsilon, "register_kv_caches", None)
    register_restore = getattr(epsilon, "register_after_restore_func", None)
    supports_restore = _callable_accepts(register_restore, "callback")
    if supports_restore is None:
        supports_restore = bool(
            explicit.get(
                "supports_after_restore",
                getattr(epsilon, "SUPPORTS_AFTER_RESTORE", False),
            )
        )
    # The external compatibility shim currently accepts this registration but
    # deliberately does not execute the callback.
    if getattr(epsilon, "_EXTERNAL_DIR", ""):
        supports_restore = False

    return EpsilonCapabilities(
        api_version=version,
        supports_timeout=bool(timeout),
        supports_inactivity_timeout=bool(inactivity_timeout),
        supports_kv_registration=callable(register_kv),
        supports_after_restore=bool(supports_restore),
        signature_source="inspect" if checkpoint is not None else "missing",
    )


class EpsilonAdapter:
    """Version-tolerant, one-call adapter for the Epsilon Python boundary."""

    def __init__(self, epsilon: Any):
        self.epsilon = epsilon
        self.capabilities = _epsilon_capabilities(epsilon)

    def snapshot_arrival(
        self,
        *,
        worker_id: int,
        worker_num: int,
        timeout: int,
        inactivity_timeout: int,
    ) -> int | None:
        checkpoint = getattr(self.epsilon, "snapstart_checkpoint", None)
        if not callable(checkpoint):
            raise EpsilonProtocolError("Epsilon snapstart_checkpoint is unavailable")

        kwargs: dict[str, Any] = {
            "wait_mode": 1,
            "worker_id": worker_id,
            "worker_num": worker_num,
        }
        if self.capabilities.supports_timeout:
            kwargs["timeout"] = timeout
        else:
            LOGGER.warning(
                "sCR Epsilon API %s has no native timeout capability; "
                "external watchdog is required",
                self.capabilities.api_version,
            )
        if self.capabilities.supports_inactivity_timeout:
            kwargs["inactivity_timeout"] = inactivity_timeout
        return _call_result(checkpoint, **kwargs)

    def register_after_restore(self, callback: Callable[..., Any]) -> int | None:
        if not self.capabilities.supports_after_restore:
            raise EpsilonProtocolError(
                "Epsilon after-restore callback is unavailable or is a no-op"
            )
        function = getattr(self.epsilon, "register_after_restore_func", None)
        if not callable(function):
            raise EpsilonProtocolError("Epsilon after-restore API is unavailable")
        return _call_result(function, callback)


@dataclass(frozen=True)
class ScrParticipantManifest:
    """Stable full-process membership for one Epsilon scheduler scope.

    The manifest only assigns IDs.  It never invokes a controller or performs
    dump/restore; each process uses its ID for the passive Epsilon arrival.
    """

    worker_num: int
    participant_ids: TypingMapping[str, int]
    generation: str = ""

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
    *,
    generation: str | None = None,
) -> ScrParticipantManifest:
    """Assign contiguous IDs to an ordered process-role sequence."""

    mapping: dict[str, int] = {}
    for worker_id, (role, instance) in enumerate(participants):
        key = f"{role}:{instance}"
        if key in mapping:
            raise ValueError(f"duplicate sCR participant {key!r}")
        mapping[key] = worker_id
    if generation is None:
        generation = _scr_generation()
    if generation == "<unset>":
        generation = ""
    manifest = ScrParticipantManifest(len(mapping), mapping, generation.strip())
    manifest.validate()
    return manifest


_registration_lock = threading.Lock()
_registrations: dict[int, ScrRegistration] = {}
_registration_locks: dict[int, threading.Lock] = {}
_registration_engines: dict[int, Any] = {}
_before_checkpoint_hooks: set[int] = set()
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
    if torch is None or not bool(getattr(torch.cuda, "_initialized", False)):
        return

    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize(device=device)


def _prepare_cuda_for_arrival(device: Any | None = None) -> None:
    """Synchronize registered GPU state, or explicitly record a CPU skip."""

    with _registration_lock:
        gpu_state_registered = any(bool(record.tensors) for record in _registrations.values())
    torch = sys.modules.get("torch")
    cuda_initialized = bool(torch is not None and getattr(torch.cuda, "_initialized", False))
    if gpu_state_registered and not cuda_initialized:
        raise RuntimeError("GPU KV state is registered but torch CUDA is not initialized")
    if not gpu_state_registered:
        LOGGER.debug("sCR arrival CUDA prepare skipped for CPU-only participant")
        return
    _synchronize_cuda(device)


def _cuda_synchronize() -> None:
    """Best-effort callback retained for Epsilon compatibility.

    The arrival path performs a result-bearing synchronization immediately
    before calling Epsilon.  This callback is still installed for providers
    that invoke it on their own thread, but an exception is logged here because
    the legacy callback API has no failure channel.
    """

    try:
        _synchronize_cuda()
    except Exception:
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
        callback = _cuda_synchronize
    else:

        def callback() -> None:
            try:
                _synchronize_cuda(device)
            except Exception:
                LOGGER.exception("captured sCR CUDA synchronize callback failed")

    def _supplementary_callback() -> None:
        # External SCR currently invokes this callback synchronously from
        # snapstart_checkpoint.  Avoid doing the same CUDA synchronization
        # twice after the arrival path's result-bearing prepare step.  A
        # provider callback running on another thread simply performs the
        # supplementary synchronization.
        if getattr(_scr_prepare_local, "skip_next", False):
            _scr_prepare_local.skip_next = False
            return
        callback()

    return _supplementary_callback


def _scr_timeouts() -> tuple[int, int]:
    """Resolve one explicit timeout budget for every Epsilon implementation."""

    def _resolve(canonical: str, aliases: tuple[str, ...], default: int) -> int:
        # The external shim may override the Python argument with SCR_TIMEOUT.
        # Therefore all supplied timeout aliases must agree; otherwise the
        # caller cannot know which budget the provider will actually use.
        names = (canonical,) + aliases
        supplied = [(name, os.environ[name]) for name in names if name in os.environ]
        if not supplied:
            return default
        values: list[int] = []
        invalid = False
        for name, raw in supplied:
            try:
                value = int(raw)
                if value < 1:
                    raise ValueError
            except (TypeError, ValueError):
                LOGGER.error("invalid %s=%r; using default=%s", name, raw, default)
                invalid = True
                continue
            values.append(value)
        if invalid and values:
            raise EpsilonProtocolError(
                f"conflicting {canonical} timeout configuration: malformed alias"
            )
        if values and any(value != values[0] for value in values[1:]):
            raise EpsilonProtocolError(
                f"conflicting {canonical} timeout configuration: {supplied!r}"
            )
        return values[0] if values else default

    timeout = _resolve(SCR_TIMEOUT_ENV, SCR_TIMEOUT_ALIASES, DEFAULT_TIMEOUT_SECONDS)
    inactivity_timeout = _resolve(
        SCR_INACTIVITY_TIMEOUT_ENV,
        SCR_INACTIVITY_TIMEOUT_ALIASES,
        DEFAULT_INACTIVITY_TIMEOUT_SECONDS,
    )
    return timeout, inactivity_timeout


def _restore_elapsed_ms() -> float | None:
    """Return restore elapsed time when the platform exported its epoch start."""

    raw = os.environ.get(SCR_RESTORE_START_TIME_ENV)
    if not raw or not raw.strip():
        return None
    try:
        return max(0.0, time.time() * 1000.0 - float(raw))
    except (TypeError, ValueError):
        LOGGER.warning("invalid %s=%r; restore elapsed time unavailable", SCR_RESTORE_START_TIME_ENV, raw)
        return None


def register_for_scr(
    engine: Any,
    *,
    model_name: str = "",
    instance: int = 0,
    rank: int | None = None,
    local_rank: int | None = None,
    after_restore: Callable[..., Any] | None = None,
) -> bool:
    """Serialize lazy registration retries for one engine identity."""

    if engine is None or not is_scr_enabled():
        return False
    engine_key = id(engine)
    with _registration_lock:
        _registration_engines[engine_key] = engine
        registration_lock = _registration_locks.setdefault(engine_key, threading.Lock())
    with registration_lock:
        return _register_for_scr_once(
            engine,
            model_name=model_name,
            instance=instance,
            rank=rank,
            local_rank=local_rank,
            after_restore=after_restore,
        )


def _register_for_scr_once(
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
    started = time.monotonic()
    if not is_scr_enabled():
        return False

    if engine is None:
        return False

    epsilon = _load_epsilon()
    if epsilon is None or not _epsilon_is_active(epsilon):
        return False
    try:
        adapter = EpsilonAdapter(epsilon)
    except Exception:
        LOGGER.exception("sCR Epsilon capability discovery failed")
        return False

    # Registration may be retried when the cache is initialized lazily. Avoid
    # registering the Epsilon callback twice for the same engine object.
    engine_key = id(engine)
    with _registration_lock:
        previous = _registrations.get(engine_key)
        if previous is not None and previous.ok:
            return True

    tensors = _engine_kv_cache_tensors(engine)

    cache_result: int | None = None
    hook_result: int | None = None
    after_restore_result: int | None = None
    ok = True

    try:
        if tensors and adapter.capabilities.supports_kv_registration:
            cache_result = _call_result(epsilon.register_kv_caches, list(tensors))
            ok = ok and cache_result in (None, 0)
        elif not tensors:
            LOGGER.warning("sCR active but no non-empty KV-cache tensors were found")
            ok = False
        else:
            LOGGER.warning("sCR active but Epsilon KV-cache registration is unavailable")
            ok = False

        with _registration_lock:
            hook_registered = engine_key in _before_checkpoint_hooks
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
                    _before_checkpoint_hooks.add(engine_key)
            ok = ok and hook_result in (None, 0)
        else:
            LOGGER.warning("sCR active but Epsilon before-checkpoint hook is unavailable")
            ok = False
        if after_restore is not None:
            if not adapter.capabilities.supports_after_restore:
                LOGGER.warning(
                    "sCR after-restore callback requested but Epsilon does not "
                    "provide an executable register_after_restore_func"
                )
                ok = False
            else:
                callback_started = time.monotonic()

                def _restore_fixup_with_timing(*args: Any, **kwargs: Any) -> Any:
                    callback_started_at = time.monotonic()
                    LOGGER.info(
                        "sCR restore fixup callback started generation=%s phase=%s",
                        _scr_generation(),
                        os.environ.get(SCR_PHASE_ENV, "<unset>"),
                    )
                    try:
                        result = after_restore(*args, **kwargs)
                    except BaseException:
                        LOGGER.exception(
                            "sCR restore fixup callback failed generation=%s elapsed_ms=%.3f",
                            _scr_generation(),
                            (time.monotonic() - callback_started_at) * 1000.0,
                        )
                        raise
                    LOGGER.info(
                        "sCR restore fixup callback completed generation=%s elapsed_ms=%.3f",
                        _scr_generation(),
                        (time.monotonic() - callback_started_at) * 1000.0,
                    )
                    return result

                after_restore_result = _call_result(
                    adapter.register_after_restore, _restore_fixup_with_timing
                )
                ok = ok and after_restore_result in (None, 0)
                LOGGER.info(
                    "sCR restore callback registered generation=%s result=%s elapsed_ms=%.3f "
                    "restore_elapsed_ms=%s",
                    _scr_generation(),
                    after_restore_result,
                    (time.monotonic() - callback_started) * 1000.0,
                    _restore_elapsed_ms(),
                )
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
        after_restore_result=after_restore_result,
        registration_duration_ms=(time.monotonic() - started) * 1000.0,
    )
    with _registration_lock:
        _registrations[engine_key] = registration
    LOGGER.info(
        "sCR registration completed generation=%s phase=%s tensors=%d cache_result=%s "
        "hook_result=%s ok=%s api_version=%s elapsed_ms=%.3f restore_elapsed_ms=%s",
        _scr_generation(),
        os.environ.get(SCR_PHASE_ENV, "<unset>"),
        len(tensors),
        cache_result,
        hook_result,
        ok,
        adapter.capabilities.api_version,
        registration.registration_duration_ms,
        _restore_elapsed_ms(),
    )
    return ok


def _parse_scr_int_env(name: str) -> int | None:
    """Parse an optional SCR integer override without silently changing scope.

    A malformed worker mapping is more dangerous than a disabled optimization:
    one rank can occupy another rank's Epsilon slot and leave the scheduler
    quorum waiting forever. Callers therefore treat a malformed value as a
    mapping error and skip arrival for this process.
    """

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def resolve_scr_worker_mapping(
    *, local_rank: int, worker_id: int | None = None, worker_num: int | None = None
) -> tuple[int, int]:
    """Resolve and validate a rank's Epsilon wait-mode mapping.

    The default scope is the current Pod: ``worker_num`` comes from
    ``LOCAL_WORLD_SIZE`` and the ID is ``local_rank``. Deployments sharing one
    scheduler may set ``RTP_LLM_SCR_WORKER_OFFSET`` (or an explicit per-process
    ``RTP_LLM_SCR_WORKER_ID``) and ``RTP_LLM_SCR_WORKER_NUM`` to describe the
    complete participant scope. In the full-process mode the launcher supplies
    an explicit mapping for CPU/frontend participants too; the default local
    mapping remains appropriate only when the scheduler scope is GPU-rank-only.
    """

    try:
        local_rank = int(local_rank)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"local_rank must be an integer, got {local_rank!r}") from exc
    if local_rank < 0:
        raise ValueError(f"local_rank must be non-negative, got {local_rank}")

    if worker_num is None:
        env_worker_num = _parse_scr_int_env(SCR_WORKER_NUM_ENV)
        if env_worker_num is not None:
            worker_num = env_worker_num
        else:
            local_world_size = _parse_scr_int_env("LOCAL_WORLD_SIZE")
            worker_num = local_world_size if local_world_size is not None else 1
    try:
        worker_num = int(worker_num)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"worker_num must be an integer, got {worker_num!r}") from exc
    if worker_num <= 0:
        raise ValueError(f"worker_num must be positive, got {worker_num}")

    if worker_id is None:
        env_worker_id = _parse_scr_int_env(SCR_WORKER_ID_ENV)
        if env_worker_id is not None:
            worker_id = env_worker_id
        else:
            offset = _parse_scr_int_env(SCR_WORKER_OFFSET_ENV)
            worker_id = (offset if offset is not None else 0) + local_rank
    try:
        worker_id = int(worker_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"worker_id must be an integer, got {worker_id!r}") from exc

    if not 0 <= worker_id < worker_num:
        raise ValueError(
            "worker_id must be in [0, worker_num), "
            f"got worker_id={worker_id}, worker_num={worker_num}"
        )
    return worker_id, worker_num


def arrive_scr_checkpoint_barrier(
    *,
    worker_id: int,
    worker_num: int,
    timeout: int | None = None,
    inactivity_timeout: int | None = None,
    generation: str | None = None,
    fail_closed: bool = False,
) -> int | None:
    """Arrive at Epsilon's rank-local snapshot barrier.

    This is the one active-looking call that remains in RTP-LLM. It is not a
    controller operation: ``scr_controller`` still initiates ``check`` /
    ``block`` / ``dump`` / ``restore`` from the control plane. The native
    Epsilon call lets each template participant announce that its process
    state is at a safe point and then wait for the controller-driven snapshot
    lifecycle. GPU-only KV registration happens separately.

    Every participating rank must call this once per snapshot generation with
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
    if not _external_shim_phase_active():
        LOGGER.info(
            "sCR external shim inactive for phase=%s; skipping arrival",
            os.environ.get(SCR_PHASE_ENV, "<unset>"),
        )
        return _raise_if_strict("sCR arrival requested in an inactive phase")

    started = time.monotonic()
    actual_generation = _scr_generation()
    if generation and generation != actual_generation:
        LOGGER.error(
            "sCR generation mismatch expected=%s actual=%s worker_id=%s worker_num=%s",
            generation,
            actual_generation,
            worker_id,
            worker_num,
        )
        return _raise_if_strict(
            f"sCR generation mismatch expected={generation} actual={actual_generation}"
        )
    try:
        default_timeout, default_inactivity = _scr_timeouts()
    except EpsilonProtocolError as exc:
        LOGGER.error(
            "sCR timeout configuration conflict generation=%s: %s",
            _scr_generation(),
            exc,
        )
        return _raise_if_strict(
            f"sCR timeout configuration conflict: {exc}", exc
        )
    try:
        resolved_timeout = int(timeout) if timeout is not None else default_timeout
        resolved_inactivity = (
            int(inactivity_timeout)
            if inactivity_timeout is not None
            else default_inactivity
        )
        # The external shim gives SCR_TIMEOUT precedence over the function
        # argument. Refuse an explicit disagreement rather than logging one
        # value while the provider waits with another.
        timeout_names = (SCR_TIMEOUT_ENV,) + SCR_TIMEOUT_ALIASES
        if timeout is not None and any(
            name in os.environ for name in timeout_names
        ) and resolved_timeout != default_timeout:
            raise EpsilonProtocolError(
                "explicit timeout disagrees with configured SCR timeout"
            )
        inactivity_names = (SCR_INACTIVITY_TIMEOUT_ENV,) + SCR_INACTIVITY_TIMEOUT_ALIASES
        if inactivity_timeout is not None and any(
            name in os.environ for name in inactivity_names
        ) and resolved_inactivity != default_inactivity:
            raise EpsilonProtocolError(
                "explicit inactivity timeout disagrees with configured SCR timeout"
            )
    except EpsilonProtocolError as exc:
        LOGGER.error(
            "invalid explicit sCR timeout configuration timeout=%r "
            "inactivity_timeout=%r generation=%s: %s; skipping arrival",
            timeout,
            inactivity_timeout,
            _scr_generation(),
            exc,
        )
        return _raise_if_strict(
            f"invalid explicit sCR timeout configuration: {exc}", exc
        )
    except (TypeError, ValueError):
        LOGGER.error(
            "invalid explicit sCR timeout configuration timeout=%r "
            "inactivity_timeout=%r generation=%s; skipping arrival",
            timeout,
            inactivity_timeout,
            _scr_generation(),
        )
        return _raise_if_strict("invalid explicit sCR timeout configuration")
    if resolved_timeout <= 0 or resolved_inactivity <= 0:
        LOGGER.error(
            "invalid sCR timeout configuration timeout=%s inactivity_timeout=%s "
            "generation=%s; skipping arrival",
            resolved_timeout,
            resolved_inactivity,
            _scr_generation(),
        )
        return _raise_if_strict(
            f"invalid sCR timeout configuration timeout={resolved_timeout} "
            f"inactivity_timeout={resolved_inactivity}"
        )

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

    epsilon = _load_epsilon()
    if epsilon is None or not _epsilon_is_active(epsilon):
        return _raise_if_strict("sCR Epsilon provider is unavailable or inactive")
    try:
        adapter = EpsilonAdapter(epsilon)
    except Exception as exc:
        LOGGER.exception("sCR Epsilon capability discovery failed")
        return _raise_if_strict("sCR Epsilon capability discovery failed", exc)
    if not callable(getattr(epsilon, "snapstart_checkpoint", None)):
        LOGGER.warning("sCR active but Epsilon snapshot barrier is unavailable")
        return _raise_if_strict("sCR snapshot barrier API is unavailable")

    try:
        LOGGER.info(
            "sCR snapshot arrival started generation=%s phase=%s worker_id=%d "
            "worker_num=%d timeout_s=%d inactivity_timeout_s=%d "
            "restore_elapsed_ms=%s",
            _scr_generation(),
            os.environ.get(SCR_PHASE_ENV, "<unset>"),
            worker_id,
            worker_num,
            resolved_timeout,
            resolved_inactivity,
            _restore_elapsed_ms(),
        )
        prepare_device = _capture_cuda_device()
        try:
            _prepare_cuda_for_arrival(prepare_device)
        except Exception:
            LOGGER.exception(
                "sCR CUDA prepare failed; refusing snapshot arrival generation=%s "
                "worker_id=%d worker_num=%d",
                _scr_generation(),
                worker_id,
                worker_num,
            )
            return _raise_if_strict("sCR CUDA preparation failed")
        # The external shim invokes its callback synchronously in this same
        # thread. Mark the successful preparation so the compatibility callback
        # does not perform an unnecessary second synchronize.
        _scr_prepare_local.skip_next = True
        try:
            result = adapter.snapshot_arrival(
                worker_id=worker_id,
                worker_num=worker_num,
                timeout=resolved_timeout,
                inactivity_timeout=resolved_inactivity,
            )
        finally:
            # Do not let an exception or a provider that invokes the callback
            # asynchronously suppress synchronization on the next arrival.
            _scr_prepare_local.skip_next = False
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if result not in (None, 0):
            LOGGER.error(
                "sCR snapshot arrival returned non-zero generation=%s phase=%s "
                "worker_id=%d worker_num=%d result=%s elapsed_ms=%.3f "
                "restore_elapsed_ms=%s",
                _scr_generation(),
                os.environ.get(SCR_PHASE_ENV, "<unset>"),
                worker_id,
                worker_num,
                result,
                elapsed_ms,
                _restore_elapsed_ms(),
            )
            if fail_closed:
                raise ScrArrivalError(f"sCR snapshot arrival returned status {result!r}")
        elif elapsed_ms >= resolved_timeout * 1000.0:
            LOGGER.error(
                "sCR snapshot arrival exceeded timeout generation=%s phase=%s "
                "worker_id=%d worker_num=%d elapsed_ms=%.3f timeout_s=%d result=%s "
                "restore_elapsed_ms=%s",
                _scr_generation(),
                os.environ.get(SCR_PHASE_ENV, "<unset>"),
                worker_id,
                worker_num,
                elapsed_ms,
                resolved_timeout,
                result,
                _restore_elapsed_ms(),
            )
            if fail_closed:
                raise ScrArrivalError(
                    "sCR snapshot arrival exceeded timeout "
                    f"generation={_scr_generation()} worker_id={worker_id} "
                    f"worker_num={worker_num} elapsed_ms={elapsed_ms:.3f}"
                )
        else:
            LOGGER.info(
                "sCR snapstart checkpoint reached pid=%d generation=%s phase=%s "
                "worker_id=%d worker_num=%d result=%s elapsed_ms=%.3f "
                "restore_elapsed_ms=%s",
                os.getpid(),
                _scr_generation(),
                os.environ.get(SCR_PHASE_ENV, "<unset>"),
                worker_id,
                worker_num,
                result,
                elapsed_ms,
                _restore_elapsed_ms(),
            )
        return result
    except Exception as exc:
        # The barrier is optional. A timeout or an unavailable sidecar must
        # not take down a serving rank; the control plane can use a fallback.
        LOGGER.exception(
            "sCR snapshot barrier arrival failed generation=%s phase=%s "
            "worker_id=%d worker_num=%d elapsed_ms=%.3f timeout_s=%d "
            "restore_elapsed_ms=%s",
            _scr_generation(),
            os.environ.get(SCR_PHASE_ENV, "<unset>"),
            worker_id,
            worker_num,
            (time.monotonic() - started) * 1000.0,
            resolved_timeout,
            _restore_elapsed_ms(),
        )
        if fail_closed:
            raise ScrArrivalError(
                "sCR snapshot barrier arrival failed "
                f"generation={_scr_generation()} worker_id={worker_id} worker_num={worker_num}: {exc}"
            ) from exc
        return None


def arrive_scr_template_barrier(
    *,
    worker_id: int,
    worker_num: int,
    timeout: int | None = None,
    inactivity_timeout: int | None = None,
    generation: str | None = None,
    fail_closed: bool = False,
) -> int | None:
    """Run lifecycle hooks and Epsilon arrival as one template barrier."""

    if not is_scr_template_phase_active():
        return arrive_scr_checkpoint_barrier(
            worker_id=worker_id,
            worker_num=worker_num,
            timeout=timeout,
            inactivity_timeout=inactivity_timeout,
            generation=generation,
            fail_closed=fail_closed,
        )

    actual_generation = generation or _scr_generation()
    if actual_generation == "<unset>":
        actual_generation = ""
    phase = os.environ.get(SCR_PHASE_ENV, "").strip().lower()
    lifecycle = get_template_lifecycle()
    lifecycle.prepare_for_template(actual_generation, phase)
    try:
        result = arrive_scr_checkpoint_barrier(
            worker_id=worker_id,
            worker_num=worker_num,
            timeout=timeout,
            inactivity_timeout=inactivity_timeout,
            generation=generation,
            fail_closed=fail_closed,
        )
        lifecycle.restore_fixup(actual_generation)
        lifecycle.release_template(actual_generation)
        return result
    except BaseException:
        lifecycle.abort_template(actual_generation)
        raise


def start_scr_checkpoint_arrival_thread(
    *,
    worker_id: int,
    worker_num: int,
    timeout: int | None = None,
    inactivity_timeout: int | None = None,
    generation: str | None = None,
    name: str = "scr-checkpoint-arrival",
) -> threading.Thread | None:
    """Compatibility helper for non-startup callers.

    This helper must not be used for a template participant's startup path:
    the daemon would let the process create listeners or serve while it is
    waiting at the barrier. Startup code must call
    :func:`arrive_scr_checkpoint_barrier` synchronously instead. It remains
    exported only for compatibility with isolated tests/legacy callers.
    """

    if not is_scr_enabled():
        return None
    if not _external_shim_phase_active():
        LOGGER.info(
            "sCR external shim inactive for phase=%s; skipping arrival thread",
            os.environ.get(SCR_PHASE_ENV, "<unset>"),
        )
        return None

    def _arrive() -> None:
        try:
            result = arrive_scr_checkpoint_barrier(
                worker_id=worker_id,
                worker_num=worker_num,
                timeout=timeout,
                inactivity_timeout=inactivity_timeout,
                generation=generation,
            )
            if result is None:
                # None means the optional integration was inactive or could
                # not reach Epsilon. This is fail-open for normal serving, but
                # the control plane must treat it as a missing quorum member.
                LOGGER.warning(
                    "sCR snapshot arrival did not complete "
                    "generation=%s worker_id=%s worker_num=%s phase=%s",
                    _scr_generation(),
                    worker_id,
                    worker_num,
                    os.environ.get(SCR_PHASE_ENV, ""),
                )
            elif result != 0:
                LOGGER.error(
                    "sCR snapshot arrival returned non-zero result=%s "
                    "generation=%s worker_id=%s worker_num=%s phase=%s",
                    result,
                    _scr_generation(),
                    worker_id,
                    worker_num,
                    os.environ.get(SCR_PHASE_ENV, ""),
                )
            else:
                LOGGER.info(
                    "sCR snapshot arrival completed "
                    "generation=%s worker_id=%s worker_num=%s phase=%s",
                    _scr_generation(),
                    worker_id,
                    worker_num,
                    os.environ.get(SCR_PHASE_ENV, ""),
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
        _registration_locks.clear()
        _registration_engines.clear()
        _before_checkpoint_hooks.clear()
    if hasattr(_scr_prepare_local, "skip_next"):
        del _scr_prepare_local.skip_next


__all__ = [
    "EpsilonAdapter",
    "EpsilonCapabilities",
    "EpsilonProtocolError",
    "ScrArrivalError",
    "ScrParticipantManifest",
    "ScrRegistration",
    "arrive_scr_checkpoint_barrier",
    "arrive_scr_template_barrier",
    "build_scr_participant_manifest",
    "epsilon_backend_mode",
    "configure_scr_environment",
    "is_scr_enabled",
    "is_scr_template_phase_active",
    "register_for_scr",
    "resolve_scr_worker_mapping",
]
