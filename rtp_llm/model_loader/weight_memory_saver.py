"""WeightMemorySaver: register model-weight GPU allocations as pausable memory.

Wraps ``torch_memory_saver`` so that every CUDA allocation that holds model
weights is registered under the ``tag="weights"`` region (via
:func:`weights_region`) with ``enable_cpu_backup`` selected by the startup sleep
level. Registering the region keeps the virtual addresses stable across a later
pause/resume (data_ptr must not change because CUDA graphs and the C++
``weights_`` aliases bake pointers in).

The actual pause (back weights up / release physical GPU pages) and resume
(remap physical pages at the same VA) are driven from the C++ sleep controller
through ``VmmBackend::pause/resume("weights")`` (see
KVCachePhysicalMemoryController.cc), NOT from this module. This module's job is
purely to *scope* the allocations at load time; there is intentionally no Python
pause/resume entry point.

Activation
----------
Disabled by default. Enable by setting the environment variable
``ENABLE_SLEEP_MODE=1`` (or programmatically from the parsed runtime config)
and having ``torch_memory_saver`` importable (typically via its LD_PRELOAD
hook shim). ``RTP_LLM_WEIGHT_MEMORY_SAVER=1`` is kept as a
low-level developer override for isolated memory-saver tests. When the switch
is off or the package is unavailable, every API in this module degrades to a
no-op so production startup paths are unaffected.

Coverage checklist (weight tensors that must land inside ``weights_region``)
----------------------------------------------------------------------------
- [covered] Main ``ModelWeights`` incl. quantization scales/zeros:
  ``ModelLoader.load_weights`` (loader.py) keeps checkpoint-reader temporary
  buffers outside the TMS region, and wraps dynamic weights (lm_head etc.) plus
  static EPLB init (``_init_eplb_weight``).
  The fastsafetensors iterator enters ``weights_region`` only after
  ``ParallelLoader`` / ``LoadWithShm`` construction so persistent tensors are
  tagged without registering pinned staging buffers. ``WeightModule.load`` /
  ``WeightModule.update`` (weight_module.py) wrap the final ``.to(device)``
  landing point for every atomic/composite/quant weight regardless of caller.
- [covered] Multimodal ViT (``mm_part``): ``BaseMultiModalMixin.__init__``
  (rtp_llm/multimodal/multimodal_mixins/base_multimodal_mixin.py) wraps both
  the module construction on device and the checkpoint weight load
  (``MultimodalMixinLoader.load_weights`` + ``load_mm_weight``).
- [covered] Static LoRA (merge_lora): merged into the main weights during
  ``ModelLoader.load_weights`` — same region.
- [covered] Dynamic LoRA adapters: ``LoraManager.add_lora``
  (rtp_llm/lora/lora_manager.py) wraps the host->device upload performed by
  the C++ ``add_lora`` (LD_PRELOAD hook intercepts the in-thread C++
  cudaMalloc).
- [covered] Draft / MTP propose models: loaded through the same
  ``BaseModel.load -> ModelLoader.load_weights`` path as the main model.
- [left-for-integration] Dynamic EPLB weight relayout: Python
  ``ExpertBalancer.load_moe_weight`` only produces CPU tensors; the GPU-side
  expert buffers are (re)allocated inside the C++ engine (EPLB plan buffers).
  Needs the C++-side region hook to be integrated.
- [left-for-integration] Runtime in-place weight update (``WeightManager``)
  copies into already-registered tensors (no new GPU allocation), but any
  future reallocation there must also be wrapped.

Threading note: ``torch_memory_saver.region`` toggles a *thread-local*
"interesting region" flag, so the region only captures allocations made on
the entering thread (including synchronous C++ calls issued from it).
"""

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

ENV_SWITCH: str = "ENABLE_SLEEP_MODE"
ENV_LEVEL: str = "SLEEP_MODE_LEVEL"
LEGACY_ENV_SWITCH: str = "RTP_LLM_WEIGHT_MEMORY_SAVER"
WEIGHTS_TAG: str = "weights"

_lock = threading.RLock()
_tms: Optional[Any] = None
_import_attempted: bool = False
_enabled_override: Optional[bool] = None
_level_override: Optional[int] = None
_region_depth = threading.local()
_region_suppressed = threading.local()


def configure_from_runtime(
    enable_sleep_mode: bool, sleep_mode_level: Optional[int] = None
) -> None:
    """Mirror parsed RuntimeConfig sleep fields into this Python helper.

    CLI arguments in RTP-LLM are bound to config objects and are not written
    back into os.environ. Weight allocation happens in Python before the C++
    sleep controller is exercised, so this explicit override keeps
    ``--enable-sleep-mode`` / ``--sleep-mode-level`` and the corresponding env
    vars equivalent. ``sleep_mode_level`` selects whether the weights region is
    opened with host cpu_backup (level 1) or as discard-only (level 2); it is
    frozen at allocation time by torch_memory_saver, so it cannot change per
    /sleep request.
    """
    global _enabled_override, _level_override, _tms, _import_attempted
    with _lock:
        _enabled_override = bool(enable_sleep_mode)
        if sleep_mode_level is not None:
            _level_override = int(sleep_mode_level)
        if not _enabled_override:
            _tms = None
            _import_attempted = False


def is_enabled() -> bool:
    """Whether the feature switch env var is on (does not check importability)."""
    if _enabled_override is not None:
        return _enabled_override
    return (
        os.environ.get(ENV_SWITCH, "0") == "1"
        or os.environ.get(LEGACY_ENV_SWITCH, "0") == "1"
    )


def sleep_mode_level() -> int:
    """Startup-selected sleep level for this process (1 = host backup, 2 = discard).

    Reads the explicit override first (set via :func:`configure_from_runtime`),
    then the ``SLEEP_MODE_LEVEL`` env var (mirrored from the parsed runtime
    config in server_args), defaulting to 1.
    """
    if _level_override is not None:
        return _level_override
    try:
        return int(os.environ.get(ENV_LEVEL, "1"))
    except (TypeError, ValueError):
        return 1


def _get_tms() -> Optional[Any]:
    """Lazily import and cache the torch_memory_saver singleton.

    Returns None when the switch is off or the package is unavailable.
    """
    global _tms, _import_attempted
    if not is_enabled():
        return None
    with _lock:
        if _import_attempted:
            return _tms
        _import_attempted = True
        try:
            from torch_memory_saver import (  # type: ignore[import-not-found]
                torch_memory_saver,
            )

            _tms = torch_memory_saver
            logging.info(
                "WeightMemorySaver enabled: torch_memory_saver available, "
                f"weights will be registered under tag={WEIGHTS_TAG!r} with cpu backup"
            )
        except Exception:
            _tms = None
            logging.warning(
                f"WeightMemorySaver: {ENV_SWITCH}=1 or {LEGACY_ENV_SWITCH}=1 but torch_memory_saver is not "
                "importable; weight memory pause/resume degrades to no-op",
                exc_info=True,
            )
        return _tms


def is_available() -> bool:
    """True only when the env switch is on and torch_memory_saver is importable."""
    return _get_tms() is not None


@contextmanager
def configure_subprocess() -> Iterator[None]:
    """Inject torch_memory_saver into child processes only when sleep mode is on.

    This mirrors torch_memory_saver's own subprocess helper while preserving
    the no-op behavior used by normal startup paths. The parent process keeps
    its environment unchanged after the child has been spawned.
    """
    if not is_enabled():
        yield
        return

    try:
        from torch_memory_saver import (
            configure_subprocess as tms_configure_subprocess,  # type: ignore[import-not-found]
        )
    except Exception:
        logging.warning(
            f"WeightMemorySaver: {ENV_SWITCH}=1 or {LEGACY_ENV_SWITCH}=1 but "
            "torch_memory_saver.configure_subprocess is not importable; "
            "subprocess starts without memory saver preload",
            exc_info=True,
        )
        yield
        return

    with tms_configure_subprocess():
        yield


def start_configured_process(process: Any) -> None:
    """Start a child process with weight memory saver preload when required."""
    with configure_subprocess():
        process.start()


@contextmanager
def weights_region() -> Iterator[None]:
    """Context manager registering CUDA allocations as pausable weight memory.

    Equivalent to ``tms.region(tag="weights", enable_cpu_backup=True)`` when
    the saver is available, ``nullcontext()`` otherwise. Re-entrant: nested
    uses on the same thread enter the underlying region only once.
    """
    # Explicitly suppressed on this thread (e.g. level-2 wake reload): the
    # resident weights already occupy their VA, so nothing allocated now should
    # join the region -- see suppress_weights_region().
    if getattr(_region_suppressed, "value", False):
        yield
        return

    tms = _get_tms()
    if tms is None:
        yield
        return

    depth: int = getattr(_region_depth, "value", 0)
    if depth > 0:
        _region_depth.value = depth + 1
        try:
            yield
        finally:
            _region_depth.value = getattr(_region_depth, "value", 1) - 1
        return

    # Drop cached allocator blocks so weight tensors cannot be served from
    # physically-backed cache blocks allocated *before* this region (those
    # would escape torch_memory_saver tracking).
    try:
        import torch

        # no-op when CUDA is not initialized yet
        torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - defensive, torch is a hard dep
        pass

    # Level 1 backs weights up to pinned host on pause (fast wake, holds host
    # RAM). Level 2 opens the region without host backup: pause frees GPU without
    # a host copy and resume remaps blank pages at the same VA; the model loader
    # then reloads the original checkpoint into those live tensors in place.
    # tms freezes this choice at allocation time, hence it is a startup-level knob.
    enable_cpu_backup = sleep_mode_level() != 2
    _region_depth.value = 1
    try:
        with tms.region(tag=WEIGHTS_TAG, enable_cpu_backup=enable_cpu_backup):
            yield
    finally:
        _region_depth.value = 0


@contextmanager
def suppress_weights_region() -> Iterator[None]:
    """Force every ``weights_region()`` on this thread to become a nullcontext.

    Used by the level-2 wake reload. At wake the resident weight tensors already
    occupy their fixed VA (remapped in place by torch_memory_saver ``resume``);
    the reload only streams transient sources and ``copy_``-s them into those
    live tensors. ``WeightModule.load`` (and the fastsafetensors iterator) would
    otherwise allocate every raw read / dequant / TP-split / final ``.to(device)``
    intermediate INSIDE the weights region -- committing them as region-backed,
    cpu-backup pages that ``empty_cache`` cannot return to the driver. Those stick
    around (and *grow with weight count*), then starve the following KV-cache
    ``resume`` -> ``cu_mem_create`` OOM. Suppressing the region keeps them as plain
    torch allocations, freed per-tensor in the reload loop. This is the
    scratch-path analogue of ``prepare_weights_fastsafetensor(in_weights_region=
    False)`` and also covers that path belt-and-suspenders.
    """
    prev = getattr(_region_suppressed, "value", False)
    _region_suppressed.value = True
    try:
        yield
    finally:
        _region_suppressed.value = prev


def _reset_for_testing() -> None:
    """Reset module-level caches/state. Test-only helper."""
    global _tms, _import_attempted, _enabled_override, _level_override
    with _lock:
        _tms = None
        _import_attempted = False
        _enabled_override = None
        # Reset the startup level override too: it is a process-level singleton, so
        # leaving it set would leak level-2 into a following test's weights_region().
        _level_override = None
    _region_depth.value = 0
    _region_suppressed.value = False
