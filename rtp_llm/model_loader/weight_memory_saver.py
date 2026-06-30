"""WeightMemorySaver (Sleep/wake_up M6): weight GPU memory pause/resume with CPU backup.

Wraps ``torch_memory_saver`` so that every CUDA allocation that holds model
weights is registered under the ``tag="weights"`` region with
``enable_cpu_backup=True``. On engine sleep, :func:`pause_weights` backs the
weight pages up to host (pinned) memory and releases the physical GPU pages
while keeping the virtual addresses stable (constraint C2: data_ptr must not
change because CUDA graphs and the C++ ``weights_`` aliases bake pointers in).
On wake_up, :func:`resume_weights` remaps physical pages at the same VA and
copies the content back (constraint C4: weight content must be preserved).

Activation
----------
Disabled by default. Enable by setting the environment variable
``ENABLE_SLEEP_MODE=1`` (or programmatically from the parsed runtime config)
and having ``torch_memory_saver`` importable (typically via its LD_PRELOAD
hook shim, see spike S1). ``RTP_LLM_WEIGHT_MEMORY_SAVER=1`` is kept as a
low-level developer override for isolated memory-saver tests. When the switch
is off or the package is unavailable, every API in this module degrades to a
no-op so production startup paths are unaffected.

Coverage checklist (weight tensors that must land inside ``weights_region``)
----------------------------------------------------------------------------
- [covered] Main ``ModelWeights`` incl. quantization scales/zeros:
  ``ModelLoader.load_weights`` (loader.py) wraps ft-style / fastsafetensors /
  scratch loading, dynamic weights (lm_head etc.) and static EPLB init
  (``_init_eplb_weight``); ``WeightModule.load`` / ``WeightModule.update``
  (weight_module.py) wrap the final ``.to(device)`` landing point for every
  atomic/composite/quant weight regardless of caller.
- [covered] Multimodal ViT (``mm_part``): ``BaseMultiModalMixin.__init__``
  (rtp_llm/multimodal/multimodal_mixins/base_multimodal_mixin.py) wraps both
  the module construction on device and the checkpoint weight load
  (``MultimodalMixinLoader.load_weights`` + ``load_mm_weight``).
- [covered] Static LoRA (merge_lora): merged into the main weights during
  ``ModelLoader.load_weights`` — same region.
- [covered] Dynamic LoRA adapters: ``LoraManager.add_lora``
  (rtp_llm/lora/lora_manager.py) wraps the host->device upload performed by
  the C++ ``add_lora`` (LD_PRELOAD hook intercepts the in-thread C++
  cudaMalloc, validated by spike S1 "cpp" scenario).
- [covered] Draft / MTP propose models: loaded through the same
  ``BaseModel.load -> ModelLoader.load_weights`` path as the main model.
- [left-for-integration] Dynamic EPLB weight relayout: Python
  ``ExpertBalancer.load_moe_weight`` only produces CPU tensors; the GPU-side
  expert buffers are (re)allocated inside the C++ engine (EPLB plan buffers).
  Needs the C++-side region hook from M5/M7 integration.
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
LEGACY_ENV_SWITCH: str = "RTP_LLM_WEIGHT_MEMORY_SAVER"
WEIGHTS_TAG: str = "weights"

_lock = threading.RLock()
_tms: Optional[Any] = None
_import_attempted: bool = False
_paused: bool = False
_enabled_override: Optional[bool] = None
_region_depth = threading.local()


def configure_from_runtime(enable_sleep_mode: bool) -> None:
    """Mirror parsed RuntimeConfig.enable_sleep_mode into this Python helper.

    CLI arguments in RTP-LLM are bound to config objects and are not written
    back into os.environ. Weight allocation happens in Python before the C++
    sleep controller is exercised, so this explicit override keeps
    ``--enable-sleep-mode`` and ``ENABLE_SLEEP_MODE=1`` equivalent.
    """
    global _enabled_override, _tms, _import_attempted, _paused
    with _lock:
        _enabled_override = bool(enable_sleep_mode)
        if not _enabled_override:
            _tms = None
            _import_attempted = False
            _paused = False


def is_enabled() -> bool:
    """Whether the feature switch env var is on (does not check importability)."""
    if _enabled_override is not None:
        return _enabled_override
    return (
        os.environ.get(ENV_SWITCH, "0") == "1"
        or os.environ.get(LEGACY_ENV_SWITCH, "0") == "1"
    )


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


def is_paused() -> bool:
    """Whether the weights region is currently paused (physical pages released)."""
    return _paused


@contextmanager
def weights_region() -> Iterator[None]:
    """Context manager registering CUDA allocations as pausable weight memory.

    Equivalent to ``tms.region(tag="weights", enable_cpu_backup=True)`` when
    the saver is available, ``nullcontext()`` otherwise. Re-entrant: nested
    uses on the same thread enter the underlying region only once.
    """
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
    # would escape torch_memory_saver tracking; see spike S1 notes).
    try:
        import torch

        # no-op when CUDA is not initialized yet
        torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - defensive, torch is a hard dep
        pass

    _region_depth.value = 1
    try:
        with tms.region(tag=WEIGHTS_TAG, enable_cpu_backup=True):
            yield
    finally:
        _region_depth.value = 0


def pause_weights() -> bool:
    """Backup weights to host and release physical GPU pages (VA preserved).

    Returns True if the weights are paused after the call. No-op (warning,
    returns False) when the saver is unavailable; idempotent when already
    paused. Intended to be called from the M1 sleep sequence *after* the KV
    cache pause.
    """
    global _paused
    tms = _get_tms()
    if tms is None:
        logging.warning(
            "WeightMemorySaver.pause_weights: saver unavailable "
            f"(enabled={is_enabled()}), skip pausing weight memory"
        )
        return False
    with _lock:
        if _paused:
            logging.info("WeightMemorySaver.pause_weights: already paused, skip")
            return True
        tms.pause(WEIGHTS_TAG)
        _paused = True
        logging.info("WeightMemorySaver: weights paused (cpu backup, VA preserved)")
        return True


def resume_weights() -> bool:
    """Remap physical pages at the same VA and copy weight content back.

    Returns True if the weights are resumed (not paused) after the call.
    No-op (warning, returns False) when the saver is unavailable; idempotent
    when not paused. Intended to be called from the M1 wake_up sequence
    *after* the KV cache physical memory is remapped.
    """
    global _paused
    tms = _get_tms()
    if tms is None:
        logging.warning(
            "WeightMemorySaver.resume_weights: saver unavailable "
            f"(enabled={is_enabled()}), skip resuming weight memory"
        )
        return False
    with _lock:
        if not _paused:
            logging.info("WeightMemorySaver.resume_weights: not paused, skip")
            return True
        tms.resume(WEIGHTS_TAG)
        _paused = False
        logging.info("WeightMemorySaver: weights resumed (content restored)")
        return True


def _reset_for_testing() -> None:
    """Reset module-level caches/state. Test-only helper."""
    global _tms, _import_attempted, _paused, _enabled_override
    with _lock:
        _tms = None
        _import_attempted = False
        _paused = False
        _enabled_override = None
    _region_depth.value = 0
