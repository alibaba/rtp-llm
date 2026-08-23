"""WeightMemorySaver: weight GPU memory pause/resume with CPU backup.

Wraps ``torch_memory_saver`` so that every CUDA allocation that holds model
weights is registered under the ``tag="weights"`` region with
``enable_cpu_backup=True``. On engine sleep, :func:`pause_weights` backs the
weight pages up to host (pinned) memory and releases the physical GPU pages
while keeping the virtual addresses stable (data_ptr must not change because
CUDA graphs and the C++ ``weights_`` aliases bake pointers in). On wake_up,
:func:`resume_weights` remaps physical pages at the same VA and copies the
content back so weight values are preserved.

Activation
----------
Disabled by default. Enable by setting the environment variable
``ENABLE_SLEEP_MODE=1`` (or programmatically from the parsed runtime config)
and having ``torch_memory_saver`` importable (typically via its LD_PRELOAD
hook shim). ``RTP_LLM_WEIGHT_MEMORY_SAVER=1`` is kept as a
low-level developer override for isolated memory-saver tests. When the switch
is off or the package is unavailable, every API in this module degrades to a
no-op so production startup paths are unaffected.

``expandable_segments:True`` can be requested alongside sleep mode, but it is
kept OFF through the whole init path (weight load + KV arena allocation + KV MR
registration) so those land at low, RDMA-registerable virtual addresses, and is
turned on only after the engine is ready so runtime forward buffers still get
the fragmentation benefit -- see :func:`_prepare_expandable_coexistence` and
:func:`enable_runtime_expandable`.

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
ENV_COLLECTIVE_RELEASE: str = "SLEEP_RELEASE_COLLECTIVE_MEMORY"
LEGACY_ENV_SWITCH: str = "RTP_LLM_WEIGHT_MEMORY_SAVER"
WEIGHTS_TAG: str = "weights"

_lock = threading.RLock()
_tms: Optional[Any] = None
_import_attempted: bool = False
_paused: bool = False
_enabled_override: Optional[bool] = None
_level_override: Optional[int] = None
_collective_release_override: Optional[bool] = None
_region_depth = threading.local()
_region_suppressed = threading.local()
_model_scope = threading.local()
# Process-global mirror of the active build scope. DSV4's py-model construction
# dispatches the actual module __init__ (MegaMoEStrategy / CompressorFP8, which
# self-register into the global registries) onto a WORKER thread -- the JIT
# warmup runs on a "Dummy-N" thread, not the MainThread that opened
# model_build_scope. A threading.local scope set on MainThread is therefore
# invisible on that worker, so the strategy stamps None and the level-2 wake
# re-derive filter (_owned) drops every mega/compressor -> blank kernel weights
# -> garbage output after wake. Model builds are strictly sequential (the main
# model's _create_python_model fully returns, joining its workers, before the
# MTP draft is built), so a single process-global unambiguously identifies the
# model under construction even across threads. current_model_scope() prefers
# the threadlocal (exact, when build and registration share a thread) and falls
# back to this global (cross-thread build).
_model_scope_global: Any = None

# Expandable-segments coexistence (see _prepare_expandable_coexistence).
_EXPANDABLE_KEY: str = "expandable_segments"
_expandable_prepared: bool = False
# The user requested expandable_segments:True, but we defer turning it on until
# after startup (see _prepare_expandable_coexistence / enable_runtime_expandable).
_expandable_requested: bool = False
# expandable_segments is currently live (only true after enable_runtime_expandable).
_expandable_active: bool = False
# The *live* torch caching-allocator expandable_segments state, mirrored from
# every _set_expandable_segments() call. Distinct from _expandable_active (which
# latches True for the whole runtime): this tracks the instantaneous setting,
# so it correctly reads False inside an expandable_segments_disabled() block.
# assert_pausable_alloc_safe() reads it to catch a sleep-persistent allocation
# about to happen while expandable is on.
_expandable_live: bool = False
# The user's PYTORCH_CUDA_ALLOC_CONF minus expandable_segments, replayed on every
# live-config write (see _capture_base_alloc_conf).
_expandable_base_conf: str = ""
_base_conf_captured: bool = False

# Init-phase segment-split cap (see limit_init_segment_splitting).
_SPLIT_CAP_KEY: str = "max_split_size_mb"
# Above torch's 20 MiB kLargeBuffer floor (smaller values are rejected) and below
# the loader's ~1 GiB staging buffers, which is the point: cap only the blocks big
# enough to strand hundreds of MiB when a resident weight splits one.
_INIT_SPLIT_CAP_MB: int = 256
_split_cap_live: bool = False


@contextmanager
def model_build_scope(token: Any) -> Iterator[None]:
    """Mark the model whose py-model modules are being constructed on this thread.

    DSV4 keeps *global* WeakSet registries of live Mega-MoE strategies and
    attention compressors (see mega_buf.py / compressor.py) so the sleep reclaim
    and the level-2 wake re-derive can reach every live instance. When a
    checkpoint-backed propose/draft model (e.g. DSV4 MTP) coexists with the main
    model, both register into the SAME registries, and their layer ids collide
    (the MTP draft is ``num_layers=1`` -> ``layer_id=0``, same as the main
    model's layer 0). The level-2 wake reload keys re-derivation by ``layer_id``,
    so without attribution the main reload would grab the MTP strategy for
    layer 0 (or vice versa) and corrupt both.

    This context manager stamps the active model's ``token`` onto every
    strategy/compressor registered while it is open, so each model's
    :class:`WeightManager` can filter the global registries down to its own
    instances. Re-entrant / nestable; restores the previous scope on exit.
    ``token`` is typically ``id(base_model)`` — stable and distinct for the two
    live models over the process lifetime.
    """
    global _model_scope_global
    prev = getattr(_model_scope, "value", None)
    prev_global = _model_scope_global
    _model_scope.value = token
    _model_scope_global = token
    try:
        yield
    finally:
        _model_scope.value = prev
        _model_scope_global = prev_global


def current_model_scope() -> Any:
    """Token of the model currently being built (None if outside a
    :func:`model_build_scope`). Stamped onto DSV4 registry entries at
    registration time so the level-2 wake reload can attribute them per model.

    Prefers the thread-local scope (exact when the module __init__ that registers
    runs on the same thread that opened the scope), falling back to the
    process-global mirror. The fallback is required because DSV4 constructs its
    Mega-MoE strategies / compressors on a worker thread while the MainThread
    holds the scope open; without it every registration stamps None and the
    level-2 wake re-derive silently drops all computed weights. See
    ``_model_scope_global``."""
    local = getattr(_model_scope, "value", None)
    if local is not None:
        return local
    return _model_scope_global


def configure_from_runtime(
    enable_sleep_mode: bool,
    sleep_mode_level: Optional[int] = None,
    release_collective_memory: Optional[bool] = None,
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
    global _enabled_override, _level_override, _collective_release_override
    global _tms, _import_attempted, _paused
    with _lock:
        _enabled_override = bool(enable_sleep_mode)
        if sleep_mode_level is not None:
            _level_override = int(sleep_mode_level)
        if release_collective_memory is not None:
            _collective_release_override = bool(release_collective_memory)
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


def release_collective_memory() -> bool:
    """Whether sleep should also release NCCL communicator GPU memory.

    Independent of the sleep level: the level selects what happens to the
    *weights* (host backup vs discard-and-reload), whereas this selects whether
    the communicator's transport buffers are handed back too. It is a separate
    switch because it carries costs the level does not -- pinned host memory
    equal to the GPU bytes released, a few seconds on each of sleep and wake, and
    a runtime NCCL new enough to expose ``ncclCommSuspend`` -- so a deployment
    that wants level 2 does not implicitly opt into those.

    Defaults to on. The feature is fail-closed when the runtime NCCL does not
    expose the suspend/resume API, and an explicit ``0`` still disables it for
    deployments that do not want the pinned-host-memory/latency trade-off.
    """
    if _collective_release_override is not None:
        return _collective_release_override
    return os.environ.get(ENV_COLLECTIVE_RELEASE, "1") == "1"


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


def is_paused() -> bool:
    """Whether the weights region is currently paused (physical pages released)."""
    return _paused


def _alloc_conf_without_expandable(conf: str) -> str:
    """Drop the ``expandable_segments`` key from a PYTORCH_CUDA_ALLOC_CONF string,
    preserving all other comma-separated ``key:value`` settings."""
    kept = [
        part.strip()
        for part in conf.split(",")
        if part.strip() and not part.strip().startswith(_EXPANDABLE_KEY + ":")
    ]
    return ",".join(kept)


def _capture_base_alloc_conf() -> None:
    """Snapshot the user's ``PYTORCH_CUDA_ALLOC_CONF`` (minus expandable_segments) once.

    The live setter *replaces* the whole allocator config rather than merging into
    it, so anything the user asked for in the env var has to be replayed on every
    write or it would be silently reverted to the torch default.
    """
    global _base_conf_captured, _expandable_base_conf
    if _base_conf_captured:
        return
    _base_conf_captured = True
    _expandable_base_conf = _alloc_conf_without_expandable(
        os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    )


def _apply_live_alloc_conf(expandable: bool, split_cap: bool) -> None:
    """Push the composed caching-allocator config to the live allocator.

    Both knobs we flip at runtime -- ``expandable_segments`` and the init-phase
    ``max_split_size_mb`` cap -- share one setter that replaces the entire config,
    so each call restates both plus the captured env base.

    Applies to future segments/allocations only; existing segments keep their
    nature. Prefers the current ``torch._C._accelerator_setAllocatorSettings`` and
    falls back to the deprecated ``torch.cuda.memory._set_allocator_settings``.
    """
    import torch

    _capture_base_alloc_conf()
    parts = [_expandable_base_conf] if _expandable_base_conf else []
    parts.append(f"{_EXPANDABLE_KEY}:{'True' if expandable else 'False'}")
    if split_cap:
        parts.append(f"{_SPLIT_CAP_KEY}:{_INIT_SPLIT_CAP_MB}")
    full = ",".join(parts)
    setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if setter is not None:
        setter(full)
    else:
        torch.cuda.memory._set_allocator_settings(full)


def _set_expandable_segments(enabled: bool) -> None:
    """Flip the *live* torch caching-allocator ``expandable_segments`` setting,
    keeping the init-phase split cap and any other allocator settings intact."""
    _apply_live_alloc_conf(enabled, _split_cap_live)
    # Mirror the applied setting so assert_pausable_alloc_safe() has a reliable
    # live view (only reached on setter success; a raising setter leaves the
    # previous value, which matches the un-applied reality).
    global _expandable_live
    _expandable_live = enabled


def _prepare_expandable_coexistence() -> None:
    """Let ``expandable_segments:True`` coexist with the torch_memory_saver pool.

    torch_memory_saver routes weight allocations through a private
    ``torch.cuda.MemPool`` backed by a CUDA-VMM pluggable allocator so pages can
    be unmapped on sleep. That pool is incompatible with expandable segments
    (pytorch/pytorch#147851), and torch_memory_saver enforces it by *raising* at
    init when ``PYTORCH_CUDA_ALLOC_CONF`` requests expandable segments -- forcing
    the whole process to choose one or the other.

    Following vllm-project/vllm#40812 we strip ``expandable_segments`` out of the
    env var so the torch_memory_saver sanity check passes, but -- unlike vllm --
    we do NOT re-apply it live yet. Enabling expandable during init lets torch's
    cuMem segments reserve the low virtual-address range while weights load;
    the (correctly non-expandable) torch_memory_saver KV arena is then pushed to
    a high VA where a plain nv_peer_mem ``ibv_reg_mr`` (PD/RDMA cache-store MR
    registration, no dmabuf) EFAULTs. So we keep expandable OFF through the whole
    init path -- weight load, KV arena allocation, and KV MR registration all
    land at low, registerable VA -- and only turn it on afterwards via
    :func:`enable_runtime_expandable`, once no more RDMA-registered buffers are
    allocated. Runtime/forward activation buffers then still get the
    fragmentation benefit. Runs once per process; a no-op unless expandable
    segments were actually requested.

    Workaround note: this deferral exists only because expandable segments and
    the torch_memory_saver MemPool cannot currently coexist for RDMA-registered
    VMM memory. If a future torch supports expandable inside a pluggable MemPool
    (pytorch/pytorch#147851), the deferral can be dropped and expandable applied
    live from here.
    """
    global _expandable_prepared, _expandable_requested
    if _expandable_prepared:
        return
    _expandable_prepared = True

    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    _capture_base_alloc_conf()
    if f"{_EXPANDABLE_KEY}:True" not in conf:
        return

    # Remove it from the env so torch_memory_saver._sanity_checks() (which reads
    # the env var, not the live setting) does not refuse to initialize, and so
    # torch does not pick it up when the caching allocator initializes.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _expandable_base_conf
    _expandable_requested = True
    # Actively force the live setting OFF as well: if the caching allocator has
    # already parsed the env (e.g. an earlier torch.cuda call latched
    # expandable_segments:True), stripping the env alone would not undo it. This
    # makes the init phase non-expandable regardless of allocator-init timing.
    try:
        _set_expandable_segments(False)
    except Exception:
        # Setter unavailable this early: the env strip above still prevents the
        # allocator from turning expandable on when it first parses the config.
        logging.warning(
            "WeightMemorySaver: could not force expandable_segments off at init; "
            "relying on the stripped env var instead",
            exc_info=True,
        )
    logging.info(
        "WeightMemorySaver: expandable_segments requested alongside sleep mode; "
        "deferred until after startup so weights/KV land at low registerable VA "
        "(see pytorch/pytorch#147851, vllm-project/vllm#40812)"
    )


def prepare_expandable_coexistence() -> None:
    """Public entry to normalize expandable_segments before CUDA initializes.

    Call once at process start (before any ``torch.cuda`` allocation) so the env
    var is stripped and expandable is forced off ahead of the caching-allocator
    init. Idempotent and a no-op unless ``expandable_segments:True`` was requested
    with sleep mode. ``weights_region`` also calls the internal helper, so this
    early call only tightens the timing; it is safe to skip on paths that do not
    touch CUDA early."""
    if not is_enabled():
        return
    _prepare_expandable_coexistence()


def enable_runtime_expandable() -> None:
    """Turn expandable_segments on for the remaining (runtime) allocations.

    Call once the engine is ready -- after weights are loaded and the KV cache
    arena has been allocated and RDMA-registered. Applies to future segments
    only, so the already-resident weights and KV arena keep their low-VA
    non-expandable form (and stay registerable), while per-forward activation
    buffers allocated from here on get the expandable fragmentation benefit.

    No-op unless ``expandable_segments:True`` was requested and successfully
    deferred by :func:`_prepare_expandable_coexistence`.
    """
    global _expandable_active
    if not _expandable_requested or _expandable_active:
        return
    try:
        _set_expandable_segments(True)
        _expandable_active = True
        logging.info(
            "WeightMemorySaver: enabled expandable_segments live for runtime "
            "allocations (engine ready; weights + KV already registered)"
        )
    except Exception:
        # Setter unavailable: keep running with expandable off (safe -- matches
        # the init state) rather than leave a half-applied setting.
        logging.warning(
            "WeightMemorySaver: could not enable expandable_segments post-startup; "
            "continuing with expandable_segments disabled",
            exc_info=True,
        )


def limit_init_segment_splitting() -> None:
    """Stop init-phase transients from becoming permanently-stranded segments.

    THE PROBLEM. Weight loading allocates and frees ~1 GiB GPU staging buffers
    (fastsafetensors batch copy on the loader's producer/consumer threads); torch
    keeps each freed buffer as a cached ~1 GiB *segment*. A later request that is
    smaller than the segment is then served by *splitting* it. When the request is
    a resident tensor -- a 512 MiB weight shard from ``WeightModule._split``, or one
    of the small permanently-live tensors built afterwards (packed FP8 scales,
    stacked wo_a, rope cos/sin cache, the graph-baked capture input) -- the segment
    is pinned by a live block for the process lifetime and the rest of it is free
    forever. ``empty_cache()`` cannot help: it only returns 100%-free segments, so
    one live block in a 1010 MiB segment costs the other 498 MiB. Measured on DSV4
    L2 sleep, this stranded ~1.5 GiB/rank -- roughly half the entire sleeping
    residual that was not CUDA context or NCCL.

    THE FIX. Cap ``max_split_size_mb`` for the init window only. Blocks above the
    cap are never split, so the big staging carcasses stay 100% free and the
    ``empty_cache()`` the loader already runs actually returns them, while resident
    tensors get their own right-sized segments. Restored by
    :func:`release_init_segment_splitting` once the engine is ready, so per-forward
    and KV allocations keep torch's default splitting behaviour -- which is why this
    is scoped rather than set through the env var: as a process-global setting it
    would trade permanent serving-time fragmentation for a sleep-only gain.

    Call before any weight allocation. No-op unless sleep mode is on (the gain only
    exists at sleep, and the non-sleep load path stays byte-for-byte unchanged).
    """
    global _split_cap_live
    if not is_enabled() or _split_cap_live:
        return
    try:
        _apply_live_alloc_conf(_expandable_live, True)
    except Exception:
        # Setter unavailable: keep torch's default splitting. Costs the sleeping
        # residual gain, breaks nothing.
        logging.warning(
            "WeightMemorySaver: could not cap max_split_size_mb for init; "
            "sleeping residual will keep the stranded load-time segments",
            exc_info=True,
        )
        return
    _split_cap_live = True
    logging.info(
        "WeightMemorySaver: capped max_split_size_mb=%d for the init phase so "
        "load-time staging segments are not split by resident weights",
        _INIT_SPLIT_CAP_MB,
    )


def release_init_segment_splitting() -> None:
    """Restore torch's default segment splitting once the engine is ready.

    Pairs with :func:`limit_init_segment_splitting`; call after weights, KV arena
    and graph capture are done, alongside :func:`enable_runtime_expandable`. The
    cap applies to future allocations only, so the init-phase segments keep their
    unsplit shape while runtime allocations go back to default behaviour.
    """
    global _split_cap_live
    if not _split_cap_live:
        return
    try:
        _apply_live_alloc_conf(_expandable_live, False)
    except Exception:
        # Leave the cap live rather than a half-applied config: it is a
        # fragmentation trade-off at worst, not a correctness problem.
        logging.warning(
            "WeightMemorySaver: could not restore default max_split_size_mb; "
            "runtime allocations keep the init-phase cap",
            exc_info=True,
        )
        return
    _split_cap_live = False
    logging.info(
        "WeightMemorySaver: restored default max_split_size_mb for runtime "
        "allocations (engine ready)"
    )


@contextmanager
def expandable_segments_disabled() -> Iterator[None]:
    """Disable expandable segments for the duration of a torch_memory_saver pool
    allocation, restoring the previous setting on exit. No-op unless coexistence
    is active (see :func:`_prepare_expandable_coexistence`).

    Used both for weight-region allocations (keep weights non-expandable) and by
    the level-2 wake reload (:meth:`WeightManager.reload_weights_from_loader`),
    which streams checkpoint transients and re-derives computed weights: those
    must NOT land in expandable segments, because an expandable-segment tensor
    read/written across the torch_memory_saver pause/resume boundary comes back
    with corrupted contents (silent -- coverage counts stay correct but the
    values are wrong -> garbage post-wake output). Forcing the whole wake weight
    path non-expandable matches the verified-correct expandable-off wake while
    leaving runtime forward buffers expandable for the fragmentation benefit."""
    if not _expandable_active:
        yield
        return
    _set_expandable_segments(False)
    try:
        yield
    finally:
        _set_expandable_segments(True)


def assert_pausable_alloc_safe(where: str) -> None:
    """Fail loudly if a sleep-persistent allocation is about to happen while
    expandable_segments is live.

    WHY (this is the corruption the guard exists to prevent): torch_memory_saver
    sleep unmaps the physical pages of the ``weights`` region and remaps fresh
    pages at the same virtual address on wake. A tensor that lives in a torch
    *expandable* cuMem segment and is read or written across that pause/resume
    boundary comes back with the WRONG bytes -- and does so SILENTLY: the
    tensor's address, shape, dtype and any coverage/count logs all stay correct,
    only the values are corrupt. The failure therefore does not surface at
    allocation, at sleep, or at wake; it surfaces much later as garbage model
    output, with nothing in between to point at the cause. (Observed on DSV4 L2
    wake: reload ``copy_`` sources and re-derive scratch had landed in expandable
    segments -> post-wake output was garbled while every count said 1480/1480 OK.)

    Any allocation whose contents must survive a sleep -- weight-region tensors,
    and the transient sources/scratch the wake reload copies FROM -- must be made
    with expandable off (see :func:`expandable_segments_disabled`). This guard
    turns that latent, near-undebuggable data corruption into an immediate,
    located ``RuntimeError`` at the offending allocation site. No-op unless
    expandable coexistence is active.
    """
    if _expandable_live:
        raise RuntimeError(
            f"[WeightMemorySaver] refusing a sleep-persistent allocation at {where!r} "
            "while expandable_segments is live. Tensors read/written across the "
            "torch_memory_saver pause/resume boundary in expandable segments come "
            "back silently corrupted (correct shape/counts, wrong values -> garbage "
            "post-wake output). Wrap the allocation in expandable_segments_disabled()."
        )


@contextmanager
def weights_region() -> Iterator[None]:
    """Context manager registering CUDA allocations as pausable weight memory.

    Equivalent to ``tms.region(tag="weights", enable_cpu_backup=True)`` when
    the saver is available, ``nullcontext()`` otherwise. Re-entrant: nested
    uses on the same thread enter the underlying region only once.
    """
    # Explicitly suppressed on this thread (e.g. level-2 wake reload): the
    # resident weights already occupy their VA, so nothing allocated now should
    # join the region -- see suppress_weights_region(). Suppression bypasses the
    # expandable_segments_disabled() guard below, so these transient allocations
    # are only safe if the CALLER already disabled expandable (the wake reload
    # does). Assert it: allocating reload scratch here with expandable live would
    # silently corrupt it across the resume boundary (see assert_pausable_alloc_safe).
    if getattr(_region_suppressed, "value", False):
        assert_pausable_alloc_safe("weights_region(suppressed)")
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

    # Normalize expandable_segments before torch_memory_saver initializes on the
    # first tms.region() below (must run before its env-var sanity check).
    _prepare_expandable_coexistence()

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
    # a host copy and resume remaps blank pages at the same VA; the sleep hooks
    # dump/reload the weights via a local-disk raw backup. tms freezes this
    # choice at allocation time, hence it is a startup-level knob.
    enable_cpu_backup = sleep_mode_level() != 2
    _region_depth.value = 1
    try:
        # Keep weights non-expandable. During init this is already the case
        # (expandable is deferred, see _prepare_expandable_coexistence); this
        # guard matters only for weights allocated after enable_runtime_expandable
        # (e.g. dynamic LoRA), which must stay out of expandable segments.
        with expandable_segments_disabled():
            # Self-check: the disable above must have actually taken effect
            # (a silently-failed setter, or enable_runtime_expandable() firing too
            # early in a future regression, would leave expandable live and let
            # these weight pages corrupt across the first sleep/wake).
            assert_pausable_alloc_safe("weights_region")
            with tms.region(tag=WEIGHTS_TAG, enable_cpu_backup=enable_cpu_backup):
                yield
    finally:
        _region_depth.value = 0


def pausable_empty(*args, **kwargs):
    """``torch.empty`` whose result joins the pausable weights region.

    The returned tensor is VMM-unmapped on sleep (``pause("weights")``) and
    remapped at the same VA on wake, exactly like a model weight -- so a
    persistent runtime workspace can be reclaimed at sleep without any
    destroy/recreate, registry, or hot-path ``None`` juggling.

    Use for PERSISTENT buffers only, and call it ONLY on the allocation
    (cache-miss) path -- never per-forward. ``weights_region()`` runs
    ``empty_cache()`` on entry, so wrapping a per-call fast path would empty the
    caching allocator on every forward. A throwaway temporary allocated here
    would also be trapped in the private weights MemPool (which ``empty_cache``
    cannot return to the driver), so keep the scope to the buffer itself.
    """
    import torch

    with weights_region():
        return torch.empty(*args, **kwargs)


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


def pause_weights() -> bool:
    """Backup weights to host and release physical GPU pages (VA preserved).

    Returns True if the weights are paused after the call. No-op (warning,
    returns False) when the saver is unavailable; idempotent when already
    paused. Intended to be called from the sleep sequence *after* the KV
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
    when not paused. Intended to be called from the wake_up sequence *after*
    the KV cache physical memory is remapped.
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
    global _level_override, _collective_release_override
    global _expandable_prepared, _expandable_requested, _expandable_active
    global _expandable_base_conf, _expandable_live
    global _base_conf_captured, _split_cap_live
    with _lock:
        _tms = None
        _import_attempted = False
        _paused = False
        _enabled_override = None
        # Reset the level and collective-release overrides too: leaving them set
        # leaks a previous test's configure_from_runtime into the next one, which
        # silently changes the level a later test believes it is exercising.
        _level_override = None
        _collective_release_override = None
        _expandable_prepared = False
        _expandable_requested = False
        _expandable_active = False
        _expandable_live = False
        _expandable_base_conf = ""
        _base_conf_captured = False
        _split_cap_live = False
    _region_depth.value = 0
    # nccl_memory latches a suspended set, a poison flag and a vote sequence that
    # outlive a single test just as stubbornly as the overrides above. Imported
    # lazily and guarded because it pulls in torch.distributed, which must not
    # become a hard dependency of resetting this module.
    try:
        from rtp_llm.utils.nccl_memory import _reset_for_testing as _reset_nccl

        _reset_nccl()
    except Exception:  # noqa: BLE001 - test helper must not fail on import order
        pass
