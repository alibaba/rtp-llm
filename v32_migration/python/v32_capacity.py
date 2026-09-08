"""v32_capacity.py — round 13: single-wave scoring over a global indexer pool.

Every decode row's indexer-K is replicated into a self-managed GPU pool
(admission bulk copy + per-step 132B appends, watermark-tracked). On steps
where the batch is stable, fresh and fully pooled, the native scoring wave is
skipped entirely: one fused (deep_gemm) call over (pool, ibt) yields top-2048
for all rows; offloaded rows then only need build/miss-fetch/write-back (C++).
Any uncertainty (new rows, width change, identity tripwire, pool exhaustion)
falls back to the r27 dual-wave path for that step.
"""

import logging
import os
import threading
import time

import torch

try:
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "v32_ctx", os.path.join(os.path.dirname(__file__), "v32_ctx.so")
    )
    _ctx = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_ctx)
    _ctx.ctx_init()
except Exception:
    _ctx = None
    logging.exception("[v32_capacity] v32_ctx.so missing — capacity disabled")

BS = 64
CHUNK = int(os.environ.get("V32_MIRROR_CHUNK", "4096"))
LAG = 256
STG_BLOCKS = int(os.environ.get("RTP_KV_OFFLOAD_STAGING_BLOCKS", "32"))
LOSSY = os.environ.get("V32_LOSSY", "0") == "1"
LOSSY_PREFETCH = int(os.environ.get("V32_LOSSY_PREFETCH", "8"))
# Scheme C: lossless offload. Reuses the lossy registration and host mirror but
# replaces drop-on-miss with a token-granular gather on the compute stream, so
# attention receives exactly the no-offload top-k.
LOSSLESS = os.environ.get("V32_LOSSLESS", "0") == "1" and hasattr(
    _ctx, "ctx_lossless_serve"
)
LOSSLESS_DIAG_MODE = int(os.environ.get("V32_LOSSLESS_DIAG_MODE", "0"))
if LOSSLESS_DIAG_MODE not in (0, 1, 2):
    raise ValueError("V32_LOSSLESS_DIAG_MODE must be 0, 1, or 2")
if LOSSLESS:
    LOSSY = True

# Optional control file holding "B0", "B8" or "C". Lets one server instance serve
# every scheme in an A/B/C sweep instead of reloading 700GB of weights per config,
# which also removes cross-instance variance from the comparison. Only switch
# between requests: a live request's hot-pool aliases would otherwise name staging
# slots the other scheme has already overwritten.
MODE_FILE = os.environ.get("V32_MODE_FILE", "")
_mode = {"prefetch": LOSSY_PREFETCH, "lossless": LOSSLESS, "mtime": -1.0}


def _refresh_mode():
    if not MODE_FILE:
        return
    try:
        m = os.stat(MODE_FILE).st_mtime
    except OSError:
        return
    if m == _mode["mtime"]:
        return
    _mode["mtime"] = m
    try:
        with open(MODE_FILE) as f:
            tag = f.read().strip().upper()
    except OSError:
        return
    if tag == "C" and hasattr(_ctx, "ctx_lossless_serve"):
        _mode.update(prefetch=0, lossless=True)
    elif tag.startswith("B"):
        _mode.update(prefetch=int(tag[1:] or 0), lossless=False)
    else:
        logging.warning(f"[v32_mode] ignoring unknown mode {tag!r}")
        return
    logging.warning(
        f"[v32_mode] scheme={tag} prefetch={_mode['prefetch']} "
        f"lossless={_mode['lossless']}"
    )
    # Counters and timers are cumulative; without this a sweep would report each
    # scheme blended with the ones before it.
    if hasattr(_ctx, "ctx_lossy_reset_counters"):
        _ctx.ctx_lossy_reset_counters()
    for k in _prof:
        _prof[k] = 0
    for k in _stats:
        _stats[k] = 0


_MIRROR_BLOCKS = os.environ.get("V32_MIRROR_BLOCKS", "1") == "1" and hasattr(
    _ctx, "ctx_mirror_blocks_d2h"
)
HOST_BUCKET = int(os.environ.get("V32_HOST_BUCKET", "8192"))
PREWARM_TOKENS = int(os.environ.get("V32_PREWARM_TOKENS", "73728"))
PREWARM_LAYERS = int(os.environ.get("V32_PREWARM_LAYERS", "61"))
ENGINE_ADMISSION_MIRROR = int(os.environ.get("RTP_KV_ADMIT_RING_BLOCKS", "0")) > 0

_hostpool = {}  # bucketed rows -> [pinned tensors]; pinning costs ~1.5 GB/s
_hostpool_lock = threading.Lock()


def _alloc_host(cap_tokens):
    rows = -(-cap_tokens // HOST_BUCKET) * HOST_BUCKET
    with _hostpool_lock:
        free = _hostpool.get(rows)
        if free:
            return free.pop()
    return torch.empty((rows, 576), dtype=torch.bfloat16, device="cpu", pin_memory=True)


def _free_host(t):
    if t is None:
        return
    with _hostpool_lock:
        _hostpool.setdefault(t.shape[0], []).append(t)


def _prewarm_host():
    held = [_alloc_host(PREWARM_TOKENS) for _ in range(PREWARM_LAYERS)]
    for t in held:
        _free_host(t)
    logging.warning(
        f"[v32_capacity] prewarmed {len(held)} host mirrors of "
        f"{held[0].shape[0] if held else 0} tokens"
    )


if PREWARM_TOKENS > 0 and not ENGINE_ADMISSION_MIRROR:
    threading.Thread(target=_prewarm_host, daemon=True).start()

STEP_TRACE = int(os.environ.get("V32_STEP_TRACE", "0"))
_trace = {"t": None, "d": []}
_plancache = {"step": -1, "min_seq": -1, "b": -1, "plan": []}
_store = {}
_admission_generation = {}  # block0 key -> current engine mirror generation
_view_cache = {}  # layer -> (main_ptr, main_flat)
_diag = {}  # request key -> "on" once offload engaged, else first step seen
# Selection trace: separates "wrong selections" from "wrong data". Two runs of the
# same scheme on the same prompt should produce identical per-step totals; if they
# do and the text still differs, the fault is in what we feed attention, not in
# which tokens we picked. Costs a device sync per layer, so it is opt-in.
SEL_TRACE = int(os.environ.get("V32_SEL_TRACE", "0"))
# Block 0's physical id is recycled, so the request key repeats: budget the trace
# per engagement generation instead, which is what actually separates requests.
_selt = {"step": -1, "acc": 0, "pre": 0, "valid": 0, "gen": 0, "kvlen": 0, "left": 0}
_gen = {"n": 0}


def _sel_trace(key, kernel_topk, row_i, kvlen, phase):
    """Checksum the selection row before ("pre") and after ("post") serve.

    Only the pre-serve sum is required to be reproducible: it is the scoring
    output. The post-serve sum encodes scratch slot numbers, which come from an
    atomicAdd and are legitimately run-dependent, so a difference there means
    nothing on its own.
    """
    if phase == "pre" and _selt["step"] != _step:
        if _selt["step"] >= 0 and _selt["left"] > 0:
            _selt["left"] -= 1
            logging.warning(
                f"[v32_seltrace] gen={_selt['gen']} n={SEL_TRACE - _selt['left']} "
                f"kvlen={_selt['kvlen']} valid={_selt['valid']} "
                f"pre={_selt['pre']} post={_selt['acc']}"
            )
        if _selt["gen"] != _gen["n"]:
            _selt["gen"], _selt["left"] = _gen["n"], SEL_TRACE
        _selt.update(step=_step, acc=0, valid=0, pre=0, kvlen=kvlen)
    if _selt["left"] <= 0:
        return
    sel = kernel_topk[row_i].reshape(-1).to(torch.int64)
    # weight by position so a reordering is not mistaken for an identical set
    idx = torch.arange(1, sel.numel() + 1, device=sel.device, dtype=torch.int64)
    h = int(((sel + 2) * idx).sum())
    if phase == "pre":
        _selt["pre"] += h
    else:
        _selt["acc"] += h
        _selt["valid"] += int((sel >= 0).sum())


_lossy_meta = {}  # key -> (jpos cpu i32, sb cpu i32): staging layout, layer-invariant
_step = 0
_stream = None
_stepcache = {"step": -1}
_batch_gate = {
    "native_only": False,
    "reason": "uninitialized",
    "native_steps": 0,
    "managed_steps": 0,
    "fallback_steps": 0,
}
# "bail" counts steps where our wave stood down. After the prefix is offloaded
# that hands scoring to the engine's native wave, which reads freed indexer-K, so
# a non-zero bail on an offloaded request means wrong tokens.
_stats = {
    "serves": 0,
    "errors": 0,
    "single": 0,
    "dual": 0,
    "bail": 0,
    "unsafe": 0,
    "fill": 0,
}
_prof = {
    "book": 0.0,
    "mirror": 0.0,
    "score": 0.0,
    "serve": 0.0,
    "proc": 0.0,
    "alloc": 0.0,
    "lreg": 0.0,
    "adopt": 0.0,
    "hostalloc": 0.0,
    "n": 0,
}


def _side_stream():
    global _stream
    if _stream is None:
        _stream = torch.cuda.Stream()
    return _stream


# Engine-owned independent indexer-K pool (V32_INDEPENDENT_IDX_POOL=1 on the
# engine side). When active, native scoring already covers the full sequence
# because the dsa_indexer_k group is never freed by the offload path, so the
# python scoring wave, the shadow indexer pool and the indexer host mirror are
# all unnecessary; only the main-KV mirror + lossless fetch/remap remain.
_engine_idx = {"v": None}


def engine_idx_active(kv_cache=None):
    v = _engine_idx["v"]
    if v is None:
        if kv_cache is None:
            return False
        p = getattr(kv_cache, "indexer_cache_base", None)
        v = p is not None and p.numel() > 0
        _engine_idx["v"] = v
    return v


def _resident_batch_reason(attention_inputs, min_seq):
    if attention_inputs is None or bool(getattr(attention_inputs, "is_prefill", False)):
        return None
    host_kbt = getattr(attention_inputs, "kv_cache_kernel_block_id", None)
    if host_kbt is not None and host_kbt.numel() > 0 and not host_kbt.is_cuda:
        if host_kbt.dim() == 2 and int(host_kbt.shape[1]) * BS < min_seq:
            return f"block_capacity={int(host_kbt.shape[1]) * BS}"
    lengths = getattr(attention_inputs, "sequence_lengths", None)
    if lengths is None or lengths.numel() == 0 or lengths.is_cuda:
        return None
    max_seq = int(lengths.max().item())
    return f"max_seq={max_seq}" if max_seq < min_seq else None


def _seed_managed_metadata(attention_inputs):
    host_kbt = getattr(attention_inputs, "kv_cache_kernel_block_id", None)
    lengths = getattr(attention_inputs, "sequence_lengths", None)
    required_width = 2 + STG_BLOCKS
    if (
        host_kbt is None
        or lengths is None
        or host_kbt.is_cuda
        or lengths.is_cuda
        or host_kbt.dim() != 2
        or int(host_kbt.shape[1]) < required_width
        or int(host_kbt.shape[0]) != int(lengths.numel())
    ):
        return False
    _stepcache.clear()
    _stepcache.update(
        step=-1,
        kvlens=[int(value) for value in lengths.tolist()],
        khead=host_kbt[:, :required_width].clone(),
        fresh_step=_step,
    )
    _plancache["step"] = -1
    _faststep.update(step=-1, armed=False)
    _fast_key["key"] = None
    return True


def begin_batch_step(kv_cache, attention_inputs, min_seq):
    global _step
    if int(kv_cache.layer_id) != 0:
        return _batch_gate["native_only"]
    previous_native = _batch_gate["native_only"]
    _batch_gate.update(native_only=False, reason="managed")
    if bool(getattr(attention_inputs, "is_prefill", False)):
        return False
    _step += 1
    _refresh_mode()
    if _step % 1000 == 0:
        _purge()
    try:
        reason = _resident_batch_reason(attention_inputs, min_seq)
    except Exception:
        logging.exception(
            "[v32_batch_gate] resident check failed at step=%d; keeping fail-closed managed path",
            _step,
        )
        _batch_gate["managed_steps"] += 1
        return False
    if reason is None:
        _batch_gate["managed_steps"] += 1
        if previous_native or "khead" not in _stepcache:
            if not _seed_managed_metadata(attention_inputs):
                raise RuntimeError(
                    f"managed transition at step {_step} lacks safe host metadata"
                )
        if previous_native:
            logging.warning(
                "[v32_batch_gate] managed step=%d native=%d managed=%d",
                _step,
                _batch_gate["native_steps"],
                _batch_gate["managed_steps"],
            )
        return False
    _batch_gate.update(native_only=True, reason=reason)
    _batch_gate["native_steps"] += 1
    if not previous_native:
        _faststep.update(step=-1, armed=False)
        _fast_key["key"] = None
        _plancache["step"] = -1
        logging.warning(
            "[v32_batch_gate] native-only step=%d reason=%s native=%d managed=%d",
            _step,
            reason,
            _batch_gate["native_steps"],
            _batch_gate["managed_steps"],
        )
    return True


def degrade_resident_batch_to_native(attention_inputs, min_seq, error):
    try:
        reason = _resident_batch_reason(attention_inputs, min_seq)
    except Exception:
        reason = None
    if reason is None:
        return False
    _batch_gate.update(native_only=True, reason=f"fallback:{reason}")
    _batch_gate["fallback_steps"] += 1
    logging.error(
        "[v32_batch_gate] resident bookkeeping failed at step=%d; degrading current batch "
        "to native reason=%s error=%r",
        _step,
        reason,
        error,
    )
    return True


def pre_topk(iop, q_fp8, weights, kv_cache, fmha_params, attention_inputs):
    """Layer-0 step bookkeeping. Always returns None: with the engine-owned
    independent indexer-K pool (auto-declared whenever offload is enabled) the
    native scoring wave is correct for offloaded and resident requests alike,
    so the python scoring wave and its shadow pool are gone (HANDOFF §11)."""
    if _ctx is None:
        return None
    kbt = getattr(attention_inputs, "kv_cache_kernel_block_id_device", None)
    kvlen_d = fmha_params.kvlen_d
    if kbt is None or q_fp8.shape[0] != kbt.shape[0]:
        return None
    layer_id = int(kv_cache.layer_id)
    if layer_id == 0:
        if STEP_TRACE:
            now = time.perf_counter()
            if _trace["t"] is not None and int(kvlen_d.max()) > 16384:
                p = _trace.get("p") or (0.0,) * 6
                cur = (
                    _prof["mirror"],
                    _prof["serve"],
                    _prof["proc"],
                    _prof["alloc"],
                    _prof["adopt"],
                    _prof["hostalloc"],
                )
                _trace["d"].append(
                    f"{(now - _trace['t']) * 1000:.0f}("
                    + "/".join(f"{(cur[k] - p[k]) * 1e3:.0f}" for k in range(6))
                    + ")"
                )
                _trace["p"] = cur
                if len(_trace["d"]) >= STEP_TRACE:
                    logging.warning(
                        "[v32_steptrace] total(mirror/serve/proc/alloc/adopt/hostalloc) "
                        + " ".join(_trace["d"])
                    )
                    _trace["d"] = []
            else:
                _trace["p"] = (
                    _prof["mirror"],
                    _prof["serve"],
                    _prof["proc"],
                    _prof["alloc"],
                    _prof["adopt"],
                    _prof["hostalloc"],
                )
            _trace["t"] = now
        _bookkeep(kbt, kvlen_d)
        engine_idx_active(kv_cache)  # detect once; consumed by the serve path
        if _step % 500 == 0:
            logging.warning(
                f"[v32_sw] step={_step} reqs={len(_store)} engine={_engine_idx['v']}"
            )
    return None


def _bookkeep(kbt, kvlen_d):
    """ZERO-SYNC step metadata: async D2H this step, decide with last step's."""
    required_width = 2 + STG_BLOCKS
    if kbt.dim() != 2 or int(kbt.shape[1]) < required_width:
        raise RuntimeError(
            f"offload bookkeeping requires block-table width >= {required_width}, "
            f"got shape={tuple(kbt.shape)}"
        )
    buf = _stepcache.get("buf")
    B = kbt.shape[0]
    if buf is None or buf[0].shape[0] < B or buf[1].shape[0] < B:
        buf = (
            torch.empty(B, dtype=kvlen_d.dtype, pin_memory=True),
            torch.empty((B, 2 + STG_BLOCKS), dtype=kbt.dtype, pin_memory=True),
            torch.cuda.Event(),
        )
        _stepcache["buf"] = buf
    pend = _stepcache.get("pend")
    if pend is not None and pend[2].query():  # last step's copy landed
        kvl_l = [v + (_step - pend[3]) for v in pend[0].tolist()]
        kh = pend[1]  # live pinned view: offload detection needs freshest block table
        _stepcache.update(kvlens=kvl_l, khead=kh, fresh_step=_step)
    elif "kvlens" not in _stepcache:
        _stepcache.update(kvlens=[], khead=None)
    s_ = _side_stream()
    s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        buf[0][:B].copy_(kvlen_d, non_blocking=True)
        buf[1][:B].copy_(kbt[:, : 2 + STG_BLOCKS], non_blocking=True)
        buf[2].record(s_)
    _stepcache["pend"] = (buf[0][:B], buf[1][:B], buf[2], _step)
    _stepcache["step"] = _step


def _entry(key, layer_id, cap_tokens, device, require_adopt=False):
    st = _store.get((key, layer_id))
    if st is not None and require_adopt:
        if hasattr(_ctx, "ctx_admission_generation"):
            generation = int(_ctx.ctx_admission_generation(key))
            if generation < 0:
                raise RuntimeError(
                    f"offloaded request {key} layer {layer_id} lost its admission mirror"
                )
            if int(st.get("generation", -1)) != generation:
                _invalidate_request_state(key)
                st = None
        elif hasattr(_ctx, "ctx_adopt"):
            current = _ctx.ctx_adopt(key, layer_id)
            if current is None:
                raise RuntimeError(
                    f"offloaded request {key} layer {layer_id} lost its admission mirror"
                )
            if len(current) >= 4 and int(st.get("generation", -1)) != int(current[3]):
                _invalidate_request_state(key)
                st = None
    if st is None:
        ad = None
        t_ad = time.perf_counter()
        if _ctx is not None and hasattr(_ctx, "ctx_adopt"):
            try:
                ad = _ctx.ctx_adopt(key, layer_id)
            except Exception:
                ad = None
        _prof["adopt"] += time.perf_counter() - t_ad
        if ad is not None:
            if len(ad) >= 4:
                kv, _ip_unused, durable, generation = ad
            else:
                kv, _ip_unused, durable = ad
                generation = -1
            st = {
                "kv": kv,
                "n": int(durable),
                "seen": _step,
                "reg": False,
                "adopted": True,
                "generation": int(generation),
            }
            if generation >= 0:
                _admission_generation[key] = int(generation)
            _store[(key, layer_id)] = st
            if layer_id == 0:
                logging.warning(
                    f"[v32_capacity] adopted admission mirror key={key} "
                    f"durable={int(durable)} cap={tuple(kv.shape)}"
                )
        else:
            if require_adopt:
                raise RuntimeError(
                    f"offloaded request {key} layer {layer_id} has no admission mirror"
                )
            t_h = time.perf_counter()
            kv_host = _alloc_host(cap_tokens)
            _prof["hostalloc"] += time.perf_counter() - t_h
            st = {
                "kv": kv_host,
                "n": 0,
                "seen": _step,
                "reg": False,
            }
            _store[(key, layer_id)] = st
    st["seen"] = _step
    return st


def _grow(st, key, layer_id, cap_tokens):
    if st["kv"].shape[0] < cap_tokens:
        t = st["kv"]
        n2 = _alloc_host(cap_tokens + 8192)
        n2[: st["n"]] = t[: st["n"]]
        st["kv"] = n2
        if not st.get("adopted"):
            _free_host(t)
        if st["reg"]:
            _ctx.ctx_update_host(key, layer_id, st["kv"])


_blkcache = {"key": None, "b0": -1, "blk": None}


def _blk_ids_host(key, bt_row, b0, b1):
    """Host copy of physical block ids [b0,b1); cached across the 61 layers."""
    c = _blkcache
    if (
        c["key"] == key
        and c["b0"] == b0
        and c["blk"] is not None
        and c["blk"].numel() == b1 - b0
    ):
        return c["blk"]
    blk = bt_row[b0:b1].to(torch.int32).cpu()
    c["key"], c["b0"], c["blk"] = key, b0, blk
    return blk


def _mirror_chunk(st, main_flat, bt_row, upto, key=None):
    """Mirror new main-KV rows to the pinned host store (indexer-K never leaves
    the GPU: the engine's independent pool keeps it resident, HANDOFF §11)."""
    lo, hi = st["n"], min(st["n"] + CHUNK, upto)
    if hi <= lo:
        return
    if (hi - 1) // BS >= bt_row.shape[0] or lo < 0:  # tripwire (should be unreachable)
        logging.error(
            f"[v32_capacity] mirror OOB trip lo={lo} hi={hi} btw={bt_row.shape[0]}"
        )
        st["bad"] = True
        return
    if lo % BS == 0 and hi // BS > lo // BS and _MIRROR_BLOCKS:
        b0, b1 = lo // BS, hi // BS
        blk = _blk_ids_host(key, bt_row, b0, b1)
        if int(blk.min()) > 0:
            _ctx.ctx_mirror_blocks_d2h(
                main_flat,
                blk,
                st["kv"],
                b0 * BS,
                BS,
                b1 * BS >= upto,  # flush only the chunk that completes the history
            )
            st["n"] = b1 * BS
            return
    pos = torch.arange(lo, hi, device=bt_row.device)
    phys = bt_row[(pos // BS).long()].long()
    keep = phys > 0
    pos, phys = pos[keep], phys[keep]
    if pos.numel() != hi - lo:
        st["bad"] = True
        return
    offs = (pos % BS).long()
    # KV D2H via ported staged copy: gather-scatter kernel + single D2H on its
    # own stream, returns when durable in the pinned host store.
    slots_cpu = (phys * BS + offs).to("cpu")
    _ctx.ctx_mirror_d2h(main_flat, slots_cpu, st["kv"][lo:hi])
    st["n"] = hi


_HAS_STEP_SERVE = _ctx is not None and hasattr(_ctx, "ctx_step_serve")
_faststep = {"step": -1, "armed": False}
_fast_key = {"key": None}  # armed request; _purge must not reap it (seen frozen)


def _invalidate_request_state(key):
    if _ctx is not None and hasattr(_ctx, "ctx_lossy_release"):
        _ctx.ctx_lossy_release(key)
    _lossy_meta.pop(key, None)
    for store_key in [k for k in _store if k[0] == key]:
        st = _store.pop(store_key)
        if not st.get("adopted"):
            _free_host(st["kv"])
    if _blkcache["key"] == key:
        _blkcache.update(key=None, b0=-1, blk=None)
    _plancache["step"] = -1
    _faststep["armed"] = False
    _fast_key["key"] = None


def process_layer_native_fast(kv_cache, kbt, kernel_topk, min_seq):
    """Steady-state C fast path entered directly from the hook boundary.

    Layer 0 runs the request-level checks once and arms the C++ step plan; every
    other layer is a single native call (step, layer, kbt, topk, pool). The first
    fast path repeated the dict/plan checks and a double map lookup per layer and
    recovered only 0.10ms of the measured 1.41ms. Registration and transition
    steps return False so the proven slow path stays authoritative.
    """
    fs = _faststep
    if fs["step"] != _step:
        fs["step"], fs["armed"] = _step, False
        _fast_key["key"] = None
        if (
            _ctx is None
            or not _HAS_STEP_SERVE
            or not _mode["lossless"]
            or not _engine_idx["v"]
        ):
            return False
        kvlens, khead = _stepcache.get("kvlens"), _stepcache.get("khead")
        if khead is None or not kvlens:
            return False
        plan = _row_plan(kvlens, khead, min_seq, kbt.shape[0])
        if len(plan) != 1:
            return False
        row_i, kvlen, _nb, key, offloaded = plan[0]
        if not offloaded:
            return False
        if hasattr(_ctx, "ctx_admission_generation"):
            generation = int(_ctx.ctx_admission_generation(key))
            if generation < 0:
                raise RuntimeError(
                    f"offloaded request {key} has no active admission generation"
                )
            previous = _admission_generation.get(key)
            if previous is not None and previous != generation:
                logging.warning(
                    f"[v32_capacity] recycled admission key={key} "
                    f"generation={previous}->{generation}; invalidating state"
                )
                _invalidate_request_state(key)
                _admission_generation[key] = generation
                return False
            _admission_generation[key] = generation
        _ctx.ctx_step_plan(_step, key, row_i, kvlen, LOSSLESS_DIAG_MODE, 128)
        _fast_key["key"] = key
        fs["armed"] = True
    if not fs["armed"]:
        return False
    return _ctx.ctx_step_serve(
        _step, int(kv_cache.layer_id), kbt, kernel_topk, kv_cache.kv_cache_base
    )


def process_layer(iop, q_fp8, weights, kv_cache, kbt, kvlen_d, kernel_topk, min_seq):
    t_all = time.perf_counter()
    try:
        _process_layer(
            iop, q_fp8, weights, kv_cache, kbt, kvlen_d, kernel_topk, min_seq
        )
    finally:
        _prof["proc"] += time.perf_counter() - t_all


def _row_plan(kvlens, khead, min_seq, b_now):
    """Rows to serve this step, with their layer-invariant metadata.

    The kernel block table and kvlens belong to the request, not the layer, so
    this is computed once per step instead of once per layer (61x).
    """
    c = _plancache
    if c["step"] == _step and c["min_seq"] == min_seq and c["b"] == b_now:
        return c["plan"]
    plan = []
    for i in range(min(len(kvlens), b_now)):
        kvlen = int(kvlens[i])
        if kvlen < min_seq:
            continue
        nb = (kvlen + BS - 1) // BS
        key = int(khead[i][0])
        if key <= 0 or nb < 2:
            continue
        offloaded = (
            nb > (2 + STG_BLOCKS)
            and int(khead[i][1 + STG_BLOCKS]) == 0
            and int(khead[i][1]) > 0
        )
        # Engagement has to be observable. The engine only reaches its offload
        # decision inside incrKVBlock, which runs when a new 64-token block is
        # needed, so the trigger lands at an arbitrary generated-token count set
        # by context_len mod 64. A run where it never fires measures exactly the
        # baseline and would otherwise look like a success.
        seen = _diag.get(key)
        if offloaded and seen != "on":
            _diag[key] = "on"
            _gen["n"] += 1
            logging.warning(
                f"[v32_plan] gen={_gen['n']} key={key} OFFLOAD ENGAGED at "
                f"step={_step} kvlen={kvlen}"
            )
        elif not offloaded and seen == "on":
            _diag[key] = _step  # request ended; block 0's id will be recycled
        elif not offloaded and seen is None:
            _diag[key] = _step
        elif not offloaded and isinstance(seen, int) and _step - seen >= 200:
            _diag[key] = _step
            logging.warning(
                f"[v32_plan] key={key} still NOT offloaded after {_step - 0} steps: "
                f"nb={nb} bt[1]={int(khead[i][1])} "
                f"bt[{1 + STG_BLOCKS}]={int(khead[i][1 + STG_BLOCKS])}"
            )
        if offloaded and not _engine_idx["v"]:
            _stats["unsafe"] += 1
            raise RuntimeError(
                f"offloaded request {key} has no engine indexer pool at step {_step}"
            )
        plan.append((i, kvlen, nb, key, offloaded))
    c.update(step=_step, min_seq=min_seq, b=b_now, plan=plan)
    return plan


def _pool_views(layer_id, main_pool):
    """Flat view of this layer's main KV pool, cached across steps.

    The pool is a per-layer object that does not move, but reshape was
    allocating a fresh TensorImpl on every layer of every step. Keyed on the
    data pointer so a reallocation invalidates the entry.
    """
    mp = main_pool.data_ptr()
    c = _view_cache.get(layer_id)
    if c is not None and c[0] == mp:
        return c[1]
    main_flat = main_pool.reshape(-1, main_pool.shape[-1])
    _view_cache[layer_id] = (mp, main_flat)
    return main_flat


def _process_layer(iop, q_fp8, weights, kv_cache, kbt, kvlen_d, kernel_topk, min_seq):
    if _ctx is None:
        return
    layer_id = int(kv_cache.layer_id)
    main_pool = kv_cache.kv_cache_base
    main_flat = _pool_views(layer_id, main_pool)
    t0 = time.perf_counter()
    if _stepcache["step"] != _step:  # pre_topk unavailable this step (safety)
        _bookkeep(kbt, kvlen_d)
    kvlens, khead = _stepcache["kvlens"], _stepcache["khead"]
    if khead is None or not kvlens:
        return
    # With the engine indexer pool the native top-k is always full/correct.
    single = bool(_engine_idx["v"])
    B_now = min(kbt.shape[0], q_fp8.shape[0], weights.shape[0], kvlen_d.shape[0])
    plan = _row_plan(kvlens, khead, min_seq, B_now)
    if not plan:
        return
    _prof["book"] += time.perf_counter() - t0

    served = []
    for i, kvlen, nb, key, offloaded in plan:
        if (
            offloaded
            and single
            and _mode["lossless"]
            and hasattr(_ctx, "ctx_lossless_try_serve")
            and _ctx.ctx_lossless_try_serve(
                key,
                layer_id,
                kbt,
                kernel_topk,
                i,
                main_flat,
                kvlen,
                LOSSLESS_DIAG_MODE,
            )
        ):
            served.append(i)
            _stats["serves"] += 1
            continue
        t_a = time.perf_counter()
        st = _entry(
            key,
            layer_id,
            kvlen + 8192,
            main_pool.device,
            require_adopt=offloaded,
        )
        fresh = _stepcache.get("fresh_step") == _step
        if fresh and "lk" in st and kvlen != st["lk"] + (_step - st["ls"]):
            _store.pop((key, layer_id), None)  # block0 recycled: stale host mirror
            if LOSSY and hasattr(_ctx, "ctx_lossy_release"):
                _ctx.ctx_lossy_release(key)
                _lossy_meta.pop(key, None)
            if not st.get("adopted"):
                _free_host(st["kv"])
            st = _entry(
                key,
                layer_id,
                kvlen + 8192,
                main_pool.device,
                require_adopt=offloaded,
            )
        if fresh:
            st["lk"], st["ls"] = kvlen, _step
        _grow(st, key, layer_id, kvlen + 1024)
        _prof["alloc"] += time.perf_counter() - t_a
        row = kbt[i]
        hist = kvlen - 1
        t1 = time.perf_counter()
        if not offloaded or (hist - st["n"]) >= LAG:
            _mirror_chunk(st, main_flat, row, hist, key)
        _prof["mirror"] += time.perf_counter() - t1
        if not offloaded:
            continue
        if st.get("bad") or st["n"] < hist - LAG:
            _stats["errors"] += 1
            raise RuntimeError(
                f"offloaded request {key} layer {layer_id} mirror is not durable: "
                f"n={st['n']} hist={hist} bad={bool(st.get('bad'))}"
            )
        if LOSSY and hasattr(_ctx, "ctx_lossy_serve"):
            if st.get("lreg") and hasattr(_ctx, "ctx_lossy_has"):
                if not _ctx.ctx_lossy_has(key, layer_id):
                    logging.error(
                        "[v32_capacity] offloaded key=%d layer=%d lost C++ state; re-registering",
                        key,
                        layer_id,
                    )
                    st["lreg"] = False
            if not st.get("lreg"):
                t_r = time.perf_counter()
                meta = _lossy_meta.get(key)
                if meta is None:
                    jpos = (
                        torch.nonzero(row[1 : 1 + STG_BLOCKS] > 0).reshape(-1) + 1
                    ).to(torch.int32)
                    sb = row.index_select(0, jpos.long()).to(torch.int32)
                    meta = (jpos.cpu(), sb.cpu())
                    _lossy_meta[key] = meta
                if meta[0].numel() == 0:
                    _stats["errors"] += 1
                    raise RuntimeError(
                        f"offloaded request {key} layer {layer_id} has no staging blocks"
                    )
                _ctx.ctx_lossy_register(
                    key,
                    layer_id,
                    meta[0],
                    meta[1],
                    nb + 128,
                    row.get_device(),
                    st["kv"],
                )
                st["lreg"] = True
                _prof["lreg"] += time.perf_counter() - t_r
            if hasattr(_ctx, "ctx_lossy_has") and not _ctx.ctx_lossy_has(key, layer_id):
                raise RuntimeError(
                    f"offloaded request {key} layer {layer_id} registration did not persist"
                )
            if SEL_TRACE:
                _sel_trace(key, kernel_topk, i, kvlen, "pre")
            t1 = time.perf_counter()
            if not single:
                _stats["errors"] += 1
                raise RuntimeError(
                    f"offloaded request {key} layer {layer_id} cannot use native scoring"
                )
            if _mode["lossless"]:
                _ctx.ctx_lossless_serve(
                    key,
                    layer_id,
                    st["kv"],
                    kbt,
                    kernel_topk,
                    i,
                    main_flat,
                    kvlen,
                    LOSSLESS_DIAG_MODE,
                )
            else:
                _ctx.ctx_lossy_serve(
                    key,
                    layer_id,
                    st["kv"],
                    kbt,
                    kernel_topk,
                    i,
                    main_flat,
                    kvlen,
                    _step,
                    _mode["prefetch"],
                )
            served.append(i)
            _stats["serves"] += 1
            _prof["serve"] += time.perf_counter() - t1
            if SEL_TRACE:
                _sel_trace(key, kernel_topk, i, kvlen, "post")
            continue
        if not st["reg"]:
            jpos = torch.nonzero(row[1 : 1 + STG_BLOCKS] > 0).reshape(-1) + 1
            sb = row[jpos].long()
            slots = (
                sb[:, None] * BS + torch.arange(BS, device=sb.device)[None, :]
            ).reshape(-1)
            logical = (
                jpos.long()[:, None] * BS + torch.arange(BS, device=sb.device)[None, :]
            ).reshape(-1)
            _ctx.ctx_register(
                key, layer_id, st["kv"], slots, logical, int(iop.index_topk)
            )
            st["reg"] = True
        t1 = time.perf_counter()
        if not single:
            # Offloaded request without the engine indexer pool: unservable
            # since the shadow-pool rescore paths were removed (HANDOFF §11).
            _stats["errors"] += 1
            continue
        # Native top-k already scored the full history from the engine pool;
        # kernel_topk row holds logical top-2048 — build/fetch/write-back only.
        _ctx.ctx_serve_wb(
            key,
            layer_id,
            kernel_topk[i],
            kbt,
            kernel_topk,
            i,
            main_flat,
            kvlen,
            _step,
            layer_id % 4 == 0,
        )
        served.append(i)
        _stats["serves"] += 1
        _prof["serve"] += time.perf_counter() - t1
    if served:
        _prof["n"] += 1
        if _prof["n"] % 3050 == 0:
            logging.warning(
                f"[v32_capacity] prof(s)={ {k: round(v,2) for k,v in _prof.items()} } stats={_stats}"
            )
            if hasattr(_ctx, "ctx_ktimings"):
                kt = _ctx.ctx_ktimings()
                if kt and kt[1]:
                    # Per-layer device time for the three pieces we add. fetch is
                    # the PCIe gather, which is the only intrinsic cost here; mask
                    # and append are launch-bound and removable in principle.
                    logging.warning(
                        "[v32_ktime] mask=%.4f (n=%d) fetch=%.4f (n=%d) "
                        "append=%.4f (n=%d) ms/layer"
                        % (
                            kt[0] / max(kt[1], 1),
                            kt[1],
                            kt[2] / max(kt[3], 1),
                            kt[3],
                            kt[4] / max(kt[5], 1),
                            kt[5],
                        )
                    )
            if LOSSY and hasattr(_ctx, "ctx_lossy_counters"):
                cn = _ctx.ctx_lossy_counters()
                t_, p_, m_, s_ = cn[:4]
                if len(cn) >= 6 and cn[5]:
                    # A skipped admit leaves the slot holding stale or zero bytes,
                    # which scores wrongly; on a repeated prompt the recycled case
                    # hides it, so any non-zero count here matters.
                    logging.error(
                        f"[v32_unsafe] admit skipped {cn[5]} of {cn[4]} tokens: "
                        f"indexer pool has holes and scores are wrong"
                    )
                tot = t_ + p_ + m_
                if tot and _mode["lossless"]:
                    logging.warning(
                        f"[v32_lossless] exact={(t_ + p_) / tot:.6f} "
                        f"(tail={t_} fetched={p_} overflow={m_} serves={s_} "
                        f"fetch%={p_ / tot:.4f} tok_per_serve={p_ / max(s_, 1):.1f})"
                    )
                elif tot:
                    logging.warning(
                        f"[v32_lossy] warm-pool hit ratio={(t_ + p_) / tot:.4f} "
                        f"(tail={t_} pool={p_} miss={m_} serves={s_} "
                        f"tail%={t_ / tot:.4f} pool%={p_ / tot:.4f})"
                    )


def _purge():
    live = _fast_key["key"]
    dead = []
    for store_key, st in _store.items():
        if _step - st["seen"] <= 1500 or store_key[0] == live:
            continue
        if st.get("adopted") and hasattr(_ctx, "ctx_admission_generation"):
            generation = int(_ctx.ctx_admission_generation(store_key[0]))
            if generation == int(st.get("generation", -1)):
                st["seen"] = _step
                continue
        dead.append(store_key)
    released = set()
    for store_key in dead:
        st = _store.pop(store_key)
        key = store_key[0]
        if key not in released:
            try:
                if _ctx is not None:
                    _ctx.ctx_release(key)
                    if hasattr(_ctx, "ctx_lossy_release"):
                        _ctx.ctx_lossy_release(key)
                    if hasattr(_ctx, "ctx_admission_release"):
                        _ctx.ctx_admission_release(key, int(st.get("generation", -1)))
            except Exception:
                logging.exception(
                    "[v32_capacity] stale request cleanup failed key=%d generation=%d",
                    key,
                    int(st.get("generation", -1)),
                )
            _lossy_meta.pop(key, None)
            _admission_generation.pop(key, None)
            released.add(key)
        if not st.get("adopted"):
            _free_host(st["kv"])


def stats():
    return dict(_stats)
