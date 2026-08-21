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

try:  # native fused scorer (same op as scheme A)
    import deep_gemm as _dg

    from rtp_llm.models_py.kernels.cuda.fast_topk import (
        fast_topk_transform_fused as _ftk,
    )
except Exception:
    _dg = _ftk = None
    logging.exception(
        "[v32_capacity] deep_gemm/fast_topk missing — falling back to own score kernel"
    )

BS = 64
CHUNK = int(os.environ.get("V32_MIRROR_CHUNK", "4096"))
LAG = 256
STG_BLOCKS = int(os.environ.get("RTP_KV_OFFLOAD_STAGING_BLOCKS", "32"))
IDXNB = int(os.environ.get("V32_IDX_POOL_BLOCKS", "4096"))
SINGLE_WAVE = os.environ.get("V32_SINGLE_WAVE", "1") == "1"
LOSSY = os.environ.get("V32_LOSSY", "0") == "1"
LOSSY_PREFETCH = int(os.environ.get("V32_LOSSY_PREFETCH", "8"))
_MIRROR_BLOCKS = os.environ.get("V32_MIRROR_BLOCKS", "1") == "1" and hasattr(
    _ctx, "ctx_mirror_blocks_d2h"
)
_MARGIN = 2
HOST_BUCKET = int(os.environ.get("V32_HOST_BUCKET", "8192"))
PREWARM_TOKENS = int(os.environ.get("V32_PREWARM_TOKENS", "73728"))
PREWARM_LAYERS = int(os.environ.get("V32_PREWARM_LAYERS", "61"))

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


if PREWARM_TOKENS > 0:
    threading.Thread(target=_prewarm_host, daemon=True).start()

STEP_TRACE = int(os.environ.get("V32_STEP_TRACE", "0"))
_trace = {"t": None, "d": []}
_plancache = {"step": -1, "min_seq": -1, "b": -1, "plan": []}
_store = {}
_t1cache = {}  # per-request: kvlen tensor / schedule metadata / cu_seqlens
_ibt_cache = {}  # capacity -> dense arange block table (dual-wave fallback)
_lossy_meta = {}  # key -> (jpos cpu i32, sb cpu i32): staging layout, layer-invariant
_step = 0
_stream = None
_stepcache = {"step": -1}
_stats = {"serves": 0, "errors": 0, "single": 0, "dual": 0}
_last_offloaded = {"step": -1, "rows": None}  # gpu int64 tensor of offloaded row ids

# ---- global indexer pool (single wave) ----
_ipool = {}  # layer_id -> uint8 [IDXNB, 64, 132]
_ifree = list(range(IDXNB))
_ireq = {}  # key -> {row(gpu i32), blocks, nb, wm, seen}
_sw = {
    "step": -1,
    "ready": False,
    "single": False,
    "admits": [],
    "ibt": None,
    "exp": None,
    "ok": None,
    "okpin": None,
    "okpend": None,
    "meta": None,
    "keys": [],
}


def offloaded_rows_hint(device):
    return _last_offloaded["rows"]


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


def _ipool_layer(layer_id, device):
    p = _ipool.get(layer_id)
    if p is None:
        p = torch.zeros((IDXNB, BS, 132), dtype=torch.uint8, device=device)
        _ipool[layer_id] = p
    return p


def _free_req(key):
    e = _ireq.pop(key, None)
    if e is not None:
        _ifree.extend(e["blocks"])


def _bookkeep_pool(kbt, kvlen_d, device):
    """Layer-0 CPU bookkeeping for the single-wave pool (zero-sync)."""
    _sw.update(step=_step, ready=False, single=False, admits=[], keys=[])
    if not SINGLE_WAVE or _dg is None:
        return
    # drain last step's tripwire (landed asynchronously)
    okp = _sw.get("okpend")
    if okp is not None and okp[1].query():
        vals = okp[0][: okp[2]].tolist()
        for j, v in enumerate(vals):
            if v == 0 and j < len(okp[3]):
                logging.warning(
                    f"[v32_capacity] tripwire row={j} key={okp[3][j]} — re-admit"
                )
                _free_req(okp[3][j])
        _sw["okpend"] = None
    if _stepcache.get("fresh_step") != _step:
        return
    kvl = _stepcache["kvlens"]
    kh = _stepcache["khead"]
    B = kbt.shape[0]
    if len(kvl) != B:
        return
    admits, keys = [], []
    for i in range(B):
        key = int(kh[i][0])
        if key <= 0:
            return  # dummy/degenerate rows: skip maintenance this step
        keys.append(key)
        kvlen = int(kvl[i])
        nb_need = (kvlen + BS - 1) // BS
        e = _ireq.get(key)
        if e is not None and kvlen != e["lk"] + (_step - e["ls"]):
            _free_req(key)  # block0 recycled to a new request: stale bytes, re-admit
            e = None
        # admission-offloaded row (0-sentinel prefix): its indexer-K history is
        # not in the main pool, so a pool bulk-admit would read garbage. Serve
        # such rows via the dual-wave side store instead.
        if (
            nb_need > (2 + STG_BLOCKS)
            and int(kh[i][1 + STG_BLOCKS]) == 0
            and int(kh[i][1]) > 0
            and (e is None or e["wm"] < kvlen - 1)
        ):
            _stats["dual"] += 1
            return
        if e is None:
            if len(_ifree) < nb_need + _MARGIN:
                _stats["dual"] += 1
                return
            blocks = [_ifree.pop() for _ in range(nb_need + _MARGIN)]
            cap = max(len(blocks) + 64, 128)
            row_cpu = torch.full((cap,), -1, dtype=torch.int32)
            row_cpu[: len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
            e = {
                "row": row_cpu.to(device),
                "blocks": blocks,
                "nb": len(blocks),
                "wm": 0,
                "seen": _step,
                "lk": kvlen,
                "ls": _step,
            }
            _ireq[key] = e
        e.update(seen=_step, lk=kvlen, ls=_step)
        while e["nb"] < nb_need + _MARGIN:
            if not _ifree:
                return
            b = _ifree.pop()
            if e["nb"] >= e["row"].shape[0]:
                nr = torch.full(
                    (e["row"].shape[0] * 2,), -1, dtype=torch.int32, device=device
                )
                nr[: e["row"].shape[0]] = e["row"]
                e["row"] = nr
            e["row"][e["nb"]] = b
            e["blocks"].append(b)
            e["nb"] += 1
        if e["wm"] < kvlen - 1:  # admission or backfill of missed steps
            admits.append((i, key, e["wm"], kvlen - 1))
        e["wm"] = kvlen
    # batch tensors: ibt / expected block0 / tripwire flags
    maxnb = max((int(v) + BS - 1) // BS + _MARGIN for v in kvl)
    ib = _sw.get("ibt")
    if ib is None or ib.shape[0] < B or ib.shape[1] < maxnb:
        ib = torch.full((max(B, 8), maxnb + 64), -1, dtype=torch.int32, device=device)
        _sw["ibt"] = ib
    ib.fill_(-1)
    for i in range(B):
        e = _ireq[keys[i]]
        n = min(e["nb"], ib.shape[1])
        ib[i, :n] = e["row"][:n]
    exp = _sw.get("exp")
    if exp is None or exp.shape[0] < B:
        _sw["exp"] = exp = torch.empty(max(B, 8), dtype=torch.int32, device=device)
        _sw["ok"] = torch.empty(max(B, 8), dtype=torch.int32, device=device)
        _sw["okpin"] = torch.empty(max(B, 8), dtype=torch.int32, pin_memory=True)
    exp[:B] = torch.tensor(keys, dtype=torch.int32)
    _sw["ok"][:B].fill_(1)
    _sw["meta"] = _dg.get_paged_mqa_logits_metadata(kvlen_d, BS, _dg.get_num_sms())
    _sw.update(ready=True, single=True, admits=admits, keys=keys)
    _stats["single"] += 1


def _finish_step_pool(B):
    # async tripwire drain (side stream, read next step)
    s_ = _side_stream()
    s_.wait_stream(torch.cuda.current_stream())
    ev = torch.cuda.Event()
    with torch.cuda.stream(s_):
        _sw["okpin"][:B].copy_(_sw["ok"][:B], non_blocking=True)
        ev.record(s_)
    _sw["okpend"] = (_sw["okpin"], ev, B, list(_sw["keys"]))


def pre_topk(iop, q_fp8, weights, kv_cache, fmha_params, attention_inputs):
    """Called before the native scoring wave. Returns the final topk tensor
    (single wave) or None (caller runs the native wave)."""
    global _step
    if _ctx is None:
        return None
    kbt = getattr(attention_inputs, "kv_cache_kernel_block_id_device", None)
    kvlen_d = fmha_params.kvlen_d
    if kbt is None or q_fp8.shape[0] != kbt.shape[0]:
        return None
    layer_id = int(kv_cache.layer_id)
    if layer_id == 0:
        _step += 1
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
        if _step % 1000 == 0:
            _purge()
        _bookkeep(kbt, kvlen_d)
        _bookkeep_pool(kbt, kvlen_d, kbt.device)
        if _sw["ready"]:
            _finish_step_pool(kbt.shape[0])
        if _step % 500 == 0:
            logging.warning(
                f"[v32_sw] step={_step} single_steps={_stats['single']} "
                f"ifree={len(_ifree)} reqs={len(_ireq)}"
            )
    if _sw["step"] != _step or not _sw["ready"]:
        return None
    idx_pool = kv_cache.kv_scale_base
    idx_pool_u8 = idx_pool.reshape(idx_pool.shape[0], -1).view(torch.uint8)
    pool_l = _ipool_layer(layer_id, kbt.device)
    B = kbt.shape[0]
    for i, key, lo, hi in _sw["admits"]:
        e = _ireq.get(key)
        if e is not None:
            _ctx.ctx_bulk_admit(pool_l, idx_pool_u8, kbt, i, e["row"], lo, hi)
    _ctx.ctx_batch_append(
        pool_l, idx_pool_u8, kbt, _sw["ibt"][:B], kvlen_d, _sw["exp"], _sw["ok"]
    )
    if not _sw["single"]:
        return None
    logits = _dg.fp8_paged_mqa_logits(
        q_fp8.unsqueeze(1),
        pool_l.unsqueeze(2),
        weights.view(-1, iop.index_n_heads),
        kvlen_d,
        _sw["ibt"][:B],
        _sw["meta"],
        _sw["ibt"].shape[1] * BS,
        clean_logits=False,
    )
    return _ftk(
        logits,
        fmha_params.expanded_seq_lens,
        attention_inputs.decode_cu_seqlens_device,
        2048,
    )


def _bookkeep(kbt, kvlen_d):
    """ZERO-SYNC step metadata: async D2H this step, decide with last step's."""
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


def _entry(key, layer_id, cap_tokens, device):
    st = _store.get((key, layer_id))
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
            kv, ip, durable = ad
            st = {
                "kv": kv,
                "idxp": ip,
                "n": int(durable),
                "seen": _step,
                "ev": torch.cuda.Event(),
                "ev_idx": torch.cuda.Event(),
                "reg": False,
                "adopted": True,
            }
            _store[(key, layer_id)] = st
            if layer_id == 0:
                logging.warning(
                    f"[v32_capacity] adopted admission mirror key={key} "
                    f"durable={int(durable)} cap={tuple(kv.shape)}"
                )
        else:
            t_h = time.perf_counter()
            kv_host = _alloc_host(cap_tokens)
            _prof["hostalloc"] += time.perf_counter() - t_h
            st = {
                "kv": kv_host,
                "idxp": None,  # dual-wave only: built on demand, see _idxp()
                "idxnb": (cap_tokens + BS - 1) // BS,
                "n": 0,
                "seen": _step,
                "ev": torch.cuda.Event(),
                "ev_idx": torch.cuda.Event(),
                "reg": False,
            }
            _store[(key, layer_id)] = st
    st["seen"] = _step
    return st


def _grow(st, key, layer_id, cap_tokens):
    if st["kv"].shape[0] < cap_tokens:
        t = st["kv"]
        n2 = _alloc_host(cap_tokens + 8192)
        n2[: t.shape[0]] = t
        st["kv"] = n2
        if not st.get("adopted"):
            _free_host(t)
        if st["reg"]:
            _ctx.ctx_update_host(key, layer_id, st["kv"])
    nb = (cap_tokens + 8192) // BS
    need = (cap_tokens + BS - 1) // BS
    st["idxnb"] = max(st.get("idxnb", 0), need)
    if st["idxp"] is not None and st["idxp"].shape[0] < need:
        t = st["idxp"]
        n2 = torch.zeros((nb, BS, 132), dtype=torch.uint8, device=t.device)
        n2[: t.shape[0]] = t
        st["idxp"] = n2


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


def _idxp(st, device):
    """Private indexer store, materialized only when the dual-wave path needs it."""
    if st["idxp"] is None:
        st["idxp"] = torch.zeros(
            (st.get("idxnb", 128), BS, 132), dtype=torch.uint8, device=device
        )
    return st["idxp"]


def _mirror_chunk(st, main_pool, idx_pool_u8, bt_row, upto, key=None):
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
            if st["idxp"] is not None:
                s = _side_stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    st["idxp"][b0:b1] = idx_pool_u8[bt_row[b0:b1].long()].view(
                        -1, BS, 132
                    )
                    st["ev_idx"].record(s)
                torch.cuda.current_stream().wait_event(st["ev_idx"])
            _ctx.ctx_mirror_blocks_d2h(
                main_pool.reshape(-1, main_pool.shape[-1]),
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
    if st["idxp"] is None:
        st["idxp"] = _idxp(st, bt_row.device)
    s = _side_stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        dstpos = pos.long()
        blk = idx_pool_u8[phys]
        fp8 = blk.gather(
            1, offs[:, None] * 128 + torch.arange(128, device=phys.device)[None, :]
        )
        sc = blk.gather(
            1,
            BS * 128 + offs[:, None] * 4 + torch.arange(4, device=phys.device)[None, :],
        )
        flatb = st["idxp"].reshape(-1)
        base = (dstpos // BS) * (132 * BS)
        flatb[
            (base + (dstpos % BS) * 128)[:, None]
            + torch.arange(128, device=phys.device)[None, :]
        ] = fp8
        flatb[
            (base + BS * 128 + (dstpos % BS) * 4)[:, None]
            + torch.arange(4, device=phys.device)[None, :]
        ] = sc
        st["ev_idx"].record(s)
    torch.cuda.current_stream().wait_event(st["ev_idx"])
    # KV D2H via ported staged copy: gather-scatter kernel + single D2H on its
    # own stream, returns when durable in the pinned host store.
    slots_cpu = (phys * BS + offs).to("cpu")
    _ctx.ctx_mirror_d2h(
        main_pool.reshape(-1, main_pool.shape[-1]), slots_cpu, st["kv"][lo:hi]
    )
    st["n"] = hi


def _dual_wave_sel(
    iop, q_fp8, weights, st, key, layer_id, idx_pool_u8, kbt, i, kvlen, nb
):
    """Per-row dual-wave scoring on the private side store; returns sel [1, 2048].

    Returns None when the private store holds no history: the single-wave global
    pool has been serving this row, so there is nothing to score against here.
    """
    if st["idxp"] is None:
        return None
    device = q_fp8.device
    _ctx.ctx_append_tok(key, layer_id, st["idxp"], idx_pool_u8, kbt, i, kvlen - 1)
    tc = _t1cache.get(key)
    if tc is None:
        tc = {
            "kvt": torch.empty(1, dtype=torch.int32, device=device),
            "cu": torch.tensor([0, 1], dtype=torch.int32, device=device),
            "step": -1,
        }
        _t1cache[key] = tc
    if tc["step"] != _step:
        tc["kvt"].fill_(kvlen)
        tc["meta"] = _dg.get_paged_mqa_logits_metadata(tc["kvt"], BS, _dg.get_num_sms())
        tc["step"] = _step
    cap = st["idxp"].shape[0]
    ibt = _ibt_cache.get(cap)
    if ibt is None:
        ibt = torch.arange(cap, dtype=torch.int32, device=device).unsqueeze(0)
        _ibt_cache[cap] = ibt
    logits = _dg.fp8_paged_mqa_logits(
        q_fp8[i : i + 1].unsqueeze(1),
        st["idxp"].unsqueeze(2),
        weights.view(-1, iop.index_n_heads)[i : i + 1],
        tc["kvt"],
        ibt[:, :nb],
        tc["meta"],
        nb * BS,
        clean_logits=False,
    )
    return _ftk(logits, tc["kvt"], tc["cu"], 2048)


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
        plan.append((i, kvlen, nb, key, offloaded))
    c.update(step=_step, min_seq=min_seq, b=b_now, plan=plan)
    return plan


def _process_layer(iop, q_fp8, weights, kv_cache, kbt, kvlen_d, kernel_topk, min_seq):
    if _ctx is None:
        return
    layer_id = int(kv_cache.layer_id)
    main_pool = kv_cache.kv_cache_base
    idx_pool = kv_cache.kv_scale_base
    idx_pool_u8 = idx_pool.reshape(idx_pool.shape[0], -1).view(torch.uint8)
    t0 = time.perf_counter()
    if _stepcache["step"] != _step:  # pre_topk unavailable this step (safety)
        _bookkeep(kbt, kvlen_d)
    kvlens, khead = _stepcache["kvlens"], _stepcache["khead"]
    if khead is None or not kvlens:
        return
    single = _sw["step"] == _step and _sw["single"]
    B_now = min(kbt.shape[0], q_fp8.shape[0], weights.shape[0], kvlen_d.shape[0])
    plan = _row_plan(kvlens, khead, min_seq, B_now)
    if not plan:
        return
    _prof["book"] += time.perf_counter() - t0

    served = []
    for i, kvlen, nb, key, offloaded in plan:
        t_a = time.perf_counter()
        st = _entry(key, layer_id, kvlen + 8192, main_pool.device)
        fresh = _stepcache.get("fresh_step") == _step
        if fresh and "lk" in st and kvlen != st["lk"] + (_step - st["ls"]):
            _store.pop((key, layer_id), None)  # block0 recycled: stale host mirror
            if LOSSY and hasattr(_ctx, "ctx_lossy_release"):
                _ctx.ctx_lossy_release(key)
                _lossy_meta.pop(key, None)
            if not st.get("adopted"):
                _free_host(st["kv"])
            st = _entry(key, layer_id, kvlen + 8192, main_pool.device)
        if fresh:
            st["lk"], st["ls"] = kvlen, _step
        _grow(st, key, layer_id, kvlen + 1024)
        _prof["alloc"] += time.perf_counter() - t_a
        row = kbt[i]
        hist = kvlen - 1
        t1 = time.perf_counter()
        if not offloaded or (hist - st["n"]) >= LAG:
            _mirror_chunk(st, main_pool, idx_pool_u8, row, hist, key)
        _prof["mirror"] += time.perf_counter() - t1
        if not offloaded:
            continue
        if st.get("bad") or st["n"] < hist - LAG:
            _stats["errors"] += 1
            continue
        if LOSSY and hasattr(_ctx, "ctx_lossy_serve"):
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
                    continue
                _ctx.ctx_lossy_register(
                    key, layer_id, meta[0], meta[1], nb + 128, row.get_device()
                )
                st["lreg"] = True
                _prof["lreg"] += time.perf_counter() - t_r
            t1 = time.perf_counter()
            if not single:
                if _dg is None or int(iop.index_topk) != 2048:
                    _stats["errors"] += 1
                    continue
                sel = _dual_wave_sel(
                    iop,
                    q_fp8,
                    weights,
                    st,
                    key,
                    layer_id,
                    idx_pool_u8,
                    kbt,
                    i,
                    kvlen,
                    nb,
                )
                if sel is None:
                    _stats["errors"] += 1
                    continue
                ktr = kernel_topk[i].reshape(-1)
                n_ = min(int(ktr.shape[0]), int(sel.numel()))
                ktr[:n_].copy_(sel.reshape(-1)[:n_])
                if ktr.shape[0] > n_:
                    ktr[n_:].fill_(-1)
            _ctx.ctx_lossy_serve(
                key,
                layer_id,
                st["kv"],
                kbt,
                kernel_topk,
                i,
                main_pool.reshape(-1, main_pool.shape[-1]),
                kvlen,
                _step,
                LOSSY_PREFETCH,
            )
            served.append(i)
            _stats["serves"] += 1
            _prof["serve"] += time.perf_counter() - t1
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
        if single:
            # single wave already scored full history from the global pool;
            # kernel_topk row holds logical top-2048 — build/fetch/write-back only.
            _ctx.ctx_serve_wb(
                key,
                layer_id,
                kernel_topk[i],
                kbt,
                kernel_topk,
                i,
                main_pool.reshape(-1, main_pool.shape[-1]),
                kvlen,
                _step,
                layer_id % 4 == 0,
            )
        elif _dg is not None and int(iop.index_topk) == 2048:
            # dual wave (r27): per-row native fused scorer on the private side store
            sel = _dual_wave_sel(
                iop, q_fp8, weights, st, key, layer_id, idx_pool_u8, kbt, i, kvlen, nb
            )
            if sel is None:
                _stats["errors"] += 1
                continue
            _ctx.ctx_serve_wb(
                key,
                layer_id,
                sel,
                kbt,
                kernel_topk,
                i,
                main_pool.reshape(-1, main_pool.shape[-1]),
                kvlen,
                _step,
                layer_id % 4 == 0,
            )
        else:  # fallback: own score kernel + topk in one C++ call
            _ctx.ctx_serve_full(
                key,
                layer_id,
                q_fp8,
                weights,
                _idxp(st, main_pool.device),
                kbt,
                kernel_topk,
                i,
                main_pool.reshape(-1, main_pool.shape[-1]),
                hist,
                iop.index_topk - 1,
                _step,
                layer_id % 4 == 0,
            )
        served.append(i)
        _stats["serves"] += 1
        _prof["serve"] += time.perf_counter() - t1
    if served:
        if _last_offloaded["step"] != _step:
            _last_offloaded["rows"] = torch.tensor(
                served, dtype=torch.int64, device=main_pool.device
            )
            _last_offloaded["step"] = _step
        _prof["n"] += 1
        if _prof["n"] % 3050 == 0:
            logging.warning(
                f"[v32_capacity] prof(s)={ {k: round(v,2) for k,v in _prof.items()} } stats={_stats}"
            )
            if LOSSY and hasattr(_ctx, "ctx_lossy_counters"):
                t_, p_, m_, s_ = _ctx.ctx_lossy_counters()
                tot = t_ + p_ + m_
                if tot:
                    logging.warning(
                        f"[v32_lossy] warm-pool hit ratio={(t_ + p_) / tot:.4f} "
                        f"(tail={t_} pool={p_} miss={m_} serves={s_} "
                        f"tail%={t_ / tot:.4f} pool%={p_ / tot:.4f})"
                    )


def _purge():
    dead = [k for k, v in _store.items() if _step - v["seen"] > 1500]
    for k in dead:
        try:
            if _ctx is not None:
                _ctx.ctx_release(k[0])
                if hasattr(_ctx, "ctx_admission_release"):
                    _ctx.ctx_admission_release(k[0])
                if hasattr(_ctx, "ctx_lossy_release"):
                    _ctx.ctx_lossy_release(k[0])
        except Exception:
            pass
        _t1cache.pop(k[0], None)
        _lossy_meta.pop(k[0], None)
        st = _store.pop(k)
        if not st.get("adopted"):
            _free_host(st["kv"])
    for k in [k for k, e in _ireq.items() if _step - e["seen"] > 1500]:
        _free_req(k)


def stats():
    return dict(_stats)
