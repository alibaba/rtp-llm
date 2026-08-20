"""v32_hitratio.py — shadow measurement: what fraction of the native DSA
top-2048 selections would be served by (a) a pure resident window
(block0 sink + last-W tokens) and (b) window + an LRU of previously-selected
blocks (1-step lag, modeling async prefetch).

Measurement-only: wraps the native scoring output, never alters behavior.
Synchronizes GPU->CPU per layer per step; use only in dedicated runs
(V32_HITRATIO=1). Aggregates dumped as JSON lines to V32_HITRATIO_OUT.
"""

import json
import logging
import os
import time
from collections import OrderedDict

import torch

BS = 64
MIN_SEQ = int(os.environ.get("V32_HITRATIO_MIN_SEQ", "16384"))
WINDOWS = (8192, 16384, 32768)  # pure-window variants (tokens)
LRU_BASE_WIN = 16384  # LRU rides on top of this window
LRU_CAPS = (32, 64, 128)  # LRU capacities in 64-token blocks
OUT = os.environ.get("V32_HITRATIO_OUT", "/home/admin/rtp-hol/logs/v32_hitratio.jsonl")
DUMP_EVERY_STEPS = int(os.environ.get("V32_HITRATIO_DUMP_STEPS", "32"))

_lru = {}  # (req_key, layer, cap) -> OrderedDict(block -> True)
_agg = {}  # (layer, cfg) -> [hits, total]
_series = {}  # (step_bucket, cfg) -> [hits, total]  (all layers pooled)
_step = {"n": 0, "last_dump": 0}


def _acc(table, k, hits, total):
    e = table.get(k)
    if e is None:
        table[k] = [hits, total]
    else:
        e[0] += hits
        e[1] += total


def observe(iop, kv_cache, fmha_params, attention_inputs, kernel_topk):
    kbt = getattr(attention_inputs, "kv_cache_kernel_block_id_device", None)
    if kbt is None or kernel_topk is None:
        return
    if kernel_topk.shape[0] != kbt.shape[0]:
        return  # prefill-shaped call
    layer_id = int(kv_cache.layer_id)
    if layer_id == 0:
        _step["n"] += 1
    step = _step["n"]
    kvl = fmha_params.kvlen_d
    B = kbt.shape[0]
    for i in range(B):
        kvlen = int(kvl[i])
        if kvlen < MIN_SEQ:
            continue
        key = int(kbt[i][0])
        if key <= 0:
            continue
        sel = kernel_topk[i].reshape(-1)
        sel = sel[(sel >= 0) & (sel < kvlen)]
        n = int(sel.numel())
        if n == 0:
            continue
        pos = sel.cpu().tolist()
        bucket = step // 16

        # pure windows (sink block0 = positions < 64)
        for w in WINDOWS:
            lo = kvlen - w
            hits = sum(1 for p in pos if p >= lo or p < BS)
            _acc(_agg, (layer_id, f"win{w//1024}k"), hits, n)
            _acc(_series, (bucket, f"win{w//1024}k"), hits, n)

        # window(16k) + LRU(cap); hits counted BEFORE this step's insertions
        lo16 = kvlen - LRU_BASE_WIN
        out_blocks = []  # selected out-of-window blocks, selection order
        seen = set()
        for p in pos:
            if p >= lo16 or p < BS:
                continue
            b = p // BS
            if b not in seen:
                seen.add(b)
                out_blocks.append(b)
        for cap in LRU_CAPS:
            lru = _lru.get((key, layer_id, cap))
            if lru is None:
                lru = OrderedDict()
                _lru[(key, layer_id, cap)] = lru
            hits = sum(1 for p in pos if p >= lo16 or p < BS or (p // BS) in lru)
            _acc(_agg, (layer_id, f"lru{cap}_w16k"), hits, n)
            _acc(_series, (bucket, f"lru{cap}_w16k"), hits, n)
            # 1-step-lag prefetch model: this step's selected out-of-window
            # blocks become resident for later steps
            for b in out_blocks:
                if b in lru:
                    lru.move_to_end(b)
                else:
                    lru[b] = True
            while len(lru) > cap:
                lru.popitem(last=False)

    if layer_id == 0 and step - _step["last_dump"] >= DUMP_EVERY_STEPS and step > 1:
        _step["last_dump"] = step
        try:
            dump(step)
        except Exception:
            logging.exception("[v32_hitratio] dump error")


def dump(step):
    cfgs = {}
    for (layer, cfg), (h, t) in _agg.items():
        e = cfgs.setdefault(cfg, [0, 0])
        e[0] += h
        e[1] += t
    overall = {cfg: round(h / max(t, 1), 6) for cfg, (h, t) in sorted(cfgs.items())}
    per_layer = {}
    for (layer, cfg), (h, t) in sorted(_agg.items()):
        per_layer.setdefault(cfg, {})[layer] = round(h / max(t, 1), 6)
    series = {}
    for (bucket, cfg), (h, t) in sorted(_series.items()):
        series.setdefault(cfg, {})[bucket * 16] = round(h / max(t, 1), 6)
    rec = {
        "ts": time.time(),
        "step": step,
        "overall": overall,
        "per_layer": per_layer,
        "series": series,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    logging.warning(f"[v32_hitratio] step={step} overall={overall}")
