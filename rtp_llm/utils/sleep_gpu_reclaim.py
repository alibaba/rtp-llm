"""Sleep-time GPU reclaim + diagnosis.

Called on the engine sleep hook (after the torch_memory_saver VMM pauses of
weights/cuda_graph and the C++ KV release) to hand as much *physical* GPU memory
back to the driver as possible, so a sleeping instance frees the device for other
processes.

The C++ hook already calls ``c10::cuda::CUDACachingAllocator::emptyCache()``, but
that only returns segments that are 100%-free. Large per-forward workspaces are
freed after each forward, yet their segments can stay resident if any long-lived
runtime tensor (module-level cache, workspace singleton) is co-tenanted in the
same segment. This module drops those known Python-held device caches first, then
empties the cache, so previously co-tenanted segments become fully free and are
returned to the driver.

It also logs a ``[SleepReclaim]`` segment breakdown (``torch.cuda.memory_snapshot``)
so any residual can be attributed to a specific live allocation rather than guessed
at from decoupled reserved/allocated counters (which under TMS do not track physical
residency -- only ``cudaMemGetInfo`` / ``mem_get_info`` is physically truthful).

Best-effort only: every step is guarded and never raises into the sleep hook (a
throw there would be mistaken for a hook failure and push the controller to ERROR).
"""

import gc
import logging
import os

import torch

_MiB = 1024.0 * 1024.0
_GiB = 1024.0**3


def _clear_module_device_caches() -> list[str]:
    """Drop long-lived Python-held device tensor caches so their segments free.

    Returns a list of human-readable notes for logging. MegaMoE symmetric-memory
    destruction is non-collective, but recreation requires a collective rendezvous
    and is therefore opt-in.
    """
    notes: list[str] = []
    # MegaMoE per-token output staging buffer: a plain torch.empty tensor cached at
    # module scope, recreated lazily on the next MoE forward (wake warmup).
    try:
        from rtp_llm.models_py.modules.dsv4.moe import mega_buf

        n, freed = mega_buf.release_mega_output_buffers()
        if n:
            notes.append(f"mega_output_cache released ({n} entries, ~{freed:.3f} GiB)")
        # MegaMoE symmetric-memory buffer (~4.4 GiB/rank). Its cross-rank state
        # (CUDA multicast binding + peer P2P imports, keyed to the physical VMM
        # handle) cannot be VMM-paused at a fixed VA -- a pause/resume would break
        # the bindings, so releasing it requires a full destroy + re-rendezvous on
        # wake. Opt-in via RTP_LLM_SLEEP_FREE_MEGA_SYMM=1: destroy at sleep, lazily
        # re-create (collective rendezvous, run in lockstep) on the first post-wake
        # forward. Default off: keep it resident (safe, no wake-side collective).
        symm = 0.0
        for key, buf in getattr(mega_buf, "_MEGA_BUF_CACHE", {}).items():
            try:
                symm += buf.buffer.numel() * buf.buffer.element_size()
            except Exception:
                pass
        if os.environ.get("RTP_LLM_SLEEP_FREE_MEGA_SYMM", "0") == "1":
            try:
                freed = mega_buf.release_mega_symm_buffers()
                notes.append(
                    f"mega_symm_buffer RELEASED ~{freed:.3f} GiB "
                    "(destroy+recreate; re-rendezvous on next forward)"
                )
            except Exception as e:
                notes.append(f"mega_symm_buffer release failed: {e}")
        elif symm:
            notes.append(
                f"mega_symm_buffer kept ~{symm / _GiB:.3f} GiB (symmetric-mem, not torch pool)"
            )
    except Exception as e:  # module may be absent on non-dsv4 models
        notes.append(f"mega_buf cache skip: {e}")
    return notes


def _snapshot_summary(device: object, top_n: int = 12) -> str:
    """Summarize ``torch.cuda.memory_snapshot`` for the given device.

    Reports total segment bytes, fully-free vs partial segments, and the top-N
    segments by size with their live fraction + largest live block -- the live
    blocks in large partially-free segments are exactly what pins physical memory
    that ``empty_cache`` cannot return.
    """
    try:
        dev_idx = torch.device(device).index
        segs = [s for s in torch.cuda.memory_snapshot() if s.get("device") == dev_idx]
    except Exception as e:
        return f"snapshot unavailable: {e}"
    if not segs:
        return "no segments"
    total = sum(s.get("total_size", 0) for s in segs)
    live = sum(s.get("allocated_size", 0) for s in segs)
    full_free = [s for s in segs if s.get("allocated_size", 0) == 0]
    partial = [
        s for s in segs if 0 < s.get("allocated_size", 0) < s.get("total_size", 0)
    ]
    free_in_partial = sum(
        s.get("total_size", 0) - s.get("allocated_size", 0) for s in partial
    )

    def _largest_live_block(seg: dict) -> int:
        return max(
            (
                b.get("size", 0)
                for b in seg.get("blocks", [])
                if b.get("state") == "active_allocated"
            ),
            default=0,
        )

    def _pool_id(seg: dict) -> tuple:
        # (0, 0) == default caching-allocator pool; anything else is a private
        # MemPool (TMS weights `_primary_mem_pool`) or a CUDA-graph private pool.
        # empty_cache() only returns fully-free segments of the DEFAULT pool to
        # the driver; free blocks trapped in a still-referenced private MemPool
        # are physically resident but empty_cache never releases them. This split
        # is exactly what attributes the sleep residual to default-vs-MemPool.
        pid = seg.get("segment_pool_id")
        if pid is None:
            return (-1, -1)  # field absent on this torch build
        return tuple(pid) if isinstance(pid, (list, tuple)) else (pid,)

    # Per-pool free-byte breakdown: which pool holds the physically-resident
    # free reserve that empty_cache cannot return.
    per_pool: dict[tuple, list[int]] = {}
    for s in segs:
        tot = s.get("total_size", 0)
        al = s.get("allocated_size", 0)
        agg = per_pool.setdefault(
            _pool_id(s), [0, 0, 0, 0]
        )  # tot, live, nseg, fully_free_segs
        agg[0] += tot
        agg[1] += al
        agg[2] += 1
        if al == 0:
            agg[3] += 1
    pool_lines = []
    for pid, (ptot, plive, nseg, nff) in sorted(
        per_pool.items(), key=lambda kv: kv[1][0], reverse=True
    ):
        pfree = ptot - plive
        kind = (
            "default" if pid == (0, 0) else ("absent" if pid == (-1, -1) else "MemPool")
        )
        pool_lines.append(
            f"    pool_id={pid} kind={kind} tot={ptot / _GiB:.2f}GiB live={plive / _GiB:.2f}GiB "
            f"free={pfree / _GiB:.2f}GiB segs={nseg} fully_free_segs={nff}"
        )

    top = sorted(segs, key=lambda s: s.get("total_size", 0), reverse=True)[:top_n]
    lines = []
    for s in top:
        tot = s.get("total_size", 0)
        al = s.get("allocated_size", 0)
        lines.append(
            f"    seg tot={tot / _MiB:.0f}MiB live={al / _MiB:.0f}MiB "
            f"({100.0 * al / tot if tot else 0:.0f}%) "
            f"pool={s.get('segment_type', '?')} pool_id={_pool_id(s)} "
            f"largest_live_block={_largest_live_block(s) / _MiB:.1f}MiB"
        )

    # Allocation backtraces for the biggest private-MemPool segments (only present
    # when RTP_LLM_RECORD_MEM_HISTORY=1 enabled torch's history recording). This
    # attributes the private-pool residual to its REAL allocator (symmetric-memory
    # vs weights-region vs a workspace) instead of inferring from pool size.
    def _seg_frames(seg: dict) -> list:
        # Prefer the segment-level frames; fall back to the first block's frames.
        frames = seg.get("frames") or []
        if not frames:
            for b in seg.get("blocks", []):
                if b.get("frames"):
                    frames = b["frames"]
                    break
        return frames

    def _frames_str(seg: dict) -> str:
        frames = _seg_frames(seg)
        if not frames:
            return "<no python frames (C++/engine alloc)>"
        picked = []
        for fr in frames:
            fn = fr.get("filename", "")
            name = fr.get("name", "")
            if (
                "rtp_llm" in fn
                or "symmetric_memory" in fn
                or "mem_pool" in name
                or "use_mem_pool" in name
                or "empty" == name
            ):
                picked.append(f"{fn.split('/')[-1]}:{fr.get('line', '?')}:{name}")
            if len(picked) >= 8:
                break
        if not picked:
            picked = [
                f"{fr.get('filename', '?').split('/')[-1]}:{fr.get('line', '?')}:{fr.get('name', '?')}"
                for fr in frames[:6]
            ]
        return " <- ".join(picked)

    def _is_kv(seg: dict) -> bool:
        # KV cache = C++ BlockPool alloc (tagged kv_cache, released at sleep).
        return any(
            "BlockPool" in (fr.get("name", "") or fr.get("filename", ""))
            for fr in _seg_frames(seg)
        )

    bt_lines: list[str] = []
    # The sleep residual is untagged live allocations that pause never unmaps.
    # KV cache (C++ BlockPool, tagged kv_cache) is released at sleep and is NOT
    # the residual, yet it dominates by raw size -- so bt it separately and give
    # the bulk of the budget to the NON-KV segments (python-model weight rebuilds
    # allocated outside weights_region), which are the actual untagged residual.
    # (0,0) non-KV segs are the prime suspects; (0,2)/private-pool segs are
    # weights_region-tagged (paused) and shown for completeness.
    non_kv = sorted(
        (s for s in segs if not _is_kv(s)),
        key=lambda s: s.get("total_size", 0),
        reverse=True,
    )
    kv = sorted(
        (s for s in segs if _is_kv(s)),
        key=lambda s: s.get("total_size", 0),
        reverse=True,
    )
    # Dedicated view of the DEFAULT-pool (0,0) NON-KV segments: after weights are
    # tagged into the private pool, these untagged (0,0) allocations (C++ engine
    # buffers / workspaces) are what remains of the physical residual. Summarize
    # their total, then bt the largest, so we can tell tag-able workspaces from
    # genuinely-needed runtime buffers.
    default_non_kv = [s for s in non_kv if _pool_id(s) == (0, 0)]
    dnk_total = sum(s.get("total_size", 0) for s in default_non_kv)
    bt_lines.append(
        f"    [default-non-kv] (0,0) non-KV segs={len(default_non_kv)} "
        f"total={dnk_total / _GiB:.2f}GiB  <- untagged residual candidates"
    )
    for s in default_non_kv[:20]:
        bt_lines.append(
            f"    [bt-0,0] seg tot={s.get('total_size', 0) / _MiB:.0f}MiB "
            f"live={s.get('allocated_size', 0) / _MiB:.0f}MiB: {_frames_str(s)}"
        )
    for s in non_kv[:10]:
        bt_lines.append(
            f"    [bt] seg tot={s.get('total_size', 0) / _MiB:.0f}MiB "
            f"pool_id={_pool_id(s)}: {_frames_str(s)}"
        )
    for s in kv[:2]:
        bt_lines.append(
            f"    [bt][kv] seg tot={s.get('total_size', 0) / _MiB:.0f}MiB "
            f"pool_id={_pool_id(s)}: {_frames_str(s)}"
        )

    header = (
        f"segments={len(segs)} total={total / _GiB:.2f}GiB live={live / _GiB:.2f}GiB "
        f"fully_free_segs={len(full_free)} partial_segs={len(partial)} "
        f"free_stuck_in_partial={free_in_partial / _GiB:.2f}GiB"
    )
    out = header + "\n" + "\n".join(pool_lines) + "\n" + "\n".join(lines)
    if bt_lines:
        out += "\n" + "\n".join(bt_lines)
    return out


def release_and_trim(device: object, reason: str = "sleep") -> None:
    """Free known Python-held device caches, then return free segments to the driver.

    Logs physical driver-free before/after and a segment breakdown so the residual
    can be attributed. Never raises.
    """
    try:
        with torch.cuda.device(device):
            free_before, total = torch.cuda.mem_get_info(device)
            logging.info(
                "[SleepReclaim][%s] BEFORE driver_free=%.0fMiB total=%.0fMiB\n%s",
                reason,
                free_before / _MiB,
                total / _MiB,
                _snapshot_summary(device),
            )
            notes = _clear_module_device_caches()
            gc.collect()
            torch.cuda.synchronize()
            try:
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001 - best-effort teardown
                logging.warning(
                    "[SleepReclaim][%s] empty_cache failed (ignored): %s", reason, e
                )
            free_after = torch.cuda.mem_get_info(device)[0]
            logging.info(
                "[SleepReclaim][%s] AFTER driver_free=%.0fMiB (reclaimed %.0fMiB) | %s\n%s",
                reason,
                free_after / _MiB,
                (free_after - free_before) / _MiB,
                "; ".join(notes) if notes else "no caches cleared",
                _snapshot_summary(device),
            )
    except Exception as e:  # noqa: BLE001 - never fail the sleep hook
        logging.warning(
            "[SleepReclaim][%s] release_and_trim failed (ignored): %s", reason, e
        )
