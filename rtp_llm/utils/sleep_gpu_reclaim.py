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

import torch

_MiB = 1024.0 * 1024.0
_GiB = 1024.0**3


def _cuda_graph_baked() -> bool:
    """Return whether any process-local graph requires stable allocations."""
    try:
        from rtp_llm.models_py.utils.cuda_graph_state import cuda_graph_baked

        graph_baked = bool(cuda_graph_baked())
    except Exception:
        # If the optional Python state cannot be imported, fail closed.  The C++
        # hook independently protects allocator emptyCache in graph mode.
        graph_baked = True
    if graph_baked:
        return True
    try:
        from rtp_llm.models_py.modules.dsv4.moe import mega_buf

        return bool(mega_buf.mega_buffers_graph_baked())
    except Exception:
        return False


def _optional_release_allowed(graph_baked: bool) -> bool:
    """Explicit opt-in release, never allowed for graph-baked pointers.

    ``RTP_LLM_SLEEP_FREE_RUNTIME_CACHES`` intentionally defaults to ``0``.
    Operators can opt into reclaim on a no-graph role, while a graph role always
    wins the safety check.
    """
    try:
        from rtp_llm.models_py.utils.cuda_graph_state import (
            runtime_cache_release_enabled,
        )

        release_enabled = runtime_cache_release_enabled()
    except Exception:
        release_enabled = False
    return not graph_baked and release_enabled


def _clear_module_device_caches() -> list[str]:
    """Drop long-lived Python-held device tensor caches so their segments free.

    Returns a list of human-readable notes for logging. MegaMoE symmetric-memory
    destruction is non-collective, but recreation requires a collective rendezvous
    and is therefore opt-in.
    """
    notes: list[str] = []
    graph_baked = _cuda_graph_baked()
    # DSV4 decode CUDA graphs capture the freqs_cis device pointer inside the
    # indexSelect/fused RoPE launches. Replacing that tensor at wake would leave
    # graph replay using the old (possibly unmapped) address and can produce a
    # warp illegal address with no Python traceback. Keep it resident until the
    # graph lifecycle grows an explicit invalidate+recapture protocol.
    # TODO(sleep): re-enable RoPE release only after graph invalidation/recapture
    # is rank-symmetric and completes before any post-wake replay.
    notes.append("DSV4 RoPE caches KEPT (CUDA-graph pointer stability)")

    if _optional_release_allowed(graph_baked):
        try:
            clear_cublas = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear_cublas is not None:
                clear_cublas()
                notes.append("cuBLAS workspaces RELEASED")
            else:
                notes.append("cuBLAS workspace clear unavailable")
        except Exception as e:
            notes.append(f"cuBLAS workspace release skipped: {e}")
    else:
        reason = "CUDA-graph pointer stability" if graph_baked else "env disabled"
        notes.append(f"cuBLAS workspaces KEPT ({reason})")
    try:
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
            CudaFp8DeepGEMMLinear,
        )

        if _optional_release_allowed(graph_baked):
            scale_bytes = CudaFp8DeepGEMMLinear.release_runtime_caches_for_sleep()
            notes.append(
                f"DeepGEMM runtime scale caches RELEASED {scale_bytes / _MiB:.1f} MiB"
            )
        else:
            reason = "CUDA-graph pointer stability" if graph_baked else "env disabled"
            notes.append(f"DeepGEMM runtime scale caches KEPT ({reason})")
    except Exception as e:
        notes.append(f"DeepGEMM runtime scale cache release skipped: {e}")
    try:
        from rtp_llm.models_py.distributed.symm_mem import (
            release_symm_mem_communicator_for_sleep,
        )

        if _optional_release_allowed(graph_baked):
            symm_bytes = release_symm_mem_communicator_for_sleep()
            notes.append(
                f"TP symmetric-memory communicator RELEASED {symm_bytes / _MiB:.1f} MiB"
            )
        else:
            reason = "CUDA-graph pointer stability" if graph_baked else "env disabled"
            notes.append(f"TP symmetric-memory communicator KEPT ({reason})")
    except Exception as e:
        notes.append(f"TP symmetric-memory communicator release skipped: {e}")
    try:
        from rtp_llm.models_py.modules.dsv4.moe import mega_buf

        # The two Mega MoE device buffers -- the symmetric-memory dispatch buffer
        # (``_mega_buf``, ~4.4 GiB/rank) and the bf16 output staging buffer
        # (``_mega_y``) -- both feed the same MoE kernel and have their pointers
        # baked into the same captured decode graph, so they are ONE coupled unit:
        # released together or kept together. Release uses the same unified
        # switch as the other runtime caches, ``RTP_LLM_SLEEP_FREE_RUNTIME_CACHES=1``
        # -- ``release_mega_symm_buffers()`` drops
        # the symm buffer AND the output buffer (destroy + collective re-rendezvous
        # lazily on the first post-wake forward, run in lockstep). The symm buffer's
        # cross-rank state (CUDA multicast binding + peer P2P imports, keyed to the
        # physical VMM handle) cannot be VMM-paused at a fixed VA, which is why the
        # release is opt-in rather than automatic.
        #
        # Hard safety interlock: when the forward is captured into a CUDA graph both
        # buffers MUST stay resident across sleep/wake -- freeing a baked buffer
        # dangles the VA a post-wake replay writes into (illegal access on every
        # rank -> abort), and the graph replay never runs Python so the lazy
        # re-create cannot fire. ``mega_buffers_graph_baked()`` keeps them resident
        # unconditionally in that mode, making the crash impossible-by-construction
        # even if the env is set on a graph engine by mistake (see mega_buf).
        graph_baked = graph_baked or mega_buf.mega_buffers_graph_baked()
        output_gib = mega_buf.mega_output_buffer_gib()
        symm = 0.0
        for _key, buf in getattr(mega_buf, "_MEGA_BUF_CACHE", {}).items():
            try:
                symm += buf.buffer.numel() * buf.buffer.element_size()
            except Exception:
                pass
        symm_gib = symm / _GiB

        if graph_baked:
            notes.append(
                f"mega buffers kept ~{output_gib:.3f} GiB output + ~{symm_gib:.3f} GiB "
                "symm (baked into CUDA graph)"
            )
        elif _optional_release_allowed(graph_baked):
            try:
                freed = mega_buf.release_mega_symm_buffers()
                notes.append(
                    f"mega buffers RELEASED ~{freed:.3f} GiB symm + ~{output_gib:.3f} GiB "
                    "output (destroy+recreate; re-rendezvous on next forward)"
                )
            except Exception as e:
                notes.append(f"mega buffers release failed: {e}")
        else:
            notes.append(
                f"mega buffers kept ~{output_gib:.3f} GiB output + ~{symm_gib:.3f} GiB "
                "symm (RTP_LLM_SLEEP_FREE_RUNTIME_CACHES not set)"
            )
    except Exception as e:  # module may be absent on non-dsv4 models
        notes.append(f"mega_buf cache skip: {e}")
    return notes


def _pool_id(seg: dict) -> tuple:
    """``segment_pool_id`` as a comparable tuple.

    ``(0, 0)`` == default caching-allocator pool; anything else is a private
    MemPool (TMS weights ``_primary_mem_pool``) or a CUDA-graph private pool.
    ``empty_cache()`` only returns fully-free segments of the DEFAULT pool to the
    driver; free blocks trapped in a still-referenced private MemPool are
    physically resident but ``empty_cache`` never releases them. This split is
    exactly what attributes the sleep residual to default-vs-MemPool.
    """
    pid = seg.get("segment_pool_id")
    if pid is None:
        return (-1, -1)  # field absent on this torch build
    return tuple(pid) if isinstance(pid, (list, tuple)) else (pid,)


def _seg_frames(seg: dict) -> list:
    """Allocation traceback for a segment, if history recording captured one.

    Prefers the segment-level frames and falls back to the first block's. Only
    populated when ``RTP_LLM_RECORD_MEM_HISTORY=1`` turned on torch's history
    recording; see ``gpu_mem_probe._maybe_enable_mem_history``.
    """
    frames = seg.get("frames") or []
    if not frames:
        for b in seg.get("blocks", []):
            if b.get("frames"):
                frames = b["frames"]
                break
    return frames


# Frames contributed by the recording machinery and the allocator itself. With
# stacks="all" every C++ traceback starts with a dozen of these, so the naive
# "first N frames" fallback shows only plumbing and hides the actual caller.
_NOISE = (
    "torch::unwind",
    "CapturedTraceback",
    "gather_with_cpp",
    "CUDACachingAllocator",
    "CachingAllocator::",
    "c10::cuda::CUDACachingAllocator",
    "empty_generic",
    "at::detail::empty_cuda",
    # The dispatcher trampoline between `torch.empty` and the allocator. Every
    # single traceback carries ~8 of these, each a multi-hundred-character
    # template instantiation, so leaving them in both hides the real caller
    # behind the 8-frame cap and makes the log line unreadable.
    "at::native::empty",
    "wrapper_CUDA",
    "wrap_kernel_functor",
    "at::_ops::",
    "c10::impl::",
    "at::(anonymous namespace)",
)

# Frame renderings longer than this are template soup, not information. Truncating
# keeps one segment's attribution on one readable log line.
_FRAME_CHARS = 90


def _fmt_frame(fr: dict) -> str:
    name = fr.get("name") or "?"
    if len(name) > _FRAME_CHARS:
        name = name[:_FRAME_CHARS] + "..."
    return f"{fr.get('filename', '?').split('/')[-1]}:{fr.get('line', '?')}:{name}"


def _frames_str(seg: dict) -> str:
    """Render a segment's/block's allocation traceback down to the actionable frames.

    Preference order is deliberate: Python frames first, then rtp_llm C++ frames,
    then anything non-noise. Some real owners have no Python frame at all (e.g. the
    CUDA-graph capture holds allocated straight from ``cuda_graph_runner.cc``), so
    the C++ tier cannot be dropped -- but when a Python frame exists it is always
    the one that says which component to change.
    """
    frames = _seg_frames(seg)
    if not frames:
        # Ambiguous on purpose: this is equally "allocated from C++ with no
        # Python frame" and "RTP_LLM_RECORD_MEM_HISTORY was not set".
        return "<no frames recorded>"
    frames = [
        fr for fr in frames if not any(n in (fr.get("name") or "") for n in _NOISE)
    ] or frames
    py = [fr for fr in frames if (fr.get("filename") or "").endswith(".py")]
    picked = [_fmt_frame(fr) for fr in (py or frames)[:8]]
    if not py:
        # No Python frame: prefer the rtp_llm C++ frames over unrelated plumbing,
        # but keep the plain head as a last resort so "recording on, owner unknown"
        # still reads differently from "recording off".
        rtp = [fr for fr in frames if "rtp_llm" in (fr.get("filename") or "")]
        if rtp:
            picked = [_fmt_frame(fr) for fr in rtp[:8]]
        else:
            picked = picked[:6]
    return " <- ".join(picked)


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
            graph_baked = _cuda_graph_baked()
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
            if graph_baked:
                logging.info(
                    "[SleepReclaim][%s] empty_cache kept skipped (CUDA-graph pointers must remain stable)",
                    reason,
                )
            else:
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
