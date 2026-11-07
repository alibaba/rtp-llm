from __future__ import annotations

import hashlib
import json
import math
import time

import msgspec

from .SGLangCache import SGLangCacheState


class SGLangState(msgspec.Struct, tag=True, kw_only=True, array_like=True):
    data: bytes


def workload_snapshot(scheduler, batch=None, prefill_started=None):
    running = {
        req.rid: req for req in scheduler.running_batch.reqs if not req.finished()
    }
    if batch is not None:
        running.update((req.rid, req) for req in batch.reqs if not req.finished())
    if scheduler.chunked_req is not None and not scheduler.chunked_req.finished():
        running[scheduler.chunked_req.rid] = scheduler.chunked_req
    waiting = {
        req.rid: req
        for req in scheduler.waiting_queue
        if req.rid not in running and not req.finished()
    }
    prefill_ids = set()
    if batch is not None and batch.forward_mode.is_extend_without_speculative():
        prefill_ids.update(req.rid for req in batch.reqs)
    if scheduler.chunked_req is not None:
        prefill_ids.add(scheduler.chunked_req.rid)

    def contributions(requests, *, waiting_for_prefill=False):
        context = tokens = squares = products = prefill = 0
        decode_count = decode_generated = decode_remaining = prompt_tokens = 0
        for req in requests:
            prompt = len(req.origin_input_ids)
            generated = len(req.output_ids)
            prompt_tokens += prompt
            if not generated or waiting_for_prefill or req.rid in prefill_ids:
                sequence_length = prompt + generated
                hit = min(sequence_length, req.prefix_len)
                miss = max(0, sequence_length - hit)
                prefill += int(miss > 0)
                context += hit
                tokens += miss
                squares += miss * miss
                products += hit * miss
            else:
                decode_count += 1
                decode_generated += generated
                decode_remaining += max(
                    0, req.sampling_params.max_new_tokens - generated
                )
        return {
            "prefill_request_count": prefill,
            "context_token_count": context,
            "token_count": tokens,
            "token_square_sum": squares,
            "context_remaining_prefill_token_product_sum": products,
            "decode_count": decode_count,
            "decode_generated": decode_generated,
            "decode_remaining": decode_remaining,
            "prompt_tokens": prompt_tokens,
        }

    run = contributions(running.values())
    wait = contributions(waiting.values(), waiting_for_prefill=True)
    result = {
        "event_time_us": time.time_ns() // 1000,
        "batch_count": int(bool(running)),
        "request_count": len(running) + len(waiting),
        "running_request_count": len(running),
        "waiting_request_count": len(waiting),
        "max_running_request_count": scheduler.max_running_requests,
        "running_capacity_slot_count": len(running),
        "running_elapsed_time_ms": sum(
            max(0, time.monotonic_ns() - started) / 1e6
            for rid, started in (prefill_started or {}).items()
            if rid in running and (not running[rid].output_ids or rid in prefill_ids)
        ),
        "running_decode_request_count": run["decode_count"],
        "running_decode_generated_token_count": run["decode_generated"],
        "running_decode_remaining_token_budget": run["decode_remaining"],
        "total_prompt_token_count": run["prompt_tokens"] + wait["prompt_tokens"],
        "in_flight_scheduled_token_count": 0,
        "admitted_schedule_batch_id": 0,
        "admitted_local_batch_id": 0,
        "admitted_request_count": 0,
        "admitted_pending_cache_tracked": False,
        "admitted_request_ids": [],
    }
    for suffix in (
        "prefill_request_count",
        "context_token_count",
        "token_count",
        "token_square_sum",
        "context_remaining_prefill_token_product_sum",
    ):
        name = (
            suffix
            if suffix.startswith("prefill_") or suffix.startswith("context_remaining")
            else "prefill_" + suffix
        )
        result["running_capacity_" + name] = run[suffix]
        result["waiting_capacity_" + name] = wait[suffix]
    result.update(
        {
            "remaining_prefill_request_count": run["prefill_request_count"]
            + wait["prefill_request_count"],
            "remaining_prefill_context_token_count": run["context_token_count"]
            + wait["context_token_count"],
            "remaining_prefill_token_count": run["token_count"] + wait["token_count"],
            "remaining_prefill_token_square_sum": run["token_square_sum"]
            + wait["token_square_sum"],
            "context_remaining_prefill_token_product_sum": run[
                "context_remaining_prefill_token_product_sum"
            ]
            + wait["context_remaining_prefill_token_product_sum"],
            "remaining_total_context_token_count": run["context_token_count"]
            + wait["context_token_count"],
            "remaining_total_token_count": run["token_count"] + wait["token_count"],
            "remaining_total_token_square_sum": run["token_square_sum"]
            + wait["token_square_sum"],
            "context_remaining_total_token_product_sum": run[
                "context_remaining_prefill_token_product_sum"
            ]
            + wait["context_remaining_prefill_token_product_sum"],
        }
    )
    return result


class SGLangObserver:
    def __init__(self, scheduler):
        from sglang.srt.runtime_context import get_parallel

        self.scheduler = scheduler
        parallel = get_parallel()
        self.enabled = (
            parallel.pp_rank == 0
            and parallel.attn_tp_rank == 0
            and parallel.attn_cp_rank == 0
        )
        if not self.enabled:
            return
        self.total_blocks = scheduler.max_total_num_tokens // scheduler.page_size
        cache = scheduler.tree_cache
        from sglang.srt.mem_cache.radix_cache import RadixCache
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        if type(cache) not in (RadixCache, UnifiedRadixCache):
            raise ValueError(
                "SGLang NaviSched requires the built-in Full KV radix cache"
            )
        self.hybrid = len(getattr(cache, "tree_components", (0,))) > 1
        self.linear_step = 1
        if hasattr(cache, "tree_components"):
            from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

            if ComponentType.FULL not in cache.tree_components:
                raise ValueError("NaviSched requires a Full KV cache component")
            if set(cache.tree_components) - {
                ComponentType.FULL,
                ComponentType.SWA,
                ComponentType.MAMBA,
            }:
                raise ValueError(
                    "SGLang NaviSched does not support this auxiliary cache component"
                )
            for component in cache.components.values():
                grid = getattr(component, "mamba_checkpoint_grid", scheduler.page_size)
                if grid % scheduler.page_size:
                    raise ValueError(
                        "Mamba checkpoint grid must contain complete KV pages"
                    )
                self.linear_step = math.lcm(
                    self.linear_step, grid // scheduler.page_size
                )
        self.cache = SGLangCacheState(
            self,
            self.total_blocks,
            scheduler.page_size,
            hybrid_tail_required=self.hybrid,
        )
        self._last_report_ns = 0
        self.prefill_started = {}
        self.pending_timings = []
        native_take = scheduler.tree_cache.take_events

        def take_events():
            events = native_take()
            for event in events:
                self.cache.on_event(event, device_ready=not self._has_pending_loads())
            if not self._has_pending_loads():
                self.cache.finish_loads()
            return events

        scheduler.tree_cache.take_events = take_events
        if self.hybrid:
            self._observe_hybrid(cache)
        self.send(
            {
                "kind": "ready",
                "total_blocks": self.total_blocks,
                "page_size": scheduler.page_size,
                "hybrid_tail_required": self.hybrid,
                "linear_step": self.linear_step,
                "latency_fingerprint": self._latency_fingerprint(),
            }
        )
        self.report(force=True)

    def _latency_fingerprint(self):
        import torch
        from sglang.version import __version__

        args = self.scheduler.server_args
        device = torch.cuda.get_device_properties(torch.cuda.current_device())
        values = {
            "framework": "sglang",
            "framework_version": __version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": device.name,
            "gpu_memory_bytes": device.total_memory,
            "compute_capability": [device.major, device.minor],
            "model_config": self.scheduler.model_config.hf_config.to_dict(),
        }
        for name in (
            "model_path",
            "revision",
            "dtype",
            "quantization",
            "kv_cache_dtype",
            "tp_size",
            "attn_cp_size",
            "ep_size",
            "attention_backend",
            "sampling_backend",
            "disable_overlap_schedule",
            "disable_cuda_graph",
            "page_size",
            "chunked_prefill_size",
            "max_running_requests",
            "mem_fraction_static",
        ):
            values[name] = getattr(args, name, None)
        values["speculative"] = {
            name: getattr(args, name)
            for name in dir(args)
            if name.startswith("speculative_")
        }
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
        # Native latency resources accept fingerprints of at most 4096 bytes.
        if len(encoded.encode()) > 4096:
            return "sglang:sha256:" + hashlib.sha256(encoded.encode()).hexdigest()
        return encoded

    def _observe_hybrid(self, cache):
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

        native_match = cache._finalize_match_result
        native_insert = cache.insert
        native_action = cache._apply_cache_action

        def finalize_match(params, result):
            value = native_match(params, result)
            if not self._has_pending_loads():
                self.cache.on_match(params.key, value)
            return value

        def insert(params):
            value = native_insert(params)
            # The publisher remains the only consumer of placement events. Flush
            # it before publishing readiness for a newly inserted device tail.
            self.scheduler.kv_events_publisher.publish_kv_events()
            if params.key is not None and len(params.key):
                cache.match_prefix(MatchPrefixParams(key=params.key))
            return value

        def apply_action(action):
            from sglang.srt.mem_cache.unified_cache.cache_action import ComponentAction

            if isinstance(action, ComponentAction) or type(action).__name__ in {
                "FreeDeviceKV",
                "FreeDeviceKVFullOnly",
            }:
                self.cache.invalidate_tails()
            return native_action(action)

        cache._finalize_match_result = finalize_match
        cache.insert = insert
        cache._apply_cache_action = apply_action

    def _has_pending_loads(self):
        cache = self.scheduler.tree_cache
        pipeline = getattr(cache, "buffer_pipeline", None)
        return bool(
            getattr(cache, "ongoing_load_back", None)
            or getattr(pipeline, "ongoing_buffer_load_back", None)
        )

    def send(self, data):
        message = SGLangState(data=msgspec.msgpack.encode(data))
        self.scheduler.ipc_channels.send_to_tokenizer.send_output(message)

    def report_cache_inserted(self, hashes):
        if hashes:
            self.send({"kind": "cache", "inserted": hashes, "removed": []})

    def report_cache_removed(self, hashes):
        if hashes:
            self.send({"kind": "cache", "inserted": [], "removed": hashes})

    def report_cache_updated(
        self, added, removed, full_added, full_removed, tail_added, tail_removed
    ):
        self.send(
            {
                "kind": "cache_ready",
                "values": [
                    added,
                    removed,
                    full_added,
                    full_removed,
                    tail_added,
                    tail_removed,
                ],
            }
        )

    def report(self, batch=None, force=False):
        if not self.enabled:
            return
        self._report_completed_timings()
        if not self._has_pending_loads():
            self.cache.finish_loads()
        now = time.monotonic_ns()
        if not force and now - self._last_report_ns < 100_000_000:
            return
        self._last_report_ns = now
        requests = list(self.scheduler.running_batch.reqs) + list(
            self.scheduler.waiting_queue
        )
        if batch is not None:
            requests.extend(batch.reqs)
        if self.scheduler.chunked_req is not None:
            requests.append(self.scheduler.chunked_req)
        requests = [req for req in requests if not req.finished()]
        self.send(
            {
                "kind": "workload",
                "values": workload_snapshot(
                    self.scheduler, batch, self.prefill_started
                ),
                "request_ids": sorted({req.rid for req in requests}),
            }
        )
        current = {req.rid for req in requests if not req.output_ids}
        self.prefill_started = {
            rid: started
            for rid, started in self.prefill_started.items()
            if rid in current
        }

    def _report_completed_timings(self):
        pending = []
        for start, end, prefill, decode in self.pending_timings:
            if end.query():
                self.send({"kind": "execution_finished"})
                self.send(
                    {
                        "kind": "execution",
                        "elapsed_ms": start.elapsed_time(end),
                        "prefill_tokens": prefill,
                        "decode_tokens": decode,
                    }
                )
            else:
                pending.append((start, end, prefill, decode))
        self.pending_timings = pending

    def close(self):
        if self.enabled:
            self.send({"kind": "stopped"})
