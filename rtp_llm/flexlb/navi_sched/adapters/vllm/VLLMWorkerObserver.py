"""Observe actual TPU planner decisions without changing execution inputs."""

import copy
import hashlib
import json
import logging
import os
import time

from vllm.v1.outputs import AsyncModelRunnerOutput

from . import ExecutionTimingProbe as _execution_timing_probe
from .VLLMCachePolicy import is_vllm_native_pd_prefill

# Use the same configured vLLM stream as the Engine observer.
logger = logging.getLogger("vllm.navi_sched.worker")


class _ObservedAsyncOutput(AsyncModelRunnerOutput):
    def __init__(self, output, geometry):
        self._output = output
        self._geometry = geometry

    def get_output(self):
        output = copy.copy(self._output.get_output())
        output.execution_geometry = self._geometry
        return output


class _WorkerObserver:
    def __init__(self, worker):
        _execution_timing_probe.install_gc_timing_audit()
        self._worker = worker
        self._eligibility = None
        self._chunks = []
        self._attention_policy = None
        self._timing_context = None

    def __getattr__(self, name):
        return getattr(self._worker, name)

    def _observe_attention_policy(self, runner):
        if self._attention_policy is not None:
            return self._attention_policy
        config = runner.vllm_config
        implementations = {}
        for layer in config.compilation_config.static_forward_context.values():
            impl = getattr(layer, "impl", None)
            if impl is None or not hasattr(impl, "_pcp_streaming"):
                continue
            key = (int(impl.num_heads), int(impl.head_size), bool(impl._pcp_streaming))
            implementations[key] = implementations.get(key, 0) + 1
        specs = {}
        for group in runner.kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            if not hasattr(spec, "num_kv_heads") or not hasattr(spec, "head_size"):
                continue
            key = (int(spec.block_size), int(spec.num_kv_heads), int(spec.head_size))
            specs[key] = specs.get(key, 0) + 1
        if not implementations or not specs:
            raise ValueError("attention implementations or cache specs unavailable")
        policy = {
            "version": 1,
            "tensor_parallel_size": int(config.parallel_config.tensor_parallel_size),
            "implementations": [
                {
                    "num_query_heads_before_mesh_sharding": heads,
                    "head_size": dim,
                    "pcp_streaming": streaming,
                    "layers": count,
                }
                for (heads, dim, streaming), count in sorted(implementations.items())
            ],
            "cache_specs": [
                {
                    "page_size": page,
                    "num_kv_heads": heads,
                    "head_size": dim,
                    "groups": count,
                }
                for (page, heads, dim), count in sorted(specs.items())
            ],
        }
        self._attention_policy = policy
        return policy

    def _observe_request_spans(self, decision, kwargs):
        batch = kwargs["input_batch"]
        start = kwargs["start_index"]
        req_ids = [
            rid
            for rid in batch.req_ids[start : start + kwargs["num_reqs"]]
            if rid is not None
        ]
        owner_starts = decision.token_owner_start_offsets_per_req
        if len(req_ids) != len(decision.spans) or len(owner_starts) != len(req_ids):
            raise ValueError("request spans and Worker order differ")
        return [
            {
                "request_id": str(rid),
                "computed_tokens": int(span.computed_tokens),
                "scheduled_tokens": int(span.scheduled_tokens),
                "prompt_tokens": int(span.prompt_tokens),
                "absolute_query_start": int(span.absolute_query_start),
                "token_owner_start": int(owner),
            }
            for rid, span, owner in zip(req_ids, decision.spans, owner_starts)
        ]

    def _observe_planner(self):
        runner = getattr(self._worker, "model_runner", None)
        planner = getattr(runner, "sequence_layout_planner", None)
        eligibility = getattr(planner, "eligibility", None)
        if eligibility is None or eligibility is self._eligibility:
            return
        evaluate = getattr(eligibility, "evaluate_runner_chunk", None)
        if evaluate is None:
            return

        def observe(*args, **kwargs):
            decision = evaluate(*args, **kwargs)
            try:
                if decision.enabled and decision.local_token_counts is not None:
                    chunk = {
                        "scheduled_tokens": sum(
                            int(span.scheduled_tokens) for span in decision.spans
                        ),
                        "pcp_size": int(decision.pcp_size),
                        "interleave_size": int(decision.interleave_size),
                        "max_local_tokens": max(
                            int(value) for value in decision.local_token_counts
                        ),
                        "local_padded_tokens": int(decision.local_padded_tokens),
                        "global_padded_tokens": int(decision.global_padded_tokens),
                        "dp_target_bucket": kwargs.get("dp_target_bucket"),
                    }
                    # Read only CPU metadata already chosen by the actual Worker.
                    # Failure of this diagnostic must not discard base geometry
                    # or change the planner's decision/execution inputs.
                    try:
                        chunk["request_spans"] = self._observe_request_spans(
                            decision, kwargs
                        )
                        chunk["attention_policy"] = self._observe_attention_policy(
                            runner
                        )
                    except Exception as exc:
                        chunk["attention_audit_error"] = type(exc).__name__
                        logger.exception("failed to observe TPU attention metadata")
                    self._chunks.append(chunk)
            except Exception:
                logger.exception("failed to observe TPU execution geometry")
            return decision

        eligibility.evaluate_runner_chunk = observe
        self._eligibility = eligibility

    def _attach(self, output):
        if output is None or not self._chunks:
            return output
        geometry = {"version": 1, "chunks": self._chunks}
        self._chunks = []
        if isinstance(output, AsyncModelRunnerOutput):
            _execution_timing_probe.instrument_async_output(output)
            return _ObservedAsyncOutput(output, geometry)
        output = copy.copy(output)
        output.execution_geometry = geometry
        return output

    def _begin_timing(self, scheduler_output):
        self._timing_context = None
        if not logger.isEnabledFor(logging.INFO):
            return
        try:
            tokens = scheduler_output.total_num_scheduled_tokens
            if tokens <= 0:
                return
            # Hash only request identities, never payloads or transfer metadata.
            ids = sorted(scheduler_output.num_scheduled_tokens)
            encoded = json.dumps(ids, separators=(",", ":")).encode()
            self._timing_context = {
                "request_set_sha256": hashlib.sha256(encoded).hexdigest(),
                "request_count": len(ids),
                "scheduled_tokens": tokens,
            }
        except Exception:
            # A diagnostic failure cannot change the underlying Worker call.
            self._timing_context = None

    def _finish_timing(self, stage, started, cpu_started, success):
        if self._timing_context is None:
            return
        try:
            finished = time.perf_counter_ns()
            cpu_finished = time.thread_time_ns()
            fields = dict(
                self._timing_context,
                version=1,
                stage=stage,
                pid=os.getpid(),
                started_perf_ns=started,
                finished_perf_ns=finished,
                thread_cpu_started_ns=cpu_started,
                thread_cpu_finished_ns=cpu_finished,
                event_unix_ns=time.time_ns(),
                success=success,
            )
            _execution_timing_probe._diagnostic_info(
                logger,
                "[navi-vllm-worker-stage-audit] %s",
                json.dumps(fields, separators=(",", ":"), sort_keys=True),
            )
        except Exception:
            pass

    def _start_stage_timing(self):
        if self._timing_context is None:
            return 0, 0
        try:
            return time.perf_counter_ns(), time.thread_time_ns()
        except Exception:
            self._timing_context = None
            return 0, 0

    def compile_or_warm_up_model(self, *args, **kwargs):
        result = self._worker.compile_or_warm_up_model(*args, **kwargs)
        _execution_timing_probe.install_worker_sampling_stage_audit(self._worker, self)
        _execution_timing_probe.install_worker_draft_compile_audit(self._worker)
        _execution_timing_probe.worker_startup_gc_sweep()
        return result

    def execute_model(self, scheduler_output, *args, **kwargs):
        self._begin_timing(scheduler_output)
        started, cpu_started = self._start_stage_timing()
        success = False
        try:
            self._chunks = []
            self._observe_planner()
            output = self._attach(
                self._worker.execute_model(scheduler_output, *args, **kwargs)
            )
            success = True
            return output
        finally:
            self._finish_timing("execute_model", started, cpu_started, success)
            if not success or output is not None:
                self._timing_context = None

    def sample_tokens(self, *args, **kwargs):
        started, cpu_started = self._start_stage_timing()
        success = False
        try:
            output = self._attach(self._worker.sample_tokens(*args, **kwargs))
            success = True
            return output
        finally:
            self._finish_timing("sample_tokens", started, cpu_started, success)
            self._timing_context = None


def create_vllm_worker_observer(vllm_config, worker):
    device = getattr(vllm_config, "device_config", None)
    if not is_vllm_native_pd_prefill() or getattr(device, "device_type", None) != "tpu":
        return worker
    return _WorkerObserver(worker)
