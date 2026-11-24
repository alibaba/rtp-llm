"""Continuous pressure against a prestarted RTP engine with the built Python SDK."""

import json
import math
import re
import struct
import threading
import time
import uuid
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

from ..config import (
    ExecutionLimits,
    mock_provider_enabled,
    mock_target_ppm,
    resolve_pressure_config,
)
from ..diagnostics import (
    ActivitySample,
    PressureCoverage,
    decode_pressure_response,
    raw_pressure_record,
    select_targets,
)
from ..ports import JsonlSink, PressurePoll
from ..results import PressureEvidence, PressureWriter
from ..seed_plan import build_seed_plan
from ..sequence import TieredRollingPlanner, freeze_seed_families
from ..session import DispatchClosed, PressureSession

REMOTE_DEFAULTS = {
    "mock_remote_cache": True,
    "mock_remote_cache_address": "127.0.0.1:18080",
    "mock_remote_cache_hit_ratio": 0.0,
    "mock_remote_cache_source_blocks": 64,
    "mock_remote_cache_read_delay_ms": 0,
    "mock_remote_cache_write_delay_ms": 0,
    "mock_remote_cache_callback_port": 19080,
    "mock_remote_cache_staging_size_mb": 256,
    "mock_remote_cache_max_inflight": 8,
    "mock_remote_cache_timeout_ms": 10000,
    "mock_memory_cache": False,
    "mock_disk_cache": False,
    "mock_memory_cache_hit_ratio": 0.0,
    "mock_disk_cache_hit_ratio": 0.0,
}


def _validate_provider(policy, tier, enabled, requested_ratio, engine_ratio):
    if requested_ratio != engine_ratio:
        raise ValueError(f"pressure and engine {tier} ratios disagree")
    if not enabled and (requested_ratio or policy == f"tree_then_mock_{tier}"):
        raise ValueError(f"requested {tier} provider is disabled")
    allowed = {
        "tree_pressure": (),
        "tree_then_mock_memory": ("memory",),
        "tree_then_mock_disk": ("memory", "disk"),
        "tree_then_mock_remote": ("memory", "disk", "remote"),
    }
    if enabled and tier not in allowed[policy]:
        raise ValueError(f"{tier} provider does not match pressure policy")


def pressure_settings(values):
    """Resolve RTP entry defaults. Mock preparation always uses real cache traffic."""
    if (
        values.get("family_schedule", "tiered_rolling") != "tiered_rolling"
        or values.get("rate_control", "adaptive_backlog") != "adaptive_backlog"
    ):
        raise ValueError(
            "RTP continuous pressure requires tiered_rolling and adaptive_backlog"
        )
    config = resolve_pressure_config(
        dict(
            seed_timeout_s=600,
            fill_timeout_s=600,
            fill_max_requests=100000,
            fill_no_progress_s=60,
            fill_input_tokens=102400,
            gpu_tolerance_blocks=1,
            activity_timeout_s=10,
            drain_timeout_s=60,
            capture_timeout_s=120,
        )
        | {
            "rate_control": "adaptive_backlog",
            "request_rate_min": 50,
            "request_rate_max": 400,
            "backlog_low": 64,
            "backlog_high": 192,
            "max_outstanding": 256,
            "cohort_size": 32,
            "family_schedule": "tiered_rolling",
            "memory_reuse_ratio": 0.5,
            "fill_host_max_ratio": 0.80,
            "profile_steps": 6,
        }
        | dict(values)
    )
    if config.evidence_policy == "mock_memory":
        raise ValueError(
            "RTP pressure requires real-fill preparation and real matching"
        )
    for tier in ("memory", "disk", "remote"):
        ratio = getattr(config, f"mock_{tier}_target_ratio")
        enabled = mock_provider_enabled(config, tier)
        _validate_provider(
            config.evidence_policy,
            tier,
            enabled,
            ratio,
            dict(config.mock_cache_settings).get(f"mock_{tier}_cache_hit_ratio", ratio),
        )
    if config.fill_host_max_ratio <= config.host_ratio:
        raise ValueError("fill_host_max_ratio must exceed host_ratio")
    if config.evidence_policy == "tree_then_mock_remote":
        options = dict(REMOTE_DEFAULTS)
        for name in ("memory", "disk", "remote"):
            ratio = getattr(config, f"mock_{name}_target_ratio")
            options[f"mock_{name}_cache_hit_ratio"] = ratio
            options[f"mock_{name}_cache"] = ratio > 0 or name == "remote"
        options.update(dict(config.mock_cache_settings))
        config = replace(config, mock_cache_settings=tuple(sorted(options.items())))
    return config


def _project(message):
    from google.protobuf.json_format import MessageToDict

    return MessageToDict(
        message, preserving_proto_field_name=True, use_integers_for_enums=True
    )


class RtpLlmPressureBackend:
    """Borrow explicit RPC clients, or own the two channels created by connect()."""

    def __init__(
        self,
        *,
        dash_stub,
        pressure_stub,
        messages,
        build_request,
        sampling,
        ranks,
        timeout,
    ):
        self.dash = dash_stub
        self.rpc = pressure_stub
        self.messages = messages
        self.build_request = build_request
        self.sampling = sampling
        self.ranks = tuple(ranks)
        if (
            not self.ranks
            or len(set(self.ranks)) != len(self.ranks)
            or any(type(r) is not int or r < 0 for r in self.ranks)
        ):
            raise ValueError("distinct nonnegative target ranks required")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("positive finite RPC timeout required")
        self.timeout = timeout
        self.channels = ExitStack()
        self.session_id = uuid.uuid4().hex
        self.instances = {}
        self.cursors = {}
        self.remote_epoch = {}
        self.local_targets = {}
        self.owner_ranks = set()
        self.remote_binding = None

    @classmethod
    def connect(cls, *, dash_endpoint, pressure_endpoint, ranks, timeout=60):
        try:
            import grpc

            from rtp_llm.cpp.model_rpc.proto import (
                model_rpc_service_pb2,
                model_rpc_service_pb2_grpc,
            )
            from rtp_llm.dash_sc.client import (
                build_model_infer_request,
                dash_sc_grpc_client_channel_options,
            )
            from rtp_llm.dash_sc.codec import SamplingParams
            from rtp_llm.dash_sc.proto import predict_v2_pb2_grpc
        except ImportError as exc:
            raise RuntimeError(
                "RTP pressure requires the built/installed DashSC SDK and generated model RPC Python protobufs"
            ) from exc
        with ExitStack() as channels:
            dash = channels.enter_context(
                grpc.insecure_channel(
                    dash_endpoint, options=dash_sc_grpc_client_channel_options()
                )
            )
            pressure = channels.enter_context(grpc.insecure_channel(pressure_endpoint))
            backend = cls(
                dash_stub=predict_v2_pb2_grpc.GRPCInferenceServiceStub(dash),
                pressure_stub=model_rpc_service_pb2_grpc.RpcServiceStub(pressure),
                messages=model_rpc_service_pb2,
                build_request=build_model_infer_request,
                sampling=SamplingParams(max_new_tokens=1, min_new_tokens=1, top_k=1),
                ranks=ranks,
                timeout=timeout,
            )
            backend.channels = channels.pop_all()
            return backend

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.channels.close()

    def _snapshot(self, timeout, **options):
        sent = time.monotonic()
        response = self.rpc.GetCachePressure(
            self.messages.CachePressureRequestPB(target_ranks=self.ranks, **options),
            timeout=timeout,
        )
        received = time.monotonic()
        return response, sent, received

    def startup(self, config):
        if config.evidence_policy == "mock_memory":
            raise ValueError("RTP pressure requires real-fill preparation")
        deadline = time.monotonic() + min(10.0, config.fill_timeout_s)
        attempts = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("startup pressure retry deadline exceeded")
            response, sent, received = self._snapshot(min(self.timeout, remaining))
            raw = raw_pressure_record(response, sent, received, to_dict=_project)
            try:
                samples = decode_pressure_response(
                    response, sent, received, self.ranks, to_dict=_project
                ).samples
                break
            except ValueError as error:
                errors = {
                    f"rank {r.world_rank}: {r.error}": r.error.lower()
                    for r in response.ranks
                    if r.error
                }
                transient = (
                    "connect failed",
                    "connection refused",
                    "failed to connect",
                    "temporarily unavailable",
                )
                remaining = deadline - time.monotonic()
                if (
                    str(error) not in errors
                    or remaining <= 0
                    or not all(
                        any(marker in value for marker in transient)
                        for value in errors.values()
                    )
                ):
                    raise
                attempts.append(raw)
                time.sleep(min(0.25, remaining))
        raw["preceding_transient_attempts"] = attempts
        targets = select_targets(samples, self.ranks)
        PressureCoverage(samples, config)
        if any(
            pool.tier == 2
            and pool.capacity
            and (pool.used - pool.pending_release) / pool.capacity
            >= config.fill_host_max_ratio
            for sample in samples
            for pool in sample.pools
        ):
            raise ValueError(
                "startup Host occupancy already reaches preparation headroom"
            )
        if config.evidence_policy == "tree_then_mock_remote":
            _validate_remote(samples, config, self.ranks)
        owners = {t.owner_rank for t in targets}
        ppm = mock_target_ppm(config)
        for sample in samples:
            if bool(sample.remote_runtime) != (
                config.evidence_policy == "tree_then_mock_remote"
            ):
                raise ValueError("Remote capability differs from selected policy")
            for tier, number in (("memory", 2), ("disk", 3)):
                active = any(
                    pool.tier == number
                    and pool.mock_source_ref_blocks > 0
                    and pool.mock_source_ref_bytes > 0
                    for pool in sample.pools
                )
                enabled = mock_provider_enabled(config, tier)
                if sample.rank in owners and config.evidence_policy != "tree_pressure":
                    actual = sample.local_mock_targets
                    ratio = getattr(config, f"mock_{tier}_target_ratio")
                    if (
                        actual is None
                        or (
                            getattr(actual, tier + "_enabled"),
                            getattr(actual, tier + "_ratio"),
                            getattr(actual, tier + "_ppm"),
                        )
                        != (enabled, ratio, ppm[tier])
                        or (ratio and not enabled)
                    ):
                        raise ValueError(
                            f"owner {sample.rank} {tier} targets differ from requested settings or are missing"
                        )
                    if enabled and (
                        sample.local_store is None
                        or getattr(
                            (
                                sample.mock_memory
                                if tier == "memory"
                                else sample.mock_cache
                            ),
                            "terminal_outcome_version",
                            0,
                        )
                        != 1
                    ):
                        raise ValueError(
                            f"owner {sample.rank} {tier} local store capability is missing"
                        )
                if active != enabled:
                    raise ValueError(
                        f"rank {sample.rank} {tier} source reservation differs from requested provider"
                    )
        self.remote_epoch = {s.rank: s.remote_runtime for s in samples}
        self.local_targets = {s.rank: s.local_mock_targets for s in samples}
        self.owner_ranks = {target.owner_rank for target in targets}
        self.instances = {s.rank: s.instance for s in samples}
        return (
            samples,
            targets,
            raw,
        )

    def request(self, payload, register, before):
        request_id = "pending"
        responses = None
        error_result = None
        exhausted = False
        started = time.perf_counter()
        try:
            request = self.build_request(
                request_id=request_id,
                model_name="rtp_llm",
                input_ids=payload["input_ids"],
                sampling=self.sampling,
                enable_thinking=False,
                force_sp_accept=True,
            )
            request_id = before(payload)
            request.id = request_id
            request.parameters["mock_cache_request"].bool_param = payload.get(
                "mock_cache_request", payload.get("mock_memory_cache_request", False)
            )
            responses = self.dash.ModelStreamInfer(
                iter((request,)),
                timeout=self.timeout,
                metadata=(("x-dashscope-inner-loggingconsent", "none"),),
            )
            register(responses)
            values = {}
            output_len = 0
            received_inference = False
            for response in responses:
                if response.error_message:
                    raise RuntimeError(response.error_message)
                if not response.HasField("infer_response"):
                    continue
                received_inference = True
                infer = response.infer_response
                parameters = infer.parameters
                error_no = (
                    parameters["error_no"].int64_param
                    if "error_no" in parameters
                    else 0
                )
                error_msg = (
                    parameters["error_msg"].string_param
                    if "error_msg" in parameters
                    else ""
                )
                status_code = (
                    parameters["status_code"].int64_param
                    if "status_code" in parameters
                    else 0
                )
                if (
                    error_no
                    or error_msg
                    or (status_code and not 200 <= status_code < 300)
                ):
                    raise RuntimeError(
                        f"DashSC business error: error_no={error_no}, "
                        f"status_code={status_code}, error_msg={error_msg}"
                    )
                for name in (
                    "engine_cost_time_us",
                    "engine_first_token_cost_time_us",
                    "engine_wait_time_us",
                ):
                    if name in infer.parameters:
                        values[name] = infer.parameters[name].int64_param / 1000.0
                for index, output in enumerate(infer.outputs):
                    if index >= len(infer.raw_output_contents):
                        continue
                    raw = infer.raw_output_contents[index]
                    if output.name == "generated_ids" and output.datatype == "INT32":
                        shape = list(output.shape)
                        output_len = max(
                            output_len, int(shape[-1]) if shape else len(raw) // 4
                        )
                    elif output.datatype == "INT32" and len(raw) >= 4:
                        values[output.name] = struct.unpack_from("<i", raw)[0]
                    elif output.datatype == "FP64" and len(raw) >= 8:
                        values[output.name] = struct.unpack_from("<d", raw)[0]
            exhausted = True
            if not received_inference:
                raise RuntimeError("Dash-SC stream ended without an inference response")
            wall_ms = (time.perf_counter() - started) * 1000.0
            return {
                "success": True,
                "request_id": request_id,
                "input_len": int(values.get("prompt_token_num", 0)),
                "output_len": output_len,
                "reuse_len": int(values.get("prompt_cached_token_num", 0)),
                "prefill_time_ms": float(
                    values.get("engine_first_token_cost_time_us", 0.0)
                ),
                "total_time_ms": float(values.get("engine_cost_time_us", 0.0)),
                "wait_time_ms": float(values.get("engine_wait_time_us", 0.0)),
                "client_wall_time_ms": wall_ms,
                "ttft_ms": wall_ms,
                "ttft_source": "client_dashsc_grpc_input_ids_wall_max_new_tokens_1",
            }
        except DispatchClosed:
            raise
        except Exception as exc:
            error_result = {
                "success": False,
                "error": repr(exc),
                "request_id": request_id,
                "client_wall_time_ms": (time.perf_counter() - started) * 1000.0,
            }
            return error_result
        finally:
            if responses is not None and not exhausted:
                try:
                    stopped = responses.cancel() or responses.done()
                except Exception as cleanup_error:
                    stopped = False
                    if error_result is not None:
                        error_result["cancel_error"] = repr(cleanup_error)
                if not stopped and error_result is not None:
                    error_result["ingress_termination_uncertain"] = True

    def diagnostics_control(self, action, phase, timeout):
        actions = dict(enable=1, phase=2, disable=3)
        sent = time.monotonic()
        record = dict(action=action, phase=phase, client_send_s=sent)
        try:
            response = self.rpc.ControlPressureDiagnostics(
                self.messages.PressureSessionControlRequestPB(
                    session_id=self.session_id,
                    action=actions[action],
                    phase=phase,
                    recorder_capacity=4096 if action == "enable" else 0,
                    target_ranks=self.ranks,
                ),
                timeout=timeout,
            )
            record["response"] = _project(response)
            errors = []
            seen = set()
            if response.schema_version != 1:
                errors.append("unsupported control schema")
            for ack in response.ranks:
                rank = ack.world_rank
                if rank in seen or rank not in self.ranks:
                    errors.append("duplicate or unexpected control rank")
                seen.add(rank)
                if (
                    ack.error
                    or not ack.service_instance
                    or ack.applied_monotonic_ns <= 0
                ):
                    errors.append("rank acknowledgement failed: " + str(rank))
                elif self.instances.get(rank) != ack.service_instance:
                    errors.append("rank instance changed: " + str(rank))
            if seen != set(self.ranks):
                errors.append("missing control ranks")
            record.update(success=not errors, errors=errors)
        except Exception as exc:
            record.update(success=False, errors=[repr(exc)])
        record["client_receive_s"] = time.monotonic()
        return record

    def poll(self, timeout, drain_remote_completions=False):
        if drain_remote_completions:
            self.barrier_attempted = True

        def service_boundary():
            if not self.remote_binding:
                return None, None
            from ..analysis.remote_artifacts import record_remote_end

            try:
                return record_remote_end(self.remote_binding), None
            except (OSError, ValueError, KeyError) as exc:
                return None, str(exc)

        before_offset, before_error = (
            (None, None) if drain_remote_completions else service_boundary()
        )
        response, sent, received = self._snapshot(
            timeout,
            session_id=self.session_id,
            after_step_sequence=self.cursors,
            max_step_records=256,
            drain_remote_completions=drain_remote_completions,
        )
        raw = dict(
            client_send_s=sent, client_receive_s=received, response=_project(response)
        )
        try:
            raw = raw_pressure_record(response, sent, received, to_dict=_project)
            if self.remote_binding:
                after_offset, after_error = service_boundary()
                raw.update(
                    service_bytes_before=before_offset, service_bytes_after=after_offset
                )
                if before_error or after_error:
                    raw["service_capture_error"] = before_error or after_error
                    if drain_remote_completions:
                        self.remote_binding["boundary_error"] = raw[
                            "service_capture_error"
                        ]
            observed_ranks = [rank.world_rank for rank in response.ranks]
            if response.schema_version != 1:
                raise ValueError("unsupported pressure schema")
            if (
                len(response.requested_ranks) != len(self.ranks)
                or set(response.requested_ranks) != set(self.ranks)
                or len(observed_ranks) != len(set(observed_ranks))
                or not set(observed_ranks).issubset(self.ranks)
            ):
                raise ValueError("pressure rank protocol mismatch")
            payload_ranks = [
                rank
                for rank in response.ranks
                if not rank.error
                or any(
                    field.name
                    not in {
                        "world_rank",
                        "service_instance",
                        "error",
                        "coordinator_send_ns",
                        "coordinator_receive_ns",
                    }
                    for field, _ in rank.ListFields()
                )
            ]
            payload_rank_ids = {rank.world_rank for rank in payload_ranks}
            if any(
                rank.world_rank not in self.instances
                or (
                    rank.service_instance
                    and self.instances[rank.world_rank] != rank.service_instance
                )
                or (
                    rank.world_rank in payload_rank_ids
                    and (not rank.service_instance or not rank.step_history_available)
                )
                or any(step.session_id != self.session_id for step in rank.steps)
                for rank in response.ranks
            ):
                raise ValueError("rank instance or diagnostic history mismatch")
            if len(observed_ranks) != len(self.ranks) or any(
                rank.error for rank in response.ranks
            ):
                concrete_samples = ()
                if payload_ranks:
                    # Preserve concrete evidence for frozen inventory/reset
                    # checks, without asserting a complete shared scope.
                    available = self.messages.CachePressureResponsePB()
                    available.CopyFrom(response)
                    available.ClearField("ranks")
                    available.ClearField("requested_ranks")
                    for rank in payload_ranks:
                        copied = available.ranks.add()
                        copied.CopyFrom(rank)
                        copied.ClearField("error")
                        available.requested_ranks.append(rank.world_rank)
                    concrete_samples = decode_pressure_response(
                        available,
                        sent,
                        received,
                        tuple(available.requested_ranks),
                        complete=False,
                        to_dict=_project,
                    ).samples
                raw["sampling_error"] = "missing rank or failed rank sample"
                # A partial page provides no complete coverage or cursor boundary.
                return PressurePoll(
                    concrete_samples,
                    raw,
                    step_history_complete=False,
                    sample_complete=False,
                )
            decoded = decode_pressure_response(
                response, sent, received, self.ranks, to_dict=_project
            )
            if any(
                self.local_targets.get(s.rank) != s.local_mock_targets
                for s in decoded.samples
            ):
                raise ValueError("owner local mock targets changed")
            if any(
                self.remote_epoch.get(s.rank) != s.remote_runtime
                for s in decoded.samples
            ):
                raise ValueError("Remote startup identity or configuration changed")
            if any(
                r.next_step_sequence < self.cursors.get(r.world_rank, 0)
                for r in response.ranks
            ):
                raise ValueError("rank step cursor regressed")
            if drain_remote_completions:
                _validate_drained(decoded.samples, self.owner_ranks)
                if self.remote_binding and "service_capture_error" not in raw:
                    self.remote_binding["end_bytes"] = raw["service_bytes_after"]
        except Exception as exc:
            raw["diagnostics_error"] = repr(exc)
            return PressurePoll(
                (), raw, step_history_complete=False, sample_complete=False
            )
        samples = {s.rank: s for s in decoded.samples}
        activity = []
        for rank in response.ranks:
            sample = samples[rank.world_rank]
            low = sample.offset_bounds[0]
            active = [
                step.wall_start_ns / 1e9 + low
                for step in rank.steps
                if not step.completed
            ]
            progress = [
                step.wall_end_ns / 1e9 + low for step in rank.steps if step.completed
            ]
            a = rank.activity
            activity.append(
                ActivitySample(
                    0,
                    max(progress, default=0),
                    max(active, default=None),
                    (
                        a.running_requests.value
                        if a.HasField("running_requests")
                        else None
                    ),
                    (
                        a.waiting_requests.value
                        if a.HasField("waiting_requests")
                        else None
                    ),
                )
            )
            self.cursors[rank.world_rank] = rank.next_step_sequence
        return PressurePoll(
            decoded.samples,
            raw,
            tuple(activity),
            any(
                sum(step.completed for step in rank.steps) >= 256
                for rank in response.ranks
            ),
            step_history_complete=not any(
                rank.step_records_lost for rank in response.ranks
            ),
        )

    def capture(self, config, directory, *, nsys_command=None):
        from .capture import KinetoCapture, NsightCapture

        name = "pressure_" + self.session_id
        common = dict(
            trace_name=name,
            directory=directory,
            ranks=self.ranks,
            instances=self.instances,
            session_id=self.session_id,
            timeout_s=config.capture_timeout_s,
        )
        if nsys_command is not None:
            return NsightCapture(
                command=nsys_command,
                duration_s=config.nsys_duration_s,
                command_timeout_s=self.timeout,
                **common,
            )

        def start():
            self.rpc.StartProfile(
                self.messages.StartProfileRequestPB(
                    trace_name=name,
                    num_steps=config.profile_steps,
                    enable_all_rank=True,
                ),
                timeout=5,
            )
            return {"accepted": True}

        def stop():
            response = self.rpc.StopProfile(
                self.messages.StopProfileRequestPB(
                    trace_name=name, enable_all_rank=True, target_ranks=self.ranks
                ),
                timeout=1,
            )
            if sorted(r.world_rank for r in response.ranks) != sorted(
                self.ranks
            ) or any(
                not r.accepted
                or r.error
                or r.service_instance != self.instances[r.world_rank]
                or r.profile.trace_name != name
                for r in response.ranks
            ):
                raise RuntimeError("incomplete native profiler stop acknowledgement")
            return _project(response)

        def status(timeout):
            response, sent, received = self._snapshot(timeout)
            decode_pressure_response(
                response, sent, received, self.ranks, to_dict=_project
            )
            return raw_pressure_record(response, sent, received, to_dict=_project)

        return KinetoCapture(
            start_capture=start,
            stop_capture=stop,
            read_status=status,
            requested_steps=config.profile_steps,
            **common,
        )

    def run(
        self,
        *,
        settings,
        cases,
        make_ids,
        encode_marker,
        decode_marker,
        directory,
        model_tokens,
        admission_requests,
        batch_tokens,
        cache_alignment,
        commit_tail_tokens,
        concurrency=16,
        seed_count=None,
        test_block_num=0,
        random_seed=0,
        rate=0,
        poisson=False,
        capture=False,
        trace_directory=None,
        nsys_command=None,
        manifest=None,
        remote_launch_manifest=None,
    ):
        """Run one fresh epoch. make_ids(length, marker) returns exact model token IDs."""
        config = pressure_settings(settings)
        cases = tuple(dict(c) for c in cases)
        validate_cases(cases, config)
        self.session_id = uuid.uuid4().hex
        self.cursors.clear()
        try:
            samples, targets, raw = self.startup(config)
        except BaseException:
            if config.evidence_policy == "tree_then_mock_remote":
                # Even a lost observer cannot suppress the owner's physical drain.
                self._snapshot(config.drain_timeout_s, drain_remote_completions=True)
            raise
        self.barrier_attempted = False
        self.remote_binding = None
        plan = build_seed_plan(
            samples,
            targets,
            cases=cases,
            cache_alignment=cache_alignment,
            commit_tail_tokens=commit_tail_tokens,
            client_inflight=concurrency,
            manual_seed_count=seed_count,
            fill_target_tokens=config.fill_input_tokens,
            test_block_num=test_block_num,
            diagnostic_only=config.evidence_policy != "tree_pressure",
        )
        config = replace(config, fill_input_tokens=plan.fill_input_tokens)
        budgets = tuple(
            (
                f"owner={g.owner_rank} group={g.owner_group_id} pool={p.pool_id}",
                p.available_blocks,
                p.block_tokens,
            )
            for g in plan.groups
            for p in g.device_pools
        )
        limits = ExecutionLimits(
            model_tokens,
            admission_requests,
            batch_tokens,
            min(b[1] for b in budgets),
            budgets[0][2],
            budgets,
        )
        for case in cases:
            limits.require_request(case["input_len"])
        payloads = _Payloads(
            config,
            plan,
            cases,
            make_ids,
            encode_marker,
            decode_marker,
            self.session_id,
            tuple(p[2] for p in budgets),
        )

        def request(payload, register, before):
            result = self.request(payload, register, before)
            if result.get("success"):
                if (
                    result["input_len"] != len(payload["input_ids"])
                    or result["output_len"] != 1
                ):
                    return dict(result, success=False, error="request shape mismatch")
                payloads.confirm(payload)
            return result

        root = Path(directory)
        # Exclusive evidence directory prevents mixing multiple runs in append-only files.
        (root / "pressure").mkdir(parents=True, exist_ok=False)
        controller = (
            self.capture(config, trace_directory or root, nsys_command=nsys_command)
            if capture
            else None
        )
        writer = None
        try:
            metadata = dict(manifest or {})
            if self.remote_epoch and config.evidence_policy == "tree_then_mock_remote":
                runtime = json.loads(next(iter(self.remote_epoch.values())))
                metadata["remote_service_capture"] = {
                    "status": "unverified",
                    "run_id": self.session_id,
                    "runtime": {
                        k: int(runtime[k])
                        for k in ("server_epoch", "session_id", "generation")
                    },
                }
                if remote_launch_manifest is not None:
                    from ..analysis.remote_artifacts import bind_remote_capture

                    try:
                        self.remote_binding = bind_remote_capture(
                            remote_launch_manifest,
                            metadata["remote_service_capture"]["runtime"],
                            runtime["options"]["mock_remote_cache_address"],
                        )
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        metadata["remote_service_capture"]["binding_error"] = str(exc)
                    else:
                        metadata["remote_service_capture"][
                            "binding"
                        ] = self.remote_binding
            writer = PressureWriter(
                root,
                metadata
                | dict(
                    session_id=self.session_id,
                    capture_requested=capture,
                    epoch=self.instances,
                    clock_bounds=raw.get("rank_offset_bounds_s"),
                    startup_pressure=raw,
                    observed_local_mock_targets={
                        str(s.rank): asdict(s.local_mock_targets)
                        for s in samples
                        if s.local_mock_targets is not None
                    },
                    startup_targets=[asdict(t) for t in targets],
                    seed_plan=asdict(plan),
                    resolved_settings=asdict(config),
                    input_identity=cases,
                ),
            )
            with JsonlSink(root / "pressure") as sink:

                def record(event):
                    if self.remote_binding and event["event"] in {
                        "frozen",
                        "fill_baseline",
                    }:
                        baseline = event["raw_baseline"]
                        before = baseline.get("service_bytes_before")
                        after = baseline.get("service_bytes_after")
                        if (
                            before is None
                            or before != after
                            or baseline.get("service_capture_error")
                        ):
                            self.remote_binding["boundary_error"] = (
                                baseline.get("service_capture_error")
                                or "missing or changing Remote baseline"
                            )
                        else:
                            self.remote_binding["begin_bytes"] = after
                    sink(event)

                evidence = PressureEvidence(
                    writer,
                    record,
                    profile_requested=capture,
                    target_ranks=self.ranks,
                    session_id=self.session_id,
                )
                session = PressureSession(
                    config,
                    session_id=self.session_id,
                    limits=limits,
                    concurrency=concurrency,
                    max_input_tokens=max(c["input_len"] for c in cases),
                    seed_plan=plan,
                    startup_targets=targets,
                    make_payload=payloads.make,
                    request=request,
                    poll=self.poll,
                    sink=evidence,
                    controller=controller,
                    diagnostics_control=self.diagnostics_control,
                    random_seed=random_seed,
                    rate=rate,
                    poisson=poisson,
                )
                try:
                    result = session.run()
                    if "remote_service_capture" in metadata:
                        result["remote_service_capture"] = metadata[
                            "remote_service_capture"
                        ]
                        result["complete"] = False
                    return result
                except BaseException as error:
                    evidence.abort(repr(error))
                    raise
        finally:
            try:
                if (
                    config.evidence_policy == "tree_then_mock_remote"
                    and not self.barrier_attempted
                ):
                    # Best-effort cleanup after failed enable/baseline or ingress timeout.
                    # A cleanup-only barrier cannot certify the missed measurement window.
                    self._snapshot(
                        config.drain_timeout_s, drain_remote_completions=True
                    )
            finally:
                if writer is not None:
                    writer.close()


def validate_cases(cases, config):
    if not cases:
        raise ValueError("nonempty pressure cases required")
    if config.cohort_size // 2 % len(cases):
        raise ValueError("pressure grid cases must divide each tier half of the cohort")
    for case in cases:
        if (
            case.get("request_groups")
            or case.get("batch_size", 1) != 1
            or case.get("output_length", 1) != 1
        ):
            raise ValueError(
                "pressure requires ungrouped batch_size=1 and output_length=1"
            )
        n, k = case.get("input_len"), case.get("cache_len")
        if type(k) is not int or type(n) is not int or not 0 <= k < n:
            raise ValueError("pressure cases require 0 <= cache_len < input_len")
        if not k:
            raise ValueError("tiered or mixed policy requires positive cache_len")


class _Payloads:
    def __init__(
        self,
        config,
        plan,
        cases,
        make_ids,
        encode_marker,
        decode_marker,
        session_id,
        block_tokens,
    ):
        self.config, self.plan, self.cases = config, plan, cases
        self.make_ids, self.session_id = make_ids, session_id
        self.cohort = config.cohort_size
        self.encode_marker, self.decode_marker = encode_marker, decode_marker
        # Two stable vocabulary IDs delimit variable-length encodings inside the block.
        signs = [list(encode_marker(c)) for c in "01"]
        if not all(signs):
            raise ValueError("marker tokenizer must encode both 0 and 1")
        self.stop_token, self.pad_token = (ids[-1] for ids in signs)
        if self.stop_token == self.pad_token:
            raise ValueError("marker tokenizer must distinguish 0 and 1")
        self.lock = threading.RLock()
        self.seeds = [None] * plan.seed_count
        self.prefixes = set()
        self.block_tokens = block_tokens
        self.effective = None
        self.planner = None
        self.rewarm_index = 0

    def _exact(self, length, nonce):
        ids = list(self.make_ids(length, nonce))
        if len(ids) != length:
            raise ValueError("token factory violated exact length")
        return ids

    def _marker(self, nonce, width, first=None):
        tokens = list(self.encode_marker(nonce))
        if not tokens or self.decode_marker(tokens) != nonce:
            raise ValueError("tokenizer cannot round-trip pressure marker")
        prefix = [] if first is None else [first]
        # TT encodes a literal T; TP terminates. No marker can be a prefix of
        # another, even if a later epoch uses a different width and filler.
        block = prefix
        for token in tokens:
            block.append(token)
            if token == self.stop_token:
                block.append(token)
        block.extend((self.stop_token, self.pad_token))
        if len(block) > width:
            raise ValueError(
                "unique marker does not fit the first affected cache block"
            )
        return block + [self.pad_token] * (width - len(block))

    def confirm(self, payload):
        if "seed_family" in payload:
            with self.lock:
                identity = tuple(payload["input_ids"][: min(self.block_tokens)])
                if identity in self.prefixes:
                    raise ValueError("seed families share the same first block prefix")
                self.prefixes.add(identity)
                self.seeds[payload["seed_family"]] = payload["input_ids"]

    def make(self, kind, job_id, formal_sequence):
        nonce = f"{job_id}_{self.session_id}_{kind}_"
        if kind in ("seed", "fill"):
            size = (
                self.plan.seed_input_tokens
                if kind == "seed"
                else self.plan.fill_input_tokens
            )
            width = min(
                size,
                min(self.block_tokens),
                min(c["cache_len"] for c in self.cases),
            )
            ids = self._exact(size, nonce)
            ids[:width] = self._marker(nonce, width)
            data = dict(input_ids=ids, nonce=nonce)
            if kind == "seed":
                data.update(seed_family=job_id, family=job_id)
            return data
        with self.lock:
            if self.planner is None:
                self.effective = freeze_seed_families(self.seeds)
                self.planner = TieredRollingPlanner(
                    effective_seed_count=self.effective,
                    cohort_size=self.cohort,
                    memory_reuse_ratio=self.config.memory_reuse_ratio,
                    case_indices=tuple(range(len(self.cases))),
                )
            if kind == "rewarm":
                assignment = self.planner.initial_device_assignments()[
                    self.rewarm_index
                ]
                self.rewarm_index += 1
            else:
                assignment = self.planner.assignment(formal_sequence)
            case_index, family = assignment.case_index, assignment.family
            epoch = assignment.cohort_id
            case = self.cases[case_index]
            length, reuse = case["input_len"], case["cache_len"]
            seed = self.seeds[family]
        if len(seed) <= reuse:
            raise ValueError("reusable seed requires a commit tail")
        size = length - reuse
        width = min(size, min(block - reuse % block for block in self.block_tokens))
        first = self.stop_token if seed[reuse] != self.stop_token else self.pad_token
        suffix = self._marker(nonce, width, first)
        if len(suffix) < size:
            suffix.extend(self._exact(size - len(suffix), " "))
        ids = seed[:reuse] + suffix
        data = dict(
            input_ids=ids,
            nonce=nonce,
            family=family,
            epoch=epoch,
            case_index=case_index,
            requested_cache_len=reuse,
        )
        data.update(
            cohort_id=assignment.cohort_id,
            cohort_lane=assignment.cohort_lane,
            expected_cache_tier=assignment.expected_cache_tier,
            promoted_from_cohort=assignment.promoted_from_cohort,
            effective_seed_count=self.effective,
        )
        if kind == "rewarm":
            data["rewarm"] = True
        return data


def _validate_drained(samples, owners):
    if not owners or not owners <= {s.rank for s in samples}:
        raise ValueError("Remote barrier missing owner")
    for sample in samples:
        if sample.rank not in owners:
            continue
        w = sample.mock_remote_write
        if w is None or any(
            getattr(w, name)
            for name in (
                "pending_requests",
                "pending_intents",
                "retained_descriptors",
                "faulted",
                "quarantined_blocks",
            )
        ):
            raise ValueError(
                "Remote barrier has missing, failed or pending write evidence"
            )


def _remote_layouts(sample, runtime, limits, options, tp):
    groups = {g.group_id: g for g in sample.groups}
    topology, cp_sizes, key_strides = {}, set(), set()
    largest = source_bytes = 0
    for layout in runtime["layouts"]:
        group_id = int(layout.get("group", 0))
        group = groups.get(group_id)
        if (
            group is None
            or group_id in topology
            or int(layout.get("pool", 0)) != group_id
        ):
            raise ValueError("Remote layout group/pool identity mismatch")
        block, payload, storage, crc = group.geometry
        tokens, cp = int(layout["tokens_per_block"]), int(layout["cp_world_size"])
        stride = int(layout["stride_bytes"])
        sources = int(layout["source_count"])
        if (
            int(layout["payload_bytes"]) != payload
            or int(layout["storage_bytes"]) != storage
            or layout.get("crc_enabled", False) != crc
            or storage != ((payload + 19) // 16 * 16 if crc else payload)
            or int(layout.get("crc_footer_offset", 0)) != (storage - 4 if crc else 0)
            or int(layout.get("crc_format_version", 0)) != 1
            or int(layout["alignment"]) != 16
            or stride != (storage + 15) // 16 * 16
            or not re.fullmatch(r"(?:mixed|dtype:[0-9]+)", layout.get("dtype", ""))
            or sources != int(options["mock_remote_cache_source_blocks"])
            or sources > limits["max_sources_per_group"]
        ):
            raise ValueError("Remote storage or CRC geometry mismatch")
        if (
            cp not in (1, tp)
            or int(layout.get("cp_rank", 0)) != (sample.rank if cp > 1 else 0)
            or (cp == 1 and tokens != block)
            or (
                cp > 1
                and group.group_type == "FULL"
                and tokens not in (block, block * cp)
            )
        ):
            raise ValueError("Remote CP rank or cache-key stride mismatch")
        cp_sizes.add(cp)
        key_strides.add(tokens)
        topology[group_id] = (
            group.group_type,
            cp,
            tokens,
            int(layout.get("attention_window", 0)),
        )
        largest = max(largest, stride)
        source_bytes += sources * (stride + 4)  # service stores a checksum per source
    if (
        set(topology) != set(range(len(groups)))
        or len(groups) > limits["max_groups_per_rank"]
        or len(cp_sizes) != 1
        or (cp_sizes != {1} and len(key_strides) != 1)
        or source_bytes > limits["source_bytes"]
    ):
        raise ValueError("Remote groups, canonical CP stride or source bank mismatch")
    return topology, largest, source_bytes


def _remote_credits(rank, tp, credit, limits, options, record):
    local = int(options["mock_remote_cache_staging_size_mb"]) << 20
    slots = limits["write_slots"]
    first = slots * rank // tp
    expected = dict(
        local_read_bytes=local // 2,
        local_write_bytes=local - local // 2,
        service_read_bytes=limits["read_staging_bytes"] // tp,
        service_write_bytes=limits["write_staging_bytes"] // tp,
        first_write_slot=first,
        write_slots=slots * (rank + 1) // tp - first,
    )
    if any(credit[k] != v for k, v in expected.items()) or any(
        v <= 0 for k, v in credit.items() if k != "first_write_slot"
    ):
        raise ValueError(
            "Remote rank staging or slot share differs from negotiated limits"
        )
    reads, writes = credit["read_operations"], credit["write_operations"]
    inflight = int(options["mock_remote_cache_max_inflight"])
    read_share = credit["local_read_bytes"] // reads
    allowance = read_share - read_share // 8
    rpc = limits["max_rpc_bytes"]
    if (
        reads
        > min(
            inflight,
            limits["max_read_operations"] // tp,
            int(options["service_reads_per_connection"]),
        )
        or writes
        > min(
            inflight,
            credit["write_slots"],
            int(options["service_writes_per_connection"]),
        )
        or credit["max_batch_records"] > min(8, limits["max_blocks_per_rpc"])
        or credit["max_slice_bytes"] > rpc
        or credit["max_write_slice_bytes"] > min(rpc, limits["write_slot_bytes"])
        or 2 * credit["max_slice_bytes"] >= credit["service_read_bytes"] // reads
        or 2 * credit["max_slice_bytes"] + record * credit["max_batch_records"]
        >= allowance
        or 2 * credit["max_write_slice_bytes"]
        >= credit["service_write_bytes"] // writes
        or 2 * credit["max_write_slice_bytes"] + record + 16
        >= credit["local_write_bytes"] // writes
        or credit["service_write_bytes"] < 29186
    ):
        raise ValueError("Remote operation or slice credit exceeds negotiated limits")


def _validate_remote(samples, config, ranks):
    identities, callbacks = set(), set()
    shared_topology = None
    source_total = 0
    tp = len(ranks)
    if set(ranks) != set(range(tp)):
        raise ValueError("Remote local TP rank set must be contiguous")
    shared_options = shared_limits = None
    for sample in samples:
        runtime = json.loads(sample.remote_runtime)
        options, credits, limits = (
            runtime["options"],
            runtime["credits"],
            runtime["limits"],
        )
        if shared_options is not None and (
            shared_options != options or shared_limits != limits
        ):
            raise ValueError("Remote rank service options or limits disagree")
        shared_options, shared_limits = options, limits
        if int(runtime.get("rank", 0)) != sample.rank or int(
            runtime["expected_ranks"]
        ) != len(ranks):
            raise ValueError("Remote rank capability mismatch")
        endpoint = runtime["callback_endpoint"]
        host, sep, port = endpoint.rpartition(":")
        if (
            not sep
            or not host
            or int(port)
            != int(options["mock_remote_cache_callback_port"]) + sample.rank
            or endpoint in callbacks
        ):
            raise ValueError("Remote reverse callback endpoint mismatch")
        callbacks.add(endpoint)
        identities.add(
            tuple(int(runtime[k]) for k in ("server_epoch", "session_id", "generation"))
        )
        for name in ("memory", "disk", "remote"):
            if float(options[f"mock_{name}_cache_hit_ratio"]) != getattr(
                config, f"mock_{name}_target_ratio"
            ):
                raise ValueError(
                    "Remote effective tier ratio differs from requested " + name
                )
        limits = {k: int(v) for k, v in limits.items()}
        credits = {k: int(v) for k, v in credits.items()}
        timeout = int(options["mock_remote_cache_timeout_ms"])
        delay = max(
            int(options["mock_remote_cache_read_delay_ms"]),
            int(options["mock_remote_cache_write_delay_ms"]),
        )
        if (
            tp
            > min(limits["max_ranks"], int(options["service_max_callback_endpoints"]))
            or timeout > int(options["service_max_timeout_ms"])
            or delay > limits["max_delay_ms"]
            or delay + 10 >= timeout
        ):
            raise ValueError(
                "Remote rank, timeout or delay exceeds service capabilities"
            )
        topology, record, source_bytes = _remote_layouts(
            sample, runtime, limits, options, tp
        )
        if shared_topology is not None and topology != shared_topology:
            raise ValueError("Remote TP ranks advertise inconsistent group topology")
        shared_topology = topology
        source_total += source_bytes
        reserved = (
            source_total
            + limits["write_slots"] * limits["write_slot_bytes"]
            + limits["read_staging_bytes"]
            + limits["write_staging_bytes"]
        )
        if source_total > limits["source_bytes"] or reserved > limits["total_bytes"]:
            raise ValueError("Remote banks and staging exceed service capacity")
        _remote_credits(sample.rank, tp, credits, limits, options, record)
    if len(identities) != 1:
        raise ValueError("Remote ranks do not share the negotiated session")


def provider_settings(args, environment, settings):
    """Read deployment options from the frozen engine arguments and environment."""
    policy = settings["evidence_policy"]
    options = {}

    def option(name, default):
        return args.get(name, environment.get(name.upper(), default))

    for key, default in REMOTE_DEFAULTS.items():
        if key.startswith("mock_remote") and policy != "tree_then_mock_remote":
            if key == "mock_remote_cache" and str(option(key, False)).lower() in {
                "1",
                "true",
            }:
                raise ValueError(
                    "enabled Remote provider requires Remote pressure policy"
                )
            continue
        value = option(key, False if type(default) is bool else default)
        if type(default) is bool:
            normalized = str(value).lower()
            if normalized not in {"0", "1", "true", "false"}:
                raise ValueError(f"invalid engine boolean {key}")
            value = normalized in {"1", "true"}
        else:
            value = type(default)(value)
        options[key] = value
    for tier in ("memory", "disk", "remote"):
        ratio = settings[f"mock_{tier}_target_ratio"]
        _validate_provider(
            policy,
            tier,
            options.get(f"mock_{tier}_cache", False),
            ratio,
            options.get(f"mock_{tier}_cache_hit_ratio", 0),
        )
    return tuple(sorted(options.items()))


def run_rtp_pressure(
    *,
    dash_endpoint,
    pressure_endpoint,
    ranks,
    settings,
    cases,
    timeout=60,
    **run_options,
):
    """Run a fresh pressure epoch against two endpoints of an already running engine."""
    cases = tuple(cases)
    if not all(
        callable(run_options.get(k)) for k in ("encode_marker", "decode_marker")
    ):
        raise ValueError("a deterministic marker encoder and decoder are required")
    validate_cases(cases, pressure_settings(settings))
    if run_options.get("rate", 0) or run_options.get("poisson", False):
        raise ValueError("adaptive pressure controls its own arrivals")
    with RtpLlmPressureBackend.connect(
        dash_endpoint=dash_endpoint,
        pressure_endpoint=pressure_endpoint,
        ranks=ranks,
        timeout=timeout,
    ) as backend:
        return backend.run(settings=settings, cases=cases, **run_options)
