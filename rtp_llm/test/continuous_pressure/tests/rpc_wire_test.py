"""Checks the real generated messages without starting a model server."""

import struct
import unittest
from unittest.mock import Mock

from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.dash_sc.client import build_model_infer_request
from rtp_llm.dash_sc.codec import SamplingParams
from rtp_llm.dash_sc.proto import predict_v2_pb2 as dash
from rtp_llm.test.continuous_pressure.adapters.rtp_llm_rpc import RtpLlmPressureBackend


class WireTest(unittest.TestCase):
    def backend(self, responses=()):
        self.prepared, self.sent = [], []

        def build(**kwargs):
            message = build_model_infer_request(**kwargs)
            self.prepared.append(message)
            return message

        def send(messages, **kwargs):
            self.sent.append(
                dash.ModelInferRequest.FromString(next(messages).SerializeToString())
            )
            return iter(responses)

        return RtpLlmPressureBackend(
            dash_stub=Mock(ModelStreamInfer=send),
            pressure_stub=Mock(),
            messages=pb,
            ranks=(0, 1),
            timeout=1,
            build_request=build,
            sampling=SamplingParams(max_new_tokens=1, min_new_tokens=1, top_k=1),
        )

    def test_wire_false_and_true_resolved_after_real_message_preparation(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                response = dash.ModelStreamInferResponse()
                response.infer_response.parameters["status_code"].int64_param = 503
                backend = self.backend([response])
                payload = {"input_ids": [7, 123456, 9]}

                def before(p):
                    self.assertEqual(self.prepared[0].id, "pending")
                    self.assertFalse(self.sent)
                    p["mock_cache_request"] = enabled
                    return "accepted"

                registered = []
                result = backend.request(payload, registered.append, before)
                self.assertIn("503", result["error"])
                self.assertEqual(len(registered), 1)
                self.assertEqual(self.sent[0].id, "accepted")
                self.assertIn("mock_cache_request", self.sent[0].parameters)
                parameter = self.sent[0].parameters["mock_cache_request"]
                self.assertTrue(parameter.HasField("bool_param"))
                self.assertIs(parameter.bool_param, enabled)
                raw = dict(
                    zip(
                        (t.name for t in self.sent[0].inputs),
                        self.sent[0].raw_input_contents,
                    )
                )
                self.assertEqual(struct.unpack("<3i", raw["input_ids"]), (7, 123456, 9))

    def test_premature_stream_exit_cancels_even_if_cleanup_raises(self):
        for failure in ("register", "business", "decode", "exhausted"):
            with self.subTest(failure=failure):
                backend = self.backend()
                peer = Mock()
                reply = dash.ModelStreamInferResponse()
                reply.infer_response.SetInParent()
                if failure == "business":
                    reply.error_message = "business failure"

                def responses():
                    yield reply
                    if failure == "decode":
                        raise ValueError("decode failure")

                peer.__iter__ = Mock(side_effect=responses)
                peer.cancel.side_effect = RuntimeError("cleanup error")
                backend.dash.ModelStreamInfer = Mock(return_value=peer)
                register = Mock(
                    side_effect=(
                        ValueError("register failure")
                        if failure == "register"
                        else None
                    )
                )
                result = backend.request(
                    {"input_ids": [1]}, register, lambda payload: "req"
                )
                if failure == "exhausted":
                    self.assertTrue(result["success"])
                    peer.cancel.assert_not_called()
                else:
                    self.assertIn(failure + " failure", result["error"])
                    peer.cancel.assert_called_once_with()
                    self.assertTrue(result["ingress_termination_uncertain"])
                    self.assertIn("cleanup error", result["cancel_error"])

    def test_last_business_error_cancels_before_owner_barrier(self):
        import time
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace

        from rtp_llm.test.continuous_pressure.config import ExecutionLimits
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage
        from rtp_llm.test.continuous_pressure.phases import PressurePhases
        from rtp_llm.test.continuous_pressure.session import PressureSession

        for cancel_ok in (True, False):
            with self.subTest(cancel_ok=cancel_ok), ThreadPoolExecutor(1) as executor:
                backend, cfg, response = self.remote_fixture()
                samples, targets, _ = backend.startup(cfg)
                stream = Mock()
                stream.__iter__ = Mock(
                    return_value=iter(
                        [dash.ModelStreamInferResponse(error_message="last error")]
                    )
                )
                stream.cancel.return_value = cancel_ok
                stream.done.return_value = False
                backend.dash.ModelStreamInfer = Mock(return_value=stream)
                observed = []

                def barrier(request, timeout):
                    if request.drain_remote_completions:
                        observed.append(stream.cancel.call_count)
                    response.ranks[0].sequence += 1
                    response.ranks[0].sampled_monotonic_ns += 1000
                    return response

                backend.rpc.GetCachePressure.side_effect = barrier
                session = PressureSession(
                    cfg,
                    session_id=backend.session_id,
                    limits=ExecutionLimits(200000, 8, 200000, 100000, 16),
                    concurrency=1,
                    max_input_tokens=16,
                    seed_plan=SimpleNamespace(seed_count=0, seed_input_tokens=16),
                    startup_targets=targets,
                    make_payload=lambda *args: {"input_ids": [7]},
                    request=backend.request,
                    poll=backend.poll,
                    sink=lambda event: None,
                )
                session.coverage = PressureCoverage(samples, cfg)
                session.executor = executor
                session.state = PressurePhases(cfg, time.monotonic())
                session.state.phase = "measurement"
                session.state.measurement_end = time.monotonic() + 1
                session.formal_origin = time.monotonic()
                session._submit("formal", 0, 0)
                self.assertIn("last error", session.sender.pending[0].result()["error"])
                session._collect()
                session.state.measurement_end = time.monotonic() - 0.1
                session.last_poll_s = time.monotonic()
                session._drain()
                self.assertEqual(observed, [1])
                self.assertEqual(session.coverage.remote_final is not None, cancel_ok)
                self.assertEqual(session.state.failure, "request_failed")

    def test_local_owner_target_proof_is_required(self):
        from rtp_llm.test.continuous_pressure.adapters.rtp_llm_rpc import (
            pressure_settings,
        )

        backend, _, response = self.remote_fixture()
        rank = response.ranks[0]
        rank.ClearField("remote_runtime")
        rank.ClearField("mock_remote_write")
        rank.ClearField("local_mock_targets")
        rank.mock_memory.terminal_outcome_version = 1
        rank.pools[1].mock_source_ref_blocks = 1
        rank.pools[1].mock_source_ref_bytes = 128
        cfg = pressure_settings(
            dict(
                evidence_policy="tree_then_mock_memory", mock_memory_target_ratio=0.125
            )
        )
        with self.assertRaisesRegex(ValueError, "owner.*target"):
            backend.startup(cfg)
        targets = rank.local_mock_targets
        targets.memory_enabled = True
        rank.local_store.SetInParent()
        for ratio, ppm in ((0.125, 125000), (0, 0), (0.000000499, 0), (0.0000005, 1)):
            targets.memory_ratio, targets.memory_ppm = ratio, ppm
            settings = dict(
                evidence_policy="tree_then_mock_memory", mock_memory_target_ratio=ratio
            )
            samples, _, _ = backend.startup(pressure_settings(settings))
            self.assertEqual(samples[0].local_mock_targets.memory_ppm, ppm)
            with self.assertRaisesRegex(ValueError, "owner.*target"):
                backend.startup(
                    pressure_settings(dict(settings, mock_memory_target_ratio=0.875))
                )
        disk = rank.pools.add()
        disk.CopyFrom(rank.pools[1])
        disk.pool_id, disk.pool_name, disk.tier = "d", "d", 3
        rank.groups[0].disk_pool_id.value = "d"
        rank.groups[0].disk_watermark.CopyFrom(rank.groups[0].host_watermark)
        targets.disk_enabled, targets.disk_ratio, targets.disk_ppm = True, 0.25, 250000
        mixed = pressure_settings(
            dict(
                evidence_policy="tree_then_mock_disk",
                mock_memory_target_ratio=0.0000005,
                mock_disk_target_ratio=0.25,
            )
        )
        samples, _, _ = backend.startup(mixed)
        self.assertTrue(samples[0].local_mock_targets.disk_enabled)
        targets.memory_ppm = 0
        with self.assertRaisesRegex(ValueError, "owner.*target"):
            backend.startup(mixed)
        rank.ClearField("local_mock_targets")
        for pool in rank.pools:
            pool.mock_source_ref_blocks = pool.mock_source_ref_bytes = 0
        backend.startup(pressure_settings({}))

    def test_control_requires_complete_matching_rank_and_instance(self):
        backend = self.backend()
        backend.instances = {0: "aa", 1: "bb"}
        response = pb.PressureSessionControlResponsePB(schema_version=1)
        response.ranks.add(world_rank=0, service_instance="aa", applied_monotonic_ns=18)
        backend.rpc.ControlPressureDiagnostics.return_value = response
        self.assertFalse(
            backend.diagnostics_control("phase", "measurement", 0.25)["success"]
        )
        ack = response.ranks.add(
            world_rank=1, service_instance="old", applied_monotonic_ns=22
        )
        self.assertFalse(
            backend.diagnostics_control("phase", "measurement", 0.25)["success"]
        )
        ack.service_instance = "bb"
        self.assertTrue(
            backend.diagnostics_control("phase", "measurement", 0.25)["success"]
        )
        args, kwargs = backend.rpc.ControlPressureDiagnostics.call_args
        self.assertEqual(
            (
                args[0].session_id,
                args[0].phase,
                list(args[0].target_ranks),
                kwargs["timeout"],
            ),
            (backend.session_id, "measurement", [0, 1], 0.25),
        )

    def test_failed_barrier_preserves_raw_error_and_cannot_certify_drain(self):
        backend = self.backend()
        response = pb.CachePressureResponsePB(schema_version=1, requested_ranks=[0, 1])
        response.ranks.add(world_rank=1, error="observation lost")
        backend.rpc.GetCachePressure.return_value = response
        backend.instances = {0: "a", 1: "b"}
        result = backend.poll(0.5, drain_remote_completions=True)
        self.assertFalse(result.sample_complete)
        self.assertTrue(
            backend.rpc.GetCachePressure.call_args.args[0].drain_remote_completions
        )
        self.assertEqual(
            result.raw["response"]["ranks"][0]["error"], "observation lost"
        )

    def remote_fixture(self):
        from rtp_llm.test.continuous_pressure.adapters.rtp_llm_rpc import (
            pressure_settings,
        )

        backend = self.backend()
        backend.ranks = (0,)
        response = pb.CachePressureResponsePB(schema_version=1, requested_ranks=[0])
        rank = response.ranks.add(
            world_rank=0,
            service_instance="i",
            sequence=1,
            sampled_monotonic_ns=1,
            step_history_available=True,
        )
        rank.ownership.mode = 1
        rank.ownership.owner_world_rank.value = 0
        rank.ownership.participant_world_ranks.append(0)
        rank.ownership.publication_scope = 1
        group = rank.groups.add(
            group_id=0,
            group_type=1,
            block_tokens=16,
            payload_bytes=128,
            storage_bytes=128,
            device_pool_ids=["g"],
        )
        group.host_pool_id.value = "h"
        for name, tier in (("g", 1), ("h", 2)):
            rank.pools.add(
                pool_id=name,
                pool_name=name,
                tier=tier,
                total_blocks=100,
                free_blocks=100,
                available_blocks=100,
                block_size_bytes=128,
            )
        for w in (group.device_watermark, group.host_watermark):
            w.low_ratio = 0.7
            w.high_ratio = 0.9
        backend.rpc.GetCachePressure.return_value = response
        cfg = pressure_settings({"evidence_policy": "tree_then_mock_remote"})
        with self.assertRaisesRegex(ValueError, "capability"):
            backend.startup(cfg)
        rt = rank.remote_runtime
        (
            rt.version,
            rt.server_epoch,
            rt.session_id,
            rt.generation,
            rt.layout_hash,
            rt.expected_ranks,
        ) = (1, 18, 7, 2, 45, 1)
        rt.callback_endpoint = "127.0.0.1:19080"
        rt.options.update({k: str(v).lower() for k, v in cfg.mock_cache_settings})
        rt.options.update(
            {
                k: "10"
                for k in (
                    "service_reads_per_connection",
                    "service_writes_per_connection",
                    "service_max_callback_endpoints",
                    "service_max_timeout_ms",
                )
            }
        )
        rt.limits.update(
            {
                k: 128
                for k in "total_bytes source_bytes write_slot_bytes read_staging_bytes write_staging_bytes write_slots max_rpc_bytes max_blocks_per_rpc max_read_operations max_ranks max_groups_per_rank max_sources_per_group max_members_per_group replay_window max_delay_ms".split()
            }
        )
        rt.credits.update(
            {
                k: 1
                for k in "local_read_bytes local_write_bytes service_read_bytes service_write_bytes read_operations write_operations write_slots max_slice_bytes max_batch_records max_write_slice_bytes".split()
            }
        )
        rt.credits["first_write_slot"] = 0
        rt.limits.update(
            dict(
                total_bytes=1 << 24,
                source_bytes=1 << 20,
                write_slots=8,
                read_staging_bytes=1 << 20,
                write_staging_bytes=1 << 20,
            )
        )
        rt.credits.update(
            dict(
                local_read_bytes=128 << 20,
                local_write_bytes=128 << 20,
                service_read_bytes=1 << 20,
                service_write_bytes=1 << 20,
                write_slots=8,
            )
        )
        rt.options["service_max_timeout_ms"] = "10000"
        rt.layouts.add(
            crc_format_version=1,
            payload_bytes=128,
            storage_bytes=128,
            stride_bytes=128,
            source_count=64,
            alignment=16,
            dtype="dtype:5",
            tokens_per_block=16,
            cp_world_size=1,
        )
        rank.local_mock_targets.SetInParent()
        rank.mock_cache.terminal_outcome_version = 1
        rank.mock_remote_write.SetInParent()
        return backend, cfg, response

    def test_readiness_and_zero_ratio_remote_tail_require_actual_native_receipts(self):
        backend, cfg, response = self.remote_fixture()
        rank = response.ranks[0]
        rt = rank.remote_runtime
        backend.startup(cfg)
        rank.mock_remote_write.pending_requests = 1
        self.assertFalse(backend.poll(1, drain_remote_completions=True).sample_complete)
        rank.mock_remote_write.pending_requests = 0
        rank.mock_remote_write.submitted = rank.mock_remote_write.completed = 1
        rank.remote_write_acknowledged_payload_bytes = 128
        result = backend.poll(1, drain_remote_completions=True)
        self.assertTrue(result.sample_complete, result.raw)
        self.assertEqual(result.samples[0].remote_write_acknowledged_payload_bytes, 128)
        rank.local_mock_targets.memory_enabled = True
        with self.assertRaisesRegex(ValueError, "owner.*target"):
            backend.startup(cfg)
        rank.local_mock_targets.memory_enabled = False
        rt.options["mock_remote_cache_read_delay_ms"] = "5"
        with self.assertRaisesRegex(ValueError, "effective engine option"):
            backend.startup(cfg)

    def test_remote_layout_and_credit_bounds_reject_contradictions(self):
        for field, value in (
            ("pool", 999),
            ("cp_world_size", 99),
            ("tokens_per_block", 999),
            ("alignment", 3),
            ("crc_footer_offset", 4),
            ("dtype", "unknown"),
            ("write_operations", 999),
            ("service_read_bytes", 1),
            ("source_bytes", 8192),
            ("total_bytes", 128),
            ("max_slice_bytes", 10**10),
        ):
            backend, cfg, response = self.remote_fixture()
            rt = response.ranks[0].remote_runtime
            if field in rt.credits:
                rt.credits[field] = value
            elif field in rt.limits:
                rt.limits[field] = value
            else:
                setattr(rt.layouts[0], field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                backend.startup(cfg)

    def test_sharded_full_and_swa_key_stride_and_inconsistent_peer(self):
        backend, cfg, response = self.remote_fixture()
        backend.ranks = (0, 1)
        response.requested_ranks.append(1)
        first = response.ranks[0]
        swa = first.groups.add()
        swa.CopyFrom(first.groups[0])
        swa.group_id = 1
        swa.group_type = 2
        swa.block_tokens = 8
        layout = first.remote_runtime.layouts.add()
        layout.CopyFrom(first.remote_runtime.layouts[0])
        layout.group = layout.pool = 1
        layout.attention_window = 64
        second = response.ranks.add()
        second.CopyFrom(first)
        for i, r in enumerate(response.ranks):
            r.world_rank = i
            r.service_instance = f"rank{i}"
            r.ownership.owner_world_rank.value = i
            r.ownership.participant_world_ranks[0] = i
            rt = r.remote_runtime
            rt.rank = i
            rt.expected_ranks = 2
            rt.callback_endpoint = f"127.0.0.1:{19080+i}"
            rt.credits.update(
                dict(
                    write_slots=4,
                    first_write_slot=4 * i,
                    service_read_bytes=1 << 19,
                    service_write_bytes=1 << 19,
                )
            )
            for l in rt.layouts:
                l.cp_world_size = 2
                l.cp_rank = i
        backend.startup(cfg)
        second.remote_runtime.layouts[1].cp_world_size = 1
        second.remote_runtime.layouts[1].cp_rank = 0
        with self.assertRaises(ValueError):
            backend.startup(cfg)

    def test_lost_service_path_preserves_native_poll_and_drain_receipts(self):
        backend, cfg, response = self.remote_fixture()
        backend.startup(cfg)
        backend.remote_binding = dict(
            launch=dict(event_path="/missing-service-events"),
            begin_bytes=0,
            device=0,
            inode=0,
        )
        response.ranks[0].remote_write_acknowledged_payload_bytes = 128
        for draining in (False, True):
            observed = backend.poll(1, drain_remote_completions=draining)
            self.assertTrue(observed.sample_complete, observed.raw)
            self.assertEqual(
                observed.samples[0].remote_write_acknowledged_payload_bytes, 128
            )
            self.assertIn("service_capture_error", observed.raw)
        self.assertTrue(
            backend.rpc.GetCachePressure.call_args.args[0].drain_remote_completions
        )
        self.assertNotIn("end_bytes", backend.remote_binding)

    def test_missing_launch_binding_records_reason_and_keeps_cleanup(self):
        import json
        import tempfile
        from contextlib import nullcontext
        from pathlib import Path
        from unittest.mock import patch

        for launch_contents in (None, "[]", "null", "injected binding exception"):
            with self.subTest(launch=launch_contents):
                backend, cfg, response = self.remote_fixture()
                for pool in response.ranks[0].pools:
                    pool.total_blocks = pool.free_blocks = pool.available_blocks = (
                        200000
                    )
                with tempfile.TemporaryDirectory() as tmp, patch(
                    "rtp_llm.test.continuous_pressure.adapters.rtp_llm_rpc.PressureSession.run",
                    side_effect=RuntimeError("injected supervisor failure"),
                ):
                    unexpected = launch_contents == "injected binding exception"
                    injection = (
                        patch(
                            "rtp_llm.test.continuous_pressure.analysis.remote_artifacts.bind_remote_capture",
                            side_effect=RuntimeError(launch_contents),
                        )
                        if unexpected
                        else nullcontext()
                    )
                    if launch_contents is not None and not unexpected:
                        (Path(tmp) / "launch.json").write_text(launch_contents)
                    error = (
                        launch_contents if unexpected else "injected supervisor failure"
                    )
                    with injection, self.assertRaisesRegex(RuntimeError, error):
                        backend.run(
                            settings={"evidence_policy": "tree_then_mock_remote"},
                            cases=[{"cache_len": 32768, "input_len": 36864}],
                            make_ids=lambda n, m: [5] * n,
                            encode_marker=lambda m: list(m.encode()),
                            decode_marker=lambda i: bytes(i).decode(),
                            directory=Path(tmp),
                            model_tokens=131072,
                            admission_requests=16,
                            batch_tokens=262144,
                            cache_alignment=16,
                            commit_tail_tokens=16,
                            remote_launch_manifest=Path(tmp) / "launch.json",
                        )
                    if not unexpected:
                        manifest = json.loads(
                            (Path(tmp) / "pressure/manifest.json").read_text()
                        )
                        self.assertIn(
                            "binding_error", manifest["remote_service_capture"]
                        )
                        self.assertEqual(
                            manifest["startup_pressure"]["response"]["ranks"][0][
                                "service_instance"
                            ],
                            "i",
                        )
                    self.assertTrue(
                        backend.rpc.GetCachePressure.call_args.args[
                            0
                        ].drain_remote_completions
                    )

    def test_startup_retries_only_transient_rank_connection_errors(self):
        from rtp_llm.test.continuous_pressure.adapters.rtp_llm_rpc import (
            pressure_settings,
        )

        backend = self.backend()
        transient = pb.CachePressureResponsePB(schema_version=1, requested_ranks=[0, 1])
        transient.ranks.add(world_rank=1, error="connection refused")
        permanent = pb.CachePressureResponsePB(schema_version=1, requested_ranks=[0, 1])
        permanent.ranks.add(world_rank=1, error="observer invalid")
        backend.rpc.GetCachePressure.side_effect = [transient, permanent]
        with self.assertRaisesRegex(ValueError, "observer invalid"):
            backend.startup(pressure_settings({}))
        self.assertEqual(backend.rpc.GetCachePressure.call_count, 2)


if __name__ == "__main__":
    unittest.main()
