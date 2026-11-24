import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.entry import plan_from_source
from rtp_llm.test.perf_test.tests.support.fixtures import (
    fixture_config,
    fixture_engine,
    fixture_workload,
)


def config(**values):
    return fixture_config(
        engine=fixture_engine(
            max_seq_len=131072,
            max_batch_tokens=262144,
            args={
                "reuse_cache": 1,
                "enable_memory_cache": 1,
                "memory_cache_size_mb": 1024,
            },
        ),
        workload=fixture_workload(
            max_input_len=36864,
            min_compute_len=4096,
            cache_lengths=[32768],
            compute_steps=[4096],
            pressure=values,
        ),
    )


class PressureWiringTest(unittest.TestCase):
    def test_finite_config_keeps_historical_fingerprints(self):
        resolved = resolve_config(fixture_config())
        self.assertNotIn("pressure", resolved.resolved_config["workload"])
        self.assertEqual(
            resolved.workload_fingerprint,
            "dbdb929abf302249c95f9acfc9dc893c3cedf0d562162802ea5e7bba2bac2de0",
        )
        self.assertEqual(
            resolved.config_fingerprint,
            "1a984a21570db3d712612db4a2c061755bc93b2cd2b55e723225550b26d481fb",
        )

    def test_planning_and_prelaunch_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"

            def plan(source):
                return plan_from_source(
                    source, source_dir=Path(directory), result_directory=str(output)
                )[0]

            p = plan(config())
            self.assertEqual(
                p.cases[0].identity["measurement_contract"], "continuous_pressure_v1"
            )
            self.assertEqual(
                p.resolved_config["workload"]["pressure"]["profile_steps"], 6
            )
            for section, key, value in [
                ("workload", "transport", "http_text"),
                ("workload", "batch_sizes", [2]),
                ("engine", "args", {"use_batch_decode_scheduler": 1}),
                ("workload", "pressure", {"host_ratio": 1.1}),
            ]:
                source = config()
                source[section][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    plan(source)
            self.assertFalse(output.exists())

    def test_kineto_step_limits_at_pressure_plan_resolution(self):
        for capture in (False, True):
            for steps in (1, 1000):
                result = resolve_config(config(capture=capture, profile_steps=steps))
                self.assertEqual(
                    result.resolved_config["workload"]["pressure"]["profile_steps"],
                    steps,
                )
            for steps in (0, 1001):
                with self.subTest(capture=capture, steps=steps), self.assertRaisesRegex(
                    ValueError, "profile_steps"
                ):
                    resolve_config(config(capture=capture, profile_steps=steps))

    def test_saved_pressure_run_and_profile_reject_unproducible_steps_before_launch(
        self,
    ):
        from types import SimpleNamespace

        from rtp_llm.test.perf_test.execution.local import UnifiedLocalExecutor

        with tempfile.TemporaryDirectory() as directory:
            plan = SimpleNamespace(
                result_directory=directory,
                resolved_config={
                    "workload": {"pressure": {"profile_steps": 1001, "capture": True}}
                },
            )
            runner = UnifiedLocalExecutor()
            for profile in (False, True):
                with self.subTest(profile=profile), self.assertRaisesRegex(
                    ValueError, "profile_steps"
                ):
                    runner._pressure_epoch(plan, profiling=profile)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_incomplete_epoch_never_healthy(self):
        from rtp_llm.test.perf_test.prefill.continuous_pressure import (
            pressure_epoch_valid,
        )

        self.assertFalse(pressure_epoch_valid({}, {}))
        manifest = {"complete": True, "session_id": "s"}
        summary = {
            "result": {
                "session_id": "s",
                "failure": None,
                "measurement_start": 1,
                "measurement_end": 2,
            },
            "window": {"measurement_requests": 1},
            "validity": {
                "execution_valid": True,
                "full_pressure_valid": True,
                "bidirectional_migration_valid": True,
            },
        }
        self.assertTrue(pressure_epoch_valid(manifest, summary))
        self.assertFalse(pressure_epoch_valid(dict(manifest, complete=False), summary))
        self.assertFalse(
            pressure_epoch_valid(dict(manifest, session_id="old"), summary)
        )

    def test_borrowed_session_fresh_epochs_report_and_request_accounting(self):
        self._exercise_framework(remote=False)

    def test_remote_explicit_certification_offline_report_and_preserved_failures(self):
        self._exercise_framework(remote=True)

    def test_large_owner_certified_archive_remains_verifiable_offline(self):
        self._exercise_framework(remote=True, large=True)

    def _exercise_framework(self, *, remote, large=False):
        from dataclasses import replace
        from unittest.mock import MagicMock

        from rtp_llm.test.continuous_pressure.analysis.remote_artifacts import (
            bind_remote_capture,
            certify_remote_epoch,
            read_remote_certificate,
            record_remote_end,
        )
        from rtp_llm.test.perf_test.execution.local import UnifiedLocalExecutor
        from rtp_llm.test.perf_test.run_directory import initialize_run_directory

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "run"
            source = config()
            event_path = Path(root) / "events.jsonl"
            launch_path = Path(root) / "launch.json"
            address = "127.0.0.1:18080"
            if remote:
                source["workload"]["pressure"].update(
                    evidence_policy="tree_then_mock_remote",
                    remote_launch_manifest=str(launch_path),
                )
                source["engine"]["args"]["mock_remote_cache"] = 1
                event_path.touch()
                if large:
                    # Complete native operations before our epoch exercise the size of
                    # the archived file without changing any of the measured receipts.
                    with event_path.open("w") as out:
                        for op in range(2, 32002):
                            for event in ("read_admitted", "read_payload", "complete"):
                                out.write(
                                    json.dumps(
                                        dict(
                                            event=event,
                                            operation=op,
                                            generation=1,
                                            steady_ns=1234567890,
                                            deadline_ns=0,
                                            payload_bytes=128,
                                        )
                                    )
                                    + "\n"
                                )
                    self.assertGreater(event_path.stat().st_size, 8 << 20)
                launch_path.write_text(
                    json.dumps(
                        dict(
                            launch_id="process",
                            event_path=str(event_path),
                            address=address,
                        )
                    )
                )
            plan, source = plan_from_source(
                source, source_dir=Path(root), result_directory=str(path)
            )
            initialize_run_directory(
                path, source_config=source, plan=plan, run_id="run", code_version="test"
            )
            engine = MagicMock()
            engine.assert_healthy = Mock()
            engine.__enter__.return_value = engine
            engine.client.port = 20000
            factory = Mock(return_value=engine)
            tokenizer = Mock()
            tokenizer.encode.return_value = [2]
            executor = UnifiedLocalExecutor(
                session_factory=factory, tokenizer_loader=lambda p: tokenizer
            )
            directories = []

            def serve(**kwargs):
                epoch = kwargs["directory"] / "pressure"
                epoch.mkdir(parents=True)
                directories.append(epoch)
                self.assertEqual(kwargs["cases"][0]["cache_len"], 32768)
                kwargs["encode_marker"]("fresh")
                tokenizer.encode.assert_called_with("fresh", add_special_tokens=False)
                kwargs["decode_marker"]([2])
                tokenizer.decode.assert_called_with(
                    [2], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                session = f"{len(directories):032x}"
                connect.return_value.__enter__.return_value.session_id = session
                manifest = dict(kwargs["manifest"], session_id=session, complete=True)
                summary = dict(
                    result=dict(
                        session_id=session,
                        failure=None,
                        measurement_start=1,
                        measurement_end=3,
                    ),
                    window={"measurement_requests": 2},
                    validity=dict(
                        execution_valid=True,
                        full_pressure_valid=True,
                        bidirectional_migration_valid=True,
                    ),
                )
                if kwargs["capture"]:
                    manifest["capture_requested"] = True
                    summary["result"]["profile"] = dict(
                        trace_valid=True, pressure_valid=True
                    )
                if remote:
                    runtime = dict(server_epoch=11, session_id=12, generation=1)
                    binding = bind_remote_capture(launch_path, runtime, address)
                    events = [
                        dict(event=name, operation=1, generation=1, payload_bytes=9)
                        for name in (
                            "write_admitted",
                            "write_received",
                            "write_ack",
                            "complete",
                        )
                    ]
                    with event_path.open("a") as out:
                        out.writelines(json.dumps(r) + "\n" for r in events)
                    binding["end_bytes"] = record_remote_end(binding)
                    manifest.update(
                        epoch={"0": "instance"},
                        remote_service_capture=dict(
                            run_id=session, runtime=runtime, binding=binding
                        ),
                        startup_pressure=dict(
                            response=dict(
                                ranks=[
                                    dict(
                                        service_instance="instance",
                                        remote_runtime=dict(
                                            runtime,
                                            options={
                                                "mock_remote_cache_address": address
                                            },
                                        ),
                                    )
                                ]
                            )
                        ),
                    )
                    summary["result"]["mock_remote"] = dict(
                        valid=True,
                        remote_read_payload_bytes=0,
                        remote_write_acknowledged_payload_bytes=9,
                    )
                (epoch / "manifest.json").write_text(json.dumps(manifest))
                (epoch / "summary.json").write_text(json.dumps(summary))
                row = dict(
                    session_id=session,
                    metadata={"case_index": 0},
                    kind="formal",
                    phase="measurement",
                    dispatch_s=1.2,
                    completed_s=3.5,
                    success=True,
                    terminal="completion",
                )
                rows = [
                    row,
                    dict(row, terminal="cancellation", success=False),
                    dict(
                        row, terminal="not_dispatched", dispatch_s=None, success=False
                    ),
                ]
                (epoch / "requests.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows)
                )
                return summary["result"]

            with patch(
                "rtp_llm.test.perf_test.prefill.continuous_pressure.RtpLlmPressureBackend.connect"
            ) as connect:
                connect.return_value.__enter__.return_value.run.side_effect = serve
                first = executor(plan)
                if remote:
                    self.assertNotEqual(first.exit_code, 0)
                    directory = directories[0]
                    report = lambda: executor(replace(plan, action="report"))
                    self.assertFalse(report()["complete"])
                    content = event_path.read_bytes()
                    event_path.write_bytes(
                        content
                        + json.dumps(
                            dict(
                                capture_complete=True,
                                event_count=content.count(b"\n"),
                                prefix_bytes=len(content),
                            )
                        ).encode()
                        + b"\n"
                    )
                    self.assertFalse(
                        report()["complete"]
                    )  # owner has not issued an attestation
                    budget = (16 if large else 8) << 20
                    if large:
                        for invalid in (True, 0, 1 << 40):
                            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                                ValueError, "budget"
                            ):
                                certify_remote_epoch(directory, max_bytes=invalid)
                    certificate = certify_remote_epoch(directory, max_bytes=budget)
                    self.assertEqual(certificate["scope"], "remote_service")
                    launch_path.rename(launch_path.with_suffix(".archived"))
                    event_path.rename(event_path.with_suffix(".archived"))
                    cert_path = directory / "remote_certificate.json"
                    saved = cert_path.read_bytes()
                    self.assertTrue(report()["complete"])
                    self.assertEqual(cert_path.read_bytes(), saved)
                    if large:
                        with self.assertRaisesRegex(ValueError, "budget"):
                            read_remote_certificate(directory, max_bytes=8 << 20)
                        with self.assertRaisesRegex(ValueError, "budget"):
                            read_remote_certificate(directory, max_bytes=1 << 40)
                        return
                    cert_path.write_text(
                        json.dumps(dict(certificate, run_id="previous"))
                    )
                    self.assertFalse(report()["complete"])
                    cert_path.write_bytes(saved)
                    # A valid service-only certificate cannot clear an original engine gate.
                    launch_path.with_suffix(".archived").rename(launch_path)
                    event_path.with_suffix(".archived").rename(event_path)
                    summary = json.loads((directory / "summary.json").read_text())
                    summary["validity"]["execution_valid"] = False
                    (directory / "summary.json").write_text(json.dumps(summary))
                    self.assertEqual(
                        certify_remote_epoch(directory)["scope"], "remote_service"
                    )
                    self.assertFalse(report()["complete"])
                    return
                self.assertEqual(first.exit_code, 0)
                metrics = first.case_results[0]["metrics"]
                self.assertEqual(
                    (metrics["completed"], metrics["cancelled"], metrics["sent"]),
                    (1, 1, 2),
                )
                self.assertEqual(
                    connect.call_args.kwargs["pressure_endpoint"], "127.0.0.1:20001"
                )
                self.assertEqual(executor(replace(plan, action="resume")).exit_code, 0)
                self.assertEqual(
                    executor(replace(plan, action="profile"))["status"], "ok"
                )
            self.assertEqual(
                (
                    factory.call_count,
                    engine.__enter__.call_count,
                    engine.__exit__.call_count,
                ),
                (3, 3, 3),
            )
            self.assertEqual(len(set(directories)), 3)
            report = executor(replace(plan, action="report"))
            self.assertTrue(report["complete"], report)
            self.assertEqual(report["epoch"]["session_id"], f"{2:032x}")
            self.assertEqual(metrics["session_id"], f"{1:032x}")
            # Prior measurement, same config, counts and case; external path remains current.
            for f in ("manifest.json", "summary.json", "requests.jsonl"):
                (directories[1] / f).write_bytes((directories[0] / f).read_bytes())
            self.assertFalse(executor(replace(plan, action="report"))["complete"])
            (directories[1] / "manifest.json").write_text("{}")
            self.assertFalse(executor(replace(plan, action="report"))["complete"])


if __name__ == "__main__":
    unittest.main()
