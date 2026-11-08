"""CPU contracts for native K3.1 role selection and strict event transport."""

import gzip
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch

from rtp_llm.test.smoke.kimi_k31_bazel import (
    k31_ci,
    pd_config,
    remote_pd_test,
)
from rtp_llm.test.smoke.kimi_k3 import result_check


class K31CiTest(unittest.TestCase):
    def test_x86_worker_retains_the_eight_gpu_and_architecture_requirements(self):
        worker = k31_ci.K31Worker(None)
        self.addCleanup(worker.locks.close)
        with patch.object(k31_ci.platform, "machine", return_value="aarch64"):
            with self.assertRaisesRegex(ValueError, "worker architecture"):
                worker._lock_gpus()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(k31_ci.platform, "machine", return_value="x86_64"),
        ):
            rows = "\n".join(f"{i}, GPU-{i}, NVIDIA L20D" for i in range(8))
            with patch(
                "rtp_llm.test.smoke.kimi_k31_bazel.worker.subprocess.check_output",
                side_effect=[rows, ""],
            ):
                worker._lock_gpus(Path(tmp))
            worker.locks.close()
            rows = "\n".join(f"{i}, GPU-{i}, NVIDIA L20D" for i in range(4))
            with patch(
                "rtp_llm.test.smoke.kimi_k31_bazel.worker.subprocess.check_output",
                return_value=rows,
            ):
                with self.assertRaisesRegex(ValueError, "exactly eight GPUs"):
                    worker._lock_gpus(Path(tmp))

    def test_worker_rejects_unknown_layout_before_gpu_reservation(self):
        worker = k31_ci.K31Worker(None)
        worker.config = {"worker_layout": "unknown"}
        self.addCleanup(worker.locks.close)
        with self.assertRaisesRegex(ValueError, "worker layout"):
            worker._lock_gpus()

    def test_cleanup_keeps_forced_exit_distinct_from_graceful_rank_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cleanup.json"
            cleanup = dict(
                stopped=True,
                gpu_contexts_verified=True,
                errors=[],
                unexpected_gpu_contexts=[],
            )
            path.write_text(json.dumps(cleanup))
            result = k31_ci.audit_service_cleanup(tmp, -9)
            self.assertTrue(result["supervisor_force_killed"])
            self.assertFalse(result["backend_rank_exitcodes_verified"])
            self.assertFalse(result["graceful_backend_shutdown_verified"])
            service_log = Path(tmp) / "service.log"
            service_log.write_text("Force killing process 123\n")
            self.assertTrue(
                k31_ci.audit_service_cleanup(tmp, 0)["forced_child_cleanup_observed"]
            )
            service_log.write_text("Fatal Python error: Aborted\n")
            with self.assertRaisesRegex(ValueError, "fatal error"):
                k31_ci.audit_service_cleanup(tmp, 0)
            for text in (
                "DeepGEMM grid sync timeout: sm=3",
                "CUDA error: unspecified launch failure",
            ):
                service_log.write_text(text + "\n")
                with (
                    self.subTest(text=text),
                    self.assertRaisesRegex(ValueError, "fatal error"),
                ):
                    k31_ci.audit_service_cleanup(tmp, -9)
            service_log.unlink()
            for code in (-6, -11, 1, None):
                with (
                    self.subTest(code=code),
                    self.assertRaisesRegex(ValueError, "supervisor exit"),
                ):
                    k31_ci.audit_service_cleanup(tmp, code)
            for field, value in (
                ("stopped", False),
                ("gpu_contexts_verified", False),
                ("errors", ["service exited unexpectedly"]),
                ("unexpected_gpu_contexts", [123]),
            ):
                path.write_text(json.dumps(dict(cleanup, **{field: value})))
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(ValueError, "verification failed"),
                ):
                    k31_ci.audit_service_cleanup(tmp, 0)

    def test_native_fixture_terminates_only_its_own_group(self):
        worker = k31_ci.K31Worker(None)
        worker.service = Mock(pid=12345)
        worker.service.poll.return_value = None
        worker.service.returncode = -15
        self.addCleanup(worker.locks.close)
        with tempfile.TemporaryDirectory() as tmp:
            worker.task = Path(tmp)
            with (
                patch.object(k31_ci.os, "getpgid", return_value=12345),
                patch.object(k31_ci.os, "getsid", return_value=12345),
                patch("rtp_llm.test.smoke.kimi_k3.worker.stop_process") as stop,
            ):
                worker.stop_service()
            stop.assert_called_once_with(worker.service, timeout=330)
            result = json.loads((Path(tmp) / "termination.json").read_text())
            self.assertEqual(result["initial_signal"], 15)
            self.assertFalse(result["supervisor_force_killed"])
            self.assertFalse(result["graceful_backend_shutdown_verified"])

    def test_native_fixture_refuses_an_unowned_group_or_session(self):
        worker = k31_ci.K31Worker(None)
        worker.service = Mock(pid=12345)
        worker.service.poll.return_value = None
        self.addCleanup(worker.locks.close)
        for group, session in ((12344, 12345), (12345, 12344)):
            with (
                self.subTest(group=group, session=session),
                patch.object(k31_ci.os, "getpgid", return_value=group),
                patch.object(k31_ci.os, "getsid", return_value=session),
                patch.object(k31_ci.os, "killpg") as kill,
                self.assertRaisesRegex(ValueError, "does not own"),
            ):
                worker.stop_service()
            kill.assert_not_called()

    def test_native_fixture_preserves_an_unexpected_supervisor_exit(self):
        worker = k31_ci.K31Worker(None)
        worker.service = Mock(pid=12345)
        worker.service.poll.return_value = -11
        self.addCleanup(worker.locks.close)
        with patch.object(k31_ci.os, "killpg") as kill:
            worker.stop_service()
        kill.assert_not_called()
        worker.service.wait.assert_not_called()

    def test_native_worker_reads_persistent_logs_even_under_bazel_console_mode(self):
        worker = k31_ci.K31Worker(None)
        worker.task = Path("/owned/task/prefill")
        with patch.dict(k31_ci.os.environ, {"FT_SERVER_TEST": "1"}):
            self.assertEqual(
                worker.engine_log(), Path("/owned/task/prefill/logs/engine.log")
            )

    def test_native_worker_combines_python_rank_and_cpp_cwd_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = k31_ci.K31Worker(None)
            worker.task = Path(tmp)
            rank_logs = worker.task / "service" / "logs"
            rank_logs.mkdir(parents=True)
            for rank in range(8):
                (rank_logs / f"main_{rank}.log").write_text(
                    "[K3_SMOKE_EVENT] "
                    + json.dumps({"time_ns": 100, "event": "python_forward"})
                    + "\n"
                )
            cpp_logs = worker.task / "logs"
            cpp_logs.mkdir()
            engine = cpp_logs / "engine.log"
            engine.write_text(
                "[RANK 4] [K3_SMOKE_EVENT] "
                + json.dumps({"time_ns": 200, "event": "cpp_cancel"})
                + "\n"
            )
            events = worker.events()
            self.assertEqual(
                [row["event"] for row in events[4]],
                ["python_forward", "cpp_cancel"],
            )
            self.assertEqual(
                [len(events[rank]) for rank in range(8)], [1, 1, 1, 1, 2, 1, 1, 1]
            )
            engine.unlink()
            with self.assertRaises(FileNotFoundError):
                worker.events()

    def test_failed_snapshot_retains_the_actual_cause_and_still_returns_503(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = k31_ci.K31Worker(None)
            worker.task = Path(tmp)

            def missing_rank_log():
                raise FileNotFoundError("main_0.log")

            token = "b" * 64
            server = k31_ci.EvidenceServer(
                ("127.0.0.1", 0), token, missing_rank_log, worker.evidence_error
            )
            self.addCleanup(server.close)
            request = urllib.request.Request(
                "http://127.0.0.1:"
                + str(server.server.server_port)
                + "/execution-events",
                headers={"Authorization": "Bearer " + token},
            )
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with self.assertRaises(urllib.error.HTTPError) as result:
                opener.open(request, timeout=3)
            self.assertEqual(result.exception.code, 503)
            diagnostic = json.loads(
                (worker.task / "execution-evidence-error.json").read_text()
            )
            self.assertEqual(
                diagnostic, {"type": "FileNotFoundError", "message": "main_0.log"}
            )

    def events(self):
        return {
            rank: [{"time_ns": 100, "event": "actual_forward"}] for rank in range(8)
        }

    def test_event_transport_preserves_all_ranks_and_rejects_stale_identity(self):
        payload = k31_ci.encode_events(self.events(), "run1", "decode")
        decoded = k31_ci.decode_events(payload, "run1", "decode")
        self.assertEqual(
            {rank: list(rows) for rank, rows in decoded.items()}, self.events()
        )
        with self.assertRaisesRegex(ValueError, "different run"):
            k31_ci.decode_events(payload, "run2", "decode")
        with self.assertRaisesRegex(ValueError, "different run"):
            k31_ci.decode_events(payload, "run1", "prefill")
        events = self.events()
        del events[7]
        with self.assertRaisesRegex(ValueError, "all eight ranks"):
            k31_ci.decode_events(
                k31_ci.encode_events(events, "run1", "decode"), "run1", "decode"
            )

    def test_compressed_payload_budget_is_checked_before_json_decode(self):
        with patch.object(k31_ci, "MAX_EVENT_BYTES", 100):
            with self.assertRaisesRegex(ValueError, "uncompressed budget"):
                k31_ci.decode_events(gzip.compress(b" " * 101), "run1", "decode")

    def test_sender_reports_the_complete_snapshot_size_when_over_budget(self):
        with patch.object(k31_ci, "MAX_EVENT_BYTES", 100):
            with self.assertRaisesRegex(
                ValueError, r"uncompressed budget: \d+ > 100 bytes"
            ):
                k31_ci.encode_events(self.events(), "run1", "decode")

    def test_peer_endpoint_requires_token_and_serves_only_events(self):
        token = "a" * 64
        payload = k31_ci.encode_events(self.events(), "run1", "decode")
        server = k31_ci.EvidenceServer(("127.0.0.1", 0), token, lambda: payload)
        self.addCleanup(server.close)
        url = "http://127.0.0.1:" + str(server.server.server_port)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with self.assertRaises(urllib.error.HTTPError) as error:
            opener.open(url + "/execution-events", timeout=3)
        self.assertEqual(error.exception.code, 403)
        headers = {"Authorization": "Bearer " + token}
        with opener.open(
            urllib.request.Request(url + "/execution-events", headers=headers),
            timeout=3,
        ) as response:
            self.assertEqual(response.read(), payload)
        with self.assertRaises(urllib.error.HTTPError) as error:
            opener.open(
                urllib.request.Request(url + "/../config.json", headers=headers),
                timeout=3,
            )
        self.assertEqual(error.exception.code, 404)

    def test_geometry_delegates_to_the_traditional_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, launch = Path(tmp) / "engine.log", Path(tmp) / "launch.json"
            engine.write_text("runtime evidence")
            launch.write_text("{}")
            with patch("rtp_llm.test.smoke.kimi_k3.k31_smoke_capacity.cache_geometry",
                       return_value={"runtime_verified": True}) as shared:
                result = k31_ci.runtime_geometry(engine, launch, "a" * 40)
            shared.assert_called_once_with(
                engine, "prefill", 8, long_case=True, prefill_tail_blocks=None
            )
            self.assertTrue(result["runtime_verified"])

    def test_native_k31_result_requires_execution_audit_not_just_request_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accuracy = root / "accuracy.json"
            accuracy.write_text('{"passed":true}')
            audit = root / "execution-audit.json"
            audit.write_text(
                json.dumps(
                    dict(passed=True, complete_smoke_passed=False, native_K31_MTP=True)
                )
            )
            self.assertFalse(k31_ci.check_ci_result(accuracy)["passed"])
            audit.write_text(
                json.dumps(
                    dict(passed=True, complete_smoke_passed=True, native_K31_MTP=True)
                )
            )
            self.assertFalse(k31_ci.check_ci_result(accuracy)["passed"])

    def test_k31_profile_keeps_shared_precision_options_and_isolates_evidence_ports(self):
        host = dict(
            checkpoint="/data/target",
            draft_checkpoint="/data/draft",
            allow_checkpoint_root="/data",
            artifact_root="/data/output",
        )
        item = dict(
            external_revision="a" * 40,
            internal_revision="b" * 40,
            smoke_profile="k31",
            decode_dp=2,
            prefill=dict(host, ip="192.0.2.1", port=27188),
            decode=dict(host, ip="192.0.2.2", port=28188),
        )
        for role, blocks, port in (("prefill", 96, 27280), ("decode", 192, 28280)):
            item[role].update(test_block_num=blocks, evidence_port=port)
        pd_config.validate_config(item, local_role="prefill")
        self.assertTrue(item["kda_nvfp4"])
        self.assertTrue(item["dsa_nvfp4"])
        item["kda_nvfp4"] = False
        pd_config.validate_config(item, local_role="prefill")
        self.assertFalse(item["kda_nvfp4"])
        self.assertTrue(item["dsa_nvfp4"])
        item["fp8_kv_cache"] = True
        pd_config.validate_config(item, local_role="prefill")
        item["decode"]["evidence_port"] = item["decode"]["port"] + 1
        with self.assertRaisesRegex(ValueError, "overlaps"):
            pd_config.validate_config(item, local_role="prefill")

    def test_native_cleanup_requires_shared_projection_and_crc_audits(self):
        worker = k31_ci.K31Worker(None)
        self.addCleanup(worker.locks.close)
        worker.service = Mock(returncode=0)
        worker.config = dict(kda_nvfp4=True, dsa_nvfp4=False)
        worker.host = dict(block_tree_transfer_worker_count=1,
                           memory_cache_max_descriptors_per_transfer_batch=4)
        worker.projection_audit_required = True
        with tempfile.TemporaryDirectory() as tmp:
            worker.task = Path(tmp)
            with (
                patch.object(k31_ci.Worker, "close"),
                patch.object(k31_ci, "audit_service_cleanup", return_value=dict(passed=True)),
                patch("rtp_llm.test.smoke.kimi_k3.nvfp4_evidence.audit_projection_precision",
                      return_value=dict(passed=True)) as precision,
                patch("rtp_llm.test.smoke.kimi_k3.k31_two_host_smoke.audit_crc_workspace",
                      return_value={}) as crc,
            ):
                worker.close()
                precision.assert_called_once_with(worker.task / "service", worker.config)
                crc.assert_called_once_with(worker.task, 1, 4)
                self.assertTrue(json.loads((worker.task / "projection-precision.json").read_text())["passed"])
                precision.side_effect = ValueError("NVFP4 projections never executed")
                with self.assertRaisesRegex(RuntimeError, "never executed"):
                    worker.close()

    def test_native_entry_selects_profile_from_pinned_config_and_broker_environment(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            pins = dict(external_revision="a" * 40, internal_revision="b" * 40)
            connection = dict(
                url="http://192.0.2.1:9000", token="a" * 64, run_id="k31-pd-unittest"
            )
            environment = dict(
                TEST_TOTAL_SHARDS="2",
                TEST_SHARD_INDEX="0",
                TEST_SHARD_STATUS_FILE=str(root / "status"),
                K31_PD_CONFIG=str(config),
            )
            for profile, embedded in (("k3", True), ("k31", False)):
                with self.subTest(profile=profile):
                    output = root / profile
                    output.mkdir()
                    payload = dict(pins, smoke_profile=profile,
                                   prefill=dict(test_block_num=96),
                                   decode=dict(test_block_num=192))
                    selected_environment = dict(
                        environment, TEST_UNDECLARED_OUTPUTS_DIR=str(output)
                    )
                    if embedded:
                        payload["broker"] = connection
                    else:
                        selected_environment.update(
                            K31_PD_BROKER_URL=connection["url"],
                            K31_PD_BROKER_TOKEN=connection["token"],
                            K31_PD_RUN_ID=connection["run_id"],
                        )
                    config.write_text(json.dumps(payload))
                    with (
                        patch.dict("os.environ", selected_environment, clear=True),
                        patch.object(
                            remote_pd_test, "advertise_ip", return_value="192.0.2.2"
                        ),
                        patch.object(remote_pd_test.signal, "signal"),
                        patch.object(remote_pd_test, "run_shard") as run,
                    ):
                        run.return_value = dict(
                            passed=True, role="prefill", errors=[], cleanup_errors=[]
                        )
                        self.assertEqual(remote_pd_test.main(), 0)
                        self.assertEqual(
                            run.call_args.args[2].run_id, connection["run_id"]
                        )
                        self.assertNotIn("broker", run.call_args.args[0])
                        if profile == "k31":
                            self.assertIs(run.call_args.args[5], k31_ci.K31Worker)
                            self.assertIs(run.call_args.args[6], k31_ci.check_ci_result)
                        else:
                            self.assertIs(run.call_args.args[5], k31_ci.Worker)

    def test_native_entry_rejects_partial_or_ambiguous_broker_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            payload = dict(external_revision="a" * 40, internal_revision="b" * 40)
            environment = dict(
                TEST_TOTAL_SHARDS="2",
                TEST_SHARD_INDEX="0",
                TEST_SHARD_STATUS_FILE=str(root / "status"),
                K31_PD_CONFIG=str(config),
                K31_PD_BROKER_URL="http://192.0.2.1:9000",
            )
            config.write_text(json.dumps(payload))
            with (
                patch.dict("os.environ", environment, clear=True),
                self.assertRaisesRegex(ValueError, "exactly one complete"),
            ):
                remote_pd_test.main()
            payload["broker"] = dict(
                url=environment["K31_PD_BROKER_URL"],
                token="a" * 64,
                run_id="k31-pd-unittest",
            )
            config.write_text(json.dumps(payload))
            environment.update(
                K31_PD_BROKER_TOKEN="a" * 64, K31_PD_RUN_ID="k31-pd-unittest"
            )
            with (
                patch.dict("os.environ", environment, clear=True),
                self.assertRaisesRegex(ValueError, "exactly one complete"),
            ):
                remote_pd_test.main()

    def test_request_adapter_uses_native_draft_full_suite_and_both_decode_owners(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = k31_ci.K31Worker.__new__(k31_ci.K31Worker)
            worker.role, worker.task = "prefill", root / "k31-pd-unittest" / "prefill"
            worker.task.mkdir(parents=True)
            worker.workspace = root / "workspace"
            worker.host = dict(
                ip="192.0.2.1",
                port=27188,
                checkpoint="/models/target",
                draft_checkpoint="/models/draft",
                test_block_num=96,
            )
            worker.config = dict(
                external_revision="a" * 40, decode=dict(ip="192.0.2.2", port=28188)
            )
            worker._request_phase = "regular"
            with (
                patch.object(
                    k31_ci, "runtime_geometry", return_value={"runtime_verified": True}
                ),
                patch.object(worker, "status", return_value={"healthy": True}),
            ):
                command = worker.request_command()
            self.assertEqual(Path(command[1]).name, "k31_full_smoke.py")
            self.assertEqual(
                command[command.index("--draft-checkpoint") + 1], "/models/draft"
            )
            self.assertNotIn("--synthetic-draft-checkpoint", command)
            self.assertEqual(
                command[command.index("--decode-batch-profile") + 1], "smoke"
            )
            owners = [
                command[index + 1]
                for index, value in enumerate(command)
                if value == "--decode-role-addr"
            ]
            self.assertEqual(owners, ["192.0.2.2:28188:28189", "192.0.2.2:28224:28225"])
            self.assertEqual(
                worker.launcher_options(),
                [
                    "--model-type",
                    "kimi_k31",
                    "--execution-evidence",
                    "--test-block-num",
                    "96",
                    "--core-dump-limit-mib",
                    "64",
                ],
            )


if __name__ == "__main__":
    unittest.main()
