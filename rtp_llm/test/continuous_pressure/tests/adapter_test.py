import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from rtp_llm.test.continuous_pressure.adapters.capture import ProfileController
from rtp_llm.test.continuous_pressure.adapters.rtp_llm_rpc import RtpLlmPressureBackend
from rtp_llm.test.continuous_pressure.analysis.remote_artifacts import (
    bind_remote_capture,
    certify_remote_epoch,
    record_remote_end,
)
from rtp_llm.test.continuous_pressure.session import DispatchClosed


class AdapterTest(unittest.TestCase):
    def test_dispatch_after_preparation_and_business_failure(self):
        events = []
        request = SimpleNamespace(
            id="pending", parameters={"mock_cache_request": SimpleNamespace()}
        )
        payload = {"input_ids": [5, 6]}

        def build(**kwargs):
            events.append(("prepare", kwargs["input_ids"]))
            return request

        class Stub:
            def ModelStreamInfer(self, messages, **kwargs):
                sent = next(messages)
                events.append(
                    ("sent", sent.id, sent.parameters["mock_cache_request"].bool_param)
                )
                return iter([SimpleNamespace(error_message="business failure")])

        backend = RtpLlmPressureBackend(
            dash_stub=Stub(),
            pressure_stub=None,
            messages=None,
            build_request=build,
            sampling=None,
            ranks=(0,),
            timeout=2,
        )

        def before(payload):
            payload["mock_cache_request"] = False
            events.append("before")
            return "actual"

        result = backend.request(
            payload, lambda call: events.append("registered"), before
        )
        self.assertEqual(
            events,
            [("prepare", [5, 6]), "before", ("sent", "actual", False), "registered"],
        )
        self.assertFalse(result["success"])
        with self.assertRaises(DispatchClosed):
            backend.request(
                payload,
                lambda call: self.fail(),
                lambda p: (_ for _ in ()).throw(DispatchClosed()),
            )
        self.assertEqual(events[-1], ("prepare", [5, 6]))

    def test_kineto_step_bound_precedes_standalone_transport(self):
        from unittest.mock import patch

        from rtp_llm.test.continuous_pressure.adapters.rtp_llm_rpc import (
            run_rtp_pressure,
        )

        options = dict(
            dash_endpoint="unused",
            pressure_endpoint="unused",
            ranks=(0,),
            cases=[dict(input_len=17, cache_len=16)],
            capture=True,
            encode_marker=lambda s: [],
            decode_marker=lambda ids: "",
        )
        with patch.object(RtpLlmPressureBackend, "connect") as connect:
            for steps in (0, 1001):
                with self.assertRaisesRegex(ValueError, "profile_steps"):
                    run_rtp_pressure(settings=dict(profile_steps=steps), **options)
            connect.assert_not_called()
            run_rtp_pressure(settings=dict(profile_steps=1000), **options)
            connect.assert_called_once()

    def test_profile_start_and_cancel_do_not_wait_on_export(self):
        started, stopped = threading.Event(), threading.Event()

        def wait():
            self.assertTrue(stopped.wait(2))
            return {}

        control = ProfileController(started.set, wait, stopped.set)
        control.start()
        self.assertTrue(started.wait(2))
        self.assertFalse(control.done())
        control.request_stop()
        self.assertTrue(stopped.wait(2))
        control.result(2)

    def test_independent_import_and_installed_sdk_guard(self):
        program = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if set(fullname.split(".")) & {"perf_test","cache_grid","torch","grpc","proto"}:
            raise ImportError(fullname)
sys.meta_path.insert(0,Guard())
from rtp_llm.test.continuous_pressure.adapters.rtp_llm_rpc import RtpLlmPressureBackend
try:
    RtpLlmPressureBackend.connect(dash_endpoint="unused",pressure_endpoint="unused",ranks=(0,))
except RuntimeError as e:
    assert "generated model RPC" in str(e)
else:
    raise AssertionError("SDK dependency guard missing")
"""
        subprocess.run(
            [sys.executable, "-S", "-c", program],
            cwd=Path(__file__).resolve().parents[4],
            check=True,
        )

    def test_post_shutdown_certificate_reconciles_only_bound_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "server.jsonl"

            def line(event, count, op=99):
                return (
                    json.dumps(
                        dict(
                            event=event, generation=2, operation=op, payload_bytes=count
                        )
                    ).encode()
                    + b"\n"
                )

            probe = line("write_ack", 1)
            path.write_bytes(probe)
            runtime = dict(server_epoch=17, session_id=31, generation=2)
            launch = dict(
                launch_id="per-process-nonce",
                event_path=str(path),
                address="127.0.0.1:18080",
            )
            source = root / "launch.json"
            source.write_text(json.dumps(launch))
            binding = bind_remote_capture(source, runtime, launch["address"])
            with path.open("ab") as stream:
                for name in ("read_admitted", "read_payload", "complete"):
                    stream.write(line(name, 7, 1))
                for name in (
                    "write_admitted",
                    "write_received",
                    "write_ack",
                    "complete",
                ):
                    stream.write(line(name, 9, 2))
            binding["end_bytes"] = record_remote_end(binding)
            native = dict(
                runtime, options={"mock_remote_cache_address": launch["address"]}
            )
            manifest = dict(
                complete=True,
                session_id="p",
                epoch={"0": "engine"},
                startup_pressure={
                    "response": {
                        "ranks": [
                            dict(service_instance="engine", remote_runtime=native)
                        ]
                    }
                },
                remote_service_capture=dict(
                    run_id="p", runtime=runtime, binding=binding
                ),
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            remote = dict(
                valid=True,
                remote_read_payload_bytes=7,
                remote_write_acknowledged_payload_bytes=9,
            )
            summary = dict(
                result=dict(session_id="p", failure=None, mock_remote=remote)
            )
            (root / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                certify_remote_epoch(root)
            content = path.read_bytes()
            footer = (
                json.dumps(
                    dict(
                        capture_complete=True, event_count=8, prefix_bytes=len(content)
                    )
                ).encode()
                + b"\n"
            )
            path.write_bytes(content + footer)
            certificate = certify_remote_epoch(root)
            self.assertEqual(certificate["remote_write_acknowledged_payload_bytes"], 9)
            self.assertEqual(certificate["window_events"], 7)
            self.assertEqual(certificate["scope"], "remote_service")
            self.assertNotIn("epoch_complete", certificate)
            summary["validity"] = {
                k: True
                for k in (
                    "execution_valid",
                    "full_pressure_valid",
                    "bidirectional_migration_valid",
                )
            }
            (root / "summary.json").write_text(json.dumps(summary))
            self.assertNotIn("epoch_complete", certify_remote_epoch(root))
            summary["result"].update(measurement_start=1.0, measurement_end=2.0)
            summary["window"] = dict(measurement_requests=2)
            (root / "summary.json").write_text(json.dumps(summary))
            self.assertNotIn("epoch_complete", certify_remote_epoch(root))
            summary["result"]["failure"] = "local_write_evidence_invalid"
            remote.update(valid=False, remote_drain_valid=True)
            (root / "summary.json").write_text(json.dumps(summary))
            self.assertEqual(certify_remote_epoch(root)["scope"], "remote_service")
            remote["remote_drain_valid"] = False
            (root / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "native Remote drain"):
                certify_remote_epoch(root)
            summary["result"]["failure"] = None
            remote.update(valid=True, remote_drain_valid=True)
            (root / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                bind_remote_capture(source, runtime, launch["address"])
            source.write_text(json.dumps(dict(launch, launch_id="other")))
            with self.assertRaisesRegex(ValueError, "launch"):
                certify_remote_epoch(root)
            source.write_text(json.dumps(launch))
            path.write_bytes(content + footer[:-2])
            with self.assertRaises(ValueError):
                certify_remote_epoch(root)


class SeedBankTest(unittest.TestCase):
    def test_freezes_successful_seeds_by_submission_and_rejects_collisions(self):
        from rtp_llm.test.continuous_pressure.adapters.rtp_llm_rpc import (
            _Payloads,
            pressure_settings,
        )

        config = pressure_settings({"cohort_size": 2})
        plan = SimpleNamespace(
            seed_count=4, seed_input_tokens=160, fill_input_tokens=160
        )
        cases = [{"input_len": 160, "cache_len": 64}]
        encode = lambda text: list(text.encode("ascii"))
        decode = lambda ids: bytes(ids).decode("ascii")

        def bank(session, encoder=encode, blocks=(64,)):
            return _Payloads(
                config,
                plan,
                cases,
                lambda n, m: [55] * n,
                encoder,
                decode,
                session,
                blocks,
            )

        first, second = bank("a" * 32), bank("b" * 32)
        for item in (first, second):
            for job in (2, 0, 1):
                item.confirm(item.make("seed", job, None))
        cold = first.make("formal", 4, 1)["input_ids"]
        promoted = first.make("formal", 5, 2)["input_ids"]
        self.assertEqual(cold[:64], promoted[:64])
        self.assertNotEqual(cold[64:128], promoted[64:128])
        self.assertNotEqual(cold[:64], second.make("formal", 4, 1)["input_ids"][:64])
        self.assertNotEqual(first.make("fill", 6, None)["input_ids"][:64], cold[:64])
        for codec in (lambda text: [9], lambda text: encode(text)[:8]):
            with self.subTest(codec=codec), self.assertRaises(ValueError):
                bank("a" * 32, codec).make("seed", 0, None)
        with self.assertRaises(ValueError):
            bank("a" * 32, blocks=(16,)).make("seed", 0, None)

    def test_variable_length_marker_prefix_codes_can_be_recovered_from_first_block(
        self,
    ):
        from rtp_llm.test.continuous_pressure.adapters.rtp_llm_rpc import (
            _Payloads,
            pressure_settings,
        )

        encoded = {"0": [3], "1": [4]}
        words = (
            "0_epoch_seed_",
            "1_epoch_seed_",
            "2_epoch_seed_",
            "3_epoch_formal_",
            "4_epoch_formal_",
        )
        for index, word in enumerate(words):
            encoded[word] = [3, 4] + [3] * index
        decoded = {tuple(ids): s for s, ids in encoded.items()}
        plan = SimpleNamespace(seed_count=3, seed_input_tokens=60, fill_input_tokens=60)
        bank = _Payloads(
            pressure_settings({"cohort_size": 2}),
            plan,
            [{"input_len": 60, "cache_len": 16}],
            lambda n, m: [9] * n,
            encoded.__getitem__,
            lambda ids: decoded[tuple(ids)],
            "epoch",
            (16,),
        )
        for i in range(3):
            payload = bank.make("seed", i, None)
            bank.confirm(payload)
        for seq, job in ((1, 3), (2, 4)):
            payload = bank.make("formal", job, seq)
            code = payload["input_ids"][17:32]  # exclude the divergent first token
            data = []
            stream = iter(code)
            for token in stream:
                if token == 3:
                    following = next(stream)
                    if following == 4:
                        break
                    self.assertEqual(following, 3)
                data.append(token)
            else:
                self.fail("self-delimiting marker has no terminator")
            self.assertEqual(decoded[tuple(data)], payload["nonce"])

    def test_fresh_seed_marker_is_unique_across_different_grid_frame_widths(self):
        from rtp_llm.test.continuous_pressure.adapters.rtp_llm_rpc import (
            _Payloads,
            pressure_settings,
        )

        epochs = ("a" * 32, "b" * 32)
        markers = {
            f"0_{epochs[0]}_seed_": [10, 11],
            f"0_{epochs[1]}_seed_": [10, 11, 7, 8, 15],
            "0": [7],
            "1": [8],
        }
        inverse = {tuple(v): k for k, v in markers.items()}
        plan = SimpleNamespace(
            seed_count=1, seed_input_tokens=32784, fill_input_tokens=32784
        )

        def exact(n, marker):
            return ([10, 11, 7, 8, 15, 7, 8, 8, 8, 8] + [15] * n)[:n]

        seeds = []
        for epoch, minimum in zip(epochs, (4, 10)):
            bank = _Payloads(
                pressure_settings({"cohort_size": 4}),
                plan,
                [
                    {"input_len": 100, "cache_len": minimum},
                    {"input_len": 32784, "cache_len": 32768},
                ],
                exact,
                markers.__getitem__,
                lambda ids: inverse[tuple(ids)],
                epoch,
                (16,),
            )
            seeds.append(bank.make("seed", 0, None)["input_ids"])
        self.assertNotEqual(seeds[0][:16], seeds[1][:16])

    def test_missing_barrier_evidence_and_control_ack(self):
        from rtp_llm.test.continuous_pressure.adapters.rtp_llm_rpc import (
            _validate_drained,
        )

        with self.assertRaisesRegex(ValueError, "pending"):
            _validate_drained(
                [
                    SimpleNamespace(
                        rank=0,
                        mock_remote_write=SimpleNamespace(
                            pending_requests=1,
                            pending_intents=0,
                            retained_descriptors=0,
                            faulted=False,
                            quarantined_blocks=0,
                            failed=0,
                            rejected=0,
                        ),
                    )
                ],
                {0},
            )

    def test_default_headroom_and_reject_legacy(self):
        from rtp_llm.test.continuous_pressure.adapters.rtp_llm_rpc import (
            pressure_settings,
        )

        self.assertEqual(pressure_settings({}).rate_control, "adaptive_backlog")
        for option in ({"family_schedule": "round_robin"}, {"rate_control": "fixed"}):
            with self.assertRaises(ValueError):
                pressure_settings(option)
        self.assertEqual(pressure_settings({}).profile_steps, 6)
        self.assertEqual(pressure_settings({}).fill_host_max_ratio, 0.80)
        with self.assertRaisesRegex(ValueError, "real-fill"):
            pressure_settings({"evidence_policy": "mock_memory"})


if __name__ == "__main__":
    unittest.main()
