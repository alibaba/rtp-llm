import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexlb_test_framework.scenario.actions import engine_control as ec


class EngineControlTests(unittest.TestCase):
    def plan(self):
        return NS(
            path="control",
            reference=lambda value, kind: self.assertEqual(kind, "string"),
        )

    def deadline(self):
        return NS(check=lambda: None, remaining=lambda: 0.25)

    def test_rejects_unknown_operations_and_perf(self):
        for params in [
            dict(operation="inject", targets=["p0"]),
            dict(operation="set_perf", targets=["p0"], perf={"invented": 1}),
            dict(operation="stop", targets=["p0"], perf={}),
        ]:
            with self.assertRaises(ValueError):
                ec.validate(params, self.plan())

    def test_rejects_nonfinite_boolean_and_invalid_caps(self):
        for perf in [
            {"decode_scale": float("nan")},
            {"prefill_fixed_ms": True},
            {"max_waiting_batches": 1.5},
            {"max_prefill_concurrency": 0},
        ]:
            with self.assertRaises(ValueError):
                ec.validate(
                    dict(operation="set_perf", targets=["p0"], perf=perf), self.plan()
                )

    def test_accepts_typed_engine_name_reference(self):
        params = dict(
            operation="set_perf",
            targets=[{"$ref": "stages.add.output.engine"}],
            perf={"prefill_fixed_ms": 10000},
        )
        self.assertEqual(ec.validate(params, self.plan()), params)

    def run_control(self, operation, response=None, after_stopped=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        stored = {}

        def register(kind, value, **kwargs):
            stored["evidence"] = value
            return dict(kind=kind, id="1", env_epoch=3)

        ctx = NS(
            ops=object(),
            resolve=lambda value: value,
            env_epoch=3,
            clock=lambda: 1,
            artifact_dir=Path(temp.name),
            register_resource=register,
        )
        before = {
            "engines": [
                dict(
                    name="p0",
                    role="prefill",
                    grpc_addr="127.1.0.1:1",
                    stopped=operation == "start",
                )
            ]
        }
        after = {
            "engines": [
                dict(
                    name="p0",
                    role="prefill",
                    grpc_addr="127.1.0.1:1",
                    stopped=(
                        (operation == "stop")
                        if after_stopped is None
                        else after_stopped
                    ),
                )
            ]
        }
        response = (
            response if response is not None else dict(status="ok", engine="p0", port=1)
        )
        params = dict(operation=operation, targets=["p0"])
        if operation == "set_perf":
            params["perf"] = {"prefill_fixed_ms": 100}
        return ctx, stored, params, [before, response, after]

    def test_stop_and_start_require_observed_state(self):
        for operation in ("stop", "start"):
            ctx, stored, params, responses = self.run_control(operation)
            with patch.object(ec, "_http", side_effect=responses):
                output = ec.execute(ctx, params, self.deadline())
            self.assertTrue(stored["evidence"]["complete"])
            self.assertTrue(stored["evidence"]["effect_verified"])
            self.assertEqual(
                stored["evidence"]["after"]["p0"]["stopped"], operation == "stop"
            )
            self.assertTrue(Path(output.artifacts[0]).exists())

    def test_perf_ack_does_not_claim_measured_latency(self):
        ctx, stored, params, responses = self.run_control("set_perf")
        with patch.object(ec, "_http", side_effect=responses):
            ec.execute(ctx, params, self.deadline())
        self.assertFalse(stored["evidence"]["effect_verified"])
        self.assertEqual(
            stored["evidence"]["requested_perf"], {"prefill_fixed_ms": 100}
        )

    def test_wrong_target_or_false_success_is_error_and_persisted(self):
        for response in (
            {"status": "error", "engine": "p0", "port": 1},
            {"status": "ok", "engine": "other", "port": 1},
        ):
            ctx, stored, params, responses = self.run_control("stop", response=response)
            with patch.object(ec, "_http", side_effect=responses), self.assertRaises(
                ValueError
            ):
                ec.execute(ctx, params, self.deadline())
            evidence = json.loads(next(ctx.artifact_dir.glob("*.json")).read_text())
            self.assertFalse(evidence["complete"])
            self.assertIn("error", evidence)
            self.assertNotIn("evidence", stored)

    def test_ack_without_effect_is_error(self):
        ctx, _, params, responses = self.run_control("stop", after_stopped=False)
        with patch.object(ec, "_http", side_effect=responses), self.assertRaisesRegex(
            ValueError, "without matching"
        ):
            ec.execute(ctx, params, self.deadline())

    def test_missing_engine_blocks_all_mutations(self):
        ctx, _, params, _ = self.run_control("stop")
        with patch.object(
            ec, "_http", return_value={"engines": []}
        ) as http, self.assertRaises(ValueError):
            ec.execute(ctx, params, self.deadline())
        self.assertEqual(http.call_count, 1)

    def test_resolves_each_target_reference_before_control(self):
        ctx, stored, params, responses = self.run_control("stop")
        params["targets"] = [{"$ref": "stages.add.output.engine"}]
        ctx.resolve = lambda value: "p0" if isinstance(value, dict) else value
        with patch.object(ec, "_http", side_effect=responses):
            ec.execute(ctx, params, self.deadline())
        self.assertEqual(stored["evidence"]["targets"], ["p0"])

    def test_real_http_receives_remaining_deadline(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"status":"ok"}'
        with patch.object(
            ec.urllib.request, "urlopen", return_value=response
        ) as open_url:
            ec._http(
                NS(mock_http_port=1), "set_perf", self.deadline(), {"engine": "p0"}
            )
        self.assertEqual(open_url.call_args.kwargs["timeout"], 0.25)


if __name__ == "__main__":
    unittest.main()
