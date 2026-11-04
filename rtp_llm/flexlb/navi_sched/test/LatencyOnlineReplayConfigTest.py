#!/usr/bin/env python3

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


EFFECTIVE_PARAMETERS = [
    -32.13333333333333,
    0.3333333333333333,
    0.04666666666666667,
    0.6666666666666666,
    0.0033333333333333335,
    0.003,
    0.0,
    310.17292760150764,
    0.0044444444444444444,
    0.0,
    0.3333333333333333,
    0.04666666666666667,
    0.6666666666666666,
    0.0033333333333333335,
    0.003,
    5.0,
    0.0,
    0.0,
    0.0,
]
COORDINATE_SCALES = [
    0.878862443703,
    0.119054644545,
    0.0881092493705,
    2.45290259606,
    0.918359382716,
    19.5646708383,
    1.0,
    1.23972355201,
    0.00229934153041,
    0.211011741932,
    1.09757235126,
    0.878792874576,
    193.929503215,
    222.305482002,
    207.484099949,
    0.0242867645882,
    1892.09311928,
    0.0299401936353,
    0.403880546944,
]
PARAMETERS = [
    value / scale
    for value, scale in zip(EFFECTIVE_PARAMETERS, COORDINATE_SCALES)
]
ZEROS = [0.0] * 19


def sample(sequence, target_ms, history_sequence=None):
    event = {
        "type": "sample",
        "sequence": sequence,
        "event_time_us": 1_000_000 + sequence * 1000,
        "target_ms": target_ms,
        "features": [
            1,
            512,
            0,
            65536,
            8_388_608,
            0,
            16,
            0,
            2048,
            262_144,
            0,
            32,
            1000.0,
            500,
            32000,
            100000,
        ],
        "base_prediction_before_ms": 1000.0,
        "prediction_before_ms": 1000.0,
    }
    if history_sequence is not None:
        event["history_sequence"] = history_sequence
    return event


def update(parameter_rate=1.0, current_sample_weight=4.0):
    rates = [1.0] * 19
    rates[17] = parameter_rate
    return {
        "type": "update",
        "sequence": 2,
        "revision_before": 1,
        "revision_after": 2,
        "exact_replayable": True,
        "step_before": 1,
        "step_after": 2,
        "parameters_before": PARAMETERS.copy(),
        "parameters_after": PARAMETERS.copy(),
        "adam_m1_before": ZEROS.copy(),
        "adam_m2_before": ZEROS.copy(),
        "adam_m1_after": ZEROS.copy(),
        "adam_m2_after": ZEROS.copy(),
        "residual_parameters_before": ZEROS.copy(),
        "residual_parameters_after": ZEROS.copy(),
        "residual_time_before": 0,
        "residual_time_after": 1_002_000,
        "fit_batches": [[1, 2]],
        "base_learning_rate": 0.05,
        "base_learning_rate_scale": 1.0,
        "current_sample_weight": current_sample_weight,
        "parameter_learning_rate_scales": rates,
        "nonlinear_scale": 1500.0,
        "zero_anchor_max_ms": 256.0,
        "residual_learning_rate": 0.8,
        "residual_update_error_floor_ms": 100.0,
        "residual_update_error_ratio": 0.1,
        "residual_idle_decay_start_us": 120_000_000,
        "residual_idle_decay_end_us": 300_000_000,
    }


class LatencyOnlineReplayConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runfiles = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        cls.binary = runfiles / "rtp_llm/flexlb/navi_sched/latency_online_replay"

    def run_replay(
        self,
        root,
        event,
        cli_weight=4,
        cli_scale=16,
        cli_residual=0.8,
        feature_scales="all",
        trace_model_state=False,
        mode="fixed",
        secondary_mode=None,
        residual_error_limit=None,
        next_update=None,
        check=True,
    ):
        input_dir = root / "input"
        input_dir.mkdir()
        events = [
            sample(1, 3000.0, 1),
            sample(2, 9000.0, 2),
            event,
            sample(3, 5000.0, 3 if next_update is not None else None),
        ]
        if next_update is not None:
            events.append(next_update)
        (input_dir / "host.jsonl").write_text(
            "".join(json.dumps(value, separators=(",", ":")) + "\n"
                    for value in events),
            encoding="utf-8",
        )
        output = root / "output.tsv"
        command = [
            str(self.binary),
            str(input_dir),
            str(output),
            str(cli_weight),
            "0",
            str(cli_scale),
            str(cli_residual),
            feature_scales,
            "1" if trace_model_state else "0",
            mode,
        ]
        if secondary_mode is not None:
            command.append(secondary_mode)
        if residual_error_limit is not None:
            if secondary_mode is None:
                raise ValueError("residual_error_limit requires secondary_mode")
            command.append("residual_error_limit:100," + str(residual_error_limit))
        process = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if check and process.returncode != 0:
            raise AssertionError(process.stderr)
        return process, output.read_text(encoding="utf-8") if output.exists() else ""

    @staticmethod
    def trajectory(output):
        rows = []
        for line in output.splitlines():
            fields = line.split("\t")
            rows.append(fields[:2] + fields[3:])
        return rows

    @staticmethod
    def update_state_from_trace(event, output):
        model_state = next(
            line.split("\t") for line in output.splitlines()
            if line.startswith("M\t")
        )
        parameter_count = int(model_state[24])
        parameter_begin = 25
        parameter_end = parameter_begin + parameter_count
        residual_count = int(model_state[parameter_end])
        residual_begin = parameter_end + 1
        residual_end = residual_begin + residual_count

        adam_trace = next(
            line.split("\t") for line in output.splitlines()
            if line.startswith("D\t")
        )
        trace_parameter_count = int(adam_trace[13])
        if trace_parameter_count != parameter_count:
            raise AssertionError("model and Adam trace parameter counts differ")
        moment1 = []
        moment2 = []
        for index in range(parameter_count):
            field = 14 + index * 15
            moment1.append(float(adam_trace[field + 3]))
            moment2.append(float(adam_trace[field + 6]))

        result = deepcopy(event)
        result["parameters_after"] = [
            float(value) for value in model_state[parameter_begin:parameter_end]
        ]
        result["residual_parameters_after"] = [
            float(value) for value in model_state[residual_begin:residual_end]
        ]
        result["adam_m1_after"] = moment1
        result["adam_m2_after"] = moment2
        return result

    def materialize_online_update(self, root):
        event = update()
        _, output = self.run_replay(
            root, event, trace_model_state=True
        )
        return self.update_state_from_trace(event, output)

    def test_removed_rate_modes_are_rejected(self):
        for mode in ("production", "adaptive", "simple_lag:0.1,0,1,0,1,12",
                     "error_rate:1,64,128,0,1,1",
                     "production_error_rate_hinge:0.5,1,0.5,0.03,0,1000,1"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                process, output = self.run_replay(
                    Path(directory), update(), mode=mode, check=False)
                self.assertNotEqual(0, process.returncode)
                self.assertIn("unsupported replay mode: " + mode, process.stderr)
                self.assertEqual("", output)

    def test_logged_effective_config_overrides_command_line_defaults(self):
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            _, left = self.run_replay(
                Path(left_dir), update(), cli_weight=1, cli_scale=0.25,
                cli_residual=0.03,
            )
            _, right = self.run_replay(
                Path(right_dir), update(), cli_weight=64, cli_scale=512,
                cli_residual=1.0,
            )
        self.assertEqual(self.trajectory(left), self.trajectory(right))
        self.assertIn("P\thost\t", left)

    def test_counterfactual_fixed_uses_command_line_rates(self):
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            _, left = self.run_replay(
                Path(left_dir), update(), cli_weight=1, cli_scale=0.25,
                cli_residual=0.03, mode="counterfactual_fixed",
            )
            _, right = self.run_replay(
                Path(right_dir), update(), cli_weight=64, cli_scale=512,
                cli_residual=1.0, mode="counterfactual_fixed",
            )
        self.assertNotEqual(self.trajectory(left), self.trajectory(right))

    def test_counterfactual_anchor_until_reanchors_without_controller(self):
        first = update()
        second = update()
        second["sequence"] = 3
        second["revision_before"] = 2
        second["revision_after"] = 3
        second["step_before"] = 2
        second["step_after"] = 3
        second["fit_batches"] = [[3]]
        with tempfile.TemporaryDirectory() as anchored_dir, tempfile.TemporaryDirectory() as diagnostic_dir, tempfile.TemporaryDirectory() as fixed_dir:
            _, anchored = self.run_replay(
                Path(anchored_dir), first, next_update=second,
                cli_weight=4, cli_scale=1, cli_residual=0,
                mode="counterfactual_anchor_until:1003500",
            )
            _, diagnostic = self.run_replay(
                Path(diagnostic_dir), first, next_update=second,
                cli_weight=4, cli_scale=1, cli_residual=0,
                mode="fixed_reanchor_diagnostic",
            )
            _, fixed = self.run_replay(
                Path(fixed_dir), first, next_update=second,
                cli_weight=4, cli_scale=1, cli_residual=0,
                mode="counterfactual_fixed",
            )
        self.assertEqual(self.trajectory(diagnostic), self.trajectory(anchored))
        self.assertNotEqual(self.trajectory(fixed), self.trajectory(anchored))

    def test_logged_reanchor_uses_logged_rates_before_reanchoring(self):
        event = update(current_sample_weight=40.0)
        event["base_learning_rate_scale"] = 2.55
        with tempfile.TemporaryDirectory() as logged_dir, tempfile.TemporaryDirectory() as diagnostic_dir, tempfile.TemporaryDirectory() as candidate_dir:
            _, logged = self.run_replay(
                Path(logged_dir), event, cli_weight=4, cli_scale=0.1,
                mode="fixed",
            )
            _, diagnostic = self.run_replay(
                Path(diagnostic_dir), event, cli_weight=4, cli_scale=0.1,
                mode="logged_reanchor_diagnostic",
            )
            _, candidate = self.run_replay(
                Path(candidate_dir), event, cli_weight=4, cli_scale=0.1,
                mode="fixed_reanchor_diagnostic",
            )
        self.assertEqual(self.trajectory(logged), self.trajectory(diagnostic))
        self.assertNotEqual(self.trajectory(candidate), self.trajectory(diagnostic))

    def test_p14_floor_projects_every_replay_fit_step(self):
        with tempfile.TemporaryDirectory() as baseline_dir, tempfile.TemporaryDirectory() as floor_dir:
            kwargs = {
                "cli_scale": 0.01,
                "cli_residual": 0,
                "trace_model_state": True,
                "mode": "counterfactual_fixed",
            }
            _, baseline = self.run_replay(Path(baseline_dir), update(), **kwargs)
            _, floored = self.run_replay(
                Path(floor_dir), update(), secondary_mode="p14_floor:0.2", **kwargs
            )
        baseline_steps = [line.split("\t") for line in baseline.splitlines()
                          if line.startswith("M\t")]
        floor_steps = [line.split("\t") for line in floored.splitlines()
                       if line.startswith("M\t")]
        self.assertEqual(1, len(baseline_steps))
        self.assertEqual(1, len(floor_steps))
        self.assertLess(float(baseline_steps[0][39]), 0.2)
        self.assertEqual(0.2, float(floor_steps[0][39]))

    def test_p14_floor_rejects_invalid_values(self):
        for value in ("-0.1", "nan", "0.2junk"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                process, _ = self.run_replay(
                    Path(directory), update(), mode="counterfactual_fixed",
                    secondary_mode="p14_floor:" + value, check=False,
                )
                self.assertNotEqual(0, process.returncode)
                self.assertIn("p14_floor", process.stderr)

    def test_default_mode_uses_command_line_rates(self):
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir:
            _, left = self.run_replay(
                Path(left_dir), update(), cli_weight=1, cli_scale=0.25,
                mode="",
            )
            _, right = self.run_replay(
                Path(right_dir), update(), cli_weight=64, cli_scale=512,
                mode="",
            )
        self.assertNotEqual(self.trajectory(left), self.trajectory(right))

    def test_counterfactual_fixed_uses_candidate_rates_with_logged_config(self):
        with tempfile.TemporaryDirectory() as left_dir, tempfile.TemporaryDirectory() as right_dir, tempfile.TemporaryDirectory() as weight_dir:
            _, left = self.run_replay(
                Path(left_dir), update(), cli_weight=4, cli_scale=0.5, cli_residual=0,
                mode="counterfactual_fixed",
            )
            _, right = self.run_replay(
                Path(right_dir), update(), cli_weight=4, cli_scale=11, cli_residual=0,
                mode="counterfactual_fixed",
            )
            _, heavier = self.run_replay(
                Path(weight_dir), update(), cli_weight=64, cli_scale=0.5, cli_residual=0,
                mode="counterfactual_fixed",
            )
        self.assertNotEqual(self.trajectory(left), self.trajectory(right))
        self.assertNotEqual(self.trajectory(left), self.trajectory(heavier))

    def test_counterfactual_fixed_ignores_invalid_logged_rates(self):
        event = update(current_sample_weight=0.0)
        event["base_learning_rate_scale"] = 0.0
        event["residual_learning_rate"] = 2.0
        with tempfile.TemporaryDirectory() as directory:
            process, output = self.run_replay(
                Path(directory), event, cli_weight=4, cli_scale=2,
                cli_residual=0.8, mode="counterfactual_fixed", check=False,
            )
        self.assertEqual(0, process.returncode, process.stderr)
        self.assertIn("P\thost\t", output)

    def test_counterfactual_residual_zero_preserves_logged_anchor(self):
        event = update()
        event["residual_parameters_before"] = [0.0] + [100.0] * 17 + [0.0]
        event["residual_parameters_after"] = event["residual_parameters_before"]
        with tempfile.TemporaryDirectory() as root:
            process, output = self.run_replay(
                Path(root), event, cli_residual=0, mode="counterfactual_fixed",
                trace_model_state=True, check=False,
            )
        self.assertEqual(0, process.returncode, process.stderr)
        predictions = [line.split("\t") for line in output.splitlines()
                       if line.startswith("P\t")]
        self.assertEqual(1, len(predictions))
        self.assertGreaterEqual(len(predictions[0]), 11, predictions)
        self.assertEqual(float(predictions[0][8]), float(predictions[0][9]))
        self.assertEqual(0.0, float(predictions[0][10]))
        state = self.update_state_from_trace(event, output)
        self.assertEqual(event["residual_parameters_before"],
                         state["residual_parameters_after"])

    def test_fixed_logged_residual_zero_skips_stale_correction(self):
        event = update()
        event["fit_batches"] = []
        event["step_after"] = event["step_before"]
        event["residual_parameters_before"] = [0.0] + [100.0] * 17 + [0.0]
        event["residual_parameters_after"] = event["residual_parameters_before"].copy()
        event["residual_learning_rate"] = 0.0
        with tempfile.TemporaryDirectory() as root:
            process, output = self.run_replay(
                Path(root), event, mode="fixed",
                check=False,
            )
        self.assertEqual(0, process.returncode, process.stderr)
        prediction = next(line.split("\t") for line in output.splitlines()
                          if line.startswith("P\t"))
        self.assertEqual(float(prediction[8]), float(prediction[9]))
        self.assertEqual(0.0, float(prediction[10]))
        zero_point = next(line.split("\t") for line in output.splitlines()
                          if line.startswith("Z\t"))
        self.assertEqual(float(zero_point[6]), float(zero_point[7]))
        update_row = next(line.split("\t") for line in output.splitlines()
                          if line.startswith("U\t"))
        self.assertEqual(0.0, float(update_row[7]))

    def test_disabled_residual_reports_divergent_base_prediction(self):
        event = update()
        event["parameters_before"][7] = 10_000_000.0
        event["parameters_after"][7] = 10_000_000.0
        with tempfile.TemporaryDirectory() as directory:
            process, _ = self.run_replay(
                Path(directory), event, cli_scale=1, cli_residual=0,
                mode="counterfactual_fixed", check=False,
            )
        self.assertNotEqual(0, process.returncode)
        self.assertIn("base prediction out of range", process.stderr)
        self.assertNotIn("residual update failed", process.stderr)

    def test_disabled_residual_preserves_logged_state_with_fixed_rates(self):
        event = update()
        del event["residual_learning_rate"]
        event["residual_parameters_before"][3] = 30.0
        event["residual_parameters_after"][3] = 30.0
        event["residual_time_before"] = 900_000
        event["residual_time_after"] = event["residual_time_before"]
        with tempfile.TemporaryDirectory() as directory:
            process, output = self.run_replay(
                Path(directory), event, cli_scale=1, cli_residual=0,
                mode="counterfactual_fixed", trace_model_state=True, check=False,
            )
        self.assertEqual(0, process.returncode, process.stderr)
        state = self.update_state_from_trace(event, output)
        self.assertEqual(event["residual_parameters_before"],
                         state["residual_parameters_after"])
        update_row = next(line.split("\t") for line in output.splitlines()
                          if line.startswith("U\t"))
        self.assertEqual(0.0, float(update_row[7]))
        self.assertEqual(0.0, float(update_row[10]))

    def test_residual_error_limit_overrides_logged_ratio(self):
        with tempfile.TemporaryDirectory() as low_dir, tempfile.TemporaryDirectory() as high_dir:
            kwargs = {
                "cli_scale": 1,
                "cli_residual": 1,
                "trace_model_state": True,
                "mode": "counterfactual_fixed",
                "secondary_mode": "parameter_rates:17=0,18=0",
            }
            _, low = self.run_replay(
                Path(low_dir), update(), residual_error_limit=0.001, **kwargs
            )
            _, high = self.run_replay(
                Path(high_dir), update(), residual_error_limit=1.0, **kwargs
            )
        low_state = self.update_state_from_trace(update(), low)
        high_state = self.update_state_from_trace(update(), high)
        self.assertGreater(
            max(abs(left - right) for left, right in zip(
                low_state["residual_parameters_after"],
                high_state["residual_parameters_after"])),
            100.0,
        )

    def test_logged_p17_rate_changes_the_continuous_trajectory(self):
        with tempfile.TemporaryDirectory() as fast_dir, tempfile.TemporaryDirectory() as slow_dir:
            _, fast = self.run_replay(Path(fast_dir), update(parameter_rate=1.0))
            _, slow = self.run_replay(Path(slow_dir), update(parameter_rate=0.01))
        fast_prediction = next(line for line in fast.splitlines() if line.startswith("P\t"))
        slow_prediction = next(line for line in slow.splitlines() if line.startswith("P\t"))
        self.assertNotEqual(fast_prediction, slow_prediction)

    def test_invalid_logged_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            process, _ = self.run_replay(
                Path(directory), update(current_sample_weight=0.0), check=False
            )
        self.assertNotEqual(0, process.returncode)
        self.assertIn("invalid per-update runtime config", process.stderr)

    def test_zero_residual_learning_rate_replays(self):
        event = update()
        event["residual_learning_rate"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            _, output = self.run_replay(
                Path(directory), event, cli_residual=0.0
            )
        self.assertIn("P\thost\t", output)

    def test_logged_online_state_replays_exactly(self):
        with tempfile.TemporaryDirectory() as materialize_dir:
            event = self.materialize_online_update(Path(materialize_dir))
        self.assertTrue(any(value != 0.0 for value in event["adam_m1_after"]))
        self.assertTrue(any(value != 0.0 for value in event["adam_m2_after"]))
        self.assertTrue(
            any(value != 0.0 for value in event["residual_parameters_after"])
        )

        with tempfile.TemporaryDirectory() as verify_dir:
            _, output = self.run_replay(Path(verify_dir), event)
        update_result = next(
            line.split("\t") for line in output.splitlines()
            if line.startswith("U\t")
        )
        for difference in update_result[6:11]:
            self.assertLessEqual(float(difference), 1e-15)

    def test_logged_online_state_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as materialize_dir:
            event = self.materialize_online_update(Path(materialize_dir))
        event["parameters_after"][3] += 1.0

        with tempfile.TemporaryDirectory() as verify_dir:
            _, output = self.run_replay(Path(verify_dir), event)
        update_result = next(
            line.split("\t") for line in output.splitlines()
            if line.startswith("U\t")
        )
        self.assertGreater(float(update_result[6]), 0.999999999999)
        for difference in update_result[7:11]:
            self.assertLessEqual(float(difference), 1e-15)

    def test_cold_start_transition_restores_after_state_without_false_mismatch(self):
        with tempfile.TemporaryDirectory() as materialize_dir:
            event = self.materialize_online_update(Path(materialize_dir))
        event["exact_replayable"] = False
        event["revision_after"] = 256
        event["parameters_before"] = [value + 1000.0 for value in PARAMETERS]
        event["adam_m1_before"] = [1000.0] * 19
        event["adam_m2_before"] = [1000.0] * 19

        next_event = deepcopy(event)
        next_event.update({
            "sequence": 3,
            "revision_before": 256,
            "revision_after": 257,
            "exact_replayable": True,
            "step_before": event["step_after"],
            "step_after": event["step_after"] + 1,
            "parameters_before": event["parameters_after"],
            "parameters_after": event["parameters_after"],
            "adam_m1_before": event["adam_m1_after"],
            "adam_m1_after": event["adam_m1_after"],
            "adam_m2_before": event["adam_m2_after"],
            "adam_m2_after": event["adam_m2_after"],
            "residual_parameters_before": event["residual_parameters_after"],
            "residual_parameters_after": event["residual_parameters_after"],
            "residual_time_before": event["residual_time_after"],
            "residual_time_after": event["residual_time_after"],
            "fit_batches": [[3]],
        })
        with tempfile.TemporaryDirectory() as materialize_next_dir:
            _, materialized_output = self.run_replay(
                Path(materialize_next_dir), event, trace_model_state=True,
                next_update=next_event,
            )
        next_event = self.update_state_from_trace(next_event, materialized_output)
        next_event["residual_time_after"] = 1_003_000

        with tempfile.TemporaryDirectory() as verify_dir:
            _, output = self.run_replay(
                Path(verify_dir), event, next_update=next_event
            )
        self.assertIn("K\thost\t", output)
        self.assertIn("\tcold_start_state_transition\n", output)
        updates = [
            line.split("\t") for line in output.splitlines()
            if line.startswith("U\t")
        ]
        self.assertEqual(1, len(updates))
        for difference in updates[0][6:11]:
            self.assertLessEqual(float(difference), 1e-12)

    def test_remaining_decode_budget_can_replace_p18_for_ablation(self):
        without_budget = "0,0,0,0,0,0,0,0"
        budget_only = "0,0,0,0,0,0,0,0,1"
        with tempfile.TemporaryDirectory() as left_dir:
            with tempfile.TemporaryDirectory() as right_dir:
                _, left = self.run_replay(
                    Path(left_dir),
                    update(),
                    feature_scales=without_budget,
                    mode="parameter_rates:17=0,18=1",
                )
                _, right = self.run_replay(
                    Path(right_dir),
                    update(),
                    feature_scales=budget_only,
                    mode="parameter_rates:17=0,18=1",
                )
        left_prediction = next(
            line for line in left.splitlines() if line.startswith("P\t")
        )
        right_prediction = next(
            line for line in right.splitlines() if line.startswith("P\t")
        )
        self.assertNotEqual(left_prediction, right_prediction)

    def test_fixed_parameter_scales_can_be_calibrated_in_the_same_replay(self):
        parameter_rates = "parameter_rates:0=2,17=0,18=0"
        with tempfile.TemporaryDirectory() as directory:
            _, output = self.run_replay(
                Path(directory),
                update(),
                trace_model_state=True,
                mode=parameter_rates,
                secondary_mode="loss_feedback:1,0.01,0.01,100,1",
            )
        summary = next(
            line.split("\t") for line in output.splitlines()
            if line.startswith("C\t")
        )
        self.assertEqual(97, len(summary))
        self.assertGreater(float(summary[3]), 0.0)
        self.assertEqual("17", summary[87])
        self.assertEqual(0.0, float(summary[88]))
        self.assertEqual("18", summary[92])
        self.assertEqual(0.0, float(summary[93]))
        adam_trace = next(
            line.split("\t") for line in output.splitlines()
            if line.startswith("D\t")
        )
        self.assertEqual(0.0, float(adam_trace[14 + 17 * 15]))
        self.assertEqual(0.0, float(adam_trace[14 + 18 * 15]))


if __name__ == "__main__":
    unittest.main()
