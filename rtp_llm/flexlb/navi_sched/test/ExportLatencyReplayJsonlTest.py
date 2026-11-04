#!/usr/bin/env python3

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from ExportLatencyReplayJsonl import export_log


PARAMETERS = [float(index + 1) for index in range(19)]
ZEROS = [0.0] * 19


def config(generation, scale, parameter_scales=None, adaptive=False):
    value = {
        "latency_learning_rate": 0.011,
        "latency_online_learning_rate_scale": scale,
        "latency_current_sample_weight": 4,
        "latency_parameter_learning_rate_scales": parameter_scales or [],
        "latency_nonlinear_scale": 1500,
        "latency_zero_anchor_max_ms": 256,
        "residual_learning_rate": 0.8,
        "residual_update_error_floor_ms": 100,
        "residual_update_error_ratio": 0.1,
        "residual_idle_decay_start_ms": 120000,
        "residual_idle_decay_end_ms": 300000,
    }
    if adaptive:
        value["latency_adaptive_learning_rate"] = {"enabled": True}
    return (
        "[navi-scheduler-config-update] generation "
        f"[{generation}] hash [hash-{generation}] effectiveConfig "
        f"[{json.dumps(value, separators=(',', ':'))}]\n"
    )


def sample(sequence, batch_id, event_time_us):
    return (
        "[navi-latency-training-sample] "
        f"sequence={sequence} batch_id={batch_id} event_time_us={event_time_us} "
        "target_ms=120.5 base_prediction_before_ms=110.25 "
        "prediction_before_ms=115.75 batch_count=1 running_batch_size=2 "
        "running_hit_tokens=3 running_miss_tokens=4 "
        "running_miss_square_sum=16 running_hit_miss_sum=12 "
        "waiting_batch_size=5 waiting_hit_tokens=6 waiting_miss_tokens=7 "
        "waiting_miss_square_sum=49 waiting_hit_miss_sum=42 "
        "preceding_request_count=8 running_elapsed_time_ms=9.5 "
        "running_decode_request_count=2 running_decode_generated_tokens=47 "
        "running_decode_remaining_token_budget=53\n"
    )


def update(
    sequence,
    step_before,
    step_after,
    event_time_us,
    revision_before=None,
    revision_after=None,
):
    revision_before = sequence if revision_before is None else revision_before
    revision_after = revision_before + 1 if revision_after is None else revision_after
    arrays = {
        "parameters_before": PARAMETERS,
        "parameters_after": PARAMETERS,
        "adam_m1_before": ZEROS,
        "adam_m1_after": ZEROS,
        "adam_m2_before": ZEROS,
        "adam_m2_after": ZEROS,
        "residual_parameters_before": ZEROS,
        "residual_parameters_after": ZEROS,
    }
    encoded = " ".join(
        f"{name}=[{','.join(map(str, values))}]"
        for name, values in arrays.items()
    )
    return (
        "[navi-latency-training-update] "
        f"sequence={sequence} revision_before={revision_before} "
        f"revision_after={revision_after} optimizer_step_before={step_before} "
        f"optimizer_step_after={step_after} {encoded} "
        f"residual_last_update_time_us_before={event_time_us - 1} "
        f"residual_last_update_time_us_after={event_time_us}\n"
    )


class ExportLatencyReplayJsonlTest(unittest.TestCase):
    def write_log(self, path, adaptive=False):
        rates = [1.0] * 19
        rates[17] = 0.01
        lines = [
            config(1, 16),
            sample(1, 101, 1001),
            "[latency-history-append] sequence=7 bucket=6 "
            "local_batch_id=101 event_time_us=1001\n",
            "[latency-history-fit] mode=online_batch current_sequence=7 "
            "sequence_ids=[7]\n",
            update(1, 1, 2, 1001),
            config(2, 32, rates, adaptive),
            sample(2, 102, 1002),
            "[latency-history-append] sequence=8 bucket=6 "
            "local_batch_id=102 event_time_us=1002\n",
            "[latency-history-fit] mode=online_batch current_sequence=8 "
            "sequence_ids=[7,8]\n",
            update(2, 2, 3, 1002),
        ]
        with gzip.open(path, "wt") as stream:
            stream.writelines(lines)

    def test_exports_history_batches_and_per_update_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "agent.log.gz"
            output = root / "agent.jsonl"
            self.write_log(source)
            export_log(source, output, require_running_elapsed=True)
            events = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(["sample", "update", "sample", "update"], [
            event["type"] for event in events
        ])
        self.assertEqual(7, events[0]["history_sequence"])
        self.assertEqual([2, 47, 53], events[0]["features"][-3:])
        self.assertEqual([[7]], events[1]["fit_batches"])
        self.assertTrue(events[1]["exact_replayable"])
        self.assertEqual(1, events[1]["revision_before"])
        self.assertEqual(2, events[1]["revision_after"])
        self.assertEqual(1, events[1]["config_generation"])
        self.assertEqual(16.0, events[1]["base_learning_rate_scale"])
        self.assertEqual([[7, 8]], events[3]["fit_batches"])
        self.assertEqual(2, events[3]["config_generation"])
        self.assertEqual(32.0, events[3]["base_learning_rate_scale"])
        self.assertEqual(0.01, events[3]["parameter_learning_rate_scales"][17])
        self.assertEqual(120000000, events[3]["residual_idle_decay_start_us"])

    def test_marks_cold_start_revision_jump_non_replayable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "agent.log.gz"
            with gzip.open(source, "wt") as stream:
                stream.writelines([
                    config(1, 1),
                    sample(1, 101, 1001),
                    "[latency-history-append] sequence=7 bucket=6 "
                    "local_batch_id=101 event_time_us=1001\n",
                    "[latency-history-fit] mode=online_batch current_sequence=7 "
                    "sequence_ids=[7]\n",
                    update(1, 1, 2, 1001, 1, 256),
                ])
            output = root / "agent.jsonl"
            export_log(source, output)
            events = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertFalse(events[1]["exact_replayable"])
        self.assertEqual(1, events[1]["revision_before"])
        self.assertEqual(256, events[1]["revision_after"])

    def test_rejects_adaptive_config_without_effective_update_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "agent.log.gz"
            self.write_log(source, adaptive=True)
            with self.assertRaisesRegex(ValueError, "cannot reconstruct adaptive"):
                export_log(source, root / "agent.jsonl")

    def test_rejects_update_before_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "agent.log.gz"
            with gzip.open(source, "wt") as stream:
                stream.write(sample(1, 101, 1001))
                stream.write(update(1, 1, 2, 1001))
            with self.assertRaisesRegex(ValueError, "precedes scheduler config"):
                export_log(source, root / "agent.jsonl")


if __name__ == "__main__":
    unittest.main()
