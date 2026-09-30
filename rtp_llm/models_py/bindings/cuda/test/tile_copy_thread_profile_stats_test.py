import copy
import unittest

from run_tile_copy_thread_profile import (
    BLOCKS,
    LAYOUTS,
    MATRIX,
    THREADS,
    summarize,
    validate,
)


class ThreadProfileStatsTest(unittest.TestCase):
    def fixture(self, repeat=80):
        settings = {"seed": 10, "iterations": 2, "warmup": 1, "correctness_only": False}
        records = [
            {
                "type": "metadata",
                "implementation": "tile_copy_thread_profile_v1",
                "seed": 10 + repeat,
                "repeat": repeat,
                "iterations": 2,
                "warmup": 1,
                "correctness_only": False,
                "threads": list(THREADS),
                "blocks": list(BLOCKS),
                "evict_multiplier": 8,
                "timing": "cuda_event_stream_latency",
                "regime": "cold",
                "seq_size_per_block": 128,
                "cp_size": 8,
                "tp_size": 8,
                "cp_mode": "CP_RR",
                "profile_capture": False,
                "l2_bytes": 1024,
                "production_default_threads": 256,
                "measured_copy_launches_per_round": 6,
                "scatter_setup_copy_launches_per_round": 1,
            }
        ]
        for layout, (payload, tiles) in LAYOUTS.items():
            records.append(
                {
                    "type": "layout",
                    "layout": layout,
                    "payload_bytes": payload,
                    "tiles": tiles,
                    "crc_staging_stride": 878176,
                    "source_bytes": 8192,
                    "evict_bytes": 8192,
                }
            )
        for layout, packing, direction, blocks, threads in sorted(MATRIX):
            case = dict(
                layout=layout,
                packing=packing,
                direction=direction,
                blocks=blocks,
                threads=threads,
            )
            records.append(
                dict(
                    case,
                    type="correctness",
                    success=True,
                    full_payload_and_guards=True,
                    rotations_checked=[0, 1],
                )
            )
            for round in range(2):
                records.append(
                    dict(
                        case,
                        type="sample",
                        repeat=repeat,
                        round=round,
                        source_plan=round,
                        position=THREADS.index(threads),
                        us=threads / 256 * (round + 1),
                    )
                )
        records.append({"type": "complete", "success": True})
        return records, settings

    def test_complete_paired_matrix_and_summary(self):
        samples = []
        for repeat in (80, 81):
            records, settings = self.fixture(repeat)
            _, rows = validate(records, settings, repeat)
            samples.extend(rows)
        summary = summarize(samples)
        self.assertEqual(len(summary), len(MATRIX))
        for row in summary:
            self.assertEqual(row["samples"], 4)
            self.assertEqual(row["median_us"], row["threads"] / 256 * 1.5)
            self.assertEqual(row["p95_us"], row["threads"] / 256 * 2)
            self.assertEqual(row["paired_ratio_to_256"], row["threads"] / 256)

    def test_reject_incomplete_duplicate_and_unpaired_samples(self):
        baseline, settings = self.fixture()
        sample_indices = [
            i for i, row in enumerate(baseline) if row["type"] == "sample"
        ]
        mutations = (
            lambda rows: rows.pop(),
            lambda rows: rows.pop(sample_indices[0]),
            lambda rows: rows.__setitem__(
                sample_indices[1], copy.deepcopy(rows[sample_indices[0]])
            ),
            lambda rows: rows[sample_indices[0]].update(source_plan=123),
            lambda rows: rows[sample_indices[0]].update(position=9),
            lambda rows: rows[sample_indices[0]].update(us=float("nan")),
            lambda rows: rows[sample_indices[0]].update(us=-1),
        )
        for mutation in mutations:
            records = copy.deepcopy(baseline)
            mutation(records)
            with self.assertRaises(ValueError):
                validate(records, settings, 80)

    def test_reject_wrong_shape_eviction_or_guard_coverage(self):
        baseline, settings = self.fixture()
        first_check = next(
            i for i, row in enumerate(baseline) if row["type"] == "correctness"
        )
        mutations = (
            lambda rows: rows[1].update(crc_staging_stride=732688),
            lambda rows: rows[1].update(payload_bytes=42),
            lambda rows: rows[1].update(evict_bytes=4096),
            lambda rows: rows[first_check].update(full_payload_and_guards=False),
            lambda rows: rows[first_check].update(rotations_checked=[0]),
            lambda rows: rows[0].update(timing="kernel"),
        )
        for mutation in mutations:
            records = copy.deepcopy(baseline)
            mutation(records)
            with self.assertRaises(ValueError):
                validate(records, settings, 80)


if __name__ == "__main__":
    unittest.main()
