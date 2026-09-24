"""CPU-only regression tests for rejecting incomplete or misleading results."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from crc_copy_benchmark_stats import (
    BLOCKS,
    DIRECTIONS,
    TILES,
    VARIANTS,
    analyze,
    read_run,
)


def fixture(exclude=False, correctness_only=False, repeat=80):
    settings = dict(
        iterations=4,
        warmup=1,
        seed=20260924,
        exclude_1d_h2d=exclude,
        correctness_only=correctness_only,
    )
    meta = dict(
        type="metadata",
        implementation="crc_copy_benchmark_v1",
        variants=list(VARIANTS),
        block_counts=list(BLOCKS),
        repeat=repeat,
        seed=settings["seed"] + repeat,
        iterations=4,
        warmup=1,
        correctness_only=correctness_only,
        exclude_1d_h2d=exclude,
        evict_multiplier=8,
        timing="wall",
        regime="cold",
        cpu_metadata="warm",
        layout_order="production",
        local_backing_count=True,
        seq_size_per_block=128,
        cp_size=8,
        tp_size=8,
        cp_mode="CP_RR",
        gen_num_per_cycle=0,
        fallback_allowed=False,
        l2_bytes=1024,
        driver=13000,
        runtime=13020,
        gpu="fixture",
        gpu_uuid="fixture-uuid",
        sm=103,
    )
    for field in (
        "boundary",
        "production_source",
        "gather_control",
        "copy3d_stream",
        "copy3d_submit_mutex",
        "host_input",
        "h2d_host_payload",
    ):
        meta[field] = "fixture contract"
    if exclude:
        meta["excluded_cases"] = [
            dict(direction="h2d", variant="copy1d_batch", reason="explicit option")
        ]
    records = [
        meta,
        dict(type="mixed_crc_selftest", variant="integrated_crc", success=True),
    ]
    for layout, tiles in TILES.items():
        payload = sum(tiles)
        encoded = (payload + 19) // 16 * 16
        records.append(
            dict(
                type="layout",
                layout=layout,
                tile_bytes=tiles,
                tiles=len(tiles),
                payload_bytes=payload,
                encoded_bytes=encoded,
                host_stride=(encoded + 4095) // 4096 * 4096,
                staging_stride=878176,
                host_pinned_verified=True,
                source_device_verified=True,
                copy3d_operations_per_backing=3,
                pool_blocks=128,
                source_bytes=payload * 128,
                destination_bytes=payload * 128,
                evict_bytes=8192,
            )
        )
        for direction in DIRECTIONS:
            active = [
                v
                for v in VARIANTS
                if not (exclude and direction == "h2d" and v == "copy1d_batch")
            ]
            for n in BLOCKS:
                for variant in VARIANTS:
                    record = dict(
                        direction=direction, layout=layout, blocks=n, variant=variant
                    )
                    if variant in active:
                        record.update(
                            type="correctness", rotations_checked=[0, 1], success=True
                        )
                    else:
                        record.update(
                            type="excluded_correctness", reason="explicit option"
                        )
                    records.append(record)
                if not correctness_only:
                    for round_id in range(4):
                        for position, variant in enumerate(active):
                            records.append(
                                dict(
                                    type="sample",
                                    direction=direction,
                                    layout=layout,
                                    blocks=n,
                                    variant=variant,
                                    repeat=repeat,
                                    round=round_id,
                                    position=position,
                                    source_plan=round_id % 2,
                                    timing="wall",
                                    regime="cold",
                                    us=100.0 + n + 10 * position + round_id,
                                )
                            )
        for n in (1, 8, 16, 32):
            records.append(
                dict(
                    type="crc_corruption",
                    layout=layout,
                    blocks=n,
                    variant="integrated_crc",
                    direction="h2d",
                    cases=6,
                    corruption_cases=6,
                    success=True,
                    all_targets_unchanged=True,
                    recovery_success=True,
                    good_bad_good=True,
                    entire_pool_guards=True,
                )
            )
    records.append(dict(type="complete", success=True))
    return settings, records


class BenchmarkStatsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def write(self, records, name="raw.jsonl"):
        path = Path(self.temp.name) / name
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return path

    def test_full_and_explicit_partial_matrices(self):
        for exclude, expected in ((False, 5120), (True, 4608)):
            with self.subTest(exclude=exclude):
                settings, first = fixture(exclude=exclude)
                _, second = fixture(exclude=exclude, repeat=81)
                result = analyze(
                    [self.write(first, "80.jsonl"), self.write(second, "81.jsonl")],
                    settings,
                )
                self.assertEqual(result["sample_count"], expected)
                self.assertEqual(result["paired_round_count"], 1024)
                self.assertEqual(len(result["cases"]), 128)
                cell = result["cases"][0]["variants"]["integrated_crc"]
                self.assertEqual(cell["count"], 8)
                self.assertEqual(cell["p50_us"], 102.5)
                self.assertEqual(cell["p95_us"], 104.0)

    def test_correctness_only_contains_no_timing(self):
        settings, first = fixture(correctness_only=True)
        _, second = fixture(correctness_only=True, repeat=81)
        result = analyze(
            [self.write(first, "80.jsonl"), self.write(second, "81.jsonl")], settings
        )
        self.assertEqual(result["sample_count"], 0)
        self.assertEqual(result["cases"], [])

    def test_reject_invalid_or_incomplete_evidence(self):
        settings, original = fixture()
        sample = next(i for i, r in enumerate(original) if r["type"] == "sample")
        shape = next(i for i, r in enumerate(original) if r["type"] == "layout")
        corruption = next(
            i for i, r in enumerate(original) if r["type"] == "crc_corruption"
        )
        mutations = {
            "mixed_integer_success": lambda r: r[1].update(success=1),
            "corruption_boolean_blocks": lambda r: r[corruption].update(blocks=True),
            "incomplete": lambda r: r.pop(),
            "integer_success": lambda r: r[-1].update(success=1),
            "boolean_blocks": lambda r: r[sample].update(blocks=True),
            "float_blocks": lambda r: r[sample].update(blocks=1.0),
            "missing_boundary": lambda r: r[0].pop("boundary"),
            "missing_sample": lambda r: r.pop(sample),
            "duplicate_variant": lambda r: r.__setitem__(
                sample + 1, copy.deepcopy(r[sample])
            ),
            "different_source": lambda r: r[sample + 1].update(source_plan=1),
            "duplicate_position": lambda r: r[sample + 1].update(position=0),
            "nonfinite": lambda r: r[sample].update(us=float("nan")),
            "zero_duration": lambda r: r[sample].update(us=0),
            "wrong_shape": lambda r: r[shape].update(payload_bytes=732673),
            "insufficient_eviction": lambda r: r[shape].update(evict_bytes=8191),
            "corruption_wrote_target": lambda r: r[corruption].update(
                all_targets_unchanged=False
            ),
            "silent_exclusion": lambda r: r[0].update(exclude_1d_h2d=True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                records = copy.deepcopy(original)
                mutate(records)
                with self.assertRaises(ValueError):
                    read_run(self.write(records), settings, 80)

    def test_exclusion_cannot_claim_success_or_accept_samples(self):
        settings, original = fixture(exclude=True)
        excluded = next(
            i for i, r in enumerate(original) if r["type"] == "excluded_correctness"
        )
        sample = next(
            i
            for i, r in enumerate(original)
            if r["type"] == "sample" and r["direction"] == "h2d"
        )
        for mutate in (
            lambda r: r[excluded].update(success=True),
            lambda r: r[sample].update(variant="copy1d_batch"),
            lambda r: r.pop(excluded),
        ):
            records = copy.deepcopy(original)
            mutate(records)
            with self.assertRaises(ValueError):
                read_run(self.write(records), settings, 80)

    def test_repeats_must_use_same_gpu(self):
        settings, first = fixture()
        _, second = fixture(repeat=81)
        second[0]["gpu_uuid"] = "other-gpu"
        with self.assertRaisesRegex(ValueError, "gpu_uuid"):
            analyze(
                [self.write(first, "80.jsonl"), self.write(second, "81.jsonl")],
                settings,
            )


if __name__ == "__main__":
    unittest.main()
