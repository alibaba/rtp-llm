"""CPU checks for rejecting incomplete evidence and preserving paired latency."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from crc_segmentation_benchmark_stats import (
    BLOCKS,
    BOUNDARY,
    CHECK_BLOCKS,
    DIRECTIONS,
    PROTOCOL,
    SEGMENT_BYTES,
    TILES,
    VARIANTS,
    analyze,
    expected_geometry,
    read_run,
    write_results,
)


def fixture(correctness_only=False, repeat=80, iterations=4):
    settings = dict(
        iterations=iterations,
        warmup=1,
        seed=20260924,
        repeats=2,
        correctness_only=correctness_only,
    )
    meta = dict(
        type="metadata",
        protocol=PROTOCOL,
        variants=list(VARIANTS),
        repeat=repeat,
        seed=settings["seed"] + repeat,
        iterations=iterations,
        warmup=1,
        correctness_only=correctness_only,
        max_backings=32,
        timing="wall",
        regime="cold",
        measured_h2d_verified=True,
        shared_workspace=True,
        forced_mode_outside_timer=True,
        equal_workspace_bytes=True,
        source_commit="a" * 40,
        base_commit="b" * 40,
        binary_sha256="c" * 64,
        gpu_name="fixture GPU",
        gpu_uuid="fixture UUID",
        sm=103,
        runtime=13020,
        driver=13000,
        l2_bytes=1024,
        profiled=False,
        fallback_allowed=False,
        evict_multiplier=8,
        block_counts=list(BLOCKS),
        segment_bytes=SEGMENT_BYTES,
        boundary=BOUNDARY,
        model="DeepSeek-V4-Flash-0731",
        target_layers=43,
        kv_cache_sharded=True,
        seq_size_per_block=128,
        kernel_tokens_per_block=128,
        cp_size=4,
        tp_size=4,
        cp_mode="CP_RR",
        gen_num_per_cycle=0,
        local_backing_count=True,
        shape_source="CacheConfigCreator+DeviceBlockPoolConfigHelper",
    )
    records = [meta] + [
        dict(type="mixed_crc_selftest", variant=variant, success=True)
        for variant in VARIANTS
    ]
    stride = (max(map(sum, TILES.values())) + 19) // 16 * 16
    for layout, tiles in TILES.items():
        payload = sum(tiles)
        encoded = (payload + 19) // 16 * 16
        records.append(
            dict(
                type="layout",
                layout=layout,
                geometry=expected_geometry(layout),
                tiles=len(tiles),
                payload_bytes=payload,
                encoded_bytes=encoded,
                host_stride=encoded,
                staging_stride=stride,
                pool_blocks=129,
                source_bytes=129 * payload,
                evict_bytes=8192,
                workspace_device_bytes=32 * stride + 8192,
                reserved_sentinel_blocks=1,
            )
        )
        for direction in DIRECTIONS:
            for blocks in BLOCKS:
                for variant in VARIANTS:
                    messages = blocks * (
                        math.ceil(payload / SEGMENT_BYTES)
                        if variant == "crc_segmented"
                        else 1
                    )
                    records.append(
                        dict(
                            type="correctness",
                            layout=layout,
                            direction=direction,
                            blocks=blocks,
                            variant=variant,
                            rotations_checked=[0, 1],
                            crc_messages=messages,
                            success=True,
                        )
                    )
                if correctness_only:
                    continue
                for round_id in range(iterations):
                    for index, variant in enumerate(VARIANTS):
                        messages = blocks * (
                            math.ceil(payload / SEGMENT_BYTES)
                            if variant == "crc_segmented"
                            else 1
                        )
                        # Different round trends make paired quantiles testable;
                        # the second process shifts both paths by the same amount.
                        latency = (
                            (
                                100 + round_id
                                if variant == "crc_whole"
                                else 90 + 2 * round_id
                            )
                            + blocks
                            + 20 * (repeat - 80)
                        )
                        records.append(
                            dict(
                                type="sample",
                                layout=layout,
                                direction=direction,
                                blocks=blocks,
                                variant=variant,
                                repeat=repeat,
                                round=round_id,
                                source_plan=round_id % 2,
                                position=(round_id + index) % 2,
                                timing="wall",
                                regime="cold",
                                us=float(latency),
                                crc_messages=messages,
                            )
                        )
        for blocks in CHECK_BLOCKS:
            for variant in VARIANTS:
                records.append(
                    dict(
                        type="crc_corruption",
                        layout=layout,
                        direction="h2d",
                        blocks=blocks,
                        variant=variant,
                        cases=6,
                        success=True,
                        all_targets_unchanged=True,
                        recovery_success=True,
                        good_bad_good=True,
                        entire_pool_guards=True,
                    )
                )
            records.append(
                dict(
                    type="cross_mode",
                    layout=layout,
                    blocks=blocks,
                    whole_store_segmented_load=True,
                    segmented_store_whole_load=True,
                    footer_equal=True,
                    success=True,
                )
            )
    records.append(dict(type="complete", success=True))
    return settings, records


class SegmentationStatsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def write(self, records, name="raw.jsonl"):
        path = Path(self.temp.name) / name
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return path

    def pair(self, correctness_only=False, iterations=4):
        settings, first = fixture(
            correctness_only=correctness_only, iterations=iterations
        )
        _, second = fixture(
            correctness_only=correctness_only, repeat=81, iterations=iterations
        )
        return settings, [self.write(first, "80.jsonl"), self.write(second, "81.jsonl")]

    def test_complete_matrix_reports_absolute_and_paired_quantiles(self):
        settings, paths = self.pair()
        result = analyze(paths, settings)
        self.assertEqual(result["sample_count"], 2048)
        self.assertEqual(result["paired_round_count"], 1024)
        self.assertEqual(len(result["cases"]), 128)
        case = result["cases"][0]
        whole = case["variants"]["crc_whole"]
        self.assertEqual(whole["count"], 8)
        self.assertEqual(whole["p50_us"], 112.5)
        self.assertEqual(whole["p95_us"], 124.0)
        self.assertEqual(whole["by_repeat"][80]["p50_us"], 102.5)
        self.assertEqual(whole["by_repeat"][81]["p50_us"], 122.5)
        self.assertEqual(whole["by_position"][0]["count"], 4)
        delta = case["paired_whole_minus_segmented"]
        self.assertEqual(delta["p50_us"], 8.5)
        self.assertEqual(delta["p95_us"], 10.0)
        self.assertEqual(delta["by_repeat"][80]["p50_us"], 8.5)
        self.assertEqual(delta["by_repeat"][80]["by_whole_position"][0]["p50_us"], 9.0)
        self.assertEqual(delta["by_repeat"][80]["by_whole_position"][1]["p50_us"], 8.0)
        self.assertTrue(
            any(
                w["reason"] == "repeat p50 difference exceeds 10%"
                for w in result["warnings"]
            )
        )
        self.assertTrue(all(len(record["sha256"]) == 64 for record in result["inputs"]))

    def test_default_matrix_contains_51200_samples(self):
        settings, paths = self.pair(iterations=100)
        result = analyze(paths, settings)
        self.assertEqual(result["sample_count"], 51200)
        self.assertEqual(result["paired_round_count"], 25600)

    def test_correctness_only_retains_checks_without_latency(self):
        settings, paths = self.pair(correctness_only=True)
        result = analyze(paths, settings)
        self.assertEqual(result["sample_count"], 0)
        self.assertEqual(result["cases"], [])
        write_results(result, self.temp.name)
        summary = (Path(self.temp.name) / "summary.md").read_text()
        self.assertIn("no latency measurements", summary)
        self.assertNotIn("Whole p50", summary)

    def test_writer_keeps_per_repeat_and_paired_results(self):
        settings, paths = self.pair()
        result = analyze(paths, settings)
        raw_before = [path.read_bytes() for path in paths]
        write_results(result, self.temp.name)
        self.assertEqual(raw_before, [path.read_bytes() for path in paths])
        summary = (Path(self.temp.name) / "summary.md").read_text()
        self.assertIn("Whole p50 R80 / R81", summary)
        self.assertIn("102.500 / 122.500", summary)
        self.assertIn("whole minus segmented", summary)
        self.assertIn(
            "repeat_81_p95_us", (Path(self.temp.name) / "latency.csv").read_text()
        )
        self.assertTrue((Path(self.temp.name) / "paired_delta.csv").is_file())
        self.assertEqual(
            json.loads((Path(self.temp.name) / "analysis.json").read_text())[
                "sample_count"
            ],
            2048,
        )

    def test_reject_incomplete_unpaired_or_wrong_policy_evidence(self):
        settings, original = fixture()
        index = {
            kind: next(i for i, r in enumerate(original) if r["type"] == kind)
            for kind in (
                "sample",
                "layout",
                "correctness",
                "crc_corruption",
                "cross_mode",
            )
        }
        sample, shape = index["sample"], index["layout"]
        corruption, cross = index["crc_corruption"], index["cross_mode"]
        mutations = {
            "missing_complete": lambda r: r.pop(),
            "integer_complete": lambda r: r[-1].update(success=1),
            "mixed_crc_failed": lambda r: r[1].update(success=False),
            "mixed_crc_integer_success": lambda r: r[1].update(success=1),
            "duplicate_mixed_crc": lambda r: r.__setitem__(2, copy.deepcopy(r[1])),
            "missing_sample": lambda r: r.pop(sample),
            "duplicate_sample": lambda r: r.__setitem__(
                sample + 1, copy.deepcopy(r[sample])
            ),
            "different_source": lambda r: r[sample + 1].update(source_plan=1),
            "duplicate_position": lambda r: r[sample + 1].update(position=0),
            "wrong_repeat": lambda r: r[sample].update(repeat=81),
            "invalid_round": lambda r: r[sample].update(round=4),
            "invalid_source_plan": lambda r: r[sample].update(source_plan=128),
            "boolean_position": lambda r: r[sample].update(position=False),
            "boolean_blocks": lambda r: r[sample].update(blocks=True),
            "float_blocks": lambda r: r[sample].update(blocks=1.0),
            "unknown_variant": lambda r: r[sample].update(variant="automatic"),
            "nan_duration": lambda r: r[sample].update(us=float("nan")),
            "infinite_duration": lambda r: r[sample].update(us=float("inf")),
            "negative_duration": lambda r: r[sample].update(us=-1),
            "zero_duration": lambda r: r[sample].update(us=0),
            "boolean_duration": lambda r: r[sample].update(us=True),
            "event_timing": lambda r: r[sample].update(timing="cuda_event"),
            "warm_regime": lambda r: r[sample].update(regime="warm"),
            "wrong_policy": lambda r: r[sample + 1].update(crc_messages=1),
            "missing_policy_evidence": lambda r: r[sample].pop("crc_messages"),
            "correctness_wrong_policy": lambda r: r[index["correctness"]].update(
                crc_messages=2
            ),
            "missing_correctness": lambda r: r.pop(index["correctness"]),
            "integer_correctness": lambda r: r[index["correctness"]].update(success=1),
            "missing_rotation": lambda r: r[index["correctness"]].update(
                rotations_checked=[0]
            ),
            "corruption_wrote_target": lambda r: r[corruption].update(
                all_targets_unchanged=False
            ),
            "corruption_did_not_recover": lambda r: r[corruption].update(
                recovery_success=False
            ),
            "corruption_case_count": lambda r: r[corruption].update(cases=5),
            "missing_cross_mode": lambda r: r.pop(cross),
            "wrong_footer": lambda r: r[cross].update(footer_equal=False),
            "cross_mode_failed": lambda r: r[cross].update(
                segmented_store_whole_load=False
            ),
            "wrong_geometry": lambda r: r[shape]["geometry"][0].update(model_layer=0),
            "wrong_payload": lambda r: r[shape].update(payload_bytes=1),
            "wrong_source_bytes": lambda r: r[shape].update(source_bytes=1),
            "small_pool": lambda r: r[shape].update(pool_blocks=32),
            "small_eviction": lambda r: r[shape].update(evict_bytes=4096),
            "small_workspace": lambda r: r[shape].update(workspace_device_bytes=1),
            "wrong_protocol": lambda r: r[0].update(protocol="crc_copy_benchmark_v5"),
            "wrong_mode_list": lambda r: r[0].update(
                variants=["crc_whole", "automatic"]
            ),
            "mode_set_inside_timer": lambda r: r[0].update(
                forced_mode_outside_timer=False
            ),
            "missing_shared_workspace": lambda r: r[0].pop("shared_workspace"),
            "unequal_workspace": lambda r: r[0].update(equal_workspace_bytes=False),
            "missing_equal_workspace": lambda r: r[0].pop("equal_workspace_bytes"),
            "unverified_h2d": lambda r: r[0].update(measured_h2d_verified=False),
            "integer_h2d_verification": lambda r: r[0].update(measured_h2d_verified=1),
            "missing_source": lambda r: r[0].pop("source_commit"),
            "invalid_base": lambda r: r[0].update(base_commit="unavailable"),
            "profiled": lambda r: r[0].update(profiled=True),
            "fallback": lambda r: r[0].update(fallback_allowed=True),
            "unknown_record": lambda r: r.insert(1, {"type": "unexpected"}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                records = copy.deepcopy(original)
                mutate(records)
                with self.assertRaises(ValueError):
                    read_run(self.write(records), settings, 80)

    def test_exact_provenance_matches_requested_binary_and_source(self):
        for field, value in (
            ("source_commit", "d" * 40),
            ("base_commit", "e" * 40),
            ("binary_sha256", "f" * 64),
        ):
            with self.subTest(field=field):
                settings, records = fixture()
                settings[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    read_run(self.write(records), settings, 80)

    def test_measurement_contract_is_required(self):
        for field in (
            "gpu_uuid",
            "binary_sha256",
            "profiled",
            "fallback_allowed",
            "evict_multiplier",
            "block_counts",
            "segment_bytes",
            "boundary",
            "model",
            "shape_source",
            "forced_mode_outside_timer",
        ):
            with self.subTest(field=field):
                settings, records = fixture()
                records[0].pop(field)
                with self.assertRaises(ValueError):
                    read_run(self.write(records), settings, 80)

    def test_repeats_must_match_gpu_binary_source_and_workspace(self):
        fields = {
            "gpu_uuid": "other GPU",
            "gpu_name": "other model",
            "sm": 90,
            "runtime": 13030,
            "driver": 13030,
            "source_commit": "d" * 40,
            "base_commit": "e" * 40,
            "binary_sha256": "f" * 64,
        }
        for field, value in fields.items():
            with self.subTest(field=field):
                settings, first = fixture()
                _, second = fixture(repeat=81)
                second[0][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    analyze(
                        [self.write(first, "80.jsonl"), self.write(second, "81.jsonl")],
                        settings,
                    )
        settings, first = fixture()
        _, second = fixture(repeat=81)
        next(r for r in second if r["type"] == "layout")["workspace_device_bytes"] += 16
        with self.assertRaisesRegex(ValueError, "workspace"):
            analyze(
                [self.write(first, "80.jsonl"), self.write(second, "81.jsonl")],
                settings,
            )

    def test_exactly_two_different_inputs_required(self):
        settings, paths = self.pair()
        for invalid in (paths[:1], [paths[0], paths[0]], paths + paths[:1]):
            with self.subTest(inputs=invalid), self.assertRaises(ValueError):
                analyze(invalid, settings)

    def test_layout_may_carry_equal_workspace_guarantee(self):
        settings, records = fixture()
        records[0].pop("equal_workspace_bytes")
        for record in records:
            if record["type"] == "layout":
                record["equal_workspace_bytes"] = True
        read_run(self.write(records), settings, 80)

    def test_malformed_jsonl_rejected(self):
        settings, _ = fixture()
        path = Path(self.temp.name) / "bad.jsonl"
        for content in ("", "{}\n{", "null\n", "[]\n", '{"type":"metadata"}\n\n'):
            with self.subTest(content=content):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_run(path, settings, 80)


if __name__ == "__main__":
    unittest.main()
