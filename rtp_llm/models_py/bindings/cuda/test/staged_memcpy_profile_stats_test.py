"""CPU-only checks for the isolated memcpy attribution report."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from crc_copy_benchmark_stats_test import fixture
from staged_memcpy_profile_stats import (
    BLOCKS,
    PROFILED_VARIANTS,
    TILES,
    analyze,
    read_run,
    write_results,
)


def profile_fixture(repeat=80):
    settings, rows = fixture(repeat=repeat)
    meta = rows[0]
    meta.update(
        implementation="staged_memcpy_profile_v2",
        profiled_variants=list(PROFILED_VARIANTS),
        profiled=True,
        block_counts=list(BLOCKS),
    )
    for field in ("source_commit", "base_commit", "binary_sha256"):
        settings[field] = meta[field]
    filtered = []
    for row in rows:
        if row["type"] in ("correctness", "sample") and row["blocks"] not in BLOCKS:
            continue
        filtered.append(row)
        if row["type"] == "sample" and row["variant"] in PROFILED_VARIANTS:
            profile = {
                field: row[field]
                for field in (
                    "direction",
                    "layout",
                    "blocks",
                    "variant",
                    "repeat",
                    "round",
                    "position",
                    "source_plan",
                )
            }
            payload = sum(TILES[row["layout"]])
            calls = row["blocks"] if row["variant"] == "main_staged" else 0
            profile.update(
                type="staged_memcpy_profile",
                wall_us=row["us"],
                payload_bytes=payload,
                memcpy_calls=calls,
                memcpy_bytes=calls * payload,
                memcpy_us=row["us"] * 0.75 if calls else 0.0,
            )
            filtered.append(profile)
    return settings, filtered


class StagedMemcpyProfileStatsTest(unittest.TestCase):
    def write(self, root, rows, repeat=80):
        path = Path(root) / f"copy_{repeat}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_complete_two_repeat_attribution(self):
        settings, rows = profile_fixture()
        with tempfile.TemporaryDirectory() as root:
            paths = [self.write(root, rows)]
            paths.append(self.write(root, profile_fixture(81)[1], 81))
            result = analyze(paths, settings)
            write_results(result, root)
            summary = (Path(root) / "summary.md").read_text()
            self.assertEqual(summary.count("| staged_no_crc |"), 12)
            self.assertEqual(summary.count("| main_staged |"), 12)
        self.assertTrue(result["diagnostic"])
        self.assertEqual(result["sample_count"], 480)
        self.assertEqual(result["profile_count"], 192)
        self.assertEqual(len(result["cases"]), 12)
        for case in result["cases"]:
            self.assertTrue(
                all(stats["count"] == 8 for stats in case["variants"].values())
            )
            optimized = case["profiles"]["staged_no_crc"]
            self.assertEqual(optimized["memcpy"]["p50_us"], 0.0)
            self.assertEqual(optimized["memcpy_fraction_p50"], 0.0)
            self.assertEqual(
                optimized["remaining"]["p50_us"],
                case["variants"]["staged_no_crc"]["p50_us"],
            )
            main = case["profiles"]["main_staged"]
            self.assertEqual(main["memcpy_fraction_p50"], 0.75)
            self.assertAlmostEqual(
                main["remaining"]["p50_us"],
                case["variants"]["main_staged"]["p50_us"] * 0.25,
            )

    def test_requires_diagnostic_protocol(self):
        for field, value in (
            ("implementation", "crc_copy_benchmark_v5"),
            ("implementation", "staged_memcpy_profile_v1"),
            ("profiled_variants", ["staged_no_crc"]),
            ("staged_host_payload_copy", "cpu_pack_unpack"),
            ("profiled", False),
            ("block_counts", list(range(1, 33))),
            ("measured_h2d_verified", False),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                settings, rows = profile_fixture()
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    read_run(self.write(root, rows), settings, 80)

    def test_rejects_missing_and_duplicate_profiles(self):
        settings, original = profile_fixture()
        index = next(
            i
            for i, row in enumerate(original)
            if row["type"] == "staged_memcpy_profile"
        )
        for duplicate in (False, True):
            with self.subTest(
                duplicate=duplicate
            ), tempfile.TemporaryDirectory() as root:
                rows = copy.deepcopy(original)
                if duplicate:
                    rows.insert(index, copy.deepcopy(rows[index]))
                else:
                    rows.pop(index)
                with self.assertRaises(ValueError):
                    read_run(self.write(root, rows), settings, 80)

    def test_rejects_incomplete_interception_and_wrong_association(self):
        for field, value in (
            ("memcpy_calls", 0),
            ("memcpy_bytes", 1),
            ("memcpy_us", 0),
            ("memcpy_us", float("nan")),
            ("memcpy_us", 100000),
            ("payload_bytes", 1),
            ("position", 1),
            ("source_plan", 9),
            ("wall_us", 1),
        ):
            with self.subTest(
                field=field, value=value
            ), tempfile.TemporaryDirectory() as root:
                settings, rows = profile_fixture()
                row = next(
                    row
                    for row in rows
                    if row["type"] == "staged_memcpy_profile"
                    and row["variant"] == "main_staged"
                )
                row[field] = value
                with self.assertRaises(ValueError):
                    read_run(self.write(root, rows), settings, 80)

    def test_rejects_optimized_payload_copy(self):
        for field, value in (
            ("memcpy_calls", 1),
            ("memcpy_bytes", 510912),
            ("memcpy_us", 0.1),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                settings, rows = profile_fixture()
                row = next(
                    row
                    for row in rows
                    if row["type"] == "staged_memcpy_profile"
                    and row["variant"] == "staged_no_crc"
                )
                row[field] = value
                with self.assertRaises(ValueError):
                    read_run(self.write(root, rows), settings, 80)

    def test_rejects_zero_positive_control(self):
        with tempfile.TemporaryDirectory() as root:
            settings, rows = profile_fixture()
            for row in rows:
                if row["type"] == "staged_memcpy_profile":
                    row.update(memcpy_calls=0, memcpy_bytes=0, memcpy_us=0.0)
            with self.assertRaises(ValueError):
                read_run(self.write(root, rows), settings, 80)

    def test_rejects_missing_candidate_and_incomplete_run(self):
        settings, original = profile_fixture()
        for kind in ("sample", "complete"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                rows = copy.deepcopy(original)
                rows.pop(next(i for i, row in enumerate(rows) if row["type"] == kind))
                with self.assertRaises(ValueError):
                    read_run(self.write(root, rows), settings, 80)


if __name__ == "__main__":
    unittest.main()
