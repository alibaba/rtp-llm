#!/usr/bin/env python3

import tempfile
import unittest
import argparse
import json
import io
from contextlib import redirect_stdout
from pathlib import Path

from AnalyzeLatencyAdamScale import (
    METRIC_OFFSETS,
    PARAMETER_COUNT,
    TRACE_PARAMETER_FIELDS,
    analyze,
    classify_pressure_phase,
    classify_sine_phase,
    load_pressure_means,
    load_host_architecture,
    parse_trace_line,
    parse_host_architecture,
    print_summary,
)


def make_trace_line(host="mi_host", event_time_us=2_000_000,
                    target_ms=100.0):
    fields = [
        "D", host, "4", "0", "1", "0", str(event_time_us),
        str(target_ms), "4", "16", "0.1", "200", "201",
        str(PARAMETER_COUNT),
    ]
    for index in range(PARAMETER_COUNT):
        parameter = [0.0] * TRACE_PARAMETER_FIELDS
        parameter[0] = index + 1.0
        parameter[7] = -(index + 2.0)
        parameter[11] = 0.1 * index
        parameter[12] = -0.2 * index
        parameter[14] = 3.0 * index
        fields.extend(str(value) for value in parameter)
    return "\t".join(fields) + "\n"


def make_model_state_line(host="mi_host", event_time_us=2_000_000,
                          target_ms=100.0, running=1, waiting=0):
    fields = [
        "M", host, "4", "0", "1", "0", str(event_time_us),
        str(target_ms), "1", str(running), "0", "0", "0", "0",
        str(waiting), "0", "0", "0", "0", "0", "0", "100.0",
        "100.0", "0.0", "19",
    ]
    fields.extend(["0"] * 19)
    fields.append("0")
    return "\t".join(fields) + "\n"


class AnalyzeLatencyAdamScaleTest(unittest.TestCase):
    def test_parse_trace_line_uses_documented_parameter_columns(self):
        record = parse_trace_line(make_trace_line())
        self.assertEqual("mi_host", record["host"])
        self.assertEqual(PARAMETER_COUNT, record["parameter_count"])
        self.assertEqual(set(METRIC_OFFSETS), set(record["values"]))
        self.assertEqual(19.0, record["values"]["raw_gradient"][18])
        self.assertEqual(-20.0, record["values"]["normalized_gradient"][18])
        self.assertEqual(0.0, record["values"]["parameter_before"][18])
        self.assertEqual(0.0, record["values"]["parameter_after"][18])
        self.assertAlmostEqual(1.8, record["values"]["candidate_update"][18])
        self.assertAlmostEqual(-3.6, record["values"]["actual_update"][18])
        self.assertEqual(54.0, record["values"]["prediction_shift_ms"][18])

    def test_classify_sine_phase(self):
        start = 1_000_000
        period = 100_000_000
        self.assertEqual("unknown", classify_sine_phase(start, None, period))
        self.assertEqual("unknown", classify_sine_phase(999_999, start, period))
        self.assertEqual("turning", classify_sine_phase(start, start, period))
        self.assertEqual("rising", classify_sine_phase(start + 25_000_000,
                                                       start, period))
        self.assertEqual("turning", classify_sine_phase(start + 50_000_000,
                                                        start, period))
        self.assertEqual("falling", classify_sine_phase(start + 75_000_000,
                                                        start, period))

    def test_analyze_separates_architecture_and_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.tsv"
            trace.write_text(
                make_trace_line("mi_host", 26_000_000, 1_000.0)
                + make_trace_line("rtx_host", 76_000_000, 1_000.0),
                encoding="utf-8",
            )
            result = analyze(
                trace,
                {"mi_host": "MI308", "rtx_host": "RTX5000"},
                sine_start_us=0,
                sine_period_us=100_000_000,
            )
        mi = result["architectures"]["MI308"]["phases"]
        rtx = result["architectures"]["RTX5000"]["phases"]
        self.assertEqual(1, mi["rising"]["update_count"])
        self.assertEqual(1, rtx["falling"]["update_count"])
        self.assertEqual(
            19.0,
            mi["all"]["metrics"]["raw_gradient"][18]["absolute_p95"],
        )
        self.assertEqual(
            -3.6,
            rtx["all"]["metrics"]["actual_update"][18]["signed_mean"],
        )
        self.assertEqual(
            15.0,
            mi["all"]["metrics"]
              ["prediction_sensitivity_ms_per_coordinate"][18]
              ["absolute_p50"],
        )

    def test_pressure_trend_uses_architecture_bin_means(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.tsv"
            trace.write_text(
                make_model_state_line("mi_host", 11_000_000, 1_000.0, 2, 1)
                + make_model_state_line("mi_host", 21_000_000, 1_000.0, 8, 4)
                + make_model_state_line("mi_host", 31_000_000, 1_000.0, 3, 1),
                encoding="utf-8",
            )
            means = load_pressure_means(trace, {"mi_host": "MI308"}, 10_000_000)
        self.assertEqual(3.0, means[("MI308", 1)])
        self.assertEqual(12.0, means[("MI308", 2)])
        self.assertEqual("rising", classify_pressure_phase(
            "MI308", 20_000_000, means, 10_000_000
        ))
        self.assertEqual("falling", classify_pressure_phase(
            "MI308", 30_000_000, means, 10_000_000
        ))

    def test_pressure_trend_analysis_uses_matching_model_state(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.tsv"
            trace.write_text(
                make_model_state_line("mi_host", 11_000_000, 1_000.0, 2, 1)
                + make_trace_line("mi_host", 11_000_000, 1_000.0)
                + make_model_state_line("mi_host", 21_000_000, 1_000.0, 8, 4)
                + make_trace_line("mi_host", 21_000_000, 1_000.0)
                + make_model_state_line("mi_host", 31_000_000, 1_000.0, 3, 1)
                + make_trace_line("mi_host", 31_000_000, 1_000.0),
                encoding="utf-8",
            )
            result = analyze(
                trace,
                {"mi_host": "MI308"},
                phase_mode="pressure-trend",
                pressure_bin_us=10_000_000,
            )
        phases = result["architectures"]["MI308"]["phases"]
        self.assertEqual(2, phases["rising"]["update_count"])
        self.assertEqual(1, phases["falling"]["update_count"])
        self.assertAlmostEqual(
            -3.6 * 0.403880546944,
            phases["all"]["metrics"]
                  ["effective_coefficient_update"][18]["signed_mean"],
        )

        output = io.StringIO()
        with redirect_stdout(output):
            print_summary(result)
        self.assertIn("0 1", output.getvalue())
        self.assertIn("n/a", output.getvalue())

    def test_rejects_wrong_trace_shape(self):
        with self.assertRaisesRegex(ValueError, "Adam trace"):
            parse_trace_line("D\tbroken\n")

    def test_parse_host_architecture_normalizes_ip(self):
        self.assertEqual(
            ("11_225_195_118", "MI308"),
            parse_host_architecture("11.225.195.118=MI308"),
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_host_architecture("11.225.195.118=unknown")

    def test_manifest_ignores_unrelated_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"nodes": [
                {"ip": "11.1.1.1", "deployment": "cn_MI308X_2TP_spot"},
                {"ip": "11.1.1.2", "deployment": "cn_H20_4TP_spot"},
            ]}), encoding="utf-8")
            self.assertEqual(
                {
                    "11_1_1_1": "MI308",
                    "cn_MI308X_2TP_spot_11_1_1_1": "MI308",
                },
                load_host_architecture(manifest),
            )

    def test_manifest_maps_deployment_prefixed_trace_host(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"nodes": [
                {"ip": "11.225.194.47",
                 "deployment": "cn_MI308X_2TP_spot_new2"},
                {"ip": "33.249.233.127",
                 "deployment": "cn_RTX_PRO_5000_72GB_4TP_spot"},
            ]}), encoding="utf-8")
            hosts = load_host_architecture(manifest)
        self.assertEqual("MI308", hosts[
            "cn_MI308X_2TP_spot_new2_11_225_194_47"
        ])
        self.assertEqual("RTX5000", hosts[
            "cn_RTX_PRO_5000_72GB_4TP_spot_33_249_233_127"
        ])

    def test_manifest_requires_target_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"nodes": [
                {"ip": "11.1.1.2", "deployment": "cn_H20_4TP_spot"},
            ]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "MI308 or RTX5000"):
                load_host_architecture(manifest)


if __name__ == "__main__":
    unittest.main()
