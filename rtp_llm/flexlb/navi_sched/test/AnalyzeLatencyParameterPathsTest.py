#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from AnalyzeLatencyAdamScaleTest import make_trace_line
from AnalyzeLatencyParameterPaths import analyze


def parameterized_line(event_time_us, before, after, update):
    fields = make_trace_line(
        host="mi_host", event_time_us=event_time_us, target_ms=0.0
    ).rstrip("\n").split("\t")
    parameter_offset = 14
    parameter_width = 15
    fields[parameter_offset + 8] = str(before)
    fields[parameter_offset + 10] = str(after)
    fields[parameter_offset + 12] = str(update)
    return "\t".join(fields) + "\n"


class AnalyzeLatencyParameterPathsTest(unittest.TestCase):
    def test_summarizes_net_change_and_total_variation_per_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.tsv"
            trace.write_text(
                parameterized_line(10, 1.0, 2.0, 1.0) +
                parameterized_line(20, 2.0, 1.5, -0.5) +
                parameterized_line(110, 1.5, 4.0, 2.5),
                encoding="utf-8",
            )
            result = analyze(
                trace, {"mi_host": "MI308"}, start_us=0,
                period_us=100, cycle_count=2)
        first = result["host_paths"][0]["parameters"][0]
        self.assertAlmostEqual(0.5 * 0.878862443703,
                               first["effective_net_change"])
        self.assertAlmostEqual(1.5 * 0.878862443703,
                               first["effective_total_variation"])
        self.assertAlmostEqual(1.0 / 3.0,
                               first["absolute_net_over_total_variation"])
        self.assertEqual(2, result["architectures"]["MI308"]
                         ["cycles"]["0"]["update_count"])


if __name__ == "__main__":
    unittest.main()
