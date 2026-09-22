"""Direct report data must survive HTML serialization without JSX escaping."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reporting import stress_report as generator
from reporting import renderer


class ReportSpecTest(unittest.TestCase):
    def test_labels_and_nullable_series_survive_rendering(self):
        label = 'engine O\'Brien "P" & <cache> {x} 中文\\path </script><script>bad()</script>'
        categories = [label, "next\nline", "tail"]
        data = [0, None, 12.3456]
        panel = generator.panel_spec(
            label,
            label,
            generator.chart_spec(
                "line",
                categories,
                [("key", label, data, "info")],
                suffix=" tok/s",
                y_max=20,
            ),
        )
        panel.update(id="p1", timeX=True, xNums=[-1, 0, 2])
        page = renderer.render(
            {
                "run_id": label,
                "panels": [panel],
                "kpis": [generator.kpi_spec(label, label, "warning")],
                "timeAxis": {"min": 0, "max": 2},
            }
        )
        self.assertNotIn("</script><script>bad()</script>", page)
        payload, _ = json.JSONDecoder().raw_decode(page.split("const SPEC = ", 1)[1])
        result = payload["panels"][0]
        self.assertEqual(result["x"], categories)
        self.assertEqual(result["series"][0]["data"], data)
        self.assertEqual(result["series"][0]["name"], label)
        self.assertEqual(result["title"], label)
        self.assertEqual(result["caption"], label)
        self.assertEqual(result["xNums"], [-1, 0, 2])
        self.assertEqual(
            payload["summary"]["kpis"][0],
            {"label": label, "value": label, "tone": "warn"},
        )

    def test_numeric_cleanup_keeps_existing_precision(self):
        self.assertEqual(
            generator.num_arr(
                [None, True, "bad", float("inf"), float("nan"), 3, 1.234567]
            ),
            [0, 1, 0, 0, 0, 3, 1.2346],
        )


if __name__ == "__main__":
    unittest.main()
