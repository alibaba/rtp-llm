"""Direct report data must survive HTML serialization without JSX escaping."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reporting import renderer


class ReportSpecTest(unittest.TestCase):
    def test_labels_and_nullable_series_survive_rendering(self):
        label = 'engine O\'Brien "P" & <cache> {x} 中文\\path </script><script>bad()</script>'
        categories = [label, "next\nline", "tail"]
        data = [0, None, 12.3456]
        panel = dict(title=label, caption=label, type="line", x=categories,
                     yMax=20, unit="tok/s", series=[dict(name=label, data=data)])
        panel.update(id="p1", timeX=True, xNums=[-1, 0, 2])
        page = renderer.render(
            {
                "run_id": label,
                "panels": [panel],
                "kpis": [dict(label=label, value=label, tone="warn")],
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


if __name__ == "__main__":
    unittest.main()
