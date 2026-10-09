import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.utils.gang_info import read_c2_gang_info


class GangInfoTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "custom-annotations"

    def test_reads_existing_annotation_format_and_refreshes_projection(self):
        for ip in ("10.0.0.2", "10.0.0.3"):
            rows = {"model_part0": {"ip": ip, "port": 1234}}
            self.path.write_text(
                'unrelated="value"\napp.c2.io/biz-detail-ganginfo='
                + json.dumps(json.dumps(rows))
                + '\nother="value"\n'
            )
            self.assertEqual(read_c2_gang_info(str(self.path)), rows)

    def test_missing_file_fails(self):
        with self.assertRaises(FileNotFoundError):
            read_c2_gang_info(str(self.path))

    def test_missing_duplicate_and_malformed_annotations_fail(self):
        line = 'app.c2.io/biz-detail-ganginfo="{}"\n'
        for text in (
            'unrelated="value"\n',
            line + line,
            'app.c2.io/biz-detail-ganginfo="not-json"\n',
        ):
            with self.subTest(text=text):
                self.path.write_text(text)
                with self.assertRaises(ValueError):
                    read_c2_gang_info(str(self.path))


if __name__ == "__main__":
    unittest.main()
