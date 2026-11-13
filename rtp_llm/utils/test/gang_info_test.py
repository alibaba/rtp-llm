import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rtp_llm.utils.gang_info import GangInfoReader, get_gang_member


class GangInfoTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "custom-annotations"
        self.reader = GangInfoReader.from_config(
            SimpleNamespace(gang_annotation_path=str(self.path))
        )

    def test_reads_existing_annotation_format_and_refreshes_projection(self):
        for ip in ("10.0.0.2", "10.0.0.3"):
            rows = {"model_part0": {"ip": ip, "port": 1234}}
            replacement = self.path.with_suffix(".new")
            replacement.write_text(
                'unrelated="value"\napp.c2.io/biz-detail-ganginfo='
                + json.dumps(json.dumps(rows))
                + '\nother="value"\n'
            )
            replacement.replace(self.path)
            self.assertEqual(self.reader.read(), rows)

    def test_missing_file_fails(self):
        with self.assertRaises(FileNotFoundError):
            self.reader.read()

    def test_member_lookup_preserves_rtp_names_and_merged_fields(self):
        for prefix in ("prefill_part", "prefill-part", "prefill-rank-"):
            with self.subTest(prefix=prefix):
                first = {"ip": "22.0.1.2", "real_ip": "10.0.0.2", "port": 1234}
                second = {"ip": "22.0.1.3", "real_ip": "10.0.0.3"}
                rows = {f"{prefix}1": second, f"{prefix}0": first}
                self.assertIs(get_gang_member(rows, 0), first)
                self.assertIs(get_gang_member(rows, 1), second)
                self.assertIsNone(get_gang_member(rows, 2))

    def test_member_lookup_distinguishes_part1_from_part10(self):
        rows = {"prefill_part10": {"ip": "22.0.1.12"}}
        self.assertIsNone(get_gang_member(rows, 1))
        self.assertEqual(get_gang_member(rows, 10), rows["prefill_part10"])

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
                    self.reader.read()


if __name__ == "__main__":
    unittest.main()
