import io
import json
import unittest
from zipfile import ZipFile

from compare_v41_long_logits import FIXTURE, RECORD, TARGET, compare, read_record


def archive(role, **changes):
    record = dict(
        schema=1,
        loader=role,
        reader="fastsafetensors",
        input_ids_sha256=FIXTURE,
        input_len=783,
        pd_sep=False,
        reuse_len=0,
        logit_1606=2.5,
        logit_28=2.0,
        argmax=1606,
        http_first_token=1606,
    )
    record.update(changes)
    nested, outer = io.BytesIO(), io.BytesIO()
    with ZipFile(nested, "w") as f:
        f.writestr(RECORD, json.dumps(record))
    with ZipFile(outer, "w") as f:
        f.writestr(
            "rtp_llm/test/smoke/" + TARGET + "/test.outputs/outputs.zip",
            nested.getvalue(),
        )
    return outer


class ComparisonTest(unittest.TestCase):
    def test_reversed_order_and_different_argmax(self):
        result = compare(archive("oldfst"), archive("newfst", logit_28=3.25, argmax=28))
        self.assertTrue(result["candidate_order_changed"])
        self.assertTrue(result["argmax_changed"])
        self.assertFalse(result["http_first_token_changed"])
        self.assertEqual(result["newfst"]["margin_1606_minus_28"], -0.75)

    def test_two_candidates_do_not_determine_global_winner(self):
        result = compare(archive("oldfst", argmax=7), archive("newfst", argmax=9))
        self.assertFalse(result["candidate_order_changed"])
        self.assertTrue(result["argmax_changed"])

    def test_wrong_role_or_mismatched_input_or_missing_logit_rejected(self):
        for item in (
            archive("newfst"),
            archive("oldfst", input_ids_sha256="else"),
            archive("oldfst", pd_sep=True),
            archive("oldfst", reader="scratch"),
            archive("oldfst", logit_28=None),
            archive("oldfst", logit_28=float("nan")),
        ):
            with self.assertRaises(ValueError):
                read_record(item, "oldfst")


if __name__ == "__main__":
    unittest.main()
