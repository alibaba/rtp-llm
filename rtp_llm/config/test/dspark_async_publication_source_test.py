"""Source contract only; tensor math and live lifecycle have separate gates."""

import unittest
from pathlib import Path


class DSparkAsyncPublicationSourceTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3]
        self.source = (
            root / "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc"
        ).read_text()
        self.body = self.source.split(
            "absl::Status MtpExecutor::dispatchDecodeAsync(", 1
        )[1]

    def test_batch_is_dspark_only_and_preserves_failed_state(self):
        self.assertIn(
            "is_dspark_ && batch_size > 0 && spec_decode_output.success.defined()",
            self.body,
        )
        batch = self.body.split("if (batch_dspark_publication) {", 1)[1].split(
            "// Select all accepted target tokens", 1
        )[0]
        self.assertEqual(batch.count("torch::where("), 3)
        self.assertIn("torch::where(ok, next_seq_len_all, prev_seq_len_all)", batch)
        self.assertIn("torch::cat(previous_tokens, 0)", batch)
        self.assertIn("same_width && previous.accept_len_gpu.defined()", batch)
        self.assertIn("toCudaInt32WithHostHold(", batch)
        for forbidden in (".item", ".cpu", "synchronize(", "data_ptr"):
            self.assertNotIn(forbidden, batch)

    def test_old_anchor_only_computed_for_mismatched_width(self):
        batch = self.body.split("if (batch_dspark_publication) {", 1)[1]
        same = batch.split("if (same_width) {", 1)[1].split("} else {", 1)[0]
        self.assertIn("previous_tokens.push_back(previous.accept_tokens_gpu)", same)
        self.assertNotIn("gather(", same)

    def test_legacy_branch_and_bookkeeping_order_retained(self):
        self.assertIn(
            "if (spec_decode_output.success.defined() && !batch_dspark_publication)",
            self.body,
        )
        self.assertIn(
            "spec_decode_output.success.defined() && !batch_dspark_publication ?",
            self.body,
        )
        self.assertLess(
            self.body.index("stream->setMtpAsyncDeviceState("),
            self.body.index("s->incPendingAsyncBookkeeping()"),
        )
        self.assertLess(
            self.body.index("s->incPendingAsyncBookkeeping()"),
            self.body.index("spec_bookkeeping_runner_.launch("),
        )
        self.assertIn("s->decPendingAsyncBookkeepingAndMaybeRelease()", self.body)


if __name__ == "__main__":
    unittest.main()
