"""CPU/source-only profiler acceptance contract; no torch/native imports."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class AcceptanceProfileSourceTest(unittest.TestCase):
    def test_annotation_uses_single_same_step_host_snapshot(self):
        source = (
            ROOT / "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc"
        ).read_text()
        collect = source.split("void MtpExecutor::collectDecodeMetrics(", 1)[1].split(
            "absl::Status MtpExecutor::dispatchDecodeOutput", 1
        )[0]
        self.assertEqual(collect.count("consumePendingAcceptLenMetrics()"), 1)
        annotation = collect.split(
            "const auto accept_len_metrics = consumePendingAcceptLenMetrics();", 1
        )[1].split("const int64_t total_accept_len", 1)[0]
        self.assertIn("if (is_dspark_ && accept_len_metrics.valid)", annotation)
        self.assertIn("RTP_LLM_PROFILE_SCOPE_DYNAMIC(", annotation)
        self.assertIn(
            "executor.mtp.decode_step(acceptance,sum=%lld,streams=%lld,verify_steps=%zu)",
            annotation,
        )
        self.assertIn("accept_len_metrics.total_accept_len", annotation)
        self.assertIn("accept_len_metrics.total_stream_num", annotation)
        self.assertIn("verifySteps()", annotation)
        for forbidden in (
            ".item",
            ".cpu",
            "synchronize(",
            "copy_(",
            "torch::",
            "RTP_LLM_LOG",
            "getenv",
        ):
            self.assertNotIn(forbidden, annotation)
        # Keep the existing parser scope and current same-step consumption.
        self.assertIn(
            'RTP_LLM_PROFILE_SCOPE("executor.mtp.decode_step(collect_metrics)")',
            collect,
        )
        decode = source.split("absl::Status MtpExecutor::decodeStep(", 1)[1].split(
            "void MtpExecutor::waitPreviousBookkeepingAndKvSwaps", 1
        )[0]
        self.assertLess(
            decode.index("stageAcceptLenMetrics("),
            decode.index("collectDecodeMetrics("),
        )
        self.assertLess(
            decode.index("collectDecodeMetrics("),
            decode.index("return dispatchDecodeOutput("),
        )

    def test_macro_only_formats_when_profiler_callbacks_active(self):
        source = (ROOT / "rtp_llm/cpp/utils/ProfilingScope.h").read_text()
        macro = source.split("#define RTP_LLM_PROFILE_SCOPE_DYNAMIC", 1)[1]
        self.assertLess(
            macro.index("if (at::hasCallbacks())"), macro.index("snprintf(")
        )

    def test_early_window_cannot_reach_output_limit(self):
        # The gate arms BEFORE releasing any GENERATE and min=max=256.
        # Include the one P-produced token in this conservative bound.
        self.assertLess(1 + 16 * (4 + 1), 256)
        self.assertLess(1 + 16 * (7 + 1), 256)
        source = (ROOT / "rtp_llm/cpp/engine_base/stream/GenerateStream.cc").read_text()
        finish = source.split("bool GenerateStream::needFinishBySPTokens()", 1)[
            1
        ].split("void GenerateStream::matchEosToken()", 1)[0]
        self.assertIn(
            "seqLength() >= generate_input_->generate_config->min_new_tokens + inputLength()",
            finish,
        )
        self.assertIn("matchEosToken();", finish)
        self.assertIn("matchStopWordsList();", finish)


if __name__ == "__main__":
    unittest.main()
