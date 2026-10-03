"""PREFILL hook routing and isolated execution of the production finish guard."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class MtpPrefillProfileHooksTest(unittest.TestCase):
    def test_role_and_work_boundaries(self):
        engine = (ROOT / "rtp_llm/cpp/normal_engine/NormalEngine.cc").read_text()
        executor = (
            ROOT / "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc"
        ).read_text()
        header = (
            ROOT / "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.h"
        ).read_text()
        init = engine.split("void NormalEngine::initExecutor", 1)[1].split(
            "void NormalEngine::initScheduler", 1
        )[0]
        mtp_init = init.split("} else {", 1)[0]
        self.assertEqual(
            mtp_init.count("params.pd_sep_config.role_type == RoleType::PREFILL ?"), 2
        )
        self.assertIn("prefill_profile_start   = nullptr", header)
        self.assertIn("prefill_profile_finish  = nullptr", header)
        self.assertIn(
            "propose_params_ && pd_sep_config.role_type != RoleType::PREFILL", engine
        )
        prefill = executor.split("absl::Status MtpExecutor::prefillStep", 1)[1].split(
            "absl::Status MtpExecutor::decodeStep", 1
        )[0]
        self.assertLess(
            prefill.index("RETURN_IF_STATUS_OR_ERROR(model_input_status)"),
            prefill.index("prefill_profile_start_();"),
        )
        self.assertLess(
            prefill.index("tpSyncModelInputs("),
            prefill.index("if (model_input.skip_run)"),
        )
        self.assertLess(
            prefill.index("if (model_input.skip_run)"),
            prefill.index("prefill_profile_start_();"),
        )
        self.assertLess(
            prefill.index("finish_prefill_profile{prefill_profile_finish_}"),
            prefill.index("releaseAllModelBuffers();"),
        )
        stop = engine.split("absl::Status NormalEngine::stop()", 1)[1]
        self.assertIn("loop_thread_->join();", stop)

    @unittest.skipUnless(shutil.which("g++"), "requires host C++ compiler")
    def test_actual_guard_returns_unwinds_and_stays_on_execution_thread(self):
        source = (
            ROOT / "rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc"
        ).read_text()
        guard = (
            "struct FinishPrefillProfile {"
            + source.split("struct FinishPrefillProfile {", 1)[1].split(
                "} finish_prefill_profile", 1
            )[0]
            + "};"
        )
        program = (
            r"""
#include <functional>
#include <cassert>
#include <thread>
#include <stdexcept>
#include <string>
#define RTP_LLM_LOG_WARNING(...) do {} while (0)
#define RTP_LLM_LOG_ERROR(...) do {} while (0)
"""
            + guard
            + r"""
int execute(const std::function<void()>& finish, bool error) {
    FinishPrefillProfile guard{finish};
    if (error) throw std::runtime_error("model failure");
    return 7;
}
int main() {
    int finished = 0;
    auto execution_thread = std::this_thread::get_id();
    std::function<void()> finish = [&]() {
        assert(std::this_thread::get_id() == execution_thread);
        ++finished;
    };
    assert(execute(finish, false) == 7 && finished == 1);
    try { execute(finish, true); assert(false); }
    catch (const std::runtime_error&) {}
    assert(finished == 2);
    assert(execute(nullptr, false) == 7);
    for (bool unknown : {false, true}) {
        std::function<void()> throwing_finish = [&]() {
            ++finished;
            if (unknown) throw 3;
            throw std::runtime_error("finish failure");
        };
        assert(execute(throwing_finish, false) == 7);
        try { execute(throwing_finish, true); assert(false); }
        catch (const std::runtime_error& error) {
            assert(std::string(error.what()) == "model failure");
        }
    }
    assert(finished == 6);
}
"""
        )
        with tempfile.TemporaryDirectory(prefix="mtp_prefill_hook_guard_") as temp:
            binary = str(Path(temp) / "guard_test")
            result = subprocess.run(
                ["g++", "-std=c++17", "-pthread", "-x", "c++", "-", "-o", binary],
                input=program,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run([binary], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
