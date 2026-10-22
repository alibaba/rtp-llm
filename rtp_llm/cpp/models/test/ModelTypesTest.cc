#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"
#include "gtest/gtest.h"

#include <array>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace rtp_llm {
namespace {

// Fork before initializing CUDA: each child owns its pinned allocator and the
// real CPU TP transport. No model forward or second control collective is used.
void checkTerminalBroadcast(int rank, const std::string& path) {
    auto& broadcaster = CpuTpBroadcaster::instance();
    broadcaster.initialize(rank, 2, path);
    ParallelismConfig config;
    config.tp_size = 2;
    config.tp_rank = rank;
    for (bool terminal : {false, true, false}) {
        for (bool skip_lm_head : {false, true}) {
            for (bool capture : {false, true}) {
                GptModelInputs inputs;
                inputs.terminal_stop         = rank == 0 ? terminal : !terminal;
                inputs.skip_run              = rank == 0 ? !terminal : terminal;
                inputs.record_execution_id   = rank == 0 ? 123 : 0;
                inputs.skip_lm_head          = rank == 0 ? skip_lm_head : !skip_lm_head;
                inputs.capture_hidden_states = rank == 0 ? capture : !capture;
                tpSyncModelInputs(inputs, config);
                EXPECT_EQ(inputs.terminal_stop, terminal);
                EXPECT_EQ(inputs.skip_run, !terminal);
                EXPECT_EQ(inputs.record_execution_id, 123);
                EXPECT_EQ(inputs.skip_lm_head, skip_lm_head);
                EXPECT_EQ(inputs.capture_hidden_states, capture);
                EXPECT_FALSE(inputs.combo_tokens.defined());
            }
        }
    }
    broadcaster.reset();
}

TEST(ModelTypesTest, TerminalBroadcastIsRootOwnedAndIndependentOfSkipRun) {
    // Protect the old wire positions, not just the newly added flag.
    static_assert(skipRun == 22 && recordExecutionId == 26 && captureHiddenStates == 28);
    static_assert(terminalStop == 29 && gptModelInputLength == 30);
    const char* tmp = std::getenv("TEST_TMPDIR");
    ASSERT_NE(tmp, nullptr);
    // Bazel's absolute TEST_TMPDIR can exceed sockaddr_un::sun_path.
    // Children use the same directory with a short relative socket name.
    const std::string    path = "terminal_" + std::to_string(::getpid());
    std::array<pid_t, 2> children;
    for (int rank = 0; rank < 2; ++rank) {
        children[rank] = ::fork();
        ASSERT_GE(children[rank], 0);
        if (children[rank] == 0) {
            if (::chdir(tmp) != 0) {
                ::_exit(3);
            }
            ::alarm(20);
            try {
                checkTerminalBroadcast(rank, path);
                ::_exit(::testing::Test::HasFailure() ? 1 : 0);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "terminal broadcast rank %d: %s\n", rank, e.what());
                ::_exit(2);
            }
        }
    }
    for (const auto child : children) {
        int status = 0;
        ASSERT_EQ(::waitpid(child, &status, 0), child);
        EXPECT_TRUE(WIFEXITED(status)) << status;
        if (WIFEXITED(status)) {
            EXPECT_EQ(WEXITSTATUS(status), 0);
        }
    }
}

}  // namespace
}  // namespace rtp_llm
