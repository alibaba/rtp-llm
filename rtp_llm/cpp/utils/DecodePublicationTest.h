#pragma once

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <utility>
#include <unistd.h>
#include <gtest/gtest.h>

namespace rtp_llm::test {
// Set the environment BEFORE entering this helper. A threadsafe death test
// re-execs the binary, including fixture setup. Keep mocks/resources local to
// function, or pass cleanup to run fixture teardown and release owned state:
// _Exit does not run the fixture destructor. Cleanup also runs after ASSERT_*
// or GTEST_SKIP returns from function. A skipped child is NOT a passing case.
// The success marker also rejects exit(0) before the body/cleanup was reached.
template<typename Function, typename Cleanup>
void withDecodePublicationStartup(Function&& function, Cleanup&& cleanup) {
    struct RestoreStyle {
        std::string previous = ::testing::GTEST_FLAG(death_test_style);
        ~RestoreStyle() { ::testing::GTEST_FLAG(death_test_style) = previous; }
    } restore;
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    ASSERT_EXIT({
        if (::dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
            std::perror("startup test stdout redirection failed");
            std::_Exit(1);
        }
        try { function(); }
        catch (const std::exception& error) { ADD_FAILURE() << "startup test exception: " << error.what(); }
        catch (...) { ADD_FAILURE() << "startup test unknown exception"; }
        try { cleanup(); }
        catch (const std::exception& error) { ADD_FAILURE() << "startup cleanup exception: " << error.what(); }
        catch (...) { ADD_FAILURE() << "startup cleanup unknown exception"; }
        // The exec child's event forwarding is suppressed by GoogleTest. The
        // default reporter still records assertion results; print them without
        // replacing that reporter or changing HasFailure/HasFatalFailure.
        // Body and cleanup must join their workers before these unlocked reads.
        const bool failed = ::testing::Test::HasFailure();
        const bool skipped = ::testing::Test::IsSkipped();
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        if (!info) {
            std::fputs("startup test has no current test info\n", stderr);
            std::fflush(stderr);
            std::_Exit(1);
        }
        const auto* results = info->result();
        for (int i = 0; i < results->total_part_count(); ++i) {
            const auto& part = results->GetTestPartResult(i);
            if (!part.failed() && !part.skipped()) continue;
            if (part.file_name() && part.line_number() >= 0) {
                std::fprintf(stderr, "%s:%d: %s: %s\n", part.file_name(), part.line_number(),
                             part.skipped() ? "skipped" : "failed", part.message());
            } else {
                std::fprintf(stderr, "unknown location: %s: %s\n",
                             part.skipped() ? "skipped" : "failed", part.message());
            }
        }
        std::fprintf(stderr, "startup-result failed=%d skipped=%d\n", failed, skipped);
        if (!failed && !skipped) {
            std::fputs("startup-child-complete\n", stderr);
        }
        std::fflush(stdout);
        std::fflush(stderr);
        std::_Exit(failed || skipped ? 1 : 0);
    }, ::testing::ExitedWithCode(0), "startup-child-complete");
}

template<typename Function>
void withDecodePublicationStartup(Function&& function) {
    withDecodePublicationStartup(std::forward<Function>(function), [] {});
}
}  // namespace rtp_llm::test
