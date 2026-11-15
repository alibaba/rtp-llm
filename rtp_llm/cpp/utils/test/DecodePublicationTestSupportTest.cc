#include "rtp_llm/cpp/utils/DecodePublicationTest.h"
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <gmock/gmock.h>
#include <gtest/gtest-spi.h>

namespace rtp_llm::test {
namespace {
// Do not install EXPECT_FATAL_FAILURE's assertion interceptor in the exec child:
// it would hide the very assertion that the helper must propagate to the parent.
bool inDeathTestChild() {
    return !::testing::GTEST_FLAG(internal_run_death_test).empty();
}
void nonfatalProbe() {
    withDecodePublicationStartup([] { EXPECT_TRUE(false) << "startup-nonfatal-probe"; });
}
void fatalProbe() {
    withDecodePublicationStartup([] { ASSERT_TRUE(false) << "startup-fatal-probe"; },
                                 [] { std::fputs("startup-cleanup-ran\n", stderr); });
}
void cleanupFailureProbe() {
    withDecodePublicationStartup([] {}, [] { EXPECT_TRUE(false) << "startup-cleanup-failure"; });
}
void skipProbe() {
    withDecodePublicationStartup([] { GTEST_SKIP() << "startup-skip-probe"; });
}
void earlyExitProbe() {
    withDecodePublicationStartup([] { std::_Exit(0); });
}
void exceptionProbe() {
    withDecodePublicationStartup([] { throw std::runtime_error("startup-exception-probe"); });
}
void fixtureMockProbe() {
    std::shared_ptr<::testing::MockFunction<void()>> mock;
    withDecodePublicationStartup([&] {
        mock = std::make_shared<::testing::MockFunction<void()>>();
        EXPECT_CALL(*mock, Call()).Times(1);
    }, [&] { mock.reset(); }); // Same explicit cleanup required for fixture-owned mocks.
}

TEST(DecodePublicationTestSupportTest, SuccessfulBodyAndCleanupPass) {
    bool initialized = false;
    withDecodePublicationStartup([&] { initialized = true; }, [&] { EXPECT_TRUE(initialized); });
}
TEST(DecodePublicationTestSupportTest, NonfatalAssertionTextReachesParent) {
    if (inDeathTestChild()) { nonfatalProbe(); return; }
    EXPECT_FATAL_FAILURE(nonfatalProbe(), "startup-nonfatal-probe");
}
TEST(DecodePublicationTestSupportTest, FatalAssertionTextReachesParent) {
    if (inDeathTestChild()) { fatalProbe(); return; }
    EXPECT_FATAL_FAILURE(fatalProbe(), "startup-fatal-probe");
}
TEST(DecodePublicationTestSupportTest, CleanupRunsAfterFatalAssertion) {
    if (inDeathTestChild()) { fatalProbe(); return; }
    EXPECT_FATAL_FAILURE(fatalProbe(), "startup-cleanup-ran");
}
TEST(DecodePublicationTestSupportTest, CleanupAssertionCannotSilentlyPass) {
    if (inDeathTestChild()) { cleanupFailureProbe(); return; }
    EXPECT_FATAL_FAILURE(cleanupFailureProbe(), "startup-cleanup-failure");
}
TEST(DecodePublicationTestSupportTest, SkipCannotSilentlyPass) {
    if (inDeathTestChild()) { skipProbe(); return; }
    EXPECT_FATAL_FAILURE(skipProbe(), "startup-result failed=0 skipped=1");
}
TEST(DecodePublicationTestSupportTest, EarlyZeroExitCannotPretendCleanupCompleted) {
    if (inDeathTestChild()) { earlyExitProbe(); return; }
    EXPECT_FATAL_FAILURE(earlyExitProbe(), "died but not with expected error");
}
TEST(DecodePublicationTestSupportTest, ExceptionIsReportedToParent) {
    if (inDeathTestChild()) { exceptionProbe(); return; }
    EXPECT_FATAL_FAILURE(exceptionProbe(), "startup-exception-probe");
}
TEST(DecodePublicationTestSupportTest, UnsatisfiedFixtureMockIsVerifiedBeforeExit) {
    if (inDeathTestChild()) { fixtureMockProbe(); return; }
    EXPECT_FATAL_FAILURE(fixtureMockProbe(), "unsatisfied and active");
}
}  // namespace
}  // namespace rtp_llm::test
