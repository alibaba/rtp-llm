#include "rtp_llm/cpp/utils/CudacoreDiagnostics.h"

#include <chrono>
#include <exception>

#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(CudacoreTerminateGuardTest, UncaughtFatalExceptionRespectsWindowAndDelegates) {
    const auto  started = std::chrono::steady_clock::now();
    const pid_t child   = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
        std::set_terminate([] { ::_exit(77); });
        cudacore_test::setCollectionWindowMsForTest(180);
        installCudacoreTerminateGuard();
        auto record =
            buildCudaRuntimeErrorRecord(719, FatalCudaErrorSite::BatchedCopyCompletion, "NoBlockCopy.cc", 1, 2);
        (void)recordFirstFatalCudaError(record);
        std::terminate();
    }

    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    const auto waited_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 77);
    EXPECT_GE(waited_ms, 100);
    EXPECT_LT(waited_ms, 2000);
}

}  // namespace
}  // namespace rtp_llm
