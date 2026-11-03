#include <csignal>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/utils/SignalUtils.h"

namespace rtp_llm {
namespace {

TEST(SignalUtilsTest, InstallSighandlerIgnoresSigpipe) {
    // A native runtime may restore the default disposition, which terminates the
    // process on the next write to a closed pipe. Installing the handler must
    // take SIGPIPE away from that path.
    ASSERT_NE(signal(SIGPIPE, SIG_DFL), SIG_ERR);
    ASSERT_TRUE(installSighandler());

    struct sigaction current;
    ASSERT_EQ(sigaction(SIGPIPE, nullptr, &current), 0);
    EXPECT_EQ(current.sa_handler, SIG_IGN);
}

TEST(SignalUtilsTest, InstallSighandlerKeepsFatalSignalsHandled) {
    ASSERT_TRUE(installSighandler());

    struct sigaction current;
    ASSERT_EQ(sigaction(SIGSEGV, nullptr, &current), 0);
    EXPECT_NE(current.sa_handler, SIG_IGN);
    EXPECT_NE(current.sa_handler, SIG_DFL);
    EXPECT_TRUE((current.sa_flags & SA_SIGINFO) != 0);
}

}  // namespace
}  // namespace rtp_llm
