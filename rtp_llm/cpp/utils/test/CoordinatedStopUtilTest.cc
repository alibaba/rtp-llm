#include "rtp_llm/cpp/utils/CoordinatedStopUtil.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <string>
#include <thread>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value): name_(name) {
        const char* old = std::getenv(name);
        if (old != nullptr) {
            old_value_ = old;
        }
        if (value != nullptr) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
    }

    ~ScopedEnvVar() {
        if (old_value_.has_value()) {
            setenv(name_.c_str(), old_value_->c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

private:
    std::string                name_;
    std::optional<std::string> old_value_;
};

TEST(CoordinatedStopUtilTest, UsesDefaultAndConfiguredTimeout) {
    {
        ScopedEnvVar env(kCoordinatedStopTimeoutEnv, nullptr);
        EXPECT_EQ(coordinatedStopTimeoutMs(), kDefaultCoordinatedStopTimeoutMs);
    }
    {
        ScopedEnvVar env(kCoordinatedStopTimeoutEnv, "12345");
        EXPECT_EQ(coordinatedStopTimeoutMs(), 12345);
    }
}

TEST(CoordinatedStopUtilTest, InvalidTimeoutFallsBackToDefault) {
    for (const char* value : {"", "abc", "0", "-1", "12ms"}) {
        ScopedEnvVar env(kCoordinatedStopTimeoutEnv, value);
        EXPECT_EQ(coordinatedStopTimeoutMs(), kDefaultCoordinatedStopTimeoutMs) << value;
    }
}

TEST(CoordinatedStopUtilTest, ShortTimeoutReproducesLongEngineStep) {
    ScopedEnvVar      env(kCoordinatedStopTimeoutEnv, "10");
    std::atomic<bool> running{true};
    std::atomic<bool> acknowledged{false};

    const auto status = waitForCoordinatedStopAck(
        running,
        [&acknowledged]() { return acknowledged.load(std::memory_order_acquire); },
        coordinatedStopTimeoutMs(),
        "arm");

    EXPECT_TRUE(absl::IsDeadlineExceeded(status));
    EXPECT_EQ(status.message(), "engine loop did not acknowledge coordinated stop arm");
}

TEST(CoordinatedStopUtilTest, ConfiguredTimeoutAllowsDelayedEngineAck) {
    ScopedEnvVar      env(kCoordinatedStopTimeoutEnv, "200");
    std::atomic<bool> running{true};
    std::atomic<bool> acknowledged{false};
    std::thread       engine_loop([&acknowledged]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        acknowledged.store(true, std::memory_order_release);
    });

    const auto status = waitForCoordinatedStopAck(
        running,
        [&acknowledged]() { return acknowledged.load(std::memory_order_acquire); },
        coordinatedStopTimeoutMs(),
        "arm");
    engine_loop.join();

    EXPECT_TRUE(status.ok()) << status;
}

}  // namespace
}  // namespace rtp_llm
