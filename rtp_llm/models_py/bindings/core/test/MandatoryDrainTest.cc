#include "rtp_llm/models_py/bindings/MandatoryDrain.h"

#include <atomic>
#include <csignal>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>

#include <gtest/gtest.h>

namespace rtp_llm::test {

TEST(MandatoryDrainTest, SubmissionFailureWaitsForDrainBeforeRethrow) {
    std::promise<void> drain_entered;
    std::promise<void> release_drain;
    auto               release = release_drain.get_future().share();
    std::atomic<bool>   caught{false};

    std::thread caller([&]() {
        try {
            runWithMandatoryDrain(
                []() { throw std::runtime_error("submit failed"); },
                [&]() {
                    drain_entered.set_value();
                    release.wait();
                    return true;
                },
                []() {});
        } catch (const std::runtime_error&) {
            caught.store(true, std::memory_order_release);
        }
    });

    const auto drain_status = drain_entered.get_future().wait_for(std::chrono::seconds(1));
    EXPECT_EQ(drain_status, std::future_status::ready);
    if (drain_status == std::future_status::ready) {
        EXPECT_FALSE(caught.load(std::memory_order_acquire));
    }
    release_drain.set_value();
    caller.join();
    EXPECT_TRUE(caught.load(std::memory_order_acquire));
}

TEST(MandatoryDrainTest, SuccessfulSubmissionStillDrains) {
    bool drained = false;
    runWithMandatoryDrain([]() {}, [&]() {
        drained = true;
        return true;
    }, []() {});
    EXPECT_TRUE(drained);
}

TEST(MandatoryDrainTest, DrainFailureAborts) {
    EXPECT_EXIT(runWithMandatoryDrain([]() { throw std::runtime_error("submit failed"); },
                                     []() { return false; },
                                     []() {}),
                ::testing::KilledBySignal(SIGABRT),
                "");
}

TEST(MandatoryDrainTest, DrainExceptionAborts) {
    EXPECT_EXIT(runWithMandatoryDrain([]() {},
                                     []() -> bool { throw std::runtime_error("drain failed"); },
                                     []() {}),
                ::testing::KilledBySignal(SIGABRT),
                "");
}

}  // namespace rtp_llm::test
