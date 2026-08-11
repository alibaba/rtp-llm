#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h"

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

class FailingRpcService final: public RpcServiceImpl {
public:
    explicit FailingRpcService(std::shared_ptr<std::atomic<int>> stop_calls): stop_calls_(std::move(stop_calls)) {}

    void stop() override {
        stop_calls_->fetch_add(1, std::memory_order_relaxed);
        throw std::runtime_error("service stop failed");
    }

private:
    std::shared_ptr<std::atomic<int>> stop_calls_;
};

TEST(RtpLLMOpShutdownTest, ExplicitTimeoutRetainsAsyncStateAndRetryJoins) {
    RtpLLMOp op;

    auto completion_signal = std::make_shared<std::promise<void>>();
    auto release_signal    = std::make_shared<std::promise<void>>();
    auto release_result    = release_signal->get_future().share();
    op.grpc_shutdown_result_ = completion_signal->get_future().share();
    op.grpc_shutdown_thread_ = std::thread([completion_signal, release_result]() {
        release_result.wait();
        completion_signal->set_value();
    });

    EXPECT_ANY_THROW(op.stopWithDeadline(std::chrono::steady_clock::now()));
    EXPECT_TRUE(op.grpc_shutdown_result_.valid());
    EXPECT_TRUE(op.grpc_shutdown_thread_.joinable());
    EXPECT_FALSE(op.is_server_shutdown_.load(std::memory_order_acquire));

    release_signal->set_value();
    EXPECT_NO_THROW(op.stopWithDeadline(std::chrono::steady_clock::now() + 5s));
    EXPECT_FALSE(op.grpc_shutdown_result_.valid());
    EXPECT_FALSE(op.grpc_shutdown_thread_.joinable());
    EXPECT_TRUE(op.is_server_shutdown_.load(std::memory_order_acquire));
}

TEST(RtpLLMOpShutdownTest, ServiceStopFailureIsTerminalAndRunsOnlyOnce) {
    RtpLLMOp op;
    auto     stop_calls = std::make_shared<std::atomic<int>>(0);
    op.model_rpc_service_ = std::make_unique<FailingRpcService>(stop_calls);

    op.startServiceStop();
    ASSERT_EQ(op.service_stop_result_.wait_for(5s), std::future_status::ready);
    EXPECT_ANY_THROW(op.waitForServiceStop(std::chrono::steady_clock::now() + 5s));
    EXPECT_EQ(stop_calls->load(std::memory_order_relaxed), 1);
    EXPECT_TRUE(op.service_stop_result_.valid());
    EXPECT_FALSE(op.service_stop_thread_.joinable());

    op.startServiceStop();
    EXPECT_ANY_THROW(op.waitForServiceStop(std::chrono::steady_clock::now() + 5s));
    EXPECT_EQ(stop_calls->load(std::memory_order_relaxed), 1);

    op.service_stop_result_ = {};
    op.model_rpc_service_.reset();
    op.is_server_shutdown_.store(true, std::memory_order_release);
}

TEST(RtpLLMOpShutdownTest, ForceStopAbortsWhenServiceWorkerMissesDeadline) {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_EXIT(
        {
            auto* op = new RtpLLMOp();
            auto  pending_signal = std::make_shared<std::promise<void>>();
            op->service_stop_result_ = pending_signal->get_future().share();
            op->service_stop_thread_ = std::thread([pending_signal]() {
                (void)pending_signal;
                std::mutex              mutex;
                std::condition_variable condition;
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock, []() { return false; });
            });

            op->forceStopNoThrow(std::chrono::steady_clock::now() + 20ms);
            std::_Exit(1);
        },
        ::testing::KilledBySignal(SIGABRT),
        "");
}

}  // namespace
}  // namespace rtp_llm
