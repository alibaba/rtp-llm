#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <ATen/record_function.h>
#include "gtest/gtest.h"
#include "torch/torch.h"

#include "rtp_llm/cpp/normal_engine/AsyncRunner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"

namespace rtp_llm {
namespace {

std::atomic<int>* g_record_function_callback_count = nullptr;

std::unique_ptr<at::ObserverContext> countRecordFunctionCallback(const at::RecordFunction&) {
    if (g_record_function_callback_count != nullptr) {
        g_record_function_callback_count->fetch_add(1, std::memory_order_relaxed);
    }
    return nullptr;
}

}  // namespace

class AsyncRunnerTest: public ::testing::Test {
protected:
    torch::Stream makeStream() {
        return cuda_graph::graphGetStreamFromPool(/*is_high_priority=*/true);
    }

    torch::Stream currentStream() {
        return cuda_graph::graphGetCurrentStream();
    }
};

TEST_F(AsyncRunnerTest, LaunchAndSyncBasic) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    std::atomic<int> counter{0};
    runner.launch([&counter] { counter.fetch_add(1); });
    runner.sync(currentStream());

    EXPECT_EQ(counter.load(), 1);
}

TEST_F(AsyncRunnerTest, SequentialLaunches) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    std::atomic<int> counter{0};
    constexpr int    kIterations = 10;
    for (int i = 0; i < kIterations; i++) {
        runner.launch([&counter] { counter.fetch_add(1); });
    }
    runner.sync(currentStream());

    EXPECT_EQ(counter.load(), kIterations);
}

TEST_F(AsyncRunnerTest, ExecutionOrder) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    std::vector<int> order;
    std::mutex       mu;

    for (int i = 0; i < 5; i++) {
        runner.launch([&order, &mu, i] {
            std::lock_guard<std::mutex> lk(mu);
            order.push_back(i);
        });
    }
    runner.sync(currentStream());

    ASSERT_EQ(order.size(), 5u);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(order[i], i);
    }
}

TEST_F(AsyncRunnerTest, CudaTensorWork) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    auto src = torch::ones({64}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto dst = torch::zeros({64}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    runner.launch([&src, &dst] { dst.copy_(src); });
    runner.sync(currentStream());

    auto result = dst.cpu();
    EXPECT_TRUE(torch::equal(result, torch::ones({64}, torch::kFloat32)));
}

TEST_F(AsyncRunnerTest, DoesNotPropagateRecordFunctionCallbacksToWorker) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    std::atomic<int>  callback_count{0};
    std::atomic<bool> allow_worker_ops{false};
    std::atomic<bool> worker_task_started{false};
    g_record_function_callback_count = &callback_count;
    auto handle = at::addThreadLocalCallback(at::RecordFunctionCallback(&countRecordFunctionCallback));

    {
        at::RecordFunction main_record(at::RecordScope::FUNCTION);
        main_record.before("async_runner.main_thread_positive_control");
        main_record.end();
    }
    EXPECT_GT(callback_count.load(std::memory_order_relaxed), 0);
    callback_count.store(0, std::memory_order_relaxed);

    runner.launch([&allow_worker_ops, &worker_task_started] {
        worker_task_started.store(true, std::memory_order_release);
        while (!allow_worker_ops.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        auto src = torch::ones({64}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto dst = src.narrow(0, 0, 32).clone();
        (void)dst;
    });

    while (!worker_task_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    callback_count.store(0, std::memory_order_relaxed);
    at::removeCallback(handle);
    allow_worker_ops.store(true, std::memory_order_release);
    runner.sync(currentStream());

    g_record_function_callback_count = nullptr;
    EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 0);
}

TEST_F(AsyncRunnerTest, DestructorJoinsThread) {
    auto             stream = makeStream();
    std::atomic<int> counter{0};

    {
        AsyncRunner runner(stream);
        runner.launch([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            counter.fetch_add(1);
        });
    }
    EXPECT_EQ(counter.load(), 1);
}

TEST_F(AsyncRunnerTest, SyncWithoutLaunch) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    runner.sync(currentStream());
}

TEST_F(AsyncRunnerTest, RethrowsWorkerExceptionFromSync) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    runner.launch([] { throw std::runtime_error("async failure"); });
    EXPECT_THROW(runner.sync(currentStream()), std::runtime_error);

    std::atomic<int> counter{0};
    runner.launch([&counter] { counter.fetch_add(1); });
    runner.sync(currentStream());
    EXPECT_EQ(counter.load(), 1);
}

TEST_F(AsyncRunnerTest, RethrowsWorkerExceptionFromNextLaunch) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    runner.launch([] { throw std::runtime_error("async failure"); });
    EXPECT_THROW(runner.launch([] {}), std::runtime_error);

    std::atomic<int> counter{0};
    runner.launch([&counter] { counter.fetch_add(1); });
    runner.sync(currentStream());
    EXPECT_EQ(counter.load(), 1);
}

TEST_F(AsyncRunnerTest, LaunchSyncInterleavedStress) {
    auto        stream = makeStream();
    AsyncRunner runner(stream);

    std::atomic<int> counter{0};
    constexpr int    kIterations = 100;

    for (int i = 0; i < kIterations; i++) {
        runner.launch([&counter] { counter.fetch_add(1); });
        runner.sync(currentStream());
        EXPECT_EQ(counter.load(), i + 1);
    }
}

}  // namespace rtp_llm
