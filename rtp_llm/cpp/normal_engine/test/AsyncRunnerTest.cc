#include <atomic>
#include <chrono>
#include <future>
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

// Keep a real stream tail pending independently of the worker's CPU return.
// The callback uses no device API; every exit path releases it before joining.
class PendingStreamTail {
public:
    explicit PendingStreamTail(torch::Stream stream): stream_(stream) {
        stream_.synchronize();
    }

    ~PendingStreamTail() {
        release();
        stream_.synchronize();
    }

    void enqueue() {
#if USING_ROCM
        const auto status = hipLaunchHostFunc(cuda_graph::toGraphStream(stream_).stream(), waitForRelease, this);
        if (status != hipSuccess) {
            throw std::runtime_error(hipGetErrorString(status));
        }
#else
        const auto status = cudaLaunchHostFunc(cuda_graph::toGraphStream(stream_).stream(), waitForRelease, this);
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }
#endif
    }

    void release() {
        released_.store(true, std::memory_order_release);
    }

    bool timedOut() const {
        return timed_out_.load(std::memory_order_acquire);
    }

private:
    static void waitForRelease(void* raw) {
        auto&      self     = *static_cast<PendingStreamTail*>(raw);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!self.released_.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() >= deadline) {
                self.timed_out_.store(true, std::memory_order_release);
                return;
            }
            std::this_thread::yield();
        }
    }

    torch::Stream     stream_;
    std::atomic<bool> released_{false};
    std::atomic<bool> timed_out_{false};
};

}  // namespace

class AsyncRunnerTest: public ::testing::Test {
protected:
    torch::Stream makeStream() {
        return cuda_graph::graphGetStreamFromPool(/*is_high_priority=*/true);
    }

    torch::Stream currentStream() {
        return cuda_graph::graphGetCurrentStream();
    }

    void expectFailureDrainsTail(bool observe_from_launch) {
        auto               stream = makeStream();
        AsyncRunner        runner(stream);
        PendingStreamTail  pending(stream);
        auto               tail            = cuda_graph::makeGraphEvent();
        auto               source_owner    = std::make_shared<int>(17);
        std::weak_ptr<int> source_lifetime = source_owner;
        std::atomic<bool>  replacement_ran{false};

        runner.launch([&pending, &tail, source_owner] {
            (void)source_owner;
            pending.enqueue();
            tail.record(cuda_graph::graphGetCurrentStream());
            throw std::runtime_error("failure after enqueue");
        });
        source_owner.reset();

        auto observation = std::async(std::launch::async, [&] {
            cuda_graph::setDevice(static_cast<int>(stream.device_index()));
            try {
                if (observe_from_launch) {
                    runner.launch([&replacement_ran] { replacement_ran.store(true); });
                } else {
                    runner.sync(currentStream());
                }
            } catch (const std::runtime_error& error) {
                return std::make_pair(std::string(error.what()), tail.query());
            }
            return std::make_pair(std::string("exception missing"), tail.query());
        });

        EXPECT_EQ(observation.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
        EXPECT_FALSE(source_lifetime.expired());
        pending.release();
        const auto result = observation.get();
        EXPECT_EQ(result.first, "failure after enqueue");
        EXPECT_TRUE(result.second);
        EXPECT_TRUE(source_lifetime.expired());
        EXPECT_FALSE(replacement_ran.load());
        EXPECT_FALSE(pending.timedOut());

        runner.launch([&replacement_ran] { replacement_ran.store(true); });
        runner.sync(currentStream());
        EXPECT_TRUE(replacement_ran.load());
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

    auto input_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    input_ready_event->record(currentStream());
    runner.launch([&src, &dst, input_ready_event] {
        input_ready_event->block(cuda_graph::graphGetCurrentStream());
        dst.copy_(src);
    });
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

TEST_F(AsyncRunnerTest, FailedTaskGpuTailAndInputsCompleteBeforeSyncThrows) {
    expectFailureDrainsTail(/*observe_from_launch=*/false);
}

TEST_F(AsyncRunnerTest, FailedTaskGpuTailAndInputsCompleteBeforeNextLaunchThrows) {
    expectFailureDrainsTail(/*observe_from_launch=*/true);
}

TEST_F(AsyncRunnerTest, FailedTaskGpuTailAndInputsCompleteBeforeDestructorReturns) {
    auto               stream = makeStream();
    auto               runner = std::make_unique<AsyncRunner>(stream);
    PendingStreamTail  pending(stream);
    auto               tail            = cuda_graph::makeGraphEvent();
    auto               source_owner    = std::make_shared<int>(17);
    std::weak_ptr<int> source_lifetime = source_owner;

    runner->launch([&pending, &tail, source_owner] {
        (void)source_owner;
        pending.enqueue();
        tail.record(cuda_graph::graphGetCurrentStream());
        throw std::runtime_error("unobserved failure after enqueue");
    });
    source_owner.reset();
    auto observation = std::async(std::launch::async, [&] {
        cuda_graph::setDevice(static_cast<int>(stream.device_index()));
        runner.reset();
        return tail.query();
    });

    EXPECT_EQ(observation.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
    EXPECT_FALSE(source_lifetime.expired());
    pending.release();
    EXPECT_TRUE(observation.get());
    EXPECT_TRUE(source_lifetime.expired());
    EXPECT_FALSE(pending.timedOut());
}

TEST_F(AsyncRunnerTest, SuccessfulSyncDoesNotDrainGpuTailOnHost) {
    auto              stream      = makeStream();
    auto              wait_stream = makeStream();
    AsyncRunner       runner(stream);
    PendingStreamTail pending(stream);
    auto              tail = cuda_graph::makeGraphEvent();
    runner.launch([&pending, &tail] {
        pending.enqueue();
        tail.record(cuda_graph::graphGetCurrentStream());
    });
    auto observation = std::async(std::launch::async, [&] {
        cuda_graph::setDevice(static_cast<int>(stream.device_index()));
        runner.sync(wait_stream);
        return tail.query();
    });

    const auto status = observation.wait_for(std::chrono::seconds(1));
    EXPECT_EQ(status, std::future_status::ready);
    // Check before releasing the gate, so a host drain cannot pass as async.
    if (status == std::future_status::ready) {
        EXPECT_FALSE(observation.get());
    }
    pending.release();
    if (status != std::future_status::ready) {
        observation.get();
    }
    wait_stream.synchronize();
    EXPECT_FALSE(pending.timedOut());
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
