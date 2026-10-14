#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

namespace rtp_llm {
namespace {
TEST(CopyStreamReuseTest, CompletedStreamsSurviveShortLivedThreads) {
    // Each worker's thread-local owner is destroyed before the next worker
    // starts. Reusing its completed stream must not alias the torch pool.
    std::set<cudaStream_t> handles;
    for (int i = 0; i < 128; ++i) {
        cudaStream_t stream = nullptr;
        std::thread  worker([&] {
            EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
            stream = getCacheCopyStream();
            EXPECT_EQ(stream, getCacheCopyStream());
            EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        });
        worker.join();
        ASSERT_NE(stream, nullptr);
        handles.insert(stream);
    }
    EXPECT_EQ(handles.size(), 1);
}

TEST(CopyStreamReuseTest, ActiveThreadsOwnDistinctStreams) {
    constexpr int             count = 16;
    std::vector<cudaStream_t> streams(count);
    std::vector<std::thread>  workers;
    std::mutex                mutex;
    std::condition_variable   ready;
    int                       arrived = 0;
    std::promise<void>        release;
    auto                      released = release.get_future().share();
    for (int i = 0; i < count; ++i) {
        workers.emplace_back([&, i] {
            EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
            streams[i] = getCacheCopyStream();
            {
                std::lock_guard<std::mutex> lock(mutex);
                ++arrived;
                ready.notify_one();
            }
            released.wait();
        });
    }
    bool all_ready;
    {
        std::unique_lock<std::mutex> lock(mutex);
        all_ready = ready.wait_for(lock, std::chrono::seconds(10), [&] { return arrived == count; });
    }
    release.set_value();
    for (auto& worker : workers) {
        worker.join();
    }
    ASSERT_TRUE(all_ready);
    EXPECT_EQ(std::set<cudaStream_t>(streams.begin(), streams.end()).size(), count);
}

TEST(CopyStreamReuseTest, PendingWorkIsNotPassedToAnotherThread) {
    struct Gate {
        std::mutex              mutex;
        std::condition_variable cv;
        bool                    released = false;
        static void CUDART_CB   wait(void* value) {
            auto&                        gate = *static_cast<Gate*>(value);
            std::unique_lock<std::mutex> lock(gate.mutex);
            gate.cv.wait(lock, [&] { return gate.released; });
        }
    } gate;
    std::promise<void> acquired, launch;
    auto               launch_ready = launch.get_future();
    cudaStream_t       busy         = nullptr;
    auto               first        = std::async(std::launch::async, [&] {
        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        busy = getCacheCopyStream();
        acquired.set_value();
        launch_ready.wait();
        EXPECT_EQ(cudaLaunchHostFunc(busy, &Gate::wait, &gate), cudaSuccess);
    });
    acquired.get_future().wait();
    // Put a distinct, completed stream in the idle pool before blocking busy.
    // This avoids creating a stream while the callback waits and makes the
    // test independent of other tests' pool contents.
    std::thread seed([&] {
        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        EXPECT_NE(getCacheCopyStream(), busy);
        EXPECT_EQ(cudaStreamSynchronize(getCacheCopyStream()), cudaSuccess);
    });
    seed.join();
    launch.set_value();
    const auto owner_done = first.wait_for(std::chrono::seconds(1));
    auto       second     = std::async(std::launch::async, [&] {
        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        auto stream = getCacheCopyStream();
        EXPECT_NE(stream, busy);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    });
    const auto copy_done  = second.wait_for(std::chrono::seconds(1));
    {
        std::lock_guard<std::mutex> lock(gate.mutex);
        gate.released = true;
        gate.cv.notify_one();
    }
    first.get();
    second.get();
    EXPECT_EQ(owner_done, std::future_status::ready);
    EXPECT_EQ(copy_done, std::future_status::ready);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}
}  // namespace
}  // namespace rtp_llm
