#include "gtest/gtest.h"
#define private public
#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"

#include <atomic>
#include <thread>
#include <vector>

#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

#if USING_CUDA || USING_ROCM
namespace {

bool hasGpuDeviceForTest() {
#if USING_CUDA
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
#elif USING_ROCM
    int device_count = 0;
    return hipGetDeviceCount(&device_count) == hipSuccess && device_count > 0;
#else
    return false;
#endif
}

int currentDeviceForTest() {
#if USING_CUDA
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return -1;
    }
    return device;
#elif USING_ROCM
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess) {
        return -1;
    }
    return device;
#else
    return -1;
#endif
}

}  // namespace
#endif

class CacheStoreAsyncWriterTest: public ::testing::Test {};

TEST_F(CacheStoreAsyncWriterTest, InitAndWaitBasic) {
    CacheStoreAsyncWriter writer;
    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);

    writer.init();
    ASSERT_FALSE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);

    std::atomic<int> counter{0};
    writer.submit([&counter]() { counter.fetch_add(1); });
    writer.submit([&counter]() { counter.fetch_add(1); });
    writer.submit([&counter]() { counter.fetch_add(1); });

    writer.waitAllDone();
    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);
    ASSERT_EQ(3, counter.load());
}

TEST_F(CacheStoreAsyncWriterTest, WaitAllDoneWhileIdleThrows) {
    CacheStoreAsyncWriter writer;
    ASSERT_ANY_THROW(writer.waitAllDone());
}

TEST_F(CacheStoreAsyncWriterTest, SubmitWhileIdleThrows) {
    CacheStoreAsyncWriter writer;
    ASSERT_ANY_THROW(writer.submit([]() {}));
}

TEST_F(CacheStoreAsyncWriterTest, InitWhileRunningThrows) {
    CacheStoreAsyncWriter writer;
    writer.init();

    ASSERT_ANY_THROW(writer.init());

    // Writer should still be functional after the failed second init.
    std::atomic<int> counter{0};
    writer.submit([&counter]() { counter.fetch_add(1); });
    writer.waitAllDone();
    ASSERT_EQ(1, counter.load());
}

TEST_F(CacheStoreAsyncWriterTest, InitWaitCycle) {
    CacheStoreAsyncWriter writer;
    std::vector<int>      order;

    writer.init();
    writer.submit([&]() { order.push_back(1); });
    writer.submit([&]() { order.push_back(2); });
    writer.waitAllDone();

    ASSERT_EQ(2u, order.size());

    writer.init();
    writer.submit([&]() { order.push_back(3); });
    writer.waitAllDone();

    ASSERT_EQ(3u, order.size());
    ASSERT_EQ(3, order.back());
}

TEST_F(CacheStoreAsyncWriterTest, AsyncExecution) {
    CacheStoreAsyncWriter writer;
    writer.init();

    auto              main_tid = std::this_thread::get_id();
    std::atomic<bool> different_thread{false};

    writer.submit([&]() {
        if (std::this_thread::get_id() != main_tid) {
            different_thread.store(true);
        }
    });
    writer.waitAllDone();

    ASSERT_TRUE(different_thread.load());
}

TEST_F(CacheStoreAsyncWriterTest, AsyncExecutionWithDeviceId) {
#if USING_CUDA || USING_ROCM
    if (!hasGpuDeviceForTest()) {
        GTEST_SKIP() << "No GPU device available";
    }

    CacheStoreAsyncWriter writer(0);
    writer.init();

    std::atomic<int> counter{0};
    std::atomic<int> observed_device{-1};
    writer.submit([&counter, &observed_device]() {
        observed_device.store(currentDeviceForTest(), std::memory_order_release);
        counter.fetch_add(1);
    });
    writer.waitAllDone();

    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);
    ASSERT_EQ(1, counter.load());
    ASSERT_EQ(0, observed_device.load(std::memory_order_acquire));
    ASSERT_EQ(0, writer.pending_count_.load());
#else
    GTEST_SKIP() << "GPU device pinning is unavailable in CPU-only builds";
#endif
}

TEST_F(CacheStoreAsyncWriterTest, ExceptionPropagation) {
    CacheStoreAsyncWriter writer;
    writer.init();

    writer.submit([]() { throw std::runtime_error("test error"); });

    ASSERT_THROW(writer.waitAllDone(), std::runtime_error);
    ASSERT_EQ(0, writer.pending_count_.load());

    // After exception, writer should be back in IDLE and re-initializable.
    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);
    writer.init();
    std::atomic<int> counter{0};
    writer.submit([&counter]() { counter.fetch_add(1); });
    writer.waitAllDone();
    ASSERT_EQ(1, counter.load());
}

TEST_F(CacheStoreAsyncWriterTest, FirstExceptionKeptOnMultipleFailures) {
    CacheStoreAsyncWriter writer;
    writer.init();

    writer.submit([]() { throw std::runtime_error("first"); });
    writer.submit([]() { throw std::runtime_error("second"); });

    try {
        writer.waitAllDone();
        FAIL() << "expected exception";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        ASSERT_TRUE(msg == "first" || msg == "second") << "unexpected: " << msg;
    }
}

TEST_F(CacheStoreAsyncWriterTest, WaitWithoutSubmit) {
    CacheStoreAsyncWriter writer;
    writer.init();
    writer.waitAllDone();
    ASSERT_TRUE(writer.state_ == CacheStoreAsyncWriter::State::IDLE);
}

TEST_F(CacheStoreAsyncWriterTest, ManyCycles) {
    CacheStoreAsyncWriter writer;
    std::atomic<int>      total{0};

    for (int cycle = 0; cycle < 50; ++cycle) {
        writer.init();
        for (int i = 0; i < 5; ++i) {
            writer.submit([&total]() { total.fetch_add(1); });
        }
        writer.waitAllDone();
    }
    ASSERT_EQ(250, total.load());
}

TEST_F(CacheStoreAsyncWriterTest, DoubleWaitAllDoneThrows) {
    CacheStoreAsyncWriter writer;
    writer.init();
    writer.waitAllDone();
    ASSERT_ANY_THROW(writer.waitAllDone());
}

}  // namespace rtp_llm
