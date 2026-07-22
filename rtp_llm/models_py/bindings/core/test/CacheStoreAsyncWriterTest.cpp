#include "gtest/gtest.h"
#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"

#include <atomic>
#include <stdexcept>
#include <string>
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

int gpuDeviceCountForTest() {
#if USING_CUDA
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess ? device_count : 0;
#elif USING_ROCM
    int device_count = 0;
    return hipGetDeviceCount(&device_count) == hipSuccess ? device_count : 0;
#else
    return 0;
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

bool setDeviceForTest(int device) {
#if USING_CUDA
    return cudaSetDevice(device) == cudaSuccess;
#elif USING_ROCM
    return hipSetDevice(device) == hipSuccess;
#else
    return false;
#endif
}

class ScopedDeviceResetForTest {
public:
    ScopedDeviceResetForTest(): original_device_(currentDeviceForTest()) {}
    ~ScopedDeviceResetForTest() {
        if (original_device_ >= 0) {
            setDeviceForTest(original_device_);
        }
    }

    ScopedDeviceResetForTest(const ScopedDeviceResetForTest&)            = delete;
    ScopedDeviceResetForTest& operator=(const ScopedDeviceResetForTest&) = delete;

private:
    int original_device_;
};

}  // namespace
#endif

class CacheStoreAsyncWriterTest: public ::testing::Test {};

TEST_F(CacheStoreAsyncWriterTest, InitAndWaitBasic) {
    CacheStoreAsyncWriter writer;

    writer.init();

    std::atomic<int> counter{0};
    writer.submit([&counter]() { counter.fetch_add(1); });
    writer.submit([&counter]() { counter.fetch_add(1); });
    writer.submit([&counter]() { counter.fetch_add(1); });

    writer.waitAllDone();
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
    std::atomic<int>      completed{0};

    writer.init();
    writer.submit([&]() { completed.fetch_add(1); });
    writer.submit([&]() { completed.fetch_add(1); });
    writer.waitAllDone();

    ASSERT_EQ(2, completed.load());

    writer.init();
    writer.submit([&]() { completed.fetch_add(1); });
    writer.waitAllDone();

    ASSERT_EQ(3, completed.load());
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
    const int device_count = gpuDeviceCountForTest();
    if (device_count < 2) {
        GTEST_SKIP() << "Need at least two GPU devices to prove non-default device pinning";
    }

    ScopedDeviceResetForTest reset_device;
    const int                kMainThreadDevice = device_count > 2 ? 1 : 0;
    // Two-device hosts cannot use both a nonzero parent and a non-default writer.
    // Prefer writer=1 there so the assertion is not satisfied by the runtime default.
    const int kWriterDevice = device_count > 2 ? 2 : 1;
    ASSERT_NE(kMainThreadDevice, kWriterDevice);
    ASSERT_GT(kWriterDevice, 0);
    ASSERT_TRUE(setDeviceForTest(kMainThreadDevice));
    ASSERT_EQ(kMainThreadDevice, currentDeviceForTest());

    CacheStoreAsyncWriter writer(kWriterDevice);
    writer.init();

    std::atomic<int> counter{0};
    std::atomic<int> observed_device{-1};
    writer.submit([&counter, &observed_device]() {
        observed_device.store(currentDeviceForTest(), std::memory_order_release);
        counter.fetch_add(1);
    });
    writer.waitAllDone();

    ASSERT_EQ(1, counter.load());
    ASSERT_EQ(kWriterDevice, observed_device.load(std::memory_order_acquire));
    ASSERT_EQ(kMainThreadDevice, currentDeviceForTest());
#else
    GTEST_SKIP() << "GPU device pinning is unavailable in CPU-only builds";
#endif
}

TEST_F(CacheStoreAsyncWriterTest, ExceptionPropagation) {
    CacheStoreAsyncWriter writer;
    writer.init();

    writer.submit([]() { throw std::runtime_error("test error"); });

    ASSERT_THROW(writer.waitAllDone(), std::runtime_error);

    // After exception, writer should be back in IDLE and re-initializable.
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
