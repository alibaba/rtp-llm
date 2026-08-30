#include "gtest/gtest.h"

#include <atomic>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/models_py/bindings/core/CacheStoreAsyncWriter.h"

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

static CacheConfig makeWriterTestCacheConfig(const std::string& tag, size_t kv_stride) {
    CacheConfig config;
    config.layer_num                 = 1;
    config.layer_all_num             = 1;
    config.block_num                 = 1;
    config.seq_size_per_block        = 1;
    config.kernel_seq_size_per_block = 1;
    config.kv_block_stride_bytes     = kv_stride;

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->tag                = tag;
    spec->seq_size_per_block = 1;

    GroupBase group;
    group.tag                       = tag;
    group.spec                      = spec;
    group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.layer_ids                 = {0};
    group.block_num                 = 1;
    group.seq_size_per_block        = 1;
    group.kernel_seq_size_per_block = 1;
    group.kv_block_stride_bytes     = kv_stride;

    config.setTopology({std::move(group)}, {{0, {tag}}});
    return config;
}

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
    std::vector<int>      order;
    std::mutex            order_mutex;

    writer.init();
    writer.submit([&]() {
        std::lock_guard<std::mutex> lock(order_mutex);
        order.push_back(1);
    });
    writer.submit([&]() {
        std::lock_guard<std::mutex> lock(order_mutex);
        order.push_back(2);
    });
    writer.waitAllDone();

    ASSERT_EQ(2u, order.size());

    writer.init();
    writer.submit([&]() {
        std::lock_guard<std::mutex> lock(order_mutex);
        order.push_back(3);
    });
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

TEST_F(CacheStoreAsyncWriterTest, SelectsRequestedMtpCacheConfig) {
    auto main_config = makeWriterTestCacheConfig("main", /*kv_stride=*/16);
    main_config.mtp_sub_configs.push_back(
        std::make_shared<CacheConfig>(makeWriterTestCacheConfig("draft", /*kv_stride=*/32)));
    auto cache_manager = std::make_shared<KVCacheManager>(main_config, /*warmup=*/true);

    CacheStoreAsyncWriter writer(
        /*device_id=*/-1, cache_manager, /*cache_model_id=*/7, /*mtp_cache_config_index=*/0);

    EXPECT_EQ(writer.cache_manager_, cache_manager);
    EXPECT_EQ(writer.cache_config_->tagForGroup(0), "draft");
    EXPECT_EQ(writer.cache_model_id_, 7);
    EXPECT_EQ(writer.cp_rank_, 0);
    EXPECT_EQ(writer.cp_size_, 1);
}

TEST_F(CacheStoreAsyncWriterTest, UsesExplicitForwardCpTopology) {
    CacheStoreAsyncWriter writer(/*device_id=*/-1,
                                 /*cache_manager=*/nullptr,
                                 /*cache_model_id=*/0,
                                 /*mtp_cache_config_index=*/std::nullopt,
                                 /*forward_cp_rank=*/1,
                                 /*forward_cp_size=*/2);

    EXPECT_EQ(writer.cp_rank_, 1);
    EXPECT_EQ(writer.cp_size_, 2);
}

TEST_F(CacheStoreAsyncWriterTest, ExceptionPropagation) {
    CacheStoreAsyncWriter writer;
    writer.init();

    writer.submit([]() { throw std::runtime_error("test error"); });

    ASSERT_THROW(writer.waitAllDone(), std::runtime_error);
    ASSERT_EQ(0, writer.pending_count_.load());
    ASSERT_EQ(CacheStoreAsyncWriter::State::IDLE, writer.state_);

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
