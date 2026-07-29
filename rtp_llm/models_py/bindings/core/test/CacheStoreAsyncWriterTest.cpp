#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
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

static CacheConfig makeWriterTestCacheConfig(size_t tokens_per_block) {
    return test::makeSimpleMhaCacheConfig(
        /*layer_num=*/1, /*block_num=*/1, tokens_per_block, DataType::TYPE_FP16);
}

class NoopCacheStore final: public CacheStore {
public:
    struct StoreRecord {
        std::string              request_id;
        std::vector<std::string> block_keys;
    };

    void store(const std::shared_ptr<RequestBlockBuffer>& buf, CacheStoreStoreDoneCallback callback) override {
        if (buf) {
            StoreRecord record;
            record.request_id = buf->getRequestId();
            for (const auto& [key, block] : buf->getBlocks()) {
                record.block_keys.push_back(key);
            }
            std::lock_guard<std::mutex> lock(records_mutex_);
            store_records_.push_back(std::move(record));
        }
        if (callback) {
            callback(true, CacheStoreErrorCode::None);
        }
    }

    std::vector<StoreRecord> storeRecords() const {
        std::lock_guard<std::mutex> lock(records_mutex_);
        return store_records_;
    }

    void load(const std::shared_ptr<RequestBlockBuffer>&,
              CacheStoreLoadDoneCallback callback,
              const std::string&,
              uint32_t,
              uint32_t,
              uint32_t,
              int,
              int) override {
        if (callback) {
            callback(true, CacheStoreErrorCode::None);
        }
    }

    std::shared_ptr<LoadContext> loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&,
                                             const std::string&,
                                             uint32_t,
                                             uint32_t,
                                             int64_t,
                                             LoadContext::CheckCancelFunc,
                                             int,
                                             int) override {
        return nullptr;
    }

    std::shared_ptr<StoreContext> storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&,
                                               int64_t) override {
        return nullptr;
    }

    std::shared_ptr<RemoteStoreTask>
    submitRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>&,
                          const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>&,
                          RemoteStoreTask::CheckCancelFunc) override {
        return nullptr;
    }

    void releaseRemoteStoreTask(const std::shared_ptr<RemoteStoreTask>&) override {}

    bool regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>&) override {
        return true;
    }

    std::shared_ptr<BlockBuffer> findUserBuffer(const std::string&) override {
        return nullptr;
    }

    const std::shared_ptr<MemoryUtil>& getMemoryUtil() const override {
        return null_memory_util_;
    }

    void debugInfo() override {}

private:
    std::shared_ptr<MemoryUtil> null_memory_util_;
    mutable std::mutex          records_mutex_;
    std::vector<StoreRecord>    store_records_;
};

class CacheStoreAsyncWriterTest: public ::testing::Test {
protected:
    void SetUp() override {
        cache_manager_ = std::make_shared<KVCacheManager>(makeWriterTestCacheConfig(/*tokens_per_block=*/1),
                                                          /*warmup=*/true);
        cache_store_   = std::make_shared<NoopCacheStore>();
        cache_manager_->setCacheStore(cache_store_);
        writer_ = std::make_unique<CacheStoreAsyncWriter>(/*device_id=*/-1, cache_manager_);
    }

    std::shared_ptr<KVCacheManager>        cache_manager_;
    std::shared_ptr<NoopCacheStore>        cache_store_;
    std::unique_ptr<CacheStoreAsyncWriter> writer_;
};

TEST_F(CacheStoreAsyncWriterTest, InitAndWaitBasic) {
    writer_->init();

    std::atomic<int> counter{0};
    writer_->submit([&counter]() { counter.fetch_add(1); });
    writer_->submit([&counter]() { counter.fetch_add(1); });
    writer_->submit([&counter]() { counter.fetch_add(1); });

    writer_->waitAllDone();
    ASSERT_EQ(3, counter.load());
}

TEST_F(CacheStoreAsyncWriterTest, ThreadPoolIsCreatedLazilyAndReusedAcrossCycles) {
    // A writer exists per PyWrappedModel (plus one per MTP draft module), but only
    // PD-separation prefill ever opens a cycle. Constructing one must not start worker
    // threads, and cycles must share one pool rather than rebuilding it per forward.
    // Private member reached via -fno-access-control, like the submit() cases below.
    ASSERT_EQ(writer_->thread_pool_, nullptr);

    writer_->init();
    const auto* pool_after_first_init = writer_->thread_pool_.get();
    ASSERT_NE(pool_after_first_init, nullptr);
    writer_->waitAllDone();

    writer_->init();
    EXPECT_EQ(writer_->thread_pool_.get(), pool_after_first_init);
    writer_->waitAllDone();
}

TEST_F(CacheStoreAsyncWriterTest, WaitAllDoneWhileIdleThrows) {
    ASSERT_ANY_THROW(writer_->waitAllDone());
}

TEST_F(CacheStoreAsyncWriterTest, SubmitWhileIdleThrows) {
    ASSERT_ANY_THROW(writer_->submit([]() {}));
}

TEST_F(CacheStoreAsyncWriterTest, InitWithoutCacheStoreFailsAndCanRetryAfterInjection) {
    // Pin fail-fast semantics: a rollback switch pre-set in the calling
    // environment must not turn this contract test into a silent pass.
    autil::EnvGuard force_fail_fast("CACHE_STORE_SKIP_WRITE_WHEN_UNREADY", "0");

    auto manager = std::make_shared<KVCacheManager>(makeWriterTestCacheConfig(/*tokens_per_block=*/1), /*warmup=*/true);
    CacheStoreAsyncWriter writer(/*device_id=*/2, manager, /*cache_model_id=*/19);

    try {
        writer.init();
        FAIL() << "expected missing CacheStore to fail initialization";
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("CacheStore"), std::string::npos);
        EXPECT_NE(message.find("initCacheStore"), std::string::npos);
        EXPECT_NE(message.find("model_id=19"), std::string::npos);
        EXPECT_NE(message.find("device_id=2"), std::string::npos);
    }

    EXPECT_EQ(CacheStoreAsyncWriter::State::IDLE, writer.state_);
    EXPECT_EQ(nullptr, writer.active_cache_store_);

    manager->setCacheStore(cache_store_);
    ASSERT_NO_THROW(writer.init());
    ASSERT_NO_THROW(writer.waitAllDone());
}

TEST_F(CacheStoreAsyncWriterTest, MissingCacheStoreRollbackSwitchSkipsCycleWrites) {
    // Outer guard pins the default fail-fast semantics (and restores the caller's
    // original value on exit); the inner guard flips the rollback switch on for
    // the degraded-skip section only.
    autil::EnvGuard force_fail_fast("CACHE_STORE_SKIP_WRITE_WHEN_UNREADY", "0");

    auto manager = std::make_shared<KVCacheManager>(makeWriterTestCacheConfig(/*tokens_per_block=*/1), /*warmup=*/true);
    CacheStoreAsyncWriter writer(/*device_id=*/-1, manager);

    torch_ext::PyCacheStoreInputs inputs;
    torch_ext::LayerKVCache       layer_cache;
    layer_cache.layer_id = 0;
    layer_cache.tag      = "full";

    {
        // Degraded-skip mode: the cycle is admitted, write() drops the work without side
        // effects, and the cycle drains clean. The early return precedes write()'s
        // CUDA/ROCm build guard.
        autil::EnvGuard rollback_switch("CACHE_STORE_SKIP_WRITE_WHEN_UNREADY", "1");
        ASSERT_NO_THROW(writer.init());
        EXPECT_TRUE(writer.skip_cycle_writes_);
        ASSERT_NO_THROW(writer.write(inputs, layer_cache));
        EXPECT_EQ(0, writer.pending_count_.load());
        ASSERT_NO_THROW(writer.waitAllDone());
        EXPECT_EQ(CacheStoreAsyncWriter::State::IDLE, writer.state_);
    }

    // Switch cleared: the default fail-fast contract is back.
    ASSERT_ANY_THROW(writer.init());

    // A later cycle with an injected CacheStore must not inherit the skip flag.
    manager->setCacheStore(cache_store_);
    ASSERT_NO_THROW(writer.init());
    EXPECT_FALSE(writer.skip_cycle_writes_);
    ASSERT_NO_THROW(writer.waitAllDone());
}

TEST_F(CacheStoreAsyncWriterTest, WriteOutsideActiveCycleThrows) {
    torch_ext::PyCacheStoreInputs inputs;
    torch_ext::LayerKVCache       layer_cache;
    layer_cache.layer_id = 4;
    layer_cache.tag      = "linear";

    // CUDA/ROCm builds must reject an out-of-cycle write() with the RUNNING-cycle
    // contract message; CPU-only builds hit the build guard before any state check.
#if USING_CUDA || USING_ROCM
    const char* expected_message = "requires an active RUNNING forward cycle";
#else
    const char* expected_message = "requires a CUDA or ROCm build";
#endif

    try {
        writer_->write(inputs, layer_cache);
        FAIL() << "expected write() before init() to fail";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(expected_message), std::string::npos);
    }

    writer_->init();
    writer_->waitAllDone();
    try {
        writer_->write(inputs, layer_cache);
        FAIL() << "expected write() after waitAllDone() to fail";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(expected_message), std::string::npos);
    }
}

TEST_F(CacheStoreAsyncWriterTest, InitWhileRunningThrows) {
    writer_->init();

    ASSERT_ANY_THROW(writer_->init());

    // Writer should still be functional after the failed second init.
    std::atomic<int> counter{0};
    writer_->submit([&counter]() { counter.fetch_add(1); });
    writer_->waitAllDone();
    ASSERT_EQ(1, counter.load());
}

TEST_F(CacheStoreAsyncWriterTest, InitWaitCycle) {
    std::vector<int> order;
    std::mutex       order_mutex;

    writer_->init();
    writer_->submit([&]() {
        std::lock_guard<std::mutex> lock(order_mutex);
        order.push_back(1);
    });
    writer_->submit([&]() {
        std::lock_guard<std::mutex> lock(order_mutex);
        order.push_back(2);
    });
    writer_->waitAllDone();

    ASSERT_EQ(2u, order.size());

    writer_->init();
    writer_->submit([&]() {
        std::lock_guard<std::mutex> lock(order_mutex);
        order.push_back(3);
    });
    writer_->waitAllDone();

    ASSERT_EQ(3u, order.size());
    ASSERT_EQ(3, order.back());
}

TEST_F(CacheStoreAsyncWriterTest, AsyncExecution) {
    writer_->init();

    auto              main_tid = std::this_thread::get_id();
    std::atomic<bool> different_thread{false};

    writer_->submit([&]() {
        if (std::this_thread::get_id() != main_tid) {
            different_thread.store(true);
        }
    });
    writer_->waitAllDone();

    ASSERT_TRUE(different_thread.load());
}

TEST_F(CacheStoreAsyncWriterTest, WaitDrainsAdmittedTaskAndRejectsLateWrite) {
    std::mutex              task_mutex;
    std::condition_variable task_cv;
    bool                    task_started  = false;
    bool                    release_task  = false;
    std::atomic<bool>       wait_returned = false;
    std::exception_ptr      wait_exception;

    writer_->init();
    writer_->submit([&]() {
        std::unique_lock<std::mutex> lock(task_mutex);
        task_started = true;
        task_cv.notify_all();
        task_cv.wait(lock, [&]() { return release_task; });
    });

    {
        std::unique_lock<std::mutex> lock(task_mutex);
        task_cv.wait(lock, [&]() { return task_started; });
    }

    std::thread waiter([&]() {
        try {
            writer_->waitAllDone();
            wait_returned.store(true, std::memory_order_release);
        } catch (...) {
            wait_exception = std::current_exception();
        }
    });

    bool       observed_draining = false;
    const auto deadline          = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lock(writer_->state_mutex_);
            observed_draining = writer_->state_ == CacheStoreAsyncWriter::State::DRAINING;
        }
        if (observed_draining) {
            break;
        }
        std::this_thread::yield();
    }

    if (observed_draining) {
        EXPECT_FALSE(wait_returned.load(std::memory_order_acquire));
        torch_ext::PyCacheStoreInputs inputs;
        torch_ext::LayerKVCache       layer_cache;
        layer_cache.layer_id = 5;
        layer_cache.tag      = "linear";
        EXPECT_THROW(writer_->write(inputs, layer_cache), std::runtime_error);
    }

    {
        std::lock_guard<std::mutex> lock(task_mutex);
        release_task = true;
    }
    task_cv.notify_all();
    waiter.join();

    ASSERT_TRUE(observed_draining);
    ASSERT_FALSE(wait_exception);
    EXPECT_TRUE(wait_returned.load(std::memory_order_acquire));
    EXPECT_EQ(0, writer_->pending_count_.load());
    EXPECT_EQ(CacheStoreAsyncWriter::State::IDLE, writer_->state_);
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

    CacheStoreAsyncWriter writer(kWriterDevice, cache_manager_);
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

TEST_F(CacheStoreAsyncWriterTest, WriteSuccessPathDeliversBlocksAndDrainsToIdle) {
#if USING_CUDA || USING_ROCM
    if (gpuDeviceCountForTest() < 1) {
        GTEST_SKIP() << "write() success path needs a GPU device for event creation";
    }

    // One request, one block, matching the fixture config
    // (makeWriterTestCacheConfig: 1 layer, 1 block, 1 token/block, FP16 MHA).
    torch_ext::PyCacheStoreInputs inputs;
    inputs.input_lengths_host    = torch::tensor({1}, torch::kInt32);
    inputs.prefix_lengths_host   = torch::tensor({0}, torch::kInt32);
    inputs.host_kv_cache_offset  = torch::tensor({{0}}, torch::kInt32);
    inputs.request_id            = torch::tensor({int64_t(42)}, torch::kInt64);
    inputs.request_pd_separation = torch::tensor({true}, torch::kBool);
    inputs.cache_keys            = torch::tensor({{int64_t(100)}}, torch::kInt64);

    torch_ext::LayerKVCache layer_cache;
    layer_cache.kv_cache_base =
        torch::zeros({1, 2, 1, 1, 1}, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    layer_cache.seq_size_per_block = 1;
    layer_cache.layer_id           = 0;
    layer_cache.tag                = "default";

    writer_->init();
    ASSERT_NO_THROW(writer_->write(inputs, layer_cache));
    ASSERT_NO_THROW(writer_->waitAllDone());

    EXPECT_EQ(0, writer_->pending_count_.load());
    EXPECT_EQ(CacheStoreAsyncWriter::State::IDLE, writer_->state_);

    const auto records = cache_store_->storeRecords();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records.front().request_id, "42");
    const auto  cache_key = makeCacheKey(/*model_id=*/0, "100", /*layer_id=*/0, "default");
    const auto& keys      = records.front().block_keys;
    EXPECT_NE(std::find(keys.begin(), keys.end(), "k_" + cache_key), keys.end());
    EXPECT_NE(std::find(keys.begin(), keys.end(), "v_" + cache_key), keys.end());

    // A drained writer must accept the next forward cycle cleanly.
    ASSERT_NO_THROW(writer_->init());
    ASSERT_NO_THROW(writer_->waitAllDone());
#else
    GTEST_SKIP() << "write() requires a CUDA or ROCm build";
#endif
}

TEST_F(CacheStoreAsyncWriterTest, SelectsRequestedMtpCacheConfig) {
    auto main_config = makeWriterTestCacheConfig(/*tokens_per_block=*/1);
    main_config.mtp_sub_configs.push_back(
        std::make_shared<CacheConfig>(makeWriterTestCacheConfig(/*tokens_per_block=*/2)));
    main_config.mtp_sub_configs.push_back(
        std::make_shared<CacheConfig>(makeWriterTestCacheConfig(/*tokens_per_block=*/3)));
    auto cache_manager = std::make_shared<KVCacheManager>(main_config, /*warmup=*/true);

    CacheStoreAsyncWriter writer(
        /*device_id=*/-1, cache_manager, /*cache_model_id=*/7, /*mtp_cache_config_index=*/1);

    EXPECT_EQ(writer.cache_manager_, cache_manager);
    EXPECT_EQ(writer.cache_config_->seq_size_per_block, 3u);
    EXPECT_EQ(writer.cache_model_id_, 7);
    EXPECT_EQ(writer.cp_rank_, 0);
    EXPECT_EQ(writer.cp_size_, 1);
}

TEST_F(CacheStoreAsyncWriterTest, ExceptionPropagation) {
    writer_->init();

    writer_->submit([]() { throw std::runtime_error("test error"); });

    ASSERT_THROW(writer_->waitAllDone(), std::runtime_error);
    ASSERT_EQ(0, writer_->pending_count_.load());
    ASSERT_EQ(CacheStoreAsyncWriter::State::IDLE, writer_->state_);

    // After exception, writer should be back in IDLE and re-initializable.
    writer_->init();
    std::atomic<int> counter{0};
    writer_->submit([&counter]() { counter.fetch_add(1); });
    writer_->waitAllDone();
    ASSERT_EQ(1, counter.load());
}

TEST_F(CacheStoreAsyncWriterTest, OneOfConcurrentExceptionsIsPropagated) {
    writer_->init();

    std::atomic<int> executed{0};
    writer_->submit([&executed]() {
        executed.fetch_add(1);
        throw std::runtime_error("first");
    });
    writer_->submit([&executed]() {
        executed.fetch_add(1);
        throw std::runtime_error("second");
    });

    try {
        writer_->waitAllDone();
        FAIL() << "expected exception";
    } catch (const std::runtime_error& e) {
        // The retained exception follows observation order, not submission order.
        std::string msg = e.what();
        ASSERT_TRUE(msg == "first" || msg == "second") << "unexpected: " << msg;
    }
    EXPECT_EQ(2, executed.load());
    EXPECT_EQ(0, writer_->pending_count_.load());
}

TEST_F(CacheStoreAsyncWriterTest, WaitWithoutSubmit) {
    writer_->init();
    writer_->waitAllDone();
}

TEST_F(CacheStoreAsyncWriterTest, ManyCycles) {
    std::atomic<int> total{0};

    for (int cycle = 0; cycle < 50; ++cycle) {
        writer_->init();
        for (int i = 0; i < 5; ++i) {
            writer_->submit([&total]() { total.fetch_add(1); });
        }
        writer_->waitAllDone();
    }
    ASSERT_EQ(250, total.load());
}

TEST_F(CacheStoreAsyncWriterTest, DoubleWaitAllDoneThrows) {
    writer_->init();
    writer_->waitAllDone();
    ASSERT_ANY_THROW(writer_->waitAllDone());
}

}  // namespace rtp_llm
