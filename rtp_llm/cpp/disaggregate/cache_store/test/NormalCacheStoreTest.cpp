#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "autil/NetUtil.h"
#include "autil/EnvUtil.h"
#include "ATen/cuda/CUDAContext.h"
#include "c10/cuda/CUDAGuard.h"
#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace rtp_llm {

class CheckpointMemoryUtil: public MemoryUtil {
public:
    bool regUserMr(void*, uint64_t, bool, uint64_t) override {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back("reg_mr");
        return available_;
    }
    bool deregUserMr(void*, bool) override {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back("dereg_mr");
        return available_;
    }
    bool isMemoryMr(void*, uint64_t, bool, bool) override {
        return isAvailable();
    }
    bool findMemoryMr(void*, void*, uint64_t, bool, bool) override {
        return isAvailable();
    }
    bool isRdmaMode() override {
        return false;
    }
    bool teardownForCheckpoint() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!available_) {
            return true;
        }
        available_ = false;
        teardown_count_++;
        events_.push_back("teardown");
        return true;
    }
    bool rebuildAfterRestore() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (available_) {
            return true;
        }
        rebuild_count_++;
        if (fail_rebuild_) {
            return false;
        }
        available_ = true;
        events_.push_back("rebuild");
        return true;
    }
    bool isAvailable() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return available_;
    }

    void setFailRebuild(bool fail) {
        std::lock_guard<std::mutex> lock(mutex_);
        fail_rebuild_ = fail;
    }
    int teardownCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return teardown_count_;
    }
    int rebuildCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return rebuild_count_;
    }
    std::vector<std::string> events() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_;
    }

private:
    mutable std::mutex       mutex_;
    bool                     available_{true};
    bool                     fail_rebuild_{false};
    int                      teardown_count_{0};
    int                      rebuild_count_{0};
    std::vector<std::string> events_;
};

class NormalCacheStoreCheckpointTest: public ::testing::Test {
protected:
    void SetUp() override {
        memory_util_ = std::make_shared<CheckpointMemoryUtil>();
        CacheStoreInitParams params;
        params.listen_port   = autil::NetUtil::randomPort();
        params.rdma_mode     = false;
        params.enable_metric = false;
        params.memory_util   = memory_util_;
        cache_store_         = NormalCacheStore::createNormalCacheStore(params);
        ASSERT_NE(cache_store_, nullptr);
    }

    void expectEmptyStore(bool expected_ok, CacheStoreErrorCode expected_error) {
        bool callback_called = false;
        auto request         = std::make_shared<RequestBlockBuffer>("checkpoint-empty-request");
        cache_store_->store(request, [&](bool ok, CacheStoreErrorCode error) {
            callback_called = true;
            EXPECT_EQ(ok, expected_ok);
            EXPECT_EQ(error, expected_error);
        });
        EXPECT_TRUE(callback_called);
    }

    std::shared_ptr<CheckpointMemoryUtil> memory_util_;
    std::shared_ptr<NormalCacheStore>     cache_store_;
};

TEST_F(NormalCacheStoreCheckpointTest, TwoCheckpointCyclesKeepIdentityAndAreIdempotent) {
    auto*                       identity  = cache_store_.get();
    std::shared_ptr<CacheStore> lifecycle = cache_store_;
    std::vector<std::string>    expected_events;

    for (int cycle = 1; cycle <= 2; ++cycle) {
        std::vector<std::thread> threads;
        std::atomic<int>         successful_calls{0};
        for (int i = 0; i < 8; ++i) {
            threads.emplace_back([&]() { successful_calls += lifecycle->beginCheckpointDrain(); });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        EXPECT_EQ(successful_calls.load(), 8);
        EXPECT_FALSE(lifecycle->isAvailable());
        expectEmptyStore(false, CacheStoreErrorCode::PushWorkerItemFailed);

        threads.clear();
        successful_calls = 0;
        for (int i = 0; i < 8; ++i) {
            threads.emplace_back([&]() { successful_calls += lifecycle->teardownForCheckpoint(); });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        EXPECT_EQ(successful_calls.load(), 8);
        EXPECT_EQ(memory_util_->teardownCount(), cycle - 1);
        EXPECT_TRUE(memory_util_->isAvailable());

        // Level 3 stops transport first. The MR owner must remain usable until
        // the controller has deregistered every KV MR.
        EXPECT_TRUE(memory_util_->deregUserMr(reinterpret_cast<void*>(0x1000 + cycle), true));
        expected_events.push_back("dereg_mr");
        EXPECT_TRUE(lifecycle->teardownMemoryOwnerAfterMrDereg());
        EXPECT_TRUE(lifecycle->teardownMemoryOwnerAfterMrDereg());
        expected_events.push_back("teardown");
        EXPECT_EQ(memory_util_->teardownCount(), cycle);
        EXPECT_FALSE(memory_util_->isAvailable());

        EXPECT_TRUE(lifecycle->rebuildMemoryOwnerBeforeMrReg());
        EXPECT_TRUE(lifecycle->rebuildMemoryOwnerBeforeMrReg());
        expected_events.push_back("rebuild");

        threads.clear();
        successful_calls = 0;
        for (int i = 0; i < 8; ++i) {
            threads.emplace_back([&]() { successful_calls += lifecycle->rebuildAfterRestore(); });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        EXPECT_EQ(successful_calls.load(), 8);
        EXPECT_EQ(memory_util_->rebuildCount(), cycle);
        EXPECT_EQ(cache_store_.get(), identity);
        EXPECT_FALSE(lifecycle->isAvailable());
        expectEmptyStore(false, CacheStoreErrorCode::PushWorkerItemFailed);

        EXPECT_TRUE(memory_util_->regUserMr(reinterpret_cast<void*>(0x2000 + cycle), 4096, true, 4096));
        expected_events.push_back("reg_mr");
        EXPECT_TRUE(lifecycle->resumeAfterCheckpoint());
        EXPECT_TRUE(lifecycle->isAvailable());
        expectEmptyStore(true, CacheStoreErrorCode::None);
        EXPECT_EQ(memory_util_->events(), expected_events);
    }
}

TEST_F(NormalCacheStoreCheckpointTest, MemoryOwnerCannotStopBeforeTransportTeardown) {
    EXPECT_FALSE(cache_store_->teardownMemoryOwnerAfterMrDereg());
    EXPECT_EQ(memory_util_->teardownCount(), 0);
    EXPECT_TRUE(memory_util_->isAvailable());

    ASSERT_TRUE(cache_store_->beginCheckpointDrain());
    EXPECT_FALSE(cache_store_->teardownMemoryOwnerAfterMrDereg());
    EXPECT_EQ(memory_util_->teardownCount(), 0);
    EXPECT_TRUE(memory_util_->isAvailable());
}

TEST_F(NormalCacheStoreCheckpointTest, DrainRejectsNewWorkAndCanBeCancelled) {
    bool                        existing_callback_called = false;
    CacheStoreStoreDoneCallback callback          = [&](bool, CacheStoreErrorCode) { existing_callback_called = true; };
    auto                        existing_callback = cache_store_->countTransfer(std::move(callback));
    ASSERT_EQ(cache_store_->activeTransferCount(), 1);

    ASSERT_TRUE(cache_store_->beginCheckpointDrain());
    expectEmptyStore(false, CacheStoreErrorCode::PushWorkerItemFailed);
    EXPECT_FALSE(cache_store_->teardownForCheckpoint());

    existing_callback(true, CacheStoreErrorCode::None);
    EXPECT_TRUE(existing_callback_called);
    EXPECT_EQ(cache_store_->activeTransferCount(), 0);
    EXPECT_TRUE(cache_store_->resumeAfterCheckpoint());
    EXPECT_TRUE(cache_store_->isAvailable());
    expectEmptyStore(true, CacheStoreErrorCode::None);
}

TEST_F(NormalCacheStoreCheckpointTest, ConcurrentAdmissionFailsClosedAfterDrain) {
    std::atomic<bool>        start{false};
    std::atomic<int>         callback_count{0};
    std::atomic<int>         success_count{0};
    std::vector<std::thread> workers;
    for (int i = 0; i < 8; ++i) {
        workers.emplace_back([&, i]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            auto request = std::make_shared<RequestBlockBuffer>("concurrent-admission-" + std::to_string(i));
            cache_store_->store(request, [&](bool ok, CacheStoreErrorCode) {
                callback_count++;
                success_count += ok;
            });
        });
    }
    start.store(true, std::memory_order_release);
    ASSERT_TRUE(cache_store_->beginCheckpointDrain());
    for (auto& worker : workers) {
        worker.join();
    }
    EXPECT_EQ(callback_count.load(), 8);
    EXPECT_LE(success_count.load(), 8);

    for (int i = 0; i < 32; ++i) {
        expectEmptyStore(false, CacheStoreErrorCode::PushWorkerItemFailed);
    }
    EXPECT_TRUE(cache_store_->teardownForCheckpoint());
    EXPECT_TRUE(cache_store_->rebuildAfterRestore());
    EXPECT_FALSE(cache_store_->isAvailable());
    EXPECT_TRUE(memory_util_->regUserMr(reinterpret_cast<void*>(0x3000), 4096, true, 4096));
    EXPECT_TRUE(cache_store_->resumeAfterCheckpoint());
}

TEST_F(NormalCacheStoreCheckpointTest, RebuildFailureStaysFailClosed) {
    ASSERT_TRUE(cache_store_->beginCheckpointDrain());
    ASSERT_TRUE(cache_store_->teardownForCheckpoint());
    ASSERT_TRUE(cache_store_->teardownMemoryOwnerAfterMrDereg());
    memory_util_->setFailRebuild(true);

    EXPECT_FALSE(cache_store_->rebuildMemoryOwnerBeforeMrReg());
    EXPECT_FALSE(cache_store_->rebuildMemoryOwnerBeforeMrReg());
    EXPECT_FALSE(cache_store_->rebuildAfterRestore());
    EXPECT_FALSE(cache_store_->resumeAfterCheckpoint());
    EXPECT_FALSE(cache_store_->isAvailable());
    EXPECT_EQ(memory_util_->rebuildCount(), 1);
    expectEmptyStore(false, CacheStoreErrorCode::PushWorkerItemFailed);
}

TEST_F(NormalCacheStoreCheckpointTest, CancelDoesNotHideDelayedPhysicalTransfer) {
    std::atomic<bool> cancelled{false};
    auto              request = std::make_shared<RemoteStoreRequest>(
        "client", "remote-store-request", "127.0.0.1", 1, 2, currentTimeUs() + 1000000, 1, 0, 1, 0);
    request->buffer_pairs["local"] = "remote";
    auto collector                 = std::make_shared<CacheStoreRemoteStoreMetricsCollector>(nullptr, 1);
    auto task                      = std::make_shared<RemoteStoreTaskImpl>(
        request, collector, [&cancelled]() { return cancelled.load(std::memory_order_acquire); });

    auto storage          = std::shared_ptr<void>(new char[16], [](void* ptr) { delete[] static_cast<char*>(ptr); });
    auto block            = std::make_shared<BlockBuffer>("local", storage, 16, false, true);
    auto transfer_request = task->makeAvailableRequest(std::vector<std::shared_ptr<BlockBuffer>>{block});
    ASSERT_NE(transfer_request, nullptr);
    cache_store_->trackPhysicalTransfer(transfer_request);
    ASSERT_EQ(cache_store_->activePhysicalTransferCount(), 1);

    cancelled.store(true, std::memory_order_release);
    task->waitDone();
    ASSERT_TRUE(task->done());
    ASSERT_TRUE(cache_store_->beginCheckpointDrain());
    EXPECT_EQ(cache_store_->activePhysicalTransferCount(), 1);
    EXPECT_FALSE(cache_store_->teardownForCheckpoint());

    transfer_request->callback(false, CacheStoreErrorCode::CallPrefillTimeout, transfer_request->buffer_pairs);
    EXPECT_EQ(cache_store_->activePhysicalTransferCount(), 0);
    EXPECT_TRUE(cache_store_->teardownForCheckpoint());
    EXPECT_TRUE(memory_util_->isAvailable());
    EXPECT_TRUE(cache_store_->teardownMemoryOwnerAfterMrDereg());
    EXPECT_FALSE(memory_util_->isAvailable());
}

class NormalCacheStoreTest: public CacheStoreTestBase {
protected:
    bool initCacheStores();

    void verifyBlock(
        const std::shared_ptr<BlockBuffer>& block, const std::string& key, uint32_t len, bool gpu_mem, char val);
    void setThreadFunction(const std::string& requestid, size_t block_size);
    void getThreadFunction(const std::string& requestid, size_t block_size);

protected:
    std::shared_ptr<NormalCacheStore> cache_store1_;
    std::shared_ptr<NormalCacheStore> cache_store2_;
    uint32_t                          port1_;
    uint32_t                          port2_;
};

bool NormalCacheStoreTest::initCacheStores() {
    if (!device_util_ || !memory_util_) {
        return false;
    }

    port1_ = autil::NetUtil::randomPort();
    port2_ = autil::NetUtil::randomPort();

    CacheStoreInitParams params1;
    params1.listen_port   = port1_;
    params1.enable_metric = false;
    params1.memory_util   = memory_util_;

    cache_store1_ = NormalCacheStore::createNormalCacheStore(params1);
    if (!cache_store1_) {
        return false;
    }
    CacheStoreInitParams params2;
    params2.listen_port   = port2_;
    params2.enable_metric = false;
    params2.memory_util   = memory_util_;

    cache_store2_ = NormalCacheStore::createNormalCacheStore(params2);
    return cache_store2_ != nullptr;
}

void NormalCacheStoreTest::verifyBlock(
    const std::shared_ptr<BlockBuffer>& block, const std::string& key, uint32_t len, bool gpu_mem, char val) {
    ASSERT_TRUE(block != nullptr) << key;

    ASSERT_EQ(key, block->key);
    ASSERT_EQ(len, block->len);
    ASSERT_EQ(gpu_mem, block->gpu_mem);

    if (len == 0) {
        return;
    }

    if (!gpu_mem) {
        ASSERT_EQ(val, ((char*)(block->addr.get()))[0]) << key;
        return;
    }

    auto buf = device_util_->mallocCPU(len);
    ASSERT_TRUE(device_util_->memcopy(buf, false, block->addr.get(), block->gpu_mem, len));
    ASSERT_EQ(val, ((char*)(buf))[0]) << key << " " << reinterpret_cast<uint64_t>(block->addr.get());

    device_util_->freeCPU(buf);
}

TEST_F(NormalCacheStoreTest, testStore_Success) {
    ASSERT_TRUE(initCacheStores());

    uint32_t    block_size  = 16;
    std::string requestid   = "test-request-id";
    auto        store_cache = std::make_shared<RequestBlockBuffer>(requestid);
    store_cache->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));
    store_cache->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, '1', true));

    std::mutex mutex;  // for sync test
    mutex.lock();
    auto store_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    };

    cache_store1_->store(store_cache, store_callback);

    mutex.lock();
    mutex.unlock();

    // save to local cache
    if (memory_util_->isRdmaMode()) {
        // rdma block cache store will store mr memory, there will be no copy
        verifyBlock(
            cache_store1_->getRequestBlockBufferStore()->getBlockBuffer(requestid, "a"), "a", block_size, true, '0');
        verifyBlock(
            cache_store1_->getRequestBlockBufferStore()->getBlockBuffer(requestid, "ab"), "ab", block_size, true, '1');
    } else {
        // cpu block cache store will store only cpu memory, so there will be a copy
        verifyBlock(
            cache_store1_->getRequestBlockBufferStore()->getBlockBuffer(requestid, "a"), "a", block_size, false, '0');
        verifyBlock(
            cache_store1_->getRequestBlockBufferStore()->getBlockBuffer(requestid, "ab"), "ab", block_size, false, '1');
    }

    ASSERT_TRUE(cache_store2_->getRequestBlockBufferStore()->getBlockBuffer(requestid, "a") == nullptr);
    ASSERT_TRUE(cache_store2_->getRequestBlockBufferStore()->getBlockBuffer(requestid, "ab") == nullptr);
}

TEST_F(NormalCacheStoreTest, testStore_emptyCache) {
    ASSERT_TRUE(initCacheStores());

    std::mutex mutex;  // for sync test
    auto       store_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    };

    std::string requestid   = "test-request-id";
    auto        store_cache = std::make_shared<RequestBlockBuffer>(requestid);

    mutex.lock();
    cache_store1_->store(store_cache, store_callback);
    mutex.lock();
    mutex.unlock();
}

TEST_F(NormalCacheStoreTest, testMarkRequestEndClearsPendingStoreTransferCount) {
    ASSERT_TRUE(initCacheStores());

    cudaEvent_t gate;
    ASSERT_EQ(cudaSuccess, cudaEventCreateWithFlags(&gate, cudaEventDisableTiming));
    auto stream = at::cuda::getStreamFromPool();
    ASSERT_EQ(cudaSuccess, cudaStreamWaitEvent(stream.stream(), gate, 0));

    const std::string requestid  = "test-pending-store-request-id";
    const uint32_t    block_size = 16;
    auto              event      = std::make_shared<torch::Event>(torch::kCUDA);
    {
        c10::cuda::CUDAStreamGuard guard(stream);
        event->record(stream);
    }

    auto store_cache = std::make_shared<RequestBlockBuffer>(requestid, event);
    store_cache->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));

    std::atomic<bool> callback_called{false};
    cache_store1_->store(store_cache, [&callback_called](bool ok, CacheStoreErrorCode ec) {
        callback_called.store(true, std::memory_order_release);
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    });

    ASSERT_EQ(cache_store1_->activeTransferCount(), 1);
    cache_store1_->markRequestEnd(requestid);

    EXPECT_TRUE(callback_called.load(std::memory_order_acquire));
    EXPECT_EQ(cache_store1_->activeTransferCount(), 0);

    ASSERT_EQ(cudaSuccess, cudaEventRecord(gate, 0));
    ASSERT_EQ(cudaSuccess, cudaEventSynchronize(gate));
    ASSERT_EQ(cudaSuccess, cudaEventDestroy(gate));
}

TEST_F(NormalCacheStoreTest, testStore_invalidParams) {
    ASSERT_TRUE(initCacheStores());

    std::mutex mutex;  // for sync test
    auto       store_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::InvalidParams, ec);
    };

    mutex.lock();
    cache_store1_->store(nullptr, store_callback);
    mutex.lock();
    mutex.unlock();

    uint32_t    block_size  = 16;
    std::string requestid   = "test-request-id";
    auto        store_cache = std::make_shared<RequestBlockBuffer>(requestid);
    store_cache->addBlock("block", nullptr, block_size, false, true);

    mutex.lock();
    cache_store1_->store(store_cache, store_callback);
    mutex.lock();
    mutex.unlock();
}

TEST_F(NormalCacheStoreTest, testStore_storeToBufferStoreFailed) {
    ASSERT_TRUE(initMockMemoryUtil());
    ASSERT_TRUE(initCacheStores());

    uint32_t    block_size  = 16;
    std::string requestid   = "test-request-id";
    auto        store_cache = std::make_shared<RequestBlockBuffer>(requestid);
    EXPECT_CALL(*mock_memory_util_, regUserMr(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(true));
    store_cache->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));
    store_cache->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, '1', true));

    std::mutex mutex;  // for sync test
    mutex.lock();
    auto store_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::StoreFailed, ec);
    };

    // mock call from isMemoryMr, then return false in tcp mode
    EXPECT_CALL(*mock_memory_util_, isMemoryMr(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(*mock_memory_util_, regUserMr(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(false));

    cache_store1_->store(store_cache, store_callback);
    mutex.lock();
    mutex.unlock();
}

TEST_F(NormalCacheStoreTest, testLoad_Success) {
    ASSERT_TRUE(initCacheStores());

    // store a to cache_store1 for local get
    uint32_t    block_size   = 16;
    std::string requestid    = "test-request-id";
    auto        store_buffer = std::make_shared<RequestBlockBuffer>(requestid);
    store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));

    cache_store2_->store(store_buffer, [](bool ok, CacheStoreErrorCode ec) {
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    });

    // store abc to cache_store2 for get success on later time
    std::thread set_block_thread([this, requestid, block_size]() {
        usleep(100 * 1000);  /// wait 100ms then set block

        auto store_buffer = std::make_shared<RequestBlockBuffer>(requestid);
        store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, '1', true));
        store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("abc", block_size, '2', true));
        cache_store2_->store(store_buffer, [](bool ok, CacheStoreErrorCode ec) {
            ASSERT_TRUE(ok);
            ASSERT_EQ(CacheStoreErrorCode::None, ec);
        });
    });

    auto load_cache = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, 'a', true));
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, 'b', true));
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("abc", block_size, 'c', true));

    std::mutex mutex;  // for sync test
    mutex.lock();
    auto load_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    };
    cache_store1_->load(load_cache, load_callback, autil::NetUtil::getBindIp(), port2_, 0, 1000);

    mutex.lock();  // wait till callback
    mutex.unlock();

    verifyBlock(load_cache->getBlock("a"), "a", block_size, true, '0');
    verifyBlock(load_cache->getBlock("ab"), "ab", block_size, true, '1');
    verifyBlock(load_cache->getBlock("abc"), "abc", block_size, true, '2');

    set_block_thread.join();
}

TEST_F(NormalCacheStoreTest, testLoad_loadBeforeStore) {
    uint32_t block_size = 16;
    ASSERT_TRUE(initCacheStores());

    std::string requestid = "test-request-id";

    std::mutex mutex;  // for sync test

    auto load_cache = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, 'a', true));
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, 'b', true));
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("abc", block_size, 'c', true));

    mutex.lock();
    auto load_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    };
    cache_store1_->load(load_cache, load_callback, "localhost", port2_, 0, 1000);

    // store a to cache_store1 for local get
    auto store_buffer = std::make_shared<RequestBlockBuffer>(requestid);
    store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));
    store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, '1', true));
    store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("abc", block_size, '2', true));
    cache_store2_->store(store_buffer, [](bool ok, CacheStoreErrorCode ec) {
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    });
    mutex.lock();  // wait till callback
    mutex.unlock();

    verifyBlock(load_cache->getBlock("a"), "a", block_size, true, '0');
    verifyBlock(load_cache->getBlock("ab"), "ab", block_size, true, '1');
    verifyBlock(load_cache->getBlock("abc"), "abc", block_size, true, '2');
}

TEST_F(NormalCacheStoreTest, testLoad_MultiThread) {
    uint32_t block_size = 16;
    ASSERT_TRUE(initCacheStores());

    std::string requestid1 = "test-request-id1";
    std::string requestid2 = "test-request-id2";
    std::string requestid3 = "test-request-id3";

    // store abc to cache_store2 for get success on later time
    std::thread set_block_thread1([this, requestid1, block_size]() {
        usleep(100 * 1000);  /// wait 100ms then set block
        auto store_buffer = std::make_shared<RequestBlockBuffer>(requestid1);
        store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));
        store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, '1', true));
        cache_store2_->store(store_buffer, [](bool ok, CacheStoreErrorCode ec) {
            ASSERT_TRUE(ok);
            ASSERT_EQ(CacheStoreErrorCode::None, ec);
        });
    });
    std::thread set_block_thread2([this, requestid2, block_size]() {
        usleep(100 * 1000);  /// wait 100ms then set block
        auto store_buffer = std::make_shared<RequestBlockBuffer>(requestid2);
        store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("abc", block_size, '2', true));
        cache_store2_->store(store_buffer, [](bool ok, CacheStoreErrorCode ec) {
            ASSERT_TRUE(ok);
            ASSERT_EQ(CacheStoreErrorCode::None, ec);
        });
    });
    std::thread set_block_thread3([this, requestid3, block_size]() {
        usleep(100 * 1000);  /// wait 100ms then set block
        auto store_buffer = std::make_shared<RequestBlockBuffer>(requestid3);
        store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("abcd", block_size, '3', true));
        cache_store2_->store(store_buffer, [](bool ok, CacheStoreErrorCode ec) {
            ASSERT_TRUE(ok);
            ASSERT_EQ(CacheStoreErrorCode::None, ec);
        });
    });

    std::thread get_block_thread1([this, requestid1, block_size]() {
        std::mutex mutex1;
        auto       load_cache = std::make_shared<RequestBlockBuffer>(requestid1);
        load_cache->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, 'a', true));
        load_cache->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, 'b', true));
        mutex1.lock();
        auto load_callback = [&mutex1](bool ok, CacheStoreErrorCode ec) {
            mutex1.unlock();
            ASSERT_TRUE(ok);
            ASSERT_EQ(CacheStoreErrorCode::None, ec);
        };
        cache_store1_->load(load_cache, load_callback, "localhost", port2_, 0, 1000);
        mutex1.lock();
        mutex1.unlock();
        verifyBlock(load_cache->getBlock("a"), "a", block_size, true, '0');
        verifyBlock(load_cache->getBlock("ab"), "ab", block_size, true, '1');
    });
    std::thread get_block_thread2([this, requestid2, block_size]() {
        std::mutex mutex2;
        auto       load_cache = std::make_shared<RequestBlockBuffer>(requestid2);
        load_cache->addBlock(block_buffer_util_->makeBlockBuffer("abc", block_size, 'c', true));
        mutex2.lock();
        auto load_callback = [&mutex2](bool ok, CacheStoreErrorCode ec) {
            mutex2.unlock();
            ASSERT_TRUE(ok);
            ASSERT_EQ(CacheStoreErrorCode::None, ec);
        };
        cache_store1_->load(load_cache, load_callback, "localhost", port2_, 0, 1000);
        mutex2.lock();
        mutex2.unlock();
        verifyBlock(load_cache->getBlock("abc"), "abc", block_size, true, '2');
    });
    std::thread get_block_thread3([this, requestid3, block_size]() {
        std::mutex mutex3;
        auto       load_cache = std::make_shared<RequestBlockBuffer>(requestid3);
        load_cache->addBlock(block_buffer_util_->makeBlockBuffer("abcd", block_size, 'd', true));
        mutex3.lock();
        auto load_callback = [&mutex3](bool ok, CacheStoreErrorCode ec) {
            mutex3.unlock();
            ASSERT_TRUE(ok);
            ASSERT_EQ(CacheStoreErrorCode::None, ec);
        };
        cache_store1_->load(load_cache, load_callback, "localhost", port2_, 0, 1000);
        mutex3.lock();
        mutex3.unlock();
        verifyBlock(load_cache->getBlock("abcd"), "abcd", block_size, true, '3');
    });

    set_block_thread1.join();
    set_block_thread2.join();
    set_block_thread3.join();

    get_block_thread1.join();
    get_block_thread2.join();
    get_block_thread3.join();
}

void NormalCacheStoreTest::setThreadFunction(const std::string& requestid, size_t block_size) {
    usleep(100 * 1000);  // 等待100毫秒后设置block
    auto store_buffer = std::make_shared<RequestBlockBuffer>(requestid);
    store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("a" + requestid, block_size, '0', true));
    store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("ab" + requestid, block_size, '1', true));
    cache_store1_->store(store_buffer, [](bool ok, CacheStoreErrorCode ec) {
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    });
}

void NormalCacheStoreTest::getThreadFunction(const std::string& requestid, size_t block_size) {
    std::mutex mutex1;
    auto       load_cache = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("a" + requestid, block_size, 'a', true));
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("ab" + requestid, block_size, 'b', true));
    mutex1.lock();
    auto load_callback = [&mutex1](bool ok, CacheStoreErrorCode ec) {
        mutex1.unlock();
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    };
    cache_store2_->load(load_cache, load_callback, "localhost", port1_, 0, 1000);
    mutex1.lock();
    mutex1.unlock();
    verifyBlock(load_cache->getBlock("a" + requestid), "a" + requestid, block_size, true, '0');
    verifyBlock(load_cache->getBlock("ab" + requestid), "ab" + requestid, block_size, true, '1');
}

TEST_F(NormalCacheStoreTest, testLoad_MultiThread2) {
    uint32_t block_size = 16;
    ASSERT_TRUE(initCacheStores());

    std::vector<std::thread> set_threads;
    for (int i = 0; i < 10; ++i) {
        std::string request_id = std::to_string(i);
        set_threads.emplace_back([this, request_id, block_size]() { this->setThreadFunction(request_id, block_size); });
    }

    std::vector<std::thread> get_threads;
    for (int i = 0; i < 10; ++i) {
        std::string request_id = std::to_string(i);
        get_threads.emplace_back([this, request_id, block_size]() { this->getThreadFunction(request_id, block_size); });
    }

    for (auto& t : set_threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    for (auto& t : get_threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}

TEST_F(NormalCacheStoreTest, testLoad_emptyCache) {
    ASSERT_TRUE(initCacheStores());

    std::mutex mutex;  // for sync test
    auto       load_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    };

    std::string requestid  = "test-request-id";
    auto        load_cache = std::make_shared<RequestBlockBuffer>(requestid);

    mutex.lock();
    cache_store1_->load(load_cache, load_callback, autil::NetUtil::getBindIp(), port2_, 0, 1000);
    mutex.lock();  // wait till callback
    mutex.unlock();
}

TEST_F(NormalCacheStoreTest, testLoad_invalidParams) {
    ASSERT_TRUE(initCacheStores());

    std::mutex mutex;  // for sync test
    auto       load_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::InvalidParams, ec);
    };

    mutex.lock();
    cache_store1_->load(nullptr, load_callback, autil::NetUtil::getBindIp(), port2_, 0, 1000);
    mutex.lock();  // wait till callback
    mutex.unlock();

    uint32_t    block_size = 16;
    std::string requestid  = "test-request-id";
    auto        load_cache = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache->addBlock("block", nullptr, block_size, true, true);

    mutex.lock();
    cache_store1_->load(load_cache, load_callback, autil::NetUtil::getBindIp(), port2_, 0, 1000);
    mutex.lock();  // wait till callback
    mutex.unlock();
}

TEST_F(NormalCacheStoreTest, testLoad_remoteCallPrefillTimeout) {
    ASSERT_TRUE(initCacheStores());

    std::string requestid  = "test-request-id";
    uint32_t    block_size = 16;

    auto load_cache = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, 'a', true));
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, 'b', true));

    std::mutex mutex;  // for sync test
    auto       load_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::LoadBufferTimeout, ec);
    };

    mutex.lock();
    cache_store1_->load(load_cache, load_callback, autil::NetUtil::getBindIp(), port2_, 0, 1000);
    mutex.lock();  // wait till callback
    mutex.unlock();

    verifyBlock(load_cache->getBlock("a"), "a", block_size, true, 'a');
    verifyBlock(load_cache->getBlock("ab"), "ab", block_size, true, 'b');
}

TEST_F(NormalCacheStoreTest, testLoad_remoteBufferExpired) {
    ASSERT_TRUE(initCacheStores());

    std::string requestid  = "test-request-id";
    uint32_t    block_size = 16;

    cache_store2_->markRequestEnd(requestid);

    auto load_cache = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, 'a', true));
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, 'b', true));

    std::mutex mutex;  // for sync test
    auto       load_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::LoadBufferTimeout, ec);
    };

    mutex.lock();
    cache_store1_->load(load_cache, load_callback, autil::NetUtil::getBindIp(), port2_, 0, 1000);
    mutex.lock();  // wait till callback
    mutex.unlock();
}

void doReg(
    const std::shared_ptr<MemoryUtil>& memory_util, void* buf, uint64_t block_size, uint64_t block_num, bool gpu_mem) {
    std::vector<void*> bufs;
    int                i = 0;
    for (; i < block_num; i++) {
        void* tmp_buf = (void*)((uint64_t)buf + block_size * i);
        if (memory_util->regUserMr(tmp_buf, block_size, gpu_mem)) {
            bufs.push_back(tmp_buf);
            continue;
        }
        break;
    }
    std::cout << "register user mr, one block_size " << block_size << "; success block count " << i << ", expect count "
              << block_num << ",  total size " << block_size * i * 1.0f / 1024 / 1024 << " mb" << std::endl;
    for (int j = 0; j < bufs.size(); j++) {
        ASSERT_TRUE(memory_util->deregUserMr(bufs[j], gpu_mem));
    }
}

TEST_F(NormalCacheStoreTest, testRdmaMemoryRegCPU) {
    uint64_t block_size = 32 * 1024;
    uint64_t block_num  = 24;

    void* ptr = nullptr;
    std::cout << "start test " << block_size << std::endl;
    ptr = malloc(block_size * block_num);
    ASSERT_TRUE(ptr != nullptr);
    doReg(memory_util_, ptr, block_size, block_num, false);
    free(ptr);
}

TEST_F(NormalCacheStoreTest, testRdmaMemoryRegGPU) {
    uint64_t block_size = 32 * 1024;
    uint64_t block_num  = 24;

    void* ptr = nullptr;
    std::cout << "start test " << block_size << std::endl;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&ptr, block_size * block_num));
    doReg(memory_util_, ptr, block_size, block_num, true);
    ASSERT_EQ(cudaSuccess, cudaFree(ptr));
}

TEST_F(NormalCacheStoreTest, testLoad_canceWhileLoad) {
    ASSERT_TRUE(initCacheStores());

    // store a to cache_store1 for local get
    uint32_t    block_size   = 16;
    std::string requestid    = "test-request-id";
    auto        store_buffer = std::make_shared<RequestBlockBuffer>(requestid);
    store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));

    cache_store2_->store(store_buffer, [](bool ok, CacheStoreErrorCode ec) {
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    });

    // store abc to cache_store2 for get success on later time
    std::thread set_block_thread([this, requestid, cache_store = cache_store2_]() {
        usleep(100 * 1000);  /// wait 100ms then set block
        cache_store->markRequestEnd(requestid);
    });

    auto load_cache = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, 'a', true));
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, 'b', true));

    std::mutex mutex;  // for sync test
    mutex.lock();
    auto load_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::LoadBufferTimeout, ec);
    };
    cache_store1_->load(load_cache, load_callback, autil::NetUtil::getBindIp(), port2_, 0, 1000);

    mutex.lock();  // wait till callback
    mutex.unlock();

    set_block_thread.join();
}

TEST_F(NormalCacheStoreTest, testLoadContext_Success) {
    ASSERT_TRUE(initCacheStores());

    // store a to cache_store1 for local get
    uint32_t    block_size   = 16;
    std::string requestid    = "test-request-id";
    auto        store_buffer = std::make_shared<RequestBlockBuffer>(requestid);
    store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));

    cache_store2_->store(store_buffer, [](bool ok, CacheStoreErrorCode ec) {
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    });

    // store abc to cache_store2 for get success on later time
    std::thread set_block_thread([this, requestid, block_size]() {
        usleep(100 * 1000);  /// wait 100ms then set block

        auto store_buffer = std::make_shared<RequestBlockBuffer>(requestid);
        store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, '1', true));
        store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("abc", block_size, '2', true));
        cache_store2_->store(store_buffer, [](bool ok, CacheStoreErrorCode ec) {
            ASSERT_TRUE(ok);
            ASSERT_EQ(CacheStoreErrorCode::None, ec);
        });
    });

    auto load_cache1 = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache1->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, 'a', true));
    auto load_cache2 = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache2->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, 'b', true));
    load_cache2->addBlock(block_buffer_util_->makeBlockBuffer("abc", block_size, 'c', true));
    std::vector<std::shared_ptr<RequestBlockBuffer>> load_caches{load_cache1, load_cache2};

    auto load_context = cache_store1_->loadBuffers(
        load_caches, autil::NetUtil::getBindIp(), port2_, 0, 1000, []() { return false; }, 1, 0);
    ASSERT_TRUE(load_context != nullptr);
    load_context->waitDone();

    ASSERT_TRUE(load_context->success());

    verifyBlock(load_cache1->getBlock("a"), "a", block_size, true, '0');
    verifyBlock(load_cache2->getBlock("ab"), "ab", block_size, true, '1');
    verifyBlock(load_cache2->getBlock("abc"), "abc", block_size, true, '2');

    set_block_thread.join();
}

TEST_F(NormalCacheStoreTest, testLoadContext_loadTimeout) {
    ASSERT_TRUE(initCacheStores());

    std::string requestid  = "test-request-id";
    uint32_t    block_size = 16;

    auto load_cache1 = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache1->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, 'a', true));
    auto load_cache2 = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache2->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, 'b', true));
    std::vector<std::shared_ptr<RequestBlockBuffer>> load_caches{load_cache1, load_cache2};

    auto load_context = cache_store1_->loadBuffers(
        load_caches, autil::NetUtil::getBindIp(), port2_, 0, 1000, []() { return false; }, 1, 0);
    ASSERT_TRUE(load_context != nullptr);
    load_context->waitDone();

    ASSERT_FALSE(load_context->success());
    ASSERT_EQ(ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT, load_context->getErrorInfo().code());

    verifyBlock(load_cache1->getBlock("a"), "a", block_size, true, 'a');
    verifyBlock(load_cache2->getBlock("ab"), "ab", block_size, true, 'b');
}

TEST_F(NormalCacheStoreTest, testLoadContext_loadCancel) {
    ASSERT_TRUE(initCacheStores());

    std::string requestid  = "test-request-id";
    uint32_t    block_size = 16;

    auto load_cache1 = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache1->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, 'a', true));
    auto load_cache2 = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache2->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, 'b', true));
    std::vector<std::shared_ptr<RequestBlockBuffer>> load_caches{load_cache1, load_cache2};

    auto start_time_ms = autil::TimeUtility::currentTimeInMilliSeconds();
    auto load_context  = cache_store1_->loadBuffers(
        load_caches,
        autil::NetUtil::getBindIp(),
        port2_,
        0,
        1000,
        [start_time_ms]() { return start_time_ms + 100 < autil::TimeUtility::currentTimeInMilliSeconds(); },
        1,
        0);

    ASSERT_TRUE(load_context != nullptr);
    load_context->waitDone();

    ASSERT_FALSE(load_context->success());
    ASSERT_EQ(ErrorCode::CANCELLED, load_context->getErrorInfo().code());

    verifyBlock(load_cache1->getBlock("a"), "a", block_size, true, 'a');
    verifyBlock(load_cache2->getBlock("ab"), "ab", block_size, true, 'b');
}

TEST_F(NormalCacheStoreTest, testStoreContext_Success) {
    ASSERT_TRUE(initCacheStores());

    uint32_t    block_size = 16;
    std::string requestid  = "test-request-id";

    std::vector<std::shared_ptr<RequestBlockBuffer>> store_caches;
    auto                                             store_cache1 = std::make_shared<RequestBlockBuffer>(requestid);
    store_cache1->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));
    store_caches.push_back(store_cache1);

    auto store_cache2 = std::make_shared<RequestBlockBuffer>(requestid);
    store_cache2->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, '1', true));
    store_caches.push_back(store_cache2);

    auto store_context = cache_store1_->storeBuffers(store_caches, 1000);
    ASSERT_TRUE(store_context != nullptr);
    store_context->waitDone();
    ASSERT_TRUE(store_context->success());

    // save to local cache
    if (memory_util_->isRdmaMode()) {
        // rdma block cache store will store mr memory, there will be no copy
        verifyBlock(
            cache_store1_->getRequestBlockBufferStore()->getBlockBuffer(requestid, "a"), "a", block_size, true, '0');
        verifyBlock(
            cache_store1_->getRequestBlockBufferStore()->getBlockBuffer(requestid, "ab"), "ab", block_size, true, '1');
    } else {
        // cpu block cache store will store only cpu memory, so there will be a copy
        verifyBlock(
            cache_store1_->getRequestBlockBufferStore()->getBlockBuffer(requestid, "a"), "a", block_size, false, '0');
        verifyBlock(
            cache_store1_->getRequestBlockBufferStore()->getBlockBuffer(requestid, "ab"), "ab", block_size, false, '1');
    }
}

TEST_F(NormalCacheStoreTest, testStoreContext_storeToBufferStoreFailed) {
    ASSERT_TRUE(initMockMemoryUtil());
    ASSERT_TRUE(initCacheStores());

    uint32_t    block_size  = 16;
    std::string requestid   = "test-request-id";
    auto        store_cache = std::make_shared<RequestBlockBuffer>(requestid);
    EXPECT_CALL(*mock_memory_util_, regUserMr(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(true));

    std::vector<std::shared_ptr<RequestBlockBuffer>> store_caches;
    auto                                             store_cache1 = std::make_shared<RequestBlockBuffer>(requestid);
    store_cache1->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));
    store_caches.push_back(store_cache1);

    auto store_cache2 = std::make_shared<RequestBlockBuffer>(requestid);
    store_cache2->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, '1', true));
    store_caches.push_back(store_cache2);

    // mock call from isMemoryMr, then return false in tcp mode
    EXPECT_CALL(*mock_memory_util_, isMemoryMr(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(*mock_memory_util_, regUserMr(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(false));

    auto store_context = cache_store1_->storeBuffers(store_caches, 1000);
    ASSERT_TRUE(store_context != nullptr);
    store_context->waitDone();
    ASSERT_FALSE(store_context->success());
}

TEST_F(NormalCacheStoreTest, testLoadPatition_Success) {
    ASSERT_TRUE(initCacheStores());

    // store a to cache_store1 for local get
    uint32_t    block_size   = 16;
    std::string requestid    = "test-request-id";
    auto        store_buffer = std::make_shared<RequestBlockBuffer>(requestid);
    store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));

    cache_store2_->store(store_buffer, [](bool ok, CacheStoreErrorCode ec) {
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    });

    // store abc to cache_store2 for get success on later time
    std::thread set_block_thread([this, requestid, block_size]() {
        usleep(100 * 1000);  /// wait 100ms then set block

        auto store_buffer = std::make_shared<RequestBlockBuffer>(requestid);
        store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("ab", block_size, '1', true));
        store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("abc", block_size, '2', true));
        cache_store2_->store(store_buffer, [](bool ok, CacheStoreErrorCode ec) {
            ASSERT_TRUE(ok);
            ASSERT_EQ(CacheStoreErrorCode::None, ec);
        });
    });

    auto block1 = block_buffer_util_->makeBlockBuffer("a", block_size, 'a', true);
    auto block2 = block_buffer_util_->makeBlockBuffer("ab", block_size, 'b', true);
    auto block3 = block_buffer_util_->makeBlockBuffer("abc", block_size, 'c', true);

    auto partition_count = 8;
    auto part_block_size = block_size / partition_count;
    ASSERT_EQ(part_block_size, 2);
    for (int i = 0; i < partition_count; i++) {
        auto                  load_cache1      = std::make_shared<RequestBlockBuffer>(requestid);
        void*                 part_block1_addr = (void*)((int64_t)block1->addr.get() + part_block_size * i);
        std::shared_ptr<void> part_block1(part_block1_addr, [](void* ptr) {});
        load_cache1->addBlock("a", part_block1, part_block_size, true, true);

        auto                  load_cache2      = std::make_shared<RequestBlockBuffer>(requestid);
        void*                 part_block2_addr = (void*)((int64_t)block2->addr.get() + part_block_size * i);
        std::shared_ptr<void> part_block2(part_block2_addr, [](void* ptr) {});
        load_cache2->addBlock("ab", part_block2, part_block_size, true, true);

        void*                 part_block3_addr = (void*)((int64_t)block3->addr.get() + part_block_size * i);
        std::shared_ptr<void> part_block3(part_block3_addr, [](void* ptr) {});
        load_cache2->addBlock("abc", part_block3, part_block_size, true, true);
        std::vector<std::shared_ptr<RequestBlockBuffer>> load_caches{load_cache1, load_cache2};

        auto load_context = cache_store1_->loadBuffers(
            load_caches, autil::NetUtil::getBindIp(), port2_, 0, 1000, []() { return false; }, partition_count, i);
        ASSERT_TRUE(load_context != nullptr);
        load_context->waitDone();
        ASSERT_TRUE(load_context->success());
        verifyBlock(load_cache1->getBlock("a"), "a", block_size / partition_count, true, '0');
        verifyBlock(load_cache2->getBlock("ab"), "ab", block_size / partition_count, true, '1');
        verifyBlock(load_cache2->getBlock("abc"), "abc", block_size / partition_count, true, '2');
        ASSERT_EQ(load_cache1->getBlock("a")->addr.get(), part_block1_addr);
        ASSERT_EQ(load_cache2->getBlock("ab")->addr.get(), part_block2_addr);
        ASSERT_EQ(load_cache2->getBlock("abc")->addr.get(), part_block3_addr);
    }

    set_block_thread.join();
}

}  // namespace rtp_llm
