#include "gtest/gtest.h"
#include <gmock/gmock.h>

#include <cstring>

#include "autil/EnvUtil.h"
#include "autil/NetUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreServiceImplContext.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpMessager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
class MockCacheLoadServiceClosure: public RPCClosure {
public:
    MockCacheLoadServiceClosure(const std::shared_ptr<MemoryUtil>&                           memory_util,
                                const std::shared_ptr<RequestBlockBuffer>                    request_block_buffer,
                                arpc::ANetRPCController*                                     controller,
                                CacheLoadRequest*                                            request,
                                CacheLoadResponse*                                           response,
                                CacheStoreLoadDoneCallback                                   callback,
                                const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector) {}

    virtual ~MockCacheLoadServiceClosure() {}

public:
    MOCK_METHOD0(Run, void());
};

class TcpCacheStoreServiceImplContextTest: public CacheStoreTestBase {
protected:
    bool                         initCacheStores();
    bool                         initContext();
    bool                         initContextWithBlock(const std::shared_ptr<BlockBuffer>& block,
                                                      int                                 partition_count,
                                                      int                                 partition_id);
    std::shared_ptr<BlockBuffer> storeHndBlock(const std::string& key);
    void                         loadThreadFunction(int id);
    void storeBlocks(int num);
    void verifyContextRunDone(int                          unloaded_block_cnt,
                              int                          write_cnt,
                              bool                         context_done_run,
                              KvCacheStoreServiceErrorCode error_code);

    void TearDown() override {
        if (done_) {
            delete done_;
        }
        cache_store1_.reset();
        cache_store2_.reset();
    }

protected:
    std::shared_ptr<NormalCacheStore>                     cache_store1_;
    std::shared_ptr<NormalCacheStore>                     cache_store2_;
    CacheLoadRequest*                                     request_;
    CacheLoadResponse*                                    response_;
    std::shared_ptr<CacheStoreServerLoadMetricsCollector> collector_;
    MockCacheLoadServiceClosure*                          done_{nullptr};
    std::shared_ptr<TimerManager>                         timer_manager_;

    std::shared_ptr<TcpCacheStoreServiceImplContext> context_;

    uint32_t port1_;
    uint32_t port2_;
    uint32_t rdma_port1_;
    uint32_t rdma_port2_;
};

bool TcpCacheStoreServiceImplContextTest::initCacheStores() {
    port1_      = autil::NetUtil::randomPort();
    port2_      = autil::NetUtil::randomPort();
    rdma_port1_ = autil::NetUtil::randomPort();
    rdma_port2_ = autil::NetUtil::randomPort();

    CacheStoreInitParams params1;
    params1.listen_port      = port1_;
    params1.rdma_listen_port = rdma_port1_;
    params1.enable_metric    = false;
    params1.memory_util      = memory_util_;

    cache_store1_ = NormalCacheStore::createNormalCacheStore(params1);
    if (!cache_store1_) {
        return false;
    }

    CacheStoreInitParams params2;
    params2.listen_port      = port2_;
    params2.rdma_listen_port = rdma_port2_;
    params2.enable_metric    = false;
    params2.memory_util      = memory_util_;

    cache_store2_ = NormalCacheStore::createNormalCacheStore(params2);
    return cache_store2_ != nullptr;
}
bool TcpCacheStoreServiceImplContextTest::initContext() {
    auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    for (int i = 0; i < 10; i++) {
        auto block = block_buffer_util_->makeBlockBuffer("b" + std::to_string(i), 1024, '0' + i, true);
        request_block_buffer->addBlock(block);
    }

    auto load_request =
        std::make_shared<LoadRequest>("1.2.3.4", 12345, 12346, request_block_buffer, nullptr, 1000, 1, 0);

    request_ = cache_store1_->messager_->makeLoadRequest(load_request);
    if (request_ == nullptr) {
        return false;
    }

    arpc::ANetRPCController* controller = new arpc::ANetRPCController();
    controller->SetExpireTime(1000);

    response_ = new CacheLoadResponse;
    if (response_ == nullptr) {
        return false;
    }
    done_ = new MockCacheLoadServiceClosure(
        memory_util_, request_block_buffer, controller, request_, response_, nullptr, nullptr);

    if (done_ == nullptr) {
        return false;
    }

    timer_manager_ = cache_store1_->messager_->timer_manager_;
    if (timer_manager_ == nullptr) {
        return false;
    }

    auto collector = std::make_shared<CacheStoreServerLoadMetricsCollector>(nullptr, 1, 1, 1);
    context_       = std::make_shared<TcpCacheStoreServiceImplContext>(
        request_, response_, collector, done_, cache_store1_->request_block_buffer_store_);
    return context_ != nullptr;
}

void TcpCacheStoreServiceImplContextTest::verifyContextRunDone(int                          unloaded_block_cnt,
                                                               int                          write_cnt,
                                                               bool                         context_done_run,
                                                               KvCacheStoreServiceErrorCode error_code) {
    ASSERT_EQ(context_->unloaded_blocks_.size(), unloaded_block_cnt);
    ASSERT_EQ(context_->write_cnt_.load(), write_cnt);
    ASSERT_EQ(context_->done_run_.load(), context_done_run);
    ASSERT_EQ(response_->error_code(), error_code);
}

void TcpCacheStoreServiceImplContextTest::storeBlocks(int num) {
    std::string requestid   = "test-request-id";
    auto        store_cache = std::make_shared<RequestBlockBuffer>(requestid);
    for (int i = 0; i < num; i++) {
        auto buffer_block = block_buffer_util_->makeBlockBuffer("b" + std::to_string(i), 1024, 'a' + i, true);
        ASSERT_NE(buffer_block, nullptr);
        store_cache->addBlock(buffer_block);
    }
    std::mutex mutex;  // for sync test
    mutex.lock();
    auto store_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    };

    cache_store2_->store(store_cache, store_callback);
    mutex.lock();
    mutex.unlock();

    auto blocks = cache_store2_->request_block_buffer_store_->request_cache_map_["test-request-id"]->blocks_;
    ASSERT_EQ(blocks.size(), 10);
}

// cache_store2 store block, cache_store1 load from cache_store2
TEST_F(TcpCacheStoreServiceImplContextTest, loadBlock_Success) {
    ASSERT_TRUE(initCacheStores());
    ASSERT_TRUE(initContext());
    ASSERT_EQ(context_->unloaded_blocks_.size(), 10);

    storeBlocks(10);

    EXPECT_CALL(*done_, Run()).Times(1);

    int  load_cnt = 0;
    auto blocks   = cache_store2_->request_block_buffer_store_->request_cache_map_["test-request-id"]->blocks_;
    for (auto& block : blocks) {
        context_->loadBlockOnTcp(true, {block.second});
        ++load_cnt;
        ASSERT_EQ(context_->unloaded_blocks_.size(), 10 - load_cnt);
    }

    verifyContextRunDone(0, 10, true, KvCacheStoreServiceErrorCode::EC_SUCCESS);

    ASSERT_EQ(response_->blocks_size(), 10);
    for (int i = 0; i < 10; i++) {
        for (int i = 0; i < response_->blocks_size(); i++) {
            const auto& block = response_->blocks(i);
            if (block.key() == "b" + std::to_string(i)) {
                ASSERT_EQ(block.content().data()[0], 'a' + i);
                ASSERT_EQ(block.len(), 1024);
                break;
            }
        }
    }
}

TEST_F(TcpCacheStoreServiceImplContextTest, loadBlock_Timeout) {
    ASSERT_TRUE(initCacheStores());
    ASSERT_TRUE(initContext());
    ASSERT_EQ(context_->unloaded_blocks_.size(), 10);

    storeBlocks(10);

    std::mutex mutex;
    mutex.lock();
    auto timer_callback = [this, &mutex]() {
        this->context_->runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
        mutex.unlock();
    };
    auto timer = timer_manager_->addTimer(request_->timeout_ms(), std::move(timer_callback));
    context_->setTimer(timer);
    ASSERT_TRUE(timer != nullptr);

    EXPECT_CALL(*done_, Run()).Times(1);
    std::vector<std::thread> load_threads;
    for (int i = 0; i < 7; ++i) {
        load_threads.emplace_back([this, i]() { this->loadThreadFunction(i); });
    }

    for (auto& t : load_threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    mutex.lock();
    mutex.unlock();

    verifyContextRunDone(3, 7, true, KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
}

TEST_F(TcpCacheStoreServiceImplContextTest, loadBlock_Canceled) {
    ASSERT_TRUE(initCacheStores());
    ASSERT_TRUE(initContext());
    ASSERT_EQ(context_->unloaded_blocks_.size(), 10);

    storeBlocks(10);

    EXPECT_CALL(*done_, Run()).Times(1);
    for (int i = 0; i < 5; i++) {
        auto block = cache_store2_->request_block_buffer_store_->request_cache_map_["test-request-id"]
                         ->blocks_["b" + std::to_string(i)];
        context_->loadBlockOnTcp(true, {block});
        ASSERT_EQ(context_->unloaded_blocks_.size(), 10 - i - 1);
    }

    ASSERT_EQ(context_->unloaded_blocks_.size(), 5);
    ASSERT_EQ(context_->write_cnt_.load(), 5);
    ASSERT_FALSE(context_->done_run_.load());

    // load canceled request, run failed
    context_->loadBlockOnTcp(false, {});

    verifyContextRunDone(5, 5, true, KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
}

TEST_F(TcpCacheStoreServiceImplContextTest, loadBlock_AfterDoneRun) {
    ASSERT_TRUE(initCacheStores());
    ASSERT_TRUE(initContext());
    ASSERT_EQ(context_->unloaded_blocks_.size(), 10);
    storeBlocks(10);

    EXPECT_CALL(*done_, Run()).Times(1);

    // force run failed
    context_->runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
    ASSERT_TRUE(context_->response_ == nullptr);
    verifyContextRunDone(10, 0, true, KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);

    auto block = cache_store2_->request_block_buffer_store_->request_cache_map_["test-request-id"]->blocks_["b0"];
    context_->loadBlockOnTcp(true, {block});
}

namespace {

// [K region][V region] with every head filled with a distinct byte, so a wrong
// slice shows up as a wrong value rather than merely a wrong length.
constexpr int      kHeads        = 4;
constexpr uint32_t kBytesPerHead = 32;
constexpr uint32_t kRegionBytes  = kHeads * kBytesPerHead;
constexpr uint32_t kBlockBytes   = 2 * kRegionBytes;

char headValue(int region, int head) {
    return static_cast<char>(region * 16 + head);
}

}  // namespace

std::shared_ptr<BlockBuffer> TcpCacheStoreServiceImplContextTest::storeHndBlock(const std::string& key) {
    auto block = block_buffer_util_->makeBlockBuffer(key, kBlockBytes, 0, /*gpu=*/false);
    if (block == nullptr) {
        return nullptr;
    }
    auto* base = static_cast<char*>(block->addr.get());
    for (int region = 0; region < 2; ++region) {
        for (int head = 0; head < kHeads; ++head) {
            memset(base + region * kRegionBytes + head * kBytesPerHead, headValue(region, head), kBytesPerHead);
        }
    }

    auto       store_cache = std::make_shared<RequestBlockBuffer>("test-request-id");
    store_cache->addBlock(block);

    std::mutex mutex;
    mutex.lock();
    auto store_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        EXPECT_TRUE(ok);
        EXPECT_EQ(CacheStoreErrorCode::None, ec);
    };
    cache_store2_->store(store_cache, store_callback);
    mutex.lock();
    mutex.unlock();

    return cache_store2_->request_block_buffer_store_->request_cache_map_["test-request-id"]->blocks_[key];
}

bool TcpCacheStoreServiceImplContextTest::initContextWithBlock(const std::shared_ptr<BlockBuffer>& block,
                                                               int                                 partition_count,
                                                               int                                 partition_id) {
    auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    request_block_buffer->addBlock(block);

    auto load_request = std::make_shared<LoadRequest>(
        "1.2.3.4", 12345, 12346, request_block_buffer, nullptr, 1000, partition_count, partition_id);

    request_ = cache_store1_->messager_->makeLoadRequest(load_request);
    if (request_ == nullptr) {
        return false;
    }

    arpc::ANetRPCController* controller = new arpc::ANetRPCController();
    controller->SetExpireTime(1000);
    response_ = new CacheLoadResponse;
    done_     = new MockCacheLoadServiceClosure(
        memory_util_, request_block_buffer, controller, request_, response_, nullptr, nullptr);

    auto collector = std::make_shared<CacheStoreServerLoadMetricsCollector>(nullptr, 1, 1, 1);
    context_       = std::make_shared<TcpCacheStoreServiceImplContext>(
        request_, response_, collector, done_, cache_store1_->request_block_buffer_store_);
    return context_ != nullptr;
}

// A CP prefill holds every KV head while a TP decode wants only its own, so each
// decode rank asks for the same partition inside the K region and the V region.
TEST_F(TcpCacheStoreServiceImplContextTest, loadBlock_KvHeadSliceOfOpaqueBlock) {
    ASSERT_TRUE(initCacheStores());

    constexpr int kPartitionCount = 2;
    for (int partition_id = 0; partition_id < kPartitionCount; ++partition_id) {
        auto stored = storeHndBlock("kv_head_slice");
        ASSERT_NE(stored, nullptr);
        ASSERT_EQ(stored->len, kBlockBytes);

        // What the decode rank owns locally: half the heads, both regions.
        auto requested = block_buffer_util_->makeBlockBuffer("kv_head_slice", kBlockBytes / kPartitionCount, 0, false);
        ASSERT_NE(requested, nullptr);
        requested->partition_count     = kPartitionCount;
        requested->partition_id        = partition_id;
        requested->partition_kv_halves = true;

        // Request-level partition stays at 1 to prove the per-block override wins.
        ASSERT_TRUE(initContextWithBlock(requested, 1, 0));
        EXPECT_CALL(*done_, Run()).Times(1);

        context_->loadBlockOnTcp(true, {stored});
        verifyContextRunDone(0, 1, true, KvCacheStoreServiceErrorCode::EC_SUCCESS);

        ASSERT_EQ(response_->blocks_size(), 1);
        const auto& got = response_->blocks(0);
        ASSERT_EQ(got.len(), kBlockBytes / kPartitionCount);
        ASSERT_EQ(got.content().size(), kBlockBytes / kPartitionCount);

        const int   heads_per_partition = kHeads / kPartitionCount;
        const int   head_begin          = partition_id * heads_per_partition;
        const char* content             = got.content().data();
        for (int region = 0; region < 2; ++region) {
            for (int i = 0; i < heads_per_partition; ++i) {
                const uint32_t off = (region * heads_per_partition + i) * kBytesPerHead;
                for (uint32_t b = 0; b < kBytesPerHead; ++b) {
                    ASSERT_EQ(content[off + b], headValue(region, head_begin + i))
                        << "partition_id=" << partition_id << " region=" << region << " head=" << head_begin + i
                        << " byte=" << b;
                }
            }
        }

        cache_store2_->request_block_buffer_store_->delRequestBlockBuffer("test-request-id");
        delete done_;
        done_ = nullptr;
    }
}

// MiniMax-M3 keeps its indexer K cache in the scale slot. It has no KV-head
// dimension, so every decode rank needs the whole thing even though the
// request-level partition splits the main KV block.
TEST_F(TcpCacheStoreServiceImplContextTest, loadBlock_PerBlockPartitionCountOneReplicates) {
    ASSERT_TRUE(initCacheStores());

    auto stored = storeHndBlock("kv_scale_replicated");
    ASSERT_NE(stored, nullptr);

    auto requested = block_buffer_util_->makeBlockBuffer("kv_scale_replicated", kBlockBytes, 0, false);
    ASSERT_NE(requested, nullptr);
    requested->partition_count = 1;
    requested->partition_id    = 0;

    // Request-level partition of 2 would have cut this block in half and failed.
    ASSERT_TRUE(initContextWithBlock(requested, 2, 1));
    EXPECT_CALL(*done_, Run()).Times(1);

    context_->loadBlockOnTcp(true, {stored});
    verifyContextRunDone(0, 1, true, KvCacheStoreServiceErrorCode::EC_SUCCESS);

    ASSERT_EQ(response_->blocks_size(), 1);
    const auto& got = response_->blocks(0);
    ASSERT_EQ(got.len(), kBlockBytes);
    for (int region = 0; region < 2; ++region) {
        for (int head = 0; head < kHeads; ++head) {
            const uint32_t off = region * kRegionBytes + head * kBytesPerHead;
            ASSERT_EQ(got.content().data()[off], headValue(region, head));
        }
    }
}

// A peer length that is not a whole partition of the stored block is rejected as
// an invalid request instead of silently transferring a wrong-sized slice.
TEST_F(TcpCacheStoreServiceImplContextTest, loadBlock_InconsistentPartitionLenRejected) {
    ASSERT_TRUE(initCacheStores());

    auto stored = storeHndBlock("kv_bad_len");
    ASSERT_NE(stored, nullptr);

    auto requested = block_buffer_util_->makeBlockBuffer("kv_bad_len", kBlockBytes, 0, false);
    ASSERT_NE(requested, nullptr);
    requested->partition_count     = 2;
    requested->partition_id        = 0;
    requested->partition_kv_halves = true;

    ASSERT_TRUE(initContextWithBlock(requested, 1, 0));
    EXPECT_CALL(*done_, Run()).Times(1);

    context_->loadBlockOnTcp(true, {stored});
    verifyContextRunDone(0, 0, true, KvCacheStoreServiceErrorCode::EC_FAILED_INVALID_REQ);
}

void TcpCacheStoreServiceImplContextTest::loadThreadFunction(int i) {
    auto block = cache_store2_->request_block_buffer_store_->request_cache_map_["test-request-id"]
                     ->blocks_["b" + std::to_string(i)];
    context_->loadBlockOnTcp(true, {block});
}

TEST_F(TcpCacheStoreServiceImplContextTest, loadBlockMultiThread) {
    ASSERT_TRUE(initCacheStores());
    ASSERT_TRUE(initContext());
    ASSERT_EQ(context_->unloaded_blocks_.size(), 10);
    storeBlocks(10);

    EXPECT_CALL(*done_, Run()).Times(1);
    std::vector<std::thread> load_threads;
    for (int i = 0; i < 10; ++i) {
        load_threads.emplace_back([this, i]() { this->loadThreadFunction(i); });
    }
    for (auto& t : load_threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    verifyContextRunDone(0, 10, true, KvCacheStoreServiceErrorCode::EC_SUCCESS);

    ASSERT_EQ(response_->blocks_size(), 10);
}
}  // namespace rtp_llm
