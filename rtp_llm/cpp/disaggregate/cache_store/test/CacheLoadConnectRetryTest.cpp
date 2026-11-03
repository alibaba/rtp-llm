#include "gtest/gtest.h"

#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "autil/NetUtil.h"

namespace rtp_llm {
namespace {

class ScriptedLoadMessager: public Messager {
public:
    ScriptedLoadMessager(): Messager(nullptr, nullptr, nullptr) {}

    bool init(MessagerInitParams) override {
        return true;
    }

    void load(const std::shared_ptr<LoadRequest>&                          request,
              const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector) override {
        const auto code = requests.size() < results.size() ? results[requests.size()] : CacheStoreErrorCode::None;
        requests.push_back(request);
        if (cancel_on_load) {
            cancelled = true;
        }
        collector->markEnd(code == CacheStoreErrorCode::None);
        // Hold only the payload for assertions. The callback owns its LoadContext.
        auto callback = std::move(request->callback);
        callback(code == CacheStoreErrorCode::None, code);
    }

    bool generateBlockInfo(::BlockBufferInfo*, const std::shared_ptr<BlockBuffer>&, uint32_t, uint32_t) override {
        return false;
    }

    std::vector<CacheStoreErrorCode>          results;
    std::vector<std::shared_ptr<LoadRequest>> requests;
    bool                                      cancel_on_load = false;
    std::atomic_bool                          cancelled{false};
};

class CacheLoadConnectRetryTest: public testing::Test {
protected:
    void SetUp() override {
        CacheStoreInitParams params;
        params.listen_port   = autil::NetUtil::randomPort();
        params.rdma_mode     = false;
        params.enable_metric = false;
        store_               = NormalCacheStore::createNormalCacheStore(params);
        ASSERT_TRUE(store_);
        messager_         = std::make_shared<ScriptedLoadMessager>();
        store_->messager_ = messager_;
        buffer_           = std::make_shared<RequestBlockBuffer>("request");
        buffer_->addBlock("block", std::make_shared<char>(0), 1, false, true);
    }

    std::shared_ptr<LoadContext> startLoad(bool combine_load = true) {
        auto load = std::make_shared<LoadContext>(store_, combine_load);
        load->load({buffer_}, "127.0.0.1", 10000, 10001, 5000, [this] { return messager_->cancelled.load(); }, 2, 1);
        return load;
    }

    std::shared_ptr<NormalCacheStore>     store_;
    std::shared_ptr<ScriptedLoadMessager> messager_;
    std::shared_ptr<RequestBlockBuffer>   buffer_;
};

TEST_F(CacheLoadConnectRetryTest, RecoversTransientConnectionErrors) {
    messager_->results = {CacheStoreErrorCode::LoadConnectFailed, CacheStoreErrorCode::LoadRdmaConnectFailed};
    auto load          = startLoad();
    EXPECT_TRUE(load->waitDoneWithConnectRetry(2).ok());
    ASSERT_EQ(messager_->requests.size(), 3);
    const auto& first = messager_->requests.front();
    const auto& last  = messager_->requests.back();
    EXPECT_EQ(last->request_block_buffer->getBlock("block"), buffer_->getBlock("block"));
    EXPECT_EQ(last->rdma_port, first->rdma_port);
    EXPECT_EQ(last->partition_count, 2);
    EXPECT_EQ(last->partition_id, 1);
    EXPECT_LE(last->timeout_ms, first->timeout_ms);
}

TEST_F(CacheLoadConnectRetryTest, ReturnsLastConnectionErrorWhenRetriesExhausted) {
    messager_->results = {CacheStoreErrorCode::LoadConnectFailed, CacheStoreErrorCode::LoadRdmaConnectFailed};
    EXPECT_EQ(startLoad()->waitDoneWithConnectRetry(1).code(), ErrorCode::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED);
    EXPECT_EQ(messager_->requests.size(), 2);
}

TEST_F(CacheLoadConnectRetryTest, DoesNotRetryWriteFailureOrDisabledRetry) {
    messager_->results = {CacheStoreErrorCode::LoadRdmaWriteFailed, CacheStoreErrorCode::LoadConnectFailed};
    EXPECT_EQ(startLoad()->waitDoneWithConnectRetry(2).code(), ErrorCode::CACHE_STORE_LOAD_RDMA_WRITE_FAILED);
    EXPECT_EQ(startLoad()->waitDoneWithConnectRetry(0).code(), ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED);
    EXPECT_EQ(messager_->requests.size(), 2);
}

TEST_F(CacheLoadConnectRetryTest, DoesNotRetryAfterCancellation) {
    messager_->results        = {CacheStoreErrorCode::LoadConnectFailed};
    messager_->cancel_on_load = true;
    EXPECT_EQ(startLoad()->waitDoneWithConnectRetry(2).code(), ErrorCode::CANCELLED);
    EXPECT_EQ(messager_->requests.size(), 1);
}

TEST_F(CacheLoadConnectRetryTest, DoesNotRetryAfterOriginalDeadline) {
    messager_->results = {CacheStoreErrorCode::LoadConnectFailed};
    auto load          = startLoad();
    load->waitDone();
    load->deadline_ms_ = 0;
    EXPECT_EQ(load->waitDoneWithConnectRetry(2).code(), ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT);
    EXPECT_EQ(messager_->requests.size(), 1);
}

TEST_F(CacheLoadConnectRetryTest, DoesNotRetryTcpBatch) {
    messager_->results = {CacheStoreErrorCode::LoadConnectFailed};
    EXPECT_EQ(startLoad(false)->waitDoneWithConnectRetry(2).code(), ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED);
    EXPECT_EQ(messager_->requests.size(), 1);
}

}  // namespace
}  // namespace rtp_llm
