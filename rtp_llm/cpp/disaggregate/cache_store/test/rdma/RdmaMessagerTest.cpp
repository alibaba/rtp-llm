#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaMessager.h"
#include "autil/NetUtil.h"
#include "autil/EnvUtil.h"

namespace rtp_llm {

class RdmaMessagerTest: public CacheStoreTestBase {

protected:
    bool initMessager();

protected:
    std::shared_ptr<RequestBlockBufferStore> client_buffer_store_;
    std::shared_ptr<RequestBlockBufferStore> server_buffer_store_;

    std::shared_ptr<Messager> client_;
    std::shared_ptr<Messager> server_;

    uint32_t                     client_port_;
    uint32_t                     client_rdma_port_;
    uint32_t                     server_port_;
    uint32_t                     server_rdma_port_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
};

bool RdmaMessagerTest::initMessager() {
    client_port_      = autil::NetUtil::randomPort();
    client_rdma_port_ = autil::NetUtil::randomPort();
    client_buffer_store_ =
        std::make_shared<RequestBlockBufferStore>(memory_util_);
    client_ = std::make_shared<RdmaMessager>(memory_util_, client_buffer_store_, metrics_reporter_);
    MessagerInitParams client_init_params{client_port_};
    client_init_params.rdma_server_port = client_rdma_port_;
    if (!client_->init(client_init_params)) {
        return false;
    }

    server_port_      = autil::NetUtil::randomPort();
    server_rdma_port_ = autil::NetUtil::randomPort();
    server_buffer_store_ =
        std::make_shared<RequestBlockBufferStore>(memory_util_);
    server_ = std::make_shared<RdmaMessager>(memory_util_, server_buffer_store_, metrics_reporter_);
    MessagerInitParams server_init_params{server_port_};
    server_init_params.rdma_server_port = server_rdma_port_;
    if (!server_->init(server_init_params)) {
        return false;
    }
    return true;
}

TEST_F(RdmaMessagerTest, testSendLoadRequest_Success) {
    if (!memory_util_->isRdmaMode()) {
        return;
    }

    ASSERT_TRUE(initMessager());
    uint32_t    block_size   = 16;
    std::string requestid    = "test-request-id";
    auto        store_buffer = std::make_shared<RequestBlockBuffer>(requestid);
    store_buffer->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));
    ASSERT_TRUE(server_buffer_store_->setRequestBlockBuffer(store_buffer));

    auto load_cache = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, 'a', true));

    std::mutex mutex;  // for sync test
    auto       load_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
    };

    mutex.lock();

    auto load_request = std::make_shared<LoadRequest>(
        autil::NetUtil::getBindIp(), server_port_, server_rdma_port_, load_cache, load_callback, 1000, 1, 0);
    auto collector = std::make_shared<CacheStoreClientLoadMetricsCollector>(nullptr, 1, 1);
    client_->load(load_request, collector);

    mutex.lock();  // wait till callback
    mutex.unlock();

    block_buffer_util_->verifyBlock(load_cache->getBlock("a"), "a", block_size, true, '0');
}

TEST_F(RdmaMessagerTest, testSendLoadRequest_connectFailed) {
    if (!memory_util_->isRdmaMode()) {
        return;
    }

    ASSERT_TRUE(initMessager());

    uint32_t    block_size = 16;
    std::string requestid  = "test-request-id";
    auto        load_cache = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, 'a', true));

    std::mutex mutex;  // for sync test
    auto       load_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::LoadBufferTimeout, ec);
    };
    mutex.lock();

    auto load_request = std::make_shared<LoadRequest>(
        autil::NetUtil::getBindIp(), server_port_, server_rdma_port_, load_cache, load_callback, 1000, 1, 0);
    auto collector = std::make_shared<CacheStoreClientLoadMetricsCollector>(nullptr, 1, 1);
    client_->load(load_request, collector);

    mutex.lock();  // wait till callback
    mutex.unlock();
}

TEST_F(RdmaMessagerTest, testSendLoadRequest_makeRequestFailed) {
    ASSERT_TRUE(initMockMemoryUtil());
    if (!memory_util_->isRdmaMode()) {
        return;
    }
    ASSERT_TRUE(initMessager());

    uint32_t    block_size = 16;
    std::string requestid  = "test-request-id";
    auto        load_cache = std::make_shared<RequestBlockBuffer>(requestid);
    EXPECT_CALL(*mock_memory_util_, regUserMr(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(true));
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, 'a', true));

    std::mutex mutex;  // for sync test
    auto       load_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::LoadSendRequestFailed, ec);
    };
    mutex.lock();

    EXPECT_CALL(*mock_memory_util_, findMemoryMr(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));

    auto load_request = std::make_shared<LoadRequest>(
        autil::NetUtil::getBindIp(), server_port_, server_rdma_port_, load_cache, load_callback, 1000, 1, 0);
    auto collector = std::make_shared<CacheStoreClientLoadMetricsCollector>(nullptr, 1, 1);
    client_->load(load_request, collector);

    mutex.lock();  // wait till callback
    mutex.unlock();
}

TEST_F(RdmaMessagerTest, testSendLoadRequest_sendRequestFailed) {
    if (!memory_util_->isRdmaMode()) {
        return;
    }

    ASSERT_TRUE(initMessager());

    uint32_t    block_size = 16;
    std::string requestid  = "test-request-id";
    auto        load_cache = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, 'a', true));

    std::mutex mutex;  // for sync test
    auto       load_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::LoadConnectFailed, ec);
    };
    mutex.lock();

    auto load_request =
        std::make_shared<LoadRequest>("1.2.3.4", server_port_, 0, load_cache, load_callback, 1000, 1, 0);
    auto collector = std::make_shared<CacheStoreClientLoadMetricsCollector>(nullptr, 1, 1);
    client_->load(load_request, collector);

    mutex.lock();  // wait till callback
    mutex.unlock();
}

}  // namespace rtp_llm
