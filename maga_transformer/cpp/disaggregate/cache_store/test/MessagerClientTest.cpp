#include "gtest/gtest.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MessagerClient.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MessagerServer.h"
#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "maga_transformer/cpp/disaggregate/cache_store/BlockBufferUtil.h"
#include "maga_transformer/cpp/disaggregate/cache_store/Interface.h"
#include "maga_transformer/cpp/disaggregate/cache_store/test/MockMemoryUtil.h"
#include "autil/NetUtil.h"
#include "autil/EnvUtil.h"

namespace rtp_llm {

class MessagerClientTest: public ::testing::Test {

protected:
    bool initMemoryUtil(bool mock);
    bool initMessager(bool mock, bool rdma = false);
    void verifyBlock(
        const std::shared_ptr<BlockBuffer>& block, const std::string& key, uint32_t len, bool gpu_mem, char val);

protected:
    MockMemoryUtil*                          mock_memory_util_{nullptr};
    std::shared_ptr<MemoryUtil>              memory_util_;
    std::shared_ptr<BlockBufferUtil>         block_buffer_util_;
    std::shared_ptr<RequestBlockBufferStore> server_buffer_store_;

    std::shared_ptr<MessagerClient> client_;
    std::shared_ptr<MessagerServer> server_;
};

bool MessagerClientTest::initMemoryUtil(bool mock) {
    if (mock) {
        mock_memory_util_ = new MockMemoryUtil(createMemoryUtilImpl(autil::EnvUtil::getEnv(kEnvRdmaMode, false)));
        memory_util_.reset(mock_memory_util_);
    } else {
        memory_util_ = std::make_shared<MemoryUtil>(createMemoryUtilImpl(autil::EnvUtil::getEnv(kEnvRdmaMode, false)));
    }

    block_buffer_util_   = std::make_shared<BlockBufferUtil>(memory_util_);
    server_buffer_store_ = std::make_shared<RequestBlockBufferStore>(memory_util_, nullptr);
    return true;
}

bool MessagerClientTest::initMessager(bool mock, bool rdma) {
    if (!initMemoryUtil(mock)) {
        return false;
    }

    auto port = autil::NetUtil::randomPort();

    client_ = std::make_shared<MessagerClient>(memory_util_);
    if (!client_->init(port, 0, false)) {
        return false;
    }

    auto timer_manager_ = std::make_shared<arpc::TimerManager>();
    auto metrics_reporter_ = std::make_shared<CacheStoreMetricsReporter>();

    server_ = std::make_shared<MessagerServer>(memory_util_, server_buffer_store_, metrics_reporter_, timer_manager_);
    if (!server_->init(port, 0, false)) {
        return false;
    }
    return true;
}

void MessagerClientTest::verifyBlock(
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

    auto buf = memory_util_->mallocCPU(len);
    ASSERT_TRUE(memory_util_->memcopy(buf, false, block->addr.get(), block->gpu_mem, len));
    ASSERT_EQ(val, ((char*)(buf))[0]) << key << " " << reinterpret_cast<uint64_t>(block->addr.get());

    memory_util_->freeCPU(buf);
}

TEST_F(MessagerClientTest, testSendLoadRequest_Success) {
    ASSERT_TRUE(initMessager(false));

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
    client_->load(autil::NetUtil::getBindIp(), load_cache, load_callback, 1000, nullptr);

    mutex.lock();  // wait till callback
    mutex.unlock();

    verifyBlock(load_cache->getBlock("a"), "a", block_size, true, '0');
}

TEST_F(MessagerClientTest, testSendLoadRequest_connectFailed) {
    ASSERT_TRUE(initMessager(false));

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

    client_->stopTcpClient();
    client_->load(autil::NetUtil::getBindIp(), load_cache, load_callback, 1000, nullptr);

    mutex.lock();  // wait till callback
    mutex.unlock();
}

TEST_F(MessagerClientTest, testSendLoadRequest_sendRequestFailed) {
    ASSERT_TRUE(initMessager(true));

    uint32_t    block_size = 16;
    std::string requestid  = "test-request-id";
    auto        load_cache = std::make_shared<RequestBlockBuffer>(requestid);
    load_cache->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, 'a', true));

    std::mutex mutex;  // for sync test
    auto       load_callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        mutex.unlock();
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::CallPrefillTimeout, ec);
    };
    mutex.lock();

    client_->load("11.22.33.44", load_cache, load_callback, 1000, nullptr);

    mutex.lock();  // wait till callback
    mutex.unlock();
}

}  // namespace rtp_llm