#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MessagerClient.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MessagerServer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "autil/NetUtil.h"
#include "autil/EnvUtil.h"

namespace rtp_llm {

class MessagerClientTest: public CacheStoreTestBase {

protected:
    bool initMessager();
    void verifyBlock(
        const std::shared_ptr<BlockBuffer>& block, const std::string& key, uint32_t len, bool gpu_mem, char val);

protected:
    std::shared_ptr<RequestBlockBufferStore> server_buffer_store_;

    std::shared_ptr<MessagerClient> client_;
    std::shared_ptr<MessagerServer> server_;

    uint32_t port_;
};

bool MessagerClientTest::initMessager() {
    server_buffer_store_ =
        std::make_shared<RequestBlockBufferStore>(memory_util_, rtp_llm::DeviceFactory::getDefaultDevice());

    port_ = autil::NetUtil::randomPort();

    client_ = std::make_shared<MessagerClient>(memory_util_);
    if (!client_->init(false)) {
        return false;
    }

    auto timer_manager_    = std::make_shared<arpc::TimerManager>();
    auto metrics_reporter_ = std::make_shared<CacheStoreMetricsReporter>();

    server_ = std::make_shared<MessagerServer>(memory_util_, server_buffer_store_, metrics_reporter_, timer_manager_);
    if (!server_->init(port_, 0, false)) {
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

    auto buf = device_util_->mallocCPU(len);
    ASSERT_TRUE(device_util_->memcopy(buf, false, block->addr.get(), block->gpu_mem, len));
    ASSERT_EQ(val, ((char*)(buf))[0]) << key << " " << reinterpret_cast<uint64_t>(block->addr.get());

    device_util_->freeCPU(buf);
}

TEST_F(MessagerClientTest, testSendLoadRequest_Success) {
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
    client_->load(autil::NetUtil::getBindIp(), port_, 0, load_cache, load_callback, 1000, nullptr, 1, 0);

    mutex.lock();  // wait till callback
    mutex.unlock();

    verifyBlock(load_cache->getBlock("a"), "a", block_size, true, '0');
}

TEST_F(MessagerClientTest, testSendLoadRequest_connectFailed) {
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

    client_->stopTcpClient();
    client_->load(autil::NetUtil::getBindIp(), port_, 0, load_cache, load_callback, 1000, nullptr, 1, 0);

    mutex.lock();  // wait till callback
    mutex.unlock();
}

TEST_F(MessagerClientTest, testSendLoadRequest_sendRequestFailed) {
    ASSERT_TRUE(initMessager());

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

    client_->load("11.22.33.44", port_, 0, load_cache, load_callback, 1000, nullptr, 1, 0);

    mutex.lock();  // wait till callback
    mutex.unlock();
}

}  // namespace rtp_llm