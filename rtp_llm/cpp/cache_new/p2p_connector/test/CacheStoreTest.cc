#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <atomic>
#include "rtp_llm/cpp/cache_new/p2p_connector/CacheStoreClient.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/CacheStoreServer.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/TcpClient.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/TcpServer.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/CommonDefs.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/test/MockKVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/test/DeviceUtil.h"
#include "rtp_llm/cpp/core/Event.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace cache_store {

class CacheStoreTest: public ::testing::Test {
public:
    CacheStoreTest() = default;
    ~CacheStoreTest();

protected:
    void SetUp() override;
    void TearDown() override {};

protected:
    std::shared_ptr<KVCacheAllocator> kv_cache_allocator_;
    std::shared_ptr<TcpClient>        tcp_client_;
    std::shared_ptr<TcpServer>        tcp_server_;
    std::shared_ptr<CacheStoreClient> cache_store_client_;
    std::shared_ptr<CacheStoreServer> cache_store_server_;
    std::shared_ptr<DeviceUtil>       device_util_;
    std::string                       server_ip_   = "127.0.0.1";
    uint32_t                          server_port_ = 12345;
};

void CacheStoreTest::SetUp() {
    server_port_ = autil::NetUtil::randomPort();

    device_util_        = std::make_shared<DeviceUtil>();
    kv_cache_allocator_ = std::make_shared<MockKVCacheAllocator>();
    tcp_client_         = std::make_shared<TcpClient>();
    tcp_server_         = std::make_shared<TcpServer>();
    ASSERT_TRUE(tcp_client_->init(1));
    ASSERT_TRUE(tcp_server_->init(1, 1, server_port_, false));
    cache_store_client_ = std::make_shared<CacheStoreClient>(tcp_client_, tcp_server_, device_util_->device_);
    ASSERT_TRUE(cache_store_client_->init());

    std::vector<CacheStoreServerWorker> worker_addrs = {CacheStoreServerWorker(server_ip_, server_port_, 0)};
    cache_store_server_                              = std::make_shared<CacheStoreServer>(
        tcp_client_, tcp_server_, 1, kv_cache_allocator_, worker_addrs, device_util_->device_);
    ASSERT_TRUE(cache_store_server_->init());
    ASSERT_TRUE(tcp_server_->start());
}

CacheStoreTest::~CacheStoreTest() {
    cache_store_server_.reset();
    cache_store_client_.reset();
    tcp_server_.reset();
    tcp_client_.reset();
    device_util_.reset();
}

class CacheStoreGetPeerWorkerInfoTest: public CacheStoreTest {
protected:
    void SetUp() override {
        CacheStoreTest::SetUp();
    }

    void TearDown() override {
        CacheStoreTest::TearDown();
    }
};

TEST_F(CacheStoreGetPeerWorkerInfoTest, ConnectionFailed) {
    auto worker_infos = cache_store_client_->getPeerWorkerInfo("127.0.0.1", 99999);
    EXPECT_TRUE(worker_infos.empty());

    worker_infos = cache_store_client_->getPeerWorkerInfo("192.168.255.255", 12345);
    EXPECT_TRUE(worker_infos.empty());
}

TEST_F(CacheStoreGetPeerWorkerInfoTest, Success) {
    auto worker_infos = cache_store_client_->getPeerWorkerInfo(server_ip_, server_port_);
    EXPECT_EQ(worker_infos.size(), 1);
    EXPECT_EQ(worker_infos[0].ip, "127.0.0.1");
    EXPECT_EQ(worker_infos[0].port, server_port_);
    EXPECT_EQ(worker_infos[0].rdma_port, 0);
}

// ==================== Mock DeviceEvent 用于测试 ====================

class MockDeviceEvent: public DeviceEvent {
public:
    MockDeviceEvent(bool ready = false): ready_(ready) {}

    void synchronize() const override {
        // Mock implementation
    }

    bool checkReadiness() const override {
        return ready_;
    }

    void setReady(bool ready) {
        ready_ = ready;
    }

private:
    mutable std::atomic<bool> ready_;
};

// ==================== Mock Watcher 用于测试 LayerCacheBuffer 回调 ====================

class MockLayerCacheBufferWatcher: public SingleLayerCacheBufferStore::Watcher {
public:
    MockLayerCacheBufferWatcher(int layer_id):
        SingleLayerCacheBufferStore::Watcher(layer_id), notify_count_(0), last_notified_buffer_(nullptr) {}

    bool notify(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer) override {
        notify_count_++;
        last_notified_buffer_ = layer_cache_buffer;
        return false;  // 不消费缓冲区
    }

    int getNotifyCount() const {
        return notify_count_;
    }

    std::shared_ptr<LayerCacheBuffer> getLastNotifiedBuffer() const {
        return last_notified_buffer_;
    }

    void reset() {
        notify_count_         = 0;
        last_notified_buffer_ = nullptr;
    }

private:
    std::atomic<int>                  notify_count_;
    std::shared_ptr<LayerCacheBuffer> last_notified_buffer_;
};

// ==================== asyncStore 测试 ====================

class CacheStoreAsyncStoreTest: public CacheStoreTest {
protected:
    void SetUp() override {
        CacheStoreTest::SetUp();
        layer_id_ = 0;
    }

    std::shared_ptr<LayerCacheBuffer> createLayerCacheBuffer(int layer_id) {
        return std::make_shared<LayerCacheBuffer>(layer_id);
    }

    int layer_id_;
};

TEST_F(CacheStoreAsyncStoreTest, AsyncStoreWithNullEvent) {
    // 测试 event 为 nullptr 的场景 - 应该立即处理
    auto           buffer     = createLayerCacheBuffer(layer_id_);
    DeviceEventPtr null_event = nullptr;

    auto watcher                  = std::make_shared<MockLayerCacheBufferWatcher>(layer_id_);
    auto layer_cache_buffer_store = cache_store_server_->getLayerCacheBufferStore();
    auto layer_store              = layer_cache_buffer_store->getSingleLayerCacheBufferStore(layer_id_);
    ASSERT_NE(layer_store, nullptr);

    int64_t deadline_us = currentTimeMs() + 5000;
    layer_store->setLayerCacheBufferWatchFunc(watcher, deadline_us);

    cache_store_server_->asyncStore(buffer, null_event, 1000);

    // 等待 storeWaitThread 处理（最多等待 100ms）
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_GE(watcher->getNotifyCount(), 1);
}

TEST_F(CacheStoreAsyncStoreTest, AsyncStoreWithNotReadyEvent) {
    auto buffer = createLayerCacheBuffer(layer_id_);
    auto event  = std::make_shared<MockDeviceEvent>(false);  // 未就绪

    auto watcher                  = std::make_shared<MockLayerCacheBufferWatcher>(layer_id_);
    auto layer_cache_buffer_store = cache_store_server_->getLayerCacheBufferStore();
    auto layer_store              = layer_cache_buffer_store->getSingleLayerCacheBufferStore(layer_id_);
    ASSERT_NE(layer_store, nullptr);
    int64_t deadline_us = currentTimeMs() + 5000;
    layer_store->setLayerCacheBufferWatchFunc(watcher, deadline_us);

    cache_store_server_->asyncStore(buffer, event, 1000);

    // 等待一段时间，buffer 不应该被处理
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    ASSERT_EQ(layer_store->layerCacheBufferMapSize(), 0);

    // 现在让 event 就绪
    event->setReady(true);

    // 等待 storeWaitThread 处理
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 现在 watcher 应该被通知
    EXPECT_GE(watcher->getNotifyCount(), 1);
}

TEST_F(CacheStoreAsyncStoreTest, AsyncStoreTimeout) {
    // 测试超时场景 - 超时后应该从队列中删除
    auto buffer = createLayerCacheBuffer(layer_id_);
    auto event  = std::make_shared<MockDeviceEvent>(false);  // 未就绪

    // 使用很短的超时时间（10ms）
    cache_store_server_->asyncStore(buffer, event, 10);

    // 等待超时
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto layer_cache_buffer_store = cache_store_server_->getLayerCacheBufferStore();
    auto layer_store              = layer_cache_buffer_store->getSingleLayerCacheBufferStore(layer_id_);
    ASSERT_NE(layer_store, nullptr);
    ASSERT_EQ(layer_store->layerCacheBufferMapSize(), 0);
}

}  // namespace cache_store
}  // namespace rtp_llm
