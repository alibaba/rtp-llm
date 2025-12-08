#include <thread>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>

#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorPrefillWorker.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/Event.h"

namespace rtp_llm {

namespace test {

// Mock DeviceEvent for testing
class MockDeviceEvent: public DeviceEvent {
public:
    MockDeviceEvent(bool ready = false): ready_(ready) {}

    void synchronize() const override {
        // No-op for testing
    }

    bool checkReadiness() const override {
        return ready_;
    }

    void setReady(bool ready) {
        ready_ = ready;
    }

private:
    mutable bool ready_;
};

class MockTransferClient: public TransferClient {
public:
    MockTransferClient(): TransferClient(nullptr, nullptr) {}

public:
    struct TransferCallInfo {
        std::string ip;
        uint32_t    port;
        std::string unique_key;
        int         layer_id;
        uint32_t    local_partition_count;
        uint32_t    local_partition_id;
        uint32_t    remote_partition_count;
        uint32_t    remote_partition_id;
    };
    void transfer(const std::string&                       ip,
                  uint32_t                                 port,
                  const std::string&                       unique_key,
                  const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                  uint32_t                                 local_partition_count,
                  uint32_t                                 local_partition_id,
                  uint32_t                                 remote_partition_count,
                  uint32_t                                 remote_partition_id,
                  std::function<void(bool)>                callback,
                  int                                      timeout_ms = 1000) override {
        // 记录调用信息
        TransferCallInfo call_info;
        call_info.ip                     = ip;
        call_info.port                   = port;
        call_info.unique_key             = unique_key;
        call_info.layer_id               = layer_cache_buffer->getLayerId();
        call_info.local_partition_count  = local_partition_count;
        call_info.local_partition_id     = local_partition_id;
        call_info.remote_partition_count = remote_partition_count;
        call_info.remote_partition_id    = remote_partition_id;
        transfer_calls_.push_back(call_info);

        // RTP_LLM_LOG_INFO("MockTransferClient transfer, call_info port: %d", call_info.port);
        // RTP_LLM_LOG_INFO("MockTransferClient transfer, call_info local_partition_count: %d",
        // call_info.local_partition_count); RTP_LLM_LOG_INFO("MockTransferClient transfer, call_info
        // local_partition_id: %d", call_info.local_partition_id); RTP_LLM_LOG_INFO("MockTransferClient transfer,
        // call_info remote_partition_count: %d", call_info.remote_partition_count);
        // RTP_LLM_LOG_INFO("MockTransferClient transfer, call_info remote_partition_id: %d",
        // call_info.remote_partition_id);

        // 根据配置决定是否成功
        bool success = should_succeed_;
        if (layer_success_map_.find(call_info.layer_id) != layer_success_map_.end()) {
            success = layer_success_map_[call_info.layer_id];
        }

        // 异步调用回调（模拟异步行为）
        if (async_callback_) {
            std::thread([callback, success]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                callback(success);
            }).detach();
        } else {
            callback(success);
        }
    }

    void setShouldSucceed(bool should_succeed) {
        should_succeed_ = should_succeed;
    }

    void setLayerSuccess(int layer_id, bool success) {
        layer_success_map_[layer_id] = success;
    }

    void setAsyncCallback(bool async) {
        async_callback_ = async;
    }

    const std::vector<TransferCallInfo>& getTransferCalls() const {
        return transfer_calls_;
    }

    void clearTransferCalls() {
        transfer_calls_.clear();
    }

    int getTransferCallCount() const {
        return transfer_calls_.size();
    }

private:
    bool                          should_succeed_ = true;
    std::map<int, bool>           layer_success_map_;
    bool                          async_callback_ = true;
    std::vector<TransferCallInfo> transfer_calls_;
};

// Test fixture for P2PConnectorPrefillWorker
class P2PConnectorPrefillWorkerTest: public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试用的 GptInitParameter
        gpt_init_parameter_.num_layers_                                     = 2;
        gpt_init_parameter_.cache_store_config.cache_store_rdma_mode        = false;
        gpt_init_parameter_.cache_store_config.messager_io_thread_count     = 1;
        gpt_init_parameter_.cache_store_config.messager_worker_thread_count = 1;
        gpt_init_parameter_.tp_size_                                        = 2;
        gpt_init_parameter_.tp_rank_                                        = 0;

        // 创建测试用的 DeviceBase (使用 nullptr，因为测试不依赖实际设备)
        device_base_ = nullptr;

        // 创建测试用的 KVCacheAllocator (使用 nullptr，因为测试不依赖实际分配器)
        kv_cache_allocator_ = nullptr;

        // 创建 P2PConnectorPrefillWorker（不调用 init，避免创建 TransferClient）
        worker_ = std::make_unique<P2PConnectorPrefillWorker>(gpt_init_parameter_, device_base_, kv_cache_allocator_);

        computed_buffers_     = worker_->getComputedBuffersStore();
        load_contexts_        = worker_->getLoadContexts();
        mock_transfer_client_ = std::make_shared<MockTransferClient>();
        worker_->setTransferClient(mock_transfer_client_);
    }

    void TearDown() override {
        worker_.reset();
    }

    // 创建测试用的 KVCacheResourceV1
    std::shared_ptr<KVCacheResourceV1> createKVCacheResource(int layer_id, int num_blocks = 2) {
        auto resource = std::make_shared<KVCacheResourceV1>();

        // 设置 layer_block_ids（只设置指定的 layer）
        for (int i = 0; i < gpt_init_parameter_.num_layers_; ++i) {
            auto block_ids = std::make_shared<BlockIds>();
            if (i == layer_id) {
                for (int j = 0; j < num_blocks; ++j) {
                    block_ids->block_indices.push_back(j);
                }
            }
            resource->layer_block_ids.push_back(block_ids);
        }

        // 设置 cache_keys
        for (int i = 0; i < num_blocks; ++i) {
            resource->cache_keys.push_back(layer_id * 1000 + i);
        }

        return resource;
    }

    // 创建 Mock DeviceEvent
    std::shared_ptr<MockDeviceEvent> createMockEvent(bool ready = false) {
        return std::make_shared<MockDeviceEvent>(ready);
    }

    // 添加 computed buffer 到 store
    void addComputedBuffer(int64_t request_id, int layer_id, int64_t deadline_ms) {
        auto layer_cache_buffer = createLayerCacheBuffer(layer_id);
        computed_buffers_->addBuffer(request_id, layer_cache_buffer, deadline_ms);
    }

    // 创建测试用的 LayerCacheBuffer
    std::shared_ptr<LayerCacheBuffer> createLayerCacheBuffer(int layer_id, int num_blocks = 2) {
        auto buffer = std::make_shared<LayerCacheBuffer>(layer_id);
        for (int i = 0; i < num_blocks; ++i) {
            int64_t cache_key = layer_id * 1000 + i;
            int     block_id  = i;
            buffer->addBlockId(cache_key, block_id);
        }
        return buffer;
    }

protected:
    GptInitParameter                               gpt_init_parameter_;
    DeviceBase*                                    device_base_;
    std::shared_ptr<KVCacheAllocator>              kv_cache_allocator_;
    std::unique_ptr<P2PConnectorPrefillWorker>     worker_;
    std::shared_ptr<ComputedLayerCacheBufferStore> computed_buffers_;
    std::shared_ptr<PrefillWorkerLoadContextStore> load_contexts_;
    std::shared_ptr<MockTransferClient>            mock_transfer_client_;
};

// ---------------------------- writeByLayer ----------------------------

TEST_F(P2PConnectorPrefillWorkerTest, WriteByLayer_ReturnTrue_WithNullEvent) {
    int            layer_id   = 0;
    int64_t        request_id = 1001;
    auto           resource   = createKVCacheResource(layer_id, 2);
    DeviceEventPtr event      = nullptr;  // null event should be treated as ready

    // 执行 writeByLayer
    bool success = worker_->writeByLayer(layer_id, resource, request_id, event);
    EXPECT_TRUE(success);
    EXPECT_TRUE(computed_buffers_->getBuffer(request_id) == nullptr);

    // process store wait thread
    worker_->storeWaitThreadProcess();
    auto computed_buffer = computed_buffers_->getBuffer(request_id);
    ASSERT_TRUE(computed_buffer != nullptr);
}

TEST_F(P2PConnectorPrefillWorkerTest, WriteByLayer_ReturnTrue_WithReadyEvent) {
    int     layer_id   = 0;
    int64_t request_id = 1001;
    auto    resource   = createKVCacheResource(layer_id, 2);
    auto    event      = createMockEvent(false);  // ready event

    // 执行 writeByLayer
    bool success = worker_->writeByLayer(layer_id, resource, request_id, event);
    EXPECT_TRUE(success);
    EXPECT_TRUE(computed_buffers_->getBuffer(request_id) == nullptr);

    worker_->storeWaitThreadProcess();
    auto computed_buffer = computed_buffers_->getBuffer(request_id);
    ASSERT_TRUE(computed_buffer == nullptr);

    event->setReady(true);
    worker_->storeWaitThreadProcess();
    ASSERT_TRUE(computed_buffers_->getBuffer(request_id) != nullptr);
}

TEST_F(P2PConnectorPrefillWorkerTest, StoreWaitThread_TimeoutExpired) {
    int     layer_id   = 0;
    int64_t request_id = 1001;
    auto    resource   = createKVCacheResource(layer_id, 2);
    auto    event      = createMockEvent(false);  // not ready event

    worker_->setStoreWaitTimeoutMs(10);

    // 执行 writeByLayer
    bool success = worker_->writeByLayer(layer_id, resource, request_id, event);
    EXPECT_TRUE(success);

    ASSERT_TRUE(computed_buffers_->getBuffer(request_id) == nullptr);

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    worker_->storeWaitThreadProcess();

    ASSERT_TRUE(computed_buffers_->getBuffer(request_id) == nullptr);
}

// ---------------------------- write ----------------------------

// 测试：多层传输成功
TEST_F(P2PConnectorPrefillWorkerTest, Write_ReturnTrue_AllLayersTransferSuccess) {
    int64_t     request_id  = 1001;
    std::string unique_key  = "test_all_success";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    // decode 2 prefill 2
    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    // 设置所有传输都成功
    mock_transfer_client_->setShouldSucceed(true);
    mock_transfer_client_->setAsyncCallback(true);

    // 预先添加所有层的 computed buffer
    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    // 在后台线程中执行 write（因为 write 会阻塞等待）
    std::atomic<bool> write_done{false};
    std::atomic<bool> write_result{false};
    std::thread       write_thread([&]() {
        write_result = worker_->write(request_id, unique_key, deadline_ms, decode_transfer_servers);
        write_done   = true;
    });

    // 等待 write 完成
    int wait_count = 0;
    while (!write_done && wait_count < 1000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (write_thread.joinable()) {
        write_thread.join();
    }

    // 验证结果
    EXPECT_TRUE(write_result);
    EXPECT_TRUE(write_done);

    // 验证 transfer 被调用了正确的次数
    // 3 层 * 1 个服务器 = 3 次调用
    EXPECT_EQ(mock_transfer_client_->getTransferCallCount(), 2);

    // 验证所有层的传输都被调用
    const auto&   calls = mock_transfer_client_->getTransferCalls();
    std::set<int> transferred_layers;
    for (const auto& call : calls) {
        transferred_layers.insert(call.layer_id);
        ASSERT_TRUE(call.layer_id == 0 || call.layer_id == 1);
        EXPECT_EQ("127.0.0.1", call.ip);
        EXPECT_EQ(12345, call.port);
    }
    EXPECT_EQ(transferred_layers.size(), 2);
    EXPECT_TRUE(transferred_layers.find(0) != transferred_layers.end());
    EXPECT_TRUE(transferred_layers.find(1) != transferred_layers.end());
}

// 测试：部分层传输失败
TEST_F(P2PConnectorPrefillWorkerTest, Write_ReturnFalse_PartialLayersTransferFailed) {
    int64_t     request_id  = 1002;
    std::string unique_key  = "test_partial_fail";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    // 设置 layer 1 的传输失败
    mock_transfer_client_->setShouldSucceed(true);
    mock_transfer_client_->setLayerSuccess(1, false);  // layer 1 失败
    mock_transfer_client_->setAsyncCallback(true);

    // 预先添加所有层的 computed buffer
    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    // 在后台线程中执行 write
    std::atomic<bool> write_done{false};
    std::atomic<bool> write_result{false};
    std::thread       write_thread([&]() {
        write_result = worker_->write(request_id, unique_key, deadline_ms, decode_transfer_servers);
        write_done   = true;
    });

    // 等待 write 完成
    int wait_count = 0;
    while (!write_done && wait_count < 1000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (write_thread.joinable()) {
        write_thread.join();
    }

    // 验证结果（部分失败应该返回 false）
    EXPECT_FALSE(write_result);
    EXPECT_TRUE(write_done);

    // 验证所有层都被尝试传输了
    EXPECT_EQ(mock_transfer_client_->getTransferCallCount(), 2);  // 2 层 * 1 个服务器
}

// 测试：部分层没有传输（没有 computed buffer）
TEST_F(P2PConnectorPrefillWorkerTest, Write_ReturnFalse_SomeLayersNotTransferred) {
    int64_t     request_id  = 1003;
    std::string unique_key  = "test_some_layers_missing";
    int64_t     deadline_ms = currentTimeMs() + 50;  // 50ms timeout

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    // 设置所有传输都成功
    mock_transfer_client_->setShouldSucceed(true);
    mock_transfer_client_->setAsyncCallback(true);

    // 只添加 layer 0 和 layer 2 的 computed buffer，缺少 layer 1
    addComputedBuffer(request_id, 0, deadline_ms);

    // 在后台线程中执行 write
    std::atomic<bool> write_done{false};
    std::atomic<bool> write_result{false};
    std::thread       write_thread([&]() {
        write_result = worker_->write(request_id, unique_key, deadline_ms, decode_transfer_servers);
        write_done   = true;
    });

    // 等待 write 完成（由于缺少 layer 1，write 会超时或返回 false）
    int wait_count = 0;
    while (!write_done && wait_count < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    EXPECT_TRUE(wait_count < 100);

    if (write_thread.joinable()) {
        write_thread.join();
    }

    // 验证结果（缺少层应该返回 false，因为 isDone() 永远不会为 true）
    EXPECT_FALSE(write_result);

    // 验证只有已添加的层被传输了
    const auto&   calls = mock_transfer_client_->getTransferCalls();
    std::set<int> transferred_layers;
    for (const auto& call : calls) {
        transferred_layers.insert(call.layer_id);
    }
    // 应该只有 layer 0 被传输
    EXPECT_EQ(transferred_layers.size(), 1);
    EXPECT_TRUE(transferred_layers.find(0) != transferred_layers.end());
}

// 测试：多层传输成功
TEST_F(P2PConnectorPrefillWorkerTest, Write_ReturnTrue_AsymmetricTP_AllLayersTransferSuccess) {
    int64_t     request_id  = 1001;
    std::string unique_key  = "test_all_success";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    // decode 2 prefill 2
    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});
    decode_transfer_servers.push_back({"127.0.0.1", 12347});
    decode_transfer_servers.push_back({"127.0.0.1", 12348});

    // 设置所有传输都成功
    mock_transfer_client_->setShouldSucceed(true);
    mock_transfer_client_->setAsyncCallback(true);

    // 预先添加所有层的 computed buffer
    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    // 在后台线程中执行 write（因为 write 会阻塞等待）
    std::atomic<bool> write_done{false};
    std::atomic<bool> write_result{false};
    std::thread       write_thread([&]() {
        write_result = worker_->write(request_id, unique_key, deadline_ms, decode_transfer_servers);
        write_done   = true;
    });

    // 等待 write 完成
    int wait_count = 0;
    while (!write_done && wait_count < 1000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (write_thread.joinable()) {
        write_thread.join();
    }

    // 验证结果
    EXPECT_TRUE(write_result);
    EXPECT_TRUE(write_done);

    // 验证 transfer 被调用了正确的次数
    // 3 层 * 1 个服务器 = 3 次调用
    EXPECT_EQ(mock_transfer_client_->getTransferCallCount(), 4);

    // 验证所有层的传输都被调用
    const auto&   calls = mock_transfer_client_->getTransferCalls();
    std::set<int> transferred_layers;
    for (int i = 0; i < calls.size(); i++) {
        const auto& call = calls[i];
        transferred_layers.insert(call.layer_id);
        if (i == 0 || i == 2) {
            EXPECT_EQ(call.ip, "127.0.0.1");
            EXPECT_EQ(call.port, 12345);
        } else if (i == 1 || i == 3) {
            EXPECT_EQ(call.ip, "127.0.0.1");
            EXPECT_EQ(call.port, 12346);
        }
    }
    EXPECT_EQ(transferred_layers.size(), 2);
    EXPECT_TRUE(transferred_layers.find(0) != transferred_layers.end());
    EXPECT_TRUE(transferred_layers.find(1) != transferred_layers.end());
}

}  // namespace test
}  // namespace rtp_llm
