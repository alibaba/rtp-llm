#include <thread>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <map>

#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorWorker.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerBlockConvertor.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
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
        return ready_.load();
    }

    void setReady(bool ready) {
        ready_.store(ready);
    }

private:
    mutable std::atomic<bool> ready_;
};

// Mock LayerBlockConvertor for testing
class MockLayerBlockConvertor: public LayerBlockConvertor {
public:
    std::vector<BufferPtr>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const override {
        // Return empty buffers for testing
        return {};
    }

    std::vector<std::pair<BufferPtr, size_t>> getAllBuffers() const override {
        return {};
    }
};

// Mock TransferClient for testing
class MockTransferClient: public TransferClient {
public:
    MockTransferClient(): TransferClient(nullptr, nullptr, nullptr) {}

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
        TransferCallInfo call_info;
        call_info.ip                     = ip;
        call_info.port                   = port;
        call_info.unique_key             = unique_key;
        call_info.layer_id               = layer_cache_buffer->getLayerId();
        call_info.local_partition_count  = local_partition_count;
        call_info.local_partition_id     = local_partition_id;
        call_info.remote_partition_count = remote_partition_count;
        call_info.remote_partition_id    = remote_partition_id;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            transfer_calls_.push_back(call_info);
        }

        bool success = should_succeed_;
        if (layer_success_map_.find(call_info.layer_id) != layer_success_map_.end()) {
            success = layer_success_map_[call_info.layer_id];
        }

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

    std::vector<TransferCallInfo> getTransferCalls() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return transfer_calls_;
    }

    void clearTransferCalls() {
        std::lock_guard<std::mutex> lock(mutex_);
        transfer_calls_.clear();
    }

    int getTransferCallCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return transfer_calls_.size();
    }

private:
    bool                          should_succeed_ = true;
    std::map<int, bool>           layer_success_map_;
    bool                          async_callback_ = true;
    mutable std::mutex            mutex_;
    std::vector<TransferCallInfo> transfer_calls_;
};

// Test fixture for P2PConnectorWorker
class P2PConnectorWorkerTest: public ::testing::Test {
protected:
    void SetUp() override {
        // 设置配置
        cache_config_                                    = KVCacheConfig();
        cache_store_config_                              = CacheStoreConfig();
        cache_store_config_.cache_store_rdma_mode        = false;
        cache_store_config_.messager_io_thread_count     = 1;
        cache_store_config_.messager_worker_thread_count = 1;

        parallelism_config_         = ParallelismConfig();
        parallelism_config_.tp_size = 2;
        parallelism_config_.tp_rank = 0;

        pd_sep_config_                         = PDSepConfig();
        pd_sep_config_.cache_store_listen_port = 0;

        model_config_            = ModelConfig();
        model_config_.num_layers = 2;

        // 创建 Mock LayerBlockConvertor
        mock_layer_block_convertor_ = std::make_shared<MockLayerBlockConvertor>();

        // 创建 P2PConnectorWorker（不调用 init，避免创建实际的 TransferClient/Server）
        worker_ = std::make_unique<P2PConnectorWorker>(cache_config_,
                                                       cache_store_config_,
                                                       parallelism_config_,
                                                       pd_sep_config_,
                                                       model_config_,
                                                       mock_layer_block_convertor_,
                                                       nullptr);

        // 获取内部存储（通过 -fno-access-control 可以访问私有成员）
        computed_buffers_ = worker_->getComputedBuffersStore();
        load_contexts_    = worker_->getLoadContexts();

        // 创建并设置 Mock TransferClient
        mock_transfer_client_     = std::make_shared<MockTransferClient>();
        worker_->transfer_client_ = mock_transfer_client_;

        // 创建 StoreWaitContextChecker
        worker_->store_wait_context_checker_ = std::make_shared<StoreWaitContextChecker>(nullptr, computed_buffers_);

        // 创建 AsymmetricTpUtil
        worker_->asymmetric_tp_util_ = std::make_shared<AsymmetricTpUtil>(parallelism_config_);

        // 创建 LayerCacheBufferTaskStore（用于 read 测试）
        task_store_                             = std::make_shared<LayerCacheBufferTaskStore>();
        worker_->layer_cache_buffer_task_store_ = task_store_;
    }

    void TearDown() override {
        worker_.reset();
    }

    // 创建测试用的 KVCacheResource
    KVCacheResourcePtr createKVCacheResource(int layer_id, int num_blocks = 2) {
        auto resource = std::make_shared<KVCacheResource>();

        for (int i = 0; i < model_config_.num_layers; ++i) {
            auto block_ids = std::make_shared<BlockIds>();
            if (i == layer_id) {
                for (int j = 0; j < num_blocks; ++j) {
                    block_ids->block_indices.push_back(j);
                }
            }
            resource->layer_block_ids.push_back(block_ids);
        }

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

    // 模拟任务完成
    void simulateTaskDone(const std::string& unique_key, const std::vector<int>& layer_ids, bool all_success = true) {
        auto task = task_store_->getTask(unique_key);
        if (task) {
            for (int layer_id : layer_ids) {
                // 使用 partition_count=1, partition_id=0 表示单分区场景
                auto layer_cache_buffer = task->loadingLayerCacheBuffer(layer_id, 1, 0);
                if (layer_cache_buffer) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    task->notifyDone(layer_id, all_success, 1, 0);
                }
            }
        }
    }

protected:
    KVCacheConfig                                  cache_config_;
    CacheStoreConfig                               cache_store_config_;
    ParallelismConfig                              parallelism_config_;
    PDSepConfig                                    pd_sep_config_;
    ModelConfig                                    model_config_;
    std::shared_ptr<MockLayerBlockConvertor>       mock_layer_block_convertor_;
    std::unique_ptr<P2PConnectorWorker>            worker_;
    std::shared_ptr<ComputedLayerCacheBufferStore> computed_buffers_;
    std::shared_ptr<PrefillWorkerLoadContextStore> load_contexts_;
    std::shared_ptr<MockTransferClient>            mock_transfer_client_;
    std::shared_ptr<LayerCacheBufferTaskStore>     task_store_;
};

// ==================== writeByLayer 测试 (Prefill 端) ====================

TEST_F(P2PConnectorWorkerTest, WriteByLayer_ReturnTrue_WithReadyEvent) {
    int     layer_id   = 0;
    int64_t request_id = 1002;
    auto    resource   = createKVCacheResource(layer_id, 2);
    auto    event      = createMockEvent(false);

    bool success = worker_->writeByLayer(layer_id, resource, request_id, event);
    EXPECT_TRUE(success);
    EXPECT_EQ(computed_buffers_->getBuffer(request_id), nullptr);

    worker_->loopCheckProc();
    auto computed_buffer = computed_buffers_->getBuffer(request_id);
    EXPECT_EQ(computed_buffer, nullptr);

    event->setReady(true);
    worker_->loopCheckProc();
    ASSERT_NE(computed_buffers_->getBuffer(request_id), nullptr);
}

// ==================== handleRead 测试 (Prefill 端) ====================

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnTrue_AllLayersTransferSuccess) {
    int64_t     request_id  = 2001;
    std::string unique_key  = "test_all_success";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    mock_transfer_client_->setShouldSucceed(true);
    mock_transfer_client_->setAsyncCallback(true);

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    std::atomic<bool> result{false};
    std::thread       write_thread([&]() {
        result = worker_->handleRead(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    int wait_count = 0;
    while (!done && wait_count < 1000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (write_thread.joinable()) {
        write_thread.join();
    }

    EXPECT_TRUE(result);
    EXPECT_TRUE(done);
    EXPECT_EQ(mock_transfer_client_->getTransferCallCount(), 2);

    auto          calls = mock_transfer_client_->getTransferCalls();
    std::set<int> transferred_layers;
    for (const auto& call : calls) {
        transferred_layers.insert(call.layer_id);
        EXPECT_EQ("127.0.0.1", call.ip);
        EXPECT_EQ(12345, call.port);
    }
    EXPECT_EQ(transferred_layers.size(), 2);
}

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnFalse_PartialLayersTransferFailed) {
    int64_t     request_id  = 2002;
    std::string unique_key  = "test_partial_fail";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    mock_transfer_client_->setShouldSucceed(true);
    mock_transfer_client_->setLayerSuccess(1, false);
    mock_transfer_client_->setAsyncCallback(true);

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    std::atomic<bool> result{false};
    std::thread       write_thread([&]() {
        result = worker_->handleRead(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    int wait_count = 0;
    while (!done && wait_count < 1000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (write_thread.joinable()) {
        write_thread.join();
    }

    EXPECT_FALSE(result);
    EXPECT_TRUE(done);
    EXPECT_EQ(mock_transfer_client_->getTransferCallCount(), 2);
}

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnFalse_SomeLayersNotTransferred) {
    int64_t     request_id  = 2003;
    std::string unique_key  = "test_some_layers_missing";
    int64_t     deadline_ms = currentTimeMs() + 50;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    mock_transfer_client_->setShouldSucceed(true);
    mock_transfer_client_->setAsyncCallback(true);

    // 只添加 layer 0
    addComputedBuffer(request_id, 0, deadline_ms);

    std::atomic<bool> done{false};
    std::atomic<bool> result{false};
    std::thread       write_thread([&]() {
        result = worker_->handleRead(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    int wait_count = 0;
    while (!done && wait_count < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    EXPECT_TRUE(wait_count < 100);

    if (write_thread.joinable()) {
        write_thread.join();
    }

    EXPECT_FALSE(result);

    auto          calls = mock_transfer_client_->getTransferCalls();
    std::set<int> transferred_layers;
    for (const auto& call : calls) {
        transferred_layers.insert(call.layer_id);
    }
    EXPECT_EQ(transferred_layers.size(), 1);
    EXPECT_TRUE(transferred_layers.find(0) != transferred_layers.end());
}

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnTrue_AsymmetricTP_AllLayersTransferSuccess) {
    int64_t     request_id  = 2004;
    std::string unique_key  = "test_asymmetric_all_success";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});
    decode_transfer_servers.push_back({"127.0.0.1", 12347});
    decode_transfer_servers.push_back({"127.0.0.1", 12348});

    mock_transfer_client_->setShouldSucceed(true);
    mock_transfer_client_->setAsyncCallback(true);

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    std::atomic<bool> result{false};
    std::thread       write_thread([&]() {
        result = worker_->handleRead(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    int wait_count = 0;
    while (!done && wait_count < 1000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (write_thread.joinable()) {
        write_thread.join();
    }

    EXPECT_TRUE(result);
    EXPECT_TRUE(done);
    EXPECT_EQ(mock_transfer_client_->getTransferCallCount(), 4);

    auto          calls = mock_transfer_client_->getTransferCalls();
    std::set<int> transferred_layers;
    for (const auto& call : calls) {
        transferred_layers.insert(call.layer_id);
    }
    EXPECT_EQ(transferred_layers.size(), 2);
}

// ==================== read 测试 (Decode 端) ====================

TEST_F(P2PConnectorWorkerTest, Read_ReturnTrue_AllLayersSuccess) {
    std::string unique_key  = "test_read_success";
    int64_t     request_id  = 3001;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));
    layer_cache_buffers.push_back(createLayerCacheBuffer(1, 2));

    std::thread completion_thread([this, unique_key]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        simulateTaskDone(unique_key, {0, 1}, true);
    });

    bool success = worker_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    completion_thread.join();

    EXPECT_TRUE(success);
    EXPECT_EQ(task_store_->getTask(unique_key), nullptr);
}

TEST_F(P2PConnectorWorkerTest, Read_ReturnFalse_PartialLayersFailed) {
    std::string unique_key  = "test_read_partial_fail";
    int64_t     request_id  = 3002;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));
    layer_cache_buffers.push_back(createLayerCacheBuffer(1, 2));

    std::thread completion_thread([this, unique_key]() {
        auto task = task_store_->getTask(unique_key);
        if (task) {
            // 使用 partition_count=1, partition_id=0 表示单分区场景
            auto buf0 = task->loadingLayerCacheBuffer(0, 1, 0);
            if (buf0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                task->notifyDone(0, true, 1, 0);
            }
            auto buf1 = task->loadingLayerCacheBuffer(1, 1, 0);
            if (buf1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                task->notifyDone(1, false, 1, 0);
            }
        }
    });

    bool success = worker_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    completion_thread.join();

    EXPECT_FALSE(success);
    EXPECT_EQ(task_store_->getTask(unique_key), nullptr);
}

TEST_F(P2PConnectorWorkerTest, Read_ReturnFalse_Timeout) {
    std::string unique_key  = "test_read_timeout";
    int64_t     request_id  = 3003;
    int64_t     deadline_ms = currentTimeMs() + 10;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));

    auto start_time_ms = currentTimeMs();

    bool success = worker_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    EXPECT_FALSE(success);

    auto end_time_ms = currentTimeMs();
    EXPECT_GE(end_time_ms - start_time_ms, 10);
    EXPECT_LE(end_time_ms - start_time_ms, 200);

    EXPECT_EQ(task_store_->getTask(unique_key), nullptr);
}

TEST_F(P2PConnectorWorkerTest, Read_ReturnTrue_EmptyBuffers) {
    std::string unique_key  = "test_read_empty";
    int64_t     request_id  = 3004;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;

    bool success = worker_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    EXPECT_TRUE(success);
}

}  // namespace test
}  // namespace rtp_llm
