#include <thread>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <map>

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerBlockConvertor.h"
#include "rtp_llm/cpp/cache/connector/p2p/ComputedLayerCacheBuffer.h"
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
                  int64_t                                  deadline_ms) override {
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

        int delay_ms = callback_delay_ms_;
        if (async_callback_) {
            std::thread([callback, success, delay_ms]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
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

    void setCallbackDelayMs(int delay_ms) {
        callback_delay_ms_ = delay_ms;
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
    bool                          async_callback_    = true;
    int                           callback_delay_ms_ = 1;
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
        layer_all_num_           = 2;

        // 创建 Mock LayerBlockConvertor
        mock_layer_block_convertor_ = std::make_shared<MockLayerBlockConvertor>();

        // 创建 P2PConnectorWorker（不调用 init，避免创建实际的 TransferClient/Server）
        worker_ = std::make_unique<P2PConnectorWorker>(cache_config_,
                                                       cache_store_config_,
                                                       parallelism_config_,
                                                       pd_sep_config_,
                                                       model_config_,
                                                       layer_all_num_,
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

    // 设置 RDMA 传输等待超时时间
    void setTransferWaitTimeout(int64_t timeout_ms) {
        cache_store_config_.rdma_transfer_wait_timeout_ms = timeout_ms;
    }

protected:
    KVCacheConfig                                  cache_config_;
    CacheStoreConfig                               cache_store_config_;
    ParallelismConfig                              parallelism_config_;
    PDSepConfig                                    pd_sep_config_;
    ModelConfig                                    model_config_;
    uint32_t                                       layer_all_num_;
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

    // write by layer end
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
        EXPECT_EQ(unique_key, call.unique_key);
        EXPECT_EQ(1, call.remote_partition_count);
        EXPECT_EQ(0, call.remote_partition_id);
        EXPECT_EQ(1, call.local_partition_count);
        EXPECT_EQ(0, call.local_partition_id);
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

    auto          calls = mock_transfer_client_->getTransferCalls();
    std::set<int> transferred_layers;
    for (const auto& call : calls) {
        transferred_layers.insert(call.layer_id);
        EXPECT_EQ(call.unique_key, unique_key);
        EXPECT_EQ(call.remote_partition_count, 1);
        EXPECT_EQ(call.remote_partition_id, 0);
        EXPECT_EQ(call.local_partition_count, 1);
        EXPECT_EQ(call.local_partition_id, 0);
        EXPECT_EQ(call.ip, "127.0.0.1");
    }
    EXPECT_EQ(transferred_layers.size(), 2);
    EXPECT_TRUE(transferred_layers.find(0) != transferred_layers.end());
    EXPECT_TRUE(transferred_layers.find(1) != transferred_layers.end());
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

    // wait till deadline
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
        EXPECT_EQ(call.unique_key, unique_key);
        EXPECT_EQ(call.remote_partition_count, 1);
        EXPECT_EQ(call.remote_partition_id, 0);
        EXPECT_EQ(call.local_partition_count, 1);
        EXPECT_EQ(call.local_partition_id, 0);
        EXPECT_EQ(call.ip, "127.0.0.1");
    }
    EXPECT_EQ(transferred_layers.size(), 1);
    EXPECT_TRUE(transferred_layers.find(0) != transferred_layers.end());
}

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnTrue_AsymmetricTP_2P4D_Success) {
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
    std::set<int> transferred_ports;
    for (const auto& call : calls) {
        transferred_layers.insert(call.layer_id);
        transferred_ports.insert(call.port);
        EXPECT_EQ(call.unique_key, unique_key);
        EXPECT_EQ(call.remote_partition_count, 1);
        EXPECT_EQ(call.remote_partition_id, 0);
        if (call.port == 12345) {
            EXPECT_EQ(call.local_partition_count, 2);
            EXPECT_EQ(call.local_partition_id, 0);
        } else if (call.port == 12346) {
            EXPECT_EQ(call.local_partition_count, 2);
            EXPECT_EQ(call.local_partition_id, 1);
        }
    }
    EXPECT_EQ(transferred_layers.size(), 2);
    EXPECT_EQ(transferred_ports.size(), 2);
    EXPECT_TRUE(transferred_ports.find(12345) != transferred_ports.end());
    EXPECT_TRUE(transferred_ports.find(12346) != transferred_ports.end());
}

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnFalse_TransferTimeout) {
    int64_t     request_id  = 2005;
    std::string unique_key  = "test_transfer_timeout";
    int64_t     deadline_ms = currentTimeMs() + 50;  // 很短的超时时间

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    // 设置 transfer 回调延迟并返回失败，模拟超时场景
    // 注意：即使超时，handleRead 也会等待所有 transfer 回调返回
    // 所以需要设置 transfer 返回失败来验证超时后的行为
    mock_transfer_client_->setShouldSucceed(false);
    mock_transfer_client_->setAsyncCallback(true);
    mock_transfer_client_->setCallbackDelayMs(200);  // 延迟 200ms，超过 deadline

    // 添加所有层的 computed buffer
    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    std::atomic<bool> result{false};
    auto              start_time_ms = currentTimeMs();

    std::thread write_thread([&]() {
        result = worker_->handleRead(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    // 等待 handleRead 完成
    int wait_count = 0;
    while (!done && wait_count < 500) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }
    EXPECT_TRUE(wait_count < 500);

    auto elapsed_ms = currentTimeMs() - start_time_ms;

    if (write_thread.joinable()) {
        write_thread.join();
    }

    // 验证返回 false（因为 transfer 失败）
    EXPECT_FALSE(result);
    EXPECT_TRUE(done);

    // 验证 worker 等待了 transfer 回调返回后才结束（elapsed >= 回调延迟时间）
    // 这证明即使超时，worker 也会等待所有 transfer 回调返回
    EXPECT_GE(elapsed_ms, 150);  // 允许一些误差

    // 验证 transfer 被调用了
    EXPECT_GT(mock_transfer_client_->getTransferCallCount(), 1);
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

// ==================== rdma_transfer_wait_timeout_ms 超时测试 ====================

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnFalse_RdmaTransferWaitTimeout) {
    // 设置很短的 rdma_transfer_wait_timeout_ms
    setTransferWaitTimeout(50);  // 50ms

    int64_t     request_id  = 4001;
    std::string unique_key  = "test_rdma_transfer_wait_timeout_handleread";
    int64_t     deadline_ms = currentTimeMs() + 5000;  // deadline 足够长

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    // 设置 transfer 回调延迟，超过 rdma_transfer_wait_timeout_ms
    mock_transfer_client_->setShouldSucceed(true);
    mock_transfer_client_->setAsyncCallback(true);
    mock_transfer_client_->setCallbackDelayMs(500);  // 500ms，超过 50ms 的超时

    // 添加所有层的 computed buffer
    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    std::atomic<bool> result{true};  // 初始化为 true，验证会变成 false
    auto              start_time_ms = currentTimeMs();

    std::thread write_thread([&]() {
        result = worker_->handleRead(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    // 等待 handleRead 完成
    int wait_count = 0;
    while (!done && wait_count < 500) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    auto elapsed_ms = currentTimeMs() - start_time_ms;

    if (write_thread.joinable()) {
        write_thread.join();
    }

    // 验证返回 false（因为 rdma_transfer_wait_timeout_ms 超时）
    EXPECT_FALSE(result);
    EXPECT_TRUE(done);

    // 验证等待时间约为 rdma_transfer_wait_timeout_ms（50ms），而不是回调延迟（500ms）
    EXPECT_GE(elapsed_ms, 40);   // 允许一些误差
    EXPECT_LE(elapsed_ms, 300);  // 应该远小于 500ms

    // 验证 transfer 被调用了
    EXPECT_GT(mock_transfer_client_->getTransferCallCount(), 0);
}

TEST_F(P2PConnectorWorkerTest, Read_ReturnFalse_RdmaTransferWaitTimeout) {
    // 设置很短的 rdma_transfer_wait_timeout_ms
    setTransferWaitTimeout(50);  // 50ms

    std::string unique_key = "test_rdma_transfer_wait_timeout_read";
    int64_t     request_id = 4002;
    // 设置较短的 deadline_ms，让第一个循环快速退出（isTimeout）
    // 然后第二个循环会等待 rdma_transfer_wait_timeout_ms 超时
    int64_t deadline_ms = currentTimeMs() + 100;  // 100ms

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));
    layer_cache_buffers.push_back(createLayerCacheBuffer(1, 2));

    auto start_time_ms = currentTimeMs();

    // 启动一个线程来模拟任务开始加载但不完成（等待超时）
    std::thread loading_thread([this, unique_key]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto task = task_store_->getTask(unique_key);
        if (task) {
            // 只开始加载，不调用 notifyDone，让它等待超时
            task->loadingLayerCacheBuffer(0, 1, 0);
            task->loadingLayerCacheBuffer(1, 1, 0);
            // 不调用 notifyDone，让 read 等待 rdma_transfer_wait_timeout_ms 超时
        }
    });

    bool success = worker_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    auto elapsed_ms = currentTimeMs() - start_time_ms;

    loading_thread.join();

    // 验证返回 false（因为超时）
    EXPECT_FALSE(success);

    // 验证等待时间约为 deadline_ms (100ms) + rdma_transfer_wait_timeout_ms (50ms) = 150ms
    // 第一个循环等待 deadline_ms 超时，第二个循环等待 rdma_transfer_wait_timeout_ms 超时
    EXPECT_GE(elapsed_ms, 100);  // 至少等待 deadline_ms
    EXPECT_LE(elapsed_ms, 400);  // 不应该等太久

    // 验证任务被取消
    EXPECT_EQ(task_store_->getTask(unique_key), nullptr);
}

TEST_F(P2PConnectorWorkerTest, Read_ReturnFalse_CancelRead) {
    std::string unique_key  = "test_read_cancel";
    int64_t     request_id  = 3005;
    int64_t     deadline_ms = currentTimeMs() + 5000;  // 足够长的 deadline

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));
    layer_cache_buffers.push_back(createLayerCacheBuffer(1, 2));

    std::atomic<bool> done{false};
    std::atomic<bool> result{true};  // 初始化为 true，验证会变成 false

    // 启动 read 线程
    std::thread read_thread([&]() {
        result = worker_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);
        done   = true;
    });

    // 等待任务创建
    int wait_count = 0;
    while (task_store_->getTask(unique_key) == nullptr && wait_count < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        wait_count++;
    }

    // 验证任务已创建
    auto task = task_store_->getTask(unique_key);
    ASSERT_NE(task, nullptr);

    // 调用 cancelRead 取消任务
    bool cancel_result = worker_->cancelRead(unique_key);
    EXPECT_TRUE(cancel_result);

    // 等待 read 完成
    wait_count = 0;
    while (!done && wait_count < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (read_thread.joinable()) {
        read_thread.join();
    }

    // 验证返回 false（因为被取消）
    EXPECT_FALSE(result);
    EXPECT_TRUE(done);

    // 验证任务已从 store 中移除
    EXPECT_EQ(task_store_->getTask(unique_key), nullptr);
}

TEST_F(P2PConnectorWorkerTest, CancelRead_ReturnFalse_TaskNotFound) {
    std::string unique_key = "test_cancel_not_found";

    // 不创建任务，直接调用 cancelRead
    bool cancel_result = worker_->cancelRead(unique_key);

    // 验证返回 false（任务不存在）
    EXPECT_FALSE(cancel_result);
}

}  // namespace test
}  // namespace rtp_llm
