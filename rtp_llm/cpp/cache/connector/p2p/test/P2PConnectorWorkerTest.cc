#include <atomic>
#include <thread>
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <chrono>
#include <map>

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerPrefill.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerDecode.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/cache/connector/p2p/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
namespace rtp_llm {

namespace test {

// Mock LayerBlockConverter for testing
class MockLayerBlockConverter: public LayerBlockConverter {
public:
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count = 1, int partition_id = 0) const override {
        return {};
    }

    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        return {};
    }
};

// Mock IKVCacheSender for testing (replaces old TransferClient mock)
class MockIKVCacheSender: public transfer::IKVCacheSender {
public:
    struct SendCallInfo {
        std::string ip;
        uint32_t    port;
        std::string layer_key;
        int         layer_id;  // parsed from layer_key
        int64_t     deadline_ms = 0;
    };

    bool regMem(const BlockInfo& /*block_info*/, uint64_t /*aligned_size*/ = 0) override {
        return true;
    }

    void send(const transfer::SendRequest&                                         request,
              std::function<void(transfer::TransferErrorCode, const std::string&)> callback) override {
        SendCallInfo info;
        info.ip          = request.ip;
        info.port        = request.port;
        info.layer_key   = request.unique_key;
        info.layer_id    = parseLayerId(request.unique_key);
        info.deadline_ms = request.deadline_ms;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            send_calls_.push_back(info);
        }

        bool success = should_succeed_;
        if (layer_success_map_.count(info.layer_id)) {
            success = layer_success_map_.at(info.layer_id);
        }

        int delay_ms = callback_delay_ms_;
        if (use_staggered_callback_delay_) {
            const int n = stagger_callback_counter_.fetch_add(1);
            delay_ms    = stagger_callback_base_ms_ * (n + 1);
        }
        auto run_cb = [callback, success, delay_ms]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            auto code = success ? transfer::TransferErrorCode::OK : transfer::TransferErrorCode::UNKNOWN;
            callback(code, success ? "" : "mock send failed");
        };

        if (async_callback_) {
            std::thread(run_cb).detach();
        } else {
            run_cb();
        }
    }

    void setShouldSucceed(bool v) {
        should_succeed_ = v;
    }
    void setLayerSuccess(int layer_id, bool success) {
        layer_success_map_[layer_id] = success;
    }
    void setAsyncCallback(bool v) {
        async_callback_ = v;
    }
    void setCallbackDelayMs(int ms) {
        callback_delay_ms_ = ms;
    }

    /// 每次 send 的回调延迟递增（1*base, 2*base, ...），用于覆盖 wait_for + 多次 notify
    void setStaggeredCallbackDelays(bool enable, int base_ms = 15) {
        use_staggered_callback_delay_ = enable;
        stagger_callback_base_ms_     = base_ms;
        stagger_callback_counter_.store(0);
    }

    int getTransferCallCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(send_calls_.size());
    }
    std::vector<SendCallInfo> getTransferCalls() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return send_calls_;
    }
    void clearTransferCalls() {
        std::lock_guard<std::mutex> lock(mutex_);
        send_calls_.clear();
    }

private:
    /// @brief Parse layer_id from layer_key = base_key + "_" + layer_id + "_" + partition_id
    static int parseLayerId(const std::string& layer_key) {
        auto last = layer_key.rfind('_');
        if (last == std::string::npos || last == 0) {
            return -1;
        }
        auto second_last = layer_key.rfind('_', last - 1);
        if (second_last == std::string::npos) {
            return -1;
        }
        try {
            return std::stoi(layer_key.substr(second_last + 1, last - second_last - 1));
        } catch (...) {
            return -1;
        }
    }

    bool                      should_succeed_ = true;
    std::map<int, bool>       layer_success_map_;
    bool                      async_callback_    = true;
    int                       callback_delay_ms_ = 1;
    bool                      use_staggered_callback_delay_{false};
    int                       stagger_callback_base_ms_{15};
    std::atomic<int>          stagger_callback_counter_{0};
    mutable std::mutex        mutex_;
    std::vector<SendCallInfo> send_calls_;
};

// Mock IKVCacheRecvTask for testing
class MockIKVCacheRecvTask: public transfer::IKVCacheRecvTask {
public:
    bool done() const override {
        return done_.load();
    }
    bool success() const override {
        return success_.load();
    }
    void cancel() override {
        success_.store(false);
        error_code_ = transfer::TransferErrorCode::CANCELLED;
        done_.store(true);
    }
    void forceCancel() override {
        cancel();
    }
    transfer::TransferErrorCode errorCode() const override {
        return error_code_;
    }
    std::string errorMessage() const override {
        return success_.load() ? "" : "mock recv task failed";
    }

    void setDone(bool success) {
        success_.store(success);
        if (!success) {
            error_code_ = transfer::TransferErrorCode::UNKNOWN;
        }
        done_.store(true);
    }

private:
    std::atomic<bool>           done_{false};
    std::atomic<bool>           success_{true};
    transfer::TransferErrorCode error_code_{transfer::TransferErrorCode::OK};
};

// Mock IKVCacheReceiver for testing
class MockIKVCacheReceiver: public transfer::IKVCacheReceiver {
public:
    bool regMem(const BlockInfo& /*block_info*/, uint64_t /*aligned_size*/ = 0) override {
        return true;
    }

    transfer::IKVCacheRecvTaskPtr recv(const transfer::RecvRequest& request) override {
        auto                        task = std::make_shared<MockIKVCacheRecvTask>();
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_[request.unique_key] = task;
        return task;
    }

    /// @brief Signal a specific layer_key task as done
    void setTaskDone(const std::string& layer_key, bool success) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (auto it = tasks_.find(layer_key); it != tasks_.end()) {
            it->second->setDone(success);
        }
    }

    void stealTask(const std::string& /*unique_key*/) override {
        steal_task_count_.fetch_add(1, std::memory_order_relaxed);
    }
    transfer::IKVCacheRecvTaskPtr getTask(const std::string& unique_key) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (auto it = tasks_.find(unique_key); it != tasks_.end()) {
            return it->second;
        }
        return nullptr;
    }

    int stealTaskCount() const {
        return steal_task_count_.load(std::memory_order_relaxed);
    }

    /// @brief Check if tasks for a base_key have been created (>= expected_count)
    bool hasEnoughTasks(const std::string& base_key, int expected_count) const {
        std::lock_guard<std::mutex> lock(mutex_);
        int                         count = 0;
        for (const auto& [key, task] : tasks_) {
            if (key.size() > base_key.size() && key.substr(0, base_key.size()) == base_key
                && key[base_key.size()] == '_') {
                ++count;
            }
        }
        return count >= expected_count;
    }

private:
    mutable std::mutex                                                     mutex_;
    std::unordered_map<std::string, std::shared_ptr<MockIKVCacheRecvTask>> tasks_;
    std::atomic<int>                                                       steal_task_count_{0};
};

// Test fixture for P2PConnectorWorker (tests Prefill and Decode sub-objects directly)
class P2PConnectorWorkerTest: public ::testing::Test {
protected:
    void SetUp() override {
        worker_config_.transfer_backend_config.cache_store_rdma_mode        = false;
        worker_config_.transfer_backend_config.messager_io_thread_count     = 1;
        worker_config_.transfer_backend_config.messager_worker_thread_count = 1;
        worker_config_.tp_size                                              = 2;
        worker_config_.tp_rank                                              = 0;
        worker_config_.transfer_backend_config.cache_store_listen_port      = 0;
        worker_config_.layer_all_num                                        = 2;

        mock_layer_block_converter_ = std::make_shared<MockLayerBlockConverter>();

        mock_sender_   = std::make_shared<MockIKVCacheSender>();
        mock_receiver_ = std::make_shared<MockIKVCacheReceiver>();

        prefill_ = std::make_unique<P2PConnectorWorkerPrefill>(
            worker_config_, mock_layer_block_converter_, nullptr, mock_sender_);
        prefill_->init(10 * 1000);

        decode_ = std::make_unique<P2PConnectorWorkerDecode>(
            worker_config_, mock_layer_block_converter_, nullptr, mock_receiver_);

        computed_buffers_ = prefill_->getComputedBuffersStore();
    }

    void TearDown() override {
        prefill_.reset();
        decode_.reset();
    }

    KVCacheResourcePtr createKVCacheResource(int layer_id, int num_blocks = 2) {
        auto             resource  = std::make_shared<KVCacheResource>();
        int              layer_num = static_cast<int>(worker_config_.layer_all_num);
        std::vector<int> layer_to_group(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            layer_to_group[i] = i;
        }
        resource->initGroups(layer_num, layer_num, layer_to_group);

        for (int i = 0; i < layer_num; ++i) {
            if (i == layer_id) {
                for (int j = 0; j < num_blocks; ++j) {
                    resource->mutableBlockIds(i).add({j});
                }
            }
        }

        for (int i = 0; i < num_blocks; ++i) {
            resource->cacheKeys().push_back(layer_id * 1000 + i);
        }

        return resource;
    }

    // Create a c10::Event that is immediately queryable (already recorded on current stream).
    std::optional<c10::Event> createReadyEvent() {
        return std::nullopt;  // nullopt means "immediately ready" in StoreWaitContext logic
    }

    void addComputedBuffer(int64_t request_id, int layer_id, int64_t deadline_ms) {
        auto layer_cache_buffer = createLayerCacheBuffer(layer_id);
        computed_buffers_->addBuffer(request_id, layer_cache_buffer, deadline_ms);
    }

    std::shared_ptr<LayerCacheBuffer> createLayerCacheBuffer(int layer_id, int num_blocks = 2) {
        auto buffer = std::make_shared<LayerCacheBuffer>(layer_id);
        for (int i = 0; i < num_blocks; ++i) {
            int64_t cache_key = layer_id * 1000 + i;
            int     block_id  = i;
            buffer->addBlockId(cache_key, block_id);
        }
        return buffer;
    }

    void simulateTaskDone(const std::string& base_key, const std::vector<int>& layer_ids, bool all_success = true) {
        for (int layer_id : layer_ids) {
            std::string layer_key = base_key + "_" + std::to_string(layer_id) + "_0";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            mock_receiver_->setTaskDone(layer_key, all_success);
        }
    }

    void setTransferWaitTimeout(int64_t timeout_ms) {
        worker_config_.transfer_backend_config.rdma_transfer_wait_timeout_ms = timeout_ms;
        prefill_.reset();
        decode_.reset();
        prefill_ = std::make_unique<P2PConnectorWorkerPrefill>(
            worker_config_, mock_layer_block_converter_, nullptr, mock_sender_);
        prefill_->init(10 * 1000);
        decode_ = std::make_unique<P2PConnectorWorkerDecode>(
            worker_config_, mock_layer_block_converter_, nullptr, mock_receiver_);
        computed_buffers_ = prefill_->getComputedBuffersStore();
    }

protected:
    P2PConnectorWorkerConfig                       worker_config_;
    std::shared_ptr<LayerBlockConverter>           mock_layer_block_converter_;
    std::unique_ptr<P2PConnectorWorkerPrefill>     prefill_;
    std::unique_ptr<P2PConnectorWorkerDecode>      decode_;
    std::shared_ptr<ComputedLayerCacheBufferStore> computed_buffers_;
    std::shared_ptr<MockIKVCacheSender>            mock_sender_;
    std::shared_ptr<MockIKVCacheReceiver>          mock_receiver_;
};

// ==================== writeByLayer 测试 (Prefill 端) ====================

TEST_F(P2PConnectorWorkerTest, WriteByLayer_ReturnTrue_WithReadyEvent) {
    int     layer_id   = 0;
    int64_t request_id = 1002;
    auto    resource   = createKVCacheResource(layer_id, 2);

    // Pass nullopt — means "immediately ready" in StoreWaitContext logic
    bool success = prefill_->writeByLayer(layer_id, resource, request_id, std::nullopt);
    EXPECT_TRUE(success);

    // Wait for cleanup thread to check once — event is immediately ready so buffer should appear
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    ASSERT_NE(computed_buffers_->getBuffer(request_id), nullptr);
}

// ==================== sendKVCache 测试 (Prefill 端) ====================

TEST_F(P2PConnectorWorkerTest, SendKVCache_SendRequestDeadline_AlignedWithReturnBefore) {
    int64_t     request_id  = 2000;
    std::string unique_key  = "test_send_deadline_align";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(false);

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    const int64_t expected_transfer_deadline = deadline_ms - worker_config_.p2p_read_return_before_deadline_ms;
    ErrorInfo     result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
    EXPECT_TRUE(result.ok());

    for (const auto& c : mock_sender_->getTransferCalls()) {
        EXPECT_EQ(c.deadline_ms, expected_transfer_deadline)
            << "SendRequest.deadline_ms should match decode recv_task_deadline (D - return_before)";
    }
}

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnTrue_AllLayersTransferSuccess) {
    int64_t     request_id  = 2001;
    std::string unique_key  = "test_all_success";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(true);

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    ErrorInfo         result;
    std::thread       write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
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

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(done);
    EXPECT_EQ(mock_sender_->getTransferCallCount(), 2);

    auto          calls = mock_sender_->getTransferCalls();
    std::set<int> transferred_layers;
    for (const auto& call : calls) {
        transferred_layers.insert(call.layer_id);
        EXPECT_EQ("127.0.0.1", call.ip);
        EXPECT_EQ(12345u, call.port);
        EXPECT_TRUE(call.layer_key.substr(0, unique_key.size()) == unique_key);
    }
    EXPECT_EQ(transferred_layers.size(), 2u);
}

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnFalse_PartialLayersTransferFailed) {
    int64_t     request_id  = 2002;
    std::string unique_key  = "test_partial_fail";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setLayerSuccess(1, false);
    mock_sender_->setAsyncCallback(true);

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    ErrorInfo         result;
    std::thread       write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
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

    EXPECT_TRUE(result.hasError());
    EXPECT_TRUE(done);
    EXPECT_EQ(mock_sender_->getTransferCallCount(), 2);

    auto          calls = mock_sender_->getTransferCalls();
    std::set<int> transferred_layers;
    for (const auto& call : calls) {
        transferred_layers.insert(call.layer_id);
        EXPECT_TRUE(call.layer_key.substr(0, unique_key.size()) == unique_key);
        EXPECT_EQ(call.ip, "127.0.0.1");
    }
    EXPECT_EQ(transferred_layers.size(), 2u);
    EXPECT_TRUE(transferred_layers.find(0) != transferred_layers.end());
    EXPECT_TRUE(transferred_layers.find(1) != transferred_layers.end());
}

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnFalse_SomeLayersNotTransferred) {
    int64_t     request_id = 2003;
    std::string unique_key = "test_some_layers_missing";
    // D 须足够大，使 return_deadline_ms=D-100 仍晚于 now，才能先发出已有 layer 再因缺层失败
    int64_t deadline_ms = currentTimeMs() + 150;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(true);

    // 只添加 layer 0
    addComputedBuffer(request_id, 0, deadline_ms);

    // wait till deadline
    std::atomic<bool> done{false};
    ErrorInfo         result;
    std::thread       write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
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
    EXPECT_TRUE(result.hasError());

    auto          calls = mock_sender_->getTransferCalls();
    std::set<int> transferred_layers;
    for (const auto& call : calls) {
        transferred_layers.insert(call.layer_id);
        EXPECT_TRUE(call.layer_key.substr(0, unique_key.size()) == unique_key);
        EXPECT_EQ(call.ip, "127.0.0.1");
    }
    EXPECT_EQ(transferred_layers.size(), 1u);
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

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(true);

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    ErrorInfo         result;
    std::thread       write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
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

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(done);
    // 2P4D: tp_size=2, tp_rank=0 负责 decode 节点 0,1（端口 12345, 12346），每层各发 1 次 → 共 4 次
    EXPECT_EQ(mock_sender_->getTransferCallCount(), 4);

    auto          calls = mock_sender_->getTransferCalls();
    std::set<int> transferred_layers;
    std::set<int> transferred_ports;
    for (const auto& call : calls) {
        transferred_layers.insert(call.layer_id);
        transferred_ports.insert(call.port);
        EXPECT_TRUE(call.layer_key.substr(0, unique_key.size()) == unique_key);
    }
    EXPECT_EQ(transferred_layers.size(), 2u);
    EXPECT_EQ(transferred_ports.size(), 2u);
    EXPECT_TRUE(transferred_ports.find(12345) != transferred_ports.end());
    EXPECT_TRUE(transferred_ports.find(12346) != transferred_ports.end());
}

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnFalse_TransferTimeout) {
    int64_t     request_id = 2005;
    std::string unique_key = "test_transfer_timeout";
    // 足够长的 D，使 return_deadline 晚于 200ms 回调延迟，仍能等到 mock 回调
    int64_t deadline_ms = currentTimeMs() + 500;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    // 设置 transfer 回调延迟并返回失败，模拟超时场景
    mock_sender_->setShouldSucceed(false);
    mock_sender_->setAsyncCallback(true);
    mock_sender_->setCallbackDelayMs(200);  // 延迟 200ms，超过 deadline

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    ErrorInfo         result;
    auto              start_time_ms = currentTimeMs();

    std::thread write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

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

    // 验证返回错误（因为 transfer 失败）
    EXPECT_TRUE(result.hasError());
    EXPECT_TRUE(done);

    // 验证 worker 等待了 transfer 回调返回后才结束（elapsed >= 回调延迟时间）
    EXPECT_GE(elapsed_ms, 150);  // 允许一些误差

    // 验证 transfer 被调用了
    EXPECT_GT(mock_sender_->getTransferCallCount(), 1);
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

    ErrorInfo error_info = decode_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    completion_thread.join();

    EXPECT_TRUE(error_info.ok());
}

TEST_F(P2PConnectorWorkerTest, Read_ReturnFalse_PartialLayersFailed) {
    std::string unique_key  = "test_read_partial_fail";
    int64_t     request_id  = 3002;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));
    layer_cache_buffers.push_back(createLayerCacheBuffer(1, 2));

    std::thread completion_thread([this, unique_key]() {
        // Layer 0 成功，layer 1 失败
        mock_receiver_->setTaskDone(unique_key + "_0_0", true);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        mock_receiver_->setTaskDone(unique_key + "_1_0", false);
    });

    ErrorInfo error_info = decode_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    completion_thread.join();

    EXPECT_TRUE(error_info.hasError());
}

TEST_F(P2PConnectorWorkerTest, Read_ReturnFalse_Timeout) {
    std::string unique_key  = "test_read_timeout";
    int64_t     request_id  = 3003;
    int64_t     deadline_ms = currentTimeMs() + 10;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));

    auto start_time_ms = currentTimeMs();

    // 不调用 simulateTaskDone；return_deadline = D - return_before_ms 已过，尽快以 TRANSFER_NOT_DONE 返回
    ErrorInfo error_info = decode_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    EXPECT_TRUE(error_info.hasError());
    EXPECT_EQ(error_info.code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE);

    auto end_time_ms = currentTimeMs();
    EXPECT_LE(end_time_ms - start_time_ms, 300);
}

TEST_F(P2PConnectorWorkerTest, Read_ReturnTrue_EmptyBuffers) {
    std::string unique_key  = "test_read_empty";
    int64_t     request_id  = 3004;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;

    ErrorInfo error_info = decode_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    EXPECT_TRUE(error_info.ok());
}

// ==================== rdma_transfer_wait_timeout_ms 超时测试 ====================

TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnFalse_RdmaTransferWaitTimeout) {
    // 设置很短的 rdma_transfer_wait_timeout_ms；return_deadline 须较近，否则 callback 等待会持续到 D-return_before
    setTransferWaitTimeout(50);  // 50ms

    int64_t     request_id = 4001;
    std::string unique_key = "test_rdma_transfer_wait_timeout_handleread";
    // return_deadline = deadline_ms - return_before_ms(100) ≈ now + 50ms，与 rdma cap 同量级
    int64_t deadline_ms = currentTimeMs() + 150;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    // 设置 transfer 回调延迟，超过 rdma_transfer_wait_timeout_ms
    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(true);
    mock_sender_->setCallbackDelayMs(500);  // 500ms，超过 50ms 的超时

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    ErrorInfo         result;
    auto              start_time_ms = currentTimeMs();

    std::thread write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    int wait_count = 0;
    while (!done && wait_count < 500) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    auto elapsed_ms = currentTimeMs() - start_time_ms;

    if (write_thread.joinable()) {
        write_thread.join();
    }

    // 验证返回错误（因为 rdma_transfer_wait_timeout_ms 超时）
    EXPECT_TRUE(result.hasError());
    EXPECT_TRUE(done);

    // 验证等待时间约为 rdma_transfer_wait_timeout_ms（50ms），而不是回调延迟（500ms）
    EXPECT_GE(elapsed_ms, 40);   // 允许一些误差
    EXPECT_LE(elapsed_ms, 300);  // 应该远小于 500ms

    // 验证 transfer 被调用了
    EXPECT_GT(mock_sender_->getTransferCallCount(), 0);
}

TEST_F(P2PConnectorWorkerTest, Read_ReturnFalse_RdmaTransferWaitTimeout) {
    std::string unique_key = "test_rdma_transfer_wait_timeout_read";
    int64_t     request_id = 4002;
    // return_deadline = D - return_before_ms ≈ now + 100ms
    int64_t deadline_ms = currentTimeMs() + 200;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));
    layer_cache_buffers.push_back(createLayerCacheBuffer(1, 2));

    auto start_time_ms = currentTimeMs();

    // 不调用 simulateTaskDone：在 return 截止前退出，返回 TRANSFER_NOT_DONE，不 forceCancel
    ErrorInfo error_info = decode_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);

    auto elapsed_ms = currentTimeMs() - start_time_ms;

    EXPECT_TRUE(error_info.hasError());
    EXPECT_EQ(error_info.code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE);

    // 至少等到接近 return 截止（约 100ms），允许退避误差
    EXPECT_GE(elapsed_ms, 85);
    EXPECT_LE(elapsed_ms, 400);
    EXPECT_GE(mock_receiver_->stealTaskCount(), 1);
}

TEST_F(P2PConnectorWorkerTest, Read_ReturnFalse_CancelRead) {
    std::string unique_key  = "test_read_cancel";
    int64_t     request_id  = 3005;
    int64_t     deadline_ms = currentTimeMs() + 5000;  // 足够长的 deadline

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));
    layer_cache_buffers.push_back(createLayerCacheBuffer(1, 2));

    std::atomic<bool> done{false};
    ErrorInfo         result;

    // 启动 read 线程
    std::thread read_thread([&]() {
        result = decode_->read(request_id, unique_key, deadline_ms, layer_cache_buffers);
        done   = true;
    });

    // 等待 mock receiver 收到 recv() 请求（表示 read() 已在等待中）
    int wait_count = 0;
    while (!mock_receiver_->hasEnoughTasks(unique_key, static_cast<int>(layer_cache_buffers.size()))
           && wait_count < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        wait_count++;
    }

    // 调用 cancelRead 取消任务（重试直到找到任务，因为 read_tasks_ 在 recv 后才插入）
    bool cancel_result = false;
    for (int i = 0; i < 50 && !cancel_result; ++i) {
        cancel_result = decode_->cancelRead(unique_key);
        if (!cancel_result) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
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

    // 验证返回错误（因为被取消）
    EXPECT_TRUE(result.hasError());
    EXPECT_EQ(result.code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED);
    EXPECT_TRUE(done);
}

TEST_F(P2PConnectorWorkerTest, CancelRead_ReturnFalse_TaskNotFound) {
    std::string unique_key = "test_cancel_not_found";

    // 不创建任务，直接调用 cancelRead
    bool cancel_result = decode_->cancelRead(unique_key);

    // 验证返回 false（任务不存在）
    EXPECT_FALSE(cancel_result);
}

TEST_F(P2PConnectorWorkerTest, CancelHandleRead_ReturnTrue_ContextNotFound) {
    std::string unique_key = "test_cancel_handle_read_not_found";

    // 不创建 context，直接调用 cancelSend
    // 由于 cancel 是尽力而为，即使 context 不存在也返回 true
    bool cancel_result = prefill_->cancelSend(unique_key);

    // 验证返回 true（因为 cancel 是 best-effort）
    EXPECT_TRUE(cancel_result);
}

TEST_F(P2PConnectorWorkerTest, CancelHandleRead_ReturnTrue_ContextFound) {
    int64_t     request_id  = 3006;
    std::string unique_key  = "test_cancel_handle_read_found";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(true);

    std::atomic<bool> done{false};
    ErrorInfo         result;

    // 启动 sendKVCache 线程（未添加 computed buffer，sendKVCache 会阻塞等待）
    std::thread handle_read_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    // 稍等，让 sendKVCache 线程启动并注册 cancel flag
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // 调用 cancelSend 取消 context
    bool cancel_result = prefill_->cancelSend(unique_key);
    EXPECT_TRUE(cancel_result);

    // 等待 sendKVCache 完成
    int wait_count = 0;
    while (!done && wait_count < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (handle_read_thread.joinable()) {
        handle_read_thread.join();
    }

    // 验证返回错误（因为被取消）
    EXPECT_TRUE(result.hasError());
    EXPECT_TRUE(done);
}

// 异步回调错峰完成，覆盖 waitSendCallbacksWithTimeout 中 result_cv 多次 wait_for 唤醒
TEST_F(P2PConnectorWorkerTest, SendKVCache_Succeeds_StaggeredAsyncCallbacks) {
    int64_t     request_id  = 2009;
    std::string unique_key  = "test_staggered_cv";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(true);
    mock_sender_->setStaggeredCallbackDelays(true, 25);
    mock_sender_->setCallbackDelayMs(0);

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    ErrorInfo         result;
    std::thread       write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
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

    EXPECT_TRUE(done);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(mock_sender_->getTransferCallCount(), 2);
    mock_sender_->setStaggeredCallbackDelays(false);
}

// ==================== sendKVCache callback wait timeout 测试 ====================

// 测试：send callback 超时未回调 -> sendKVCache 应报告失败而非成功
TEST_F(P2PConnectorWorkerTest, HandleRead_ReturnFalse_CallbackWaitTimeout) {
    int64_t     request_id  = 4001;
    std::string unique_key  = "test_callback_wait_timeout";
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    // callback 永不触发（大 delay），验证超时保护下不会误判成功
    mock_sender_->setShouldSucceed(true);
    mock_sender_->setAsyncCallback(true);
    mock_sender_->setCallbackDelayMs(10000);  // 10s delay，远超 rdma_transfer_wait_timeout_ms

    // 设置很短的 rdma_transfer_wait_timeout_ms
    setTransferWaitTimeout(50);

    addComputedBuffer(request_id, 0, deadline_ms);
    addComputedBuffer(request_id, 1, deadline_ms);

    std::atomic<bool> done{false};
    ErrorInfo         result;
    std::thread       write_thread([&]() {
        result = prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
        done   = true;
    });

    int wait_count = 0;
    while (!done && wait_count < 500) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (write_thread.joinable()) {
        write_thread.join();
    }

    EXPECT_TRUE(done);
    // callback 未收齐，必须报告失败（fix: 修复前此处可能误判成功）
    EXPECT_TRUE(result.hasError());
    EXPECT_EQ(result.code(), ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT);
}

// ==================== LayerCacheBufferUtil 边界测试 ====================

class LayerCacheBufferUtilTest: public ::testing::Test {
protected:
    KVCacheResourcePtr createResource(int num_layers, int blocks_per_layer) {
        auto             resource = std::make_shared<KVCacheResource>();
        std::vector<int> layer_to_group(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            layer_to_group[i] = i;
        }
        resource->initGroups(num_layers, num_layers, layer_to_group);
        for (int layer = 0; layer < num_layers; ++layer) {
            for (int i = 0; i < blocks_per_layer; ++i) {
                resource->mutableBlockIds(layer).add({i});
            }
        }
        for (int i = 0; i < blocks_per_layer; ++i) {
            resource->cacheKeys().push_back(1000 + i);
        }
        return resource;
    }
};

TEST_F(LayerCacheBufferUtilTest, ConvertLayer_ReturnNull_StartIdxEqualActualCount) {
    auto resource = createResource(2, 3);
    // start_block_idx == actual_block_count (3) -> out of range
    auto buf = LayerCacheBufferUtil::convertLayer(*resource, 0, 0, 3, -1);
    EXPECT_EQ(buf, nullptr);
}

TEST_F(LayerCacheBufferUtilTest, ConvertLayer_ReturnNull_StartIdxGreaterThanActualCount) {
    auto resource = createResource(2, 3);
    // start_block_idx > actual_block_count
    auto buf = LayerCacheBufferUtil::convertLayer(*resource, 0, 0, 10, -1);
    EXPECT_EQ(buf, nullptr);
}

TEST_F(LayerCacheBufferUtilTest, ConvertLayer_ReturnNull_BlockCountLessThanNegativeOne) {
    auto resource = createResource(2, 3);
    // block_count < -1 is undefined/illegal
    auto buf = LayerCacheBufferUtil::convertLayer(*resource, 0, 0, 0, -2);
    EXPECT_EQ(buf, nullptr);
}

TEST_F(LayerCacheBufferUtilTest, ConvertLayer_ReturnNull_BlockCountZero) {
    auto resource = createResource(2, 3);
    auto buf      = LayerCacheBufferUtil::convertLayer(*resource, 0, 0, 0, 0);
    EXPECT_EQ(buf, nullptr);
}

TEST_F(LayerCacheBufferUtilTest, ConvertLayer_ReturnPartial_BlockCountLimitsResult) {
    auto resource = createResource(2, 4);
    // start=1, count=2 -> should return 2 blocks
    auto buf = LayerCacheBufferUtil::convertLayer(*resource, 0, 0, 1, 2);
    ASSERT_NE(buf, nullptr);
    EXPECT_EQ(static_cast<int>(buf->blockIdMap().size()), 2);
}

TEST_F(LayerCacheBufferUtilTest, ConvertLayer_ReturnAll_BlockCountNegativeOne) {
    auto resource = createResource(2, 3);
    // block_count=-1 means "all remaining"
    auto buf = LayerCacheBufferUtil::convertLayer(*resource, 0, 0, 0, -1);
    ASSERT_NE(buf, nullptr);
    EXPECT_EQ(static_cast<int>(buf->blockIdMap().size()), 3);
}

TEST_F(LayerCacheBufferUtilTest, ConvertLayer_ReturnNull_StartIdxNegative) {
    auto resource = createResource(2, 3);
    auto buf      = LayerCacheBufferUtil::convertLayer(*resource, 0, 0, -1, -1);
    EXPECT_EQ(buf, nullptr);
}

}  // namespace test
}  // namespace rtp_llm
