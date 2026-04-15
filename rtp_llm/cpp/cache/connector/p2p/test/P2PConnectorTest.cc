#include <atomic>
#include <future>
#include <string>
#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorResourceStore.h"
#include "rtp_llm/cpp/cache/connector/p2p/test/MockGenerateStream.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/Exception.h"
#include "rtp_llm/cpp/cache/connector/p2p/test/TestRpcServer.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/BlockInfo.h"

namespace rtp_llm {

// Mock LayerBlockConverter for testing
class MockLayerBlockConverter: public LayerBlockConverter {
public:
    std::vector<BlockInfo> convertIndexToBuffer(int /*layer_id*/,
                                                int /*block_id*/,
                                                int /*partition_count*/,
                                                int /*partition_id*/) const override {
        return {};
    }

    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        return {};
    }
};

class P2PConnectorTest: public ::testing::Test {
protected:
    void SetUp() override {
        RuntimeConfig runtime_config;
        for (int i = 0; i < 2; ++i) {
            auto service = std::make_unique<TestRpcService>();
            auto server  = std::make_unique<TestRpcServer>(std::move(service));
            ASSERT_TRUE(server->start());
            tp_broadcast_servers_.push_back(std::move(server));
            tp_broadcast_addrs_.push_back("127.0.0.1:" + std::to_string(tp_broadcast_servers_.back()->listenPort()));
        }
        runtime_config.worker_grpc_addrs = tp_broadcast_addrs_;

        CacheStoreConfig cache_store_config;
        cache_store_config.cache_store_rdma_mode        = false;
        cache_store_config.messager_io_thread_count     = 1;
        cache_store_config.messager_worker_thread_count = 1;

        ParallelismConfig parallelism_config;
        parallelism_config.tp_size = 2;
        parallelism_config.tp_rank = 0;

        PDSepConfig pd_sep_config;
        pd_sep_config.role_type                      = RoleType::PREFILL;
        pd_sep_config.cache_store_listen_port        = 0;
        pd_sep_config.decode_polling_call_prefill_ms = 30;

        config_ = P2PConnectorConfig::create(
            runtime_config, cache_store_config, parallelism_config, pd_sep_config, /*layer_all_num=*/2);

        mock_layer_block_converter_ = std::make_shared<MockLayerBlockConverter>();

        connector_ = std::make_unique<P2PConnector>(config_, mock_layer_block_converter_, nullptr);
        ASSERT_TRUE(connector_->init());
    }

    void TearDown() override {
        connector_.reset();
        tp_broadcast_servers_.clear();
    }

    // 创建有效的 KVCacheResource（使用 initGroups + groupBlocks/blocks/cacheKeys 公开 API）
    KVCacheResourcePtr createValidKVCacheResource(int num_layers = 2, int blocks_per_layer = 2) {
        auto             resource = std::make_shared<KVCacheResource>();
        std::vector<int> layer_to_group(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            layer_to_group[i] = i;
        }
        resource->initGroups(num_layers, num_layers, layer_to_group);

        for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
            for (int i = 0; i < blocks_per_layer; ++i) {
                resource->mutableBlockIds(layer_id).add({i});
            }
        }

        for (int i = 0; i < num_layers * blocks_per_layer; ++i) {
            resource->cacheKeys().push_back(1000 + i);
        }

        return resource;
    }

    // 创建有效的 P2PConnectorStartLoadRequest
    P2PConnectorStartLoadRequestPB
    createValidStartLoadRequest(const std::string& unique_key, int64_t deadline_ms, int num_workers = 1) {
        P2PConnectorStartLoadRequestPB request;
        request.set_unique_key(unique_key);
        request.set_deadline_ms(deadline_ms);

        for (int i = 0; i < num_workers; ++i) {
            auto* worker = request.add_workers();
            worker->set_ip("127.0.0.1");
            worker->set_cache_store_port(12345 + i);
        }

        return request;
    }

    /// 构造一个最小可用的 GenerateStream，通过 GenerateInput/GenerateConfig 直接设置
    /// P2P 路由所需字段（unique_key, request_id, deadline_ms），返回 owning shared_ptr。
    /// timeout_ms 是超时毫秒数（相对值），begin_time_us 是当前微秒时间戳，
    /// deadlineMs() = timeout_ms + begin_time_us/1000。
    std::shared_ptr<GenerateStream> createGenerateStream(const std::string& unique_key,
                                                         int64_t            request_id,
                                                         int64_t            timeout_ms) {
        auto config         = std::make_shared<GenerateConfig>();
        config->unique_key  = unique_key;
        config->timeout_ms  = static_cast<int>(timeout_ms);  // timeout in milliseconds (relative)

        auto input                 = std::make_shared<GenerateInput>();
        input->request_id          = request_id;
        input->generate_config     = config;
        input->input_ids           = torch::zeros({1}, torch::kInt32);
        input->begin_time_us       = currentTimeUs();  // Set begin_time_us to current time

        return std::make_shared<MockGenerateStream>(input);
    }

    /// 构造 MockMeta，路由信息从 GenerateStream 中读取（与生产路径一致）。
    std::shared_ptr<MockMeta> createMockMeta(GenerateStream* stream, int prefill_tp_size = 1) {
        auto meta = std::make_shared<MockMeta>();
        meta->setRequestId(stream->streamId());
        meta->setUniqueKey(stream->uniqueKey());
        meta->setDeadlineMs(stream->deadlineMs());
        meta->setPrefillTpSize(prefill_tp_size);
        meta->setGenerateStream(stream);
        return meta;
    }

protected:
    P2PConnectorConfig                          config_;
    std::shared_ptr<MockLayerBlockConverter>    mock_layer_block_converter_;
    std::unique_ptr<P2PConnector>               connector_;
    std::vector<std::unique_ptr<TestRpcServer>> tp_broadcast_servers_;
    std::vector<std::string>                    tp_broadcast_addrs_;
};

// ==================== handleRead 测试 ====================

// 测试: stream_store_ 为 nullptr，返回 INTERNAL 错误
TEST_F(P2PConnectorTest, HandleRead_ReturnInternal_WhenStreamStoreIsNull) {
    auto connector = std::make_unique<P2PConnector>(config_, mock_layer_block_converter_, nullptr);

    // 2. 创建 request
    std::string unique_key  = "test_stream_store_null";
    int64_t     deadline_ms = currentTimeMs() + 5000;
    auto        request     = createValidStartLoadRequest(unique_key, deadline_ms);

    P2PConnectorStartLoadResponsePB response;
    connector->handleRead(request, response);

    EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
}

// 测试: waitForResourceEntry 超时，返回 INTERNAL 错误
TEST_F(P2PConnectorTest, HandleRead_ReturnInternal_WhenWaitResourceEntryTimeout) {
    // 1. 创建并初始化 P2PConnector（已在 SetUp 中完成）
    // 2. 不添加 resource entry 到 stream_store_
    // 3. 调用 handleRead，使用已过期的 deadline_ms
    std::string unique_key  = "test_wait_timeout";
    int64_t     deadline_ms = currentTimeMs() - 100;  // 使用已过期的时间
    auto        request     = createValidStartLoadRequest(unique_key, deadline_ms);

    P2PConnectorStartLoadResponsePB response;
    connector_->handleRead(request, response);

    EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
}

// 测试: waitForResourceEntry 被取消，返回 CANCELLED 错误
TEST_F(P2PConnectorTest, HandleRead_ReturnCancelled_WhenWaitResourceEntryCancelled) {
    // 1. 创建并初始化 P2PConnector（已在 SetUp 中完成）
    // 2. 不添加 resource entry 到 stream_store_
    // 3. 调用 handleRead，使用 is_cancelled lambda 立即返回 true
    std::string unique_key  = "test_wait_cancelled";
    int64_t     deadline_ms = currentTimeMs() + 5000;
    auto        request     = createValidStartLoadRequest(unique_key, deadline_ms);

    std::atomic<bool> cancelled_flag(true);  // 立即设置为 true
    auto              is_cancelled = [&cancelled_flag]() { return cancelled_flag.load(); };

    P2PConnectorStartLoadResponsePB response;
    connector_->handleRead(request, response, is_cancelled);

    EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_NE(response.error_message().find("cancelled"), std::string::npos);
}

// 测试: scheduler_->sendKVCache 失败，返回 INTERNAL 错误
TEST_F(P2PConnectorTest, HandleRead_ReturnInternal_WhenSchedulerHandleReadFailed) {
    // 1. 创建并初始化 P2PConnector（已在 SetUp 中完成）
    // 2. 添加有效的 resource entry
    std::string unique_key  = "test_scheduler_handle_read_failed";
    int64_t     request_id  = 5003;
    int64_t     timeout_ms  = 5000;  // timeout in milliseconds (relative)
    int64_t     deadline_ms = currentTimeMs() + timeout_ms;  // absolute deadline for request
    auto        resource    = createValidKVCacheResource(2, 2);
    auto        stream      = createGenerateStream(unique_key, request_id, timeout_ms);
    auto        meta        = createMockMeta(stream.get());
    connector_->asyncMatch(resource, meta);

    // 3. 设置 TestRpcServer 返回失败（用于 scheduler_->sendKVCache）
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PResponseSuccess(false);
    }

    // 4. 创建 request
    auto request = createValidStartLoadRequest(unique_key, deadline_ms, 2);

    P2PConnectorStartLoadResponsePB response;
    connector_->handleRead(request, response);

    EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
}

// 测试: waitSideChannelReady 超时（first token not found），返回 INTERNAL 错误
TEST_F(P2PConnectorTest, HandleRead_ReturnInternal_WhenWaitSideChannelTimeout) {
    // 1. 添加有效的 resource entry
    std::string unique_key  = "test_wait_side_channel_timeout";
    int64_t     request_id  = 5004;
    int64_t     timeout_ms  = 5000;  // timeout in milliseconds (relative)
    int64_t     deadline_ms = currentTimeMs() + timeout_ms;  // absolute deadline for request
    auto        resource    = createValidKVCacheResource(2, 2);
    auto        stream      = createGenerateStream(unique_key, request_id, timeout_ms);
    auto        meta        = createMockMeta(stream.get());
    connector_->asyncMatch(resource, meta);

    // 2. 设置 TestRpcServer 返回成功（用于 scheduler_->sendKVCache）
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PResponseSuccess(true);
    }

    // 3. 创建 request
    auto request = createValidStartLoadRequest(unique_key, deadline_ms, 2);

    P2PConnectorStartLoadResponsePB response;
    connector_->handleRead(request, response);

    // 4. 由于没有调用 notifySideChannelReady，waitSideChannelReady 会超时
    EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
}

// 测试: 成功场景，使用 notifySideChannelReady 机制，返回 OK, 验证 response 中包含了所有字段
// 注意：这个测试验证了 side-channel 机制的基本流程
// 1. asyncMatch 添加 entry 到 stream_store
// 2. notifySideChannelReady 设置 side-channel data
// 3. handleRead -> waitAndStealResource -> waitAndFillResponse
// 4. waitAndFillResponse 检查 side_channel_ready，发现已经是 true，立即返回
TEST_F(P2PConnectorTest, HandleRead_ReturnOk_WithNotifySideChannelMechanism) {
    // 1. 创建有效的 resource entry
    std::string unique_key  = "test_notify_side_channel_success";
    int64_t     request_id  = 5001;
    int64_t     timeout_ms  = 5000;  // timeout in milliseconds (relative)
    int64_t     deadline_ms = currentTimeMs() + timeout_ms;  // absolute deadline for request
    auto        resource    = createValidKVCacheResource(2, 2);
    auto        stream      = createGenerateStream(unique_key, request_id, timeout_ms);
    auto        meta        = createMockMeta(stream.get());
    connector_->asyncMatch(resource, meta);

    // 2. 设置 TestRpcServer 返回成功（用于 scheduler_->sendKVCache）
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PResponseSuccess(true);
    }

    // 3. 在调用 handleRead 之前，先调用 notifySideChannelReady
    //    这会在 entry 上设置 side_channel_ready=true
    //    然后 handleRead steal entry 时，entry 已经是 ready 状态
    //    waitAndFillResponse 会立即返回
    P2PConnectorResourceEntry::SideChannelData data;
    data.first_token_id   = 12345;
    data.total_reuse_len  = 10;
    data.local_reuse_len  = 5;
    data.remote_reuse_len = 5;
    data.memory_reuse_len = 0;
    data.propose_tokens   = {1001, 1002, 1003};
    data.position_ids     = {1, 2, 3, 4};
    data.propose_probs.set_data_type(TensorPB::FP32);
    data.propose_hidden.set_data_type(TensorPB::FP32);

    // 注意：notifySideChannelReady 需要在 handleRead steal entry 之前调用
    // 这样 entry 的 side_channel_ready 才会被设置
    connector_->streamStore()->notifySideChannelReady(unique_key, data);

    // 4. 创建 request 并调用 handleRead
    //    使用 num_workers = 1 简化测试
    auto request = createValidStartLoadRequest(unique_key, deadline_ms, 1);

    P2PConnectorStartLoadResponsePB response;
    connector_->handleRead(request, response);

    // 5. 验证响应
    EXPECT_EQ(response.error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_EQ(response.payload().first_generate_token_id(), 12345);
    EXPECT_EQ(response.payload().total_reuse_len(), 10);

    // 6. 验证 scheduler_->sendKVCache 被调用
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_GE(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
    }
}

}  // namespace rtp_llm
