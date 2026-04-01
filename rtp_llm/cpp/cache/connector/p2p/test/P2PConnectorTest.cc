#include <atomic>
#include <string>
#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/IGenerateStream.h"
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

/// Wraps MockGenerateStream for P2PConnector::asyncMatch (requires Meta::generateStream()).
class P2PMetaForConnectorTest final: public Meta {
public:
    explicit P2PMetaForConnectorTest(std::shared_ptr<MockGenerateStream> gs): gs_(std::move(gs)) {}

    bool enableMemoryCache() const override {
        return false;
    }
    bool enableRemoteCache() const override {
        return false;
    }
    const std::string& trace_id() const override {
        static const std::string kEmpty;
        return kEmpty;
    }
    const std::string& unique_id() const override {
        static const std::string kEmpty;
        return kEmpty;
    }
    const std::vector<int64_t>& tokens() const override {
        static const std::vector<int64_t> kEmpty;
        return kEmpty;
    }
    std::shared_ptr<IGenerateStream> generateStream() const override {
        return gs_;
    }

private:
    std::shared_ptr<MockGenerateStream> gs_;
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
    std::string unique_key      = "test_scheduler_handle_read_failed";
    int64_t     request_id      = 5003;
    int64_t     deadline_ms     = currentTimeMs() + 5000;
    auto        resource        = createValidKVCacheResource(2, 2);
    auto        generate_stream = std::make_shared<MockGenerateStream>();
    generate_stream->setUniqueKey(unique_key);
    generate_stream->setRequestId(request_id);
    generate_stream->setDeadlineMs(deadline_ms);
    auto meta = std::make_shared<P2PMetaForConnectorTest>(generate_stream);
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

// 测试: fillResponseWithStreamInfo 失败（first token not found），返回 INTERNAL 错误
TEST_F(P2PConnectorTest, HandleRead_ReturnInternal_WhenFillResponseFailed) {
    // 1. 创建并初始化 P2PConnector（已在 SetUp 中完成）
    // 2. 添加有效的 resource entry
    std::string unique_key      = "test_fill_response_failed";
    int64_t     request_id      = 5004;
    int64_t     deadline_ms     = currentTimeMs() + 5000;
    auto        resource        = createValidKVCacheResource(2, 2);
    auto        generate_stream = std::make_shared<MockGenerateStream>();

    // 3. Mock generate_stream->currentExecuteTokens() 返回空（first token not found）
    // 不设置任何 token，这样 currentExecuteTokens 会返回空
    // generate_stream->setTokenIds(0, {});  // 默认就是空的

    generate_stream->setUniqueKey(unique_key);
    generate_stream->setRequestId(request_id);
    generate_stream->setDeadlineMs(deadline_ms);
    auto meta = std::make_shared<P2PMetaForConnectorTest>(generate_stream);
    connector_->asyncMatch(resource, meta);

    // 4. 设置 TestRpcServer 返回成功（用于 scheduler_->sendKVCache）
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PResponseSuccess(true);
    }

    // 5. 创建 request
    auto request = createValidStartLoadRequest(unique_key, deadline_ms, 2);

    P2PConnectorStartLoadResponsePB response;
    connector_->handleRead(request, response);

    EXPECT_NE(response.error_code(), ErrorCodePB::NONE_ERROR);
}

// 测试: 成功场景，返回 OK, 验证 response 中包含了所有字段
TEST_F(P2PConnectorTest, HandleRead_ReturnOk_WhenAllSuccess) {
    // 1. 创建有效的 resource entry
    std::string unique_key      = "test_handle_read_success";
    int64_t     request_id      = 5001;
    int64_t     deadline_ms     = currentTimeMs() + 5000;
    auto        resource        = createValidKVCacheResource(2, 2);
    auto        generate_stream = std::make_shared<MockGenerateStream>();

    // 2. 设置 generate_stream 的返回值
    // 设置 first token (最后一个 token)
    std::vector<int> tokens = {100, 200, 300, 12345};  // 最后一个 token 是 12345
    generate_stream->setTokenIds(0, tokens);

    // 设置 position_ids
    std::vector<int32_t> position_ids = {1, 2, 3, 4};
    generate_stream->setPositionIds(position_ids);

    // 设置 reuse_length
    generate_stream->setReuseLength(10, 5, 5);

    // 设置 propose_info
    std::vector<int> propose_tokens = {1001, 1002, 1003};
    TensorPB         propose_probs;
    TensorPB         propose_hidden;
    generate_stream->setProposeInfo(propose_tokens, propose_probs, propose_hidden);

    generate_stream->setUniqueKey(unique_key);
    generate_stream->setRequestId(request_id);
    generate_stream->setDeadlineMs(deadline_ms);
    auto meta = std::make_shared<P2PMetaForConnectorTest>(generate_stream);
    connector_->asyncMatch(resource, meta);

    // 4. 创建 request
    auto request = createValidStartLoadRequest(unique_key, deadline_ms, 2);

    // 5. 设置 TestRpcServer 返回成功（用于 scheduler_->sendKVCache）
    for (auto& server : tp_broadcast_servers_) {
        server->service()->setP2PResponseSuccess(true);
    }

    P2PConnectorStartLoadResponsePB response;
    connector_->handleRead(request, response);

    EXPECT_EQ(response.error_code(), ErrorCodePB::NONE_ERROR);

    // 8. 验证 response 中包含了正确的 first_token
    EXPECT_EQ(response.first_generate_token_id(), 12345);

    // 9. 验证 response 中包含了正确的 position_ids
    ASSERT_EQ(response.position_ids_size(), position_ids.size());
    for (size_t i = 0; i < position_ids.size(); ++i) {
        EXPECT_EQ(response.position_ids(i), position_ids[i]);
    }

    // 10. 验证 response 中包含了正确的 reuse_length
    EXPECT_EQ(response.total_reuse_len(), 10);
    EXPECT_EQ(response.local_reuse_len(), 5);
    EXPECT_EQ(response.remote_reuse_len(), 5);
    EXPECT_EQ(response.memory_reuse_len(), 0);  // mock setReuseLength(10, 5, 5) leaves memory default 0

    // 11. 验证 response 中包含了正确的 propose_info
    ASSERT_EQ(response.propose_token_ids_size(), propose_tokens.size());
    for (size_t i = 0; i < propose_tokens.size(); ++i) {
        EXPECT_EQ(response.propose_token_ids(i), propose_tokens[i]);
    }

    // 12. 验证 scheduler_->sendKVCache 被调用（通过验证 BroadcastTp 被调用）
    for (size_t i = 0; i < tp_broadcast_servers_.size(); ++i) {
        EXPECT_GE(tp_broadcast_servers_[i]->service()->getBroadcastTpCallCount(), 1);
    }
}

}  // namespace rtp_llm
