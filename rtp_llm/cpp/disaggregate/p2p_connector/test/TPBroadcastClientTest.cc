#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/TPBroadcastClient.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/Exception.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/test/TestRpcServer.h"

namespace rtp_llm {

class TPBroadcastClientTest: public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试用的 RPC 服务器
        for (int i = 0; i < 2; ++i) {
            auto service = std::make_unique<TestRpcService>();
            auto server  = std::make_unique<TestRpcServer>(std::move(service));
            ASSERT_TRUE(server->start());
            servers_.push_back(std::move(server));
            server_addrs_.push_back("127.0.0.1:" + std::to_string(servers_.back()->listenPort()));
        }

        // 创建 GptInitParameter
        gpt_init_parameter_.worker_grpc_addrs_ = server_addrs_;

        // 创建 TPBroadcastClient
        client_ = std::make_unique<TPBroadcastClient>(gpt_init_parameter_);
        ASSERT_TRUE(client_->init());
    }

    void TearDown() override {
        client_.reset();
        servers_.clear();
    }

protected:
    std::vector<std::unique_ptr<TestRpcServer>> servers_;
    std::vector<std::string>                    server_addrs_;
    GptInitParameter                            gpt_init_parameter_;
    std::unique_ptr<TPBroadcastClient>          client_;

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
};

// ---------------------------- broadcast ----------------------------

TEST_F(TPBroadcastClientTest, Broadcast_ReturnNotNull_AllRequestsSuccess) {
    std::string unique_key  = "test_broadcast_1";
    int64_t     request_id  = 1001;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    // 创建 LayerCacheBuffer
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));
    layer_cache_buffers.push_back(createLayerCacheBuffer(1, 2));

    // 创建 decode_transfer_servers
    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});
    decode_transfer_servers.push_back({"127.0.0.1", 12346});

    // 执行 broadcast
    auto result = client_->broadcast(request_id, layer_cache_buffers, decode_transfer_servers, unique_key, deadline_ms);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->unique_key, unique_key);
    ASSERT_NE(result->result, nullptr);

    // 等待完成
    result->result->waitDone();
    EXPECT_TRUE(result->success());
    EXPECT_TRUE(result->result->success());

    // 验证响应
    auto responses = result->result->responses();
    EXPECT_EQ(responses.size(), server_addrs_.size());
    for (size_t i = 0; i < responses.size(); ++i) {
        EXPECT_TRUE(responses[i]->has_p2p_response());
        EXPECT_TRUE(responses[i]->p2p_response().success());
    }

    // 验证 BroadcastTp 被调用（每个服务器应该被调用一次）
    for (size_t i = 0; i < servers_.size(); ++i) {
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

TEST_F(TPBroadcastClientTest, Broadcast_ReturnNull_Timeout) {
    // 设置服务器延迟响应
    for (auto& server : servers_) {
        server->service()->setSleepMillis(200);
    }

    std::string unique_key  = "test_broadcast_timeout";
    int64_t     request_id  = 1002;
    int64_t     deadline_ms = currentTimeMs() + 10;  // 很短的超时时间

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    // 执行 broadcast，应该会因为超时返回 nullptr 或失败
    auto result =
        client_->broadcast(request_id, layer_cache_buffers, decode_transfer_servers, unique_key, deadline_ms, 50);

    // 注意：当发生超时时，waitDone() 会抛出 RTPException
    // broadcast 可能返回非空，但调用 waitDone() 时会因为超时抛出异常
    ASSERT_NE(result, nullptr);
    ASSERT_NE(result->result, nullptr);

    // 期望 waitDone() 抛出 RTPException 异常
    EXPECT_THROW(result->result->waitDone(), RTPException);

    // 验证 BroadcastTp 被调用（即使超时，也应该被调用）
    for (size_t i = 0; i < servers_.size(); ++i) {
        EXPECT_GE(servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

TEST_F(TPBroadcastClientTest, Broadcast_ReturnNotNull_PartialResponseFailed) {
    // 设置第一个服务器返回失败
    servers_[0]->service()->setP2PResponseSuccess(false);

    std::string unique_key  = "test_broadcast_partial_fail";
    int64_t     request_id  = 1003;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    auto result = client_->broadcast(request_id, layer_cache_buffers, decode_transfer_servers, unique_key, deadline_ms);
    ASSERT_NE(result, nullptr);
    ASSERT_NE(result->result, nullptr);
    result->result->waitDone();

    EXPECT_FALSE(result->success());
    EXPECT_TRUE(result->result->success());
    auto responses = result->result->responses();
    EXPECT_EQ(2, responses.size());
    EXPECT_TRUE(responses[0]->has_p2p_response());
    EXPECT_FALSE(responses[0]->p2p_response().success());
    EXPECT_TRUE(responses[1]->has_p2p_response());
    EXPECT_TRUE(responses[1]->p2p_response().success());

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < servers_.size(); ++i) {
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

TEST_F(TPBroadcastClientTest, Broadcast_ReturnNotNull_AllResponseFailed) {
    // 设置第一个服务器返回失败
    servers_[0]->service()->setP2PResponseSuccess(false);
    servers_[1]->service()->setP2PResponseSuccess(false);

    std::string unique_key  = "test_broadcast_partial_fail";
    int64_t     request_id  = 1003;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    auto result = client_->broadcast(request_id, layer_cache_buffers, decode_transfer_servers, unique_key, deadline_ms);
    ASSERT_NE(result, nullptr);
    ASSERT_NE(result->result, nullptr);
    result->result->waitDone();

    EXPECT_FALSE(result->success());
    EXPECT_TRUE(result->result->success());
    const auto& responses = result->result->responses();
    EXPECT_EQ(2, responses.size());
    for (const auto& response : responses) {
        EXPECT_TRUE(response->has_p2p_response());
        EXPECT_FALSE(response->p2p_response().success());
    }

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < servers_.size(); ++i) {
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

TEST_F(TPBroadcastClientTest, Broadcast_ReturnNotNull_RpcStatusFailed) {
    // 设置第一个服务器返回失败
    servers_[0]->service()->setRpcResponseStatus(::grpc::Status(grpc::StatusCode::INTERNAL, "Internal error"));

    std::string unique_key  = "test_broadcast_partial_fail";
    int64_t     request_id  = 1003;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    auto result = client_->broadcast(request_id, layer_cache_buffers, decode_transfer_servers, unique_key, deadline_ms);
    ASSERT_NE(result, nullptr);
    ASSERT_NE(result->result, nullptr);
    result->result->waitDone();

    EXPECT_FALSE(result->success());
    EXPECT_FALSE(result->result->success());
    auto responses = result->result->responses();
    EXPECT_EQ(2, responses.size());
    EXPECT_FALSE(responses[0]->has_p2p_response());
    EXPECT_FALSE(responses[1]->has_p2p_response());
}

// ---------------------------- cancel ----------------------------

TEST_F(TPBroadcastClientTest, Cancel_ReturnSuccess) {
    std::string unique_key  = "test_cancel";
    int64_t     request_id  = 2001;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    auto result = std::make_shared<TPBroadcastClient::Result>(unique_key, nullptr);
    EXPECT_NO_THROW(client_->cancel(result));

    // 验证 cancel 调用（每个服务器应该被调用一次 cancel）
    for (size_t i = 0; i < servers_.size(); ++i) {
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCancelCallCount(), 1);
    }
}

TEST_F(TPBroadcastClientTest, Cancel_ReturnFail) {
    std::string unique_key  = "test_cancel";
    int64_t     request_id  = 2002;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    servers_[0]->service()->setP2PResponseSuccess(false);

    auto result = std::make_shared<TPBroadcastClient::Result>(unique_key, nullptr);
    EXPECT_THROW(client_->cancel(result), RTPException);

    // 验证 cancel 调用（每个服务器应该被调用一次 cancel）
    for (size_t i = 0; i < servers_.size(); ++i) {
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCancelCallCount(), 1);
    }
}

TEST_F(TPBroadcastClientTest, Cancel_ReturnFail_Timeout) {
    std::string unique_key  = "test_cancel";
    int64_t     request_id  = 2002;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    servers_[0]->service()->setSleepMillis(200);

    auto result = std::make_shared<TPBroadcastClient::Result>(unique_key, nullptr);
    EXPECT_THROW(client_->cancel(result, 50), RTPException);

    // 验证 cancel 调用（每个服务器应该被调用一次 cancel）
    for (size_t i = 0; i < servers_.size(); ++i) {
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCancelCallCount(), 1);
    }
}

TEST_F(TPBroadcastClientTest, Cancel_ReturnFail_RpcStatusFailed) {
    servers_[0]->service()->setRpcResponseStatus(::grpc::Status(grpc::StatusCode::INTERNAL, "Internal error"));

    std::string unique_key  = "test_cancel";
    int64_t     request_id  = 2002;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    auto result = std::make_shared<TPBroadcastClient::Result>(unique_key, nullptr);
    EXPECT_THROW(client_->cancel(result), RTPException);

    EXPECT_EQ(servers_[0]->service()->getBroadcastTpCancelCallCount(), 1);
    // Server 1 可能因为 cancel 请求太快，没有被调用
}

}  // namespace rtp_llm