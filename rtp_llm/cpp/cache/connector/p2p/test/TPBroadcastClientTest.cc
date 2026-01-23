#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/TPBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/Exception.h"
#include "rtp_llm/cpp/cache/connector/p2p/test/TestRpcServer.h"

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

        // 创建 TPBroadcastClient
        client_ = std::make_unique<TPBroadcastClient>(server_addrs_);
        ASSERT_TRUE(client_->init());
    }

    void TearDown() override {
        client_.reset();
        servers_.clear();
    }

    // 等待 Result 完成
    void waitDone(std::shared_ptr<TPBroadcastClient::Result>& result, int timeout_ms = 1000) {
        int waited_ms = 0;
        while (!result->done() && waited_ms < timeout_ms) {
            result->checkDone();
            if (!result->done()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                waited_ms += 10;
            }
        }
    }

protected:
    std::vector<std::unique_ptr<TestRpcServer>> servers_;
    std::vector<std::string>                    server_addrs_;
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
    auto result = client_->broadcast(request_id,
                                     layer_cache_buffers,
                                     decode_transfer_servers,
                                     unique_key,
                                     deadline_ms,
                                     P2PConnectorBroadcastType::READ);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->uniqueKey(), unique_key);

    // 等待完成
    waitDone(result);
    EXPECT_TRUE(result->done());
    EXPECT_TRUE(result->success());

    // 验证 BroadcastTp 被调用（每个服务器应该被调用一次）
    for (size_t i = 0; i < servers_.size(); ++i) {
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

TEST_F(TPBroadcastClientTest, Broadcast_ReturnNotNull_Timeout) {
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

    client_->setExtraWaitTimeMs(50);

    // 执行 broadcast
    auto result = client_->broadcast(request_id,
                                     layer_cache_buffers,
                                     decode_transfer_servers,
                                     unique_key,
                                     deadline_ms,
                                     P2PConnectorBroadcastType::READ);

    ASSERT_NE(result, nullptr);

    // 等待时会因超时抛出 RTPException
    // TPBroadcastResult::waitDone 在 gRPC 超时（DEADLINE_EXCEEDED）时会调用 RTP_LLM_FAIL
    EXPECT_THROW(waitDone(result, 500), RTPException);
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

    auto result = client_->broadcast(request_id,
                                     layer_cache_buffers,
                                     decode_transfer_servers,
                                     unique_key,
                                     deadline_ms,
                                     P2PConnectorBroadcastType::READ);
    ASSERT_NE(result, nullptr);

    waitDone(result);

    EXPECT_TRUE(result->done());
    EXPECT_FALSE(result->success());

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < servers_.size(); ++i) {
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

TEST_F(TPBroadcastClientTest, Broadcast_ReturnNotNull_AllResponseFailed) {
    // 设置所有服务器返回失败
    servers_[0]->service()->setP2PResponseSuccess(false);
    servers_[1]->service()->setP2PResponseSuccess(false);

    std::string unique_key  = "test_broadcast_all_fail";
    int64_t     request_id  = 1004;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    auto result = client_->broadcast(request_id,
                                     layer_cache_buffers,
                                     decode_transfer_servers,
                                     unique_key,
                                     deadline_ms,
                                     P2PConnectorBroadcastType::READ);
    ASSERT_NE(result, nullptr);

    waitDone(result);

    EXPECT_TRUE(result->done());
    EXPECT_FALSE(result->success());

    // 验证 BroadcastTp 被调用
    for (size_t i = 0; i < servers_.size(); ++i) {
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCallCount(), 1);
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCancelCallCount(), 0);
    }
}

TEST_F(TPBroadcastClientTest, Broadcast_ReturnNotNull_RpcStatusFailed) {
    // 设置第一个服务器返回 RPC 错误
    servers_[0]->service()->setRpcResponseStatus(::grpc::Status(grpc::StatusCode::INTERNAL, "Internal error"));

    std::string unique_key  = "test_broadcast_rpc_fail";
    int64_t     request_id  = 1005;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.push_back(createLayerCacheBuffer(0, 2));

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    decode_transfer_servers.push_back({"127.0.0.1", 12345});

    auto result = client_->broadcast(request_id,
                                     layer_cache_buffers,
                                     decode_transfer_servers,
                                     unique_key,
                                     deadline_ms,
                                     P2PConnectorBroadcastType::READ);
    ASSERT_NE(result, nullptr);

    waitDone(result);

    EXPECT_FALSE(result->success());
}

TEST_F(TPBroadcastClientTest, Cancel_ReturnNotNull_Success) {
    std::string unique_key = "test_cancel_success";

    // 执行 cancel
    auto result = client_->cancel(unique_key, P2PConnectorBroadcastType::CANCEL_READ);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->uniqueKey(), unique_key);

    // 等待完成
    waitDone(result);
    EXPECT_TRUE(result->done());
    EXPECT_TRUE(result->success());

    // 验证 CANCEL_READ 被发送给所有 worker
    for (size_t i = 0; i < servers_.size(); ++i) {
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCallCount(), 0);
        EXPECT_EQ(servers_[i]->service()->getBroadcastTpCancelCallCount(), 1);
    }
}

}  // namespace rtp_llm
