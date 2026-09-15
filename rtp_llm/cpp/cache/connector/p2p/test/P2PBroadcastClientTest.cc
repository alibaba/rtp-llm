#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/Exception.h"
#include "rtp_llm/cpp/cache/connector/p2p/test/TestRpcServer.h"

namespace rtp_llm {

class P2PBroadcastClientTest: public ::testing::Test {
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

        client_ = std::make_unique<P2PBroadcastClient>(server_addrs_);
        ASSERT_TRUE(client_->init());
    }

    void TearDown() override {
        client_.reset();
        servers_.clear();
    }

    void waitDone(std::shared_ptr<P2PBroadcastClient::Result>& result, int timeout_ms = 1000) {
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
    std::unique_ptr<P2PBroadcastClient>         client_;

    /// 本 worker 的一份传输计划：一条 route 带它自己的块视图。
    static TransferRoutePB createRoute(int32_t route_id, int64_t cache_key, int32_t block_id) {
        TransferRoutePB route;
        route.set_route_id(route_id);
        route.set_cache_tag("full");
        auto* layer_block = route.add_layer_blocks();
        layer_block->set_layer_id(0);
        layer_block->add_cache_keys(cache_key);
        layer_block->add_block_ids(block_id);
        return route;
    }

    static P2PBroadcastClient::BroadcastParams
    createParams(int64_t request_id, const std::string& unique_key, int64_t deadline_ms, int64_t request_deadline_ms) {
        P2PBroadcastClient::BroadcastParams params;
        params.request_id          = request_id;
        params.unique_key          = unique_key;
        params.deadline_ms         = deadline_ms;
        params.request_deadline_ms = request_deadline_ms;
        params.type                = P2PConnectorBroadcastType::READ;
        return params;
    }
};

// ---------------------------- broadcast ----------------------------

TEST_F(P2PBroadcastClientTest, Broadcast_ReturnNotNull_AllRequestsSuccess) {
    std::string unique_key  = "test_broadcast_1";
    int64_t     request_id  = 1001;
    int64_t     deadline_ms = currentTimeMs() + 5000;
    int64_t     request_deadline_ms = deadline_ms + 5000;

    auto params         = createParams(request_id, unique_key, deadline_ms, request_deadline_ms);
    params.peer_workers = {{"127.0.0.1", 12345}, {"127.0.0.1", 12346}};

    // 执行 broadcast
    auto result = client_->broadcast(std::move(params));
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
        EXPECT_EQ(servers_[i]->service()->getLastBroadcastTpRequest().deadline_ms(), deadline_ms);
        EXPECT_EQ(servers_[i]->service()->getLastBroadcastTpRequest().request_deadline_ms(), request_deadline_ms);
        EXPECT_EQ(servers_[i]->service()->getLastBroadcastTpRequest().peer_workers_size(), 2);
    }
}

TEST_F(P2PBroadcastClientTest, BroadcastSendsEachWorkerItsOwnRoutes) {
    const int64_t  deadline_ms = currentTimeMs() + 5000;
    const uint64_t plan_digest = 0xfeedface;

    auto params        = createParams(1010, "cp-rank-view", deadline_ms, deadline_ms);
    params.routes      = {{createRoute(0, 100, 10)}, {createRoute(1, 101, 11)}};
    params.plan_digest = plan_digest;

    auto result = client_->broadcast(std::move(params));
    ASSERT_NE(result, nullptr);
    waitDone(result);
    ASSERT_TRUE(result->success());

    // 每个 worker 只拿到自己那份 route，块视图随 route 内嵌下发。
    for (size_t rank = 0; rank < servers_.size(); ++rank) {
        const auto request = servers_[rank]->service()->getLastBroadcastTpRequest();
        ASSERT_EQ(request.routes_size(), 1);
        EXPECT_EQ(request.routes(0).route_id(), static_cast<int32_t>(rank));
        ASSERT_EQ(request.routes(0).layer_blocks_size(), 1);
        EXPECT_EQ(request.routes(0).layer_blocks(0).cache_keys(0), 100 + static_cast<int64_t>(rank));
        EXPECT_EQ(request.routes(0).layer_blocks(0).block_ids(0), 10 + static_cast<int32_t>(rank));
        EXPECT_EQ(request.plan_digest(), plan_digest);
    }
}

TEST_F(P2PBroadcastClientTest, BroadcastAllowsEmptyRoutesForOneWorker) {
    const int64_t deadline_ms = currentTimeMs() + 5000;

    auto params   = createParams(1011, "cp-empty-rank-view", deadline_ms, deadline_ms);
    params.routes = {{createRoute(0, 100, 10)}, {}};

    auto result = client_->broadcast(std::move(params));
    ASSERT_NE(result, nullptr);
    waitDone(result);
    ASSERT_TRUE(result->success());

    // rank1 的空 routes 即权威的「本 worker 无任务」；顶层 layer_blocks 不再上线。
    EXPECT_EQ(servers_[0]->service()->getLastBroadcastTpRequest().routes_size(), 1);
    EXPECT_EQ(servers_[1]->service()->getLastBroadcastTpRequest().routes_size(), 0);
    EXPECT_EQ(servers_[0]->service()->getLastBroadcastTpRequest().layer_blocks_size(), 0);
    EXPECT_EQ(servers_[1]->service()->getLastBroadcastTpRequest().layer_blocks_size(), 0);
}

TEST_F(P2PBroadcastClientTest, BroadcastRejectsMismatchedRouteCount) {
    const int64_t deadline_ms = currentTimeMs() + 5000;

    auto params   = createParams(1012, "cp-invalid-rank-view", deadline_ms, deadline_ms);
    params.routes = {{createRoute(0, 100, 10)}};

    EXPECT_EQ(client_->broadcast(std::move(params)), nullptr);
}

TEST_F(P2PBroadcastClientTest, Broadcast_ReturnNotNull_Timeout) {
    // 设置服务器延迟响应
    for (auto& server : servers_) {
        server->service()->setSleepMillis(200);
    }

    std::string unique_key  = "test_broadcast_timeout";
    int64_t     request_id  = 1002;
    int64_t     deadline_ms = currentTimeMs() + 10;  // 很短的超时时间

    // 执行 broadcast
    auto result = client_->broadcast(createParams(request_id, unique_key, deadline_ms, deadline_ms));

    ASSERT_NE(result, nullptr);

    waitDone(result, 500);
    EXPECT_TRUE(result->done());
    EXPECT_FALSE(result->success());
}

TEST_F(P2PBroadcastClientTest, Broadcast_ReturnNotNull_PartialResponseFailed) {
    // 设置第一个服务器返回失败
    servers_[0]->service()->setP2PResponseSuccess(false);

    std::string unique_key  = "test_broadcast_partial_fail";
    int64_t     request_id  = 1003;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    auto result = client_->broadcast(createParams(request_id, unique_key, deadline_ms, deadline_ms));
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

TEST_F(P2PBroadcastClientTest, Broadcast_ReturnNotNull_AllResponseFailed) {
    // 设置所有服务器返回失败
    servers_[0]->service()->setP2PResponseSuccess(false);
    servers_[1]->service()->setP2PResponseSuccess(false);

    std::string unique_key  = "test_broadcast_all_fail";
    int64_t     request_id  = 1004;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    auto result = client_->broadcast(createParams(request_id, unique_key, deadline_ms, deadline_ms));
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

TEST_F(P2PBroadcastClientTest, Broadcast_MissingP2PResponseHasNonSuccessError) {
    servers_[0]->service()->setOmitP2PResponse(true);
    const int64_t deadline_ms = currentTimeMs() + 5000;

    auto result = client_->broadcast(createParams(1006, "test_broadcast_missing_response", deadline_ms, deadline_ms));
    ASSERT_NE(result, nullptr);
    waitDone(result);

    EXPECT_TRUE(result->done());
    EXPECT_FALSE(result->success());
    EXPECT_EQ(result->errorCode(), ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED);
    EXPECT_NE(result->errorMessage().find("missing p2p_response"), std::string::npos);
}

TEST_F(P2PBroadcastClientTest, Broadcast_ReturnNotNull_RpcStatusFailed) {
    // 设置第一个服务器返回 RPC 错误
    servers_[0]->service()->setRpcResponseStatus(::grpc::Status(grpc::StatusCode::INTERNAL, "Internal error"));

    std::string unique_key  = "test_broadcast_rpc_fail";
    int64_t     request_id  = 1005;
    int64_t     deadline_ms = currentTimeMs() + 5000;

    auto result = client_->broadcast(createParams(request_id, unique_key, deadline_ms, deadline_ms));
    ASSERT_NE(result, nullptr);

    waitDone(result);

    EXPECT_FALSE(result->success());
}

TEST_F(P2PBroadcastClientTest, Cancel_ReturnNotNull_Success) {
    std::string unique_key          = "test_cancel_success";
    int64_t     request_id          = 3013;
    int64_t     deadline_ms         = currentTimeMs() + 1000;
    int64_t     request_deadline_ms = currentTimeMs() + 5000;

    // 执行 cancel
    auto result = client_->cancel(
        unique_key, P2PConnectorBroadcastType::CANCEL_READ, request_deadline_ms, request_id, deadline_ms);
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
        EXPECT_EQ(servers_[i]->service()->getLastBroadcastTpRequest().request_id(), request_id);
        EXPECT_EQ(servers_[i]->service()->getLastBroadcastTpRequest().deadline_ms(), deadline_ms);
        EXPECT_EQ(servers_[i]->service()->getLastBroadcastTpRequest().request_deadline_ms(), request_deadline_ms);
    }
}


TEST_F(P2PBroadcastClientTest, CancelSurvivesCallerReleaseAndReclaimsAfterFinish) {
    for (auto& server : servers_) {
        server->service()->setP2PRequestSleepMillis(P2PConnectorBroadcastType::CANCEL_HANDLE_READ, 50);
    }
    auto result = client_->cancel("cancel-release", P2PConnectorBroadcastType::CANCEL_HANDLE_READ,
                                  currentTimeMs() + 5000, 3014, currentTimeMs() + 1000);
    ASSERT_NE(result, nullptr);
    std::weak_ptr<P2PBroadcastClient::TpBroadcastResult> pending = result->tp_broadcast_result_;
    result.reset();
    const auto deadline = currentTimeMs() + 2000;
    while (!pending.expired() && currentTimeMs() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(pending.expired());
    for (auto& server : servers_) {
        EXPECT_EQ(server->service()->getBroadcastTpCancelCallCount(), 1);
    }
}

}  // namespace rtp_llm
