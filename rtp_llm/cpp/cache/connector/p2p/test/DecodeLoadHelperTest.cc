#include <thread>
#include <atomic>
#include <future>
#include <mutex>
#include <unordered_map>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/DecodeLoadHelper.h"
#include "rtp_llm/cpp/utils/Exception.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/test/TestRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"

namespace rtp_llm {

class DecodeLoadHelperTest: public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试用的 RPC 服务器
        auto service = std::make_unique<TestRpcService>();
        server_      = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server_->start());
        server_addr_ = "127.0.0.1:" + std::to_string(server_->listenPort());

        // worker_addrs_ 格式: "host:cache_store_port:grpc_port"
        worker_addrs_.push_back("127.0.0.1:12345:" + std::to_string(server_->listenPort()));

        // 创建 DecodeLoadHelper
        client_ = std::make_unique<DecodeLoadHelper>(worker_addrs_);
    }

    void TearDown() override {
        client_.reset();
        server_.reset();
    }

    // Completion must progress without a caller polling its CQ.
    bool waitDone(std::shared_ptr<DecodeLoadHelper::Result>& result, int timeout_ms = 5000) {
        auto completion = std::make_shared<std::promise<void>>();
        auto ready      = completion->get_future();
        result->setDoneCallback([completion] { completion->set_value(); });
        return ready.wait_for(std::chrono::milliseconds(timeout_ms)) == std::future_status::ready && result->success();
    }

protected:
    std::unique_ptr<TestRpcServer>     server_;
    std::string                        server_addr_;
    std::vector<std::string>           worker_addrs_;
    std::unique_ptr<DecodeLoadHelper> client_;
};

TEST(DecodeLoadHelperWorkerAddrTest, ConstructorParsesHostIpv4AndIpv6WorkerAddrs) {
    DecodeLoadHelper caller({
        "127.0.0.1:12345:23456",
        "prefill-decode.local:12346:23457",
        "[::1]:12347:23458",
        "fe80::1:12348:23459",
        "::ffff:10.0.0.1:12349:23460",
        "[::ffff:10.0.0.2]:12350:23461",
    });

    ASSERT_EQ(caller.tp_worker_infos_.size(), 6);
    EXPECT_EQ(caller.tp_worker_infos_[0].ip(), "127.0.0.1");
    EXPECT_EQ(caller.tp_worker_infos_[0].cache_store_port(), 12345);
    EXPECT_EQ(caller.tp_worker_infos_[1].ip(), "prefill-decode.local");
    EXPECT_EQ(caller.tp_worker_infos_[1].cache_store_port(), 12346);
    EXPECT_EQ(caller.tp_worker_infos_[2].ip(), "::1");
    EXPECT_EQ(caller.tp_worker_infos_[2].cache_store_port(), 12347);
    EXPECT_EQ(caller.tp_worker_infos_[3].ip(), "fe80::1");
    EXPECT_EQ(caller.tp_worker_infos_[3].cache_store_port(), 12348);
    EXPECT_EQ(caller.tp_worker_infos_[4].ip(), "::ffff:10.0.0.1");
    EXPECT_EQ(caller.tp_worker_infos_[4].cache_store_port(), 12349);
    EXPECT_EQ(caller.tp_worker_infos_[5].ip(), "::ffff:10.0.0.2");
    EXPECT_EQ(caller.tp_worker_infos_[5].cache_store_port(), 12350);
}

TEST(DecodeLoadHelperWorkerAddrTest, ConstructorRejectsMalformedWorkerAddrOrInvalidPorts) {
    EXPECT_THROW(DecodeLoadHelper({"127.0.0.1:0:23456"}), rtp_llm::RTPException);
    EXPECT_THROW(DecodeLoadHelper({"127.0.0.1:12345:65536"}), rtp_llm::RTPException);
    EXPECT_THROW(DecodeLoadHelper({"127.0.0.1:12345:not-a-port"}), rtp_llm::RTPException);
    EXPECT_THROW(DecodeLoadHelper({"[::1]12345:23456"}), rtp_llm::RTPException);
    EXPECT_THROW(DecodeLoadHelper({"fe80::1"}), rtp_llm::RTPException);
    EXPECT_THROW(DecodeLoadHelper({"127.0.0.1:12345:23456:34567"}), rtp_llm::RTPException);
    EXPECT_THROW(DecodeLoadHelper({"::ffff:999.0.0.1:12345:23456"}), rtp_llm::RTPException);
}

// ---------------------------- load ----------------------------

TEST_F(DecodeLoadHelperTest, Load_ReturnNotNull_RequestSuccess) {
    std::string unique_key   = "test_load_1";
    int64_t     request_id   = 1001;
    int64_t     deadline_ms  = currentTimeMs() + 5000;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(server_->listenPort());

    // 执行 load
    auto result = client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms, deadline_ms, 5000);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->request_id, request_id);
    EXPECT_EQ(result->request.unique_key(), unique_key);

    // 等待完成
    bool success = waitDone(result);
    EXPECT_TRUE(success);
    EXPECT_TRUE(result->done());
    EXPECT_TRUE(result->success());
    EXPECT_EQ(result->response.error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_EQ(result->side_channel_payload.disk_reuse_len, 3);

    // 验证 StartLoad 被调用
    EXPECT_EQ(server_->service()->getStartLoadCallCount(), 1);
}

TEST_F(DecodeLoadHelperTest, LoadCarriesOnlyLoadTimeoutAndKeepsLocalDeadline) {
    const int64_t request_deadline_ms  = currentTimeMs() + 5000;
    const int64_t transfer_deadline_ms = currentTimeMs() + 500;
    constexpr uint64_t plan_digest     = 0xfedcba9876543210ULL;

    auto result = client_->load(1010,
                                "127.0.0.1",
                                static_cast<uint32_t>(server_->listenPort()),
                                "test_split_deadline",
                                request_deadline_ms,
                                transfer_deadline_ms,
                                1000,
                                false,
                                plan_digest);

    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->request.timeout_ms(), 1000);
    EXPECT_EQ(result->request.plan_digest(), plan_digest);
    EXPECT_GT(result->timeout_ms, 0);
    EXPECT_LE(result->timeout_ms, 500);
    EXPECT_TRUE(waitDone(result));
    EXPECT_EQ(server_->service()->getLastStartLoadRequest().plan_digest(), plan_digest);
}

TEST_F(DecodeLoadHelperTest, LoadReturnsPlanDigestMismatchBeforeTransferDeadline) {
    const auto error_code = ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED;
    const std::string message = "StartLoad plan digest mismatch: decode=1 prefill=2 decode_tp_size=1";
    server_->service()->setStartLoadApplicationError(transErrorCodeToRPC(error_code), message);
    const int64_t deadline_ms = currentTimeMs() + 60000;

    auto result = client_->load(1012,
                                "127.0.0.1",
                                static_cast<uint32_t>(server_->listenPort()),
                                "test_plan_digest_mismatch",
                                deadline_ms,
                                deadline_ms,
                                5000,
                                false,
                                1);
    ASSERT_NE(result, nullptr);
    EXPECT_FALSE(waitDone(result, 1000));
    ASSERT_TRUE(result->done());
    EXPECT_FALSE(result->success());
    EXPECT_EQ(result->error_code, error_code);
    EXPECT_EQ(result->error_message,
              message
                  + " [received_at location=DecodeLoadHelper::complete observer_role=DECODE operation=StartLoad peer="
                  + "127.0.0.1:" + std::to_string(server_->listenPort())
                  + " request_id=1012 key=test_plan_digest_mismatch wire_code=13]");
    EXPECT_EQ(server_->service()->getLastStartLoadRequest().plan_digest(), 1);
}

TEST_F(DecodeLoadHelperTest, LoadRejectsSuccessfulResponseWithoutFirstToken) {
    server_->service()->setFirstGenerateTokenId(0);
    const int64_t deadline_ms = currentTimeMs() + 5000;

    auto result = client_->load(1011,
                                "127.0.0.1",
                                static_cast<uint32_t>(server_->listenPort()),
                                "test_missing_first_token",
                                deadline_ms,
                                deadline_ms,
                                5000);

    ASSERT_NE(result, nullptr);
    EXPECT_FALSE(waitDone(result));
    EXPECT_TRUE(result->done());
    EXPECT_FALSE(result->success());
    EXPECT_EQ(result->error_code, ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED);
}

TEST_F(DecodeLoadHelperTest, Load_ReturnNotNull_RequestFailed) {
    // 设置服务器返回失败
    server_->service()->setStartLoadResponseSuccess(false);

    std::string unique_key   = "test_load_fail";
    int64_t     request_id   = 1002;
    int64_t     deadline_ms  = currentTimeMs() + 5000;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(server_->listenPort());

    // 执行 load
    auto result = client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms, deadline_ms, 5000);
    ASSERT_NE(result, nullptr);

    // 等待完成
    bool success = waitDone(result);
    EXPECT_FALSE(success);
    EXPECT_TRUE(result->done());
    EXPECT_FALSE(result->success());
    EXPECT_NE(result->response.error_code(), ErrorCodePB::NONE_ERROR);

    // 验证 StartLoad 被调用（即使失败也应该被调用）
    EXPECT_EQ(server_->service()->getStartLoadCallCount(), 1);
}

TEST_F(DecodeLoadHelperTest, Load_ReturnNotNull_Timeout) {
    // 设置服务器延迟响应
    server_->service()->setSleepMillis(200);

    std::string unique_key   = "test_load_timeout";
    int64_t     request_id   = 1003;
    int64_t     deadline_ms  = currentTimeMs() + 10;  // 很短的超时时间
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(server_->listenPort());

    // 执行 load
    auto result = client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms, deadline_ms, 5000);
    ASSERT_NE(result, nullptr);

    // 等待完成，应该会因为超时返回 false
    bool success = waitDone(result, 1000);
    EXPECT_FALSE(success);
    EXPECT_FALSE(result->success());

    // 验证 StartLoad 被调用（即使超时也应该被调用）
    EXPECT_GE(server_->service()->getStartLoadCallCount(), 1);
}

TEST_F(DecodeLoadHelperTest, CancelAndFinishNotifyExactlyOnce) {
    for (int iteration = 0; iteration < 50; ++iteration) {
        auto result = std::make_shared<DecodeLoadHelper::Result>();
        result->response.mutable_payload()->set_has_first_generate_token(true);
        auto callbacks = std::make_shared<std::atomic<int>>(0);
        result->setDoneCallback([callbacks] { callbacks->fetch_add(1); });
        std::promise<void> start;
        auto               gate = start.get_future().share();
        std::thread        finish([result, gate] {
            gate.wait();
            result->complete(true);
        });
        std::thread        cancel([result, gate] {
            gate.wait();
            result->cancel();
        });
        start.set_value();
        finish.join();
        cancel.join();
        EXPECT_TRUE(result->done());
        EXPECT_EQ(callbacks->load(), 1);
    }
}

TEST_F(DecodeLoadHelperTest, LateFinishCannotOverwriteCancellation) {
    auto result = std::make_shared<DecodeLoadHelper::Result>();
    result->response.mutable_payload()->set_has_first_generate_token(true);
    result->cancel();
    result->complete(true);
    EXPECT_TRUE(result->done());
    EXPECT_FALSE(result->success());
    EXPECT_EQ(result->error_code, ErrorCode::CANCELLED);
    EXPECT_FALSE(result->side_channel_payload.has_data);
}

TEST_F(DecodeLoadHelperTest, CompletionBeforeCallbackRegistrationNotifiesImmediately) {
    auto result = std::make_shared<DecodeLoadHelper::Result>();
    result->response.mutable_payload()->set_has_first_generate_token(true);
    result->complete(true);
    int calls = 0;
    result->setDoneCallback([&calls] { ++calls; });
    EXPECT_EQ(calls, 1);
    EXPECT_TRUE(result->success());
}

TEST_F(DecodeLoadHelperTest, Load_ReturnNull_InvalidServerAddr) {
    std::string unique_key   = "test_load_invalid_addr";
    int64_t     request_id   = 1004;
    int64_t     deadline_ms  = currentTimeMs() + 5000;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = 99999;  // 无效端口

    // 执行 load，应该返回 nullptr（因为无法连接）
    auto result = client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms, deadline_ms, 5000);
    // 注意：由于 RPCPool 的行为，可能返回非空但 waitDone 会失败
    // 这里主要测试接口调用不会崩溃
    if (result != nullptr) {
        bool success = waitDone(result);
        EXPECT_FALSE(success);
    }
}

TEST_F(DecodeLoadHelperTest, Load_NormalizesRawIpv6ServerAddr) {
    std::string unique_key   = "test_load_ipv6_addr";
    int64_t     request_id   = 1008;
    int64_t     deadline_ms  = currentTimeMs() + 100;
    std::string prefill_ip   = "::1";
    uint32_t    prefill_port = 65535;

    auto result = client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms, deadline_ms, 5000);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->server_addr, "[::1]:65535");
    result->cancel();
}

TEST_F(DecodeLoadHelperTest, Load_ReturnNull_InvalidTargetPort) {
    const int64_t deadline_ms = currentTimeMs() + 100;
    auto          result      = client_->load(1009, "::1", 0, "test_load_bad_port", deadline_ms, deadline_ms, 5000);
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->done());
    EXPECT_EQ(result->error_code, ErrorCode::INVALID_PARAMS);
    EXPECT_NE(result->error_message.find("test_load_bad_port"), std::string::npos);
}

TEST_F(DecodeLoadHelperTest, CompletionRejectsMissingPayload) {
    auto result = std::make_shared<DecodeLoadHelper::Result>();
    result->complete(true);
    EXPECT_TRUE(result->done());
    EXPECT_FALSE(result->success());
    EXPECT_EQ(result->error_code, ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED);
    EXPECT_FALSE(result->side_channel_payload.has_data);
}

TEST_F(DecodeLoadHelperTest, CompletionPreservesRngOffsetPresence) {
    for (uint64_t offset : {uint64_t{0}, uint64_t{1} << 63}) {
        auto  result  = std::make_shared<DecodeLoadHelper::Result>();
        auto* payload = result->response.mutable_payload();
        payload->set_has_first_generate_token(true);
        payload->set_first_generate_token_id(7);
        payload->set_has_rng_state(true);
        payload->set_rng_offset(offset);
        result->complete(true);
        ASSERT_TRUE(result->success());
        EXPECT_TRUE(result->side_channel_payload.has_rng_state);
        EXPECT_EQ(result->side_channel_payload.rng_offset, offset);
    }
}


TEST_F(DecodeLoadHelperTest, CompletionAcceptsExplicitZeroFirstToken) {
    auto  result  = std::make_shared<DecodeLoadHelper::Result>();
    auto* payload = result->response.mutable_payload();
    payload->set_has_first_generate_token(true);
    payload->set_first_generate_token_id(0);
    result->complete(true);
    EXPECT_TRUE(result->success());
    EXPECT_TRUE(result->side_channel_payload.has_first_token);
    EXPECT_EQ(result->side_channel_payload.first_token_id, 0);
}

TEST_F(DecodeLoadHelperTest, Load_ReturnNotNull_RpcStatusFailed) {
    // 设置服务器返回 RPC 错误状态
    server_->service()->setRpcResponseStatus(::grpc::Status(grpc::StatusCode::INTERNAL, "Internal error"));

    std::string unique_key   = "test_load_rpc_fail";
    int64_t     request_id   = 1006;
    int64_t     deadline_ms  = currentTimeMs() + 5000;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(server_->listenPort());

    // 执行 load
    auto result = client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms, deadline_ms, 5000);
    ASSERT_NE(result, nullptr);

    // 等待完成
    bool success = waitDone(result);
    EXPECT_FALSE(success);
    EXPECT_TRUE(result->done());
    EXPECT_FALSE(result->success());
    EXPECT_FALSE(result->status.ok());

    // 验证 StartLoad 被调用
    EXPECT_EQ(server_->service()->getStartLoadCallCount(), 1);
}

// ---------------------------- checkDone ----------------------------

TEST_F(DecodeLoadHelperTest, CheckDone_NotDoneInitially) {
    // 设置服务器延迟响应
    server_->service()->setSleepMillis(500);

    std::string unique_key   = "test_check_done";
    int64_t     request_id   = 2001;
    int64_t     deadline_ms  = currentTimeMs() + 5000;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(server_->listenPort());

    auto result = client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms, deadline_ms, 5000);
    ASSERT_NE(result, nullptr);

    // 初始状态应该是 not done
    EXPECT_FALSE(result->done());
    EXPECT_FALSE(result->success());

    // No caller-side CQ polling is required.
    // 由于服务器延迟，应该还没完成
    // 注意：这个测试可能有时间敏感性

    // 最终等待完成
    waitDone(result);
    EXPECT_TRUE(result->done());
}

TEST_F(DecodeLoadHelperTest, CheckDone_TotalCostTimeUs) {
    std::string unique_key   = "test_cost_time";
    int64_t     request_id   = 2002;
    int64_t     deadline_ms  = currentTimeMs() + 5000;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(server_->listenPort());

    auto result = client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms, deadline_ms, 5000);
    ASSERT_NE(result, nullptr);

    waitDone(result);
    EXPECT_TRUE(result->done());

    // 验证总耗时被记录
    int64_t cost_time_us = result->totalCostTimeUs();
    EXPECT_GT(cost_time_us, 0);
}

// ---------------------------- cancel (bounded drain — P0-1 fix) ----------------------------

// Regression for the 8-min decode-side stall (DingTalk doc §7). Before the fix,
// shutdownAndDrainCompletionQueue used an unbounded Next() that could block for the
// remaining gRPC deadline when the peer channel was unhealthy. With the bounded-drain
// fix, cancel() must return while the remote handler is still blocked.
TEST_F(DecodeLoadHelperTest, Cancel_BoundedByDrainDeadline_WhenServerSlow) {
    auto entered = std::make_shared<std::promise<void>>();
    auto reached = entered->get_future();
    auto release = std::make_shared<std::promise<void>>();
    auto gate = release->get_future().share();
    auto exited = std::make_shared<std::atomic<bool>>(false);
    auto cleanup = std::shared_ptr<void>(nullptr, [release](void*) { release->set_value(); });
    server_->service()->setStartLoadGate([entered, gate, exited] {
        entered->set_value();
        gate.wait();
        exited->store(true);
    });

    std::string unique_key   = "test_cancel_bounded";
    int64_t     request_id   = 3001;
    int64_t     deadline_ms  = currentTimeMs() + 30000;  // long deadline so timeout isn't the bound
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(server_->listenPort());

    auto result = client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms, deadline_ms, 5000);
    ASSERT_NE(result, nullptr);

    const auto reached_status = reached.wait_for(std::chrono::seconds(5));
    if (reached_status != std::future_status::ready) cleanup.reset();
    ASSERT_EQ(reached_status, std::future_status::ready);
    EXPECT_FALSE(result->done());
    auto cancelled = std::async(std::launch::async, [result] { result->cancel(); });
    const auto ready = cancelled.wait_for(std::chrono::seconds(5));
    // Causal check: cancellation returns while the remote handler remains gated.
    EXPECT_EQ(ready, std::future_status::ready);
    EXPECT_FALSE(exited->load());
    cleanup.reset();
    cancelled.get();
    EXPECT_TRUE(result->done());
    EXPECT_FALSE(result->success());
}

TEST_F(DecodeLoadHelperTest, Cancel_Idempotent) {
    server_->service()->setSleepMillis(200);

    std::string unique_key   = "test_cancel_idempotent";
    int64_t     request_id   = 3002;
    int64_t     deadline_ms  = currentTimeMs() + 5000;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(server_->listenPort());

    auto result = client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms, deadline_ms, 5000);
    ASSERT_NE(result, nullptr);

    result->cancel();
    EXPECT_TRUE(result->done());
    EXPECT_FALSE(result->success());
    EXPECT_EQ(result->error_code, ErrorCode::CANCELLED);

    // Idempotence preserves the completed outcome.
    const auto error = result->error_code;
    result->cancel();
    EXPECT_TRUE(result->done());
    EXPECT_FALSE(result->success());
    EXPECT_EQ(result->error_code, error);

}

}  // namespace rtp_llm
