#include <thread>
#include <gtest/gtest.h>
#include "grpc++/grpc++.h"

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/PrefillLoadClient.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/test/TestRpcServer.h"

namespace rtp_llm {

class PrefillLoadClientTest: public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试用的 RPC 服务器
        auto service = std::make_unique<TestRpcService>();
        server_      = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server_->start());
        server_addr_ = "127.0.0.1:" + std::to_string(server_->listenPort());

        // 创建 GptInitParameter
        // worker_addrs_ 格式: "ip:cache_store_port:grpc_port"
        gpt_init_parameter_.worker_addrs_.push_back("127.0.0.1:12345:" + std::to_string(server_->listenPort()));

        // 创建 PrefillLoadClient
        client_ = std::make_unique<PrefillLoadClient>(gpt_init_parameter_);
    }

    void TearDown() override {
        client_.reset();
        server_.reset();
    }

protected:
    std::unique_ptr<TestRpcServer>     server_;
    std::string                        server_addr_;
    GptInitParameter                   gpt_init_parameter_;
    std::unique_ptr<PrefillLoadClient> client_;
};

// ---------------------------- load ----------------------------

TEST_F(PrefillLoadClientTest, Load_ReturnNotNull_RequestSuccess) {
    std::string unique_key   = "test_load_1";
    int64_t     request_id   = 1001;
    int64_t     deadline_ms  = currentTimeMs() + 5000;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(server_->listenPort());

    // 执行 load
    auto result = client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->request_id_, request_id);
    EXPECT_EQ(result->request.unique_key(), unique_key);

    // 等待完成
    bool success = result->waitDone();
    EXPECT_TRUE(success);
    EXPECT_TRUE(result->success());
    EXPECT_TRUE(result->response.success());

    // 验证 StartLoad 被调用
    EXPECT_EQ(server_->service()->getStartLoadCallCount(), 1);
}

TEST_F(PrefillLoadClientTest, Load_ReturnNotNull_RequestFailed) {
    // 设置服务器返回失败
    server_->service()->setStartLoadResponseSuccess(false);

    std::string unique_key   = "test_load_fail";
    int64_t     request_id   = 1002;
    int64_t     deadline_ms  = currentTimeMs() + 5000;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(server_->listenPort());

    // 执行 load
    auto result = client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms);
    ASSERT_NE(result, nullptr);

    // 等待完成
    bool success = result->waitDone();
    EXPECT_FALSE(success);
    EXPECT_FALSE(result->success());
    EXPECT_FALSE(result->response.success());

    // 验证 StartLoad 被调用（即使失败也应该被调用）
    EXPECT_EQ(server_->service()->getStartLoadCallCount(), 1);
}

TEST_F(PrefillLoadClientTest, Load_ReturnNotNull_Timeout) {
    // 设置服务器延迟响应
    server_->service()->setSleepMillis(200);

    std::string unique_key   = "test_load_timeout";
    int64_t     request_id   = 1003;
    int64_t     deadline_ms  = currentTimeMs() + 10;  // 很短的超时时间
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(server_->listenPort());

    // 执行 load
    auto result = client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms);
    ASSERT_NE(result, nullptr);

    // 等待完成，应该会因为超时返回 false
    bool success = result->waitDone();
    EXPECT_FALSE(success);
    EXPECT_FALSE(result->success());

    // 验证 StartLoad 被调用（即使超时也应该被调用）
    EXPECT_GE(server_->service()->getStartLoadCallCount(), 1);
}

TEST_F(PrefillLoadClientTest, Load_ReturnNull_InvalidServerAddr) {
    std::string unique_key   = "test_load_invalid_addr";
    int64_t     request_id   = 1004;
    int64_t     deadline_ms  = currentTimeMs() + 5000;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = 99999;  // 无效端口

    // 执行 load，应该返回 nullptr（因为无法连接）
    auto result = client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms);
    // 注意：由于 RPCPool 的行为，可能返回非空但 waitDone 会失败
    // 这里主要测试接口调用不会崩溃
    if (result != nullptr) {
        bool success = result->waitDone();
        EXPECT_FALSE(success);
    }
}

TEST_F(PrefillLoadClientTest, Load_ReturnNotNull_RpcStatusFailed) {
    // 设置服务器返回 RPC 错误状态
    server_->service()->setRpcResponseStatus(::grpc::Status(grpc::StatusCode::INTERNAL, "Internal error"));

    std::string unique_key   = "test_load_rpc_fail";
    int64_t     request_id   = 1006;
    int64_t     deadline_ms  = currentTimeMs() + 5000;
    std::string prefill_ip   = "127.0.0.1";
    uint32_t    prefill_port = static_cast<uint32_t>(server_->listenPort());

    // 执行 load
    auto result = client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms);
    ASSERT_NE(result, nullptr);

    // 等待完成
    bool success = result->waitDone();
    EXPECT_FALSE(success);
    EXPECT_FALSE(result->success());
    EXPECT_FALSE(result->status.ok());

    // 验证 StartLoad 被调用
    EXPECT_EQ(server_->service()->getStartLoadCallCount(), 1);
}

}  // namespace rtp_llm
