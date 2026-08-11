#include <chrono>
#include <string>
#include <thread>

#include "grpcpp/impl/codegen/time.h"
#include "gtest/gtest.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RemoteLoadFence.h"

namespace rtp_llm {
namespace {

using namespace std::chrono_literals;

int64_t unixMs(std::chrono::system_clock::time_point time_point) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(time_point.time_since_epoch()).count();
}

void setDeadline(grpc::ServerContext& context, std::chrono::milliseconds remaining) {
    grpc::Timepoint2Timespec(std::chrono::system_clock::now() + remaining, &context.deadline_);
}

std::string makePastDeadlineToken(const std::string& identity, int64_t& deadline_unix_ms) {
    deadline_unix_ms = unixMs(std::chrono::system_clock::now()) - 1;
    auto token = makeRemoteLoadAllocationToken("decode-owner", identity, deadline_unix_ms);
    EXPECT_TRUE(token.ok()) << token.status();
    return token.ok() ? *token : std::string();
}

class TestableDecodeRpcServer: public DecodeRpcServer {
public:
    void setDpRank(int dp_rank) {
        maga_init_params_.parallelism_config.dp_rank = dp_rank;
    }

    void setLoadCacheTimeoutMs(int64_t timeout_ms) {
        maga_init_params_.pd_sep_config.load_cache_timeout_ms = timeout_ms;
    }

    absl::Status beginRemoteLoadForTest(const std::string& token,
                                        int64_t            deadline_unix_ms,
                                        std::chrono::steady_clock::time_point local_expiry) {
        auto operation = remote_load_fences_.begin(token, deadline_unix_ms, local_expiry);
        return operation.ok() ? absl::OkStatus() : operation.status();
    }
};

TEST(RemoteLoadAdmissionTest, ReceivingRpcUsesLocalTransportDeadlineAcrossClockSkew) {
    TestableDecodeRpcServer server;
    server.setDpRank(0);

    int64_t deadline_unix_ms = 0;
    const auto token = makePastDeadlineToken("clock-skew", deadline_unix_ms);
    BroadcastLoadRequestPB request;
    request.set_allocation_token(token);
    request.set_load_deadline_unix_ms(deadline_unix_ms);
    request.set_timeout_ms(1'000);
    request.set_dp_rank(1);

    grpc::ServerContext    context;
    BroadcastLoadResponsePB response;
    setDeadline(context, 1s);

    EXPECT_TRUE(server.RemoteLoad(&context, &request, &response).ok());
    EXPECT_NE(response.error_info().error_code(), ErrorCodePB::LOAD_CACHE_TIMEOUT);
    EXPECT_NE(response.error_info().error_message().find("wrong data-parallel rank"), std::string::npos);
}

TEST(RemoteLoadAdmissionTest, QuiesceBeforeRemoteLoadSealsDelayedOperation) {
    TestableDecodeRpcServer server;
    server.setDpRank(0);
    server.setLoadCacheTimeoutMs(10);

    int64_t deadline_unix_ms = 0;
    const auto token = makePastDeadlineToken("delayed-operation", deadline_unix_ms);
    RemoteLoadQuiesceRequestPB quiesce_request;
    quiesce_request.set_allocation_token(token);
    quiesce_request.set_load_deadline_unix_ms(deadline_unix_ms);
    quiesce_request.set_local_only(true);
    quiesce_request.set_retention_timeout_ms(500);

    grpc::ServerContext          quiesce_context;
    RemoteLoadQuiesceResponsePB quiesce_response;
    setDeadline(quiesce_context, 20ms);
    EXPECT_TRUE(server.QuiesceRemoteLoad(&quiesce_context, &quiesce_request, &quiesce_response).ok());
    ASSERT_TRUE(quiesce_response.quiesced());
    ASSERT_EQ(quiesce_response.error_info().error_code(), ErrorCodePB::NONE_ERROR);

    std::this_thread::sleep_for(50ms);
    const auto delayed_begin = server.beginRemoteLoadForTest(
        token, deadline_unix_ms, std::chrono::steady_clock::now() + 100ms);
    EXPECT_EQ(delayed_begin.code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_NE(delayed_begin.message().find("sealed"), std::string::npos);
}

}  // namespace
}  // namespace rtp_llm
