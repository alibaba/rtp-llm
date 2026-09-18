#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "gtest/gtest.h"
#include <mutex>
#include <string>

#include "rtp_llm/cpp/model_rpc/PrefillRpcServerNew2.h"

namespace rtp_llm {

TEST(PrefillRpcServerNew2Test, GetPeerInfoReturnsOnlySelectedPeerLayoutWithoutWorkerAddresses) {
    PrefillRpcServerNew2 server;
    auto& pc = server.maga_init_params_.parallelism_config;
    pc.tp_size = 4;
    pc.dp_size = 3;
    pc.dp_rank = 2;
    // No worker address list is needed to report the selected endpoint's layout.
    for (bool sharded : {false, true}) {
        pc.prefill_cp_config.kv_cache_sharded = sharded;
        grpc::ServerContext   context;
        GetPeerInfoRequestPB  request;
        GetPeerInfoResponsePB response;
        ASSERT_TRUE(server.GetPeerInfo(&context, &request, &response).ok());
        EXPECT_EQ(response.tp_size(), 4);
        EXPECT_EQ(response.cp_size(), sharded ? 4 : 1);
    }
}

TEST(PrefillRpcServerNew2Test, GetPeerInfoRejectsInvalidTpSize) {
    PrefillRpcServerNew2 server;
    server.maga_init_params_.parallelism_config.tp_size = 0;
    grpc::ServerContext   context;
    GetPeerInfoRequestPB  request;
    GetPeerInfoResponsePB response;
    const auto status = server.GetPeerInfo(&context, &request, &response);
    ASSERT_FALSE(status.ok());
    EXPECT_NE(status.error_message().find("invalid tp_size=0"), std::string::npos);
}

TEST(PrefillRpcServerNew2Test, OnflightScopeTracksStepAndCleansOnReturn) {
    PrefillRpcServerNew2 server;

    {
        PrefillRpcServerNew2::OnflightScope scope(&server, 9001);
        {
            std::lock_guard<std::mutex> lock(server.onflight_trackers_mutex_);
            ASSERT_EQ(server.onflight_trackers_.size(), 1);
            ASSERT_NE(server.onflight_trackers_.find(9001), server.onflight_trackers_.end());
            EXPECT_EQ(server.onflight_trackers_.at(9001)->step.load(),
                      static_cast<int>(PrefillRpcServerNew2::GenerateStreamStep::kEntry));
        }

        scope.markStep(PrefillRpcServerNew2::GenerateStreamStep::kAfterEngineEnqueue);
        {
            std::lock_guard<std::mutex> lock(server.onflight_trackers_mutex_);
            EXPECT_EQ(server.onflight_trackers_.at(9001)->step.load(),
                      static_cast<int>(PrefillRpcServerNew2::GenerateStreamStep::kAfterEngineEnqueue));
        }
    }

    std::lock_guard<std::mutex> lock(server.onflight_trackers_mutex_);
    EXPECT_TRUE(server.onflight_trackers_.empty());
}

TEST(PrefillRpcServerNew2Test, StartLoadRejectsMissingEngine) {
    PrefillRpcServerNew2            server;
    grpc::ServerContext             context;
    P2PConnectorStartLoadRequestPB  request;
    P2PConnectorStartLoadResponsePB response;

    auto status = server.StartLoad(&context, &request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(status.error_message(), "engine is null");
}

TEST(PrefillRpcServerNew2Test, GenerateStreamCallRejectsMissingRequestTimeout) {
    PrefillRpcServerNew2 server;
    grpc::ServerContext context;
    GenerateInputPB request;
    request.set_request_id(43);
    request.add_token_ids(1);
    auto* config = request.mutable_generate_config();
    config->set_max_new_tokens(8);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);
    config->set_unique_key("missing_request_deadline");
    auto status = server.GenerateStreamCall(&context, &request, nullptr);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
}

TEST(PrefillRpcServerNew2Test, GenerateStreamCallRejectsPdRequestWithoutUniqueKey) {
    PrefillRpcServerNew2 server;
    grpc::ServerContext  context;
    GenerateInputPB      request;
    request.set_request_id(42);
    request.add_token_ids(1);
    auto* config = request.mutable_generate_config();
    config->set_timeout_ms(5000);
    config->set_max_new_tokens(8);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);

    auto status = server.GenerateStreamCall(&context, &request, nullptr);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.error_message(), "decode_entrance handoff requires non-empty unique_key");
}

}  // namespace rtp_llm
