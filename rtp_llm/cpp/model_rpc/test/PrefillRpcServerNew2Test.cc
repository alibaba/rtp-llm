#include "gtest/gtest.h"
#include <mutex>
#include <string>

#include "rtp_llm/cpp/model_rpc/PrefillRpcServerNew2.h"

namespace rtp_llm {

namespace {

bool hasSuffix(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size()
           && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

TEST(PrefillRpcServerNew2Test, ParseP2PWorkerGrpcAddrSupportsIpv4HostAndBracketIpv6) {
    std::string grpc_addr;

    ASSERT_TRUE(PrefillRpcServerNew2::parseP2PWorkerGrpcAddr("127.0.0.1:8000:9000", &grpc_addr));
    EXPECT_EQ(grpc_addr, "127.0.0.1:9000");

    ASSERT_TRUE(PrefillRpcServerNew2::parseP2PWorkerGrpcAddr("prefill-0.service:8001:9001", &grpc_addr));
    EXPECT_EQ(grpc_addr, "prefill-0.service:9001");

    ASSERT_TRUE(PrefillRpcServerNew2::parseP2PWorkerGrpcAddr("[::1]:8002:9002", &grpc_addr));
    EXPECT_EQ(grpc_addr, "[::1]:9002");

    ASSERT_TRUE(PrefillRpcServerNew2::parseP2PWorkerGrpcAddr("fe80::1:8003:9003", &grpc_addr));
    EXPECT_EQ(grpc_addr, "[fe80::1]:9003");
}

TEST(PrefillRpcServerNew2Test, ParseP2PWorkerGrpcAddrRejectsMalformedAddressOrPort) {
    std::string grpc_addr;

    EXPECT_FALSE(PrefillRpcServerNew2::parseP2PWorkerGrpcAddr("", &grpc_addr));
    EXPECT_FALSE(PrefillRpcServerNew2::parseP2PWorkerGrpcAddr("127.0.0.1:8000", &grpc_addr));
    EXPECT_FALSE(PrefillRpcServerNew2::parseP2PWorkerGrpcAddr("fe80::1", &grpc_addr));
    EXPECT_FALSE(PrefillRpcServerNew2::parseP2PWorkerGrpcAddr("[::1]8000:9000", &grpc_addr));
    EXPECT_FALSE(PrefillRpcServerNew2::parseP2PWorkerGrpcAddr("127.0.0.1:0:9000", &grpc_addr));
    EXPECT_FALSE(PrefillRpcServerNew2::parseP2PWorkerGrpcAddr("127.0.0.1:8000:65536", &grpc_addr));
    EXPECT_FALSE(PrefillRpcServerNew2::parseP2PWorkerGrpcAddr("127.0.0.1:8000:not-a-port", &grpc_addr));
}

TEST(PrefillRpcServerNew2Test, GetPeerInfoUsesPrecomputedDpGrpcAddrs) {
    PrefillRpcServerNew2 server;
    server.maga_init_params_.parallelism_config.tp_size = 2;
    server.maga_init_params_.parallelism_config.dp_size = 2;
    server.dp_grpc_addrs_ = {"10.0.0.1:9000", "[::1]:9002"};

    grpc::ServerContext  context;
    GetPeerInfoRequestPB request;
    GetPeerInfoResponsePB response;

    auto status = server.GetPeerInfo(&context, &request, &response);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(response.tp_size(), 2);
    EXPECT_EQ(response.dp_size(), 2);
    ASSERT_EQ(response.dp_grpc_addrs_size(), 2);
    EXPECT_EQ(response.dp_grpc_addrs(0), "10.0.0.1:9000");
    EXPECT_EQ(response.dp_grpc_addrs(1), "[::1]:9002");
}

TEST(PrefillRpcServerNew2Test, GetPeerInfoFallbackSkipsInvalidComputedPorts) {
    PrefillRpcServerNew2 server;
    server.maga_init_params_.parallelism_config.tp_size = 4;
    server.maga_init_params_.parallelism_config.dp_size = 3;
    server.maga_init_params_.parallelism_config.tp_rank = 0;
    server.maga_init_params_.parallelism_config.dp_rank = 0;
    server.maga_init_params_.pd_sep_config.worker_port_offset = 40000;
    server.local_rpc_port_ = 1000;
    server.dp_grpc_addrs_.clear();

    grpc::ServerContext  context;
    GetPeerInfoRequestPB request;
    GetPeerInfoResponsePB response;

    auto status = server.GetPeerInfo(&context, &request, &response);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(response.dp_grpc_addrs_size(), 1);
    EXPECT_TRUE(hasSuffix(response.dp_grpc_addrs(0), ":1000"));
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

TEST(PrefillRpcServerNew2Test, GenerateStreamCallRejectsPdRequestWithoutUniqueKey) {
    PrefillRpcServerNew2 server;
    grpc::ServerContext  context;
    GenerateInputPB      request;
    request.set_request_id(42);
    request.add_token_ids(1);
    auto* config = request.mutable_generate_config();
    config->set_max_new_tokens(8);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);

    auto status = server.GenerateStreamCall(&context, &request, nullptr);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.error_message(), "decode_entrance handoff requires non-empty unique_key");
}

}  // namespace rtp_llm
