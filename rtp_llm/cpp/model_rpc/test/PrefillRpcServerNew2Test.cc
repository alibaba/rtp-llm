#include "gtest/gtest.h"
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "rtp_llm/cpp/model_rpc/PrefillRpcServerNew2.h"

namespace rtp_llm {

namespace {

class TestablePrefillRpcServerNew2: public PrefillRpcServerNew2 {
public:
    bool cancelled{false};
    bool has_engine{true};
    bool has_cache_manager{true};
    bool handle_read_called{false};
    bool handle_read_observed_cancelled{false};
    bool local_generate_called{false};
    bool enqueue_called{false};
    bool poll_called{false};
    bool mtp_eagle{false};
    std::string                    handled_unique_key;
    std::shared_ptr<GenerateInput> captured_input;
    grpc::Status                   local_generate_status{grpc::Status::OK};
    grpc::Status                   poll_status{grpc::Status::OK};

protected:
    bool isContextCancelled(grpc::ServerContext*) const override {
        return cancelled;
    }

    bool hasStartLoadEngine() const override {
        return has_engine;
    }

    bool hasStartLoadCacheManager() const override {
        return has_cache_manager;
    }

    void handleStartLoadRead(const P2PConnectorStartLoadRequestPB& request,
                             P2PConnectorStartLoadResponsePB&,
                             std::function<bool()> is_cancelled) override {
        handle_read_called              = true;
        handled_unique_key              = request.unique_key();
        handle_read_observed_cancelled  = is_cancelled();
    }

    grpc::Status callLocalGenerateStream(grpc::ServerContext*,
                                         const GenerateInputPB*,
                                         grpc::ServerWriter<GenerateOutputsPB>*) override {
        local_generate_called = true;
        return local_generate_status;
    }

    bool engineIsMtpEagle() const override {
        return mtp_eagle;
    }

    std::shared_ptr<GenerateStream> enqueuePrefillStream(const std::shared_ptr<GenerateInput>& input) override {
        enqueue_called = true;
        captured_input = input;
        return nullptr;
    }

    grpc::Status pollPrefillStreamOutput(grpc::ServerContext*,
                                         const std::string&,
                                         WriterInterface*,
                                         std::shared_ptr<GenerateStream>&) override {
        poll_called = true;
        return poll_status;
    }
};

bool hasSuffix(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size()
           && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

GenerateInputPB makeGenerateRequest(bool pd_separation, const std::string& unique_key = "") {
    GenerateInputPB request;
    request.set_request_id(42);
    request.add_token_ids(1);
    auto* config = request.mutable_generate_config();
    config->set_max_new_tokens(pd_separation ? 8 : 1);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);
    if (!unique_key.empty()) {
        config->set_unique_key(unique_key);
    }
    return request;
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

TEST(PrefillRpcServerNew2Test, StartLoadReturnsCancelledBeforeEngineLookup) {
    TestablePrefillRpcServerNew2 server;
    server.cancelled = true;
    server.has_engine = false;

    grpc::ServerContext             context;
    P2PConnectorStartLoadRequestPB  request;
    P2PConnectorStartLoadResponsePB response;
    request.set_unique_key("cancelled-key");

    auto status = server.StartLoad(&context, &request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_FALSE(server.handle_read_called);
}

TEST(PrefillRpcServerNew2Test, StartLoadRejectsMissingEngineOrCacheManager) {
    grpc::ServerContext             context;
    P2PConnectorStartLoadRequestPB  request;
    P2PConnectorStartLoadResponsePB response;
    request.set_unique_key("load-key");

    TestablePrefillRpcServerNew2 missing_engine;
    missing_engine.has_engine = false;
    auto engine_status = missing_engine.StartLoad(&context, &request, &response);
    EXPECT_EQ(engine_status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(engine_status.error_message(), "engine is null");
    EXPECT_FALSE(missing_engine.handle_read_called);

    TestablePrefillRpcServerNew2 missing_cache;
    missing_cache.has_engine = true;
    missing_cache.has_cache_manager = false;
    auto cache_status = missing_cache.StartLoad(&context, &request, &response);
    EXPECT_EQ(cache_status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(cache_status.error_message(), "cache manager is null");
    EXPECT_FALSE(missing_cache.handle_read_called);
}

TEST(PrefillRpcServerNew2Test, StartLoadDelegatesHandleReadOnHealthyRequest) {
    TestablePrefillRpcServerNew2 server;
    server.has_engine = true;
    server.has_cache_manager = true;

    grpc::ServerContext             context;
    P2PConnectorStartLoadRequestPB  request;
    P2PConnectorStartLoadResponsePB response;
    request.set_unique_key("handoff-key");

    auto status = server.StartLoad(&context, &request, &response);

    ASSERT_TRUE(status.ok());
    EXPECT_TRUE(server.handle_read_called);
    EXPECT_EQ(server.handled_unique_key, "handoff-key");
    EXPECT_FALSE(server.handle_read_observed_cancelled);
}

TEST(PrefillRpcServerNew2Test, GenerateStreamCallFallsBackToLocalForNonPdRequest) {
    TestablePrefillRpcServerNew2 server;
    grpc::ServerContext          context;
    auto                         request = makeGenerateRequest(/*pd_separation=*/false);

    auto status = server.GenerateStreamCall(&context, &request, nullptr);

    ASSERT_TRUE(status.ok());
    EXPECT_TRUE(server.local_generate_called);
    EXPECT_FALSE(server.enqueue_called);
    EXPECT_FALSE(server.poll_called);
}

TEST(PrefillRpcServerNew2Test, GenerateStreamCallRejectsPdRequestWithoutUniqueKey) {
    TestablePrefillRpcServerNew2 server;
    grpc::ServerContext          context;
    auto                         request = makeGenerateRequest(/*pd_separation=*/true);

    auto status = server.GenerateStreamCall(&context, &request, nullptr);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.error_message(), "decode_entrance handoff requires non-empty unique_key");
    EXPECT_FALSE(server.local_generate_called);
    EXPECT_FALSE(server.enqueue_called);
    EXPECT_FALSE(server.poll_called);
}

TEST(PrefillRpcServerNew2Test, GenerateStreamCallEnqueuesAndPollsPdRequestWithUniqueKey) {
    TestablePrefillRpcServerNew2 server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    grpc::ServerContext context;
    auto                request = makeGenerateRequest(/*pd_separation=*/true, "prefill-handoff-key");

    auto status = server.GenerateStreamCall(&context, &request, nullptr);

    ASSERT_TRUE(status.ok());
    EXPECT_FALSE(server.local_generate_called);
    EXPECT_TRUE(server.enqueue_called);
    EXPECT_TRUE(server.poll_called);
    ASSERT_TRUE(server.captured_input);
    ASSERT_TRUE(server.captured_input->generate_config);
    EXPECT_EQ(server.captured_input->generate_config->unique_key, "prefill-handoff-key");
    EXPECT_TRUE(server.captured_input->generate_config->pd_separation);
    EXPECT_TRUE(server.captured_input->generate_config->force_disable_sp_run);
    std::lock_guard<std::mutex> lock(server.onflight_trackers_mutex_);
    EXPECT_TRUE(server.onflight_trackers_.empty());
}

}  // namespace rtp_llm
