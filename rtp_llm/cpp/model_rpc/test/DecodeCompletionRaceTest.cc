#include <chrono>
#include <future>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm {
namespace {

class CompletionPeer: public RpcService::Service {
public:
    grpc::Status
    GetPeerInfo(grpc::ServerContext*, const GetPeerInfoRequestPB*, GetPeerInfoResponsePB* response) override {
        response->set_tp_size(2);
        response->set_cp_size(2);
        return grpc::Status::OK;
    }
};

class CompletionEngine: public EngineBase {
public:
    CompletionEngine(bool commit_before_poll, bool fail):
        EngineBase(EngineInitParams{}), commit_before_poll_(commit_before_poll), fail_(fail) {}

    GenerateStreamPtr makeStream(const std::shared_ptr<GenerateInput>& input) override {
        ModelConfig model;
        model.max_seq_len = 4096;
        model.vocab_size  = 32000;
        return std::make_shared<NormalGenerateStream>(input, model, RuntimeConfig{}, ResourceContext{}, nullptr);
    }

    GenerateStreamPtr enqueue(const std::shared_ptr<GenerateInput>& input) override {
        auto stream = makeStream(input);
        enqueue(stream);
        return stream;
    }

    void enqueue(GenerateStreamPtr& stream) override {
        stream->generate_status_->status.store(StreamState::RUNNING);
        stream->reportEvent(StreamEvents::GenerateDone);
        if (fail_) {
            stream->reportError(ErrorCode::MALLOC_FAILED, "completion test failure");
        }
        if (commit_before_poll_) {
            stream->moveToNext();
        }
        enqueued.set_value(stream);
    }

    absl::Status stop() override {
        return absl::OkStatus();
    }

    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("completion test does not execute a model");
    }

    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }

    std::promise<GenerateStreamPtr> enqueued;

private:
    bool commit_before_poll_;
    bool fail_;
};

void checkCompletion(bool commit_before_poll, bool fail) {
    CompletionPeer      peer;
    grpc::ServerBuilder builder;
    int                 port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&peer);
    auto peer_server = builder.BuildAndStart();
    ASSERT_TRUE(peer_server);

    DecodeRpcServer server;
    server.meta_                  = std::make_shared<RpcServerRuntimeMeta>();
    server.cancel_registry_       = std::make_unique<PDCancelRegistry>(server.meta_);
    server.prefill_server_caller_ = std::make_shared<PrefillServerCaller>();
    auto engine                   = std::make_shared<CompletionEngine>(commit_before_poll, fail);
    server.engine_                = engine;
    auto enqueued                 = engine->enqueued.get_future();

    GenerateInputPB request;
    request.set_request_id(201635);
    request.set_enqueued_by_master(true);
    request.add_token_ids(1);
    auto* config = request.mutable_generate_config();
    config->set_max_new_tokens(35);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);
    config->set_timeout_ms(5000);
    auto* address = config->add_role_addrs();
    address->set_role(RoleAddrPB::PREFILL);
    address->set_ip("127.0.0.1");
    address->set_grpc_port(port);

    grpc::ServerContext context;
    auto rpc = std::async(std::launch::async, [&] { return server.GenerateStreamCall(&context, &request, nullptr); });
    ASSERT_EQ(enqueued.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    auto stream = enqueued.get();
    auto handle = server.cancel_registry_->find(request.request_id());
    ASSERT_TRUE(handle);

    bool       cleanup_decided = false;
    const auto deadline        = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lock(server.cancel_registry_->mutex_);
            cleanup_decided = handle->canceled.load() || handle->downstream_done;
        }
        if (cleanup_decided) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(cleanup_decided);
    EXPECT_EQ(stream->getStatus(), commit_before_poll ? StreamState::FINISHED : StreamState::RUNNING);
    EXPECT_EQ(handle->canceled.load(), fail);
    EXPECT_EQ(stream->hasError(), fail) << stream->statusInfo().ToString();
    EXPECT_EQ(stream->moveToNext(), StreamState::FINISHED);

    const auto status = rpc.get();
    EXPECT_EQ(status.ok(), !fail) << status.error_message();
    EXPECT_EQ(stream->statusInfo().code(), fail ? ErrorCode::MALLOC_FAILED : ErrorCode::NONE_ERROR);
    EXPECT_TRUE(handle->local_done.load());
    peer_server->Shutdown();
}

TEST(DecodeCompletionRaceTest, PendingSuccessfulCompletionIsNotCanceled) {
    checkCompletion(false, false);
}

TEST(DecodeCompletionRaceTest, CommittedSuccessfulCompletionIsNotCanceled) {
    checkCompletion(true, false);
}

TEST(DecodeCompletionRaceTest, PendingFailureStillCancelsAndPreservesCause) {
    checkCompletion(false, true);
}

}  // namespace
}  // namespace rtp_llm
