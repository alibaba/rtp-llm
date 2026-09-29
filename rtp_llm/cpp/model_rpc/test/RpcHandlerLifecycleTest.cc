#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"

// Prefill reads through ClientReaderWriterInterface. Explicitly instantiate
// the stream so HIP also emits Read's secondary-base virtual thunk.
template class grpc::ClientReaderWriter<GenerateRequestPB, GenerateOutputsPB>;

namespace rtp_llm::test {
namespace {

constexpr int64_t     kRequestId      = 3101;
constexpr int64_t     kBatchId        = 71;
constexpr const char* kUnexpectedExit = "GenerateContext destroyed before RPC handling completed";
constexpr const char* kFailureReason  = "injected output failure";

enum class OutputAction {
    COMPLETE,
    FAIL,
    THROW,
    WAIT
};
enum class Handler {
    LOCAL,
    PREFILL
};

// Only model output is controlled. Stream state transitions, cache ownership,
// handler scope guards and context destructors remain production implementations.
class HandlerTestStream: public NormalGenerateStream {
public:
    HandlerTestStream(const std::shared_ptr<GenerateInput>& input,
                      const EngineInitParams&               params,
                      const ResourceContext&                resource,
                      OutputAction                          action,
                      std::promise<void>&                   polling):
        NormalGenerateStream(input, params.model_config_, params.runtime_config, resource, nullptr),
        action_(action),
        polling_(polling) {}

    ErrorResult<GenerateOutputs> nextOutput(int64_t wait_timeout_ms = 0) override {
        std::call_once(polling_once_, [this] { polling_.set_value(); });
        switch (action_) {
            case OutputAction::COMPLETE:
                // Output is drained while the engine still owns a RUNNING stream.
                return ErrorResult<GenerateOutputs>(ErrorCode::FINISHED, "output drained");
            case OutputAction::FAIL:
                reportError(ErrorCode::PRIORITY_PREEMPTED, kFailureReason);
                return ErrorResult<GenerateOutputs>(statusInfo());
            case OutputAction::THROW:
                throw std::runtime_error(kFailureReason);
            case OutputAction::WAIT:
                return NormalGenerateStream::nextOutput(wait_timeout_ms);
        }
        throw std::logic_error("unknown output action");
    }

private:
    OutputAction        action_;
    std::promise<void>& polling_;
    std::once_flag      polling_once_;
};

class SteppedEngine: public EngineBase {
public:
    static EngineInitParams parameters(bool pipeline = false) {
        EngineInitParams params;
        params.model_config_.max_seq_len                                  = 32;
        params.model_config_.vocab_size                                   = 128;
        params.runtime_config.max_generate_batch_size                     = 8;
        params.runtime_config.fifo_scheduler_config.max_batch_tokens_size = 32;
        if (pipeline) {
            params.model_config_.num_layers = 3;
            params.parallelism_config.pp_size = 3;
            params.parallelism_config.tp_size = 1;
            params.parallelism_config.pp_stage_layer_counts = {1, 1, 1};
        }
        return params;
    }

    static CacheConfig defaultCacheConfig() {
        return makeSimpleMhaCacheConfig(1, 8, 8, DataType::TYPE_FP16, 1, 4);
    }

    explicit SteppedEngine(OutputAction action,
                           bool pipeline = false,
                           const CacheConfig& cache_config = defaultCacheConfig()):
        EngineBase(parameters(pipeline)), action_(action), pipeline_(pipeline) {
        resource_context_.cache_manager = std::make_shared<KVCacheManager>(cache_config);
        if (!resource_context_.cache_manager->init()) {
            throw std::runtime_error("test cache initialization failed");
        }
        const auto params = parameters(pipeline_);
        scheduler_        = std::make_unique<FIFOScheduler>(params.runtime_config,
                                                     params.model_config_,
                                                     params.pd_sep_config,
                                                     params.parallelism_config,
                                                     params.model_specific_config,
                                                     resource_context_.cache_manager);
    }

    ~SteppedEngine() override {
        (void)stop();
    }

    GenerateStreamPtr makeStream(const std::shared_ptr<GenerateInput>& input) override {
        stream = std::make_shared<HandlerTestStream>(input, parameters(pipeline_), resource_context_, action_, polling);
        stream->setPipelineParallel(pipeline_);
        return stream;
    }
    GenerateStreamPtr enqueue(const std::shared_ptr<GenerateInput>& input) override {
        auto result = makeStream(input);
        enqueue(result);
        return result;
    }
    void enqueue(GenerateStreamPtr& value) override {
        const auto status = scheduler_->enqueue(value);
        if (!status.ok()) {
            throw std::runtime_error(status.ToString());
        }
        auto scheduled = scheduler_->schedule();
        if (!scheduled.ok() || scheduled.value().streams.size() != 1 || value->getStatus() != StreamState::RUNNING) {
            throw std::runtime_error("test stream was not admitted to RUNNING: " + value->statusInfo().ToString());
        }
        // No background engine loop: the test explicitly advances the next tick.
    }
    absl::Status stop() override {
        return scheduler_->stop();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused in lifecycle test");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }
    FIFOScheduler& scheduler() {
        return static_cast<FIFOScheduler&>(*scheduler_);
    }

    std::promise<void> polling;
    GenerateStreamPtr  stream;

private:
    OutputAction action_;
    bool pipeline_;
};

// Isolate remote computation/cache transport, not the handler under test.
class LifecyclePeer: public RpcService::Service {
public:
    explicit LifecyclePeer(bool send_load_response = true): send_load_response_(send_load_response) {}

    grpc::Status RemoteGenerate(grpc::ServerContext*, ServerStream* rpc) override {
        GenerateRequestPB request;
        GenerateOutputsPB response;
        if (!rpc->Read(&request) || !rpc->Write(response)) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "allocate exchange failed");
        }
        if (rpc->Read(&request) && send_load_response_ && !rpc->Write(response)) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "load exchange failed");
        }
        while (rpc->Read(&request)) {}
        return grpc::Status::OK;
    }
    grpc::Status
    RemoteLoad(grpc::ServerContext*, const BroadcastLoadRequestPB* request, BroadcastLoadResponsePB* response) override {
        {
            std::lock_guard<std::mutex> lock(load_requests_mutex_);
            load_requests_.push_back(*request);
        }
        response->set_done_time_us(currentTimeUs());
        return grpc::Status::OK;
    }

    std::vector<BroadcastLoadRequestPB> loadRequests() {
        std::lock_guard<std::mutex> lock(load_requests_mutex_);
        return load_requests_;
    }

private:
    // An early local failure never consumes LOAD's response. Keep this peer
    // quiescent so Finish tests handler teardown, not unread-message draining.
    bool send_load_response_;
    std::mutex load_requests_mutex_;
    std::vector<BroadcastLoadRequestPB> load_requests_;
};

class ScopedRpcServer {
public:
    explicit ScopedRpcServer(RpcService::Service* service) {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(service);
        server_ = builder.BuildAndStart();
        if (!server_ || port == 0) {
            throw std::runtime_error("test gRPC server failed to start");
        }
    }
    ~ScopedRpcServer() {
        server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));
        server_->Wait();
    }
    std::string address() const {
        return "127.0.0.1:" + std::to_string(port);
    }
    int port = 0;

private:
    std::unique_ptr<grpc::Server> server_;
};

class LifecycleDecodeService: public RpcService::Service {
public:
    grpc::Status RemoteGenerate(grpc::ServerContext* context, ServerStream* rpc) override {
        auto status = handler.RemoteGenerate(context, rpc);
        returned.set_value(status);  // The real context destructor has already run.
        return status;
    }
    DecodeRpcServer            handler;
    std::promise<grpc::Status> returned;
};

GenerateInputPB lifecycleRequest() {
    GenerateInputPB request;
    request.set_request_id(kRequestId);
    request.mutable_group_id()->set_value(kBatchId);
    request.add_token_ids(1);
    request.add_token_ids(2);
    auto* config = request.mutable_generate_config();
    config->set_max_new_tokens(8);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_timeout_ms(10000);
    return request;
}

class RpcHandlerLifecycleTest: public DeviceTestBase {
protected:
    void expectFinishedOnce(const std::shared_ptr<RpcServerRuntimeMeta>& meta, ErrorCode error) {
        const auto info = meta->getEngineScheduleInfo(-1);
        EXPECT_TRUE(info.running_task_info_list.empty());
        ASSERT_EQ(info.finished_task_info_list.size(), 1);
        const auto& task = info.finished_task_info_list.front();
        EXPECT_EQ(task.request_id, kRequestId);
        EXPECT_EQ(task.batch_id, kBatchId);
        EXPECT_EQ(task.error_code, static_cast<int64_t>(error));
        if (error == ErrorCode::PRIORITY_PREEMPTED || error == ErrorCode::EXECUTION_EXCEPTION) {
            EXPECT_NE(task.error_message.find(kFailureReason), std::string::npos);
        }
        // Decode's ordinary completion must not invent the Prefill control
        // owner's CANCELING/CANCELED overlay or emit a duplicate completion.
        EXPECT_EQ(task.priority_preemption_progress, PriorityPreemptionProgress::NONE);
    }

    void reap(const std::shared_ptr<SteppedEngine>& engine, size_t free_before) {
        auto next = engine->scheduler().schedule();
        ASSERT_TRUE(next.ok());
        EXPECT_TRUE(next.value().streams.empty());
        EXPECT_EQ(engine->stream->getStatus(), StreamState::FINISHED);
        EXPECT_TRUE(engine->scheduler().empty());
        EXPECT_EQ(engine->scheduler().onflightStreams(), 0);
        EXPECT_TRUE(engine->scheduler().runningTaskList().empty());
        EXPECT_EQ(engine->getCacheManager()->freeBlocksNum(), free_before);
    }

    void runSynchronousHandler(Handler kind, OutputAction action) {
        TestLogCapture   capture("rpc_handler_lifecycle");
        auto             engine      = std::make_shared<SteppedEngine>(action);
        auto             meta        = std::make_shared<RpcServerRuntimeMeta>();
        const auto       free_before = engine->getCacheManager()->freeBlocksNum();
        LifecyclePeer    peer(action == OutputAction::COMPLETE);
        ScopedRpcServer  peer_server(&peer);
        LocalRpcServer   local;
        PrefillRpcServer prefill;
        auto&            handler          = kind == Handler::LOCAL ? local : static_cast<LocalRpcServer&>(prefill);
        handler.engine_                   = engine;
        handler.meta_                     = meta;
        handler.propose_maga_init_params_ = nullptr;
        handler.maga_init_params_.pd_sep_config.prefill_retry_times                 = 0;
        handler.maga_init_params_.pd_sep_config.prefill_stop_stream_wait_timeout_ms = 1;
        // No local cache-store transport is needed for this handler test.
        prefill.resource().workers.clear();
        auto request = lifecycleRequest();
        request.mutable_generate_config()->set_can_use_pd_separation(kind == Handler::PREFILL);
        auto* role = request.mutable_generate_config()->add_role_addrs();
        role->set_role(RoleAddrPB::DECODE);
        role->set_ip("127.0.0.1");
        role->set_grpc_port(peer_server.port);
        auto polling = engine->polling.get_future();
        auto call =
            std::async(std::launch::async, [&] { return handler.GenerateStreamCall(nullptr, &request, nullptr); });
        // No output payload is emitted by this fixture, so a writer is unnecessary.
        const bool reached_output = polling.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        const bool returned       = call.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        if (!returned) {
            // A regression must fail the test rather than leave a future blocked.
            (void)engine->stop();
        }
        EXPECT_TRUE(reached_output);
        EXPECT_TRUE(returned) << "handler waited for scheduler completion";
        if (kind == Handler::LOCAL && action == OutputAction::THROW) {
            EXPECT_THROW(call.get(), std::runtime_error);
        } else {
            auto status = call.get();
            EXPECT_EQ(status.ok(), action == OutputAction::COMPLETE);
            if (action != OutputAction::COMPLETE) {
                EXPECT_NE(status.error_message().find(kFailureReason), std::string::npos);
            }
        }
        ASSERT_TRUE(reached_output);
        ASSERT_TRUE(returned);
        ASSERT_NE(engine->stream, nullptr);
        EXPECT_EQ(engine->stream->getStatus(), StreamState::RUNNING);
        const auto expected_error = action == OutputAction::COMPLETE ? ErrorCode::NONE_ERROR :
                                    action == OutputAction::FAIL     ? ErrorCode::PRIORITY_PREEMPTED :
                                    kind == Handler::LOCAL           ? ErrorCode::CANCELLED :
                                                                       ErrorCode::EXECUTION_EXCEPTION;
        EXPECT_EQ(engine->stream->statusInfo().code(), expected_error);
        expectFinishedOnce(meta, expected_error);
        const auto diagnostic = capture.content().find(kUnexpectedExit);
        if (kind == Handler::LOCAL && action == OutputAction::THROW) {
            EXPECT_NE(diagnostic, std::string::npos);
            EXPECT_EQ(capture.content().find(kUnexpectedExit, diagnostic + std::string(kUnexpectedExit).size()),
                      std::string::npos);
        } else {
            EXPECT_EQ(diagnostic, std::string::npos);
        }
        if (action == OutputAction::COMPLETE) {
            engine->stream->reportEvent(StreamEvents::GenerateDone);
        }
        reap(engine, free_before);
    }

    void runDecodeHandler(OutputAction action,
                          bool pipeline = false,
                          const CacheConfig& cache_config = SteppedEngine::defaultCacheConfig()) {
        TestLogCapture         capture("decode_handler_lifecycle");
        auto                   engine      = std::make_shared<SteppedEngine>(action, pipeline, cache_config);
        auto                   meta        = std::make_shared<RpcServerRuntimeMeta>();
        const auto             free_before = engine->getCacheManager()->freeBlocksNum();
        LifecyclePeer          peer;
        ScopedRpcServer        peer_server(&peer);
        LifecycleDecodeService service;
        service.handler.engine_                   = engine;
        service.handler.meta_                     = meta;
        service.handler.propose_maga_init_params_ = nullptr;
        // Exercise real async RemoteLoad fan-out, with only remote KV transport faked.
        service.handler.resource().workers      = {"rank0", "rank1"};
        service.handler.resource().grpc_workers = {peer_server.address(), peer_server.address()};
        if (pipeline) {
            service.handler.maga_init_params_ = SteppedEngine::parameters(true);
            service.handler.resource().workers = {"rank0", "rank1", "rank2"};
            service.handler.resource().grpc_workers = {
                peer_server.address(), peer_server.address(), peer_server.address()};
        }
        ScopedRpcServer server(&service);
        auto stub = RpcService::NewStub(grpc::CreateChannel(server.address(), grpc::InsecureChannelCredentials()));
        grpc::ClientContext client;
        client.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(15));
        auto              polling  = engine->polling.get_future();
        auto              returned = service.returned.get_future();
        auto              rpc      = stub->RemoteGenerate(&client);
        GenerateRequestPB request;
        request.set_stage(RemoteStage::ALLOCATE);
        request.set_request_id(kRequestId);
        request.set_client_id("lifecycle");
        *request.mutable_input() = lifecycleRequest();
        request.add_peer_addrs("prefill:1:2");
        if (pipeline) {
            /** Unequal PP stage counts must not trigger the legacy total-worker ratio check. */
            request.add_peer_addrs("prefill-last:1:2");
            auto* first = request.add_stage_peer_groups();
            first->set_layer_begin(0);
            first->set_layer_count(1);
            first->add_peer_addrs("prefill:1:2");
            auto* last = request.add_stage_peer_groups();
            last->set_layer_begin(1);
            last->set_layer_count(2);
            last->add_peer_addrs("prefill-last:1:2");
            last->set_is_last_stage(true);
        }
        ASSERT_TRUE(rpc->Write(request));
        GenerateOutputsPB response;
        ASSERT_TRUE(rpc->Read(&response));
        ASSERT_TRUE(response.supports_prefill_completion());
        request.set_stage(RemoteStage::LOAD);
        ASSERT_TRUE(rpc->Write(request));
        ASSERT_TRUE(rpc->Read(&response));
        ASSERT_EQ(response.error_info().error_code(), ErrorCodePB::NONE_ERROR);
        if (pipeline) {
            const auto received = peer.loadRequests();
            ASSERT_EQ(received.size(), 3);
            const auto& blocks = engine->stream->kvCachePtr()->cacheResource(0).groupBlockIds();
            const auto tags = blocks.orderedTags();
            ASSERT_FALSE(tags.empty());
            ASSERT_EQ(tags.size(), cache_config.groupNums());
            if (tags.size() == 2) {
                /** Distinct allocated rows detect truncation and accidental reuse of another tag's blocks. */
                ASSERT_NE(blocks.blocks(tags[0]), blocks.blocks(tags[1]));
            }
            for (const auto& load : received) {
                EXPECT_EQ(load.peer_addrs_size(), 0);
                ASSERT_EQ(load.stage_peer_groups_size(), request.stage_peer_groups_size());
                for (int i = 0; i < request.stage_peer_groups_size(); ++i) {
                    EXPECT_EQ(load.stage_peer_groups(i).SerializeAsString(),
                              request.stage_peer_groups(i).SerializeAsString());
                }
                ASSERT_EQ(load.tagged_group_block_ids_size(), tags.size());
                for (size_t i = 0; i < tags.size(); ++i) {
                    const auto& row = load.tagged_group_block_ids(static_cast<int>(i));
                    EXPECT_EQ(row.tag(), tags[i]);
                    EXPECT_EQ(BlockIndicesType(row.block_ids().begin(), row.block_ids().end()), blocks.blocks(tags[i]));
                }
            }
        }
        request.set_stage(RemoteStage::GENERATE);
        request.set_first_generate_token_id(3);
        ASSERT_TRUE(rpc->Write(request));
        ASSERT_EQ(polling.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        if (action == OutputAction::WAIT) {
            EXPECT_EQ(engine->stream->getStatus(), StreamState::RUNNING);
            EXPECT_LT(engine->getCacheManager()->freeBlocksNum(), free_before);
            client.TryCancel();
        }
        const bool returned_before_tick = returned.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        if (!returned_before_tick) {
            client.TryCancel();
            (void)engine->stop();
        }
        EXPECT_TRUE(returned_before_tick) << "Decode handler waited for scheduler completion";
        ASSERT_EQ(returned.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        const auto server_status = returned.get();
        rpc->WritesDone();
        while (rpc->Read(&response)) {}
        const auto client_status = rpc->Finish();
        ASSERT_TRUE(returned_before_tick);
        EXPECT_EQ(server_status.ok(), action == OutputAction::COMPLETE);
        EXPECT_EQ(client_status.ok(), action == OutputAction::COMPLETE);
        if (action == OutputAction::WAIT) {
            EXPECT_EQ(server_status.error_code(), grpc::StatusCode::CANCELLED);
            EXPECT_EQ(client_status.error_code(), grpc::StatusCode::CANCELLED);
        }
        const auto expected_error = action == OutputAction::COMPLETE ? ErrorCode::NONE_ERROR :
                                    action == OutputAction::FAIL     ? ErrorCode::PRIORITY_PREEMPTED :
                                    action == OutputAction::THROW    ? ErrorCode::EXECUTION_EXCEPTION :
                                                                       ErrorCode::CANCELLED;
        EXPECT_EQ(engine->stream->statusInfo().code(), expected_error);
        EXPECT_EQ(engine->stream->getStatus(), StreamState::RUNNING);
        EXPECT_EQ(engine->scheduler().runningStreamsSize(), 1);
        EXPECT_EQ(engine->scheduler().onflightStreams(), 1);
        const auto running_tasks = engine->scheduler().runningTaskList();
        ASSERT_EQ(running_tasks.size(), 1);
        EXPECT_EQ(running_tasks.front().request_id, kRequestId);
        EXPECT_EQ(running_tasks.front().batch_id, kBatchId);
        EXPECT_LT(engine->getCacheManager()->freeBlocksNum(), free_before);
        expectFinishedOnce(meta, expected_error);
        EXPECT_EQ(capture.content().find(kUnexpectedExit), std::string::npos);
        if (action == OutputAction::COMPLETE) {
            engine->stream->reportEvent(StreamEvents::GenerateDone);
        }
        reap(engine, free_before);
        expectFinishedOnce(meta, expected_error);
    }
};

TEST_F(RpcHandlerLifecycleTest, LocalSuccessReturnsBeforeSchedulerWithoutCanceling) {
    runSynchronousHandler(Handler::LOCAL, OutputAction::COMPLETE);
}
TEST_F(RpcHandlerLifecycleTest, LocalFailurePreservesPriorityPreemption) {
    runSynchronousHandler(Handler::LOCAL, OutputAction::FAIL);
}
TEST_F(RpcHandlerLifecycleTest, LocalExceptionUnwindsAndDiagnosesOnce) {
    runSynchronousHandler(Handler::LOCAL, OutputAction::THROW);
}
TEST_F(RpcHandlerLifecycleTest, PrefillSuccessReturnsBeforeSchedulerWithoutCanceling) {
    runSynchronousHandler(Handler::PREFILL, OutputAction::COMPLETE);
}
TEST_F(RpcHandlerLifecycleTest, PrefillEarlyFailurePreservesPriorityPreemption) {
    runSynchronousHandler(Handler::PREFILL, OutputAction::FAIL);
}
TEST_F(RpcHandlerLifecycleTest, PrefillCaughtExceptionCompletesHandlingWithoutUnexpectedExit) {
    runSynchronousHandler(Handler::PREFILL, OutputAction::THROW);
}
TEST_F(RpcHandlerLifecycleTest, DecodeSuccessReturnsBeforeSchedulerWithoutCanceling) {
    runDecodeHandler(OutputAction::COMPLETE);
}
TEST_F(RpcHandlerLifecycleTest, PipelineDecodeFansOutStageMetadataAndCompleteTagRows) {
    runDecodeHandler(OutputAction::COMPLETE, /*pipeline=*/true);

    auto config = SteppedEngine::defaultCacheConfig();
    config.seq_size_per_block = 1;
    config.model_has_multiple_cache_groups = true;
    /** Full and tail-only linear allocation produce distinct rows for the two-token prompt. */
    config.fromGroupedSpecs(
        {makeMhaSpec("full", 1, DataType::TYPE_FP16, 1, 4), makeLinearSpec("linear", 1, DataType::TYPE_FP16, 1, 4)},
        {{0}, {0}}, {CacheGroupType::FULL, CacheGroupType::LINEAR}, {"full", "linear"});
    config.finalizeBlockNums(16, RuntimeConfig{});
    runDecodeHandler(OutputAction::COMPLETE, /*pipeline=*/true, config);
}
TEST_F(RpcHandlerLifecycleTest, DecodePriorityPreemptionReclaimsKvOnNextSchedulerTick) {
    runDecodeHandler(OutputAction::FAIL);
}
TEST_F(RpcHandlerLifecycleTest, DecodeCaughtExceptionPreservesCauseAndReclaimsKv) {
    runDecodeHandler(OutputAction::THROW);
}
TEST_F(RpcHandlerLifecycleTest, DecodeRunningCancellationReturnsBeforeSchedulerReclaimsKv) {
    runDecodeHandler(OutputAction::WAIT);
}

}  // namespace
}  // namespace rtp_llm::test
