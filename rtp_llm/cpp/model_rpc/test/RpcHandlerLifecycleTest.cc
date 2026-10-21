#include <chrono>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorPrefill.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorResourceStore.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"

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
    PREFILL,
    PREFILL_LOCAL
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
    static EngineInitParams parameters() {
        EngineInitParams params;
        params.model_config_.max_seq_len                                  = 32;
        params.model_config_.vocab_size                                   = 128;
        params.runtime_config.max_generate_batch_size                     = 8;
        params.runtime_config.fifo_scheduler_config.max_batch_tokens_size = 32;
        return params;
    }

    explicit SteppedEngine(OutputAction action): EngineBase(parameters()), action_(action) {
        resource_context_.cache_manager =
            std::make_shared<KVCacheManager>(DeviceTestBase::makeMhaCacheConfig(1, 8, 1, 4, 8, DataType::TYPE_FP16));
        if (!resource_context_.cache_manager->init()) {
            throw std::runtime_error("test cache initialization failed");
        }
        const auto params = parameters();
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
        stream = std::make_shared<HandlerTestStream>(input, parameters(), resource_context_, action_, polling);
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
        // P2P registration completes synchronously, but admission still visits
        // WAITING -> LOADING_CACHE -> WAITING -> RUNNING on separate scheduler ticks.
        // Advance those ticks without adding a background engine or waiting for model output.
        for (int tick = 0; tick < 3; ++tick) {
            auto scheduled = scheduler_->schedule();
            if (!scheduled.ok()) {
                throw std::runtime_error("test scheduler failed: " + scheduled.status().ToString());
            }
            if (value->getStatus() == StreamState::RUNNING) {
                if (scheduled.value().size() != 1) {
                    throw std::runtime_error("test scheduler did not return its RUNNING stream");
                }
                return;
            }
            if (value->getStatus() == StreamState::FINISHED || value->hasError()) {
                break;
            }
        }
        throw std::runtime_error("test stream was not admitted to RUNNING: " + value->statusInfo().ToString());
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
};

class BatchAdmissionEngine: public SteppedEngine {
public:
    BatchAdmissionEngine(): SteppedEngine(OutputAction::WAIT) {}
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>& inputs) override {
        admitted_inputs = inputs;
        if (throw_on_enqueue) throw std::runtime_error("injected batch enqueue exception");
        if (mismatch_results) return {{true}, {}};
        std::vector<bool> accepted;
        std::vector<GenerateStreamPtr> streams;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto value = makeStream(inputs[i]);
            const bool ok = i == 0;
            if (!ok) value->reportError(ErrorCode::PRIORITY_PREEMPTED, "injected admission rejection");
            accepted.push_back(ok);
            streams.push_back(std::move(value));
        }
        return {accepted, streams};
    }
    bool throw_on_enqueue = false;
    bool mismatch_results = false;
    std::vector<std::shared_ptr<GenerateInput>> admitted_inputs;
};

class LifecycleService: public RpcService::Service {
public:
    explicit LifecycleService(LocalRpcServer& handler): handler_(handler) {}

    grpc::Status GenerateStreamCall(grpc::ServerContext* context,
                                    const GenerateInputPB* request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        try {
            auto status = handler_.GenerateStreamCall(context, request, writer);
            returned.set_value(status);
            return status;
        } catch (...) {
            returned.set_exception(std::current_exception());
            return grpc::Status(grpc::StatusCode::INTERNAL, kFailureReason);
        }
    }

    std::promise<grpc::Status> returned;

private:
    LocalRpcServer& handler_;
};

class ScopedRpcServer {
public:
    explicit ScopedRpcServer(RpcService::Service* service) {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(service);
        server_ = builder.BuildAndStart();
        if (!server_ || port_ == 0) {
            throw std::runtime_error("test gRPC server failed to start");
        }
    }
    ~ScopedRpcServer() {
        server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));
        server_->Wait();
    }
    std::string address() const {
        return "127.0.0.1:" + std::to_string(port_);
    }

private:
    int port_ = 0;
    std::unique_ptr<grpc::Server> server_;
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
        // Ordinary completion must not invent a cancellation overlay or emit a duplicate completion.
        EXPECT_EQ(task.priority_preemption_progress, PriorityPreemptionProgress::NONE);
    }

    void reap(const std::shared_ptr<SteppedEngine>& engine, size_t free_before) {
        auto next = engine->scheduler().schedule();
        ASSERT_TRUE(next.ok());
        EXPECT_TRUE(next.value().empty());
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
        LocalRpcServer   local;
        PrefillRpcServer prefill;
        auto&            handler          = kind == Handler::LOCAL ? local : static_cast<LocalRpcServer&>(prefill);
        handler.engine_                   = engine;
        handler.meta_                     = meta;
        handler.propose_maga_init_params_ = nullptr;
        prefill.cancel_registry_ = std::make_unique<PDCancelRegistry>(meta);
        if (kind == Handler::PREFILL) {
            P2PConnectorConfig config;
            config.role_type = RoleType::PREFILL;
            auto connector = std::make_shared<P2PConnector>(config, nullptr, nullptr);
            connector->prefill_ = std::make_unique<P2PConnectorPrefill>(config, nullptr, nullptr);
            connector->prefill_->stream_store_ = std::make_shared<P2PConnectorResourceStore>(nullptr);
            engine->getCacheManager()->p2p_connector_ = connector;
        }
        auto request = lifecycleRequest();
        request.mutable_generate_config()->set_can_use_pd_separation(kind == Handler::PREFILL);
        request.mutable_generate_config()->set_unique_key("lifecycle-p2p");
        LifecycleService service(handler);
        ScopedRpcServer  server(&service);
        auto stub = RpcService::NewStub(grpc::CreateChannel(server.address(), grpc::InsecureChannelCredentials()));
        grpc::ClientContext client;
        client.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        auto polling = engine->polling.get_future();
        auto call    = service.returned.get_future();
        auto response_call = std::async(std::launch::async, [&] {
            auto reader = stub->GenerateStreamCall(&client, request);
            GenerateOutputsPB response;
            while (reader->Read(&response)) {}
            return reader->Finish();
        });
        const bool reached_output = polling.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        const bool returned       = call.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        if (!returned) {
            // A regression must fail the test rather than leave a future blocked.
            (void)engine->stop();
        }
        EXPECT_TRUE(reached_output);
        EXPECT_TRUE(returned) << "handler waited for scheduler completion";
        auto status = call.get();
        EXPECT_EQ(status.ok(), action == OutputAction::COMPLETE) << status.error_message();
        if (action != OutputAction::COMPLETE) {
            EXPECT_NE(status.error_message().find(kFailureReason), std::string::npos);
        }
        EXPECT_EQ(response_call.get().ok(), action == OutputAction::COMPLETE);
        ASSERT_TRUE(reached_output);
        ASSERT_TRUE(returned);
        ASSERT_NE(engine->stream, nullptr);
        EXPECT_EQ(engine->stream->getStatus(), StreamState::RUNNING);
        const auto expected_error = action == OutputAction::COMPLETE ? ErrorCode::NONE_ERROR :
                                    action == OutputAction::FAIL     ? ErrorCode::PRIORITY_PREEMPTED :
                                    kind == Handler::PREFILL         ? ErrorCode::EXECUTION_EXCEPTION :
                                                                     ErrorCode::CANCELLED;
        EXPECT_EQ(engine->stream->statusInfo().code(), expected_error);
        expectFinishedOnce(meta, expected_error);
        const auto diagnostic = capture.content().find(kUnexpectedExit);
        if (action == OutputAction::THROW) {
            EXPECT_NE(diagnostic, std::string::npos);
            EXPECT_EQ(capture.content().find(kUnexpectedExit, diagnostic + std::string(kUnexpectedExit).size()),
                      std::string::npos);
        } else {
            EXPECT_EQ(diagnostic, std::string::npos);
        }
        if (action == OutputAction::COMPLETE) {
            engine->stream->reportEvent(StreamEvents::GenerateDone);
        }
        if (kind == Handler::PREFILL) {
            // This fixture has no Decode/StartLoad consumer. Settle its P2P
            // registration explicitly, as downstream transfer completion would;
            // scheduler completion alone must not drop the store's KV hold.
            EXPECT_LT(engine->getCacheManager()->freeBlocksNum(), free_before);
            auto store = engine->getCacheManager()->p2p_connector_->streamStore();
            ASSERT_NE(store, nullptr);
            store->markTerminal(engine->stream->uniqueKey(), engine->stream->deadlineMs());
        }
        reap(engine, free_before);
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
TEST_F(RpcHandlerLifecycleTest, PrefillNonPdFallbackCompletesThroughLocalHandler) {
    runSynchronousHandler(Handler::PREFILL_LOCAL, OutputAction::COMPLETE);
}
TEST_F(RpcHandlerLifecycleTest, PrefillNonPdFallbackPreservesExecutionFailure) {
    runSynchronousHandler(Handler::PREFILL_LOCAL, OutputAction::FAIL);
}
TEST_F(RpcHandlerLifecycleTest, PrefillNonPdFallbackCleansUpAfterException) {
    runSynchronousHandler(Handler::PREFILL_LOCAL, OutputAction::THROW);
}
TEST_F(RpcHandlerLifecycleTest, PrefillSuccessReturnsBeforeSchedulerWithoutCanceling) {
    runSynchronousHandler(Handler::PREFILL, OutputAction::COMPLETE);
}
TEST_F(RpcHandlerLifecycleTest, PrefillEarlyFailurePreservesPriorityPreemption) {
    runSynchronousHandler(Handler::PREFILL, OutputAction::FAIL);
}
TEST_F(RpcHandlerLifecycleTest, PrefillExceptionUnwindsAndDiagnosesOnce) {
    runSynchronousHandler(Handler::PREFILL, OutputAction::THROW);
}

TEST_F(RpcHandlerLifecycleTest, ProductionBatchCombinesValidationAndEngineAdmissionResults) {
    auto engine = std::make_shared<BatchAdmissionEngine>();
    auto meta = std::make_shared<RpcServerRuntimeMeta>();
    PrefillRpcServer server;
    server.engine_ = engine;
    server.meta_ = meta;
    server.cancel_registry_ = std::make_unique<PDCancelRegistry>(meta);
    P2PConnectorConfig config;
    config.role_type = RoleType::PREFILL;
    auto connector = std::make_shared<P2PConnector>(config, nullptr, nullptr);
    connector->prefill_ = std::make_unique<P2PConnectorPrefill>(config, nullptr, nullptr);
    connector->prefill_->stream_store_ = std::make_shared<P2PConnectorResourceStore>(nullptr);
    engine->getCacheManager()->p2p_connector_ = connector;

    EnqueueBatchRequestPB request;
    request.set_batch_id(kBatchId);
    auto* slot = request.add_dp_slots();
    slot->set_dp_rank(0);
    for (int i = 0; i < 3; ++i) {
        auto* item = slot->add_requests()->mutable_input();
        *item = lifecycleRequest();
        item->set_request_id(kRequestId + i);
        if (i == 0) item->mutable_generate_config()->set_max_new_tokens(-1);
    }
    grpc::ServerContext context;
    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.EnqueueBatch(&context, &request, &response).ok());
    ASSERT_EQ(engine->admitted_inputs.size(), 2u);
    for (const auto& input : engine->admitted_inputs) EXPECT_EQ(input->group_size, 2);
    ASSERT_EQ(response.successes_size(), 1);
    EXPECT_EQ(response.successes(0).request_id(), kRequestId + 1);
    ASSERT_EQ(response.errors_size(), 2);
    EXPECT_EQ(response.errors(0).request_id(), kRequestId);
    EXPECT_EQ(response.errors(0).error_info().error_code(), static_cast<int64_t>(ErrorCode::INVALID_PARAMS));
    EXPECT_EQ(response.errors(1).request_id(), kRequestId + 2);
    EXPECT_EQ(response.errors(1).error_info().error_code(), static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
    ASSERT_EQ(server.batch_entries_.size(), 1u);
    EXPECT_EQ(server.onflightRequestNum(), 1u);
    auto& entry = server.batch_entries_.begin()->second;
    ASSERT_NE(entry.context, nullptr);
    entry.context->markRpcHandlingCompleted();
    server.batch_entries_.clear();
    EXPECT_EQ(server.onflightRequestNum(), 0u);
}

TEST_F(RpcHandlerLifecycleTest, ProductionBatchEnqueueFailuresReleaseAllReservations) {
    for (const bool throw_error : {true, false}) {
        auto engine = std::make_shared<BatchAdmissionEngine>();
        engine->throw_on_enqueue = throw_error;
        engine->mismatch_results = !throw_error;
        auto meta = std::make_shared<RpcServerRuntimeMeta>();
        PrefillRpcServer server;
        server.engine_ = engine;
        server.meta_ = meta;
        server.cancel_registry_ = std::make_unique<PDCancelRegistry>(meta);
        P2PConnectorConfig config;
        config.role_type = RoleType::PREFILL;
        auto connector = std::make_shared<P2PConnector>(config, nullptr, nullptr);
        connector->prefill_ = std::make_unique<P2PConnectorPrefill>(config, nullptr, nullptr);
        connector->prefill_->stream_store_ = std::make_shared<P2PConnectorResourceStore>(nullptr);
        engine->getCacheManager()->p2p_connector_ = connector;
        EnqueueBatchRequestPB request;
        request.set_batch_id(kBatchId);
        auto* slot = request.add_dp_slots();
        slot->set_dp_rank(0);
        for (int i = 0; i < 2; ++i) {
            auto* item = slot->add_requests()->mutable_input();
            *item = lifecycleRequest();
            item->set_request_id(kRequestId + i);
        }
        grpc::ServerContext context;
        EnqueueBatchResponsePB response;
        EXPECT_EQ(server.EnqueueBatch(&context, &request, &response).error_code(), grpc::StatusCode::INTERNAL);
        EXPECT_EQ(response.successes_size(), 0);
        EXPECT_TRUE(server.batch_entries_.empty());
        EXPECT_EQ(server.onflightRequestNum(), 0u);
        ASSERT_EQ(engine->admitted_inputs.size(), 2u);
        for (int i = 0; i < 2; ++i) {
            auto handle = server.cancel_registry_->find(kRequestId + i);
            ASSERT_NE(handle, nullptr);
            EXPECT_TRUE(handle->local_done.load());
        }
    }
}

TEST_F(RpcHandlerLifecycleTest, ProductionBatchDeadlineAndPreprocessRejectionsNeverReachEngine) {
    for (const bool terminal_deadline : {true, false}) {
        auto engine = std::make_shared<BatchAdmissionEngine>();
        auto meta = std::make_shared<RpcServerRuntimeMeta>();
        PrefillRpcServer server;
        server.engine_ = engine;
        server.meta_ = meta;
        server.cancel_registry_ = std::make_unique<PDCancelRegistry>(meta);
        P2PConnectorConfig config;
        config.role_type = RoleType::PREFILL;
        auto connector = std::make_shared<P2PConnector>(config, nullptr, nullptr);
        connector->prefill_ = std::make_unique<P2PConnectorPrefill>(config, nullptr, nullptr);
        auto store = std::make_shared<P2PConnectorResourceStore>(nullptr);
        connector->prefill_->stream_store_ = store;
        engine->getCacheManager()->p2p_connector_ = connector;
        EnqueueBatchRequestPB request;
        request.set_batch_id(kBatchId);
        auto* slot = request.add_dp_slots();
        slot->set_dp_rank(0);
        auto* item = slot->add_requests()->mutable_input();
        *item = lifecycleRequest();
        if (terminal_deadline) {
            const auto key = masterEnqueuedHandoffUniqueKey(kRequestId);
            const auto deadline = store->requestDeadline(key, 10000);
            store->markCancelled(key, deadline);
        } else {
            item->add_multimodal_inputs()->set_multimodal_url("missing-processor-image");
        }
        grpc::ServerContext context;
        EnqueueBatchResponsePB response;
        ASSERT_TRUE(server.EnqueueBatch(&context, &request, &response).ok());
        EXPECT_EQ(response.successes_size(), 0);
        ASSERT_EQ(response.errors_size(), 1);
        EXPECT_EQ(response.errors(0).request_id(), kRequestId);
        EXPECT_EQ(response.errors(0).error_info().error_code(), static_cast<int64_t>(
            terminal_deadline ? ErrorCode::GENERATE_TIMEOUT : ErrorCode::MM_NOT_SUPPORTED_ERROR));
        EXPECT_TRUE(engine->admitted_inputs.empty());
        EXPECT_TRUE(server.batch_entries_.empty());
        EXPECT_EQ(server.onflightRequestNum(), 0u);
        if (!terminal_deadline) {
            auto handle = server.cancel_registry_->find(kRequestId);
            ASSERT_NE(handle, nullptr);
            EXPECT_TRUE(handle->local_done.load());
            EXPECT_TRUE(handle->downstream_done);
        }
    }
}

TEST_F(RpcHandlerLifecycleTest, PrefillBatchTerminalCleanupMarksHandlingCompleted) {
    for (auto action : {OutputAction::COMPLETE, OutputAction::FAIL}) {
        TestLogCapture capture("prefill_batch_lifecycle");
        auto engine = std::make_shared<SteppedEngine>(action);
        auto meta = std::make_shared<RpcServerRuntimeMeta>();
        PrefillRpcServer prefill;
        prefill.engine_ = engine;
        prefill.meta_ = meta;
        prefill.cancel_registry_ = std::make_unique<PDCancelRegistry>(meta);
        const auto free_before = engine->getCacheManager()->freeBlocksNum();
        auto request = lifecycleRequest();
        auto stream = engine->enqueue(QueryConverter::transQuery(&request));
        auto context = std::make_unique<GenerateContext>(kRequestId, 10000, nullptr, prefill.metrics_reporter_, meta);
        context->setStream(stream);
        auto& entry = prefill.batch_entries_["lifecycle-batch"];
        entry.reserved = true;
        entry.context = std::move(context);
        if (action == OutputAction::COMPLETE) {
            stream->reportEvent(StreamEvents::GenerateDone);
            reap(engine, free_before);
        } else {
            stream->reportError(ErrorCode::PRIORITY_PREEMPTED, kFailureReason);
        }
        prefill.batchContextCleanupTick();
        EXPECT_TRUE(prefill.batch_entries_.empty());
        EXPECT_EQ(capture.content().find(kUnexpectedExit), std::string::npos);
        if (action == OutputAction::FAIL) {
            reap(engine, free_before);
        }
    }
}

}  // namespace
}  // namespace rtp_llm::test
