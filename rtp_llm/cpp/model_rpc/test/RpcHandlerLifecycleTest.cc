#include <chrono>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
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
        auto scheduled = scheduler_->schedule();
        if (!scheduled.ok() || scheduled.value().size() != 1 || value->getStatus() != StreamState::RUNNING) {
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
        if (action == OutputAction::THROW) {
            EXPECT_THROW(call.get(), std::runtime_error);
        } else {
            auto status = call.get();
            EXPECT_EQ(status.ok(), action == OutputAction::COMPLETE);
            if (action != OutputAction::COMPLETE) {
                EXPECT_NE(status.error_message().find(kFailureReason), std::string::npos);
            }
        }
        EXPECT_EQ(response_call.get().ok(), action == OutputAction::COMPLETE);
        ASSERT_TRUE(reached_output);
        ASSERT_TRUE(returned);
        ASSERT_NE(engine->stream, nullptr);
        EXPECT_EQ(engine->stream->getStatus(), StreamState::RUNNING);
        const auto expected_error = action == OutputAction::COMPLETE ? ErrorCode::NONE_ERROR :
                                    action == OutputAction::FAIL     ? ErrorCode::PRIORITY_PREEMPTED :
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
TEST_F(RpcHandlerLifecycleTest, PrefillSuccessReturnsBeforeSchedulerWithoutCanceling) {
    runSynchronousHandler(Handler::PREFILL, OutputAction::COMPLETE);
}
TEST_F(RpcHandlerLifecycleTest, PrefillEarlyFailurePreservesPriorityPreemption) {
    runSynchronousHandler(Handler::PREFILL, OutputAction::FAIL);
}
TEST_F(RpcHandlerLifecycleTest, PrefillExceptionUnwindsAndDiagnosesOnce) {
    runSynchronousHandler(Handler::PREFILL, OutputAction::THROW);
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
