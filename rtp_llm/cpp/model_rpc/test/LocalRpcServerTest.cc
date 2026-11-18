#include <gtest/gtest.h>

#include <memory>
#include <new>
#include <array>
#include <cstring>
#include <chrono>
#include <functional>
#include <future>
#include <grpc++/grpc++.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServiceImpl.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm {
namespace {

class WorkerStatusTestServer: public LocalRpcServer {
public:
    WorkerStatusTestServer(RoleType role, int concurrency_limit) {
        maga_init_params_.pd_sep_config.role_type              = role;
        maga_init_params_.concurrency_config.concurrency_limit = concurrency_limit;
    }

    // Requests can be in KV preparation or waiting for remote decode without
    // a local runtime task. No engine or GPU is needed for these status tests.
    EngineScheduleInfo getEngineScheduleInfo(int64_t) override {
        return EngineScheduleInfo{};
    }

    std::unique_ptr<AtomicGuard> startRequest() {
        return std::make_unique<AtomicGuard>(onflight_requests_);
    }
};

class BatchTestServer: public LocalRpcServer {
public:
    using LocalRpcServer::pollStreamOutput;
    using LocalRpcServer::prepareInput;
    using LocalRpcServer::serializeErrorMsg;
    void setEngine(const std::shared_ptr<EngineBase>& engine) {
        engine_ = engine;
        meta_   = std::make_shared<RpcServerRuntimeMeta>();
    }
};

class BatchTestStream: public GenerateStream {
public:
    explicit BatchTestStream(const std::shared_ptr<GenerateInput>& input):
        GenerateStream(input, modelConfig(), RuntimeConfig{}, ResourceContext{}, nullptr) {}

    ErrorResult<GenerateOutputs> nextOutput() override {
        ++reads;
        if (throw_on_read) {
            throw std::runtime_error("injected output failure");
        }
        if (hasError()) {
            return statusInfo();
        }
        if (!output_ready) {
            throw std::runtime_error("legacy output read entered without readiness");
        }
        if (reads > 1) {
            return {ErrorCode::FINISHED, "finished"};
        }
        GenerateOutputs output{};
        output.request_id = streamId();
        output.generate_outputs.emplace_back();
        output.generate_outputs.back().finished = true;
        return std::move(output);
    }

    void updateOutput(const StreamUpdateInfo&) override {}
    bool hasOutput() override {
        if (on_poll) {
            auto notify = std::exchange(on_poll, {});
            notify();
        }
        return output_ready;
    }

    int                   reads         = 0;
    bool                  throw_on_read = false;
    bool                  output_ready  = true;
    std::function<void()> on_poll;

private:
    static ModelConfig modelConfig() {
        ModelConfig config;
        config.max_seq_len = 8;
        return config;
    }
};

class BatchTestEngine: public EngineBase {
public:
    BatchTestEngine(): EngineBase(EngineInitParams{}) {}

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        throw std::runtime_error("batch must not call single enqueue");
    }
    void enqueue(std::shared_ptr<GenerateStream>&) override {
        throw std::runtime_error("batch must not call single enqueue");
    }
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>& inputs) override {
        ++batch_calls;
        if (before_enqueue) {
            before_enqueue();
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto stream = std::make_shared<BatchTestStream>(inputs[i]);
            if (i == 1) {
                stream->reportError(item_error_code, item_error_message);
            }
            stream->throw_on_read = throw_on_first && i == 0;
            stream->output_ready  = !idle_streams;
            if (i == 0) {
                stream->on_poll = std::move(on_first_poll);
            }
            streams.push_back(std::move(stream));
        }
        auto accepted = std::vector<bool>(inputs.size(), true);
        if (accepted.size() > 1) {
            accepted[1] = accept_failed_item;
        }
        if (wrong_cardinality) {
            accepted.pop_back();
        }
        return {std::move(accepted), std::vector<GenerateStreamPtr>(streams.begin(), streams.end())};
    }
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused in batch RPC test");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }

    ErrorCode                                     item_error_code = ErrorCode::HIDDEN_STATE_CAPTURE_BACKPRESSURE;
    std::string                                   item_error_message = "capture quota occupied";
    bool                                          accept_failed_item = false;
    int                                           batch_calls       = 0;
    bool                                          wrong_cardinality = false;
    bool                                          throw_on_first    = false;
    bool                                          idle_streams      = false;
    std::function<void()>                         before_enqueue;
    std::function<void()>                         on_first_poll;
    std::vector<std::shared_ptr<BatchTestStream>> streams;
};

class RpcBoundaryTestEngine: public BatchTestEngine {
public:
    RpcBoundaryTestEngine() {
        RuntimeConfig config;
        config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 2;
        scheduler_ = std::make_unique<BatchDecodeScheduler>(config, nullptr, nullptr);
    }

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>& input) override {
        ModelConfig model;
        model.max_seq_len = 16;
        model.vocab_size  = 128;
        stream = std::make_shared<NormalGenerateStream>(input, model, RuntimeConfig{}, ResourceContext{}, nullptr);
        prepare_stream(stream);
        return stream;
    }

    std::function<void(const std::shared_ptr<NormalGenerateStream>&)> prepare_stream;
    std::shared_ptr<NormalGenerateStream>                            stream;
};

class RpcBoundaryTestService: public LocalRpcServiceImpl {
public:
    explicit RpcBoundaryTestService(const std::shared_ptr<LocalRpcServer>& server) {
        local_server_ = server;
    }
};

class LocalRpcServerBoundaryTest: public ::testing::Test {
protected:
    void SetUp() override {
        local  = std::make_shared<BatchTestServer>();
        engine = std::make_shared<RpcBoundaryTestEngine>();
        local->setEngine(engine);
        service = std::make_unique<RpcBoundaryTestService>(local);
        grpc::ServerBuilder builder;
        builder.RegisterService(service.get());
        server = builder.BuildAndStart();
        ASSERT_NE(server, nullptr);
        stub = RpcService::NewStub(server->InProcessChannel(grpc::ChannelArguments{}));
    }

    void TearDown() override {
        if (server) {
            server->Shutdown();
            server->Wait();
        }
    }

    grpc::Status updateScheduler(const std::string& json) {
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        UpdateSchedulerInfoRequestPB request;
        request.set_scheduler_info(json);
        EmptyPB response;
        return stub->UpdateSchedulerInfo(&context, request, &response);
    }

    GenerateInputPB streamRequest() {
        GenerateInputPB request;
        request.set_request_id(101);
        request.add_token_ids(1);
        auto* config = request.mutable_generate_config();
        config->set_max_new_tokens(3);
        config->set_num_beams(1);
        config->set_num_return_sequences(1);
        config->set_ignore_eos(true);
        config->set_frontend_metric_streaming(true);
        return request;
    }

    std::shared_ptr<BatchTestServer>        local;
    std::shared_ptr<RpcBoundaryTestEngine>  engine;
    std::unique_ptr<RpcBoundaryTestService> service;
    std::unique_ptr<grpc::Server>           server;
    std::unique_ptr<RpcService::Stub>       stub;
};

TEST_F(LocalRpcServerBoundaryTest, SchedulerRpcRejectsInvalidValuesAndJsonWithoutChangingState) {
    auto& scheduler = static_cast<BatchDecodeScheduler&>(engine->getScheduler());
    for (const auto* mode : {"decode", "prefill"}) {
        ASSERT_TRUE(updateScheduler(std::string("{\"batch_size\":3,\"mode\":\"") + mode + "\"}").ok());
        const auto        old_mode = scheduler.scheduler_type_;
        const std::string other_mode = old_mode == BatchDecodeScheduler::kBatchDecode ? "prefill" : "decode";
        for (const auto* batch_size : {"0", "-1", "4294967296", "\"bad\"", "null", "[]", "{}"}) {
            SCOPED_TRACE(batch_size);
            const auto status = updateScheduler(std::string("{\"batch_size\":") + batch_size
                                                + ",\"mode\":\"" + other_mode + "\"}");
            EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
            ErrorDetailsPB details;
            ASSERT_TRUE(details.ParseFromString(status.error_details()));
            EXPECT_EQ(details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
            EXPECT_FALSE(details.error_message().empty());
            EXPECT_EQ(details.error_message(), status.error_message());
            EXPECT_EQ(scheduler.batch_size_, 3u);
            EXPECT_EQ(scheduler.scheduler_type_, old_mode);
        }
        for (const auto* json : {"{", "{}", "[]", "{\"batch_size\":4,\"mode\":42}"}) {
            SCOPED_TRACE(json);
            const auto status = updateScheduler(json);
            EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
            ErrorDetailsPB details;
            ASSERT_TRUE(details.ParseFromString(status.error_details()));
            EXPECT_EQ(details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
            EXPECT_FALSE(details.error_message().empty());
            EXPECT_EQ(details.error_message(), status.error_message());
            EXPECT_EQ(scheduler.batch_size_, 3u);
            EXPECT_EQ(scheduler.scheduler_type_, old_mode);
        }
    }
    for (const auto* batch_size : {"1", "4294967295"}) {
        ASSERT_TRUE(updateScheduler(std::string("{\"batch_size\":") + batch_size + ",\"mode\":\"decode\"}").ok());
        EXPECT_EQ(scheduler.batch_size_, std::stoull(batch_size));
        EXPECT_EQ(scheduler.scheduler_type_, BatchDecodeScheduler::kBatchDecode);
    }
}

TEST_F(LocalRpcServerBoundaryTest, SchedulerRpcDoesNotTranslateNonParameterExceptions) {
    class ThrowingScheduler: public BatchDecodeScheduler {
    public:
        using BatchDecodeScheduler::BatchDecodeScheduler;
        void updateSchedulerInfo(const std::string&) override {
            fail();
        }
        std::function<void()> fail;
    };
    RuntimeConfig config;
    config.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 2;
    auto scheduler = std::make_unique<ThrowingScheduler>(config, nullptr, nullptr);
    auto* throwing = scheduler.get();
    engine->scheduler_ = std::move(scheduler);
    UpdateSchedulerInfoRequestPB request;
    request.set_scheduler_info("{\"batch_size\":3}");
    EmptyPB response;
    // Exceptions intentionally stay outside gRPC's worker threads in this test.
    throwing->fail = [] { throw std::bad_alloc(); };
    EXPECT_THROW(local->UpdateSchedulerInfo(nullptr, &request, &response), std::bad_alloc);
    throwing->fail = [] { throw std::runtime_error("scheduler internal failure"); };
    EXPECT_THROW(local->UpdateSchedulerInfo(nullptr, &request, &response), std::runtime_error);
    throwing->fail = [] { throw autil::legacy::RuntimeError("scheduler internal failure"); };
    EXPECT_THROW(local->UpdateSchedulerInfo(nullptr, &request, &response), autil::legacy::RuntimeError);
    throwing->fail = [] { throw 42; };
    EXPECT_THROW(local->UpdateSchedulerInfo(nullptr, &request, &response), int);
}

TEST_F(LocalRpcServerBoundaryTest, StreamRpcRejectsMissingGenerateConfigBeforeEnqueue) {
    auto request = streamRequest();
    request.clear_generate_config();
    const auto before = request.SerializeAsString();
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    auto reader = stub->GenerateStreamCall(&context, request);
    GenerateOutputsPB output;
    EXPECT_FALSE(reader->Read(&output));
    const auto status = reader->Finish();
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
    EXPECT_EQ(details.error_message(), "generate_config must be present");
    EXPECT_EQ(details.error_message(), status.error_message());
    EXPECT_EQ(engine->stream, nullptr);
    EXPECT_TRUE(engine->streams.empty());
    EXPECT_EQ(engine->batch_calls, 0);
    EXPECT_EQ(local->onflightRequestNum(), 0u);
    EXPECT_EQ(request.SerializeAsString(), before);
}

TEST_F(LocalRpcServerBoundaryTest, ZeroOutputStreamRpcPreservesTerminalErrorCodeAndReason) {
    for (const auto code : {ErrorCode::INVALID_PARAMS,
                            ErrorCode::HIDDEN_STATE_CAPTURE_BACKPRESSURE,
                            ErrorCode::GENERATE_TIMEOUT,
                            ErrorCode::LONG_PROMPT_ERROR,
                            ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN,
                            ErrorCode::MALLOC_FAILED,
                            ErrorCode::DECODE_MALLOC_FAILED,
                            ErrorCode::LOAD_KV_CACHE_FAILED,
                            ErrorCode::WAIT_TO_RUN_TIMEOUT,
                            static_cast<ErrorCode>(99999),
                            ErrorCode::EXECUTION_EXCEPTION,
                            ErrorCode::CANCELLED}) {
        SCOPED_TRACE(static_cast<int>(code));
        const std::string reason = "rejected before first output: " + ErrorCodeToString(code);
        engine->prepare_stream = [code, reason](const auto& stream) { stream->reportError(code, reason); };
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        auto reader = stub->GenerateStreamCall(&context, streamRequest());
        GenerateOutputsPB output;
        EXPECT_FALSE(reader->Read(&output));
        const auto status = reader->Finish();
        EXPECT_EQ(status.error_code(), transErrorCodeToGrpc(code));
        EXPECT_EQ(status.error_message(), reason);
        ErrorDetailsPB details;
        ASSERT_TRUE(details.ParseFromString(status.error_details()));
        EXPECT_EQ(details.error_code(), static_cast<int>(code));
        EXPECT_EQ(details.error_message(), reason);
        ASSERT_NE(engine->stream, nullptr);
        EXPECT_FALSE(engine->stream->hasOutput());
        EXPECT_EQ(engine->stream->statusInfo().code(), code);
        EXPECT_EQ(engine->stream->statusInfo().ToString(), reason);
        EXPECT_EQ(local->onflightRequestNum(), 0u);
    }
}

TEST_F(LocalRpcServerBoundaryTest, StreamRpcDrainsMetricBeforeTerminalErrorOrFinalBusinessOutput) {
    for (bool fail : {false, true}) {
        SCOPED_TRACE(fail);
        engine->prepare_stream = [fail](const auto& stream) {
            for (int i = 0; i < (fail ? 2 : 3); ++i) {
                stream->update({torch::tensor({{7 + i}}, torch::kInt32), 1});
            }
            if (fail) {
                stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "failed after queued metric");
            } else {
                stream->generate_status_->status.store(StreamState::FINISHED);
            }
        };
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        auto reader = stub->GenerateStreamCall(&context, streamRequest());
        GenerateOutputsPB output;
        ASSERT_TRUE(reader->Read(&output));
        EXPECT_TRUE(output.frontend_metric_only());
        if (!fail) {
            ASSERT_TRUE(reader->Read(&output));
            EXPECT_FALSE(output.frontend_metric_only());
            ASSERT_EQ(output.flatten_output().finished_size(), 1);
            EXPECT_TRUE(output.flatten_output().finished(0));
            ASSERT_EQ(output.flatten_output().aux_info_size(), 1);
            EXPECT_EQ(output.flatten_output().aux_info(0).output_len(), 3);
        }
        EXPECT_FALSE(reader->Read(&output));
        const auto status = reader->Finish();
        if (fail) {
            EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
            ErrorDetailsPB details;
            ASSERT_TRUE(details.ParseFromString(status.error_details()));
            EXPECT_EQ(details.error_code(), static_cast<int>(ErrorCode::EXECUTION_EXCEPTION));
            EXPECT_EQ(details.error_message(), "failed after queued metric");
        } else {
            EXPECT_TRUE(status.ok()) << status.error_message();
        }
    }
}

TEST_F(LocalRpcServerBoundaryTest, ClosedOutputConsumerStillCancelsWithoutReplacingExistingStreamError) {
    class RejectingWriter: public LocalRpcServer::WriterInterface {
    public:
        bool Write(const GenerateOutputsPB& output, grpc::WriteOptions) override {
            ++writes;
            EXPECT_TRUE(output.frontend_metric_only());
            return false;
        }
        int writes = 0;
    };
    for (bool existing_error : {false, true}) {
        SCOPED_TRACE(existing_error);
        auto input                                       = std::make_shared<GenerateInput>();
        input->request_id                                = 101;
        input->input_ids                                 = torch::tensor({1}, torch::kInt32);
        input->generate_config                           = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens           = 3;
        input->generate_config->is_streaming             = false;
        input->generate_config->ignore_eos               = true;
        input->generate_config->frontend_metric_streaming = true;
        engine->prepare_stream = [existing_error](const auto& stream) {
            stream->update({torch::tensor({{7}}, torch::kInt32), 1});
            if (existing_error) {
                stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "original execution error");
            }
        };
        auto            stream = engine->enqueue(input);
        RejectingWriter writer;
        const auto      status = local->pollStreamOutput(nullptr, "101", &writer, stream);
        EXPECT_EQ(writer.writes, 1);
        EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
        EXPECT_EQ(stream->statusInfo().code(), existing_error ? ErrorCode::EXECUTION_EXCEPTION : ErrorCode::CANCELLED);
        EXPECT_EQ(stream->statusInfo().ToString(),
                  existing_error ? "original execution error" : "write outputs pb failed");
    }
}

class MultiStepBatchTestEngine: public BatchTestEngine {
public:
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>& inputs) override {
        ++batch_calls;
        std::vector<GenerateStreamPtr> results;
        ModelConfig                    model;
        model.max_seq_len = 16;
        model.vocab_size  = 128;
        for (size_t i = 0; i < inputs.size(); ++i) {
            EXPECT_FALSE(inputs[i]->generate_config->is_streaming);
            auto stream =
                std::make_shared<NormalGenerateStream>(inputs[i], model, RuntimeConfig{}, ResourceContext{}, nullptr);
            for (int step = 0; step < 3; ++step) {
                const int32_t token = 7 + static_cast<int32_t>(i) * 3 + step;
                stream->update({torch::tensor({{token}}, torch::kInt32), 1});
                EXPECT_EQ(stream->hasOutput(), step == 2);
            }
            EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
            // No scheduler runs in this fixture; publish its terminal state
            // only after the real stream has produced its final business output.
            stream->generate_status_->status.store(StreamState::FINISHED);
            results.push_back(std::move(stream));
        }
        return {std::vector<bool>(inputs.size(), true), std::move(results)};
    }
};

TEST(LocalRpcServerBatchTest, DirectStreamingBatchReturnsAllStepsWithoutMutatingWireInputs) {
    for (bool streaming : {false, true}) {
        SCOPED_TRACE(streaming);
        BatchTestServer server;
        auto            engine = std::make_shared<MultiStepBatchTestEngine>();
        server.setEngine(engine);
        BatchGenerateInputPB request;
        for (int i = 0; i < 2; ++i) {
            auto* input = request.add_inputs();
            input->set_request_id(100 + i);
            input->add_token_ids(i + 1);
            auto* config = input->mutable_generate_config();
            config->set_max_new_tokens(3);
            config->set_is_streaming(streaming);
            config->set_ignore_eos(true);
            config->set_num_beams(1);
            config->set_num_return_sequences(1);
        }
        const auto             before = request.SerializeAsString();
        BatchGenerateOutputsPB response;
        ASSERT_TRUE(server.BatchGenerateCall(nullptr, &request, &response).ok());
        EXPECT_EQ(engine->batch_calls, 1);
        EXPECT_EQ(request.SerializeAsString(), before);
        ASSERT_EQ(response.results_size(), 2);
        for (int i = 0; i < 2; ++i) {
            const auto& result = response.results(i);
            ASSERT_FALSE(result.has_error_info());
            ASSERT_TRUE(result.has_final_output());
            EXPECT_EQ(result.final_output().request_id(), 100 + i);
            const auto& output = result.final_output().flatten_output();
            ASSERT_EQ(output.finished_size(), 1);
            EXPECT_TRUE(output.finished(0));
            ASSERT_EQ(output.aux_info_size(), 1);
            EXPECT_EQ(output.aux_info(0).output_len(), 3);
            const auto&            data = output.output_ids().int32_data();
            std::array<int32_t, 3> tokens{};
            ASSERT_EQ(data.size(), sizeof(tokens));
            std::memcpy(tokens.data(), data.data(), sizeof(tokens));
            EXPECT_EQ(tokens, (std::array<int32_t, 3>{7 + 3 * i, 8 + 3 * i, 9 + 3 * i}));
        }
    }
}

BatchGenerateInputPB batchRequest() {
    BatchGenerateInputPB request;
    for (int i = 0; i < 3; ++i) {
        auto* input = request.add_inputs();
        input->set_request_id(100 + i);
        input->add_token_ids(i + 1);
        input->mutable_generate_config()->set_prefill_only(true);
        input->mutable_generate_config()->set_capture_hidden_states(true);
        input->mutable_generate_config()->set_reuse_cache(false);
    }
    return request;
}

class BatchPollService: public RpcService::Service {
public:
    explicit BatchPollService(BatchTestServer& server): server_(server) {}

    grpc::Status BatchGenerateCall(grpc::ServerContext*        context,
                                   const BatchGenerateInputPB* request,
                                   BatchGenerateOutputsPB*     response) override {
        const auto result = server_.BatchGenerateCall(context, request, response);
        completed.set_value();
        return result;
    }

    std::promise<void> completed;

private:
    BatchTestServer& server_;
};

TEST(LocalRpcServerBatchTest, IdleNativeBatchObservesClientCancellationAndDeadline) {
    for (bool use_deadline : {false, true}) {
        SCOPED_TRACE(use_deadline);
        BatchTestServer local;
        auto            engine = std::make_shared<BatchTestEngine>();
        engine->idle_streams   = true;
        std::promise<void> polling;
        auto               entered = polling.get_future();
        engine->on_first_poll      = [&polling] { polling.set_value(); };
        local.setEngine(engine);
        BatchPollService    service(local);
        auto                completed = service.completed.get_future();
        grpc::ServerBuilder builder;
        builder.RegisterService(&service);
        auto server = builder.BuildAndStart();
        ASSERT_NE(server, nullptr);
        auto                stub = RpcService::NewStub(server->InProcessChannel(grpc::ChannelArguments{}));
        grpc::ClientContext context;
        if (use_deadline) {
            context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(1));
        }
        const auto             request = batchRequest();
        BatchGenerateOutputsPB response;
        auto                   call =
            std::async(std::launch::async, [&] { return stub->BatchGenerateCall(&context, request, &response); });
        const auto entered_state = entered.wait_for(std::chrono::seconds(3));
        EXPECT_EQ(entered_state, std::future_status::ready);
        if (!use_deadline || entered_state != std::future_status::ready) {
            context.TryCancel();
        }
        EXPECT_EQ(call.wait_for(std::chrono::seconds(3)), std::future_status::ready);
        const auto status = call.get();
        EXPECT_EQ(status.error_code(),
                  use_deadline ? grpc::StatusCode::DEADLINE_EXCEEDED : grpc::StatusCode::CANCELLED);
        const auto completed_state = completed.wait_for(std::chrono::seconds(3));
        EXPECT_EQ(completed_state, std::future_status::ready);
        // Release a failed waiter before destroying the test server; the timed
        // assertion above still detects an unbounded native cancellation path.
        if (entered_state == std::future_status::ready && completed_state != std::future_status::ready) {
            for (const auto& stream : engine->streams) {
                stream->reportError(ErrorCode::CANCELLED, "test cleanup");
            }
        }
        server->Shutdown();
        server->Wait();
        ASSERT_EQ(engine->streams.size(), 3u);
        EXPECT_EQ(engine->streams[0]->statusInfo().code(), ErrorCode::CANCELLED);
        EXPECT_EQ(engine->streams[2]->statusInfo().code(), ErrorCode::CANCELLED);
        EXPECT_EQ(engine->streams[0]->reads, 0);
        EXPECT_EQ(engine->streams[2]->reads, 0);
    }
}

TEST(LocalRpcServerBatchTest, CaptureErrorRoundTripsBothWirePathsWithoutRenumberingGrammar) {
    const auto code = ErrorCode::HIDDEN_STATE_CAPTURE_BACKPRESSURE;
    EXPECT_EQ(static_cast<int>(ErrorCode::GRAMMAR_COMPILE_OVERLOADED), 8105);
    EXPECT_EQ(static_cast<int>(code), 8107);
    EXPECT_EQ(static_cast<int>(ErrorCodePB::HIDDEN_STATE_CAPTURE_BACKPRESSURE), 28);
    EXPECT_EQ(ErrorCodeToString(code), "HIDDEN_STATE_CAPTURE_BACKPRESSURE");
    EXPECT_EQ(transErrorCodeToRPC(code), ErrorCodePB::HIDDEN_STATE_CAPTURE_BACKPRESSURE);
    EXPECT_EQ(transRPCErrorCode(transErrorCodeToRPC(code)), code);
    EXPECT_EQ(transErrorCodeToGrpc(code), grpc::StatusCode::RESOURCE_EXHAUSTED);
    BatchTestServer server;
    const auto      status = server.serializeErrorMsg("capture", ErrorInfo(code, "capture quota occupied"));
    EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.error_code(), 8107);
    EXPECT_EQ(details.error_message(), "capture quota occupied");
}

TEST_F(LocalRpcServerBoundaryTest, BatchRpcPreservesNativeAdmissionAndTerminalErrorsWithSuccessSiblings) {
    for (const auto code : {ErrorCode::GENERATE_TIMEOUT,
                           ErrorCode::LONG_PROMPT_ERROR,
                           ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN,
                           ErrorCode::MALLOC_FAILED,
                           ErrorCode::DECODE_MALLOC_FAILED,
                           ErrorCode::LOAD_KV_CACHE_FAILED,
                           ErrorCode::WAIT_TO_RUN_TIMEOUT,
                           ErrorCode::INVALID_PARAMS,
                           ErrorCode::EXECUTION_EXCEPTION,
                           ErrorCode::CANCELLED,
                           ErrorCode::HIDDEN_STATE_CAPTURE_BACKPRESSURE,
                           static_cast<ErrorCode>(99999)}) {
        for (bool accepted : {false, true}) {
            SCOPED_TRACE(static_cast<int>(code));
            SCOPED_TRACE(accepted);
            auto batch_engine = std::make_shared<BatchTestEngine>();
            batch_engine->item_error_code = code;
            batch_engine->item_error_message = "original native failure " + std::to_string(static_cast<int>(code));
            batch_engine->accept_failed_item = accepted;
            local->setEngine(batch_engine);
            auto request = batchRequest();
            const auto before = request.SerializeAsString();
            grpc::ClientContext context;
            context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
            BatchGenerateOutputsPB response;
            const auto status = stub->BatchGenerateCall(&context, request, &response);
            ASSERT_TRUE(status.ok()) << status.error_message();
            EXPECT_EQ(batch_engine->batch_calls, 1);
            EXPECT_EQ(request.SerializeAsString(), before);
            ASSERT_EQ(response.results_size(), 3);
            for (int index : {0, 2}) {
                const auto& success = response.results(index);
                ASSERT_TRUE(success.has_final_output());
                EXPECT_FALSE(success.has_error_info());
                EXPECT_FALSE(success.has_error_details());
                EXPECT_EQ(success.final_output().request_id(), request.inputs(index).request_id());
                EXPECT_EQ(batch_engine->streams[index]->reads, 2);
            }
            const auto& failure = response.results(1);
            EXPECT_FALSE(failure.has_final_output());
            ASSERT_TRUE(failure.has_error_info());
            ASSERT_TRUE(failure.has_error_details());
            EXPECT_EQ(failure.error_info().error_code(), transErrorCodeToRPC(code));
            EXPECT_EQ(failure.error_details().error_code(), static_cast<int64_t>(code));
            EXPECT_EQ(failure.error_details().error_message(), batch_engine->item_error_message);
            EXPECT_EQ(failure.error_info().error_message(), failure.error_details().error_message());
            EXPECT_EQ(batch_engine->streams[1]->statusInfo().code(), code);
            EXPECT_EQ(local->onflightRequestNum(), 0u);
        }
    }
}

TEST(LocalRpcServerBatchTest, NativeBatchPreservesMixedResultsAndDrainsSuccessAfterRejection) {
    BatchTestServer server;
    auto            engine = std::make_shared<BatchTestEngine>();
    server.setEngine(engine);
    auto                   request = batchRequest();
    BatchGenerateOutputsPB response;
    ASSERT_TRUE(server.BatchGenerateCall(nullptr, &request, &response).ok());
    EXPECT_EQ(engine->batch_calls, 1);
    ASSERT_EQ(response.results_size(), 3);
    EXPECT_EQ(response.results(0).final_output().request_id(), 100);
    EXPECT_EQ(response.results(1).error_info().error_code(), ErrorCodePB::HIDDEN_STATE_CAPTURE_BACKPRESSURE);
    ASSERT_TRUE(response.results(1).has_error_details());
    EXPECT_EQ(response.results(1).error_details().error_code(), 8107);
    EXPECT_EQ(response.results(1).error_details().error_message(), "capture quota occupied");
    EXPECT_EQ(response.results(2).final_output().request_id(), 102);
    EXPECT_EQ(engine->streams[0]->reads, 2);
    EXPECT_EQ(engine->streams[1]->reads, 0);
    EXPECT_EQ(engine->streams[2]->reads, 2);
    EXPECT_EQ(server.onflightRequestNum(), 0);
}

TEST(LocalRpcServerBatchTest, BatchFailurePreservesCardinalityAndCancelsUnconsumedStreams) {
    for (bool wrong_cardinality : {false, true}) {
        BatchTestServer server;
        auto            engine    = std::make_shared<BatchTestEngine>();
        engine->wrong_cardinality = wrong_cardinality;
        engine->throw_on_first    = !wrong_cardinality;
        server.setEngine(engine);
        auto                   request = batchRequest();
        BatchGenerateOutputsPB response;
        response.add_results();
        ASSERT_TRUE(server.BatchGenerateCall(nullptr, &request, &response).ok());
        ASSERT_EQ(response.results_size(), 3);
        for (const auto& result : response.results()) {
            EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::EXECUTION_EXCEPTION);
            ASSERT_TRUE(result.has_error_details());
            EXPECT_EQ(result.error_details().error_code(), static_cast<int64_t>(ErrorCode::EXECUTION_EXCEPTION));
            EXPECT_EQ(result.error_details().error_message(), result.error_info().error_message());
            EXPECT_FALSE(result.has_final_output());
        }
        EXPECT_EQ(engine->streams[0]->statusInfo().code(), ErrorCode::CANCELLED);
        EXPECT_EQ(engine->streams[1]->statusInfo().code(), ErrorCode::HIDDEN_STATE_CAPTURE_BACKPRESSURE);
        EXPECT_EQ(engine->streams[2]->statusInfo().code(), ErrorCode::CANCELLED);
        EXPECT_EQ(engine->streams[2]->reads, 0);
    }
}

TEST(LocalRpcServerBatchTest, EnqueueExceptionsKeepExecutionErrorDetailsForEveryItem) {
    for (bool unknown : {false, true}) {
        SCOPED_TRACE(unknown);
        BatchTestServer server;
        auto engine = std::make_shared<BatchTestEngine>();
        engine->before_enqueue = [unknown] {
            if (unknown) {
                throw 42;
            }
            throw std::runtime_error("injected enqueue failure");
        };
        server.setEngine(engine);
        auto request = batchRequest();
        const auto before = request.SerializeAsString();
        BatchGenerateOutputsPB response;
        ASSERT_TRUE(server.BatchGenerateCall(nullptr, &request, &response).ok());
        EXPECT_EQ(engine->batch_calls, 1);
        EXPECT_TRUE(engine->streams.empty());
        EXPECT_EQ(request.SerializeAsString(), before);
        ASSERT_EQ(response.results_size(), request.inputs_size());
        for (const auto& result : response.results()) {
            ASSERT_TRUE(result.has_error_info());
            ASSERT_TRUE(result.has_error_details());
            EXPECT_FALSE(result.has_final_output());
            EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::EXECUTION_EXCEPTION);
            EXPECT_EQ(result.error_details().error_code(), static_cast<int64_t>(ErrorCode::EXECUTION_EXCEPTION));
            EXPECT_EQ(result.error_details().error_message(), result.error_info().error_message());
            EXPECT_EQ(result.error_details().error_message(),
                      unknown ? "BatchGenerateCall failed with unknown exception" :
                                "BatchGenerateCall failed with exception: injected enqueue failure");
        }
        EXPECT_EQ(server.onflightRequestNum(), 0u);
    }
}

TEST(LocalRpcServerBatchTest, PrepareInputRequiresConfigPresenceButAcceptsExplicitDefaults) {
    BatchTestServer server;
    GenerateInputPB request;
    request.set_request_id(100);
    request.add_token_ids(1);
    auto output = std::make_shared<GenerateInput>();
    const auto missing = server.prepareInput(request, output);
    EXPECT_EQ(missing.code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(missing.ToString(), "generate_config must be present");
    EXPECT_EQ(output, nullptr);
    EXPECT_FALSE(request.has_generate_config());

    request.mutable_generate_config();
    const auto before = request.SerializeAsString();
    const auto valid = server.prepareInput(request, output);
    ASSERT_TRUE(valid.ok()) << valid.ToString();
    ASSERT_NE(output, nullptr);
    ASSERT_NE(output->generate_config, nullptr);
    EXPECT_EQ(output->request_id, 100);
    EXPECT_EQ(output->generate_config->max_new_tokens, 0);
    EXPECT_EQ(request.SerializeAsString(), before);
}

TEST(LocalRpcServerBatchTest, MissingConfigAtAnyIndexRejectsEntireBatchBeforeEnqueue) {
    for (int missing_index = 0; missing_index < 3; ++missing_index) {
        SCOPED_TRACE(missing_index);
        BatchTestServer server;
        auto engine = std::make_shared<BatchTestEngine>();
        server.setEngine(engine);
        auto request = batchRequest();
        request.mutable_inputs(missing_index)->clear_generate_config();
        const auto before = request.SerializeAsString();
        BatchGenerateOutputsPB response;
        response.add_results()->mutable_final_output()->set_request_id(-1);
        ASSERT_TRUE(server.BatchGenerateCall(nullptr, &request, &response).ok());
        EXPECT_EQ(engine->batch_calls, 0);
        EXPECT_TRUE(engine->streams.empty());
        EXPECT_EQ(server.onflightRequestNum(), 0u);
        EXPECT_EQ(request.SerializeAsString(), before);
        ASSERT_EQ(response.results_size(), request.inputs_size());
        for (int i = 0; i < response.results_size(); ++i) {
            SCOPED_TRACE(i);
            const auto& result = response.results(i);
            ASSERT_TRUE(result.has_error_info());
            EXPECT_FALSE(result.has_final_output());
            EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::INVALID_PARAMS);
            ASSERT_TRUE(result.has_error_details());
            EXPECT_EQ(result.error_details().error_code(), static_cast<int64_t>(ErrorCode::INVALID_PARAMS));
            EXPECT_EQ(result.error_details().error_message(), result.error_info().error_message());
            EXPECT_EQ(result.error_info().error_message(),
                      i == missing_index ? "generate_config must be present" :
                                           "batch aborted due to input preparation failure at index "
                                               + std::to_string(missing_index));
        }
    }
}

TEST(LocalRpcServerBatchTest, InputPreparationFailureIsAtomicBeforeEnqueue) {
    BatchTestServer server;
    auto            engine = std::make_shared<BatchTestEngine>();
    server.setEngine(engine);
    auto request = batchRequest();
    request.mutable_inputs(1)->mutable_generate_config()->set_min_new_tokens(1);
    BatchGenerateOutputsPB response;
    ASSERT_TRUE(server.BatchGenerateCall(nullptr, &request, &response).ok());
    EXPECT_EQ(engine->batch_calls, 0);
    ASSERT_EQ(response.results_size(), 3);
    for (const auto& result : response.results()) {
        EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::INVALID_PARAMS);
        ASSERT_TRUE(result.has_error_details());
        EXPECT_EQ(result.error_details().error_code(), static_cast<int64_t>(ErrorCode::INVALID_PARAMS));
        EXPECT_EQ(result.error_details().error_message(), result.error_info().error_message());
    }
}

TEST(LocalRpcServerBatchTest, EmptyNullUninitializedAndServiceReadiness) {
    BatchTestServer        server;
    BatchGenerateInputPB   empty;
    BatchGenerateOutputsPB response;
    EXPECT_TRUE(server.BatchGenerateCall(nullptr, &empty, &response).ok());
    EXPECT_EQ(response.results_size(), 0);
    EXPECT_EQ(server.BatchGenerateCall(nullptr, nullptr, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    auto request = batchRequest();
    EXPECT_EQ(server.BatchGenerateCall(nullptr, &request, nullptr).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_TRUE(server.BatchGenerateCall(nullptr, &request, &response).ok());
    ASSERT_EQ(response.results_size(), 3);
    for (const auto& result : response.results()) {
        EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::EXECUTION_EXCEPTION);
        ASSERT_TRUE(result.has_error_details());
        EXPECT_EQ(result.error_details().error_code(), static_cast<int64_t>(ErrorCode::EXECUTION_EXCEPTION));
        EXPECT_EQ(result.error_details().error_message(), result.error_info().error_message());
    }
    LocalRpcServiceImpl service;
    EXPECT_EQ(service.BatchGenerateCall(nullptr, &request, &response).error_code(), grpc::StatusCode::UNAVAILABLE);
}

class LocalRpcServerWorkerStatusTest: public ::testing::TestWithParam<RoleType> {};

TEST_P(LocalRpcServerWorkerStatusTest, TracksRequestsOutsideTheLocalTaskList) {
    WorkerStatusTestServer server(GetParam(), 2);
    auto                  status = server.getWorkerStatusInfo(-1);
    EXPECT_EQ(status.role, GetParam());
    EXPECT_EQ(status.available_concurrency, 2);

    auto first = server.startRequest();
    status     = server.getWorkerStatusInfo(-1);
    EXPECT_TRUE(status.engine_schedule_info.running_task_info_list.empty());
    EXPECT_EQ(status.available_concurrency, 1);

    auto second = server.startRequest();
    EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 0);

    // Direct RPCs may exceed the configured limit. Never report a negative
    // value or wrap around the unsigned in-flight counter.
    auto third = server.startRequest();
    EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 0);
    second.reset();
    EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 0);
    first.reset();
    EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 1);
    third.reset();
    EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 2);
}

TEST_P(LocalRpcServerWorkerStatusTest, NonPositiveLimitReportsNoAvailableConcurrency) {
    for (const int limit : {0, -1}) {
        WorkerStatusTestServer server(GetParam(), limit);
        EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 0);
        auto request = server.startRequest();
        EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 0);
        request.reset();
        EXPECT_EQ(server.getWorkerStatusInfo(-1).available_concurrency, 0);
    }
}

INSTANTIATE_TEST_SUITE_P(Roles,
                        LocalRpcServerWorkerStatusTest,
                        ::testing::Values(RoleType::PREFILL, RoleType::DECODE, RoleType::PDFUSION));

}  // namespace
}  // namespace rtp_llm
