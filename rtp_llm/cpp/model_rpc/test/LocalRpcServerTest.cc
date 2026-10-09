#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "opentelemetry/exporters/memory/in_memory_span_data.h"
#include "opentelemetry/exporters/memory/in_memory_span_exporter_factory.h"
#include "opentelemetry/sdk/trace/span_data.h"
#include "rtp_llm/cpp/telemetry/TelemetryRuntime.h"

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

using namespace ::testing;

namespace rtp_llm {

class MockGenerateStream: public GenerateStream {
public:
    MockGenerateStream(const std::shared_ptr<GenerateInput>& input,
                       const ModelConfig&                    model_config,
                       const RuntimeConfig&                  runtime_config):
        GenerateStream(input, model_config, runtime_config, ResourceContext{}, nullptr) {}

    MOCK_METHOD((ErrorResult<GenerateOutputs>), nextOutput, (int64_t), (override));
    MOCK_METHOD(void, updateOutput, (const StreamUpdateInfo&), (override));
};

class TestLocalRpcServer: public LocalRpcServer {
public:
    grpc::Status poll(std::shared_ptr<GenerateStream>& stream) {
        return pollStreamOutput(nullptr, "request", nullptr, stream);
    }

    grpc::Status poll(WriterInterface* writer, std::shared_ptr<GenerateStream>& stream) {
        return pollStreamOutput(nullptr, "request", writer, stream);
    }

    ErrorInfo collect(std::shared_ptr<GenerateStream>& stream) {
        GenerateOutputs last_outputs;
        return collectStreamOutput(nullptr, stream, nullptr, last_outputs);
    }

    void configureAllocatorDump(bool enabled, std::string auth_token, double cooldown_seconds = 0.0) {
        torch_allocator_dump_enabled_                = enabled;
        torch_allocator_dump_auth_token_             = std::move(auth_token);
        torch_allocator_dump_cooldown_seconds_       = cooldown_seconds;
        maga_init_params_.parallelism_config.tp_rank = 0;
        maga_init_params_.parallelism_config.tp_size = 1;
    }

    void configureInternalAllocatorDumpPeer(bool allowed) {
        internal_allocator_dump_peer_allowed_        = allowed;
        maga_init_params_.parallelism_config.tp_size = 2;
    }

    grpc::Status authorizeAllocatorDump(const TorchAllocatorDumpRequestPB& request) const {
        return authorizeTorchAllocatorDump(request);
    }

    grpc::Status beginAllocatorDump(const std::string& dump_id) {
        return beginTorchAllocatorDump(dump_id);
    }

    void finishAllocatorDump() {
        finishTorchAllocatorDump();
    }

    size_t allocatorDumpReplayHistorySize() const {
        return torch_allocator_dump_ids_.size();
    }

    grpc::Status aggregateAllocatorDumpResults(const std::string&                             dump_id,
                                               const std::vector<TorchAllocatorDumpResultPB>& results,
                                               TorchAllocatorDumpResponsePB*                  response) const {
        return aggregateTorchAllocatorDumpResults(dump_id, results, response);
    }

    void setAllocatorDumpCallback(std::function<TorchAllocatorDumpResultPB(const std::string&)> callback) {
        allocator_dump_callback_ = std::move(callback);
    }

    std::future<void> cancellationChecked() {
        return cancellation_checked_.get_future();
    }

    void setBatchEngine(std::shared_ptr<EngineBase> engine) {
        engine_ = std::move(engine);
    }

    std::atomic<bool> cancelled{false};

protected:
    bool isCancelled(grpc::ServerContext*) const override {
        std::call_once(cancellation_check_once_, [this] { cancellation_checked_.set_value(); });
        return cancelled.load();
    }

    bool isTorchAllocatorDumpInternalPeer(grpc::ServerContext*) const override {
        return internal_allocator_dump_peer_allowed_;
    }

    TorchAllocatorDumpResultPB dumpTorchAllocatorOnCurrentProcess(const std::string& dump_id) override {
        if (allocator_dump_callback_) {
            return allocator_dump_callback_(dump_id);
        }
        TorchAllocatorDumpResultPB result;
        result.set_dump_id(dump_id);
        result.set_success(true);
        return result;
    }

private:
    std::function<TorchAllocatorDumpResultPB(const std::string&)> allocator_dump_callback_;
    bool                                                          internal_allocator_dump_peer_allowed_{false};
    mutable std::once_flag                                        cancellation_check_once_;
    mutable std::promise<void>                                    cancellation_checked_;
};

class RecordingWriter: public LocalRpcServer::WriterInterface {
public:
    bool Write(const GenerateOutputsPB& outputs, grpc::WriteOptions) override {
        outputs_.push_back(outputs);
        return true;
    }

    std::vector<GenerateOutputsPB> outputs_;
};

class FailingWriter: public LocalRpcServer::WriterInterface {
public:
    bool Write(const GenerateOutputsPB&, grpc::WriteOptions) override {
        ++write_count_;
        return false;
    }

    size_t write_count_{0};
};

enum class WakeReason {
    OUTPUT,
    FINISHED,
    STREAM_ERROR,
    TIMEOUT
};

std::shared_ptr<MockGenerateStream> createMockStream() {
    auto input             = std::make_shared<GenerateInput>();
    input->generate_config = std::make_shared<GenerateConfig>();
    input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);

    ModelConfig model_config;
    model_config.max_seq_len = 3;
    return std::make_shared<MockGenerateStream>(input, model_config, RuntimeConfig{});
}

std::shared_ptr<NormalGenerateStream> createNormalStream() {
    auto input             = std::make_shared<GenerateInput>();
    input->generate_config = std::make_shared<GenerateConfig>();
    input->begin_time_us   = autil::TimeUtility::currentTimeInMicroSeconds();
    input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);

    ModelConfig model_config;
    model_config.max_seq_len = 3;
    return std::make_shared<NormalGenerateStream>(input, model_config, RuntimeConfig{}, ResourceContext{}, nullptr);
}

ErrorResult<GenerateOutputs> wakeResult(WakeReason reason) {
    switch (reason) {
        case WakeReason::OUTPUT: {
            GenerateOutputs outputs;
            return ErrorResult<GenerateOutputs>(std::move(outputs));
        }
        case WakeReason::FINISHED:
            return ErrorResult<GenerateOutputs>(ErrorCode::FINISHED, "finished");
        case WakeReason::STREAM_ERROR:
            return ErrorResult<GenerateOutputs>(ErrorCode::EXECUTION_EXCEPTION, "failed");
        case WakeReason::TIMEOUT:
            return ErrorResult<GenerateOutputs>(ErrorCode::GENERATE_TIMEOUT, "timeout");
    }
    return ErrorResult<GenerateOutputs>(ErrorCode::UNKNOWN_ERROR, "unknown wake reason");
}

void publishWakeError(MockGenerateStream* stream, WakeReason reason) {
    if (reason == WakeReason::STREAM_ERROR) {
        stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "failed");
    } else if (reason == WakeReason::TIMEOUT) {
        stream->reportError(ErrorCode::GENERATE_TIMEOUT, "timeout");
    }
}

ErrorCode expectedStreamError(WakeReason reason) {
    if (reason == WakeReason::STREAM_ERROR) {
        return ErrorCode::EXECUTION_EXCEPTION;
    }
    if (reason == WakeReason::TIMEOUT) {
        return ErrorCode::GENERATE_TIMEOUT;
    }
    return ErrorCode::CANCELLED;
}

class BatchTraceEngine: public EngineBase {
public:
    BatchTraceEngine(): EngineBase(EngineInitParams{}) {}
    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        return {};
    }
    void enqueue(std::shared_ptr<GenerateStream>&) override {}
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>& inputs) override {
        ++calls;
        request_ids.clear();
        for (const auto& input : inputs) {
            request_ids.push_back(input->request_id);
        }
        return {{true, true}, streams};
    }
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }
    int                            calls = 0;
    std::vector<int64_t>           request_ids;
    std::vector<GenerateStreamPtr> streams;
};

TEST(LocalRpcServerTest, BatchTraceKeepsPerItemParentsErrorsAndTransportOk) {
    namespace memory = opentelemetry::exporter::memory;
    for (const bool enabled : {false, true}) {
        for (const bool cancelled : {false, true}) {
            telemetry::TelemetryRuntime::shutdown(5000);
            auto data = std::make_shared<memory::InMemorySpanData>(128);
            if (enabled) {
                telemetry::TelemetryConfig config;
                config.enabled = true;
                config.role    = "test";
                ASSERT_TRUE(telemetry::TelemetryRuntime::initWithExporter(
                    memory::InMemorySpanExporterFactory::Create(data), config));
            }
            TestLocalRpcServer server;
            server.cancelled = cancelled;
            auto engine      = std::make_shared<BatchTraceEngine>();
            auto first       = createMockStream();
            auto second      = createMockStream();
            EXPECT_CALL(*first, nextOutput(_)).WillOnce(Return(wakeResult(WakeReason::STREAM_ERROR)));
            EXPECT_CALL(*second, nextOutput(_)).WillOnce(Return(wakeResult(WakeReason::FINISHED)));
            engine->streams = {first, second};
            server.setBatchEngine(engine);
            BatchGenerateInputPB request;
            for (int i = 0; i < 2; ++i) {
                auto* input = request.add_inputs();
                input->set_request_id(801 + i);
                input->add_token_ids(1);
                input->mutable_generate_config()->set_max_new_tokens(1);
                input->mutable_request_info()->mutable_trace_context()->set_traceparent(
                    i == 0 ? "00-11111111111111111111111111111111-2222222222222222-01" :
                             "00-11111111111111111111111111111111-3333333333333333-01");
            }
            grpc::ServerContext    context;
            BatchGenerateOutputsPB response;
            EXPECT_TRUE(server.BatchGenerateCall(&context, &request, &response).ok());
            ASSERT_EQ(response.results_size(), 2);
            EXPECT_TRUE(response.results(0).has_error_info());
            EXPECT_EQ(response.results(1).has_error_info(), cancelled);
            EXPECT_EQ(engine->calls, 1);
            EXPECT_EQ(engine->request_ids, (std::vector<int64_t>{801, 802}));
            EXPECT_TRUE(telemetry::TelemetryRuntime::shutdown(5000));
            auto   spans = data->GetSpans();
            size_t items = 0;
            size_t rpcs  = 0;
            for (const auto& span : spans) {
                if (span->GetName() == "rtp_llm.batch_generate_call") {
                    ++rpcs;
                    EXPECT_EQ(span->GetStatus(), opentelemetry::trace::StatusCode::kOk);
                }
                if (span->GetName() != "rtp_llm.batch_generate_request") {
                    continue;
                }
                ++items;
                char parent[16];
                span->GetParentSpanId().ToLowerBase16(parent);
                const bool is_first = std::string(parent, 16) == "2222222222222222";
                EXPECT_EQ(std::string(parent, 16), is_first ? "2222222222222222" : "3333333333333333");
                EXPECT_EQ(span->GetStatus(),
                          (is_first || cancelled) ? opentelemetry::trace::StatusCode::kError :
                                                    opentelemetry::trace::StatusCode::kOk);
                const auto& attrs = span->GetAttributes();
                EXPECT_EQ(opentelemetry::nostd::get<std::string>(attrs.at("request_id")), is_first ? "801" : "802");
                EXPECT_EQ(attrs.count("rtp_llm.request_id"), 0u);
                EXPECT_EQ(attrs.count("rpc.response.status_code"), 0u);
            }
            EXPECT_EQ(items, enabled ? 2u : 0u);
            EXPECT_EQ(rpcs, enabled ? 1u : 0u);
        }
    }
}

TEST(LocalRpcServerTest, AllocatorDumpAuthorizationRequiresEnablementAndSecret) {
    TestLocalRpcServer          server;
    TorchAllocatorDumpRequestPB request;
    request.set_auth_token("secret");
    request.set_dump_id("dump-123");

    EXPECT_EQ(server.authorizeAllocatorDump(request).error_code(), grpc::StatusCode::PERMISSION_DENIED);

    server.configureAllocatorDump(true, "expected-secret");
    EXPECT_EQ(server.authorizeAllocatorDump(request).error_code(), grpc::StatusCode::UNAUTHENTICATED);

    request.set_auth_token("expected-secret");
    EXPECT_TRUE(server.authorizeAllocatorDump(request).ok());
}

TEST(LocalRpcServerTest, AllocatorDumpAuthorizationRejectsUnsafeCorrelationId) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    TorchAllocatorDumpRequestPB request;
    request.set_auth_token("secret");
    request.set_dump_id("../leak-path");

    EXPECT_EQ(server.authorizeAllocatorDump(request).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(LocalRpcServerTest, RejectedAllocatorDumpRpcReturnsNoBackendDetails) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "expected-secret");
    grpc::ServerContext          context;
    TorchAllocatorDumpRequestPB  request;
    TorchAllocatorDumpResponsePB response;
    TorchAllocatorDumpResultPB   internal_response;
    request.set_auth_token("wrong-secret");
    request.set_dump_id("dump-123");
    response.add_results()->set_file_path("/stale/private/path");
    internal_response.set_file_path("/stale/private/path");
    internal_response.set_pid(1234);

    const auto status          = server.DumpTorchAllocator(&context, &request, &response);
    const auto internal_status = server.DumpTorchAllocatorInternal(&context, &request, &internal_response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAUTHENTICATED);
    EXPECT_EQ(internal_status.error_code(), grpc::StatusCode::UNAUTHENTICATED);
    EXPECT_EQ(response.results_size(), 0);
    EXPECT_TRUE(internal_response.file_path().empty());
    EXPECT_EQ(internal_response.pid(), 0);
}

TEST(LocalRpcServerTest, DirectAllocatorDumpRpcRejectsReplayAndCooldown) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret", 60.0);
    grpc::ServerContext          context;
    TorchAllocatorDumpRequestPB  request;
    TorchAllocatorDumpResponsePB response;
    request.set_auth_token("secret");
    request.set_dump_id("dump-first");

    EXPECT_TRUE(server.DumpTorchAllocator(&context, &request, &response).ok());
    ASSERT_EQ(response.results_size(), 1);
    EXPECT_EQ(response.results(0).dump_id(), "dump-first");

    EXPECT_EQ(server.DumpTorchAllocator(&context, &request, &response).error_code(), grpc::StatusCode::ALREADY_EXISTS);
    EXPECT_EQ(response.results_size(), 0);

    request.set_dump_id("dump-second");
    EXPECT_EQ(server.DumpTorchAllocator(&context, &request, &response).error_code(),
              grpc::StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_EQ(response.results_size(), 0);
}

TEST(LocalRpcServerTest, DirectAllocatorDumpRpcIsSingleFlight) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    std::promise<void> dump_started;
    std::promise<void> release_dump;
    auto               release_future = release_dump.get_future().share();
    std::atomic<int>   dump_calls{0};
    server.setAllocatorDumpCallback([&](const std::string& dump_id) {
        if (dump_calls.fetch_add(1) == 0) {
            dump_started.set_value();
            release_future.wait();
        }
        TorchAllocatorDumpResultPB result;
        result.set_dump_id(dump_id);
        result.set_success(true);
        return result;
    });

    TorchAllocatorDumpRequestPB first_request;
    first_request.set_auth_token("secret");
    first_request.set_dump_id("dump-in-flight");
    grpc::ServerContext          first_context;
    TorchAllocatorDumpResponsePB first_response;
    auto                         first_call = std::async(
        std::launch::async, [&] { return server.DumpTorchAllocator(&first_context, &first_request, &first_response); });
    const auto dump_started_status = dump_started.get_future().wait_for(std::chrono::seconds(5));
    if (dump_started_status != std::future_status::ready) {
        release_dump.set_value();
        first_call.wait();
        ADD_FAILURE() << "first allocator dump did not start";
        return;
    }

    TorchAllocatorDumpRequestPB second_request;
    second_request.set_auth_token("secret");
    second_request.set_dump_id("dump-concurrent");
    grpc::ServerContext          second_context;
    TorchAllocatorDumpResponsePB second_response;
    EXPECT_EQ(server.DumpTorchAllocator(&second_context, &second_request, &second_response).error_code(),
              grpc::StatusCode::ABORTED);

    release_dump.set_value();
    ASSERT_EQ(first_call.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(first_call.get().ok());
    EXPECT_EQ(dump_calls.load(), 1);
}

TEST(LocalRpcServerTest, InternalAllocatorDumpRequiresTrustedTpPeer) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    server.configureInternalAllocatorDumpPeer(false);
    TorchAllocatorDumpRequestPB request;
    request.set_auth_token("secret");
    request.set_dump_id("dump-internal");
    grpc::ServerContext        context;
    TorchAllocatorDumpResultPB response;

    const auto status = server.DumpTorchAllocatorInternal(&context, &request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::PERMISSION_DENIED);
    EXPECT_TRUE(response.dump_id().empty());
}

TEST(LocalRpcServerTest, InternalAllocatorDumpEnforcesAdmissionAndReplayProtection) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    server.configureInternalAllocatorDumpPeer(true);
    std::promise<void> dump_started;
    std::promise<void> release_dump;
    auto               release_future = release_dump.get_future().share();
    server.setAllocatorDumpCallback([&](const std::string& dump_id) {
        dump_started.set_value();
        release_future.wait();
        TorchAllocatorDumpResultPB result;
        result.set_dump_id(dump_id);
        result.set_success(true);
        return result;
    });

    TorchAllocatorDumpRequestPB first_request;
    first_request.set_auth_token("secret");
    first_request.set_dump_id("dump-internal-first");
    grpc::ServerContext        first_context;
    TorchAllocatorDumpResultPB first_response;
    auto                       first_call          = std::async(std::launch::async, [&] {
        return server.DumpTorchAllocatorInternal(&first_context, &first_request, &first_response);
    });
    const auto                 dump_started_status = dump_started.get_future().wait_for(std::chrono::seconds(5));
    if (dump_started_status != std::future_status::ready) {
        release_dump.set_value();
        first_call.wait();
        ADD_FAILURE() << "internal allocator dump did not start";
        return;
    }

    TorchAllocatorDumpRequestPB second_request;
    second_request.set_auth_token("secret");
    second_request.set_dump_id("dump-internal-second");
    grpc::ServerContext        second_context;
    TorchAllocatorDumpResultPB second_response;
    EXPECT_EQ(server.DumpTorchAllocatorInternal(&second_context, &second_request, &second_response).error_code(),
              grpc::StatusCode::ABORTED);

    release_dump.set_value();
    ASSERT_EQ(first_call.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(first_call.get().ok());
    EXPECT_EQ(first_response.dump_id(), first_request.dump_id());

    grpc::ServerContext        replay_context;
    TorchAllocatorDumpResultPB replay_response;
    EXPECT_EQ(server.DumpTorchAllocatorInternal(&replay_context, &first_request, &replay_response).error_code(),
              grpc::StatusCode::ALREADY_EXISTS);
}

TEST(LocalRpcServerTest, InternalAllocatorDumpEnforcesCooldownOnEachRank) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret", 60.0);
    server.configureInternalAllocatorDumpPeer(true);
    TorchAllocatorDumpRequestPB request;
    request.set_auth_token("secret");
    request.set_dump_id("dump-internal-first");
    grpc::ServerContext        first_context;
    TorchAllocatorDumpResultPB first_response;
    ASSERT_TRUE(server.DumpTorchAllocatorInternal(&first_context, &request, &first_response).ok());

    request.set_dump_id("dump-internal-second");
    grpc::ServerContext        second_context;
    TorchAllocatorDumpResultPB second_response;
    EXPECT_EQ(server.DumpTorchAllocatorInternal(&second_context, &request, &second_response).error_code(),
              grpc::StatusCode::RESOURCE_EXHAUSTED);
}

TEST(LocalRpcServerTest, LeaderAdmissionAllowsExactlyOneMatchingInternalFanout) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    server.configureInternalAllocatorDumpPeer(true);
    ASSERT_TRUE(server.beginAllocatorDump("dump-fanout").ok());

    TorchAllocatorDumpRequestPB request;
    request.set_auth_token("secret");
    request.set_dump_id("dump-fanout");
    grpc::ServerContext        context;
    TorchAllocatorDumpResultPB response;
    EXPECT_TRUE(server.DumpTorchAllocatorInternal(&context, &request, &response).ok());
    EXPECT_EQ(response.dump_id(), request.dump_id());

    grpc::ServerContext        replay_context;
    TorchAllocatorDumpResultPB replay_response;
    EXPECT_EQ(server.DumpTorchAllocatorInternal(&replay_context, &request, &replay_response).error_code(),
              grpc::StatusCode::ALREADY_EXISTS);

    EXPECT_EQ(server.beginAllocatorDump("dump-other").error_code(), grpc::StatusCode::ABORTED);
    server.finishAllocatorDump();
}

TEST(LocalRpcServerTest, AllocatorDumpReplayHistoryIsBounded) {
    TestLocalRpcServer server;
    server.configureAllocatorDump(true, "secret");
    for (int index = 0; index <= 1024; ++index) {
        ASSERT_TRUE(server.beginAllocatorDump("dump-" + std::to_string(index)).ok());
        server.finishAllocatorDump();
    }

    EXPECT_EQ(server.allocatorDumpReplayHistorySize(), 1024);
    EXPECT_TRUE(server.beginAllocatorDump("dump-0").ok());
    server.finishAllocatorDump();
    EXPECT_EQ(server.allocatorDumpReplayHistorySize(), 1024);
}

TEST(LocalRpcServerTest, AllocatorDumpAggregationRejectsMismatchedIds) {
    TestLocalRpcServer         server;
    TorchAllocatorDumpResultPB matching;
    matching.set_dump_id("expected-id");
    matching.set_world_rank(0);
    TorchAllocatorDumpResultPB mismatched;
    mismatched.set_dump_id("other-id");
    mismatched.set_world_rank(1);
    TorchAllocatorDumpResponsePB response;
    response.add_results()->set_dump_id("stale-id");

    const auto status = server.aggregateAllocatorDumpResults("expected-id", {matching, mismatched}, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::DATA_LOSS);
    EXPECT_EQ(response.results_size(), 0);
}

TEST(LocalRpcServerTest, PollChecksCancellationBeforeHandlingEveryWakeReason) {
    for (const auto reason :
         std::array{WakeReason::OUTPUT, WakeReason::FINISHED, WakeReason::STREAM_ERROR, WakeReason::TIMEOUT}) {
        TestLocalRpcServer server;
        auto               mock_stream     = createMockStream();
        auto*              mock_stream_ptr = mock_stream.get();
        EXPECT_CALL(*mock_stream, nextOutput(_)).WillOnce(InvokeWithoutArgs([&server, mock_stream_ptr, reason] {
            publishWakeError(mock_stream_ptr, reason);
            server.cancelled = true;
            return wakeResult(reason);
        }));
        std::shared_ptr<GenerateStream> stream = mock_stream;

        const auto status = server.poll(stream);

        EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
        EXPECT_EQ(stream->statusInfo().code(), expectedStreamError(reason));
    }
}

TEST(LocalRpcServerTest, CollectChecksCancellationBeforeHandlingEveryWakeReason) {
    for (const auto reason :
         std::array{WakeReason::OUTPUT, WakeReason::FINISHED, WakeReason::STREAM_ERROR, WakeReason::TIMEOUT}) {
        TestLocalRpcServer server;
        auto               mock_stream     = createMockStream();
        auto*              mock_stream_ptr = mock_stream.get();
        EXPECT_CALL(*mock_stream, nextOutput(_)).WillOnce(InvokeWithoutArgs([&server, mock_stream_ptr, reason] {
            publishWakeError(mock_stream_ptr, reason);
            server.cancelled = true;
            return wakeResult(reason);
        }));
        std::shared_ptr<GenerateStream> stream = mock_stream;

        const auto status = server.collect(stream);

        EXPECT_EQ(status.code(), ErrorCode::CANCELLED);
        EXPECT_EQ(stream->statusInfo().code(), expectedStreamError(reason));
    }
}

TEST(LocalRpcServerTest, PollInterruptsBlockedNextOutputAfterClientCancellation) {
    TestLocalRpcServer              server;
    auto                            cancellation_checked = server.cancellationChecked();
    std::shared_ptr<GenerateStream> stream               = createNormalStream();
    auto poll_result = std::async(std::launch::async, [&server, &stream] { return server.poll(stream); });

    EXPECT_EQ(cancellation_checked.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    server.cancelled = true;

    const auto wait_status = poll_result.wait_for(std::chrono::seconds(5));
    if (wait_status != std::future_status::ready) {
        stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "test poll cancellation timed out");
    }
    EXPECT_EQ(wait_status, std::future_status::ready);
    EXPECT_EQ(poll_result.get().error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
}

TEST(LocalRpcServerTest, PollClassifiesClosedResponseStreamAsCancelled) {
    TestLocalRpcServer server;
    FailingWriter      writer;
    auto               mock_stream = createMockStream();
    EXPECT_CALL(*mock_stream, nextOutput(_)).WillOnce(InvokeWithoutArgs([] {
        GenerateOutputs outputs;
        outputs.request_id = 1;
        return ErrorResult<GenerateOutputs>(std::move(outputs));
    }));
    std::shared_ptr<GenerateStream> stream = mock_stream;

    const auto status = server.poll(&writer, stream);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(status.error_message(), "request output consumer closed");
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
    EXPECT_EQ(writer.write_count_, 1);
}

TEST(LocalRpcServerTest, CollectInterruptsBlockedNextOutputAfterClientCancellation) {
    TestLocalRpcServer              server;
    auto                            cancellation_checked = server.cancellationChecked();
    std::shared_ptr<GenerateStream> stream               = createNormalStream();
    auto collect_result = std::async(std::launch::async, [&server, &stream] { return server.collect(stream); });

    EXPECT_EQ(cancellation_checked.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    server.cancelled = true;

    const auto wait_status = collect_result.wait_for(std::chrono::seconds(5));
    if (wait_status != std::future_status::ready) {
        stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "test collect cancellation timed out");
    }
    EXPECT_EQ(wait_status, std::future_status::ready);
    EXPECT_EQ(collect_result.get().code(), ErrorCode::CANCELLED);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
}

TEST(LocalRpcServerTest, PollWritesFinalLocalOutputBeforeRemoteHandoff) {
    TestLocalRpcServer              server;
    RecordingWriter                 writer;
    auto                            normal_stream = createNormalStream();
    std::shared_ptr<GenerateStream> stream        = normal_stream;
    normal_stream->setNeedReleaseResource(true);
    normal_stream->generate_status_->status.store(StreamState::RUNNING);

    {
        std::lock_guard<std::mutex> lock(*normal_stream->mutex_);
        GenerateOutputs             outputs;
        outputs.request_id = 123;
        normal_stream->enqueueGenerateOutput(std::move(outputs));
        normal_stream->reportEventWithoutLock(StreamEvents::NeedRemoteGenerate);
    }

    const auto status = server.poll(&writer, stream);

    EXPECT_TRUE(status.ok());
    ASSERT_EQ(writer.outputs_.size(), 1);
    EXPECT_EQ(writer.outputs_[0].request_id(), 123);
    EXPECT_TRUE(stream->hasEvent(StreamEvents::NeedRemoteGenerate));
    EXPECT_EQ(stream->getStatus(), StreamState::RUNNING);
    EXPECT_FALSE(normal_stream->stream_cache_resource_->isResourceReleased());
    EXPECT_FALSE(normal_stream->hasOutput());
}

}  // namespace rtp_llm
