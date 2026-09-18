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

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
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

    ErrorInfo prepare(const GenerateInputPB& input, std::shared_ptr<GenerateInput>& output) {
        return prepareInput(input, output);
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

    void setEngineForTest(const std::shared_ptr<EngineBase>& engine) {
        engine_ = engine;
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

enum class EngineFailureMode {
    THROW_STD_EXCEPTION,
    THROW_UNKNOWN_EXCEPTION,
    BATCH_SIZE_MISMATCH,
    REJECT_ALL,
    REJECT_ALL_WITH_STREAM_ERROR,
};

class FailingEngine: public EngineBase {
public:
    explicit FailingEngine(EngineFailureMode mode): EngineBase(EngineInitParams()), mode_(mode) {}

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        if (mode_ == EngineFailureMode::THROW_STD_EXCEPTION) {
            throw std::runtime_error("injected engine enqueue failure");
        }
        if (mode_ == EngineFailureMode::THROW_UNKNOWN_EXCEPTION) {
            throw 42;
        }
        return streams.empty() ? nullptr : streams.front();
    }

    void enqueue(std::shared_ptr<GenerateStream>&) override {}

    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>& inputs) override {
        if (mode_ == EngineFailureMode::THROW_STD_EXCEPTION) {
            throw std::runtime_error("injected engine enqueue failure");
        }
        if (mode_ == EngineFailureMode::THROW_UNKNOWN_EXCEPTION) {
            throw 42;
        }
        if (mode_ == EngineFailureMode::REJECT_ALL || mode_ == EngineFailureMode::REJECT_ALL_WITH_STREAM_ERROR) {
            if (streams.size() != inputs.size()) {
                throw std::runtime_error("reject-all engine requires one stream per input");
            }
            if (mode_ == EngineFailureMode::REJECT_ALL_WITH_STREAM_ERROR) {
                for (const auto& stream : streams) {
                    stream->reportError(ErrorCode::INVALID_PARAMS, "injected invalid batch parameters");
                }
            }
            return {std::vector<bool>(inputs.size(), false), streams};
        }
        if (streams.size() != inputs.size() + 1) {
            throw std::runtime_error("batch-size mismatch engine requires one extra prebuilt stream");
        }
        return {std::vector<bool>(streams.size(), true), streams};
    }

    absl::Status stop() override {
        return absl::OkStatus();
    }

    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused in LocalRpcServerTest");
    }

    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }

    std::vector<GenerateStreamPtr> streams;

private:
    EngineFailureMode mode_;
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

TEST(GenerateContextTest, CollectBasicMetricsUsesFinalErrorInfoForBareGrpcStatuses) {
    struct TestCase {
        grpc::StatusCode grpc_code;
        ErrorCode        expected_error_code;
        bool             expected_error_qps;
        bool             expected_cancel_qps;
    };
    const std::array<TestCase, 6> test_cases{{
        {grpc::StatusCode::OK, ErrorCode::NONE_ERROR, false, false},
        {grpc::StatusCode::CANCELLED, ErrorCode::CANCELLED, true, true},
        {grpc::StatusCode::INVALID_ARGUMENT, ErrorCode::INVALID_PARAMS, true, false},
        {grpc::StatusCode::DEADLINE_EXCEEDED, ErrorCode::DEADLINE_EXCEEDED, true, false},
        {grpc::StatusCode::RESOURCE_EXHAUSTED, ErrorCode::MALLOC_FAILED, true, false},
        {grpc::StatusCode::INTERNAL, ErrorCode::EXECUTION_EXCEPTION, true, false},
    }};

    for (const auto& test_case : test_cases) {
        SCOPED_TRACE(static_cast<int>(test_case.grpc_code));
        kmonitor::MetricsReporterPtr metrics_reporter;
        auto                         meta = std::make_shared<RpcServerRuntimeMeta>();
        GenerateContext              context(1, 0, nullptr, metrics_reporter, meta);
        context.error_status = test_case.grpc_code == grpc::StatusCode::OK ?
                                   grpc::Status::OK :
                                   grpc::Status(test_case.grpc_code, "bare grpc error");

        RpcMetricsCollector collector;
        context.collectBasicMetrics(collector);

        EXPECT_TRUE(collector.qps);
        EXPECT_EQ(collector.error_qps, test_case.expected_error_qps);
        EXPECT_EQ(collector.cancel_qps, test_case.expected_cancel_qps);
        EXPECT_EQ(collector.error_code, test_case.expected_error_code);
    }
}

TEST(LocalRpcServerTest, PrepareInputReturnsInvalidParamsMappedToInvalidArgument) {
    TestLocalRpcServer server;
    GenerateInputPB    input;
    input.add_token_ids(0);
    input.mutable_generate_config()->set_max_new_tokens(-1);
    auto output = std::make_shared<GenerateInput>();

    ErrorInfo result;
    EXPECT_NO_THROW(result = server.prepare(input, output));

    EXPECT_EQ(result.code(), ErrorCode::INVALID_PARAMS);
    EXPECT_THAT(result.ToString(), HasSubstr("max_new_tokens"));
    EXPECT_EQ(output, nullptr);
    EXPECT_EQ(transErrorCodeToGrpc(result.code()), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(LocalRpcServerTest, PrepareInputRejectsInvalidRoleAsInvalidArgument) {
    TestLocalRpcServer server;
    GenerateInputPB    input;
    input.add_token_ids(0);
    input.mutable_generate_config()->add_role_addrs()->set_role_str("INVALID_ROLE");
    std::shared_ptr<GenerateInput> output;

    ErrorInfo result;
    EXPECT_NO_THROW(result = server.prepare(input, output));
    EXPECT_EQ(result.code(), ErrorCode::INVALID_PARAMS);
    EXPECT_THAT(result.ToString(), HasSubstr("unknown RoleAddrPB role_str"));
    EXPECT_EQ(output, nullptr);

    grpc::ServerContext context;
    const auto          status = server.GenerateStreamCall(&context, &input, nullptr);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB error_details;
    ASSERT_TRUE(error_details.ParseFromString(status.error_details()));
    EXPECT_EQ(error_details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
}

TEST(LocalRpcServerTest, GenerateStreamCallQueryConversionFailureUsesInvalidArgumentWithDetails) {
    TestLocalRpcServer server;
    GenerateInputPB    input;
    input.add_token_ids(0);
    input.mutable_generate_config()->set_max_new_tokens(-1);
    grpc::ServerContext context;

    const auto status = server.GenerateStreamCall(&context, &input, nullptr);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB error_details;
    ASSERT_TRUE(error_details.ParseFromString(status.error_details()));
    EXPECT_EQ(error_details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
    EXPECT_THAT(error_details.error_message(), HasSubstr("max_new_tokens"));
}

TEST(LocalRpcServerTest, BatchGenerateCallSerializesUnexpectedEngineExceptions) {
    for (const auto mode : {EngineFailureMode::THROW_STD_EXCEPTION, EngineFailureMode::THROW_UNKNOWN_EXCEPTION}) {
        TestLocalRpcServer server;
        server.setEngineForTest(std::make_shared<FailingEngine>(mode));
        BatchGenerateInputPB request;
        for (int i = 0; i < 3; ++i) {
            auto* input = request.add_inputs();
            input->set_request_id(123 + i);
            input->add_token_ids(i);
        }
        grpc::ServerContext    context;
        BatchGenerateOutputsPB response;

        grpc::Status status;
        EXPECT_NO_THROW(status = server.BatchGenerateCall(&context, &request, &response));
        EXPECT_TRUE(status.ok());
        ASSERT_EQ(response.results_size(), request.inputs_size());
        for (const auto& result : response.results()) {
            ASSERT_TRUE(result.has_error_info());
            EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::EXECUTION_EXCEPTION);
            EXPECT_THAT(result.error_info().error_message(),
                        HasSubstr(mode == EngineFailureMode::THROW_STD_EXCEPTION ? "injected engine enqueue failure" :
                                                                                   "unknown exception"));
        }
    }
}

TEST(LocalRpcServerTest, GenerateHandlersRejectNullRpcArgumentsWithoutDereferencingThem) {
    TestLocalRpcServer  server;
    grpc::ServerContext context;

    auto stream_status = server.GenerateStreamCall(&context, nullptr, nullptr);
    EXPECT_EQ(stream_status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB stream_details;
    ASSERT_TRUE(stream_details.ParseFromString(stream_status.error_details()));
    EXPECT_EQ(stream_details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
    EXPECT_THAT(stream_details.error_message(), HasSubstr("must not be null"));

    BatchGenerateOutputsPB response;
    auto                   batch_status = server.BatchGenerateCall(&context, nullptr, &response);
    EXPECT_EQ(batch_status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

    BatchGenerateInputPB request;
    batch_status = server.BatchGenerateCall(&context, &request, nullptr);
    EXPECT_EQ(batch_status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(LocalRpcServerTest, EmptyBatchIsRejectedAndClearsResponse) {
    TestLocalRpcServer     server;
    BatchGenerateInputPB   request;
    BatchGenerateOutputsPB response;
    response.add_results();
    grpc::ServerContext context;

    const auto status = server.BatchGenerateCall(&context, &request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB error_details;
    ASSERT_TRUE(error_details.ParseFromString(status.error_details()));
    EXPECT_EQ(error_details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
    EXPECT_THAT(error_details.error_message(), HasSubstr("at least one input"));
    EXPECT_EQ(response.results_size(), 0);
}

TEST(LocalRpcServerTest, GenerateStreamCallRejectsNullWriterBeforeEngineAccess) {
    TestLocalRpcServer server;
    GenerateInputPB    input;
    input.set_request_id(123);
    input.add_token_ids(1);
    grpc::ServerContext context;

    const auto status = server.GenerateStreamCall(&context, &input, nullptr);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
    EXPECT_THAT(details.error_message(), HasSubstr("writer must not be null"));
}

TEST(RpcErrorCodeTest, InvalidParamsUsesDeclaredRpcEnum) {
    EXPECT_EQ(transErrorCodeToRPC(ErrorCode::INVALID_PARAMS), ErrorCodePB::INVALID_PARAMS);
    EXPECT_EQ(transRPCErrorCode(ErrorCodePB::INVALID_PARAMS), ErrorCode::INVALID_PARAMS);
    EXPECT_NE(static_cast<int>(ErrorCodePB::INVALID_PARAMS), static_cast<int>(ErrorCode::INVALID_PARAMS));
}

TEST(LocalRpcServerTest, UninitializedServerRejectsValidBatchRequests) {
    TestLocalRpcServer  server;
    grpc::ServerContext context;

    BatchGenerateInputPB batch_input;
    batch_input.add_inputs()->add_token_ids(1);
    batch_input.add_inputs()->add_token_ids(2);
    BatchGenerateOutputsPB batch_output;
    const auto             batch_status = server.BatchGenerateCall(&context, &batch_input, &batch_output);
    EXPECT_TRUE(batch_status.ok());
    ASSERT_EQ(batch_output.results_size(), batch_input.inputs_size());
    for (const auto& result : batch_output.results()) {
        ASSERT_TRUE(result.has_error_info());
        EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::EXECUTION_EXCEPTION);
        EXPECT_THAT(result.error_info().error_message(), HasSubstr("engine is not initialized"));
    }
}

TEST(LocalRpcServerTest, BatchPrepareInputFailurePreservesCardinalityAndInvalidParams) {
    TestLocalRpcServer   server;
    BatchGenerateInputPB request;
    for (int i = 0; i < 3; ++i) {
        auto* input = request.add_inputs();
        input->add_token_ids(i);
        input->mutable_generate_config()->set_max_new_tokens(1);
    }
    auto* invalid_config = request.mutable_inputs(1)->mutable_generate_config();
    invalid_config->set_max_new_tokens(0);
    invalid_config->set_prefill_only(true);
    invalid_config->set_min_new_tokens(1);

    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    const auto             status = server.BatchGenerateCall(&context, &request, &response);

    EXPECT_TRUE(status.ok());
    ASSERT_EQ(response.results_size(), request.inputs_size());
    for (int i = 0; i < response.results_size(); ++i) {
        SCOPED_TRACE(i);
        const auto& result = response.results(i);
        ASSERT_TRUE(result.has_error_info());
        EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::INVALID_PARAMS);
        EXPECT_EQ(transRPCErrorCode(result.error_info().error_code()), ErrorCode::INVALID_PARAMS);
        EXPECT_THAT(result.error_info().error_message(), Not(HasSubstr("multimodal")));
        if (i == 1) {
            EXPECT_THAT(result.error_info().error_message(), HasSubstr("min_new_tokens"));
        } else {
            EXPECT_THAT(result.error_info().error_message(), HasSubstr("invalid parameters at index 1"));
        }
    }
}

TEST(LocalRpcServerTest, BatchGenerateCallReturnsTerminalErrorForRejectedEnqueue) {
    TestLocalRpcServer server;
    auto               engine  = std::make_shared<FailingEngine>(EngineFailureMode::REJECT_ALL);
    auto               stream0 = createMockStream();
    auto               stream1 = createMockStream();
    auto               stream2 = createMockStream();
    engine->streams            = {stream0, stream1, stream2};
    EXPECT_CALL(*stream0, nextOutput(_)).Times(0);
    EXPECT_CALL(*stream1, nextOutput(_)).Times(0);
    EXPECT_CALL(*stream2, nextOutput(_)).Times(0);
    server.setEngineForTest(engine);

    BatchGenerateInputPB request;
    for (int i = 0; i < 3; ++i) {
        request.add_inputs()->add_token_ids(i);
    }
    BatchGenerateOutputsPB response;
    grpc::ServerContext    context;

    const auto status = server.BatchGenerateCall(&context, &request, &response);

    EXPECT_TRUE(status.ok());
    ASSERT_EQ(response.results_size(), request.inputs_size());
    for (int i = 0; i < response.results_size(); ++i) {
        SCOPED_TRACE(i);
        const auto& result = response.results(i);
        ASSERT_TRUE(result.has_error_info());
        EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::EXECUTION_EXCEPTION);
        EXPECT_THAT(result.error_info().error_message(), HasSubstr("rejected input at index " + std::to_string(i)));
    }
}

TEST(LocalRpcServerTest, BatchGenerateCallPassesThroughRejectedStreamError) {
    TestLocalRpcServer server;
    auto               engine  = std::make_shared<FailingEngine>(EngineFailureMode::REJECT_ALL_WITH_STREAM_ERROR);
    auto               stream0 = createMockStream();
    auto               stream1 = createMockStream();
    engine->streams            = {stream0, stream1};
    EXPECT_CALL(*stream0, nextOutput(_)).Times(0);
    EXPECT_CALL(*stream1, nextOutput(_)).Times(0);
    server.setEngineForTest(engine);

    BatchGenerateInputPB request;
    request.add_inputs()->add_token_ids(1);
    request.add_inputs()->add_token_ids(2);
    BatchGenerateOutputsPB response;
    grpc::ServerContext    context;

    const auto status = server.BatchGenerateCall(&context, &request, &response);

    EXPECT_TRUE(status.ok());
    ASSERT_EQ(response.results_size(), request.inputs_size());
    for (const auto& result : response.results()) {
        ASSERT_TRUE(result.has_error_info());
        EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::INVALID_PARAMS);
        EXPECT_THAT(result.error_info().error_message(), HasSubstr("injected invalid batch parameters"));
    }
}

TEST(LocalRpcServerTest, BatchGenerateCallCancelsEnqueuedStreamsOnOuterExceptionAndPreservesCardinality) {
    TestLocalRpcServer server;
    auto               engine = std::make_shared<FailingEngine>(EngineFailureMode::BATCH_SIZE_MISMATCH);
    for (int i = 0; i < 4; ++i) {
        engine->streams.push_back(createMockStream());
    }
    engine->streams.front()->reportEvent(StreamEvents::GenerateDone);
    server.setEngineForTest(engine);

    BatchGenerateInputPB request;
    for (int i = 0; i < 3; ++i) {
        auto* input = request.add_inputs();
        input->set_request_id(100 + i);
        input->add_token_ids(i);
    }
    BatchGenerateOutputsPB response;
    response.add_results();
    grpc::ServerContext context;

    grpc::Status status;
    EXPECT_NO_THROW(status = server.BatchGenerateCall(&context, &request, &response));
    EXPECT_TRUE(status.ok());
    ASSERT_EQ(response.results_size(), request.inputs_size());
    for (const auto& result : response.results()) {
        ASSERT_TRUE(result.has_error_info());
        EXPECT_EQ(result.error_info().error_code(), ErrorCodePB::EXECUTION_EXCEPTION);
        EXPECT_THAT(result.error_info().error_message(), HasSubstr("enqueueMultiple result size mismatch"));
    }
    ASSERT_EQ(engine->streams.size(), request.inputs_size() + 1);
    ASSERT_NE(engine->streams.front(), nullptr);
    EXPECT_FALSE(engine->streams.front()->hasError());
    EXPECT_TRUE(engine->streams.front()->hasEvent(StreamEvents::GenerateDone));
    for (size_t i = 1; i < engine->streams.size(); ++i) {
        ASSERT_NE(engine->streams[i], nullptr);
        EXPECT_TRUE(engine->streams[i]->hasError());
        EXPECT_EQ(engine->streams[i]->statusInfo().code(), ErrorCode::CANCELLED);
    }
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
