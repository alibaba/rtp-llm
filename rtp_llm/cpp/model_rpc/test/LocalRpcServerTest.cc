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
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"
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
    void setEngineForRuntimePolicyTest(std::shared_ptr<EngineBase> engine) {
        engine_ = std::move(engine);
    }

    ErrorInfo prepareForRuntimePolicyTest(const GenerateInputPB& input, std::shared_ptr<GenerateInput>& output) {
        return prepareInput(input, output);
    }

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

class RuntimePolicyEngine: public EngineBase {
public:
    explicit RuntimePolicyEngine(bool mtp): EngineBase(EngineInitParams()), mtp_(mtp) {}

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }
    void         enqueue(std::shared_ptr<GenerateStream>&) override {}
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("not used");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }
    bool isMTPEagle() override {
        return mtp_;
    }

private:
    bool mtp_;
};

GenerateInputPB makeRuntimePolicyInputPb(bool with_embeddings) {
    GenerateInputPB input_pb;
    input_pb.set_request_id(1);
    input_pb.add_token_ids(1);
    input_pb.mutable_generate_config()->set_max_new_tokens(1);
    if (with_embeddings) {
        auto* tensor = input_pb.mutable_input_embeddings()->add_embeddings();
        tensor->set_data_type(TensorPB::FP32);
        tensor->add_shape(1);
        tensor->add_shape(1);
        const float value = 1.0f;
        tensor->set_fp32_data(&value, sizeof(value));
        input_pb.mutable_input_embeddings()->add_embedding_locs(0);
    }
    return input_pb;
}

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

TEST(LocalRpcServerTest, SingleAndBatchPreparationRejectMtpInputEmbeddings) {
    TestLocalRpcServer server;
    server.setEngineForRuntimePolicyTest(std::make_shared<RuntimePolicyEngine>(true));

    std::shared_ptr<GenerateInput> single_input;
    const auto single_status = server.prepareForRuntimePolicyTest(makeRuntimePolicyInputPb(true), single_input);
    EXPECT_FALSE(single_status.ok());
    EXPECT_EQ(single_status.code(), ErrorCode::INVALID_PARAMS);

    BatchGenerateInputPB batch_pb;
    *batch_pb.add_inputs() = makeRuntimePolicyInputPb(true);
    *batch_pb.add_inputs() = makeRuntimePolicyInputPb(true);
    BatchGenerateOutputsPB batch_response;
    const auto             batch_rpc_status = server.BatchGenerateCall(nullptr, &batch_pb, &batch_response);
    EXPECT_TRUE(batch_rpc_status.ok());
    ASSERT_EQ(batch_response.results_size(), 2);
    EXPECT_THAT(batch_response.results(0).error_info().error_message(), HasSubstr("input_embeddings"));
    EXPECT_THAT(batch_response.results(1).error_info().error_message(), HasSubstr("batch aborted"));
}

TEST(LocalRpcServerTest, SingleAndBatchPreparationAllowTokenOnlyMtpRequests) {
    TestLocalRpcServer server;
    server.setEngineForRuntimePolicyTest(std::make_shared<RuntimePolicyEngine>(true));

    std::shared_ptr<GenerateInput> single_input;
    EXPECT_TRUE(server.prepareForRuntimePolicyTest(makeRuntimePolicyInputPb(false), single_input).ok());

    BatchGenerateInputPB batch_pb;
    *batch_pb.add_inputs() = makeRuntimePolicyInputPb(false);
    *batch_pb.add_inputs() = makeRuntimePolicyInputPb(false);
    for (const auto& input_pb : batch_pb.inputs()) {
        std::shared_ptr<GenerateInput> batch_input;
        EXPECT_TRUE(server.prepareForRuntimePolicyTest(input_pb, batch_input).ok());
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

TEST(LocalRpcServerTest, InputEmbeddingRequestPolicyRejectsBeforeInference) {
    TestLocalRpcServer server;
    server.maga_init_params_.model_supports_input_embeddings = true;
    server.maga_init_params_.model_config_.hidden_size       = 1;
    std::shared_ptr<GenerateInput> input;
    EXPECT_TRUE(server.prepareForRuntimePolicyTest(makeRuntimePolicyInputPb(true), input).ok());
    server.maga_init_params_.model_config_.hidden_size = 2;
    EXPECT_THAT(server.prepareForRuntimePolicyTest(makeRuntimePolicyInputPb(true), input).ToString(),
                HasSubstr("hidden size"));
    server.maga_init_params_.model_config_.hidden_size       = 1;
    server.maga_init_params_.model_supports_input_embeddings = false;
    EXPECT_THAT(server.prepareForRuntimePolicyTest(makeRuntimePolicyInputPb(true), input).ToString(),
                HasSubstr("loaded model"));
    server.maga_init_params_.model_supports_input_embeddings                 = true;
    server.maga_init_params_.ffn_disaggregate_config.enable_ffn_disaggregate = true;
    EXPECT_THAT(server.prepareForRuntimePolicyTest(makeRuntimePolicyInputPb(true), input).ToString(),
                HasSubstr("disaggregation"));
    EXPECT_TRUE(server.prepareForRuntimePolicyTest(makeRuntimePolicyInputPb(false), input).ok());
}

TEST(LocalRpcServerTest, InputEmbeddingDeploymentCapabilityMatchesRuntimePolicy) {
    TestLocalRpcServer server;
    auto&              params              = server.maga_init_params_;
    params.model_config_.hidden_size       = 128;
    params.model_supports_input_embeddings = true;
    EXPECT_TRUE(validateInputEmbeddingsRuntimeSupport(server.inputEmbeddingsRuntimePolicy()).ok());
    params.parallelism_config.tp_size = 2;
    EXPECT_FALSE(validateInputEmbeddingsRuntimeSupport(server.inputEmbeddingsRuntimePolicy()).ok());
    params.parallelism_config.tp_size                      = 1;
    params.ffn_disaggregate_config.enable_ffn_disaggregate = true;
    EXPECT_FALSE(validateInputEmbeddingsRuntimeSupport(server.inputEmbeddingsRuntimePolicy()).ok());
    params.ffn_disaggregate_config.enable_ffn_disaggregate = false;
    params.sp_config.type                                  = SP_TYPE_MTP;
    EXPECT_FALSE(validateInputEmbeddingsRuntimeSupport(server.inputEmbeddingsRuntimePolicy()).ok());
}

// Model a rolling-upgrade peer which advertises support, but implements only
// legacy generation RPCs. The default generated handlers reject the new names.
class LegacyEmbeddingService: public RpcService::Service {
public:
    std::atomic<int> status_calls{0};
    std::atomic<int> single_calls{0};
    std::atomic<int> batch_calls{0};
    std::atomic<int> remote_calls{0};

    grpc::Status GetWorkerStatus(grpc::ServerContext*, const StatusVersionPB*, WorkerStatusPB* response) override {
        ++status_calls;
        response->set_supports_input_embeddings(true);
        return grpc::Status::OK;
    }
    grpc::Status
    GenerateStreamCall(grpc::ServerContext*, const GenerateInputPB*, grpc::ServerWriter<GenerateOutputsPB>*) override {
        ++single_calls;
        return grpc::Status::OK;
    }
    grpc::Status
    BatchGenerateCall(grpc::ServerContext*, const BatchGenerateInputPB*, BatchGenerateOutputsPB*) override {
        ++batch_calls;
        return grpc::Status::OK;
    }
    grpc::Status RemoteGenerate(grpc::ServerContext*, ServerStream* stream) override {
        ++remote_calls;
        GenerateRequestPB request;
        if (stream->Read(&request)) {
            stream->Write(GenerateOutputsPB());
        }
        return grpc::Status::OK;
    }
};

class InProcessEmbeddingRpc {
public:
    explicit InProcessEmbeddingRpc(RpcService::Service* service) {
        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(service);
        server  = builder.BuildAndStart();
        channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials());
        stub    = RpcService::NewStub(channel);
    }
    ~InProcessEmbeddingRpc() {
        if (server) {
            server->Shutdown();
        }
    }
    std::unique_ptr<grpc::Server>     server;
    std::shared_ptr<grpc::Channel>    channel;
    std::unique_ptr<RpcService::Stub> stub;
};

TEST(LocalRpcServerTest, EmbeddingMethodsRejectLegacySingleMixedBatchAndDecodeWithoutFallback) {
    LegacyEmbeddingService service;
    InProcessEmbeddingRpc  rpc(&service);
    ASSERT_NE(rpc.server, nullptr);
    auto                input_pb = makeRuntimePolicyInputPb(true);
    grpc::ClientContext single_context;
    single_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    auto              reader = rpc.stub->GenerateStreamWithInputEmbeddings(&single_context, input_pb);
    GenerateOutputsPB output;
    EXPECT_FALSE(reader->Read(&output));
    EXPECT_EQ(reader->Finish().error_code(), grpc::StatusCode::UNIMPLEMENTED);

    BatchGenerateInputPB batch;
    *batch.add_inputs() = makeRuntimePolicyInputPb(false);
    *batch.add_inputs() = input_pb;
    BatchGenerateOutputsPB response;
    grpc::ClientContext    batch_context;
    batch_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    EXPECT_EQ(rpc.stub->BatchGenerateWithInputEmbeddings(&batch_context, batch, &response).error_code(),
              grpc::StatusCode::UNIMPLEMENTED);

    PrefillRpcServer       server;
    RPCContext             rpc_context{&input_pb, nullptr};
    PrefillGenerateContext context(
        &server.resource(), rpc_context, 5000, nullptr, server.metrics_reporter_, server.meta_);
    context.generate_input       = QueryConverter::transQuery(&input_pb);
    context.grpc_connection.stub = RpcService::NewStub(rpc.channel);
    server.remoteAllocateResource(context);
    EXPECT_EQ(context.error_info.code(), ErrorCode::INVALID_PARAMS);
    EXPECT_FALSE(context.shouldRetry());
    EXPECT_THAT(context.error_info.ToString(), HasSubstr("does not support input_embeddings"));
    EXPECT_EQ(service.single_calls.load(), 0);
    EXPECT_EQ(service.batch_calls.load(), 0);
    EXPECT_EQ(service.remote_calls.load(), 0);
    EXPECT_EQ(service.status_calls.load(), 0);

    // The same old peer remains usable for ordinary text generation.
    auto                text_pb = makeRuntimePolicyInputPb(false);
    grpc::ClientContext text_context;
    auto                text_reader = rpc.stub->GenerateStreamCall(&text_context, text_pb);
    EXPECT_FALSE(text_reader->Read(&output));
    EXPECT_TRUE(text_reader->Finish().ok());
    RPCContext             text_rpc_context{&text_pb, nullptr};
    PrefillGenerateContext text_pd_context(
        &server.resource(), text_rpc_context, 5000, nullptr, server.metrics_reporter_, server.meta_);
    text_pd_context.generate_input       = QueryConverter::transQuery(&text_pb);
    text_pd_context.grpc_connection.stub = RpcService::NewStub(rpc.channel);
    server.remoteAllocateResource(text_pd_context);
    EXPECT_TRUE(text_pd_context.ok());
    EXPECT_TRUE(text_pd_context.closeGrpcStream().ok());
    EXPECT_EQ(service.single_calls.load(), 1);
    EXPECT_EQ(service.remote_calls.load(), 1);
    EXPECT_EQ(service.status_calls.load(), 0);
}

class DispatchRecordingEmbeddingService: public RemoteRpcServiceImpl {
public:
    std::atomic<int>     single_calls{0};
    std::atomic<int>     batch_calls{0};
    std::atomic<int>     remote_calls{0};
    std::atomic<int64_t> remote_deadline_ms{0};
    grpc::Status
    GenerateStreamCall(grpc::ServerContext*, const GenerateInputPB*, grpc::ServerWriter<GenerateOutputsPB>*) override {
        ++single_calls;
        return grpc::Status::OK;
    }
    grpc::Status
    BatchGenerateCall(grpc::ServerContext*, const BatchGenerateInputPB*, BatchGenerateOutputsPB*) override {
        ++batch_calls;
        return grpc::Status::OK;
    }
    grpc::Status RemoteGenerate(grpc::ServerContext* context, ServerStream* stream) override {
        ++remote_calls;
        remote_deadline_ms.store(
            std::chrono::duration_cast<std::chrono::milliseconds>(context->deadline().time_since_epoch()).count());
        GenerateRequestPB request;
        if (!stream->Read(&request)) {
            return grpc::Status(grpc::StatusCode::CANCELLED, "missing allocation request");
        }
        stream->Write(GenerateOutputsPB());
        return grpc::Status::OK;
    }
};

TEST(LocalRpcServerTest, EmbeddingServiceAliasesDispatchExistingVirtualHandlers) {
    DispatchRecordingEmbeddingService service;
    InProcessEmbeddingRpc             rpc(&service);
    ASSERT_NE(rpc.server, nullptr);
    auto                input = makeRuntimePolicyInputPb(true);
    grpc::ClientContext single_context;
    auto                reader = rpc.stub->GenerateStreamWithInputEmbeddings(&single_context, input);
    GenerateOutputsPB   output;
    EXPECT_FALSE(reader->Read(&output));
    EXPECT_TRUE(reader->Finish().ok());
    BatchGenerateInputPB batch;
    *batch.add_inputs() = input;
    BatchGenerateOutputsPB response;
    grpc::ClientContext    batch_context;
    EXPECT_TRUE(rpc.stub->BatchGenerateWithInputEmbeddings(&batch_context, batch, &response).ok());
    grpc::ClientContext remote_context;
    auto                stream = rpc.stub->RemoteGenerateWithInputEmbeddings(&remote_context);
    GenerateRequestPB   request;
    *request.mutable_input() = input;
    ASSERT_TRUE(stream->Write(request));
    stream->WritesDone();
    EXPECT_TRUE(stream->Read(&output));
    EXPECT_FALSE(stream->Read(&output));
    EXPECT_TRUE(stream->Finish().ok());
    EXPECT_EQ(service.single_calls.load(), 1);
    EXPECT_EQ(service.batch_calls.load(), 1);
    EXPECT_EQ(service.remote_calls.load(), 1);
}

TEST(LocalRpcServerTest, EmbeddingServiceAliasesRunActualLocalAndPrefillAdmission) {
    LocalRpcServiceImpl service;
    service.local_server_ = std::make_shared<LocalRpcServer>();
    // No engine is installed: unsupported models must fail before any enqueue.
    InProcessEmbeddingRpc rpc(&service);
    ASSERT_NE(rpc.server, nullptr);
    auto                input = makeRuntimePolicyInputPb(true);
    grpc::ClientContext single_context;
    auto                reader = rpc.stub->GenerateStreamWithInputEmbeddings(&single_context, input);
    GenerateOutputsPB   output;
    EXPECT_FALSE(reader->Read(&output));
    EXPECT_EQ(reader->Finish().error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    BatchGenerateInputPB batch;
    *batch.add_inputs() = makeRuntimePolicyInputPb(false);
    *batch.add_inputs() = input;
    BatchGenerateOutputsPB response;
    grpc::ClientContext    batch_context;
    EXPECT_TRUE(rpc.stub->BatchGenerateWithInputEmbeddings(&batch_context, batch, &response).ok());
    ASSERT_EQ(response.results_size(), 2);
    EXPECT_TRUE(response.results(0).has_error_info());
    EXPECT_THAT(response.results(1).error_info().error_message(), HasSubstr("loaded model"));

    RemoteRpcServiceImpl prefill_service;
    prefill_service.prefill_server_          = std::make_shared<PrefillBatchRpcServer>();
    prefill_service.prefill_server_->engine_ = std::make_shared<RuntimePolicyEngine>(false);
    InProcessEmbeddingRpc prefill_rpc(&prefill_service);
    ASSERT_NE(prefill_rpc.server, nullptr);
    input.mutable_generate_config()->set_max_new_tokens(2);
    input.mutable_generate_config()->set_can_use_pd_separation(true);
    grpc::ClientContext prefill_context;
    auto                prefill_reader = prefill_rpc.stub->GenerateStreamWithInputEmbeddings(&prefill_context, input);
    EXPECT_FALSE(prefill_reader->Read(&output));
    // local_server_ is deliberately null: this requires Remote's real Prefill route.
    const auto status = prefill_reader->Finish();
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_THAT(status.error_message(), HasSubstr("loaded model"));
}

TEST(LocalRpcServerTest, PrefillDeterministicInputFailureIsNotRetried) {
    PrefillRpcServer server;
    server.engine_                  = std::make_shared<RuntimePolicyEngine>(false);
    auto                   input_pb = makeRuntimePolicyInputPb(true);
    RPCContext             rpc_context{&input_pb, nullptr};
    PrefillGenerateContext context(
        &server.resource(), rpc_context, 1000, nullptr, server.metrics_reporter_, server.meta_);
    // Undeclared model capability is rejected before a decode connection or stream allocation.
    server.maga_init_params_.pd_sep_config.prefill_retry_times = 3;
    const auto status                                          = server.syncPrefix(context);
    EXPECT_FALSE(status.ok());
    EXPECT_FALSE(context.shouldRetry());
    EXPECT_EQ(context.retry_times, 1);
    EXPECT_EQ(context.error_info.code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(context.getStream(), nullptr);
}

TEST(LocalRpcServerTest, SharedPolicyPreservesDeferredMultimodalRangeCheck) {
    GenerateInput input;
    input.input_ids             = torch::tensor({1}, torch::kInt32);
    input.input_embeddings      = std::vector<torch::Tensor>{torch::ones({1, 2})};
    input.input_embeddings_locs = std::vector<int32_t>{3};
    InputEmbeddingsRuntimePolicy policy{2, true, 1, false, false, false};
    EXPECT_TRUE(validateInputEmbeddingsForRequest(input, policy, false).ok());
    EXPECT_FALSE(validateInputEmbeddingsForRequest(input, policy).ok());
    policy.hidden_size = 3;
    EXPECT_FALSE(validateInputEmbeddingsForRequest(input, policy, false).ok());
}

TEST(LocalRpcServerTest, ExpiredPrefillEmbeddingAllocationDoesNotOpenRpcOrResetBudget) {
    PrefillRpcServer       server;
    auto                   input_pb = makeRuntimePolicyInputPb(true);
    RPCContext             rpc_context{&input_pb, nullptr};
    PrefillGenerateContext context(&server.resource(), rpc_context, 1, nullptr, server.metrics_reporter_, server.meta_);
    context.generate_input   = QueryConverter::transQuery(&input_pb);
    context.request_deadline = std::chrono::system_clock::now() - std::chrono::seconds(1);
    // No stub: an expired request must return before attempting the network.
    server.remoteAllocateResource(context);
    EXPECT_EQ(context.error_info.code(), ErrorCode::GENERATE_TIMEOUT);
    EXPECT_EQ(context.client_stream, nullptr);
    EXPECT_EQ(std::atomic_load(&context.client_context), nullptr);
}

TEST(LocalRpcServerTest, PrefillEmbeddingAllocationKeepsOriginalRequestDeadline) {
    DispatchRecordingEmbeddingService service;
    InProcessEmbeddingRpc             rpc(&service);
    ASSERT_NE(rpc.server, nullptr);
    PrefillRpcServer server;
    auto             input_pb = makeRuntimePolicyInputPb(true);
    RPCContext       rpc_context{&input_pb, nullptr};
    {
        PrefillGenerateContext context(
            &server.resource(), rpc_context, 10000, nullptr, server.metrics_reporter_, server.meta_);
        context.generate_input       = QueryConverter::transQuery(&input_pb);
        context.grpc_connection.stub = RpcService::NewStub(rpc.channel);
        context.request_deadline     = std::chrono::system_clock::now() + std::chrono::seconds(4);
        const auto original_deadline_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(context.request_deadline->time_since_epoch()).count();
        // Preparation's shorter retry budget must not truncate the generation stream.
        context.setRetryTimeoutMs(1000);
        server.maga_init_params_.pd_sep_config.max_rpc_timeout_ms = 10000;
        server.remoteAllocateResource(context);
        EXPECT_TRUE(context.ok());
        EXPECT_EQ(service.remote_calls.load(), 1);
        EXPECT_LE(service.remote_deadline_ms.load(), original_deadline_ms + 100);
        EXPECT_GE(service.remote_deadline_ms.load(), original_deadline_ms - 100);
        EXPECT_TRUE(context.closeGrpcStream().ok());
    }
    {
        PrefillGenerateContext context(
            &server.resource(), rpc_context, 10000, nullptr, server.metrics_reporter_, server.meta_);
        context.generate_input       = QueryConverter::transQuery(&input_pb);
        context.grpc_connection.stub = RpcService::NewStub(rpc.channel);
        context.request_deadline     = std::chrono::system_clock::now() - std::chrono::seconds(1);
        const int calls_before       = service.remote_calls.load();
        server.remoteAllocateResource(context);
        EXPECT_EQ(context.error_info.code(), ErrorCode::GENERATE_TIMEOUT);
        EXPECT_EQ(context.client_stream, nullptr);
        EXPECT_EQ(std::atomic_load(&context.client_context), nullptr);
        EXPECT_EQ(service.remote_calls.load(), calls_before);
    }
}

TEST(LocalRpcServerTest, EmptyInputEmbeddingDimensionsAreRejected) {
    for (const auto& shape : std::vector<std::vector<int64_t>>{{0}, {1, 0}, {0, 1}}) {
        std::vector<torch::Tensor> embeddings{torch::empty(shape, torch::kFloat32)};
        EXPECT_FALSE(validateInputEmbeddings(embeddings, {0}, 1).ok());
        EXPECT_FALSE(validateAndNormalizeInputEmbeddings(embeddings, {0}, 1).ok());
    }
}

}  // namespace rtp_llm
