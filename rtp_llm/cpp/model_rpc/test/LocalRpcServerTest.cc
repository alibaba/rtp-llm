#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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

    void setBatchRuntime(std::shared_ptr<EngineBase> engine, std::shared_ptr<RpcServerRuntimeMeta> meta) {
        engine_ = std::move(engine);
        meta_   = std::move(meta);
    }

    void failPrepareFor(int64_t request_id, ErrorCode error_code) {
        prepare_failure_ = std::make_pair(request_id, error_code);
    }

    EngineScheduleInfo scheduleInfo(int64_t latest_finished_version = -1) {
        return meta_->getEngineScheduleInfo(latest_finished_version);
    }

    std::future<void> cancellationChecked() {
        return cancellation_checked_.get_future();
    }

    std::atomic<bool> cancelled{false};

protected:
    ErrorInfo prepareInput(const GenerateInputPB& input_pb, std::shared_ptr<GenerateInput>& output) override {
        if (prepare_failure_ && prepare_failure_->first == input_pb.request_id()) {
            return ErrorInfo(prepare_failure_->second, "injected prepare failure");
        }
        return LocalRpcServer::prepareInput(input_pb, output);
    }

    bool isCancelled(grpc::ServerContext*) const override {
        std::call_once(cancellation_check_once_, [this] { cancellation_checked_.set_value(); });
        return cancelled.load();
    }

private:
    mutable std::once_flag     cancellation_check_once_;
    mutable std::promise<void> cancellation_checked_;
    std::optional<std::pair<int64_t, ErrorCode>> prepare_failure_;
};

class FixedBatchEngine: public EngineBase {
public:
    explicit FixedBatchEngine(std::vector<GenerateStreamPtr> streams):
        EngineBase(EngineInitParams()), streams_(std::move(streams)) {}

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }

    void enqueue(std::shared_ptr<GenerateStream>&) override {}

    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>& inputs) override {
        EXPECT_EQ(inputs.size(), streams_.size());
        for (size_t i = 0; i < std::min(inputs.size(), streams_.size()); ++i) {
            EXPECT_EQ(inputs[i]->request_id, streams_[i]->generateInput()->request_id);
            EXPECT_EQ(inputs[i]->group_id, streams_[i]->generateInput()->group_id);
        }
        return {std::vector<bool>(streams_.size(), true), streams_};
    }

    absl::Status stop() override {
        return absl::OkStatus();
    }

    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("not used by LocalRpcServer batch tests");
    }

    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }

private:
    std::vector<GenerateStreamPtr> streams_;
};

class ThrowingBatchEngine: public FixedBatchEngine {
public:
    ThrowingBatchEngine(): FixedBatchEngine({}) {}

    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>&) override {
        throw std::runtime_error("injected enqueue failure");
    }
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

std::shared_ptr<MockGenerateStream> createMockStream(int64_t request_id = 0, int64_t group_id = -1) {
    auto input             = std::make_shared<GenerateInput>();
    input->generate_config = std::make_shared<GenerateConfig>();
    input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);
    input->request_id      = request_id;
    input->group_id        = group_id;

    ModelConfig model_config;
    model_config.max_seq_len = 3;
    return std::make_shared<MockGenerateStream>(input, model_config, RuntimeConfig{});
}

void addBatchInput(BatchGenerateInputPB& request, int64_t request_id, int64_t batch_id) {
    auto* input = request.add_inputs();
    input->set_request_id(request_id);
    input->add_token_ids(1);
    input->mutable_group_id()->set_value(batch_id);
    auto* config = input->mutable_generate_config();
    config->set_max_new_tokens(1);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
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

TEST(LocalRpcServerTest, BatchGeneratePublishesEveryMemberLifecycle) {
    constexpr int64_t  batch_id = 900;
    TestLocalRpcServer server;
    auto               first  = createMockStream(101, batch_id);
    auto               second = createMockStream(102, batch_id);
    auto               meta   = std::make_shared<RpcServerRuntimeMeta>();
    server.setBatchRuntime(std::make_shared<FixedBatchEngine>(std::vector<GenerateStreamPtr>{first, second}), meta);

    EXPECT_CALL(*first, nextOutput(_))
        .WillOnce(InvokeWithoutArgs([&server] {
            const auto info = server.scheduleInfo();
            EXPECT_EQ(info.running_task_info_list.size(), 2);
            EXPECT_TRUE(info.finished_task_info_list.empty());
            GenerateOutputs outputs;
            outputs.request_id = 101;
            return ErrorResult<GenerateOutputs>(std::move(outputs));
        }))
        .WillOnce(InvokeWithoutArgs([] { return wakeResult(WakeReason::FINISHED); }));
    EXPECT_CALL(*second, nextOutput(_))
        .WillOnce(InvokeWithoutArgs([&server, batch_id] {
            const auto info = server.scheduleInfo();
            EXPECT_EQ(info.running_task_info_list.size(), 1);
            EXPECT_EQ(info.finished_task_info_list.size(), 1);
            if (!info.finished_task_info_list.empty()) {
                EXPECT_EQ(info.finished_task_info_list[0].request_id, 101);
                EXPECT_EQ(info.finished_task_info_list[0].batch_id, batch_id);
            }
            GenerateOutputs outputs;
            outputs.request_id = 102;
            return ErrorResult<GenerateOutputs>(std::move(outputs));
        }))
        .WillOnce(InvokeWithoutArgs([] { return wakeResult(WakeReason::FINISHED); }));

    BatchGenerateInputPB request;
    for (const auto request_id : {101, 102}) {
        auto* input = request.add_inputs();
        input->set_request_id(request_id);
        input->add_token_ids(1);
        input->mutable_group_id()->set_value(batch_id);
        auto* config = input->mutable_generate_config();
        config->set_max_new_tokens(1);
        config->set_num_beams(1);
        config->set_num_return_sequences(1);
    }
    BatchGenerateOutputsPB response;

    const auto status = server.BatchGenerateCall(nullptr, &request, &response);

    EXPECT_TRUE(status.ok());
    EXPECT_EQ(response.results_size(), 2);
    EXPECT_TRUE(response.results(0).has_final_output());
    EXPECT_TRUE(response.results(1).has_final_output());
    const auto info = server.scheduleInfo();
    EXPECT_TRUE(info.running_task_info_list.empty());
    ASSERT_EQ(info.finished_task_info_list.size(), 2);
    EXPECT_EQ(info.finished_task_info_list[0].request_id, 101);
    EXPECT_EQ(info.finished_task_info_list[0].batch_id, batch_id);
    EXPECT_EQ(info.finished_task_info_list[1].request_id, 102);
    EXPECT_EQ(info.finished_task_info_list[1].batch_id, batch_id);
}

TEST(LocalRpcServerTest, BatchGeneratePublishesEveryMemberWhenInputPreparationFails) {
    constexpr int64_t  batch_id = 901;
    TestLocalRpcServer server;
    auto               meta = std::make_shared<RpcServerRuntimeMeta>();
    server.setBatchRuntime(nullptr, meta);
    server.failPrepareFor(102, ErrorCode::LOAD_CACHE_TIMEOUT);
    BatchGenerateInputPB request;
    addBatchInput(request, 101, batch_id);
    addBatchInput(request, 102, batch_id);
    BatchGenerateOutputsPB response;

    const auto status = server.BatchGenerateCall(nullptr, &request, &response);

    EXPECT_TRUE(status.ok());
    ASSERT_EQ(response.results_size(), 2);
    EXPECT_EQ(response.results(0).error_info().error_code(), ErrorCodePB::CANCELLED);
    EXPECT_EQ(response.results(1).error_info().error_code(), ErrorCodePB::LOAD_CACHE_TIMEOUT);
    const auto info = server.scheduleInfo();
    EXPECT_TRUE(info.running_task_info_list.empty());
    ASSERT_EQ(info.finished_task_info_list.size(), 2);
    EXPECT_EQ(info.finished_task_info_list[0].request_id, 101);
    EXPECT_EQ(info.finished_task_info_list[0].batch_id, batch_id);
    EXPECT_EQ(info.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::CANCELLED));
    EXPECT_EQ(info.finished_task_info_list[1].request_id, 102);
    EXPECT_EQ(info.finished_task_info_list[1].batch_id, batch_id);
    EXPECT_EQ(info.finished_task_info_list[1].error_code,
              static_cast<int64_t>(ErrorCode::LOAD_CACHE_TIMEOUT));
}

TEST(LocalRpcServerTest, BatchGeneratePublishesEveryMemberWhenEngineEnqueueThrows) {
    constexpr int64_t  batch_id = 902;
    TestLocalRpcServer server;
    auto               meta = std::make_shared<RpcServerRuntimeMeta>();
    server.setBatchRuntime(std::make_shared<ThrowingBatchEngine>(), meta);
    BatchGenerateInputPB request;
    addBatchInput(request, 201, batch_id);
    addBatchInput(request, 202, batch_id);
    BatchGenerateOutputsPB response;

    EXPECT_THROW(server.BatchGenerateCall(nullptr, &request, &response), std::runtime_error);

    const auto info = server.scheduleInfo();
    EXPECT_TRUE(info.running_task_info_list.empty());
    ASSERT_EQ(info.finished_task_info_list.size(), 2);
    for (const auto& task : info.finished_task_info_list) {
        EXPECT_EQ(task.batch_id, batch_id);
        EXPECT_EQ(task.error_code, static_cast<int64_t>(ErrorCode::EXECUTION_EXCEPTION));
    }
}

TEST(LocalRpcServerTest, BatchGeneratePreservesTypedPerItemErrorCode) {
    constexpr int64_t  batch_id = 903;
    TestLocalRpcServer server;
    auto               stream = createMockStream(301, batch_id);
    auto               meta   = std::make_shared<RpcServerRuntimeMeta>();
    server.setBatchRuntime(std::make_shared<FixedBatchEngine>(std::vector<GenerateStreamPtr>{stream}), meta);
    EXPECT_CALL(*stream, nextOutput(_)).WillOnce(InvokeWithoutArgs([] {
        return ErrorResult<GenerateOutputs>(ErrorCode::LOAD_CACHE_TIMEOUT, "load cache timed out");
    }));
    BatchGenerateInputPB request;
    addBatchInput(request, 301, batch_id);
    BatchGenerateOutputsPB response;

    const auto status = server.BatchGenerateCall(nullptr, &request, &response);

    EXPECT_TRUE(status.ok());
    ASSERT_EQ(response.results_size(), 1);
    EXPECT_EQ(response.results(0).error_info().error_code(), ErrorCodePB::LOAD_CACHE_TIMEOUT);
    const auto info = server.scheduleInfo();
    ASSERT_EQ(info.finished_task_info_list.size(), 1);
    EXPECT_EQ(info.finished_task_info_list[0].error_code,
              static_cast<int64_t>(ErrorCode::LOAD_CACHE_TIMEOUT));
    EXPECT_EQ(info.finished_task_info_list[0].batch_id, batch_id);
}

}  // namespace rtp_llm
