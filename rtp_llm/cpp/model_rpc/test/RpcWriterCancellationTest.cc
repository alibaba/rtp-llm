#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/telemetry/TelemetryRuntime.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"

namespace rtp_llm {
namespace {

class RejectingWriter: public grpc::internal::WriterInterface<GenerateOutputsPB> {
public:
    bool Write(const GenerateOutputsPB& response, grpc::WriteOptions) override {
        ++write_calls;
        last_response.CopyFrom(response);
        return false;
    }

    int               write_calls = 0;
    GenerateOutputsPB last_response;
};

class SingleOutputStream: public GenerateStream {
public:
    explicit SingleOutputStream(GenerationPrefillCudaGraphStatus generation_prefill_cuda_graph_status =
                                    GenerationPrefillCudaGraphStatus::NOT_REQUESTED):
        GenerateStream(makeInput(), makeModelConfig(), RuntimeConfig{}, ResourceContext{}, nullptr),
        generation_prefill_cuda_graph_status_(generation_prefill_cuda_graph_status) {}

    ErrorResult<GenerateOutputs> nextOutput(int64_t /*wait_timeout_ms*/ = 0) override {
        GenerateOutputs outputs;
        GenerateOutput  output;
        output.output_ids = torch::ones({1, 1}, torch::kInt32);
        output.finished   = false;
        outputs.generate_outputs.push_back(std::move(output));
        return ErrorResult<GenerateOutputs>(std::move(outputs));
    }

    void updateOutput(const StreamUpdateInfo&) override {}

    GenerationPrefillCudaGraphStatus generationPrefillCudaGraphStatus() const override {
        return generation_prefill_cuda_graph_status_;
    }

private:
    GenerationPrefillCudaGraphStatus generation_prefill_cuda_graph_status_;

    static std::shared_ptr<GenerateInput> makeInput() {
        auto input             = std::make_shared<GenerateInput>();
        input->request_id      = 41;
        input->generate_config = std::make_shared<GenerateConfig>();
        input->input_ids       = torch::tensor({1}, torch::kInt32);
        return input;
    }

    static ModelConfig makeModelConfig() {
        ModelConfig config;
        config.max_seq_len = 8;
        return config;
    }
};

TEST(RpcWriterCancellationTest, LocalWriteFailureCancelsStreamAndReturnsCancelled) {
    LocalRpcServer                  server;
    RejectingWriter                 writer;
    std::shared_ptr<GenerateStream> stream = std::make_shared<SingleOutputStream>();

    const auto status = server.pollStreamOutput(nullptr, "41", &writer, stream);

    EXPECT_EQ(writer.write_calls, 1);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
}

TEST(RpcWriterCancellationTest, ContextCleanupPropagatesSpecificTerminalError) {
    auto                         stream = std::make_shared<SingleOutputStream>();
    auto                         meta   = std::make_shared<RpcServerRuntimeMeta>();
    kmonitor::MetricsReporterPtr metrics_reporter;
    {
        GenerateContext context(49, 0, nullptr, metrics_reporter, meta);
        context.setStream(stream);
        context.error_info = ErrorInfo(ErrorCode::MALLOC_FAILED, "allocation failed");
    }

    ASSERT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::MALLOC_FAILED);
    const auto schedule_info = meta->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(schedule_info.finished_task_info_list.size(), 1);
    EXPECT_EQ(schedule_info.finished_task_info_list[0].request_id, 49);
    EXPECT_EQ(schedule_info.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::MALLOC_FAILED));
}

TEST(RpcWriterCancellationTest, ContextCleanupPreservesExistingStreamError) {
    auto stream = std::make_shared<SingleOutputStream>();
    stream->reportError(ErrorCode::GENERATE_TIMEOUT, "original terminal error");
    {
        kmonitor::MetricsReporterPtr metrics_reporter;
        auto                         meta = std::make_shared<RpcServerRuntimeMeta>();
        GenerateContext              context(50, 0, nullptr, metrics_reporter, meta);
        context.stream_    = stream;
        context.error_info = ErrorInfo(ErrorCode::MALLOC_FAILED, "later context error");
    }

    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::GENERATE_TIMEOUT);
}

TEST(RpcWriterCancellationTest, RequestGaugeSamplesLiveAtomicValue) {
    std::atomic<size_t>          onflight_requests{3};
    kmonitor::MetricsReporterPtr metrics_reporter;
    auto                         meta = std::make_shared<RpcServerRuntimeMeta>();
    GenerateContext              context(51, 0, nullptr, metrics_reporter, meta);
    context.onflight_requests = &onflight_requests;

    RpcMetricsCollector collector;
    context.collectBasicMetrics(collector);
    EXPECT_EQ(collector.onflight_request, 3);

    onflight_requests.store(1);
    context.collectBasicMetrics(collector);
    EXPECT_EQ(collector.onflight_request, 1);
}

}  // namespace
}  // namespace rtp_llm
