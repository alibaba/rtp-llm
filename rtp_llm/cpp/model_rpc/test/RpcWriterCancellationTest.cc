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
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "opentelemetry/exporters/memory/in_memory_span_data.h"
#include "opentelemetry/exporters/memory/in_memory_span_exporter_factory.h"
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

TEST(RpcWriterCancellationTest, ContextMetricsPreserveTypedRpcErrorWithoutStream) {
    for (auto code : {ErrorCode::INVALID_PARAMS, ErrorCode::GENERATE_TIMEOUT, ErrorCode::PRIORITY_PREEMPTED}) {
        SCOPED_TRACE(static_cast<int>(code));
        kmonitor::MetricsReporterPtr metrics_reporter;
        auto meta = std::make_shared<RpcServerRuntimeMeta>();
        GenerateContext context(52, 0, nullptr, metrics_reporter, meta);
        context.error_status = grpcStatusFromErrorInfo({code, "early RPC failure"});
        RpcMetricsCollector collector;
        context.collectBasicMetrics(collector);
        EXPECT_TRUE(collector.qps);
        EXPECT_TRUE(collector.error_qps);
        EXPECT_EQ(collector.error_code, code);
        EXPECT_FALSE(collector.cancel_qps);
        context.markRpcHandlingCompleted();
    }
}

TEST(RpcWriterCancellationTest, ContextMetricsKeepExplicitFirstErrorAheadOfGrpcFallback) {
    kmonitor::MetricsReporterPtr metrics_reporter;
    auto meta = std::make_shared<RpcServerRuntimeMeta>();
    GenerateContext context(53, 0, nullptr, metrics_reporter, meta);
    context.error_info = {ErrorCode::MALLOC_FAILED, "first failure"};
    context.error_status = grpcStatusFromErrorInfo({ErrorCode::EXECUTION_EXCEPTION, "later failure"});
    RpcMetricsCollector collector;
    context.collectBasicMetrics(collector);
    EXPECT_TRUE(collector.error_qps);
    EXPECT_EQ(collector.error_code, ErrorCode::MALLOC_FAILED);
    context.markRpcHandlingCompleted();
}

TEST(RpcWriterCancellationTest, PdPhaseSynthesisExportsRoleSpecificFailureSpans) {
    namespace memory = opentelemetry::exporter::memory;
    telemetry::TelemetryRuntime::shutdown(5000);
    std::shared_ptr<memory::InMemorySpanData> span_data;
    auto exporter = memory::InMemorySpanExporterFactory::Create(span_data);
    telemetry::TelemetryConfig config;
    config.enabled = true;
    config.role = "test";
    ASSERT_TRUE(telemetry::TelemetryRuntime::initWithExporter(std::move(exporter), config));
    for (auto role : {telemetry::PhaseRole::Prefill, telemetry::PhaseRole::Decode}) {
        kmonitor::MetricsReporterPtr reporter;
        auto meta = std::make_shared<RpcServerRuntimeMeta>();
        GenerateContext context(54, 0, nullptr, reporter, meta);
        auto stream = std::make_shared<SingleOutputStream>();
        const auto now = currentTimeUs();
        stream->begin_time_us_ = now - 3000;
        stream->running_started_ = true;
        stream->running_started_time_us_ = now - 2000;
        context.setStream(stream);
        context.error_status = grpcStatusFromErrorInfo({ErrorCode::GENERATE_TIMEOUT, "phase timeout"});
        auto span = telemetry::startRpcServerSpan("pd-test-request", nullptr, true, "RpcService/GenerateStreamCall");
        context.trace_span_guard =
            std::make_unique<telemetry::GrpcStatusSpanGuard>(span, &context.error_status);
        context.synthesizeTracePhases(role, false);
        context.markRpcHandlingCompleted();
    }
    EXPECT_TRUE(telemetry::TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    size_t prefill_count = 0;
    size_t decode_count = 0;
    size_t server_count = 0;
    for (const auto& span : spans) {
        const auto name = span->GetName();
        if (name == "pd-test-request") {
            ++server_count;
            EXPECT_EQ(span->GetSpanKind(), opentelemetry::trace::SpanKind::kServer);
            EXPECT_EQ(span->GetStatus(), opentelemetry::trace::StatusCode::kError);
        } else if (name == "prefill" || name == "decode") {
            prefill_count += name == "prefill";
            decode_count += name == "decode";
            EXPECT_EQ(span->GetStatus(), opentelemetry::trace::StatusCode::kError);
            const auto& attributes = span->GetAttributes();
            const auto found = attributes.find(telemetry::kAttrGenAiPdRole);
            EXPECT_NE(found, attributes.end());
            if (found != attributes.end()) {
                const auto* role = opentelemetry::nostd::get_if<std::string>(&found->second);
                EXPECT_NE(role, nullptr);
                if (role) {
                    EXPECT_EQ(*role, name == "prefill" ? "producer" : "consumer");
                }
            }
        }
    }
    EXPECT_EQ(server_count, 2u);
    EXPECT_EQ(prefill_count, 1u);
    EXPECT_EQ(decode_count, 1u);
}

TEST(RpcWriterCancellationTest, KvLoadTraceKeepsCompletedSuccessAndAttributesDependencyFailure) {
    namespace memory = opentelemetry::exporter::memory;
    telemetry::TelemetryRuntime::shutdown(5000);
    std::shared_ptr<memory::InMemorySpanData> span_data;
    auto exporter = memory::InMemorySpanExporterFactory::Create(span_data);
    telemetry::TelemetryConfig config;
    config.enabled = true;
    config.role = "test";
    ASSERT_TRUE(telemetry::TelemetryRuntime::initWithExporter(std::move(exporter), config));
    for (bool load_success : {false, true}) {
        kmonitor::MetricsReporterPtr reporter;
        auto meta = std::make_shared<RpcServerRuntimeMeta>();
        GenerateContext context(55, 0, nullptr, reporter, meta);
        auto stream = std::make_shared<SingleOutputStream>();
        stream->loading_cache_start_time_us_ = currentTimeUs() - 3000;
        if (load_success) {
            stream->recordLoadingCacheDoneTime();
            // Failure after transfer completion must not repaint load_cache.
            stream->reportError(ErrorCode::CANCELLED, "later generation cancellation");
            context.error_status = grpcStatusFromErrorInfo(stream->statusInfo());
        } else {
            FirstError dependency;
            dependency.record({ErrorCode::CONNECT_TIMEOUT, "Prefill dependency timed out"});
            context.dependency_failure = dependency.snapshot();
            stream->reportError(ErrorCode::CONNECT_TIMEOUT, "Prefill dependency timed out");
            stream->recordLoadingCacheDoneTime();
            context.error_status = grpcStatusFromErrorInfo(stream->statusInfo());
        }
        context.setStream(stream);
        auto span = telemetry::startRpcServerSpan("kv-load-test", nullptr, true);
        context.trace_span_guard =
            std::make_unique<telemetry::GrpcStatusSpanGuard>(span, &context.error_status);
        context.synthesizeTracePhases(telemetry::PhaseRole::Decode, false);
        context.markRpcHandlingCompleted();
    }
    EXPECT_TRUE(telemetry::TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    size_t successful_loads = 0;
    size_t failed_loads = 0;
    for (const auto& span : spans) {
        if (span->GetName() != "load_cache") {
            continue;
        }
        const auto& attrs = span->GetAttributes();
        EXPECT_EQ(attrs.count(telemetry::kAttrGenAiPdRole), 0u);
        if (span->GetStatus() == opentelemetry::trace::StatusCode::kOk) {
            ++successful_loads;
            EXPECT_EQ(attrs.count(telemetry::kAttrErrorType), 0u);
        } else {
            ++failed_loads;
            const auto type = attrs.find(telemetry::kAttrErrorType);
            ASSERT_NE(type, attrs.end());
            EXPECT_EQ(opentelemetry::nostd::get<std::string>(type->second), "DependencyFailure");
            const auto code = attrs.find(telemetry::kAttrRtpLlmErrorCode);
            ASSERT_NE(code, attrs.end());
            EXPECT_EQ(opentelemetry::nostd::get<int64_t>(code->second), static_cast<int64_t>(ErrorCode::CONNECT_TIMEOUT));
        }
    }
    EXPECT_EQ(successful_loads, 1u);
    EXPECT_EQ(failed_loads, 1u);
}

class ObservedLoadingStream: public SingleOutputStream {
public:
    std::atomic<StreamState> observed{StreamState::LOADING_CACHE};
    StreamState getStatus() const override {
        return observed.load();
    }
};

TEST(RpcWriterCancellationTest, StageMetricsSampleLiveLoadingStreamsAndExcludeCancelOverlays) {
    kmonitor::MetricsReporterPtr reporter;
    auto meta = std::make_shared<RpcServerRuntimeMeta>();
    auto first = std::make_shared<ObservedLoadingStream>();
    auto second = std::make_shared<ObservedLoadingStream>();
    meta->enqueue(56, first);
    meta->enqueue(57, second);
    meta->markCancellationPending(TaskIdentity{58, -1}, false);
    std::atomic<size_t> onflight{2};
    GenerateContext context(56, 0, nullptr, reporter, meta);
    context.onflight_requests = &onflight;
    RpcMetricsCollector stage;
    stage.onflight_request = 999;
    stage.loading_cache_request = 999;
    context.reportMetrics(stage);
    EXPECT_EQ(stage.onflight_request, 2);
    EXPECT_EQ(stage.loading_cache_request, 2);

    first->observed.store(StreamState::RUNNING);
    RpcMetricsCollector basic;
    context.collectBasicMetrics(basic);
    EXPECT_EQ(basic.loading_cache_request, 1);
    second->observed.store(StreamState::FINISHED);
    meta->dequeue(57, second);
    onflight.store(1);
    context.reportMetrics(stage);
    EXPECT_EQ(stage.onflight_request, 1);
    EXPECT_EQ(stage.loading_cache_request, 0);
    context.markRpcHandlingCompleted();
}

TEST(RpcWriterCancellationTest, PdValidationAndAdmissionErrorsEndServerSpanBeforeEngineAccess) {
    namespace memory = opentelemetry::exporter::memory;
    telemetry::TelemetryRuntime::shutdown(5000);
    std::shared_ptr<memory::InMemorySpanData> span_data;
    auto exporter = memory::InMemorySpanExporterFactory::Create(span_data);
    telemetry::TelemetryConfig config;
    config.enabled = true;
    config.role = "test";
    ASSERT_TRUE(telemetry::TelemetryRuntime::initWithExporter(std::move(exporter), config));
    {
        GenerateInputPB request;
        request.set_request_id(60);
        request.mutable_request_info()->set_trace_id("prefill-rejected-trace");
        request.add_token_ids(1);
        auto* generate = request.mutable_generate_config();
        generate->set_can_use_pd_separation(true);
        generate->set_max_new_tokens(8);
        generate->set_timeout_ms(1000);
        grpc::ServerContext context;
        PrefillRpcServer server;
        server.maga_init_params_.pd_sep_config.role_type = RoleType::PREFILL;
        // Missing handoff key must fail before the uninitialized engine is used.
        const auto status = server.GenerateStreamCall(&context, &request, nullptr);
        EXPECT_EQ(errorInfoFromGrpcStatus(status).code(), ErrorCode::INVALID_PARAMS);
        EXPECT_NE(status.error_message().find("trace_id=prefill-rejected-trace"), std::string::npos);
        EXPECT_EQ(server.onflight_requests_.load(), 0u);
        EXPECT_TRUE(server.onflight_trackers_.empty());
    }
    {
        GenerateInputPB request;
        request.set_request_id(61);
        request.mutable_request_info()->set_trace_id("decode-rejected-trace");
        request.set_enqueued_by_master(true);
        auto* generate = request.mutable_generate_config();
        generate->set_can_use_pd_separation(true);
        generate->set_max_new_tokens(8);
        generate->set_timeout_ms(1000);
        grpc::ServerContext context;
        DecodeRpcServer server;
        server.maga_init_params_.pd_sep_config.role_type = RoleType::DECODE;
        server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
        server.cancel_registry_ = std::make_unique<PDCancelRegistry>(server.meta_);
        EXPECT_EQ(server.cancel_registry_->cancel(61, {ErrorCode::GENERATE_TIMEOUT, "late admission"}),
                  CANCEL_STATUS_TOMBSTONED);
        const auto status = server.GenerateStreamCall(&context, &request, nullptr);
        EXPECT_EQ(errorInfoFromGrpcStatus(status).code(), ErrorCode::GENERATE_TIMEOUT);
        EXPECT_NE(status.error_message().find("trace_id=decode-rejected-trace"), std::string::npos);
        EXPECT_EQ(server.onflight_requests_.load(), 0u);
        EXPECT_TRUE(server.cancel_registry_->pending().empty());
    }
    EXPECT_TRUE(telemetry::TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    size_t server_count = 0;
    for (const auto& span : spans) {
        if (span->GetName() == "rtp_llm.generate_stream_call") {
            ++server_count;
            EXPECT_EQ(span->GetSpanKind(), opentelemetry::trace::SpanKind::kServer);
            EXPECT_EQ(span->GetStatus(), opentelemetry::trace::StatusCode::kError);
            EXPECT_EQ(span->GetAttributes().count(telemetry::kAttrErrorType), 1u);
        }
    }
    EXPECT_EQ(server_count, 2u);
}

TEST(RpcWriterCancellationTest, RecordedRpcFailureFeedsMetricsAndPreservesExistingStreamCause) {
    kmonitor::MetricsReporterPtr reporter;
    auto meta = std::make_shared<RpcServerRuntimeMeta>();
    GenerateContext context(62, 0, nullptr, reporter, meta);
    context.recordRpcFailure({ErrorCode::INVALID_PARAMS, "invalid input"});
    RpcMetricsCollector metrics;
    context.collectBasicMetrics(metrics);
    EXPECT_TRUE(metrics.error_qps);
    EXPECT_EQ(metrics.error_code, ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(errorInfoFromGrpcStatus(context.error_status).code(), ErrorCode::INVALID_PARAMS);
    context.recordRpcFailure({ErrorCode::EXECUTION_EXCEPTION, "later exception"});
    EXPECT_EQ(context.error_info.code(), ErrorCode::INVALID_PARAMS);
    context.markRpcHandlingCompleted();

    GenerateContext with_stream(63, 0, nullptr, reporter, meta);
    auto stream = std::make_shared<SingleOutputStream>();
    stream->reportError(ErrorCode::PRIORITY_PREEMPTED, "original stream failure");
    with_stream.setStream(stream);
    with_stream.recordRpcFailure({ErrorCode::EXECUTION_EXCEPTION, "later exception"});
    EXPECT_EQ(with_stream.error_info.code(), ErrorCode::PRIORITY_PREEMPTED);
    EXPECT_EQ(errorInfoFromGrpcStatus(with_stream.error_status).code(), ErrorCode::PRIORITY_PREEMPTED);
    with_stream.markRpcHandlingCompleted();
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
