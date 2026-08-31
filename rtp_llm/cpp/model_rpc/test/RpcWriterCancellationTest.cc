#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include "opentelemetry/exporters/memory/in_memory_span_data.h"
#include "opentelemetry/exporters/memory/in_memory_span_exporter_factory.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/telemetry/TelemetryRuntime.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"

namespace rtp_llm {
namespace {

namespace memory_exporter = opentelemetry::exporter::memory;
namespace nostd           = opentelemetry::nostd;

class RejectingWriter: public grpc::internal::WriterInterface<GenerateOutputsPB> {
public:
    bool Write(const GenerateOutputsPB&, grpc::WriteOptions) override {
        ++write_calls;
        return false;
    }

    int write_calls = 0;
};

class SingleOutputStream: public GenerateStream {
public:
    SingleOutputStream(): GenerateStream(makeInput(), makeModelConfig(), RuntimeConfig{}, ResourceContext{}, nullptr) {}

    ErrorResult<GenerateOutputs> nextOutput(int64_t /*wait_timeout_ms*/ = 0) override {
        GenerateOutputs outputs;
        GenerateOutput  output;
        output.output_ids = torch::ones({1, 1}, torch::kInt32);
        output.finished   = false;
        outputs.generate_outputs.push_back(std::move(output));
        return ErrorResult<GenerateOutputs>(std::move(outputs));
    }

    void updateOutput(const StreamUpdateInfo&) override {}

private:
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

class SingleResponseClientStream: public grpc::ClientReaderWriterInterface<GenerateRequestPB, GenerateOutputsPB> {
public:
    explicit SingleResponseClientStream(grpc::Status finish_status = grpc::Status(grpc::StatusCode::CANCELLED,
                                                                                  "cancelled by client")):
        finish_status_(std::move(finish_status)) {}

    bool Read(GenerateOutputsPB* response) override {
        ++read_calls;
        if (read_calls != 1) {
            return false;
        }
        response->mutable_flatten_output()->add_aux_info();
        return true;
    }

    bool NextMessageSize(uint32_t*) override {
        return false;
    }

    bool Write(const GenerateRequestPB&, grpc::WriteOptions) override {
        return true;
    }

    void WaitForInitialMetadata() override {}

    bool WritesDone() override {
        ++writes_done_calls;
        return true;
    }

    grpc::Status Finish() override {
        ++finish_calls;
        return finish_status_;
    }

    int read_calls        = 0;
    int writes_done_calls = 0;
    int finish_calls      = 0;

private:
    grpc::Status finish_status_;
};

class DecodeFirstReadService final: public RpcService::Service {
public:
    grpc::Status RemoteGenerate(grpc::ServerContext* context, ServerStream* stream) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            entered_ = true;
        }
        condition_.notify_all();

        auto status = decode_server_.RemoteGenerate(context, stream);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return_status_ = status;
        }
        condition_.notify_all();
        return status;
    }

    bool waitUntilEntered(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, timeout, [this] { return entered_; });
    }

    std::optional<grpc::Status> waitUntilReturned(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!condition_.wait_for(lock, timeout, [this] { return return_status_.has_value(); })) {
            return std::nullopt;
        }
        return return_status_;
    }

private:
    DecodeRpcServer             decode_server_;
    std::mutex                  mutex_;
    std::condition_variable     condition_;
    bool                        entered_{false};
    std::optional<grpc::Status> return_status_;
};

std::shared_ptr<memory_exporter::InMemorySpanData> startTelemetryForCloseTest() {
    telemetry::TelemetryRuntime::shutdown(5000);
    std::shared_ptr<memory_exporter::InMemorySpanData> span_data;
    auto                       exporter = memory_exporter::InMemorySpanExporterFactory::Create(span_data);
    telemetry::TelemetryConfig config;
    config.enabled = true;
    config.role    = "test";
    config.tp_rank = 0;
    EXPECT_TRUE(telemetry::TelemetryRuntime::initWithExporter(std::move(exporter), config));
    return span_data;
}

void expectClientSpanError(const std::shared_ptr<memory_exporter::InMemorySpanData>& span_data,
                           const std::string&                                        error_type,
                           const std::optional<std::string>&                         rpc_status = std::nullopt) {
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), 1u);
    ASSERT_NE(spans[0], nullptr);
    const auto& span = spans[0];
    EXPECT_EQ(span->GetStatus(), opentelemetry::trace::StatusCode::kError);
    const auto& attributes = span->GetAttributes();
    ASSERT_NE(attributes.find("error.type"), attributes.end());
    EXPECT_EQ(nostd::get<std::string>(attributes.at("error.type")), error_type);
    if (rpc_status.has_value()) {
        ASSERT_NE(attributes.find("rpc.response.status_code"), attributes.end());
        EXPECT_EQ(nostd::get<std::string>(attributes.at("rpc.response.status_code")), *rpc_status);
    }
}

void expectClientSpanOk(const std::shared_ptr<memory_exporter::InMemorySpanData>& span_data) {
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), 1u);
    ASSERT_NE(spans[0], nullptr);
    EXPECT_EQ(spans[0]->GetStatus(), opentelemetry::trace::StatusCode::kOk);
}

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

TEST(RpcWriterCancellationTest, DecodeFirstReadCancellationReturnsCancelled) {
    test::TestLogCapture   log_capture("decode_first_read_cancel");
    DecodeFirstReadService service;
    int                    listen_port = 0;
    grpc::ServerBuilder    builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &listen_port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    ASSERT_NE(server, nullptr);
    ASSERT_NE(listen_port, 0);

    auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(listen_port), grpc::InsecureChannelCredentials());
    auto stub    = RpcService::NewStub(channel);
    grpc::ClientContext client_context;
    client_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    auto stream = stub->RemoteGenerate(&client_context);
    ASSERT_NE(stream, nullptr);

    EXPECT_TRUE(service.waitUntilEntered(std::chrono::seconds(5)));
    client_context.TryCancel();

    const auto client_status = stream->Finish();
    const auto server_status = service.waitUntilReturned(std::chrono::seconds(5));

    server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));
    server->Wait();

    EXPECT_EQ(client_status.error_code(), grpc::StatusCode::CANCELLED);
    ASSERT_TRUE(server_status.has_value());
    EXPECT_EQ(server_status->error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_NE(log_capture.content().find("request [pending peer="), std::string::npos);
    EXPECT_NE(log_capture.content().find("read allocate request failed"), std::string::npos);
}

TEST(RpcWriterCancellationTest, RemoteWriteFailureCancelsGrpcStreamClosure) {
    PrefillRpcServer server;
    RejectingWriter  writer;
    GenerateInputPB  request;
    request.set_request_id(42);
    RPCContext                   rpc_context{&request, &writer};
    RemoteServerResource         resource;
    kmonitor::MetricsReporterPtr metrics_reporter;
    auto                         meta = std::make_shared<RpcServerRuntimeMeta>();
    PrefillGenerateContext       context(&resource, rpc_context, 0, nullptr, metrics_reporter, meta);
    context.stream_        = std::make_shared<SingleOutputStream>();
    context.client_context = std::make_shared<grpc::ClientContext>();
    auto client_stream     = std::make_shared<SingleResponseClientStream>();
    context.client_stream  = client_stream;

    server.pollRemoteOutput(context);

    EXPECT_EQ(client_stream->read_calls, 1);
    EXPECT_EQ(writer.write_calls, 1);
    EXPECT_TRUE(context.cancelled());
    EXPECT_EQ(context.error_status.error_code(), grpc::StatusCode::CANCELLED);
    const auto close_status = context.closeGrpcStream();
    EXPECT_EQ(close_status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(client_stream->writes_done_calls, 1);
    EXPECT_EQ(client_stream->finish_calls, 1);

    context.stream_.reset();
}

TEST(RpcWriterCancellationTest, CloseGrpcStreamIgnoresLateOverrideAndDoesNotRefinish) {
    auto                 span_data = startTelemetryForCloseTest();
    test::TestLogCapture log_capture("close_grpc_stream_late_override");
    {
        GenerateInputPB request;
        request.set_request_id(43);
        RPCContext                   rpc_context{&request, nullptr};
        RemoteServerResource         resource;
        kmonitor::MetricsReporterPtr metrics_reporter;
        PrefillGenerateContext       context(&resource, rpc_context, 0, nullptr, metrics_reporter, nullptr);
        auto                         client_stream =
            std::make_shared<SingleResponseClientStream>(grpc::Status(grpc::StatusCode::INTERNAL, "transport failure"));
        context.client_stream        = client_stream;
        context.pd_client_span_guard = std::make_unique<telemetry::RequestSpanGuard>(
            telemetry::TelemetryRuntime::tracer()->StartSpan("attempt_override"));

        EXPECT_EQ(context.closeGrpcStream().error_code(), grpc::StatusCode::INTERNAL);
        EXPECT_EQ(context.closeGrpcStream("REMOTE_GENERATE_FAILED").error_code(), grpc::StatusCode::INTERNAL);
        EXPECT_EQ(client_stream->finish_calls, 1);
    }
    EXPECT_EQ(log_capture.content().find("ignored attempt_error_override"), std::string::npos);
    ASSERT_TRUE(telemetry::TelemetryRuntime::shutdown(5000));
    expectClientSpanError(span_data, "Internal");
}

TEST(RpcWriterCancellationTest, CloseGrpcStreamFallsBackToTransportErrorOnSecondCall) {
    auto span_data = startTelemetryForCloseTest();
    {
        GenerateInputPB request;
        request.set_request_id(44);
        RPCContext                   rpc_context{&request, nullptr};
        RemoteServerResource         resource;
        kmonitor::MetricsReporterPtr metrics_reporter;
        PrefillGenerateContext       context(&resource, rpc_context, 0, nullptr, metrics_reporter, nullptr);
        context.client_stream =
            std::make_shared<SingleResponseClientStream>(grpc::Status(grpc::StatusCode::CANCELLED, "cancelled"));
        context.pd_client_span_guard = std::make_unique<telemetry::RequestSpanGuard>(
            telemetry::TelemetryRuntime::tracer()->StartSpan("attempt_transport"));

        EXPECT_EQ(context.closeGrpcStream().error_code(), grpc::StatusCode::CANCELLED);
    }
    ASSERT_TRUE(telemetry::TelemetryRuntime::shutdown(5000));
    expectClientSpanError(span_data, "Cancelled");
}

TEST(RpcWriterCancellationTest, CloseGrpcStreamIncludesInFlightAllocateTime) {
    auto span_data = startTelemetryForCloseTest();
    {
        GenerateInputPB request;
        request.set_request_id(47);
        RPCContext                   rpc_context{&request, nullptr};
        RemoteServerResource         resource;
        kmonitor::MetricsReporterPtr metrics_reporter;
        PrefillGenerateContext       context(&resource, rpc_context, 0, nullptr, metrics_reporter, nullptr);
        context.client_stream =
            std::make_shared<SingleResponseClientStream>(grpc::Status(grpc::StatusCode::INTERNAL, "allocate failed"));
        context.pd_client_span_guard = std::make_unique<telemetry::RequestSpanGuard>(
            telemetry::TelemetryRuntime::tracer()->StartSpan("attempt_allocate_failure"));
        context.stat_info.stage      = PrefillStatInfo::remoteAllocateResource;
        context.stat_info.begin_time = currentTimeUs() - 1000;

        EXPECT_EQ(context.closeGrpcStream().error_code(), grpc::StatusCode::INTERNAL);
    }
    ASSERT_TRUE(telemetry::TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), 1u);
    const auto& attributes  = spans[0]->GetAttributes();
    const auto  allocate_rt = attributes.find("rtp_llm.allocate_rt_us");
    ASSERT_NE(allocate_rt, attributes.end());
    EXPECT_GE(nostd::get<int64_t>(allocate_rt->second), 1000);
}

TEST(RpcWriterCancellationTest, CloseGrpcStreamMarksSuccessfulTransportAsOk) {
    auto span_data = startTelemetryForCloseTest();
    {
        GenerateInputPB request;
        request.set_request_id(45);
        RPCContext                   rpc_context{&request, nullptr};
        RemoteServerResource         resource;
        kmonitor::MetricsReporterPtr metrics_reporter;
        PrefillGenerateContext       context(&resource, rpc_context, 0, nullptr, metrics_reporter, nullptr);
        context.client_stream        = std::make_shared<SingleResponseClientStream>(grpc::Status::OK);
        context.pd_client_span_guard = std::make_unique<telemetry::RequestSpanGuard>(
            telemetry::TelemetryRuntime::tracer()->StartSpan("attempt_ok"));

        EXPECT_TRUE(context.closeGrpcStream().ok());
    }
    ASSERT_TRUE(telemetry::TelemetryRuntime::shutdown(5000));
    expectClientSpanOk(span_data);
}

TEST(RpcWriterCancellationTest, CloseGrpcStreamUsesOverrideOnFirstOkClose) {
    auto span_data = startTelemetryForCloseTest();
    {
        GenerateInputPB request;
        request.set_request_id(46);
        RPCContext                   rpc_context{&request, nullptr};
        RemoteServerResource         resource;
        kmonitor::MetricsReporterPtr metrics_reporter;
        PrefillGenerateContext       context(&resource, rpc_context, 0, nullptr, metrics_reporter, nullptr);
        context.client_stream        = std::make_shared<SingleResponseClientStream>(grpc::Status::OK);
        context.pd_client_span_guard = std::make_unique<telemetry::RequestSpanGuard>(
            telemetry::TelemetryRuntime::tracer()->StartSpan("attempt_override_first"));

        EXPECT_TRUE(context.closeGrpcStream("REMOTE_GENERATE_FAILED").ok());
    }
    ASSERT_TRUE(telemetry::TelemetryRuntime::shutdown(5000));
    expectClientSpanError(span_data, "REMOTE_GENERATE_FAILED");
}

TEST(RpcWriterCancellationTest, PriorityPreemptionOverridesOkAndCancelledTransportStatus) {
    const std::vector<std::pair<grpc::Status, std::string>> cases = {
        {grpc::Status::OK, "OK"},
        {grpc::Status(grpc::StatusCode::CANCELLED, "cancelled"), "CANCELLED"},
    };
    for (const auto& [finish_status, expected_rpc_status] : cases) {
        auto span_data = startTelemetryForCloseTest();
        {
            GenerateInputPB request;
            request.set_request_id(48);
            RPCContext                   rpc_context{&request, nullptr};
            RemoteServerResource         resource;
            kmonitor::MetricsReporterPtr metrics_reporter;
            PrefillGenerateContext       context(&resource, rpc_context, 0, nullptr, metrics_reporter, nullptr);
            context.client_stream        = std::make_shared<SingleResponseClientStream>(finish_status);
            context.pd_client_span_guard = std::make_unique<telemetry::RequestSpanGuard>(
                telemetry::TelemetryRuntime::tracer()->StartSpan("priority_preempt"));

            EXPECT_EQ(context.requestPriorityPreempt(), PriorityPreemptionRequestResult::INSTALLED);
            EXPECT_TRUE(context.finalizePriorityPreemption());
        }
        ASSERT_TRUE(telemetry::TelemetryRuntime::shutdown(5000));
        expectClientSpanError(span_data, "PRIORITY_PREEMPTED", expected_rpc_status);
    }
}

}  // namespace
}  // namespace rtp_llm
