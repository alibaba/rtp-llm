#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
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
        response->mutable_flatten_output()->add_finished(false);
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
    // Short BSP tick: with the 5000ms default equal to the shutdown deadline,
    // one lost worker wakeup burns the whole budget and flakes the test.
    config.schedule_delay_ms = 200;
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
    // The handler checks CHECK_REQUEST_CANCELLED before prepareGenerateContext,
    // so a cancel that lands too early short-circuits without ever attempting
    // the read this test is about. Both outcomes return CANCELLED, and the only
    // evidence distinguishing them is the warning logged after the failed read
    // (DecodeRpcServer.cc:236-241) -- nothing is logged before it, so there is no
    // marker to wait on. A fixed settle window is therefore a race: 500ms was
    // enough on an idle box and not enough on a loaded CI worker.
    //
    // Retry the scenario with a growing window and stop at the first attempt
    // that reaches the read path.
    static constexpr const char*    kReadFailureMarker = "read allocate request failed";
    const std::chrono::milliseconds settle_windows[]   = {
        std::chrono::milliseconds(500), std::chrono::milliseconds(1500), std::chrono::milliseconds(4000)};
    bool reached_read_path = false;

    for (const auto settle : settle_windows) {
        test::TestLogCapture   log_capture("decode_first_read_cancel");
        DecodeFirstReadService service;
        int                    listen_port = 0;
        grpc::ServerBuilder    builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &listen_port);
        builder.RegisterService(&service);
        auto server = builder.BuildAndStart();
        ASSERT_NE(server, nullptr);
        ASSERT_NE(listen_port, 0);

        auto channel =
            grpc::CreateChannel("127.0.0.1:" + std::to_string(listen_port), grpc::InsecureChannelCredentials());
        auto                stub = RpcService::NewStub(channel);
        grpc::ClientContext client_context;
        client_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));
        auto stream = stub->RemoteGenerate(&client_context);
        ASSERT_NE(stream, nullptr);

        EXPECT_TRUE(service.waitUntilEntered(std::chrono::seconds(5)));
        // waitUntilEntered fires before the fake delegates to the real handler,
        // so give the handler this long to get past the gate and park in Read
        // (the client sends nothing, so it parks there indefinitely).
        std::this_thread::sleep_for(settle);
        client_context.TryCancel();

        const auto client_status = stream->Finish();
        const auto server_status = service.waitUntilReturned(std::chrono::seconds(5));

        // Read the evidence before tearing the server down.
        reached_read_path = log_capture.waitFor(kReadFailureMarker, std::chrono::seconds(5));

        server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));
        server->Wait();

        // Holds on both paths; the cancel is observed either way.
        EXPECT_EQ(client_status.error_code(), grpc::StatusCode::CANCELLED);
        ASSERT_TRUE(server_status.has_value());
        EXPECT_EQ(server_status->error_code(), grpc::StatusCode::CANCELLED);
        EXPECT_EQ(log_capture.content().find("GenerateContext destroyed before RPC handling completed"),
                  std::string::npos);

        if (reached_read_path) {
            break;
        }
    }

    EXPECT_TRUE(reached_read_path) << "cancel never landed while the handler was blocked in Read, so the "
                                      "read-failure path was not exercised in any attempt";
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
    context.stream_        = std::make_shared<SingleOutputStream>(GenerationPrefillCudaGraphStatus::REPLAYED);
    context.client_context = std::make_shared<grpc::ClientContext>();
    auto client_stream     = std::make_shared<SingleResponseClientStream>();
    context.client_stream  = client_stream;

    server.pollRemoteOutput(context);

    EXPECT_EQ(client_stream->read_calls, 1);
    EXPECT_EQ(writer.write_calls, 1);
    ASSERT_EQ(writer.last_response.flatten_output().aux_info_size(), 1);
    EXPECT_EQ(writer.last_response.flatten_output().aux_info(0).generation_prefill_cuda_graph_status(), "replayed");
    EXPECT_TRUE(context.cancelled());
    EXPECT_EQ(context.error_status.error_code(), grpc::StatusCode::CANCELLED);
    const auto close_status = context.closeGrpcStream();
    EXPECT_EQ(close_status.error_code(), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(client_stream->writes_done_calls, 1);
    EXPECT_EQ(client_stream->finish_calls, 1);

    context.stream_.reset();
}

// These tests use real local gRPC calls: a closed client must interrupt the
// server even when neither the model nor Decode publishes another packet.
class ScopedCancellationRpcServer {
public:
    explicit ScopedCancellationRpcServer(grpc::Service* service) {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(service);
        server_ = builder.BuildAndStart();
        if (!server_ || port_ == 0) {
            throw std::runtime_error("failed to start cancellation test RPC server");
        }
    }

    ~ScopedCancellationRpcServer() {
        server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(1));
        server_->Wait();
    }

    std::shared_ptr<grpc::Channel> channel() const {
        return grpc::CreateChannel("127.0.0.1:" + std::to_string(port_), grpc::InsecureChannelCredentials());
    }

private:
    int                           port_ = 0;
    std::unique_ptr<grpc::Server> server_;
};

bool waitForCancellationFlag(const std::atomic<bool>& flag) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!flag.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return flag.load();
}

class WaitingDecodeRpcService: public RpcService::Service {
public:
    grpc::Status RemoteGenerate(grpc::ServerContext*                                            context,
                                grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB>* stream) override {
        GenerateRequestPB request;
        if (!stream->Read(&request) || !stream->Write(GenerateOutputsPB())) {
            return grpc::Status(grpc::StatusCode::CANCELLED, "allocation closed");
        }
        allocated.store(true);
        while (!context->IsCancelled() && !stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        cancelled.store(context->IsCancelled());
        return grpc::Status(grpc::StatusCode::CANCELLED, "decode cancelled");
    }

    std::atomic<bool> allocated{false};
    std::atomic<bool> cancelled{false};
    std::atomic<bool> stop{false};
};

class CompletingDecodeRpcService: public RpcService::Service {
public:
    grpc::Status RemoteGenerate(grpc::ServerContext*,
                                grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB>* stream) override {
        GenerateRequestPB request;
        if (!stream->Read(&request) || !stream->Write(GenerateOutputsPB())) {
            return grpc::Status(grpc::StatusCode::CANCELLED, "allocation closed");
        }
        GenerateOutputsPB response;
        response.set_request_id(request.request_id());
        response.mutable_flatten_output()->add_finished(true);
        response.mutable_flatten_output()->add_aux_info()->set_output_len(1);
        return stream->Write(response) ? grpc::Status::OK : grpc::Status(grpc::StatusCode::CANCELLED, "output closed");
    }
};

class WaitingPrefillRpcService: public RpcService::Service {
public:
    explicit WaitingPrefillRpcService(std::shared_ptr<grpc::Channel> channel, bool late_parent = false):
        channel_(std::move(channel)), late_parent_(late_parent) {}

    grpc::Status GenerateStreamCall(grpc::ServerContext*                   server_context,
                                    const GenerateInputPB*                 request,
                                    grpc::ServerWriter<GenerateOutputsPB>* writer) override {
        RPCContext             rpc_context{request, writer};
        auto                   meta = std::make_shared<RpcServerRuntimeMeta>();
        PrefillGenerateContext context(
            &server_.resource(), rpc_context, 10000, late_parent_ ? nullptr : server_context, reporter_, meta);
        context.grpc_connection.channel = channel_;
        context.grpc_connection.stub    = RpcService::NewStub(channel_);
        server_.remoteAllocateResource(context);
        if (late_parent_) {
            // EnqueueGroup creates the child before FetchResponse supplies its parent.
            context.server_context = server_context;
        }
        std::atomic_store(&child_context_, context.client_context);
        child_registered.store(true);
        if (!context.ok()) {
            returned.store(true);
            return context.error_status;
        }
        // Exercise the production forwarding loop, including blocked Read,
        // normal EOF, and cancellation after Fetch supplies its context.
        context.stream_ = std::make_shared<SingleOutputStream>();
        server_.pollRemoteOutput(context);
        const auto status = context.error_status;
        context.stream_.reset();
        returned.store(true);
        return status;
    }

    void cleanup() {
        auto child = std::atomic_load(&child_context_);
        if (child) {
            child->TryCancel();
        }
    }

    std::atomic<bool> returned{false};
    std::atomic<bool> child_registered{false};

private:
    PrefillRpcServer                     server_;
    kmonitor::MetricsReporterPtr         reporter_;
    std::shared_ptr<grpc::Channel>       channel_;
    std::shared_ptr<grpc::ClientContext> child_context_;
    bool                                 late_parent_ = false;
};

TEST(RpcWriterCancellationTest, ClientCancellationInterruptsBlockedDecodeRead) {
    WaitingDecodeRpcService     decode;
    ScopedCancellationRpcServer decode_server(&decode);
    WaitingPrefillRpcService    prefill(decode_server.channel());
    ScopedCancellationRpcServer prefill_server(&prefill);
    auto                        stub = RpcService::NewStub(prefill_server.channel());
    grpc::ClientContext         context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    GenerateInputPB request;
    request.set_request_id(44);
    request.mutable_generate_config()->set_timeout_ms(10000);
    auto reader = stub->GenerateStreamCall(&context, request);
    EXPECT_TRUE(waitForCancellationFlag(decode.allocated));
    EXPECT_TRUE(waitForCancellationFlag(prefill.child_registered));
    context.TryCancel();
    EXPECT_TRUE(waitForCancellationFlag(decode.cancelled)) << "upstream cancellation never reaches Decode";
    // Cleanup is separate from the assertion so the negative control cannot hang.
    prefill.cleanup();
    decode.stop.store(true);
    EXPECT_TRUE(waitForCancellationFlag(prefill.returned));
    EXPECT_EQ(reader->Finish().error_code(), grpc::StatusCode::CANCELLED);
}
TEST(RpcWriterCancellationTest, FetchCancellationInterruptsPreviouslyAllocatedDecodeRead) {
    WaitingDecodeRpcService     decode;
    ScopedCancellationRpcServer decode_server(&decode);
    WaitingPrefillRpcService    prefill(decode_server.channel(), true);
    ScopedCancellationRpcServer prefill_server(&prefill);
    auto                        stub = RpcService::NewStub(prefill_server.channel());
    grpc::ClientContext         context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    GenerateInputPB request;
    request.set_request_id(44);
    request.mutable_generate_config()->set_timeout_ms(10000);
    auto reader = stub->GenerateStreamCall(&context, request);
    EXPECT_TRUE(waitForCancellationFlag(decode.allocated));
    EXPECT_TRUE(waitForCancellationFlag(prefill.child_registered));
    context.TryCancel();
    EXPECT_TRUE(waitForCancellationFlag(decode.cancelled)) << "upstream cancellation never reaches Decode";
    // Cleanup is separate from the assertion so the negative control cannot hang.
    prefill.cleanup();
    decode.stop.store(true);
    EXPECT_TRUE(waitForCancellationFlag(prefill.returned));
    EXPECT_EQ(reader->Finish().error_code(), grpc::StatusCode::CANCELLED);
}

TEST(RpcWriterCancellationTest, AllocatedDecodeStreamCompletesNormallyAfterFetchBindsParent) {
    CompletingDecodeRpcService  decode;
    ScopedCancellationRpcServer decode_server(&decode);
    WaitingPrefillRpcService    prefill(decode_server.channel(), true);
    ScopedCancellationRpcServer prefill_server(&prefill);
    auto                        stub = RpcService::NewStub(prefill_server.channel());
    grpc::ClientContext         context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    GenerateInputPB request;
    request.set_request_id(45);
    request.mutable_generate_config()->set_timeout_ms(10000);
    auto              reader = stub->GenerateStreamCall(&context, request);
    GenerateOutputsPB response;
    int               frames = 0;
    while (reader->Read(&response)) {
        ++frames;
        EXPECT_EQ(response.request_id(), 45);
        ASSERT_EQ(response.flatten_output().finished_size(), 1);
        EXPECT_TRUE(response.flatten_output().finished(0));
        ASSERT_EQ(response.flatten_output().aux_info_size(), 1);
        EXPECT_TRUE(response.flatten_output().aux_info(0).pd_sep());
    }
    EXPECT_TRUE(reader->Finish().ok());
    EXPECT_EQ(frames, 1);
    EXPECT_TRUE(prefill.returned.load());
}

TEST(RpcWriterCancellationTest, FetchDeadlineInterruptsPreviouslyAllocatedDecodeRead) {
    WaitingDecodeRpcService     decode;
    ScopedCancellationRpcServer decode_server(&decode);
    WaitingPrefillRpcService    prefill(decode_server.channel(), true);
    ScopedCancellationRpcServer prefill_server(&prefill);
    auto                        stub = RpcService::NewStub(prefill_server.channel());
    grpc::ClientContext         context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(1));
    GenerateInputPB request;
    request.set_request_id(46);
    request.mutable_generate_config()->set_timeout_ms(10000);
    auto reader = stub->GenerateStreamCall(&context, request);
    EXPECT_TRUE(waitForCancellationFlag(decode.allocated));
    EXPECT_TRUE(waitForCancellationFlag(prefill.child_registered));
    EXPECT_EQ(reader->Finish().error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_TRUE(waitForCancellationFlag(decode.cancelled));
    prefill.cleanup();
    decode.stop.store(true);
    EXPECT_TRUE(waitForCancellationFlag(prefill.returned));
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

TEST(RpcWriterCancellationTest, DecodeStageSettlementIsIdempotent) {
    DecodeStatInfo stat_info;
    stat_info.stage      = DecodeStatInfo::allocateResource;
    stat_info.begin_time = currentTimeUs() - 1000;

    stat_info.finishStage();
    const auto recorded = stat_info.allocate_resource_rt_us;

    EXPECT_GE(recorded, 1000);
    EXPECT_EQ(stat_info.begin_time, 0);
    stat_info.finishStage();
    EXPECT_EQ(stat_info.allocate_resource_rt_us, recorded);
    EXPECT_EQ(stat_info.load_cache_from_prefill_rt_us, 0);
}

TEST(RpcWriterCancellationTest, PrefillStageSettlementIsIdempotent) {
    PrefillStatInfo stat_info;
    stat_info.stage      = PrefillStatInfo::remoteAllocateResource;
    stat_info.begin_time = currentTimeUs() - 1000;

    stat_info.finishStage();
    const auto recorded = stat_info.remote_allocate_resource_rt_us;

    EXPECT_GE(recorded, 1000);
    EXPECT_EQ(stat_info.begin_time, 0);
    stat_info.finishStage();
    EXPECT_EQ(stat_info.remote_allocate_resource_rt_us, recorded);
    EXPECT_EQ(stat_info.enqueue_request_rt_us, 0);
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
