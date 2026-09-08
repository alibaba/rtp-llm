#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "grpc++/grpc++.h"
#include "opentelemetry/exporters/memory/in_memory_span_data.h"
#include "opentelemetry/exporters/memory/in_memory_span_exporter_factory.h"
#include "opentelemetry/sdk/trace/span_data.h"
#include "opentelemetry/trace/context.h"

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/telemetry/GrpcTraceCarrier.h"
#include "rtp_llm/cpp/telemetry/RpcTraceHelper.h"
#include "rtp_llm/cpp/telemetry/TelemetryRuntime.h"

namespace rtp_llm {
namespace telemetry {

namespace trace_api       = opentelemetry::trace;
namespace trace_sdk       = opentelemetry::sdk::trace;
namespace memory_exporter = opentelemetry::exporter::memory;
namespace nostd           = opentelemetry::nostd;

namespace {

std::string toHex(const trace_api::TraceId& trace_id) {
    char buf[32];
    trace_id.ToLowerBase16(buf);
    return std::string(buf, 32);
}

std::string toHex(const trace_api::SpanId& span_id) {
    char buf[16];
    span_id.ToLowerBase16(buf);
    return std::string(buf, 16);
}

const trace_sdk::SpanData* findSpanByName(const std::vector<std::unique_ptr<trace_sdk::SpanData>>& spans,
                                          const std::string&                                       name) {
    for (const auto& span : spans) {
        if (span->GetName() == name) {
            return span.get();
        }
    }
    return nullptr;
}

// RAII guard so the client span is always ended on every exit path (including
// an ASSERT early-return), never leaked.
struct SpanEndGuard {
    nostd::shared_ptr<trace_api::Span> span;
    ~SpanEndGuard() {
        if (span) {
            span->End();
        }
    }
};

// Minimal RpcService whose handler runs the real production entry point
// startRpcServerSpan(): it builds the SERVER span from the remote parent in the
// gRPC metadata, so the exported span data reflects the actual propagation.
class TracingHealthService final: public RpcService::Service {
public:
    grpc::Status
    CheckHealth(grpc::ServerContext* context, const EmptyPB* /*request*/, CheckHealthResponsePB* response) override {
        auto span =
            startRpcServerSpan("rtp_llm.check_health", context, /*pd_separation=*/false, "RpcService/CheckHealth");
        if (span == nullptr) {
            response->set_health("inactive");
            return grpc::Status::OK;
        }
        const auto span_context = span->GetContext();
        response->set_health(span_context.IsValid() ? toHex(span_context.trace_id()) : "invalid");
        span->End();
        return grpc::Status::OK;
    }
};

class GrpcPropagationTest: public ::testing::Test {
protected:
    void SetUp() override {
        TelemetryRuntime::shutdown(5000);
        auto            exporter = memory_exporter::InMemorySpanExporterFactory::Create(span_data_);
        TelemetryConfig config;
        config.enabled = true;
        config.role    = "test";
        config.tp_rank = 0;
        ASSERT_TRUE(TelemetryRuntime::initWithExporter(std::move(exporter), config));

        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(&service_);
        server_ = builder.BuildAndStart();
        ASSERT_NE(server_, nullptr);
        ASSERT_GT(port_, 0);

        channel_ = grpc::CreateChannel("127.0.0.1:" + std::to_string(port_), grpc::InsecureChannelCredentials());
        stub_    = RpcService::NewStub(channel_);
    }

    void TearDown() override {
        if (server_) {
            // Bounded shutdown so a stuck server degrades to a fast failure
            // instead of a whole-target Bazel timeout.
            server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(5));
        }
        TelemetryRuntime::shutdown(5000);
    }

    std::shared_ptr<memory_exporter::InMemorySpanData> span_data_;
    TracingHealthService                               service_;
    int                                                port_ = 0;
    std::unique_ptr<grpc::Server>                      server_;
    std::shared_ptr<grpc::Channel>                     channel_;
    std::unique_ptr<RpcService::Stub>                  stub_;
};

TEST_F(GrpcPropagationTest, ServerSpanAdoptsRemoteParentAcrossRealTransport) {
    auto               tracer = TelemetryRuntime::tracer();
    trace_api::TraceId client_trace_id;
    trace_api::SpanId  client_span_id;
    {
        // End the client span BEFORE shutdown() (inner scope) so it is flushed
        // in this same export cycle instead of ending only at test-function
        // exit, after the BSP has already drained.
        SpanEndGuard client{tracer->StartSpan("client_root")};
        ASSERT_TRUE(client.span->GetContext().IsValid());
        client_trace_id = client.span->GetContext().trace_id();
        client_span_id  = client.span->GetContext().span_id();

        grpc::ClientContext client_context;
        client_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        injectSpanToClientContext(&client_context, client.span);

        EmptyPB               request;
        CheckHealthResponsePB response;
        const auto            status = stub_->CheckHealth(&client_context, request, &response);
        ASSERT_TRUE(status.ok()) << status.error_code() << ": " << status.error_message();
    }

    // Flush the BSP, then assert the exported SERVER span's parentage rather
    // than trusting the response payload alone.
    ASSERT_TRUE(TelemetryRuntime::shutdown(5000));
    auto        spans       = span_data_->GetSpans();
    const auto* server_span = findSpanByName(spans, "rtp_llm.check_health");
    ASSERT_NE(server_span, nullptr);
    EXPECT_EQ(toHex(server_span->GetTraceId()), toHex(client_trace_id));
    EXPECT_EQ(toHex(server_span->GetParentSpanId()), toHex(client_span_id));
    EXPECT_EQ(server_span->GetSpanKind(), trace_api::SpanKind::kServer);
    const auto& attrs = server_span->GetAttributes();
    ASSERT_NE(attrs.find("rpc.system"), attrs.end());
    EXPECT_EQ(nostd::get<std::string>(attrs.at("rpc.system")), "grpc");
    EXPECT_EQ(nostd::get<std::string>(attrs.at("rpc.method")), "RpcService/CheckHealth");
}

TEST_F(GrpcPropagationTest, ServerSpanStartsNewRootWithoutRemoteMetadata) {
    grpc::ClientContext client_context;
    client_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    EmptyPB               request;
    CheckHealthResponsePB response;
    const auto            status = stub_->CheckHealth(&client_context, request, &response);
    ASSERT_TRUE(status.ok()) << status.error_code() << ": " << status.error_message();

    ASSERT_TRUE(TelemetryRuntime::shutdown(5000));
    auto        spans       = span_data_->GetSpans();
    const auto* server_span = findSpanByName(spans, "rtp_llm.check_health");
    ASSERT_NE(server_span, nullptr);
    // No traceparent on the wire: start a fresh root, never adopt a bogus parent.
    EXPECT_FALSE(server_span->GetParentSpanId().IsValid());
    EXPECT_EQ(server_span->GetSpanKind(), trace_api::SpanKind::kServer);
}

TEST_F(GrpcPropagationTest, ServerSpanAdoptsForeignTraceparentHeader) {
    // Simulate a Python frontend / Java caller that writes the W3C header
    // directly, without going through the C++ propagator. The SERVER span must
    // still adopt it as its remote parent.
    const std::string trace_id_hex = "0123456789abcdef0123456789abcdef";
    const std::string span_id_hex  = "0123456789abcdef";

    grpc::ClientContext client_context;
    client_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
    client_context.AddMetadata("traceparent", "00-" + trace_id_hex + "-" + span_id_hex + "-01");

    EmptyPB               request;
    CheckHealthResponsePB response;
    const auto            status = stub_->CheckHealth(&client_context, request, &response);
    ASSERT_TRUE(status.ok()) << status.error_code() << ": " << status.error_message();

    ASSERT_TRUE(TelemetryRuntime::shutdown(5000));
    auto        spans       = span_data_->GetSpans();
    const auto* server_span = findSpanByName(spans, "rtp_llm.check_health");
    ASSERT_NE(server_span, nullptr);
    EXPECT_EQ(toHex(server_span->GetTraceId()), trace_id_hex);
    EXPECT_EQ(toHex(server_span->GetParentSpanId()), span_id_hex);
    EXPECT_EQ(server_span->GetSpanKind(), trace_api::SpanKind::kServer);
}

}  // namespace

}  // namespace telemetry
}  // namespace rtp_llm
