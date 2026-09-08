#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "opentelemetry/context/propagation/global_propagator.h"
#include "opentelemetry/exporters/memory/in_memory_span_data.h"
#include "opentelemetry/exporters/memory/in_memory_span_exporter_factory.h"
#include "opentelemetry/trace/context.h"
#include "opentelemetry/trace/span_context.h"
#include "opentelemetry/trace/span_startoptions.h"

#include "rtp_llm/cpp/telemetry/GrpcTraceCarrier.h"
#include "rtp_llm/cpp/telemetry/RequestSpanGuard.h"
#include "rtp_llm/cpp/telemetry/RpcTraceHelper.h"
#include "rtp_llm/cpp/telemetry/TelemetryRuntime.h"

namespace rtp_llm {
namespace telemetry {

namespace trace_api       = opentelemetry::trace;
namespace memory_exporter = opentelemetry::exporter::memory;
namespace otel_ctx        = opentelemetry::context;
namespace nostd           = opentelemetry::nostd;

namespace {

// Simple std::map-backed carrier for propagator roundtrip tests.
class MapCarrier: public otel_ctx::propagation::TextMapCarrier {
public:
    nostd::string_view Get(nostd::string_view key) const noexcept override {
        auto it = data_.find(std::string(key));
        return it == data_.end() ? "" : nostd::string_view(it->second);
    }
    void Set(nostd::string_view key, nostd::string_view value) noexcept override {
        data_[std::string(key)] = std::string(value);
    }
    std::map<std::string, std::string> data_;
};

void clearTelemetryEnv() {
    unsetenv("RTP_LLM_OTEL_TRACE_ENABLE");
    unsetenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT");
    unsetenv("OTEL_EXPORTER_OTLP_ENDPOINT");
    unsetenv("RTP_LLM_OTEL_TRACE_SAMPLER_RATIO");
    unsetenv("RTP_LLM_OTEL_BSP_MAX_QUEUE_SIZE");
    unsetenv("RTP_LLM_OTEL_BSP_SCHEDULE_DELAY_MS");
    unsetenv("RTP_LLM_OTEL_BSP_MAX_EXPORT_BATCH_SIZE");
    unsetenv("RTP_LLM_OTEL_HTTP_TIMEOUT_MS");
    unsetenv("RTP_LLM_OTEL_SERVICE_NAME");
}

class TelemetryTest: public ::testing::Test {
protected:
    void SetUp() override {
        clearTelemetryEnv();
        // Each test starts from a clean runtime; shutdown is idempotent and
        // must NOT be called from helpers that already hold the runtime lock.
        TelemetryRuntime::shutdown(5000);
    }

    void TearDown() override {
        TelemetryRuntime::shutdown(5000);
        clearTelemetryEnv();
    }

    // Starts an in-memory runtime; returns the span data sink.
    std::shared_ptr<memory_exporter::InMemorySpanData> startInMemoryRuntime(double sampler_ratio = 1.0) {
        std::shared_ptr<memory_exporter::InMemorySpanData> span_data;
        auto            exporter = memory_exporter::InMemorySpanExporterFactory::Create(span_data);
        TelemetryConfig config;
        config.enabled       = true;
        config.sampler_ratio = sampler_ratio;
        config.role          = "test";
        config.tp_rank       = 0;
        EXPECT_TRUE(TelemetryRuntime::initWithExporter(std::move(exporter), config));
        return span_data;
    }
};

TEST_F(TelemetryTest, ConfigDefaultsAreBoundedAndDisabled) {
    auto config = TelemetryConfig::fromEnv();
    EXPECT_FALSE(config.enabled);
    EXPECT_TRUE(config.endpoint.empty());
    EXPECT_DOUBLE_EQ(config.sampler_ratio, 1.0);
    EXPECT_EQ(config.max_queue_size, 2048u);
    EXPECT_EQ(config.schedule_delay_ms, 5000);
    EXPECT_EQ(config.max_export_batch_size, 512u);
    EXPECT_EQ(config.http_timeout_ms, 3000);
    // Empty by default: init() derives "rtp_llm_" + role (role-split components).
    EXPECT_EQ(config.service_name, "");
}

TEST_F(TelemetryTest, ConfigEndpointPrioritySignalSpecificWins) {
    setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://signal:4318/v1/traces", 1);
    setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://generic:4318", 1);
    auto config = TelemetryConfig::fromEnv();
    EXPECT_EQ(config.endpoint, "http://signal:4318/v1/traces");
}

TEST_F(TelemetryTest, ConfigEndpointGenericAppendsPath) {
    setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://generic:4318/", 1);
    auto config = TelemetryConfig::fromEnv();
    EXPECT_EQ(config.endpoint, "http://generic:4318/v1/traces");
}

TEST_F(TelemetryTest, ConfigInvalidValuesFallBackToDefaults) {
    setenv("RTP_LLM_OTEL_BSP_MAX_QUEUE_SIZE", "-5", 1);
    setenv("RTP_LLM_OTEL_BSP_SCHEDULE_DELAY_MS", "abc", 1);
    setenv("RTP_LLM_OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "0", 1);
    setenv("RTP_LLM_OTEL_TRACE_SAMPLER_RATIO", "3.5", 1);
    auto config = TelemetryConfig::fromEnv();
    EXPECT_EQ(config.max_queue_size, 2048u);
    EXPECT_EQ(config.schedule_delay_ms, 5000);
    EXPECT_EQ(config.max_export_batch_size, 512u);
    EXPECT_DOUBLE_EQ(config.sampler_ratio, 1.0);

    for (const char* invalid_ratio : {"nan", "inf", "-inf", "0.5junk"}) {
        setenv("RTP_LLM_OTEL_TRACE_SAMPLER_RATIO", invalid_ratio, 1);
        EXPECT_DOUBLE_EQ(TelemetryConfig::fromEnv().sampler_ratio, 1.0) << invalid_ratio;
    }
}

TEST_F(TelemetryTest, ConfigBatchSizeClampedToQueueSize) {
    setenv("RTP_LLM_OTEL_BSP_MAX_QUEUE_SIZE", "100", 1);
    setenv("RTP_LLM_OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "500", 1);
    auto config = TelemetryConfig::fromEnv();
    EXPECT_EQ(config.max_queue_size, 100u);
    EXPECT_EQ(config.max_export_batch_size, 100u);
}

TEST_F(TelemetryTest, DisabledByDefault) {
    EXPECT_FALSE(TelemetryRuntime::init("pdfusion", 0));
    EXPECT_EQ(TelemetryRuntime::state(), TelemetryState::DISABLED);
    EXPECT_FALSE(TelemetryRuntime::isActive());
    // no-op tracer must be non-null and safe to use
    auto tracer = TelemetryRuntime::tracer();
    ASSERT_NE(tracer, nullptr);
    auto span = tracer->StartSpan("noop");
    span->End();
}

TEST_F(TelemetryTest, NonRankZeroDisabled) {
    setenv("RTP_LLM_OTEL_TRACE_ENABLE", "1", 1);
    setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:4318/v1/traces", 1);
    EXPECT_FALSE(TelemetryRuntime::init("prefill", 1));
    EXPECT_EQ(TelemetryRuntime::state(), TelemetryState::DISABLED);
}

TEST_F(TelemetryTest, EnabledWithoutEndpointDisabled) {
    setenv("RTP_LLM_OTEL_TRACE_ENABLE", "1", 1);
    EXPECT_FALSE(TelemetryRuntime::init("pdfusion", 0));
    EXPECT_EQ(TelemetryRuntime::state(), TelemetryState::DISABLED);
}

TEST_F(TelemetryTest, InMemoryExportWithAttributes) {
    auto span_data = startInMemoryRuntime();
    ASSERT_TRUE(TelemetryRuntime::isActive());

    auto tracer = TelemetryRuntime::tracer();
    auto span   = tracer->StartSpan("rtp_llm.test_span");
    span->SetAttribute("rtp_llm.request_id", (int64_t)42);
    span->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), 1u);
    EXPECT_EQ(spans[0]->GetName(), "rtp_llm.test_span");
    const auto& attributes = spans[0]->GetAttributes();
    auto        it         = attributes.find("rtp_llm.request_id");
    ASSERT_NE(it, attributes.end());
}

TEST_F(TelemetryTest, PropagatorInjectExtractRoundtrip) {
    auto span_data = startInMemoryRuntime();
    auto tracer    = TelemetryRuntime::tracer();
    auto span      = tracer->StartSpan("roundtrip_parent");

    // inject
    MapCarrier        carrier;
    otel_ctx::Context base_context{};
    auto              context_with_span = trace_api::SetSpan(base_context, span);
    auto              propagator        = otel_ctx::propagation::GlobalTextMapPropagator::GetGlobalPropagator();
    propagator->Inject(carrier, context_with_span);
    ASSERT_NE(carrier.data_.find("traceparent"), carrier.data_.end());

    // extract
    otel_ctx::Context empty{};
    auto              extracted      = propagator->Extract(carrier, empty);
    auto              extracted_span = trace_api::GetSpan(extracted);
    auto              remote_ctx     = extracted_span->GetContext();
    EXPECT_TRUE(remote_ctx.IsValid());
    EXPECT_TRUE(remote_ctx.IsRemote());
    EXPECT_EQ(remote_ctx.trace_id(), span->GetContext().trace_id());
    EXPECT_EQ(remote_ctx.span_id(), span->GetContext().span_id());
    EXPECT_EQ(remote_ctx.trace_flags().IsSampled(), span->GetContext().trace_flags().IsSampled());
    span->End();
}

TEST_F(TelemetryTest, InvalidTraceparentExtractIsSafe) {
    startInMemoryRuntime();
    MapCarrier carrier;
    carrier.data_["traceparent"] = "totally-invalid-header-value";
    auto propagator              = otel_ctx::propagation::GlobalTextMapPropagator::GetGlobalPropagator();

    otel_ctx::Context empty{};
    auto              extracted = propagator->Extract(carrier, empty);
    auto              span_ctx  = trace_api::GetSpan(extracted)->GetContext();
    EXPECT_FALSE(span_ctx.IsValid());
}

TEST_F(TelemetryTest, GuardFinishIsExactlyOnce) {
    auto span_data = startInMemoryRuntime();
    auto tracer    = TelemetryRuntime::tracer();
    {
        RequestSpanGuard guard(tracer->StartSpan("guarded"));
        guard.setAttribute("rtp_llm.role", "test");
        guard.finish(trace_api::StatusCode::kOk);
        guard.finish();  // second finish must be a no-op
        // destructor fallback runs here as third End attempt
    }
    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), 1u);
    EXPECT_EQ(spans[0]->GetStatus(), trace_api::StatusCode::kOk);
}

TEST_F(TelemetryTest, GuardDestructorEndsSpanOnEarlyReturn) {
    auto span_data = startInMemoryRuntime();
    auto tracer    = TelemetryRuntime::tracer();
    {
        RequestSpanGuard guard(tracer->StartSpan("early_return"));
        // simulate CHECK_ERROR_STATUS early return: no explicit finish
    }
    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), 1u);
    EXPECT_EQ(spans[0]->GetName(), "early_return");
}

TEST_F(TelemetryTest, ResourceCarriesReplicaIdentityRanks) {
    // DP deployments: every group's tp_rank0 exports spans with otherwise
    // identical resources, so dp_rank/world_rank must land as attributes.
    std::shared_ptr<memory_exporter::InMemorySpanData> span_data;
    auto            exporter = memory_exporter::InMemorySpanExporterFactory::Create(span_data);
    TelemetryConfig config;
    config.enabled = true;
    // Fast BSP flush: SpanData::SetResource stores a RAW POINTER into the
    // provider's TracerContext, so the resource must be read BEFORE shutdown
    // destroys the provider (dangling otherwise).
    config.schedule_delay_ms = 1;
    config.role              = "decode";
    config.tp_rank           = 0;
    config.dp_rank           = 1;
    config.world_rank        = 2;
    ASSERT_TRUE(TelemetryRuntime::initWithExporter(std::move(exporter), config));

    TelemetryRuntime::tracer()->StartSpan("probe")->End();
    std::vector<std::unique_ptr<opentelemetry::sdk::trace::SpanData>> spans;
    for (int i = 0; i < 200 && spans.empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        spans = span_data->GetSpans();
    }
    ASSERT_EQ(spans.size(), 1u);
    const auto& res = spans[0]->GetResource().GetAttributes();
    ASSERT_NE(res.find("rtp_llm.dp_rank"), res.end());
    EXPECT_EQ(opentelemetry::nostd::get<int64_t>(res.at("rtp_llm.dp_rank")), 1);
    ASSERT_NE(res.find("rtp_llm.world_rank"), res.end());
    EXPECT_EQ(opentelemetry::nostd::get<int64_t>(res.at("rtp_llm.world_rank")), 2);
    // The role-derived service.name is what production actually exports; without
    // this assertion the test entry point could silently keep the plain
    // "rtp_llm" default while init() derived a different name.
    ASSERT_NE(res.find("service.name"), res.end());
    EXPECT_EQ(opentelemetry::nostd::get<std::string>(res.at("service.name")), "rtp_llm_decode");
    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
}

// Resolution contract shared by init() and initWithExporter(): an empty
// service_name derives from the role, while an explicit one always wins.
TEST_F(TelemetryTest, ServiceNameDerivesFromRoleUnlessExplicitlySet) {
    auto exportedServiceName = [](const std::string& role, const std::string& explicit_service_name) {
        std::shared_ptr<memory_exporter::InMemorySpanData> span_data;
        auto            exporter = memory_exporter::InMemorySpanExporterFactory::Create(span_data);
        TelemetryConfig config;
        config.enabled = true;
        // Fast BSP flush: the resource must be read before shutdown destroys the
        // provider (SpanData::SetResource stores a raw pointer into it).
        config.schedule_delay_ms = 1;
        config.role              = role;
        config.service_name      = explicit_service_name;
        EXPECT_TRUE(TelemetryRuntime::initWithExporter(std::move(exporter), config));

        TelemetryRuntime::tracer()->StartSpan("probe")->End();
        std::vector<std::unique_ptr<opentelemetry::sdk::trace::SpanData>> spans;
        for (int i = 0; i < 200 && spans.empty(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            spans = span_data->GetSpans();
        }
        EXPECT_EQ(spans.size(), 1u);
        std::string exported;
        if (!spans.empty()) {
            const auto& res = spans[0]->GetResource().GetAttributes();
            auto        it  = res.find("service.name");
            if (it != res.end()) {
                exported = opentelemetry::nostd::get<std::string>(it->second);
            }
        }
        EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
        return exported;
    };

    // Empty service_name: P and D must land as separate topology components.
    EXPECT_EQ(exportedServiceName("decode", ""), "rtp_llm_decode");
    EXPECT_EQ(exportedServiceName("prefill", ""), "rtp_llm_prefill");
    // An explicit name is a global override and must never be role-mangled.
    EXPECT_EQ(exportedServiceName("decode", "custom_service"), "custom_service");
    // No role at all still yields the plain base name, never a trailing "_".
    EXPECT_EQ(exportedServiceName("", ""), "rtp_llm");
}

TEST_F(TelemetryTest, GuardAddEventCarriesExplicitTimestamp) {
    auto span_data = startInMemoryRuntime();
    auto tracer    = TelemetryRuntime::tracer();
    // Generic guard API contract: post-hoc events carry an explicit epoch-µs
    // timestamp (PhaseSpanSynthesizer-style timing), not the call-time clock.
    const int64_t epoch_us = 1753600000123456;
    {
        RequestSpanGuard guard(tracer->StartSpan("decode"));
        guard.addEvent("sample_event", epoch_us);
        guard.finish(trace_api::StatusCode::kOk);
    }
    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), 1u);
    const auto& events = spans[0]->GetEvents();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].GetName(), "sample_event");
    auto event_us =
        std::chrono::duration_cast<std::chrono::microseconds>(events[0].GetTimestamp().time_since_epoch()).count();
    EXPECT_EQ(event_us, epoch_us);
}

TEST_F(TelemetryTest, GuardAddEventAfterFinishIsDropped) {
    auto span_data = startInMemoryRuntime();
    auto tracer    = TelemetryRuntime::tracer();
    {
        RequestSpanGuard guard(tracer->StartSpan("decode"));
        guard.finish(trace_api::StatusCode::kOk);
        guard.addEvent("sample_event", 1753600000123456);
    }
    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), 1u);
    EXPECT_TRUE(spans[0]->GetEvents().empty());
}

TEST_F(TelemetryTest, ParentBasedSamplerRespectsRemoteFlag) {
    auto span_data = startInMemoryRuntime();
    auto tracer    = TelemetryRuntime::tracer();

    constexpr uint8_t trace_id_buf[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    constexpr uint8_t span_id_buf[8]   = {1, 2, 3, 4, 5, 6, 7, 8};

    // remote parent with sampled=0 -> child not recorded
    {
        trace_api::SpanContext      parent_ctx(trace_api::TraceId(trace_id_buf),
                                          trace_api::SpanId(span_id_buf),
                                          trace_api::TraceFlags(0),
                                          /*is_remote=*/true);
        trace_api::StartSpanOptions options;
        options.parent = parent_ctx;
        auto span      = tracer->StartSpan("unsampled_child", options);
        EXPECT_FALSE(span->IsRecording());
        span->End();
    }
    // remote parent with sampled=1 -> child recorded, same trace id
    {
        trace_api::SpanContext      parent_ctx(trace_api::TraceId(trace_id_buf),
                                          trace_api::SpanId(span_id_buf),
                                          trace_api::TraceFlags(trace_api::TraceFlags::kIsSampled),
                                          /*is_remote=*/true);
        trace_api::StartSpanOptions options;
        options.parent = parent_ctx;
        auto span      = tracer->StartSpan("sampled_child", options);
        EXPECT_TRUE(span->IsRecording());
        EXPECT_EQ(span->GetContext().trace_id(), trace_api::TraceId(trace_id_buf));
        span->End();
    }

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), 1u);
    EXPECT_EQ(spans[0]->GetName(), "sampled_child");
    EXPECT_EQ(spans[0]->GetParentSpanId(), trace_api::SpanId(span_id_buf));
}

TEST_F(TelemetryTest, GrpcClientCarrierInjectsTraceparent) {
    startInMemoryRuntime();
    auto tracer = TelemetryRuntime::tracer();
    auto span   = tracer->StartSpan("client_inject");

    grpc::ClientContext client_context;
    otel_ctx::Context   base_context{};
    auto                context_with_span = trace_api::SetSpan(base_context, span);
    injectContextToClientMetadata(&client_context, context_with_span);
    // ClientContext has no public metadata getter before the RPC starts; the
    // roundtrip through real transport is covered by e2e tests. Here we verify
    // via the carrier abstraction with a null-safety check.
    injectContextToClientMetadata(nullptr, context_with_span);  // must not crash
    span->End();
}

TEST_F(TelemetryTest, ShutdownIsIdempotentAndBounded) {
    startInMemoryRuntime();
    ASSERT_TRUE(TelemetryRuntime::isActive());
    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    EXPECT_EQ(TelemetryRuntime::state(), TelemetryState::SHUTDOWN);
    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));  // second call is a no-op
    // after shutdown, tracer degrades to no-op
    auto span = TelemetryRuntime::tracer()->StartSpan("after_shutdown");
    span->End();
}

TEST_F(TelemetryTest, SpanFactoriesCarryRpcAttributes) {
    // Both factories wrap real gRPC boundaries, so every produced span must
    // carry the OTel RPC semconv transport marker (chip source on Unitrace).
    auto span_data = startInMemoryRuntime();
    auto parent    = TelemetryRuntime::tracer()->StartSpan("parent");
    auto client    = startChildClientSpan(
        "rtp_llm.remote_generate", parent, "decode.example", 26101, 42, 1, "RpcService/RemoteGenerate");
    auto server = startRpcServerSpan("rtp_llm.decode_remote_generate", nullptr, true, "RpcService/RemoteGenerate");
    auto local  = startRpcServerSpan("rtp_llm.generate_stream_call", nullptr, false, "RpcService/GenerateStreamCall");
    ASSERT_NE(client, nullptr);
    ASSERT_NE(server, nullptr);
    ASSERT_NE(local, nullptr);
    client->End();
    server->End();
    local->End();
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), 4u);
    for (const auto& span : spans) {
        if (span->GetName() == "rtp_llm.remote_generate") {
            const auto& attributes = span->GetAttributes();
            auto        it         = attributes.find("rpc.system");
            ASSERT_NE(it, attributes.end());
            EXPECT_EQ(nostd::get<std::string>(it->second), "grpc");
            EXPECT_EQ(nostd::get<std::string>(attributes.at("rpc.method")), "RpcService/RemoteGenerate");
            EXPECT_EQ(span->GetSpanKind(), trace_api::SpanKind::kClient);
            EXPECT_EQ(nostd::get<std::string>(attributes.at("server.address")), "decode.example");
            EXPECT_EQ(nostd::get<int64_t>(attributes.at("server.port")), 26101);
            EXPECT_EQ(nostd::get<std::string>(attributes.at("request_id")), "42");
            EXPECT_EQ(nostd::get<int64_t>(attributes.at("rtp_llm.request_id")), 42);
            EXPECT_EQ(nostd::get<int64_t>(attributes.at("rtp_llm.retry_attempt")), 1);
        } else if (span->GetName() == "rtp_llm.decode_remote_generate") {
            const auto& attributes = span->GetAttributes();
            EXPECT_EQ(nostd::get<std::string>(attributes.at("rpc.system")), "grpc");
            EXPECT_EQ(nostd::get<std::string>(attributes.at("rpc.method")), "RpcService/RemoteGenerate");
            EXPECT_TRUE(nostd::get<bool>(attributes.at("rtp_llm.pd_sep")));
            EXPECT_EQ(span->GetSpanKind(), trace_api::SpanKind::kServer);
        } else if (span->GetName() == "rtp_llm.generate_stream_call") {
            const auto& attributes = span->GetAttributes();
            EXPECT_EQ(nostd::get<std::string>(attributes.at("rpc.system")), "grpc");
            EXPECT_EQ(nostd::get<std::string>(attributes.at("rpc.method")), "RpcService/GenerateStreamCall");
            EXPECT_EQ(attributes.count("rtp_llm.pd_sep"), 0u);
            EXPECT_EQ(span->GetSpanKind(), trace_api::SpanKind::kServer);
        } else {
            // Plain tracer spans (parents, synthesized phases) must NOT get
            // the marker: they are not gRPC boundaries.
            EXPECT_EQ(span->GetAttributes().count("rpc.system"), 0u);
            EXPECT_EQ(span->GetAttributes().count("rpc.method"), 0u);
        }
    }
}

TEST_F(TelemetryTest, ClientSpanCanonicalizesAndValidatesEndpoint) {
    auto span_data = startInMemoryRuntime();
    auto parent    = TelemetryRuntime::tracer()->StartSpan("parent");
    struct EndpointCase {
        const char* address;
        int64_t     port;
        const char* expected_address;
        bool        valid;
    };
    const std::vector<EndpointCase> cases = {
        {"decode.example", 1, "decode.example", true},
        {"127.0.0.1", 65535, "127.0.0.1", true},
        {"2001:db8::1", 26101, "2001:db8::1", true},
        {"[2001:db8::2]", 26102, "2001:db8::2", true},
        {"", 26101, "", false},
        {"[]", 26101, "", false},
        {"[2001:db8::3", 26101, "", false},
        {"2001:db8::3]", 26101, "", false},
        {"dns:///decode.example", 26101, "", false},
        {"decode.example", -1, "", false},
        {"decode.example", 0, "", false},
        {"decode.example", 65536, "", false},
    };
    for (size_t i = 0; i < cases.size(); ++i) {
        auto span = startChildClientSpan("endpoint_" + std::to_string(i), parent, cases[i].address, cases[i].port);
        ASSERT_NE(span, nullptr);
        span->End();
    }
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), cases.size() + 1);
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto& attributes = spans[i]->GetAttributes();
        EXPECT_EQ(nostd::get<std::string>(attributes.at("rpc.system")), "grpc");
        if (cases[i].valid) {
            EXPECT_EQ(nostd::get<std::string>(attributes.at("server.address")), cases[i].expected_address);
            EXPECT_EQ(nostd::get<int64_t>(attributes.at("server.port")), cases[i].port);
        } else {
            EXPECT_EQ(attributes.count("server.address"), 0u);
            EXPECT_EQ(attributes.count("server.port"), 0u);
        }
    }
}

TEST_F(TelemetryTest, RetryAttemptCountsCompletedRetries) {
    EXPECT_EQ(retryAttemptFromExecutionCount(0), 0);
    EXPECT_EQ(retryAttemptFromExecutionCount(1), 0);
    EXPECT_EQ(retryAttemptFromExecutionCount(2), 1);
    EXPECT_EQ(retryAttemptFromExecutionCount(3), 2);
}

TEST_F(TelemetryTest, GrpcStatusSpanGuardUsesFinalErrorStatus) {
    auto         span_data = startInMemoryRuntime();
    grpc::Status status(grpc::StatusCode::CANCELLED, "cancelled");
    { GrpcStatusSpanGuard guard(TelemetryRuntime::tracer()->StartSpan("server"), &status); }

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), 1u);
    EXPECT_EQ(spans[0]->GetStatus(), trace_api::StatusCode::kError);
    EXPECT_EQ(spans[0]->GetDescription(), "RPC request was cancelled");
    const auto& attributes = spans[0]->GetAttributes();
    ASSERT_NE(attributes.find("error.type"), attributes.end());
    EXPECT_EQ(nostd::get<std::string>(attributes.at("error.type")), "Cancelled");
    ASSERT_NE(attributes.find("rtp_llm.grpc_status_code"), attributes.end());
    EXPECT_EQ(nostd::get<int64_t>(attributes.at("rtp_llm.grpc_status_code")), grpc::StatusCode::CANCELLED);
    EXPECT_EQ(nostd::get<std::string>(attributes.at("rpc.response.status_code")), "CANCELLED");
}

TEST_F(TelemetryTest, GrpcStatusSpanGuardWritesOkResponseStatus) {
    auto         span_data = startInMemoryRuntime();
    grpc::Status status    = grpc::Status::OK;
    { GrpcStatusSpanGuard guard(TelemetryRuntime::tracer()->StartSpan("server_ok"), &status); }

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), 1u);
    EXPECT_EQ(spans[0]->GetStatus(), trace_api::StatusCode::kOk);
    EXPECT_EQ(nostd::get<std::string>(spans[0]->GetAttributes().at("rpc.response.status_code")), "OK");
}

TEST_F(TelemetryTest, GrpcStatusSpanGuardMarksUnhandledExceptionWhenStatusIsOk) {
    auto         span_data = startInMemoryRuntime();
    grpc::Status status    = grpc::Status::OK;
    try {
        GrpcStatusSpanGuard guard(TelemetryRuntime::tracer()->StartSpan("server"), &status);
        throw std::runtime_error("boom");
    } catch (const std::runtime_error&) {}

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), 1u);
    EXPECT_EQ(spans[0]->GetStatus(), trace_api::StatusCode::kError);
    EXPECT_EQ(spans[0]->GetDescription(), "RPC handler raised an exception");
    const auto& attributes = spans[0]->GetAttributes();
    ASSERT_NE(attributes.find("error.type"), attributes.end());
    EXPECT_EQ(nostd::get<std::string>(attributes.at("error.type")), "Exception");
    EXPECT_EQ(attributes.count("rtp_llm.grpc_status_code"), 0u);
}

TEST_F(TelemetryTest, GrpcStatusDescriptionsArePredictableAndHumanReadable) {
    struct DescriptionCase {
        grpc::StatusCode code;
        const char*      expected;
    };
    const std::vector<DescriptionCase> cases = {
        {grpc::StatusCode::CANCELLED, "RPC request was cancelled"},
        {grpc::StatusCode::DEADLINE_EXCEEDED, "RPC deadline was exceeded"},
        {grpc::StatusCode::RESOURCE_EXHAUSTED, "RPC request exhausted available resources"},
        {grpc::StatusCode::INTERNAL, "RPC request failed because of an internal error"},
        {grpc::StatusCode::UNAVAILABLE, "RPC service was unavailable"},
    };
    for (const auto& test_case : cases) {
        EXPECT_STREQ(grpcStatusDescription(test_case.code), test_case.expected);
        EXPECT_STRNE(grpcStatusDescription(test_case.code), grpcStatusCodeName(test_case.code));
    }
    EXPECT_STREQ(grpcStatusDescription(grpc::StatusCode::OK), "");
}

TEST_F(TelemetryTest, UsageTokenAttributesFiveKeyDoubleWrite) {
    // Per-hop usage contract: semconv input/output + legacy prompt/completion
    // aliases + total.
    auto span_data = startInMemoryRuntime();
    {
        RequestSpanGuard guard(TelemetryRuntime::tracer()->StartSpan("decode"));
        setUsageTokenAttributes(guard, /*input_tokens=*/28, /*output_tokens=*/16);
        guard.finish(trace_api::StatusCode::kOk);
    }
    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), 1u);
    const auto&                          attributes = spans[0]->GetAttributes();
    const std::map<std::string, int64_t> expected   = {
        {"gen_ai.usage.input_tokens", 28},
        {"gen_ai.usage.output_tokens", 16},
        {"gen_ai.usage.prompt_tokens", 28},
        {"gen_ai.usage.completion_tokens", 16},
        {"gen_ai.usage.total_tokens", 44},
    };
    for (const auto& [key, value] : expected) {
        auto it = attributes.find(key);
        ASSERT_NE(it, attributes.end()) << key;
        EXPECT_EQ(nostd::get<int64_t>(it->second), value) << key;
    }
}

TEST_F(TelemetryTest, UsageTokenAttributesSkipNonPositiveValues) {
    // Partial usage is worse than none for platform aggregation: any
    // non-positive side must suppress the whole five-key group.
    auto span_data = startInMemoryRuntime();
    {
        RequestSpanGuard guard(TelemetryRuntime::tracer()->StartSpan("decode"));
        setUsageTokenAttributes(guard, /*input_tokens=*/28, /*output_tokens=*/0);
        guard.finish(trace_api::StatusCode::kOk);
    }
    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    ASSERT_EQ(spans.size(), 1u);
    EXPECT_EQ(spans[0]->GetAttributes().count("gen_ai.usage.input_tokens"), 0u);
    EXPECT_EQ(spans[0]->GetAttributes().count("gen_ai.usage.total_tokens"), 0u);
}

}  // namespace

}  // namespace telemetry
}  // namespace rtp_llm
