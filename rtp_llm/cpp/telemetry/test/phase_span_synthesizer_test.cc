#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "gtest/gtest.h"

#include "opentelemetry/exporters/memory/in_memory_span_data.h"
#include "opentelemetry/exporters/memory/in_memory_span_exporter_factory.h"
#include "opentelemetry/trace/span_context.h"

#include "rtp_llm/cpp/telemetry/PhaseSpanSynthesizer.h"
#include "rtp_llm/cpp/telemetry/TelemetryRuntime.h"

namespace rtp_llm {
namespace telemetry {

namespace trace_api       = opentelemetry::trace;
namespace memory_exporter = opentelemetry::exporter::memory;
namespace nostd           = opentelemetry::nostd;

namespace {

class PhaseSpanSynthesizerTest: public ::testing::Test {
protected:
    void SetUp() override {
        TelemetryRuntime::shutdown(5000);
        auto            exporter = memory_exporter::InMemorySpanExporterFactory::Create(span_data_);
        TelemetryConfig config;
        config.enabled = true;
        config.role    = "test";
        config.tp_rank = 0;
        ASSERT_TRUE(TelemetryRuntime::initWithExporter(std::move(exporter), config));
    }

    void TearDown() override {
        TelemetryRuntime::shutdown(5000);
    }

    // Creates a parent SERVER span and returns it.
    nostd::shared_ptr<trace_api::Span> createParentSpan(const std::string& name) {
        auto                        tracer = TelemetryRuntime::tracer();
        trace_api::StartSpanOptions options;
        options.kind = trace_api::SpanKind::kServer;
        return tracer->StartSpan(name, options);
    }

    // Finds a span by name in the exported span data.
    const opentelemetry::sdk::trace::SpanData*
    findSpan(const std::vector<std::unique_ptr<opentelemetry::sdk::trace::SpanData>>& spans, const std::string& name) {
        for (const auto& s : spans) {
            if (s->GetName() == name) {
                return s.get();
            }
        }
        return nullptr;
    }

    // Returns the string value of a span attribute, or "" when absent.
    std::string getStringAttribute(const opentelemetry::sdk::trace::SpanData* span, const std::string& key) {
        const auto& attrs = span->GetAttributes();
        auto        it    = attrs.find(key);
        if (it == attrs.end()) {
            return "";
        }
        if (const auto* value = opentelemetry::nostd::get_if<std::string>(&it->second)) {
            return *value;
        }
        return "";
    }

    bool getBoolAttribute(const opentelemetry::sdk::trace::SpanData* span, const std::string& key) {
        const auto& attrs = span->GetAttributes();
        auto        it    = attrs.find(key);
        return it != attrs.end() && opentelemetry::nostd::get<bool>(it->second);
    }

    int64_t getInt64Attribute(const opentelemetry::sdk::trace::SpanData* span, const std::string& key) {
        const auto& attrs = span->GetAttributes();
        auto        it    = attrs.find(key);
        return it == attrs.end() ? -1 : opentelemetry::nostd::get<int64_t>(it->second);
    }

    std::shared_ptr<memory_exporter::InMemorySpanData> span_data_;
};

PhaseTiming completedTiming(
    int64_t begin, int64_t running, int64_t first_token, int64_t done, int64_t synthesis_end, int64_t request_id = -1) {
    PhaseTiming timing;
    timing.begin_time_us           = begin;
    timing.running_started         = true;
    timing.running_started_time_us = running;
    timing.first_token_committed   = true;
    timing.first_token_time_us     = first_token;
    timing.generation_done         = true;
    timing.generation_done_time_us = done;
    timing.synthesis_end_time_us   = synthesis_end;
    timing.request_id              = request_id;
    return timing;
}

TEST_F(PhaseSpanSynthesizerTest, FusionModeProducesThreeChildSpans) {
    auto parent          = createParentSpan("rtp_llm.generate_stream_call");
    auto parent_span_id  = parent->GetContext().span_id();
    auto parent_trace_id = parent->GetContext().trace_id();

    auto timing = completedTiming(1000000, 1005000, 1020000, 1100000, 1105000, 42);

    synthesizePhaseSpans(parent, timing, PhaseRole::Fusion, /*request_ok=*/true);
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();

    // 1 parent + 3 children = 4 spans
    ASSERT_EQ(spans.size(), 4u);

    // Verify wait span
    auto* wait = findSpan(spans, "wait");
    ASSERT_NE(wait, nullptr);
    EXPECT_EQ(wait->GetSpanKind(), trace_api::SpanKind::kInternal);
    EXPECT_EQ(wait->GetParentSpanId(), parent_span_id);
    EXPECT_EQ(wait->GetTraceId(), parent_trace_id);
    // start = begin_time = 1000000us = 1s epoch
    auto wait_start_ns = wait->GetStartTime().time_since_epoch().count();
    EXPECT_EQ(wait_start_ns, int64_t(1000000) * 1000);  // µs -> ns
    // duration = 5000us = 5ms
    EXPECT_EQ(wait->GetDuration().count(), int64_t(5000) * 1000);  // µs -> ns
    // Bailian index key: string request_id on every synthesized child span
    EXPECT_EQ(getStringAttribute(wait, "request_id"), "42");
    EXPECT_EQ(getInt64Attribute(wait, "rtp_llm.request_id"), 42);
    // Completed phases carry explicit OK rather than Unset.
    EXPECT_EQ(wait->GetStatus(), trace_api::StatusCode::kOk);

    // Verify prefill span
    auto* prefill = findSpan(spans, "prefill");
    ASSERT_NE(prefill, nullptr);
    EXPECT_EQ(prefill->GetParentSpanId(), parent_span_id);
    // start = begin + wait = 1005000us
    auto prefill_start_ns = prefill->GetStartTime().time_since_epoch().count();
    EXPECT_EQ(prefill_start_ns, int64_t(1005000) * 1000);
    // duration = ttft - wait = 20000 - 5000 = 15000us
    EXPECT_EQ(prefill->GetDuration().count(), int64_t(15000) * 1000);
    EXPECT_EQ(getStringAttribute(prefill, "request_id"), "42");
    EXPECT_EQ(getInt64Attribute(prefill, "rtp_llm.request_id"), 42);
    EXPECT_EQ(prefill->GetStatus(), trace_api::StatusCode::kOk);

    // Verify decode span
    auto* decode = findSpan(spans, "decode");
    ASSERT_NE(decode, nullptr);
    EXPECT_EQ(decode->GetParentSpanId(), parent_span_id);
    // start = first_token_time = 1020000us
    auto decode_start_ns = decode->GetStartTime().time_since_epoch().count();
    EXPECT_EQ(decode_start_ns, int64_t(1020000) * 1000);
    // duration = cost - ttft = 100000 - 20000 = 80000us
    EXPECT_EQ(decode->GetDuration().count(), int64_t(80000) * 1000);
    EXPECT_EQ(getStringAttribute(decode, "request_id"), "42");
    EXPECT_EQ(getInt64Attribute(decode, "rtp_llm.request_id"), 42);
    EXPECT_EQ(decode->GetStatus(), trace_api::StatusCode::kOk);
}

TEST_F(PhaseSpanSynthesizerTest, NegativeRequestIdSkipsAttribute) {
    auto parent = createParentSpan("parent");

    auto timing = completedTiming(1000000, 1005000, 1020000, 1100000, 1105000);

    synthesizePhaseSpans(parent, timing, PhaseRole::Fusion, /*request_ok=*/true);
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();
    ASSERT_EQ(spans.size(), 4u);

    auto* wait = findSpan(spans, "wait");
    ASSERT_NE(wait, nullptr);
    EXPECT_EQ(wait->GetAttributes().count("request_id"), 0u);
    EXPECT_EQ(wait->GetAttributes().count("rtp_llm.request_id"), 0u);
}

TEST_F(PhaseSpanSynthesizerTest, PrefillModeProducesWaitAndPrefillOnly) {
    auto parent         = createParentSpan("rtp_llm.prefill_generate_stream_call");
    auto parent_span_id = parent->GetContext().span_id();

    auto timing = completedTiming(2000000, 2003000, 2050000, 2051000, 2060000);

    synthesizePhaseSpans(parent, timing, PhaseRole::Prefill, /*request_ok=*/true);
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();

    // 1 parent + 2 children (wait + prefill) = 3 spans
    ASSERT_EQ(spans.size(), 3u);

    auto* wait = findSpan(spans, "wait");
    ASSERT_NE(wait, nullptr);
    EXPECT_EQ(wait->GetParentSpanId(), parent_span_id);
    EXPECT_EQ(wait->GetDuration().count(), int64_t(3000) * 1000);

    auto* prefill = findSpan(spans, "prefill");
    ASSERT_NE(prefill, nullptr);
    EXPECT_EQ(prefill->GetParentSpanId(), parent_span_id);
    // duration = ttft - wait = 50000 - 3000 = 47000us
    EXPECT_EQ(prefill->GetDuration().count(), int64_t(47000) * 1000);

    // No decode span
    EXPECT_EQ(findSpan(spans, "decode"), nullptr);
}

TEST_F(PhaseSpanSynthesizerTest, DecodeModeProducesWaitAndDecodeOnly) {
    auto parent         = createParentSpan("rtp_llm.decode_remote_generate");
    auto parent_span_id = parent->GetContext().span_id();

    // The remote first token predates the local Decode begin. Decode ignores
    // that boundary and uses only RUNNING -> GenerateDone.
    auto timing = completedTiming(3000000, 3002000, 2999000, 3200000, 3205000);

    synthesizePhaseSpans(parent, timing, PhaseRole::Decode, /*request_ok=*/true);
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();

    // 1 parent + 2 children (wait + decode) = 3 spans
    ASSERT_EQ(spans.size(), 3u);

    auto* wait = findSpan(spans, "wait");
    ASSERT_NE(wait, nullptr);
    EXPECT_EQ(wait->GetParentSpanId(), parent_span_id);

    auto* decode = findSpan(spans, "decode");
    ASSERT_NE(decode, nullptr);
    EXPECT_EQ(decode->GetParentSpanId(), parent_span_id);
    // Decode starts at scheduled = begin + wait = 3002000us
    auto decode_start_ns = decode->GetStartTime().time_since_epoch().count();
    EXPECT_EQ(decode_start_ns, int64_t(3002000) * 1000);
    // Decode duration = cost - wait = 200000 - 2000 = 198000us
    EXPECT_EQ(decode->GetDuration().count(), int64_t(198000) * 1000);

    // No prefill span
    EXPECT_EQ(findSpan(spans, "prefill"), nullptr);
}

TEST_F(PhaseSpanSynthesizerTest, ZeroWaitSkipsWaitSpan) {
    auto parent = createParentSpan("parent");

    auto timing = completedTiming(1000000, 1000000, 1010000, 1050000, 1051000);

    synthesizePhaseSpans(parent, timing, PhaseRole::Fusion, /*request_ok=*/true);
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();

    // 1 parent + 2 children (prefill + decode, no wait) = 3 spans
    ASSERT_EQ(spans.size(), 3u);
    EXPECT_EQ(findSpan(spans, "wait"), nullptr);
    EXPECT_NE(findSpan(spans, "prefill"), nullptr);
    EXPECT_NE(findSpan(spans, "decode"), nullptr);
}

TEST_F(PhaseSpanSynthesizerTest, InvalidTimingProducesNoSpans) {
    auto parent = createParentSpan("parent");

    // begin_time_us = 0 -> guard triggers
    PhaseTiming timing;
    timing.begin_time_us         = 0;
    timing.synthesis_end_time_us = 100000;

    synthesizePhaseSpans(parent, timing, PhaseRole::Fusion, /*request_ok=*/false);
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();

    // Only the parent span
    ASSERT_EQ(spans.size(), 1u);
    EXPECT_EQ(spans[0]->GetName(), "parent");
}

TEST_F(PhaseSpanSynthesizerTest, NullParentIsSafe) {
    // Must not crash with null parent
    PhaseTiming timing;
    timing.begin_time_us         = 1000000;
    timing.synthesis_end_time_us = 1100000;

    synthesizePhaseSpans(nullptr, timing, PhaseRole::Fusion, /*request_ok=*/false);

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();
    EXPECT_EQ(spans.size(), 0u);
}

TEST_F(PhaseSpanSynthesizerTest, KvLoadSpanCoversLoadWindowOnSuccess) {
    auto parent         = createParentSpan("rtp_llm.decode_remote_generate");
    auto parent_span_id = parent->GetContext().span_id();

    // load window [4000000, 4069000): 69ms KV-arrival wait
    // Even if the caller supplies NONE_ERROR, a successful operation must not
    // carry any error classification attributes.
    synthesizeKvLoadSpan(parent, 4000000, 4069000, 42, /*ok=*/true, nullptr, 0, "NONE_ERROR");
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();
    ASSERT_EQ(spans.size(), 2u);

    auto* load = findSpan(spans, "load_cache");
    ASSERT_NE(load, nullptr);
    EXPECT_EQ(load->GetSpanKind(), trace_api::SpanKind::kInternal);
    EXPECT_EQ(load->GetParentSpanId(), parent_span_id);
    EXPECT_EQ(load->GetStartTime().time_since_epoch().count(), int64_t(4000000) * 1000);
    EXPECT_EQ(load->GetDuration().count(), int64_t(69000) * 1000);
    EXPECT_EQ(getStringAttribute(load, "request_id"), "42");
    EXPECT_EQ(getInt64Attribute(load, "rtp_llm.request_id"), 42);
    EXPECT_EQ(load->GetStatus(), trace_api::StatusCode::kOk);
    EXPECT_EQ(load->GetDescription(), "");
    EXPECT_EQ(load->GetAttributes().count("rtp_llm.error.code"), 0u);
    EXPECT_EQ(load->GetAttributes().count("rtp_llm.error.reason"), 0u);
}

TEST_F(PhaseSpanSynthesizerTest, KvLoadSpanMarksErrorOnFailure) {
    auto parent = createParentSpan("rtp_llm.decode_remote_generate");

    // Failure exit (e.g. CACHE_STORE_LOAD_BUFFER_TIMEOUT): span still
    // synthesized so the timeout window stays visible, but status = kError.
    synthesizeKvLoadSpan(parent,
                         4000000,
                         31000000,
                         42,
                         /*ok=*/false,
                         "DependencyFailure",
                         8307,
                         "CACHE_STORE_LOAD_BUFFER_TIMEOUT");
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();
    ASSERT_EQ(spans.size(), 2u);

    auto* load = findSpan(spans, "load_cache");
    ASSERT_NE(load, nullptr);
    EXPECT_EQ(load->GetStatus(), trace_api::StatusCode::kError);
    EXPECT_EQ(load->GetDescription(), "KV cache loading timed out while waiting for a buffer");
    EXPECT_EQ(getStringAttribute(load, "error.type"), "DependencyFailure");
    EXPECT_EQ(getInt64Attribute(load, "rtp_llm.error.code"), 8307);
    EXPECT_EQ(getStringAttribute(load, "rtp_llm.error.reason"), "CACHE_STORE_LOAD_BUFFER_TIMEOUT");
}

TEST_F(PhaseSpanSynthesizerTest, KvLoadSpanUsesLowCardinalityCancelledType) {
    auto parent = createParentSpan("rtp_llm.decode_remote_generate");

    synthesizeKvLoadSpan(parent,
                         4000000,
                         4069000,
                         42,
                         /*ok=*/false,
                         "Cancelled",
                         8100,
                         "CANCELLED");
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    const auto spans = span_data_->GetSpans();
    ASSERT_EQ(spans.size(), 2u);
    auto* load = findSpan(spans, "load_cache");
    ASSERT_NE(load, nullptr);
    EXPECT_EQ(load->GetStatus(), trace_api::StatusCode::kError);
    EXPECT_EQ(load->GetDescription(), "KV cache loading was cancelled");
    EXPECT_EQ(getStringAttribute(load, "error.type"), "Cancelled");
    EXPECT_EQ(getInt64Attribute(load, "rtp_llm.error.code"), 8100);
    EXPECT_EQ(getStringAttribute(load, "rtp_llm.error.reason"), "CANCELLED");
}

TEST_F(PhaseSpanSynthesizerTest, KvLoadSpanInvalidWindowIsSkipped) {
    auto parent = createParentSpan("rtp_llm.decode_remote_generate");

    synthesizeKvLoadSpan(parent, 0, 4069000, 42, true);         // begin unset
    synthesizeKvLoadSpan(parent, 4069000, 4000000, 42, true);   // end < begin
    synthesizeKvLoadSpan(nullptr, 4000000, 4069000, 42, true);  // null parent
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();
    ASSERT_EQ(spans.size(), 1u);  // parent only
}

TEST_F(PhaseSpanSynthesizerTest, FailureBeforeRunningTruncatesWait) {
    auto parent = createParentSpan("parent");

    PhaseTiming timing;
    timing.begin_time_us         = 1000000;
    timing.synthesis_end_time_us = 1100000;
    timing.request_id            = 42;
    timing.error_type            = "DeadlineExceeded";
    synthesizePhaseSpans(parent, timing, PhaseRole::Fusion, /*request_ok=*/false);
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();
    ASSERT_EQ(spans.size(), 2u);
    auto* wait = findSpan(spans, "wait");
    ASSERT_NE(wait, nullptr);
    EXPECT_EQ(wait->GetDuration().count(), int64_t(100000) * 1000);
    EXPECT_EQ(wait->GetStatus(), trace_api::StatusCode::kError);
    EXPECT_EQ(wait->GetDescription(), "Request deadline was exceeded while waiting for execution");
    EXPECT_TRUE(getBoolAttribute(wait, "rtp_llm.phase.truncated"));
    EXPECT_EQ(getStringAttribute(wait, "error.type"), "DeadlineExceeded");
}

TEST_F(PhaseSpanSynthesizerTest, DependencyFailureBeforeRunningExplainsTruncatedWait) {
    auto parent = createParentSpan("parent");

    PhaseTiming timing;
    timing.begin_time_us         = 1000000;
    timing.synthesis_end_time_us = 1100000;
    timing.request_id            = 42;
    timing.error_type            = "DependencyFailure";
    synthesizePhaseSpans(parent, timing, PhaseRole::Decode, /*request_ok=*/false);
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();
    ASSERT_EQ(spans.size(), 2u);
    auto* wait = findSpan(spans, "wait");
    ASSERT_NE(wait, nullptr);
    EXPECT_EQ(wait->GetStatus(), trace_api::StatusCode::kError);
    EXPECT_EQ(wait->GetDescription(), "Request stopped before execution because a dependency failed");
    EXPECT_TRUE(getBoolAttribute(wait, "rtp_llm.phase.truncated"));
    EXPECT_EQ(getStringAttribute(wait, "error.type"), "DependencyFailure");
}

TEST_F(PhaseSpanSynthesizerTest, FusionFailureBeforeFirstTokenTruncatesPrefill) {
    auto parent = createParentSpan("parent");

    PhaseTiming timing;
    timing.begin_time_us           = 1000000;
    timing.running_started         = true;
    timing.running_started_time_us = 1005000;
    timing.synthesis_end_time_us   = 1040000;
    timing.error_type              = "DeadlineExceeded";
    synthesizePhaseSpans(parent, timing, PhaseRole::Fusion, /*request_ok=*/false);
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();
    ASSERT_EQ(spans.size(), 3u);
    auto* wait    = findSpan(spans, "wait");
    auto* prefill = findSpan(spans, "prefill");
    ASSERT_NE(wait, nullptr);
    ASSERT_NE(prefill, nullptr);
    EXPECT_EQ(wait->GetStatus(), trace_api::StatusCode::kOk);
    EXPECT_EQ(getStringAttribute(wait, "error.type"), "");
    EXPECT_EQ(prefill->GetStatus(), trace_api::StatusCode::kError);
    EXPECT_EQ(prefill->GetDescription(), "Prefill exceeded the request deadline");
    EXPECT_EQ(prefill->GetDuration().count(), int64_t(35000) * 1000);
    EXPECT_TRUE(getBoolAttribute(prefill, "rtp_llm.phase.truncated"));
    EXPECT_EQ(getStringAttribute(prefill, "error.type"), "DeadlineExceeded");
    EXPECT_EQ(findSpan(spans, "decode"), nullptr);
}

TEST_F(PhaseSpanSynthesizerTest, FusionFailureAfterFirstTokenTruncatesDecode) {
    auto parent = createParentSpan("parent");

    PhaseTiming timing;
    timing.begin_time_us           = 1000000;
    timing.running_started         = true;
    timing.running_started_time_us = 1005000;
    timing.first_token_committed   = true;
    timing.first_token_time_us     = 1020000;
    timing.synthesis_end_time_us   = 1080000;
    timing.error_type              = "Cancelled";
    synthesizePhaseSpans(parent, timing, PhaseRole::Fusion, /*request_ok=*/false);
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();
    ASSERT_EQ(spans.size(), 4u);
    auto* prefill = findSpan(spans, "prefill");
    auto* decode  = findSpan(spans, "decode");
    ASSERT_NE(prefill, nullptr);
    ASSERT_NE(decode, nullptr);
    EXPECT_EQ(prefill->GetStatus(), trace_api::StatusCode::kOk);
    EXPECT_EQ(getStringAttribute(prefill, "error.type"), "");
    EXPECT_EQ(decode->GetStatus(), trace_api::StatusCode::kError);
    EXPECT_EQ(decode->GetDescription(), "Decode was cancelled");
    EXPECT_EQ(decode->GetDuration().count(), int64_t(60000) * 1000);
    EXPECT_TRUE(getBoolAttribute(decode, "rtp_llm.phase.truncated"));
    EXPECT_EQ(getStringAttribute(decode, "error.type"), "Cancelled");
}

TEST_F(PhaseSpanSynthesizerTest, FusionTransportFailureAfterGenerateDoneKeepsPhasesOk) {
    auto parent = createParentSpan("parent");
    auto timing = completedTiming(1000000, 1005000, 1020000, 1100000, 1200000);

    synthesizePhaseSpans(parent, timing, PhaseRole::Fusion, /*request_ok=*/false);
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();
    ASSERT_EQ(spans.size(), 4u);
    auto* decode = findSpan(spans, "decode");
    ASSERT_NE(decode, nullptr);
    EXPECT_EQ(decode->GetStatus(), trace_api::StatusCode::kOk);
    EXPECT_EQ(decode->GetDuration().count(), int64_t(80000) * 1000);
    EXPECT_EQ(decode->GetAttributes().count("rtp_llm.phase.truncated"), 0u);
    EXPECT_EQ(decode->GetAttributes().count("error.type"), 0u);
}

TEST_F(PhaseSpanSynthesizerTest, PrefillFailureAfterFirstTokenKeepsLocalPhaseOk) {
    auto parent = createParentSpan("parent");

    PhaseTiming timing;
    timing.begin_time_us           = 1000000;
    timing.running_started         = true;
    timing.running_started_time_us = 1005000;
    timing.first_token_committed   = true;
    timing.first_token_time_us     = 1020000;
    timing.synthesis_end_time_us   = 1080000;
    synthesizePhaseSpans(parent, timing, PhaseRole::Prefill, /*request_ok=*/false);
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();
    ASSERT_EQ(spans.size(), 3u);
    auto* prefill = findSpan(spans, "prefill");
    ASSERT_NE(prefill, nullptr);
    EXPECT_EQ(prefill->GetStatus(), trace_api::StatusCode::kOk);
    EXPECT_EQ(prefill->GetAttributes().count("rtp_llm.phase.truncated"), 0u);
    EXPECT_EQ(prefill->GetAttributes().count("error.type"), 0u);
    EXPECT_EQ(findSpan(spans, "decode"), nullptr);
}

TEST_F(PhaseSpanSynthesizerTest, DecodeFailureBeforeGenerateDoneTruncatesDecode) {
    auto parent = createParentSpan("parent");

    PhaseTiming timing;
    timing.begin_time_us           = 3000000;
    timing.running_started         = true;
    timing.running_started_time_us = 3002000;
    timing.first_token_committed   = true;
    timing.first_token_time_us     = 2999000;
    timing.synthesis_end_time_us   = 3100000;
    timing.error_type              = "Cancelled";
    synthesizePhaseSpans(parent, timing, PhaseRole::Decode, /*request_ok=*/false);
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();
    ASSERT_EQ(spans.size(), 3u);
    auto* decode = findSpan(spans, "decode");
    ASSERT_NE(decode, nullptr);
    EXPECT_EQ(decode->GetStatus(), trace_api::StatusCode::kError);
    EXPECT_EQ(decode->GetDuration().count(), int64_t(98000) * 1000);
    EXPECT_TRUE(getBoolAttribute(decode, "rtp_llm.phase.truncated"));
    EXPECT_EQ(getStringAttribute(decode, "error.type"), "Cancelled");
}

TEST_F(PhaseSpanSynthesizerTest, DecodeTransportFailureAfterGenerateDoneKeepsDecodeOk) {
    auto parent = createParentSpan("parent");
    auto timing = completedTiming(3000000, 3002000, 2999000, 3100000, 3200000);

    synthesizePhaseSpans(parent, timing, PhaseRole::Decode, /*request_ok=*/false);
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();
    ASSERT_EQ(spans.size(), 3u);
    auto* decode = findSpan(spans, "decode");
    ASSERT_NE(decode, nullptr);
    EXPECT_EQ(decode->GetStatus(), trace_api::StatusCode::kOk);
    EXPECT_EQ(decode->GetDuration().count(), int64_t(98000) * 1000);
    EXPECT_EQ(decode->GetAttributes().count("rtp_llm.phase.truncated"), 0u);
    EXPECT_EQ(decode->GetAttributes().count("error.type"), 0u);
}

TEST_F(PhaseSpanSynthesizerTest, InvalidProgressIsSkippedWithoutClamping) {
    auto parent = createParentSpan("parent");

    PhaseTiming invalid_running;
    invalid_running.begin_time_us           = 1000000;
    invalid_running.running_started         = true;
    invalid_running.running_started_time_us = 1100001;
    invalid_running.synthesis_end_time_us   = 1100000;
    synthesizePhaseSpans(parent, invalid_running, PhaseRole::Fusion, /*request_ok=*/false);

    PhaseTiming invalid_done;
    invalid_done.begin_time_us           = 2000000;
    invalid_done.running_started         = true;
    invalid_done.running_started_time_us = 2005000;
    invalid_done.generation_done         = true;
    invalid_done.generation_done_time_us = 2200001;
    invalid_done.synthesis_end_time_us   = 2200000;
    synthesizePhaseSpans(parent, invalid_done, PhaseRole::Decode, /*request_ok=*/false);
    parent->End();

    EXPECT_TRUE(TelemetryRuntime::shutdown(5000));
    auto spans = span_data_->GetSpans();
    ASSERT_EQ(spans.size(), 2u);
    auto* wait = findSpan(spans, "wait");
    ASSERT_NE(wait, nullptr);
    EXPECT_EQ(wait->GetStartTime().time_since_epoch().count(), int64_t(2000000) * 1000);
    EXPECT_EQ(findSpan(spans, "decode"), nullptr);
}

TEST_F(PhaseSpanSynthesizerTest, SynthesisScopeReportsNormalAndExceptionExit) {
    bool normal_unwinding = true;
    {
        PhaseSpanSynthesisScope scope([&](bool unwinding) { normal_unwinding = unwinding; });
    }
    EXPECT_FALSE(normal_unwinding);

    bool exception_unwinding = false;
    try {
        PhaseSpanSynthesisScope scope([&](bool unwinding) { exception_unwinding = unwinding; });
        throw std::runtime_error("boom");
    } catch (const std::runtime_error&) {}
    EXPECT_TRUE(exception_unwinding);
}

TEST_F(PhaseSpanSynthesizerTest, SynthesisScopeIsFailOpen) {
    {
        PhaseSpanSynthesisScope scope([](bool) { throw std::runtime_error("boom"); });
    }
    { PhaseSpanSynthesisScope scope(nullptr); }
}

}  // namespace

}  // namespace telemetry
}  // namespace rtp_llm
