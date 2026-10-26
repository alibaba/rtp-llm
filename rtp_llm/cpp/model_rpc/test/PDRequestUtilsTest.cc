#include <gtest/gtest.h>
#include <functional>
#include <chrono>
#include <thread>
#include "opentelemetry/exporters/memory/in_memory_span_data.h"
#include "opentelemetry/exporters/memory/in_memory_span_exporter_factory.h"
#include "rtp_llm/cpp/telemetry/RpcTraceHelper.h"

#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"

namespace rtp_llm {
namespace {

GenerateInputPB makeRequest() {
    GenerateInputPB request;
    request.set_request_id(42);
    request.mutable_generate_config()->set_timeout_ms(12345);
    request.add_token_ids(10);
    request.add_token_ids(1);
    request.add_token_ids(20);
    auto* config = request.mutable_generate_config();
    config->set_max_new_tokens(8);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);
    config->set_unique_key("pd-test");
    return request;
}

class TestMultimodalProcessor: public MultimodalProcessor {
public:
    TestMultimodalProcessor(): MultimodalProcessor(py::object{}, MMModelConfig{true, {{1}}, false}, 100) {}

    int  calls            = 0;
    int  embedding_length = 3;
    bool fail             = false;
    int                      failures_remaining = 0;
    ErrorCode                failure_code       = ErrorCode::MM_PROCESS_ERROR;
    std::function<void(int)> on_call;
    grpc::ServerContext* seen_context = nullptr;

private:
    ErrorResult<MultimodalOutput>
    MultimodalEmbedding(const std::vector<MultimodalInput>, std::string, int64_t, grpc::ServerContext* context) override {
        ++calls;
        seen_context = context;
        if (on_call)
            on_call(calls);
        if (fail || failures_remaining > 0) {
            if (failures_remaining > 0)
                --failures_remaining;
            return {failure_code, "test MM failure"};
        }
        MultimodalOutput output;
        output.mm_features     = {torch::zeros({embedding_length, 2})};
        output.mm_position_ids = std::vector<torch::Tensor>{torch::arange(embedding_length, torch::kInt64)};
        output.mm_extra_input  = std::vector<torch::Tensor>{torch::tensor({2, 3}, torch::kInt32)};
        return output;
    }
};

GenerateInputPB makeMMRequest() {
    auto request = makeRequest();
    request.add_multimodal_inputs()->set_multimodal_url("test-image");
    return request;
}

}  // namespace

TEST(PDRequestUtilsTest, PrefillMmTracePreservesRequestContextAndTypedFailure) {
    namespace memory = opentelemetry::exporter::memory;
    telemetry::TelemetryRuntime::shutdown(5000);
    std::shared_ptr<memory::InMemorySpanData> span_data;
    auto exporter = memory::InMemorySpanExporterFactory::Create(span_data);
    telemetry::TelemetryConfig config;
    config.enabled = true;
    config.role = "test";
    ASSERT_TRUE(telemetry::TelemetryRuntime::initWithExporter(std::move(exporter), config));
    grpc::ServerContext context;
    auto request = makeMMRequest();
    auto input = QueryConverter::transQuery(&request);
    TestMultimodalProcessor processor;
    processor.fail = true;
    auto request_span = telemetry::startRpcServerSpan("pd-mm-request", &context, true);
    const auto request_span_id = request_span->GetContext().span_id();
    const auto result = preprocessForPD(input, &processor, false, RoleType::PREFILL, &context, request_span);
    EXPECT_EQ(result.code(), ErrorCode::MM_PROCESS_ERROR);
    EXPECT_EQ(processor.calls, 1);
    EXPECT_EQ(processor.seen_context, &context);
    request_span->End();
    EXPECT_TRUE(telemetry::TelemetryRuntime::shutdown(5000));
    auto spans = span_data->GetSpans();
    size_t mm_spans = 0;
    for (const auto& span : spans) {
        if (span->GetName() != "rtp_llm.vit") {
            continue;
        }
        ++mm_spans;
        EXPECT_EQ(span->GetSpanKind(), opentelemetry::trace::SpanKind::kInternal);
        EXPECT_EQ(span->GetParentSpanId(), request_span_id);
        EXPECT_EQ(span->GetStatus(), opentelemetry::trace::StatusCode::kError);
        const auto& attributes = span->GetAttributes();
        const auto found = attributes.find(telemetry::kAttrRtpLlmErrorCode);
        EXPECT_NE(found, attributes.end());
        if (found != attributes.end()) {
            EXPECT_EQ(opentelemetry::nostd::get<int64_t>(found->second),
                      static_cast<int64_t>(ErrorCode::MM_PROCESS_ERROR));
        }
    }
    EXPECT_EQ(mm_spans, 1u);
}

TEST(PDRequestUtilsTest, MasterEnqueuedHandoffKeyIsStablePerRequest) {
    EXPECT_EQ(masterEnqueuedHandoffUniqueKey(42), "master_enqueued_42");
    EXPECT_EQ(masterEnqueuedHandoffUniqueKey(42), masterEnqueuedHandoffUniqueKey(42));
    EXPECT_NE(masterEnqueuedHandoffUniqueKey(42), masterEnqueuedHandoffUniqueKey(43));
}

TEST(PDRequestUtilsTest, PromptLogitsFallsBackEvenWithLargerWireTokenLimit) {
    auto request = makeRequest();
    ASSERT_TRUE(checkPDSupport(request).supported);
    request.mutable_generate_config()->set_return_prompt_logits(true);
    EXPECT_FALSE(checkPDSupport(request).supported);
}

TEST(PDRequestUtilsTest, PDSupportRejectsEachUnsupportedRequestShape) {
    const auto baseline = makeRequest();
    ASSERT_TRUE(checkPDSupport(baseline).supported);
    auto request = baseline;
    request.mutable_generate_config()->set_can_use_pd_separation(false);
    EXPECT_FALSE(checkPDSupport(request).supported);
    request = baseline;
    request.mutable_generate_config()->set_max_new_tokens(1);
    EXPECT_FALSE(checkPDSupport(request).supported);
    request = baseline;
    request.mutable_generate_config()->set_num_beams(2);
    EXPECT_FALSE(checkPDSupport(request).supported);
    request = baseline;
    request.mutable_generate_config()->add_variable_num_beams(2);
    EXPECT_FALSE(checkPDSupport(request).supported);
    request = baseline;
    request.mutable_generate_config()->set_num_return_sequences(2);
    EXPECT_FALSE(checkPDSupport(request).supported);
}

TEST(PDRequestUtilsTest, TextPreprocessingPreservesRequestAndAppliesSameSpPolicy) {
    const auto request = makeRequest();
    for (bool is_mtp_eagle : {false, true}) {
        auto input = QueryConverter::transQuery(&request);
        ASSERT_TRUE(preprocessForPD(input, nullptr, is_mtp_eagle, RoleType::PREFILL, nullptr, {}).ok());
        EXPECT_TRUE(input->generate_config->pd_separation);
        EXPECT_EQ(input->generate_config->force_disable_sp_run, !is_mtp_eagle);
        EXPECT_EQ(input->generate_config->unique_key, "pd-test");
        EXPECT_EQ(input->request_deadline_ms, 0);
        EXPECT_EQ(input->generate_config->timeout_ms, 12345);
        EXPECT_TRUE(torch::equal(input->input_ids, torch::tensor({10, 1, 20}, torch::kInt32)));
    }
    EXPECT_EQ(request.token_ids_size(), 3);
}

TEST(PDRequestUtilsTest, PrefillExpansionIsDeterministicAcrossWireRoundTrip) {
    const auto              request = makeMMRequest();
    auto                    decode  = QueryConverter::transQuery(&request);
    TestMultimodalProcessor decode_processor;
    TestMultimodalProcessor prefill_processor;
    ASSERT_TRUE(preprocessForPD(decode, &decode_processor, false, RoleType::PREFILL, nullptr, {}).ok());
    GenerateInputPB handoff;
    ASSERT_TRUE(handoff.ParseFromString(request.SerializeAsString()));
    auto prefill = QueryConverter::transQuery(&handoff);
    ASSERT_TRUE(preprocessForPD(prefill, &prefill_processor, false, RoleType::PREFILL, nullptr, {}).ok());
    EXPECT_EQ(decode_processor.calls, 1);
    EXPECT_EQ(prefill_processor.calls, 1);
    EXPECT_EQ(decode->input_ids.numel(), 5);
    EXPECT_EQ(request.token_ids_size(), 3);
    EXPECT_EQ(handoff.token_ids_size(), 3);

    // Keep expansion consistency assertions in tests instead of the request path.
    const auto expect_equal = [](const torch::Tensor& a, const torch::Tensor& b) {
        EXPECT_EQ(a.scalar_type(), b.scalar_type());
        EXPECT_EQ(a.sizes(), b.sizes());
        EXPECT_TRUE(torch::equal(a, b));
    };
    expect_equal(decode->input_ids, prefill->input_ids);
    ASSERT_TRUE(decode->text_tokens_mask && prefill->text_tokens_mask);
    expect_equal(*decode->text_tokens_mask, *prefill->text_tokens_mask);
    ASSERT_TRUE(decode->mm_locs && prefill->mm_locs);
    expect_equal(*decode->mm_locs, *prefill->mm_locs);
    ASSERT_TRUE(decode->mm_position_ids && prefill->mm_position_ids);
    ASSERT_TRUE(decode->mm_extra_input && prefill->mm_extra_input);
    ASSERT_TRUE(decode->multimodal_features && prefill->multimodal_features);
    const auto expect_tensors_equal = [&](const auto& a, const auto& b) {
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            expect_equal(a[i], b[i]);
        }
    };
    expect_tensors_equal(*decode->mm_position_ids, *prefill->mm_position_ids);
    expect_tensors_equal(*decode->mm_extra_input, *prefill->mm_extra_input);
    expect_tensors_equal(*decode->multimodal_features, *prefill->multimodal_features);
}

TEST(PDRequestUtilsTest, MissingProcessorOrProcessingFailureAreRejected) {
    auto request = makeMMRequest();
    auto input   = QueryConverter::transQuery(&request);
    EXPECT_EQ(preprocessForPD(input, nullptr, false, RoleType::PREFILL, nullptr, {}).code(), ErrorCode::MM_NOT_SUPPORTED_ERROR);
    TestMultimodalProcessor processor;
    processor.fail    = true;
    const auto result = preprocessForPD(input, &processor, false, RoleType::PREFILL, nullptr, {});
    EXPECT_EQ(result.code(), ErrorCode::MM_PROCESS_ERROR);
    EXPECT_EQ(result.ToString(), "test MM failure");
}

TEST(PDRequestUtilsTest, DecodeConsumesFrontendExpandedIdsWithoutProcessor) {
    auto request = makeMMRequest();
    request.clear_token_ids();
    for (int id : {10, -101, -102, -103, 20}) {
        request.add_token_ids(id);
    }
    auto* span = request.mutable_multimodal_token_layout()->add_spans();
    span->set_offset(1);
    span->set_length(3);
    auto        input        = QueryConverter::transQuery(&request);
    const auto* original_ids = input->input_ids.data_ptr<int32_t>();
    ASSERT_TRUE(preprocessForPD(input, nullptr, false, RoleType::DECODE, nullptr, {}).ok());
    EXPECT_EQ(input->input_ids.data_ptr<int32_t>(), original_ids);
    EXPECT_TRUE(torch::equal(input->input_ids, torch::tensor({10, -101, -102, -103, 20}, torch::kInt32)));
    ASSERT_TRUE(input->text_tokens_mask);
    EXPECT_TRUE(torch::equal(*input->text_tokens_mask, torch::tensor({1, 0, 0, 0, 1}, torch::kInt32)));
    ASSERT_TRUE(input->mm_locs);
    EXPECT_TRUE(torch::equal(*input->mm_locs, torch::tensor({1}, torch::kInt32)));
    EXPECT_FALSE(input->multimodal_features);
    EXPECT_TRUE(input->generate_config->pd_separation);
    EXPECT_TRUE(input->generate_config->force_disable_sp_run);
}

TEST(PDRequestUtilsTest, DecodeRejectsRawOrMalformedMultimodalIds) {
    auto request = makeMMRequest();
    auto input   = QueryConverter::transQuery(&request);
    EXPECT_EQ(preprocessForPD(input, nullptr, false, RoleType::DECODE, nullptr, {}).code(), ErrorCode::MM_WRONG_FORMAT_ERROR);
    auto* span = request.mutable_multimodal_token_layout()->add_spans();
    span->set_offset(2);
    span->set_length(2);
    input = QueryConverter::transQuery(&request);
    EXPECT_EQ(preprocessForPD(input, nullptr, false, RoleType::DECODE, nullptr, {}).code(), ErrorCode::MM_WRONG_FORMAT_ERROR);
    EXPECT_FALSE(input->text_tokens_mask);
}

TEST(PDRequestUtilsTest, PrefillRetryRebuildsPartialInputWithoutRenewingRequestBudget) {
    const auto request         = makeMMRequest();
    auto       input           = QueryConverter::transQuery(&request);
    const auto deadline        = currentTimeMs() + 60000;
    const auto begin_time      = input->begin_time_us;
    input->request_deadline_ms = deadline;
    PDSepConfig config;
    config.prefill_retry_times      = 1;
    config.prefill_retry_timeout_ms = 0;
    TestMultimodalProcessor processor;
    processor.failures_remaining = 1;
    processor.on_call            = [&](int attempt) {
        if (attempt == 1) {
            input->input_ids.fill_(777);
            input->generate_config->unique_key = "partial";
            input->begin_time_us               = 0;
            input->request_deadline_ms         = deadline + 10000;
        } else {
            EXPECT_TRUE(torch::equal(input->input_ids, torch::tensor({10, 1, 20}, torch::kInt32)));
            EXPECT_EQ(input->generate_config->unique_key, "pd-test");
            EXPECT_EQ(input->begin_time_us, begin_time);
            EXPECT_EQ(input->request_deadline_ms, deadline);
        }
    };
    ASSERT_TRUE(preprocessPrefillWithRetry(request, input, &processor, false, config, nullptr, {}, {}).ok());
    EXPECT_EQ(processor.calls, 2);
    auto                    expected = QueryConverter::transQuery(&request);
    TestMultimodalProcessor successful;
    ASSERT_TRUE(preprocessForPD(expected, &successful, false, RoleType::PREFILL, nullptr, {}).ok());
    EXPECT_TRUE(torch::equal(input->input_ids, expected->input_ids));
    EXPECT_EQ(request.token_ids(0), 10);
    EXPECT_EQ(input->request_deadline_ms, deadline);
    EXPECT_EQ(input->begin_time_us, begin_time);
}

TEST(PDRequestUtilsTest, PrefillRetryCountIsBoundedAndPermanentErrorsDoNotRetry) {
    const auto  request = makeMMRequest();
    PDSepConfig config;
    config.prefill_retry_times      = 2;
    config.prefill_retry_timeout_ms = 0;
    for (const auto error : {ErrorCode::MM_REMOTE_RPC_FAILED, ErrorCode::MM_WRONG_FORMAT_ERROR}) {
        auto                    input = QueryConverter::transQuery(&request);
        TestMultimodalProcessor processor;
        processor.fail         = true;
        processor.failure_code = error;
        const auto status      = preprocessPrefillWithRetry(request, input, &processor, false, config, nullptr, {}, {});
        EXPECT_EQ(status.code(), error);
        EXPECT_EQ(status.ToString(), "test MM failure");
        EXPECT_EQ(processor.calls, error == ErrorCode::MM_REMOTE_RPC_FAILED ? 3 : 1);
    }
    auto                    input = QueryConverter::transQuery(&request);
    TestMultimodalProcessor disabled;
    disabled.fail              = true;
    config.prefill_retry_times = 0;
    EXPECT_EQ(preprocessPrefillWithRetry(request, input, &disabled, false, config, nullptr, {}, {}).code(),
              ErrorCode::MM_PROCESS_ERROR);
    EXPECT_EQ(disabled.calls, 1);
}

TEST(PDRequestUtilsTest, PrefillRetryBudgetStopsFurtherAttemptsAndPreservesLastError) {
    const auto  request = makeMMRequest();
    auto        input   = QueryConverter::transQuery(&request);
    PDSepConfig config;
    config.prefill_retry_times      = 5;
    config.prefill_retry_timeout_ms = 1;
    TestMultimodalProcessor processor;
    processor.fail    = true;
    processor.on_call = [](int) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); };
    const auto result = preprocessPrefillWithRetry(request, input, &processor, false, config, nullptr, {}, {});
    EXPECT_EQ(result.code(), ErrorCode::MM_PROCESS_ERROR);
    EXPECT_EQ(processor.calls, 1);
}

TEST(PDRequestUtilsTest, PrefillRetryHonorsCancellationCauseAndExpiredRequest) {
    const auto  request = makeMMRequest();
    auto        input   = QueryConverter::transQuery(&request);
    PDSepConfig config;
    config.prefill_retry_times      = 5;
    config.prefill_retry_timeout_ms = 0;
    TestMultimodalProcessor processor;
    processor.fail    = true;
    bool canceled     = false;
    processor.on_call = [&](int) { canceled = true; };
    const auto result = preprocessPrefillWithRetry(request, input, &processor, false, config, nullptr, {}, [&] {
        return canceled ? ErrorInfo(ErrorCode::PRIORITY_PREEMPTED, "preempted") : ErrorInfo::OkStatus();
    });
    EXPECT_EQ(result.code(), ErrorCode::PRIORITY_PREEMPTED);
    EXPECT_EQ(processor.calls, 1);
    input                      = QueryConverter::transQuery(&request);
    input->request_deadline_ms = currentTimeMs() - 1;
    TestMultimodalProcessor expired;
    EXPECT_EQ(preprocessPrefillWithRetry(request, input, &expired, false, config, nullptr, {}, {}).code(),
              ErrorCode::GENERATE_TIMEOUT);
    EXPECT_EQ(expired.calls, 0);
}

}  // namespace rtp_llm
