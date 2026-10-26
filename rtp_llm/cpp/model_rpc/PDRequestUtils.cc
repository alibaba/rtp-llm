#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/telemetry/RpcTraceHelper.h"
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <thread>
#include "rtp_llm/cpp/multimodal_processor/MultimodalError.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

ErrorInfo updateMultimodalFeaturesWithTrace(
    MultimodalProcessor& processor,
    std::shared_ptr<GenerateInput>& input,
    const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& parent_span,
    grpc::ServerContext* server_context) {
    if (!telemetry::TelemetryRuntime::isActive() || parent_span == nullptr || !input->multimodal_inputs
        || input->multimodal_inputs->empty()) {
        return processor.updateMultimodalFeatures(input, server_context);
    }

    opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> span;
    try {
        opentelemetry::trace::StartSpanOptions options;
        options.parent = parent_span->GetContext();
        options.kind   = opentelemetry::trace::SpanKind::kInternal;
        span           = telemetry::TelemetryRuntime::tracer()->StartSpan("rtp_llm.vit", options);
    } catch (...) {}
    grpc::Status                   status;
    telemetry::GrpcStatusSpanGuard guard(span, &status, telemetry::SpanStatusSemantics::Logical);
    try {
        guard.setAttribute(telemetry::kAttrRequestId, std::to_string(input->request_id));
    } catch (...) {}

    // Include preprocessing, embedding transport and token expansion in one operation.
    auto result = processor.updateMultimodalFeatures(input, server_context);
    if (!result.ok()) {
        status = grpc::Status(grpc::StatusCode::INTERNAL, "");
        try {
            const auto reason = ErrorCodeToString(result.code());
            guard.setLogicalErrorType(reason);
            guard.setAttribute(telemetry::kAttrRtpLlmErrorCode, static_cast<int64_t>(result.code()));
            guard.setAttribute(telemetry::kAttrRtpLlmErrorReason, reason);
        } catch (...) {}
    }
    return result;
}

std::string masterEnqueuedHandoffUniqueKey(int64_t request_id) {
    return "master_enqueued_" + std::to_string(request_id);
}

PDSupportDecision checkPDSupport(const GenerateInputPB& request) {
    const auto& config = request.generate_config();
    if (!config.can_use_pd_separation()) {
        return {false, "PD disabled by request"};
    }
    int max_new_tokens;
    try {
        max_new_tokens = QueryConverter::resolveMaxNewTokens(config);
    } catch (const std::invalid_argument&) {
        return {false, "invalid token generation configuration"};
    }
    if (max_new_tokens <= 1) {
        return {false, "PD requires more than one generated token"};
    }
    if (config.num_beams() > 1 || config.variable_num_beams_size() != 0) {
        return {false, "PD does not support beam search"};
    }
    if (config.num_return_sequences() > 1) {
        return {false, "PD does not support multiple return sequences"};
    }
    return {true, ""};
}

ErrorInfo validatePDHandoff(const GenerateInputPB& request) {
    if (request.generate_config().timeout_ms() <= 0) {
        return ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "invalid P2P request timeout");
    }
    if (request.generate_config().unique_key().empty()) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "PD handoff requires non-empty unique_key");
    }
    return ErrorInfo::OkStatus();
}

ErrorInfo preprocessForPD(std::shared_ptr<GenerateInput>& input,
                          MultimodalProcessor*            processor,
                          bool                            is_mtp_eagle,
                          RoleType                        role,
                          grpc::ServerContext*            server_context,
                          const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& span) {
    input->generate_config->pd_separation        = true;
    input->generate_config->force_disable_sp_run = !is_mtp_eagle;
    if (role == RoleType::DECODE) {
        // Decode consumes the FE-expanded IDs. Only Prefill fetches embeddings;
        // its handoff carries the position IDs needed by subsequent decode steps.
        if (input->multimodal_token_layout) {
            const auto& ids   = input->input_ids;
            const auto& spans = input->multimodal_token_layout->spans;
            if (!ids.defined() || !ids.device().is_cpu() || ids.scalar_type() != torch::kInt32 || ids.dim() != 1
                || !ids.is_contiguous() || spans.empty()) {
                return {ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid frontend multimodal token layout"};
            }
            int64_t previous_end = 0;
            for (const auto& [offset, length] : spans) {
                const int64_t end = int64_t(offset) + length;
                if (offset < previous_end || length <= 0 || end > ids.numel()) {
                    return {ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid frontend multimodal span"};
                }
                previous_end = end;
            }
            auto mask = torch::ones_like(ids);
            auto locs = torch::empty({static_cast<int64_t>(spans.size())}, torch::kInt32);
            for (size_t i = 0; i < spans.size(); ++i) {
                const auto& [offset, length] = spans[i];
                mask.narrow(0, offset, length).zero_();
                locs.data_ptr<int32_t>()[i] = offset;
            }
            input->text_tokens_mask = std::move(mask);
            input->mm_locs          = std::move(locs);
        } else if (input->multimodal_inputs && !input->multimodal_inputs->empty() && !input->v41_inputs) {
            return {ErrorCode::MM_WRONG_FORMAT_ERROR, "Decode requires frontend-expanded multimodal token IDs"};
        }
        return ErrorInfo::OkStatus();
    }
    if (input->multimodal_inputs) {
        if (!processor) {
            return {ErrorCode::MM_NOT_SUPPORTED_ERROR, "PD multimodal request requires a multimodal processor"};
        }
        return updateMultimodalFeaturesWithTrace(*processor, input, span, server_context);
    }
    return ErrorInfo::OkStatus();
}

ErrorInfo preprocessPrefillWithRetry(const GenerateInputPB&                                              request,
                                     std::shared_ptr<GenerateInput>&                                     input,
                                     MultimodalProcessor*                                                processor,
                                     bool                                                                is_mtp_eagle,
                                     const PDSepConfig&                                                  config,
                                     grpc::ServerContext*                                                server_context,
                                     const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& span,
                                     const std::function<ErrorInfo()>&                                   cancellation) {
    if (!input || !input->generate_config) {
        return {ErrorCode::INVALID_PARAMS, "Prefill preprocessing requires input and generate config"};
    }
    const auto request_deadline_ms = input->request_deadline_ms;
    const auto begin_time_us       = input->begin_time_us;
    const auto retry_start         = std::chrono::steady_clock::now();
    const auto retries             = std::max<int64_t>(0, config.prefill_retry_times);
    const auto stopped             = [&]() -> ErrorInfo {
        if (cancellation) {
            auto status = cancellation();
            if (!status.ok())
                return status;
        }
        if (request_deadline_ms > 0 && currentTimeMs() >= request_deadline_ms) {
            return {ErrorCode::GENERATE_TIMEOUT, "Prefill preprocessing request deadline exceeded"};
        }
        if (server_context && server_context->IsCancelled()) {
            return {ErrorCode::CANCELLED, "Prefill preprocessing canceled"};
        }
        return ErrorInfo::OkStatus();
    };
    const auto retry_budget_exhausted = [&]() {
        return config.prefill_retry_timeout_ms > 0
               && std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - retry_start)
                          .count()
                      >= config.prefill_retry_timeout_ms;
    };
    auto last_error = ErrorInfo::OkStatus();
    for (int64_t attempt = 0;; ++attempt) {
        auto stop = stopped();
        if (!stop.ok())
            return stop;
        if (attempt > 0) {
            if (retry_budget_exhausted())
                return last_error;
            input                      = QueryConverter::transQuery(&request);
            input->request_deadline_ms = request_deadline_ms;
            input->begin_time_us       = begin_time_us;
        }
        last_error = preprocessForPD(input, processor, is_mtp_eagle, RoleType::PREFILL, server_context, span);
        stop       = stopped();
        if (!stop.ok())
            return stop;
        if (last_error.ok() || !isRetryableMultimodalError(last_error.code()) || attempt >= retries
            || retry_budget_exhausted()) {
            return last_error;
        }
        // Match the existing Prefill retry pacing; the next attempt rechecks
        // cancellation, request expiry and the monotonic retry budget.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

}  // namespace rtp_llm
