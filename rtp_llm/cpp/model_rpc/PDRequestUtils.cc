#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/telemetry/RpcTraceHelper.h"
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <thread>
#include <future>
#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransport.h"
#if USING_CUDA
#include <c10/cuda/CUDAFunctions.h>
#endif
#include "rtp_llm/cpp/multimodal_processor/MultimodalError.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

namespace {
template<typename Run>
ErrorInfo traceMultimodalWork(int64_t                                                             request_id,
                              const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& parent_span,
                              Run&&                                                               run) {
    if (!telemetry::TelemetryRuntime::isActive() || parent_span == nullptr)
        return run();

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
        guard.setAttribute(telemetry::kAttrRequestId, std::to_string(request_id));
    } catch (...) {}

    // Include preprocessing, embedding transport and token expansion in one operation.
    auto result = run();
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

}  // namespace

ErrorInfo updateMultimodalFeaturesWithTrace(MultimodalProcessor&            processor,
                                            std::shared_ptr<GenerateInput>& input,
                                            const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& span,
                                            grpc::ServerContext* context) {
    auto run = [&] { return processor.updateMultimodalFeatures(input, context); };
    if (!input->multimodal_inputs || input->multimodal_inputs->empty())
        return run();
    return traceMultimodalWork(input->request_id, span, run);
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

ErrorInfo preprocessForPD(std::shared_ptr<GenerateInput>&                                     input,
                          MultimodalProcessor*                                                processor,
                          bool                                                                is_mtp_eagle,
                          RoleType                                                            role,
                          grpc::ServerContext*                                                server_context,
                          const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& span) {
    input->generate_config->pd_separation        = true;
    input->generate_config->force_disable_sp_run = !is_mtp_eagle;
    if (role == RoleType::DECODE) {
        // Remote ViT uses the frontend layout. With LOCAL ViT, both engines
        // expand the original request independently before using its KV keys.
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
            // Decode only uses the expanded layout; the original wire request is forwarded to Prefill.
            input->multimodal_request.reset();
        } else if (processor
                   && (input->v41_inputs || (input->multimodal_inputs && !input->multimodal_inputs->empty()))) {
            return updateMultimodalFeaturesWithTrace(*processor, input, span, server_context);
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

namespace {
template<typename Run>
ErrorInfo retryPrefillMedia(const PDSepConfig&                config,
                            int64_t                           request_deadline_ms,
                            grpc::ServerContext*              server_context,
                            const std::function<ErrorInfo()>& cancellation,
                            Run&&                             run) {
    const auto retry_start = std::chrono::steady_clock::now();
    const auto retries     = std::max<int64_t>(0, config.prefill_retry_times);
    const auto stopped     = [&]() -> ErrorInfo {
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
        if (attempt > 0 && retry_budget_exhausted())
            return last_error;
        last_error = run(attempt);
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

// Keep span/hash validation aligned with the exact media sent to the ViT.
MultimodalInputsPB selectPrefillImages(GenerateInput& part, size_t first) {
    auto& suffix = part.multimodal_token_layout->spans;
    suffix.erase(suffix.begin(), suffix.begin() + first);
    if (part.multimodal_cache_spans && first) {
        const auto& all             = *part.multimodal_cache_spans;
        part.multimodal_cache_spans = std::make_shared<MultimodalCacheSpans>(all.begin() + first, all.end());
    }
    const auto&        source = *part.multimodal_request;
    MultimodalInputsPB request;
    request.set_request_id(source.request_id());
    request.set_region(source.region());
    request.set_timeout_ms(source.timeout_ms());
    request.set_inspection_ticket(source.inspection_ticket());
    if (source.has_input_inspection_enabled())
        *request.mutable_input_inspection_enabled() = source.input_inspection_enabled();
    for (int i = first; i < source.multimodal_inputs_size(); ++i)
        *request.add_multimodal_inputs() = source.multimodal_inputs(i);
    return request;
}

struct PrefillMediaJob {
    PrefillMediaJob(grpc::ServerContext* parent, bool inherit_deadline):
        call(std::make_shared<DeferredMMRpc>(parent, inherit_deadline)) {
#if USING_CUDA
        if (c10::cuda::device_count())
            device.emplace(torch::kCUDA, c10::cuda::current_device());
#endif
    }
    std::optional<c10::Device> device;
    ~PrefillMediaJob() {
        call->cancel();
    }
    std::shared_ptr<DeferredMMRpc>          call;
    std::future<ErrorResult<GenerateInput>> fetched;
};

}  // namespace

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
    return retryPrefillMedia(config, request_deadline_ms, server_context, cancellation, [&](int64_t attempt) {
        if (attempt) {
            input                      = QueryConverter::transQuery(&request);
            input->request_deadline_ms = request_deadline_ms;
            input->begin_time_us       = begin_time_us;
        }
        return preprocessForPD(input, processor, is_mtp_eagle, RoleType::PREFILL, server_context, span);
    });
}

ErrorInfo deferPrefillMedia(std::shared_ptr<GenerateInput>&                                     input,
                            const std::shared_ptr<MultimodalProcessor>&                         processor,
                            bool                                                                is_mtp_eagle,
                            const PDSepConfig&                                                  config,
                            grpc::ServerContext*                                                context,
                            const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& span,
                            const std::function<ErrorInfo()>&                                   cancellation,
                            std::function<bool(std::function<void()>)>                          submit,
                            bool                                                                inherit_rpc_deadline) {
    input->generate_config->pd_separation        = true;
    input->generate_config->force_disable_sp_run = !is_mtp_eagle;
    auto valid                                   = processor->preparePrefillLayout(*input);
    if (!valid.ok())
        return valid;
    // Keep original (pre-system-prompt) coordinates and the immutable envelope.
    // Do not retain the stream or a gRPC ServerContext from an EnqueueBatch RPC.
    auto original                     = std::make_shared<GenerateInput>();
    original->request_id              = input->request_id;
    original->generate_config         = std::make_shared<GenerateConfig>(*input->generate_config);
    original->input_ids               = input->input_ids;
    original->multimodal_token_layout = input->multimodal_token_layout;
    original->multimodal_cache_spans  = input->multimodal_cache_spans;
    original->multimodal_request      = input->multimodal_request;
    original->request_deadline_ms     = input->request_deadline_ms;
    auto       job                    = std::make_shared<PrefillMediaJob>(context, inherit_rpc_deadline);
    const auto remaining_ms           = std::max<int64_t>(0, input->request_deadline_ms - currentTimeMs());
    job->call->deadline =
        std::min(job->call->deadline, std::chrono::system_clock::now() + std::chrono::milliseconds(remaining_ms));
    input->prepare_prefill_media = [original, processor, job, span, config, cancellation, submit = std::move(submit)](
                                       GenerateInput& admitted, int reuse) -> ErrorResult<bool> {
        if (!job->fetched.valid()) {
            const auto& spans   = original->multimodal_token_layout->spans;
            const int   covered = reuse - admitted.prefix_length;
            size_t      first   = 0;
            while (first < spans.size() && int64_t(spans[first].first) + spans[first].second <= covered)
                ++first;
            if (first == spans.size())
                return true;
            auto task = std::make_shared<std::packaged_task<ErrorResult<GenerateInput>()>>(
                [original, processor, first, config, cancellation, span, device = job->device, call = job->call]()
                    -> ErrorResult<GenerateInput> {
                    const c10::OptionalDeviceGuard guard(device);
                    GenerateInput                  part    = *original;
                    auto                           request = selectPrefillImages(part, first);
                    auto                           run     = [&](int64_t) {
                        return processor->updateMultimodalFeatures(part, request, nullptr, call.get());
                    };
                    auto result = traceMultimodalWork(original->request_id, span, [&] {
                        return retryPrefillMedia(config, original->request_deadline_ms, nullptr, cancellation, run);
                    });
                    if (!result.ok())
                        return result;
                    return std::move(part);
                });
            job->fetched = task->get_future();
            if (!submit([task] { (*task)(); }))
                return {ErrorCode::CONCURRENCY_LIMIT_ERROR, "Prefill media queue is full"};
        }
        if (job->fetched.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            return false;
        auto result = job->fetched.get();
        if (!result.ok())
            return result.status();
        auto& part                   = result.value();
        admitted.mm_locs             = admitted.prefix_length ? *part.mm_locs + admitted.prefix_length : *part.mm_locs;
        admitted.multimodal_features = std::move(part.multimodal_features);
        admitted.mm_position_ids     = std::move(part.mm_position_ids);
        admitted.mm_extra_input      = std::move(part.mm_extra_input);
        return true;
    };
    input->multimodal_request.reset();
    return ErrorInfo::OkStatus();
}

}  // namespace rtp_llm
