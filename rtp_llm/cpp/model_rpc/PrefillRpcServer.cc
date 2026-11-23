#include <unordered_set>
#include "autil/Scope.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/Host.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalError.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/telemetry/PhaseSpanSynthesizer.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
#include <limits.h>
#include <c10/core/InferenceMode.h>
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/RpcTimeoutUtils.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/RpcCompletionQueue.h"
#include <exception>
#include <tuple>
#include <utility>

namespace rtp_llm {

namespace {

class BatchTraceCarrier: public opentelemetry::context::propagation::TextMapCarrier {
public:
    explicit BatchTraceCarrier(const TraceContextPB& context): context_(context) {}
    opentelemetry::nostd::string_view Get(opentelemetry::nostd::string_view key) const noexcept override {
        if (key == "traceparent" && context_.traceparent().size() <= 256) {
            return {context_.traceparent().data(), context_.traceparent().size()};
        }
        if (key == "tracestate" && context_.tracestate().size() <= 512) {
            return {context_.tracestate().data(), context_.tracestate().size()};
        }
        return {};
    }
    void Set(opentelemetry::nostd::string_view, opentelemetry::nostd::string_view) noexcept override {}

private:
    const TraceContextPB& context_;
};

class PrefillBatchGenerateContext final: public GenerateContext {
public:
    using GenerateContext::GenerateContext;
    ~PrefillBatchGenerateContext() override {
        try {
            auto& stream = getStream();
            if (stream && stream->hasError()) {
                recordRpcFailure(stream->firstError().error);
            } else if (!rpc_handling_completed_) {
                recordRpcFailure(
                    {std::uncaught_exceptions() > 0 ? ErrorCode::EXECUTION_EXCEPTION : ErrorCode::CANCELLED,
                     "PD batch request cleanup before completion"});
            }
            if (trace_span_guard && error_info.hasError()) {
                const auto reason = ErrorCodeToString(error_info.code());
                trace_span_guard->setLogicalErrorType(reason);
                trace_span_guard->setAttribute(telemetry::kAttrRtpLlmErrorCode,
                                               static_cast<int64_t>(error_info.code()));
                trace_span_guard->setAttribute(telemetry::kAttrRtpLlmErrorReason, reason);
            }
            synthesizeTracePhases(telemetry::PhaseRole::Prefill, std::uncaught_exceptions() > 0);
        } catch (...) {
            // Observability must not prevent the base context from cleaning up.
        }
    }
};

void addBatchSuccess(EnqueueBatchResponsePB* response, int64_t request_id) {
    response->add_successes()->set_request_id(request_id);
}

void addBatchError(EnqueueBatchResponsePB* response, int64_t request_id, ErrorCode code, const std::string& message) {
    auto* error = response->add_errors();
    error->set_request_id(request_id);
    error->mutable_error_info()->set_error_code(static_cast<int64_t>(code));
    error->mutable_error_info()->set_error_message(message);
    RTP_LLM_LOG_WARNING(
        "P2P EnqueueBatch item failed, batch_id=%ld request_id=%ld error_code=%d error_name=%s error=%s",
        response->batch_id(),
        request_id,
        static_cast<int>(code),
        ErrorCodeToString(code).c_str(),
        message.c_str());
}

}  // namespace

PrefillRpcServer::~PrefillRpcServer() {
    {
        std::lock_guard<std::mutex> lock(batch_event_owner_->mutex);
        batch_event_owner_->server = nullptr;
    }
    decltype(batch_entries_) remaining;
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        remaining.swap(batch_entries_);
    }
    for (auto& entry : remaining) {
        if (entry.second.cancel_alarm)
            entry.second.cancel_alarm();
    }
    if (hang_diag_thread_) {
        hang_diag_thread_->stop();
        hang_diag_thread_.reset();
    }
}

const char* PrefillRpcServer::stepName(int step) {
    switch (static_cast<GenerateStreamStep>(step)) {
        case GenerateStreamStep::kEntry:
            return "entry";
        case GenerateStreamStep::kAfterTransQuery:
            return "after-transQuery";
        case GenerateStreamStep::kAfterEngineEnqueue:
            return "after-engine-enqueue";
        case GenerateStreamStep::kAfterPollStream:
            return "after-pollStream";
    }
    return "unknown";
}

PrefillRpcServer::OnflightScope::OnflightScope(PrefillRpcServer* owner, int64_t request_id):
    owner_(owner), request_id_(request_id) {
    tracker_             = std::make_shared<OnflightTracker>();
    tracker_->request_id = request_id;
    tracker_->start_us   = currentTimeUs();
    tracker_->step.store(static_cast<int>(GenerateStreamStep::kEntry));
    std::lock_guard<std::mutex> lock(owner_->onflight_trackers_mutex_);
    owner_->onflight_trackers_[request_id] = tracker_;
}

PrefillRpcServer::OnflightScope::~OnflightScope() {
    std::lock_guard<std::mutex> lock(owner_->onflight_trackers_mutex_);
    owner_->onflight_trackers_.erase(request_id_);
}

void PrefillRpcServer::OnflightScope::markStep(GenerateStreamStep s) {
    if (tracker_) {
        tracker_->step.store(static_cast<int>(s));
    }
}

void PrefillRpcServer::hangDiagTick() {
    int64_t                                       now_us = currentTimeUs();
    std::vector<std::shared_ptr<OnflightTracker>> stuck;
    {
        std::lock_guard<std::mutex> lock(onflight_trackers_mutex_);
        for (const auto& [rid, t] : onflight_trackers_) {
            if ((now_us - t->start_us) / 1000 > hang_diag_warn_threshold_ms_) {
                stuck.push_back(t);
            }
        }
    }
    for (const auto& t : stuck) {
        int64_t age_ms = (now_us - t->start_us) / 1000;
        RTP_LLM_LOG_WARNING(
            "[HANG-DIAG] PrefillRpcServer::GenerateStreamCall stuck request_id=%ld age_ms=%ld last_step=%s",
            t->request_id,
            age_ms,
            stepName(t->step.load()));
    }
}

std::unique_ptr<GenerateContext> PrefillRpcServer::makeBatchRequestContext(
    const GenerateInputPB& request, const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& batch_span) {
    auto context = std::make_unique<PrefillBatchGenerateContext>(
        request.request_id(), request.generate_config().timeout_ms(), nullptr, metrics_reporter_, meta_);
    context->request_info      = QueryConverter::transRequestInfo(request.request_info());
    context->onflight_requests = &onflight_requests_;
    if (telemetry::TelemetryRuntime::isActive()) {
        opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> span;
        try {
            opentelemetry::trace::StartSpanOptions options;
            options.kind = opentelemetry::trace::SpanKind::kInternal;
            if (batch_span) {
                options.parent = batch_span->GetContext();
            }
            BatchTraceCarrier carrier(request.request_info().trace_context());
            auto propagator = opentelemetry::context::propagation::GlobalTextMapPropagator::GetGlobalPropagator();
            opentelemetry::context::Context empty_context;
            auto                            remote      = propagator->Extract(carrier, empty_context);
            const auto                      remote_span = opentelemetry::trace::GetSpan(remote);
            if (remote_span && remote_span->GetContext().IsValid() && remote_span->GetContext().IsRemote()) {
                options.parent = remote;
            }
            span = telemetry::TelemetryRuntime::tracer()->StartSpan("rtp_llm.prefill_batch_request", options);
        } catch (...) {
            // Span creation is best-effort.
        }
        context->trace_span_guard = std::make_unique<telemetry::GrpcStatusSpanGuard>(
            span, &context->error_status, telemetry::SpanStatusSemantics::Logical);
        context->trace_span_guard->setAttribute(telemetry::kAttrRequestId, std::to_string(request.request_id()));
        context->trace_span_guard->setAttribute(telemetry::kAttrGenAiPdRole, telemetry::kValPdRoleProducer);
        context->trace_span_guard->setAttribute(telemetry::kAttrRtpLlmPdSep, true);
        // FE wall-clock stamp to Prefill receipt, including routing/coalescing.
        // An unset stamp or negative cross-machine clock delta is not a sample.
        const auto received_time_us = currentTimeUs();
        if (request.start_time() > 0 && request.start_time() <= received_time_us) {
            context->trace_span_guard->setAttribute(telemetry::kAttrRtpLlmPrefillHandoffDelayUs,
                                                    received_time_us - request.start_time());
        }
    }
    return context;
}

std::function<void()> PrefillRpcServer::scheduleBatchEvent(std::weak_ptr<BatchEventOwner>         owner,
                                                           int64_t                                delay_ms,
                                                           std::function<void(PrefillRpcServer&)> callback) {
    return RpcCompletionQueue::instance().scheduleAfter(delay_ms, [owner, callback = std::move(callback)](bool ok) {
        auto state = owner.lock();
        if (!ok || !state)
            return;
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->server)
            callback(*state->server);
    });
}

void PrefillRpcServer::registerBatchAttach(const std::string&       key,
                                           const GenerateStreamPtr& stream,
                                           int64_t                  timeout_ms,
                                           int64_t                  request_deadline_ms,
                                           int64_t                  now_ms) {
    constexpr int64_t kDefaultAttachTimeoutMs = 10 * 60 * 1000;
    const auto        ttl_ms                  = timeout_ms > 0 ? timeout_ms : kDefaultAttachTimeoutMs;
    const auto        delay_ms                = std::min(ttl_ms, std::max<int64_t>(0, request_deadline_ms - now_ms));
    const auto        deadline_ms             = now_ms + delay_ms;
    uint64_t          generation;
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        auto&                       entry = batch_entries_[key];
        generation = entry.generation = ++batch_generation_;
        entry.stream                  = stream;
        entry.deadline_ms             = deadline_ms;
        entry.request_deadline_ms     = request_deadline_ms;
        entry.attach_pending          = !entry.attached;
        if (entry.attach_pending) {
            entry.cancel_alarm = scheduleBatchEvent(
                batch_event_owner_, delay_ms, [key, generation, deadline_ms](PrefillRpcServer& server) {
                    server.expireBatchAttachment(key, generation, std::max(deadline_ms, currentTimeMs()));
                });
        }
    }
    if (stream) {
        stream->onLocalCleanupProgress([owner       = std::weak_ptr<BatchEventOwner>(batch_event_owner_),
                                        weak_stream = std::weak_ptr<GenerateStream>(stream),
                                        key,
                                        generation] {
            if (auto stream = weak_stream.lock(); stream && stream->isFinished()) {
                scheduleBatchEvent(owner, 0, [key, generation](PrefillRpcServer& server) {
                    server.finishBatchStream(key, generation);
                });
            }
        });
    }
}

bool PrefillRpcServer::attachBatch(const std::string& key, int64_t now_ms) {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    auto                        it = batch_entries_.find(key);
    if (it == batch_entries_.end())
        return true;
    auto& entry = it->second;
    if (entry.attach_pending && (entry.expired || now_ms >= entry.deadline_ms))
        return false;
    entry.attached       = true;
    entry.attach_pending = false;
    entry.stream.reset();
    if (entry.cancel_alarm) {
        entry.cancel_alarm();
        entry.cancel_alarm = {};
    }
    if (!entry.reserved && !entry.context)
        batch_entries_.erase(it);
    return true;
}

void PrefillRpcServer::expireBatchAttachment(const std::string& key, uint64_t generation, int64_t now_ms) {
    GenerateStreamPtr stream;
    int64_t           request_deadline_ms;
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        auto                        it = batch_entries_.find(key);
        if (it == batch_entries_.end() || it->second.generation != generation || !it->second.attach_pending
            || now_ms < it->second.deadline_ms)
            return;
        auto& entry = it->second;
        if (entry.cancel_alarm)
            entry.cancel_alarm();
        entry.cancel_alarm  = {};
        request_deadline_ms = entry.request_deadline_ms;
        if (entry.expired) {
            if (now_ms >= request_deadline_ms && !entry.reserved && !entry.context)
                batch_entries_.erase(it);
            return;
        }

        entry.expired = true;
        stream        = entry.stream.lock();
        entry.stream.reset();
    }
    if (stream) {
        stream->reportError(ErrorCode::GENERATE_TIMEOUT, "PD batch result attachment deadline exceeded");
    }
    if (engine_ && engine_->getCacheManager()) {
        engine_->getCacheManager()->expirePrefillRequest(key, request_deadline_ms);
    }

    // Connector now holds the terminal fence. Keep the RPC rejection until
    // the request deadline, or longer if Engine still owns the batch context.
    std::lock_guard<std::mutex> lock(batch_mutex_);
    auto                        it = batch_entries_.find(key);
    if (it == batch_entries_.end() || it->second.generation != generation)
        return;
    if (now_ms < request_deadline_ms) {
        it->second.cancel_alarm = scheduleBatchEvent(
            batch_event_owner_,
            request_deadline_ms - now_ms,
            [key, generation, request_deadline_ms](PrefillRpcServer& server) {
                server.expireBatchAttachment(key, generation, std::max(request_deadline_ms, currentTimeMs()));
            });
    } else if (!it->second.reserved && !it->second.context) {
        batch_entries_.erase(it);
    }
}

void PrefillRpcServer::finishBatchStream(const std::string& key, uint64_t generation) {
    std::unique_ptr<GenerateContext> context;
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        auto                        it = batch_entries_.find(key);
        if (it == batch_entries_.end() || it->second.generation != generation || !it->second.context)
            return;
        auto& entry    = it->second;
        context        = std::move(entry.context);
        entry.reserved = false;
        if (!entry.attach_pending || (entry.expired && currentTimeMs() >= entry.request_deadline_ms)) {
            if (entry.cancel_alarm)
                entry.cancel_alarm();
            batch_entries_.erase(it);
        }
    }
    const auto request_id = context->request_id;
    context->markRpcHandlingCompleted();
    context.reset();  // Publish the final RuntimeMeta snapshot before finishLocal.
    if (auto handle = cancel_registry_->find(request_id)) {
        cancel_registry_->finishLocal(handle);
    }
}

grpc::Status PrefillRpcServer::BatchGenerateCall(grpc::ServerContext*        context,
                                                 const BatchGenerateInputPB* request,
                                                 BatchGenerateOutputsPB*     response) {
    return generateLocalBatch(context, request, response, cancel_registry_.get());
}

grpc::Status PrefillRpcServer::init(const EngineInitParams&                                maga_init_params,
                                    std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                    py::object                                             mm_process_engine) {
    auto ret = LocalRpcServer::init(maga_init_params, std::move(propose_params), mm_process_engine);
    if (!ret.ok()) {
        RTP_LLM_LOG_ERROR("prefill rpc server new2 init failed, err: %s", ret.error_message().c_str());
        return ret;
    }

    if (autil::EnvUtil::getEnv("ENABLE_PREFILL_REQUEST_LOG", false)) {
        try {
            request_logger_ = std::make_unique<PrefillRequestLogger>(
                autil::EnvUtil::getEnv("LOG_PATH", std::string("logs")),
                maga_init_params.parallelism_config.world_rank,
                maga_init_params.profiling_debug_logging_config.log_file_backup_count);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("Failed to initialize prefill request logger: %s", e.what());
        }
    }
    auto kvcache_manager = engine_->getCacheManager();
    if (!kvcache_manager) {
        RTP_LLM_LOG_WARNING("prefill rpc server new2 init failed, kvcache manager is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "kvcache manager is null");
    }
    if (!kvcache_manager->hasP2PConnector()) {
        RTP_LLM_LOG_WARNING("prefill rpc server new2 init failed, PD requires P2P connector");
        return grpc::Status(grpc::StatusCode::INTERNAL, "PD requires P2P connector");
    }

    cancel_registry_ = std::make_shared<PDCancelRegistry>(meta_);

    // Start [HANG-DIAG] watchdog: every 30s, scan onflight_trackers_ and WARN
    // about any GenerateStreamCall that has been alive longer than the
    // threshold (default 60s). Captures 5/22-style hangs where the thread
    // entered prefill but never returned and printed nothing in between.
    hang_diag_thread_ =
        autil::LoopThread::createLoopThread([this]() { hangDiagTick(); }, 30 * 1000 * 1000, "PrefillRpcServerHangDiag");
    if (!hang_diag_thread_) {
        RTP_LLM_LOG_WARNING("prefill rpc server new2: failed to start hang_diag_thread_, watchdog disabled");
    }
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::GenerateStreamCall(grpc::ServerContext*                   server_context,
                                                  const GenerateInputPB*                 request,
                                                  grpc::ServerWriter<GenerateOutputsPB>* response_writer) try {
    c10::InferenceMode inference_guard(true);
    if (!request) {
        return serializeErrorMsg("unknown", ErrorInfo(ErrorCode::INVALID_PARAMS, "null generate request"));
    }
    if (request_logger_) {
        request_logger_->log(*request);
    }
    const bool pd_separation = checkPDSupport(*request).supported;
    if (!pd_separation) {
        RTP_LLM_LOG_INFO("pd separation is disabled, call local rpc server");
        if (!cancel_registry_) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Prefill is not initialized");
        }
        return generateLocalStream(server_context, request, response_writer, cancel_registry_.get());
    }
    PDCancelRegistry::Handle cancel_handle;
    // Activated only after successful admission; destruction still follows
    // GenerateContext cleanup, so local completion cannot precede RPC joins.
    std::optional<PDCancelRegistry::CallGuard> cancel_guard;
    AtomicGuard                                request_guard(onflight_requests_);
    auto                                       request_id = request->request_id();
    // [HANG-DIAG] step 1/4: entry. Coupled with OnflightScope so the watchdog
    // can report which step a stuck request last reached even if no further
    // step log gets emitted. See 5/22 P1-B analysis where 4 prefill threads
    // entered RemoteRpcServiceImpl::GenerateStreamCall (entry log present) but
    // never produced any other log line for the remaining 3h36min.
    OnflightScope onflight_scope(this, request_id);
    auto          generate_context =
        GenerateContext(request_id, request->generate_config().timeout_ms(), server_context, metrics_reporter_, meta_);
    generate_context.onflight_requests = &onflight_requests_;
    generate_context.request_info      = QueryConverter::transRequestInfo(request->request_info());
    if (generate_context.request_info.trace_id.empty()) {
        generate_context.request_info.trace_id = request->generate_config().trace_id().value();
    }
    if (generate_context.request_info.request_id.empty()) {
        generate_context.request_info.request_id = std::to_string(request_id);
    }
    if (telemetry::TelemetryRuntime::isActive()) {
        auto span = telemetry::startRpcServerSpan(
            "rtp_llm.generate_stream_call", server_context, true, "RpcService/GenerateStreamCall");
        generate_context.trace_span_guard =
            std::make_unique<telemetry::GrpcStatusSpanGuard>(span, &generate_context.error_status);
        generate_context.trace_span_guard->setAttribute(telemetry::kAttrRequestId, std::to_string(request_id));
        generate_context.trace_span_guard->setAttribute(telemetry::kAttrGenAiPdRole, telemetry::kValPdRoleProducer);
    }
    telemetry::PhaseSpanSynthesisScope phase_span_scope([&generate_context](bool exception_unwinding) {
        generate_context.synthesizeTracePhases(telemetry::PhaseRole::Prefill, exception_unwinding);
    });
    const int                          uncaught_exceptions = std::uncaught_exceptions();
    autil::ScopeGuard                  rpc_completion_guard([&generate_context, uncaught_exceptions] {
        if (std::uncaught_exceptions() == uncaught_exceptions) {
            generate_context.markRpcHandlingCompleted();
        }
    });
    try {
        auto handoff_status = validatePDHandoff(*request);
        if (!handoff_status.ok()) {
            generate_context.error_info = handoff_status;
            generate_context.error_status =
                serializeErrorMsg(generate_context.request_key, generate_context.request_info, handoff_status);
            return generate_context.error_status;
        }
        ErrorInfo  registration_error;
        const auto local_deadline = engine_->getCacheManager()->prefillRequestDeadline(
            request->generate_config().unique_key(), request->generate_config().timeout_ms(), &registration_error);
        if (registration_error.hasError()) {
            generate_context.error_info = registration_error;
            generate_context.error_status =
                serializeErrorMsg(generate_context.request_key, generate_context.request_info, registration_error);
            return generate_context.error_status;
        }
        const auto admission = cancel_registry_->admit(
            *request, request->generate_config().unique_key(), "", local_deadline, cancel_handle);
        if (!admission.ok()) {
            generate_context.error_info = admission;
            generate_context.error_status =
                serializeErrorMsg(generate_context.request_key, generate_context.request_info, admission);
            return generate_context.error_status;
        }
        cancel_guard.emplace(*cancel_registry_, cancel_handle);
        auto input                   = QueryConverter::transQuery(request);
        input->request_deadline_ms   = local_deadline;
        const bool request_cancelled = server_context && server_context->IsCancelled();
        if (local_deadline <= currentTimeMs() || request_cancelled) {
            generate_context.error_info   = request_cancelled ?
                                                ErrorInfo(ErrorCode::CANCELLED, "prefill request cancelled") :
                                                ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "prefill request expired");
            generate_context.error_status = serializeErrorMsg(
                generate_context.request_key, generate_context.request_info, generate_context.error_info);
            return generate_context.error_status;
        }
        int64_t mm_cost_us = 0, enqueue_cost_us = 0, poll_cost_us = 0;
        int64_t phase_start = currentTimeUs();

        auto preprocess_status = preprocessPrefillWithRetry(
            *request,
            input,
            mm_processor_.get(),
            engine_->isMTPEagle(),
            maga_init_params_.pd_sep_config,
            server_context,
            generate_context.trace_span_guard ? generate_context.trace_span_guard->sharedSpan() :
                                                opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>{},
            [cancel_handle] {
                return cancel_handle->canceled.load() ? cancel_handle->cancel_reason : ErrorInfo::OkStatus();
            });
        mm_cost_us = currentTimeUs() - phase_start;
        if (!preprocess_status.ok()) {
            generate_context.error_status =
                serializeErrorMsg(generate_context.request_key, generate_context.request_info, preprocess_status);
        }
        if (generate_context.finished || generate_context.hasError()) {
            // mm_processor (or upstream CHECK_ERROR_STATUS path) early-aborted.
            // Print a paired exit log so this thread's [HANG-DIAG] trail is self-
            // contained (entry -> early-exit), and so watchdog never sees a
            // request perma-stuck at step=entry without an obvious cause.
            RTP_LLM_LOG_WARNING("[HANG-DIAG] PrefillRpcServer::GenerateStreamCall step=early-exit-at-mm, "
                                "request_id=%ld, status=%s",
                                request_id,
                                generate_context.error_status.error_message().c_str());
            return generate_context.error_status;
        }

        // [HANG-DIAG] step 2/4: transQuery + mm_processor done.
        onflight_scope.markStep(GenerateStreamStep::kAfterTransQuery);
        RTP_LLM_LOG_DEBUG("[HANG-DIAG] PrefillRpcServer::GenerateStreamCall step=after-transQuery, "
                          "request_id=%ld, mm_us=%ld",
                          request_id,
                          mm_cost_us);

        if (cancel_handle->canceled.load()) {
            generate_context.error_info   = cancel_handle->cancel_reason;
            generate_context.error_status = serializeErrorMsg(
                std::to_string(request_id), generate_context.request_info, generate_context.error_info);
            return generate_context.error_status;
        }
        phase_start     = currentTimeUs();
        auto stream     = engine_->enqueue(input);
        enqueue_cost_us = currentTimeUs() - phase_start;
        generate_context.setStream(stream);
        cancel_registry_->attach(cancel_handle, stream);

        // [HANG-DIAG] step 3/4: engine_->enqueue returned (NormalGenerateStream
        // constructed and pushed to FIFOScheduler). If this log is missing, the
        // hang is inside the stream-construction / scheduler-enqueue path; check
        // FIFOScheduler::enqueue INFO for the stream_id to disambiguate further.
        onflight_scope.markStep(GenerateStreamStep::kAfterEngineEnqueue);
        RTP_LLM_LOG_DEBUG("[HANG-DIAG] PrefillRpcServer::GenerateStreamCall step=after-engine-enqueue, "
                          "request_id=%ld, enqueue_us=%ld",
                          request_id,
                          enqueue_cost_us);

        phase_start                   = currentTimeUs();
        generate_context.error_status = pollStreamOutput(
            server_context, generate_context.request_key, response_writer, generate_context.getStream());
        poll_cost_us = currentTimeUs() - phase_start;
        // [HANG-DIAG] step 4/4: pollStreamOutput returned. The dominant phase for
        // normal requests; absence here while step 3 fired means the stream never
        // produced output (cuda/scheduler hang) or pollStream itself is stuck.
        onflight_scope.markStep(GenerateStreamStep::kAfterPollStream);

        int64_t total_us = mm_cost_us + enqueue_cost_us + poll_cost_us;
        if (total_us >= 2000000) {
            RTP_LLM_LOG_WARNING("[PD-DIAG] PrefillRpcServer slow GenerateStreamCall request_id=%ld total_us=%ld "
                                "mm_us=%ld enqueue_us=%ld poll_us=%ld",
                                request_id,
                                total_us,
                                mm_cost_us,
                                enqueue_cost_us,
                                poll_cost_us);
        }
        // Final exit log paired with the entry log. WARN if non-OK so error
        // paths (pollStreamOutput surfacing stream error, etc.) are always
        // visible regardless of total_us; INFO for healthy exit gives the
        // watchdog a definitive "request done" marker.
        if (!generate_context.error_status.ok()) {
            RTP_LLM_LOG_WARNING("[HANG-DIAG] PrefillRpcServer::GenerateStreamCall step=exit-error, "
                                "request_id=%ld, total_us=%ld, status=%s",
                                request_id,
                                total_us,
                                generate_context.error_status.error_message().c_str());
        }
        meta_->dequeue(generate_context.request_id, generate_context.getStream());
        return generate_context.error_status;
    } catch (const std::invalid_argument& error) {
        generate_context.recordRpcFailure({ErrorCode::INVALID_PARAMS, error.what()});
        throw;
    } catch (const std::exception& error) {
        generate_context.recordRpcFailure({ErrorCode::EXECUTION_EXCEPTION, error.what()});
        throw;
    } catch (...) {
        generate_context.recordRpcFailure({ErrorCode::EXECUTION_EXCEPTION, "unknown PD handler exception"});
        throw;
    }
} catch (const std::invalid_argument& e) {
    const auto request_key = request ? std::to_string(request->request_id()) : "unknown";
    const auto error_info =
        ErrorInfo(ErrorCode::INVALID_PARAMS, "request [" + request_key + "] catch invalid argument [" + e.what() + "]");
    return serializeErrorMsg(
        request_key, request ? QueryConverter::transRequestInfo(request->request_info()) : RequestInfo(), error_info);
} catch (const std::exception& e) {
    const auto request_key = request ? std::to_string(request->request_id()) : "unknown";
    return serializeErrorMsg(
        request_key,
        request ? QueryConverter::transRequestInfo(request->request_info()) : RequestInfo(),
        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                  "request [" + request_key + "] catch exception [" + std::string(e.what()) + "]"));
} catch (...) {
    const auto request_key = request ? std::to_string(request->request_id()) : "unknown";
    return serializeErrorMsg(
        request_key,
        request ? QueryConverter::transRequestInfo(request->request_info()) : RequestInfo(),
        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "request [" + request_key + "] catch unknown exception"));
}

void PrefillRpcServer::beginShutdown() {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    batch_stopping_ = true;
}

size_t PrefillRpcServer::onflightRequestNum() {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    const auto                  now_ms = currentTimeMs();
    size_t                      count  = onflight_requests_.load();
    for (const auto& [key, entry] : batch_entries_) {
        (void)key;
        // Published KV is still in flight until Decode attaches or its bounded
        // attachment window expires. Expired rejection fences need no drain.
        if (entry.reserved || entry.context || (entry.attach_pending && !entry.expired && now_ms < entry.deadline_ms)) {
            ++count;
        }
    }
    return count;
}

grpc::Status PrefillRpcServer::EnqueueBatch(grpc::ServerContext*         context,
                                            const EnqueueBatchRequestPB* request,
                                            EnqueueBatchResponsePB*      response) {
    grpc::Status trace_status;
    auto batch_span = telemetry::startRpcServerSpan("rtp_llm.enqueue_batch", context, true, "RpcService/EnqueueBatch");
    telemetry::GrpcStatusSpanGuard batch_trace(batch_span, &trace_status);
    batch_trace.setAttribute(telemetry::kAttrGenAiPdRole, telemetry::kValPdRoleProducer);
    c10::InferenceMode inference_guard(true);
    RTP_LLM_PROFILE_FUNCTION();
    AtomicGuard request_guard(onflight_requests_);
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        if (batch_stopping_) {
            trace_status = grpc::Status(grpc::StatusCode::UNAVAILABLE, "Prefill batch server is shutting down");
            return trace_status;
        }
    }
    response->Clear();
    response->set_batch_id(request->batch_id());

    const auto& parallelism_config = maga_init_params_.parallelism_config;
    if (parallelism_config.dp_size != 1) {
        trace_status = grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "P2P EnqueueBatch requires single-DP mode");
        return trace_status;
    }
    const int local_dp_rank = static_cast<int>(parallelism_config.dp_rank);

    int                         input_count = 0;
    std::unordered_set<int64_t> request_ids;
    bool                        duplicate_request_id = false;
    for (const auto& dp_slot : request->dp_slots()) {
        for (const auto& external_input : dp_slot.requests()) {
            ++input_count;
            if (external_input.has_input() && !request_ids.insert(external_input.input().request_id()).second) {
                duplicate_request_id = true;
            }
        }
    }
    if (duplicate_request_id) {
        for (const auto& dp_slot : request->dp_slots()) {
            for (const auto& external_input : dp_slot.requests()) {
                addBatchError(response,
                              external_input.has_input() ? external_input.input().request_id() : 0,
                              ErrorCode::INVALID_PARAMS,
                              "duplicate request_id in P2P EnqueueBatch");
            }
        }
        return grpc::Status::OK;
    }

    std::vector<std::shared_ptr<GenerateInput>>   inputs;
    std::vector<int64_t>                          admitted_request_ids;
    std::vector<std::unique_ptr<GenerateContext>> admitted_contexts;
    const auto                                    release_request_id = [this](int64_t request_id) {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        batch_entries_.erase(masterEnqueuedHandoffUniqueKey(request_id));
    };
    const auto release_admitted_request_ids =
        [this, &admitted_request_ids, &admitted_contexts](const ErrorInfo& error) {
            for (auto& context : admitted_contexts) {
                if (context) {
                    context->recordRpcFailure(error);
                    context->markRpcHandlingCompleted();
                }
            }
            std::lock_guard<std::mutex> lock(batch_mutex_);
            for (const auto request_id : admitted_request_ids) {
                batch_entries_.erase(masterEnqueuedHandoffUniqueKey(request_id));
                if (auto handle = cancel_registry_->find(request_id))
                    cancel_registry_->finishLocal(handle);
            }
        };
    inputs.reserve(input_count);
    admitted_request_ids.reserve(input_count);
    admitted_contexts.reserve(input_count);
    for (const auto& dp_slot : request->dp_slots()) {
        for (const auto& external_input : dp_slot.requests()) {
            if (!external_input.has_input()) {
                addBatchError(response, 0, ErrorCode::INVALID_PARAMS, "P2P EnqueueBatch input is missing");
                continue;
            }
            const auto request_id = external_input.input().request_id();
            if (external_input.input().multimodal_inputs_size() != 0 || external_input.input().has_v41_inputs()
                || external_input.input().has_multimodal_token_layout()) {
                addBatchError(
                    response, request_id, ErrorCode::INVALID_PARAMS, "EnqueueBatch does not support multimodal inputs");
                continue;
            }
            const auto& config = external_input.input().generate_config();
            if (config.num_beams() > 1 || config.variable_num_beams_size() != 0) {
                addBatchError(
                    response, request_id, ErrorCode::INVALID_PARAMS, "EnqueueBatch does not support beam search");
                continue;
            }
            if (config.num_return_sequences() > 1) {
                addBatchError(response,
                              request_id,
                              ErrorCode::INVALID_PARAMS,
                              "EnqueueBatch does not support multiple return sequences");
                continue;
            }
            if (dp_slot.dp_rank() != local_dp_rank) {
                addBatchError(response,
                              request_id,
                              ErrorCode::INVALID_PARAMS,
                              "P2P EnqueueBatch dp_rank mismatch, request dp_rank " + std::to_string(dp_slot.dp_rank())
                                  + ", local dp_rank " + std::to_string(local_dp_rank));
                continue;
            }
            if (context && context->IsCancelled()) {
                release_admitted_request_ids({ErrorCode::CANCELLED, "EnqueueBatch cancelled before admission"});
                trace_status = grpc::Status(grpc::StatusCode::CANCELLED, "P2P EnqueueBatch cancelled before admission");
                return trace_status;
            }
            {
                std::lock_guard<std::mutex> lock(batch_mutex_);
                if (!batch_entries_.try_emplace(masterEnqueuedHandoffUniqueKey(request_id)).second) {
                    addBatchError(response,
                                  request_id,
                                  ErrorCode::INVALID_PARAMS,
                                  "request_id is already active in P2P EnqueueBatch");
                    continue;
                }
                batch_entries_.at(masterEnqueuedHandoffUniqueKey(request_id)).reserved = true;
            }

            PDCancelRegistry::Handle         cancel_handle;
            std::unique_ptr<GenerateContext> item_context;
            const auto                       reject_item = [&](ErrorCode code, const std::string& message) {
                if (item_context) {
                    item_context->recordRpcFailure({code, message});
                    item_context->markRpcHandlingCompleted();
                }
                if (cancel_handle) {
                    cancel_registry_->finishLocal(cancel_handle);
                    cancel_registry_->finishDownstream(cancel_handle);
                    cancel_registry_->complete(cancel_handle);
                }
                release_request_id(request_id);
                addBatchError(response, request_id, code, message);
            };
            try {
                GenerateInputPB item;
                item.CopyFrom(external_input.input());
                if (request_logger_) {
                    request_logger_->log(item);
                }
                item.mutable_group_id()->set_value(request->batch_id());
                auto* config = item.mutable_generate_config();
                config->set_unique_key(masterEnqueuedHandoffUniqueKey(request_id));
                config->set_timeout_ms(clampRpcTimeoutMsToInt32(
                    normalizeRpcTimeoutMs(config->timeout_ms(), maga_init_params_.pd_sep_config.max_rpc_timeout_ms)));
                item_context               = makeBatchRequestContext(item, batch_span);
                auto input                 = QueryConverter::transQuery(&item);
                item_context->request_info = input->request_info;
                if (input->generate_config->isPrefillOnly()) {
                    reject_item(ErrorCode::INVALID_PARAMS, "EnqueueBatch does not support prefill-only inputs");
                    continue;
                }
                const auto support = checkPDSupport(item);
                if (!support.supported) {
                    reject_item(ErrorCode::INVALID_PARAMS, support.reason);
                    continue;
                }
                ErrorInfo registration_error;
                input->request_deadline_ms = engine_->getCacheManager()->prefillRequestDeadline(
                    config->unique_key(), config->timeout_ms(), &registration_error);
                if (registration_error.hasError()) {
                    reject_item(registration_error.code(), registration_error.ToString());
                    continue;
                }
                if (input->request_deadline_ms <= currentTimeMs()) {
                    reject_item(ErrorCode::GENERATE_TIMEOUT, "P2P EnqueueBatch request expired before admission");
                    continue;
                }
                const auto admission =
                    cancel_registry_->admit(item, config->unique_key(), "", input->request_deadline_ms, cancel_handle);
                if (!admission.ok()) {
                    reject_item(admission.code(), admission.ToString());
                    continue;
                }
                const auto item_span         = item_context->trace_span_guard ?
                                                   item_context->trace_span_guard->sharedSpan() :
                                                   opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>{};
                const auto preprocess_status = preprocessPrefillWithRetry(item,
                                                                          input,
                                                                          mm_processor_.get(),
                                                                          engine_->isMTPEagle(),
                                                                          maga_init_params_.pd_sep_config,
                                                                          context,
                                                                          item_span,
                                                                          [cancel_handle] {
                                                                              return cancel_handle->canceled.load() ?
                                                                                         cancel_handle->cancel_reason :
                                                                                         ErrorInfo::OkStatus();
                                                                          });
                if (!preprocess_status.ok()) {
                    reject_item(preprocess_status.code(), preprocess_status.ToString());
                    continue;
                }
                if (cancel_handle->canceled.load()) {
                    reject_item(cancel_handle->cancel_reason.code(), cancel_handle->cancel_reason.ToString());
                    continue;
                }
                admitted_request_ids.push_back(request_id);
                inputs.push_back(std::move(input));
                admitted_contexts.push_back(std::move(item_context));
            } catch (const std::invalid_argument& e) {
                reject_item(ErrorCode::INVALID_PARAMS, e.what());
            } catch (const std::exception& e) {
                reject_item(ErrorCode::EXECUTION_EXCEPTION, e.what());
            } catch (...) {
                reject_item(ErrorCode::EXECUTION_EXCEPTION, "PD batch input preparation failed");
            }
        }
    }
    if (inputs.empty()) {
        return grpc::Status::OK;
    }
    // Group membership is the batch submitted to the engine, after per-item
    // validation/preprocessing. Rejected inputs must not inflate group_size.
    for (const auto& input : inputs) {
        input->group_size = static_cast<int>(inputs.size());
    }

    std::vector<bool>              enqueue_successes;
    std::vector<GenerateStreamPtr> streams;
    try {
        std::tie(enqueue_successes, streams) = engine_->enqueueMultiple(inputs);
    } catch (const std::exception& e) {
        release_admitted_request_ids({ErrorCode::EXECUTION_EXCEPTION, e.what()});
        trace_status = grpc::Status(grpc::StatusCode::INTERNAL,
                                    "P2P EnqueueBatch enqueueMultiple exception: " + std::string(e.what()));
        return trace_status;
    } catch (...) {
        release_admitted_request_ids({ErrorCode::EXECUTION_EXCEPTION, "enqueueMultiple unknown exception"});
        trace_status = grpc::Status(grpc::StatusCode::INTERNAL, "P2P EnqueueBatch enqueueMultiple unknown exception");
        return trace_status;
    }
    if (enqueue_successes.size() != inputs.size() || streams.size() != inputs.size()) {
        release_admitted_request_ids({ErrorCode::EXECUTION_EXCEPTION, "enqueueMultiple result size mismatch"});
        trace_status = grpc::Status(grpc::StatusCode::INTERNAL, "P2P EnqueueBatch result size mismatch");
        return trace_status;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (enqueue_successes[i]) {
            auto batch_context = std::move(admitted_contexts[i]);
            batch_context->setStream(streams[i]);
            cancel_registry_->attach(cancel_registry_->find(admitted_request_ids[i]), streams[i]);
            {
                std::lock_guard<std::mutex> lock(batch_mutex_);
                batch_entries_.at(inputs[i]->generate_config->unique_key).context = std::move(batch_context);
            }
            registerBatchAttach(inputs[i]->generate_config->unique_key,
                                streams[i],
                                request->fetch_attach_timeout_ms(),
                                inputs[i]->request_deadline_ms,
                                currentTimeMs());
            addBatchSuccess(response, admitted_request_ids[i]);
            continue;
        }
        auto error = streams[i] ? streams[i]->statusInfo() : ErrorInfo(ErrorCode::UNKNOWN_ERROR, "null stream");
        admitted_contexts[i]->recordRpcFailure(
            error.hasError() ?
                error :
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "scheduler rejected request"));
        admitted_contexts[i]->markRpcHandlingCompleted();
        if (auto handle = cancel_registry_->find(admitted_request_ids[i])) {
            cancel_registry_->finishLocal(handle);
            cancel_registry_->finishDownstream(handle);
        }
        release_request_id(admitted_request_ids[i]);
        addBatchError(response,
                      admitted_request_ids[i],
                      error.hasError() ? error.code() : ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                      error.hasError() ? error.ToString() : "scheduler rejected request");
    }
    RTP_LLM_CHECK_WITH_INFO(response->successes_size() + response->errors_size() == input_count,
                            "P2P EnqueueBatch result size mismatch: request=%d response=%d",
                            input_count,
                            response->successes_size() + response->errors_size());
    return grpc::Status::OK;
}

::grpc::Status PrefillRpcServer::StartLoad(::grpc::ServerContext*                context,
                                           const P2PConnectorStartLoadRequestPB* request,
                                           P2PConnectorStartLoadResponsePB*      response) try {
    AtomicGuard request_guard(onflight_requests_);
    RTP_LLM_LOG_DEBUG(
        "StartLoad gRPC entry, unique_key=%s, peer=%s", request->unique_key().c_str(), context->peer().c_str());
    RTP_LLM_LOG_DEBUG("receive start load request from client: %s, request: [%s]",
                      context->peer().c_str(),
                      request->DebugString().c_str());
    const std::string error_context = "reporter_role=PREFILL stage=StartLoad key=" + request->unique_key()
                                      + " tp_rank=" + std::to_string(maga_init_params_.parallelism_config.tp_rank);
    if (!engine_) {
        RTP_LLM_LOG_WARNING("P2P Prefill StartLoad failed, unique_key=%s peer=%s grpc_code=%d error=engine is null",
                            request->unique_key().c_str(),
                            context->peer().c_str(),
                            static_cast<int>(grpc::StatusCode::INTERNAL));
        return grpcStatusFromErrorInfo(
            ErrorInfo(ErrorCode::P2P_CONNECTOR_TRANSFER_NOT_INITIALIZED, error_context + ": engine is null"));
    }
    auto cache_manager = engine_->getCacheManager();
    if (!cache_manager) {
        RTP_LLM_LOG_WARNING(
            "P2P Prefill StartLoad failed, unique_key=%s peer=%s grpc_code=%d error=cache manager is null",
            request->unique_key().c_str(),
            context->peer().c_str(),
            static_cast<int>(grpc::StatusCode::INTERNAL));
        return grpcStatusFromErrorInfo(
            ErrorInfo(ErrorCode::P2P_CONNECTOR_TRANSFER_NOT_INITIALIZED, error_context + ": cache manager is null"));
    }
    if (!attachBatch(request->unique_key(), currentTimeMs())) {
        response->set_error_code(transErrorCodeToRPC(ErrorCode::GENERATE_TIMEOUT));
        response->set_error_message("PD batch result attachment deadline exceeded");
        return grpc::Status::OK;
    }
    PDCancelRegistry::ReadGuard read_guard(*cancel_registry_, request->unique_key());
    auto                        is_cancelled = [this, context, key = request->unique_key()]() {
        return context->IsCancelled() || cancel_registry_->isCanceled(key);
    };
    int64_t handle_read_start = currentTimeUs();
    cache_manager->handleRead(*request, *response, std::move(is_cancelled));
    int64_t handle_read_cost = currentTimeUs() - handle_read_start;
    if (response->error_code() != ErrorCodePB::NONE_ERROR) {
        response->set_error_message(error_context + ": " + response->error_message());
        RTP_LLM_LOG_WARNING("P2P Prefill StartLoad failed, unique_key=%s peer=%s wire_code=%d "
                            "error_code=%d error_name=%s error=%s",
                            request->unique_key().c_str(),
                            context->peer().c_str(),
                            static_cast<int>(response->error_code()),
                            static_cast<int>(transRPCErrorCode(response->error_code())),
                            ErrorCodeToString(transRPCErrorCode(response->error_code())).c_str(),
                            response->error_message().c_str());
    }
    if (handle_read_cost >= 2000000) {
        RTP_LLM_LOG_WARNING("[PD-DIAG] StartLoad slow handleRead cost_us=%ld, unique_key=%s",
                            handle_read_cost,
                            request->unique_key().c_str());
    }
    return grpc::Status::OK;
} catch (const std::exception& error) {
    RTP_LLM_LOG_ERROR("Prefill StartLoad failed for key=%s: %s", request->unique_key().c_str(), error.what());
    return grpc::Status(grpc::StatusCode::INTERNAL, "Prefill StartLoad failed");
} catch (...) {
    RTP_LLM_LOG_ERROR("Prefill StartLoad failed for key=%s with an unknown exception", request->unique_key().c_str());
    return grpc::Status(grpc::StatusCode::INTERNAL, "Prefill StartLoad failed");
}

grpc::Status
PrefillRpcServer::Cancel(grpc::ServerContext*, const CancelRequestPB* request, CancelResponsePB* response) {
    if (!request || request->request_id() <= 0 || !response)
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "cancel request missing request_id");
    if (!cancel_registry_)
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Prefill is not initialized");
    const auto cancellation = cancel_registry_->cancel(
        request->request_id(),
        {request->cancel_error_code() == 0 ? ErrorCode::PRIORITY_PREEMPTED :
                                             static_cast<ErrorCode>(request->cancel_error_code()),
         request->cancel_error_message().empty() ? "PD request canceled" : request->cancel_error_message()},
        [this](const auto& handle) {
            // A completed Prefill can still have unconsumed KV. Fence it
            // before the registry can publish local cleanup.
            if (!handle->unique_key.empty() && engine_ && engine_->getCacheManager()) {
                engine_->getCacheManager()->cancelPrefillRequest(handle->unique_key, handle->deadline_ms);
            }
        });
    if (!cancellation.ok()) {
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, std::string(cancellation.status().message()));
    }
    response->set_status(cancellation.value());
    return grpc::Status::OK;
}

::grpc::Status PrefillRpcServer::GetPeerInfo(::grpc::ServerContext*      context,
                                             const GetPeerInfoRequestPB* request,
                                             GetPeerInfoResponsePB*      response) {
    const auto& pc = maga_init_params_.parallelism_config;
    if (pc.tp_size <= 0) {
        return grpcStatusFromErrorInfo(
            ErrorInfo(ErrorCode::INVALID_PARAMS, "GetPeerInfo invalid tp_size=" + std::to_string(pc.tp_size)));
    }
    response->set_tp_size(static_cast<int32_t>(pc.tp_size));
    const auto& cp = pc.prefill_cp_config;
    response->set_cp_size(static_cast<int32_t>(cp.is_enabled() && cp.kv_cache_sharded ? pc.tp_size : 1));
    // Advertise actual cache geometry for single-stage peers as well as pipelines.
    if (engine_ && engine_->getCacheManager()) {
        response->set_pipeline_layout(engine_->getCacheManager()->p2pPipelineLayout());
    }
    RTP_LLM_LOG_INFO("GetPeerInfo: tp_size=%ld, cp_size=%d", pc.tp_size, response->cp_size());
    return grpc::Status::OK;
}

}  // namespace rtp_llm
