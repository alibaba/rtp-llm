#include "rtp_llm/cpp/model_rpc/PrefillGenerateContext.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"

using grpc::Status;
using grpc::ClientContext;

namespace rtp_llm {

PrefillStatInfo::ExecuteStage PrefillStatInfo::saveStage() const {
    return stage;
}

void PrefillStatInfo::restoreStage(PrefillStatInfo::ExecuteStage stage_) {
    stage = stage_;
}

void PrefillStatInfo::nextStage() {
    stage             = static_cast<PrefillStatInfo::ExecuteStage>(static_cast<int>(stage) + 1);
    auto cost_time_us = currentTimeUs() - begin_time;
    begin_time        = currentTimeUs();
    switch (stage) {
        case getRpcConnection: {
            break;
        }
        case multimodalProcess: {
            get_rpc_connection_rt_us += cost_time_us;
            break;
        }
        case remoteAllocateResource: {
            multimodal_process_rt_us += cost_time_us;
            break;
        }
        case enqueueRequest: {
            remote_allocate_resource_rt_us += cost_time_us;
            break;
        }
        case remoteLoadCacheStart: {
            enqueue_request_rt_us += cost_time_us;
            break;
        }
        case pollLocalOutput: {
            remote_load_cache_start_rt_us += cost_time_us;
            break;
        }
        case remoteLoadCacheEnd: {
            poll_local_output_rt_us += cost_time_us;
            break;
        }
        case RemoteGenerate: {
            remote_load_cache_end_rt_us += cost_time_us;
            break;
        }
        case pollRemoteOutput: {
            remote_generate_rt_us += cost_time_us;
            break;
        }
        case finish: {
            poll_remote_output_rt_us += cost_time_us;
            break;
        }
        default: {
            RTP_LLM_CHECK_WITH_INFO(false, "error stage");
        }
    }
}

PrefillGenerateContext::~PrefillGenerateContext() {
    reportTime();
    closeGrpcStream();
    stopStream();
}

void PrefillGenerateContext::setStream(const std::shared_ptr<GenerateStream>& stream) {
    stream_ = stream;
    if (stream) {
        meta->enqueue(task_identity_, stream_);
    }
}

void PrefillGenerateContext::stopStream() {
    if (stream_) {
        // if is waiting, cancel it
        dequeueStreamFromRuntimeMeta();
        if (stream_->getStatus() != StreamState::FINISHED) {
            // The scheduler's moveToNext() runs BEFORE process() in each step(),
            // so GenerateDone set during process() won't be detected until the
            // NEXT iteration. Wait for the scheduler to move the stream to FINISHED
            // naturally, which sets FINISHED and triggers releaseResource() →
            // tryReleaseKVBlock() → insertIntoCache() to persist KV cache.
            // Only reportError for genuine errors (no GenerateDone, or hasError).
            if (!(stream_->hasEvent(StreamEvents::GenerateDone) && !stream_->hasError())) {
                stream_->reportError(ErrorCode::CANCELLED, "cancel stream");
            }
        }
        // if is running, waiting util done
        int wait_iters = 0;
        while (stream_->getStatus() == StreamState::RUNNING) {
            RTP_LLM_LOG_DEBUG("waiting prefill stream [%d] running done to cancel",
                              stream_->generateInput()->request_id);
            usleep(1000);
            if (++wait_iters > prefill_stop_stream_wait_timeout_ms_) {
                RTP_LLM_LOG_WARNING("stopStream timeout (%ld ms) waiting for Engine Loop, "
                                    "forcing cancel for request [%d]",
                                    prefill_stop_stream_wait_timeout_ms_,
                                    stream_->generateInput()->request_id);
                stream_->reportError(ErrorCode::CANCELLED, "stopStream timeout waiting for Engine Loop");
                break;
            }
        }
        // stream status will only be set to finished by scheduler.
        markRequestEnd();
        stream_.reset();
    }
}
grpc::Status PrefillGenerateContext::closeGrpcStream(const std::string& attempt_error_override,
                                                     bool               override_transport_error) {
    if (grpc_stream_closed) {
        // The first close owns the transport/application terminal state. A
        // later settlement callback is expected during stream teardown; it
        // must remain idempotent and quiet rather than suggesting that the
        // late override changed the already-finished attempt.
        return last_grpc_stream_closed_status;
    }
    grpc_stream_closed = true;
    if (cancelled() || isRequestCancelled()) {
        tryCancelDownstream();
    }
    if (client_stream) {
        client_stream->WritesDone();
        last_grpc_stream_closed_status = client_stream->Finish();
    } else {
        last_grpc_stream_closed_status = grpc::Status::OK;
    }
    // P->D CLIENT span reflects the bidi-stream terminal status;
    // idempotent, destructor of the guard is the final fallback.
    if (pd_client_span_guard) {
        // Session stage breakdown (values already accumulated by
        // PrefillStatInfo::nextStage for kmonitor): the span covers the whole
        // ALLOCATE -> load-cache -> GENERATE session. Keys mirror the
        // PrefillStatInfo field names 1:1; these three dominate the span
        // duration (allocate RTT + wait local prefill + wait remote decode
        // token stream), so their sum ~= span length minus us-level stages.
        // Written here so the final retry attempt's guard gets the settled
        // values.
        // pollRemoteOutput() calls closeGrpcStream() as its own last step,
        // i.e. before the stage is settled by the trailing nextStage() in
        // GenerateStreamCall (and this method is idempotent, so the value
        // would stay 0 forever). Settle the in-flight stage locally without
        // touching stat_info so the kmonitor path keeps its own accounting.
        int64_t poll_remote_output_rt_us = stat_info.poll_remote_output_rt_us;
        if (stat_info.stage == PrefillStatInfo::pollRemoteOutput) {
            poll_remote_output_rt_us += currentTimeUs() - stat_info.begin_time;
        }
        int64_t remote_allocate_resource_rt_us = stat_info.remote_allocate_resource_rt_us;
        if (stat_info.stage == PrefillStatInfo::remoteAllocateResource) {
            // A failed allocation closes the attempt from inside the gRPC error
            // macro, before GenerateStreamCall can advance and settle the stage.
            remote_allocate_resource_rt_us += currentTimeUs() - stat_info.begin_time;
        }
        pd_client_span_guard->setAttribute(telemetry::kAttrRtpLlmAllocateRtUs, remote_allocate_resource_rt_us);
        pd_client_span_guard->setAttribute(telemetry::kAttrRtpLlmPollLocalOutputRtUs,
                                           stat_info.poll_local_output_rt_us);
        pd_client_span_guard->setAttribute(telemetry::kAttrRtpLlmPollRemoteOutputRtUs, poll_remote_output_rt_us);
        pd_client_span_guard->setAttribute(telemetry::kAttrRpcResponseStatusCode,
                                           telemetry::grpcStatusCodeValue(last_grpc_stream_closed_status.error_code()));
        if (!attempt_error_override.empty() && (last_grpc_stream_closed_status.ok() || override_transport_error)) {
            // The caller knows the semantic first cause. Retry attempts use this
            // only when transport is OK; priority preemption explicitly keeps it
            // authoritative even when TryCancel makes Finish() return CANCELLED.
            pd_client_span_guard->setAttribute(telemetry::kAttrErrorType, attempt_error_override);
            pd_client_span_guard->finish(opentelemetry::trace::StatusCode::kError,
                                         "Prefill-to-decode RPC attempt failed before receiving a response");
        } else if (last_grpc_stream_closed_status.ok()) {
            pd_client_span_guard->finish(opentelemetry::trace::StatusCode::kOk);
        } else {
            const char* error_name = telemetry::grpcStatusCodeName(last_grpc_stream_closed_status.error_code());
            pd_client_span_guard->setAttribute(telemetry::kAttrErrorType, error_name);
            pd_client_span_guard->finish(opentelemetry::trace::StatusCode::kError,
                                         telemetry::grpcStatusDescription(last_grpc_stream_closed_status.error_code()));
        }
    }
    return last_grpc_stream_closed_status;
}

void PrefillGenerateContext::closeGrpcConnection() {
    if (!decode_addr.empty()) {
        resource->rpc_pool.removeConnection(decode_addr);
    }
}

void PrefillGenerateContext::reset() {
    const bool discard_incomplete_input = hasError() && multimodal_attempt_started_ && !multimodal_processed_;
    GenerateContext::reset();
    if (discard_incomplete_input) {
        // Rebuild from the immutable PB after an incomplete multimodal attempt.
        RTP_LLM_CHECK_WITH_INFO(!token_ids_expanded_, "incomplete multimodal input cannot contain expanded ids");
        generate_input.reset();
        multimodal_attempt_started_ = false;
    }
    // The gRPC stream keeps a raw pointer to ClientContext; destroy the stream
    // before dropping the atomically published context reference.
    client_stream.reset();
    std::atomic_store(&client_context, std::shared_ptr<grpc::ClientContext>());
    grpc_stream_closed             = false;
    last_grpc_stream_closed_status = grpc::Status::OK;
}

bool PrefillGenerateContext::isRequestCancelled() const {
    // cancel state for Async BatchRequest
    return GenerateContext::isRequestCancelled() || (cancel_state && cancel_state->load());
}

PriorityPreemptionRequestResult PrefillGenerateContext::requestPriorityPreempt() {
    PriorityPreemptionRequestResult result;
    {
        // The first-cause transition and the CANCELING overlay are one
        // observable operation relative to ordinary runtime-meta dequeue.
        std::lock_guard<std::mutex> lock(terminal_transition_mu_);
        auto                        expected = PrefillTerminalCause::ACTIVE;
        if (!terminal_cause_.compare_exchange_strong(expected,
                                                     PrefillTerminalCause::PRIORITY_PREEMPTION,
                                                     std::memory_order_acq_rel,
                                                     std::memory_order_acquire)) {
            if (expected != PrefillTerminalCause::PRIORITY_PREEMPTION) {
                return PriorityPreemptionRequestResult::REJECTED;
            }
            result = PriorityPreemptionRequestResult::ALREADY_INSTALLED;
        } else {
            if (meta) {
                meta->markPriorityPreemptionCanceling(task_identity_);
            }
            result = PriorityPreemptionRequestResult::INSTALLED;
        }
    }
    // seq_cst pairs with remoteAllocateResource's seq_cst publication/check:
    // either this thread observes the ClientContext, or the publisher observes
    // this cancel bit and cancels the newly published context.
    cancel_state->store(true, std::memory_order_seq_cst);
    // Reuse the same downstream cancellation mechanism as an upstream client
    // disconnect. Decode observes its ServerContext cancellation and follows
    // the existing stream cleanup path.
    tryCancelDownstream();
    return result;
}

bool PrefillGenerateContext::isPriorityPreempted() const {
    return terminalCause() == PrefillTerminalCause::PRIORITY_PREEMPTION;
}

bool PrefillGenerateContext::tryMarkOtherTerminal() {
    std::lock_guard<std::mutex> lock(terminal_transition_mu_);
    auto                        expected = PrefillTerminalCause::ACTIVE;
    if (terminal_cause_.compare_exchange_strong(
            expected, PrefillTerminalCause::OTHER, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return true;
    }
    return expected == PrefillTerminalCause::OTHER;
}

void PrefillGenerateContext::dequeueStreamFromRuntimeMeta() {
    std::lock_guard<std::mutex> lock(terminal_transition_mu_);
    if (meta && stream_) {
        meta->dequeue(request_id, stream_);
    }
}

PrefillTerminalCause PrefillGenerateContext::terminalCause() const {
    return terminal_cause_.load(std::memory_order_acquire);
}

void PrefillGenerateContext::tryCancelDownstream() {
    auto context = std::atomic_load(&client_context);
    if (context) {
        context->TryCancel();
    }
}

void PrefillGenerateContext::setLocalStreamSchedulerOwned(bool owned) {
    local_stream_scheduler_owned_ = owned;
}

bool PrefillGenerateContext::finalizePriorityPreemption() {
    std::lock_guard<std::mutex> lock(priority_finalize_mu_);
    if (priority_finalized_) {
        return true;
    }
    if (!isPriorityPreempted()) {
        return false;
    }

    error_info = ErrorInfo(ErrorCode::PRIORITY_PREEMPTED, "preempted by a higher-priority request");
    ErrorDetailsPB details;
    details.set_error_code(static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
    details.set_error_message(error_info.ToString());
    std::string serialized_details;
    details.SerializeToString(&serialized_details);
    error_status =
        grpc::Status(transErrorCodeToGrpc(ErrorCode::PRIORITY_PREEMPTED), error_info.ToString(), serialized_details);

    // TryCancel is only the stop trigger. Finish joins the existing P->D RPC
    // execution; Decode's cancellation finalizer runs before Finish returns.
    tryCancelDownstream();
    (void)closeGrpcStream(ErrorCodeToString(ErrorCode::PRIORITY_PREEMPTED), true);

    const auto finalized_stream = stream_;
    if (finalized_stream) {
        stream_->reportError(ErrorCode::PRIORITY_PREEMPTED, "preempted by a higher-priority request");
        // A Prefill stream is scheduler-owned once published. Retry on the
        // managed finalizer executor until the scheduler has completed its
        // terminal transition; never occupy a worker with an unbounded poll.
        if (local_stream_scheduler_owned_ && stream_->getStatus() != StreamState::FINISHED) {
            return false;
        }
        stream_->waitPendingAsyncBookkeeping();
        stream_->releaseResource();
        markRequestEnd();
        stream_.reset();
    }

    if (meta) {
        meta->markPriorityPreemptionCanceled(request_id,
                                             static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED),
                                             "preempted by a higher-priority request",
                                             finalized_stream);
    }
    priority_finalized_ = true;
    return true;
}

void PrefillGenerateContext::nextStage() {
    stat_info.nextStage();
}

void PrefillGenerateContext::markRequestEnd() {
    int64_t real_id = request_id;
    if (stream_) {
        real_id = stream_->streamId();
    }
    if (!resource->isTensorParallel()) {
        resource->cache_store->markRequestEnd(std::to_string(real_id));
        return;
    }
    const auto&           prefill_workers = resource->grpc_workers;
    RemoteFinishRequestPB finish_request;
    finish_request.set_request_id(real_id);
    for (int i = 0; i < prefill_workers.size(); i++) {
        auto& prefill_worker = prefill_workers[i];
        auto  connect_status = resource->rpc_pool.getConnection(prefill_worker);
        if (!connect_status.ok()) {
            RTP_LLM_LOG_WARNING("request [%d], get grpc connection for ip %s failed, ignore markRequestEnd for it",
                                real_id,
                                prefill_worker.c_str());
            continue;
        }
        auto          stub = connect_status.value().stub.get();
        ClientContext client_context;
        EmptyPB       response;
        auto          grpc_status = stub->RemoteFinish(&client_context, finish_request, &response);
        if (!grpc_status.ok()) {
            RTP_LLM_LOG_WARNING("request [%d], remote finish for ip %s failed, ignore markRequestEnd for it",
                                real_id,
                                prefill_worker.c_str());
            continue;
        }
    }
}

void PrefillGenerateContext::reportTime() {
    RpcMetricsCollector collector;

    collectBasicMetrics(collector);

    collector.loading_cache_request                 = loading_cache_requests;
    collector.get_rpc_connection_rt_us              = stat_info.get_rpc_connection_rt_us;
    collector.remote_allocate_resource_rt_us        = stat_info.remote_allocate_resource_rt_us;
    collector.multimodal_process_rt_us              = stat_info.multimodal_process_rt_us;
    collector.enqueue_request_rt_us                 = stat_info.enqueue_request_rt_us;
    collector.remote_load_cache_start_rt_us         = stat_info.remote_load_cache_start_rt_us;
    collector.remote_load_cache_wait_stream_rt_us   = stat_info.remote_load_cache_wait_stream_rt_us;
    collector.remote_load_cache_write_request_rt_us = stat_info.remote_load_cache_write_request_rt_us;
    collector.poll_local_output_rt_us               = stat_info.poll_local_output_rt_us;
    collector.remote_load_cache_end_rt_us           = stat_info.remote_load_cache_end_rt_us;
    collector.remote_generate_rt_us                 = stat_info.remote_generate_rt_us;
    collector.poll_remote_output_rt_us              = stat_info.poll_remote_output_rt_us;

    reportMetrics(collector);
    metrics_reporter.reset();  // avoid to report metrics in base class
}

}  // namespace rtp_llm
