#include <unordered_set>
#include <c10/core/InferenceMode.h>
#include "rtp_llm/cpp/model_rpc/PrefillRpcServerNew2.h"
#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <utility>

namespace rtp_llm {

PrefillRpcServerNew2::~PrefillRpcServerNew2() {
    if (hang_diag_thread_) {
        hang_diag_thread_->stop();
        hang_diag_thread_.reset();
    }
}

const char* PrefillRpcServerNew2::stepName(int step) {
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

PrefillRpcServerNew2::OnflightScope::OnflightScope(PrefillRpcServerNew2* owner, int64_t request_id):
    owner_(owner), request_id_(request_id) {
    tracker_             = std::make_shared<OnflightTracker>();
    tracker_->request_id = request_id;
    tracker_->start_us   = currentTimeUs();
    tracker_->step.store(static_cast<int>(GenerateStreamStep::kEntry));
    std::lock_guard<std::mutex> lock(owner_->onflight_trackers_mutex_);
    owner_->onflight_trackers_[request_id] = tracker_;
}

PrefillRpcServerNew2::OnflightScope::~OnflightScope() {
    std::lock_guard<std::mutex> lock(owner_->onflight_trackers_mutex_);
    owner_->onflight_trackers_.erase(request_id_);
}

void PrefillRpcServerNew2::OnflightScope::markStep(GenerateStreamStep s) {
    if (tracker_) {
        tracker_->step.store(static_cast<int>(s));
    }
}

void PrefillRpcServerNew2::hangDiagTick() {
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
        RTP_LLM_LOG_WARNING("[HANG-DIAG] PrefillNew2::GenerateStreamCall stuck request_id=%ld age_ms=%ld last_step=%s",
                            t->request_id,
                            age_ms,
                            stepName(t->step.load()));
    }
}

grpc::Status PrefillRpcServerNew2::init(const EngineInitParams&                                maga_init_params,
                                        std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                        py::object                                             mm_process_engine) {
    // decode_entrance path uses P2P connector for KV cache transfer, not CacheStore.
    // Skip CacheStore init to avoid double-registering the same GPU memory region
    // (CacheStore registers the full KVCache area; P2P connector registers per-block).
    auto ret = RemoteRpcServer::init(maga_init_params, std::move(propose_params), mm_process_engine, false);
    if (!ret.ok()) {
        RTP_LLM_LOG_ERROR("prefill rpc server new2 init failed, err: %s", ret.error_message().c_str());
        return ret;
    }

    auto kvcache_manager = engine_->getCacheManager();
    if (!kvcache_manager) {
        RTP_LLM_LOG_WARNING("prefill rpc server new2 init failed, kvcache manager is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "kvcache manager is null");
    }
    if (!kvcache_manager->hasP2PConnector()) {
        RTP_LLM_LOG_WARNING("prefill rpc server new2 init failed, decode_entrance requires P2P connector");
        return grpc::Status(grpc::StatusCode::INTERNAL, "decode_entrance requires P2P connector");
    }

    // Start [HANG-DIAG] watchdog: every 30s, scan onflight_trackers_ and WARN
    // about any GenerateStreamCall that has been alive longer than the
    // threshold (default 60s). Captures 5/22-style hangs where the thread
    // entered prefill but never returned and printed nothing in between.
    hang_diag_thread_ =
        autil::LoopThread::createLoopThread([this]() { hangDiagTick(); }, 30 * 1000 * 1000, "PrefillNew2HangDiag");
    if (!hang_diag_thread_) {
        RTP_LLM_LOG_WARNING("prefill rpc server new2: failed to start hang_diag_thread_, watchdog disabled");
    }
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServerNew2::GenerateStreamCall(grpc::ServerContext*                   server_context,
                                                      const GenerateInputPB*                 request,
                                                      grpc::ServerWriter<GenerateOutputsPB>* response_writer) {
    const bool pd_separation = checkPDSupport(*request).supported;
    if (!pd_separation) {
        RTP_LLM_LOG_INFO("pd separation is disabled, call local rpc server");
        return LocalRpcServer::GenerateStreamCall(server_context, request, response_writer);
    }
    auto handoff_status = validatePDHandoff(*request);
    if (!handoff_status.ok()) {
        return serializeErrorMsg(std::to_string(request->request_id()), handoff_status);
    }

    const auto local_deadline = engine_->getCacheManager()->prefillRequestDeadline(
        request->generate_config().unique_key(), request->generate_config().timeout_ms());
    AtomicGuard request_guard(onflight_requests_);
    auto        request_id = request->request_id();
    // [HANG-DIAG] step 1/4: entry. Coupled with OnflightScope so the watchdog
    // can report which step a stuck request last reached even if no further
    // step log gets emitted. See 5/22 P1-B analysis where 4 prefill threads
    // entered RemoteRpcServiceImpl::GenerateStreamCall (entry log present) but
    // never produced any other log line for the remaining 3h36min.
    OnflightScope onflight_scope(this, request_id);
    auto generate_context =
        GenerateContext(request_id, request->generate_config().timeout_ms(), server_context, metrics_reporter_, meta_);
    auto input                            = QueryConverter::transQuery(request);
    input->request_deadline_ms = local_deadline;
    if (local_deadline <= currentTimeMs() || server_context->IsCancelled()) {
        return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "prefill request expired");
    }
    int64_t mm_cost_us = 0, enqueue_cost_us = 0, poll_cost_us = 0;
    int64_t phase_start = currentTimeUs();

    auto preprocess_status = preprocessForPD(input, mm_processor_.get(), engine_->isMTPEagle());
    mm_cost_us = currentTimeUs() - phase_start;
    if (!preprocess_status.ok()) {
        generate_context.error_status = serializeErrorMsg(generate_context.request_key, preprocess_status);
    }
    if (generate_context.finished || generate_context.hasError()) {
        // mm_processor (or upstream CHECK_ERROR_STATUS path) early-aborted.
        // Print a paired exit log so this thread's [HANG-DIAG] trail is self-
        // contained (entry -> early-exit), and so watchdog never sees a
        // request perma-stuck at step=entry without an obvious cause.
        RTP_LLM_LOG_WARNING("[HANG-DIAG] PrefillNew2::GenerateStreamCall step=early-exit-at-mm, "
                            "request_id=%ld, status=%s",
                            request_id,
                            generate_context.error_status.error_message().c_str());
        return generate_context.error_status;
    }

    // [HANG-DIAG] step 2/4: transQuery + mm_processor done.
    onflight_scope.markStep(GenerateStreamStep::kAfterTransQuery);
    RTP_LLM_LOG_DEBUG("[HANG-DIAG] PrefillNew2::GenerateStreamCall step=after-transQuery, "
                      "request_id=%ld, mm_us=%ld",
                      request_id,
                      mm_cost_us);

    phase_start     = currentTimeUs();
    auto stream     = engine_->enqueue(input);
    enqueue_cost_us = currentTimeUs() - phase_start;
    generate_context.setStream(stream);

    // [HANG-DIAG] step 3/4: engine_->enqueue returned (NormalGenerateStream
    // constructed and pushed to FIFOScheduler). If this log is missing, the
    // hang is inside the stream-construction / scheduler-enqueue path; check
    // FIFOScheduler::enqueue INFO for the stream_id to disambiguate further.
    onflight_scope.markStep(GenerateStreamStep::kAfterEngineEnqueue);
    RTP_LLM_LOG_DEBUG("[HANG-DIAG] PrefillNew2::GenerateStreamCall step=after-engine-enqueue, "
                      "request_id=%ld, enqueue_us=%ld",
                      request_id,
                      enqueue_cost_us);

    phase_start = currentTimeUs();
    generate_context.error_status =
        pollStreamOutput(server_context, generate_context.request_key, response_writer, generate_context.getStream());
    poll_cost_us = currentTimeUs() - phase_start;
    // [HANG-DIAG] step 4/4: pollStreamOutput returned. The dominant phase for
    // normal requests; absence here while step 3 fired means the stream never
    // produced output (cuda/scheduler hang) or pollStream itself is stuck.
    onflight_scope.markStep(GenerateStreamStep::kAfterPollStream);

    int64_t total_us = mm_cost_us + enqueue_cost_us + poll_cost_us;
    if (total_us >= 2000000) {
        RTP_LLM_LOG_WARNING("[PD-DIAG] PrefillNew2 slow GenerateStreamCall request_id=%ld total_us=%ld "
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
        RTP_LLM_LOG_WARNING("[HANG-DIAG] PrefillNew2::GenerateStreamCall step=exit-error, "
                            "request_id=%ld, total_us=%ld, status=%s",
                            request_id,
                            total_us,
                            generate_context.error_status.error_message().c_str());
    }
    meta_->dequeue(generate_context.request_id, generate_context.getStream());
    return generate_context.error_status;
}

grpc::Status PrefillRpcServerNew2::BatchGenerateCall(grpc::ServerContext*        context,
                                                     const BatchGenerateInputPB* request,
                                                     BatchGenerateOutputsPB*     response) {
    c10::InferenceMode inference_guard(true);
    response->Clear();
    if (request->inputs_size() == 0)
        return grpc::Status::OK;
    bool pd      = false;
    auto support = checkPDBatchSupport(*request, pd);
    if (!support.ok())
        return serializeErrorMsg("batch", support);
    if (!pd)
        return LocalRpcServer::BatchGenerateCall(context, request, response);
    AtomicGuard                                 request_guard(onflight_requests_);
    std::vector<std::shared_ptr<GenerateInput>> inputs;
    std::unordered_set<std::string>             keys;
    std::vector<int64_t> local_deadlines;
    for (int i = 0; i < request->inputs_size(); ++i) {
        const auto& item   = request->inputs(i);
        auto        status = validatePDHandoff(item);
        if (!status.ok())
            return serializeErrorMsg("batch item " + std::to_string(i), status);
        local_deadlines.push_back(engine_->getCacheManager()->prefillRequestDeadline(
            item.generate_config().unique_key(), item.generate_config().timeout_ms()));
        if (!keys.insert(item.generate_config().unique_key()).second) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "duplicate batch handoff key");
        }
    }
    for (int i = 0; i < request->inputs_size(); ++i) {
        if (context->IsCancelled())
            return grpc::Status(grpc::StatusCode::CANCELLED, "batch cancelled by user");
        const auto& item   = request->inputs(i);
        auto        input  = QueryConverter::transQuery(&item);
        input->request_deadline_ms = local_deadlines[i];
        auto        status = preprocessForPD(input, mm_processor_.get(), engine_->isMTPEagle());
        if (!status.ok())
            return serializeErrorMsg("batch item " + std::to_string(i), status);
        inputs.push_back(std::move(input));
    }
    std::vector<GenerateStreamPtr>                streams;
    std::vector<std::unique_ptr<GenerateContext>> contexts;
    std::vector<std::unique_ptr<OnflightScope>>   scopes;
    for (const auto& input : inputs) {
        auto stream = engine_->makeStream(input);
        auto item   = std::make_unique<GenerateContext>(
            input->request_id, input->generate_config->timeout_ms, context, metrics_reporter_, meta_);
        item->setStream(stream);
        streams.push_back(std::move(stream));
        contexts.push_back(std::move(item));
        scopes.emplace_back(std::make_unique<OnflightScope>(this, input->request_id));
        scopes.back()->markStep(GenerateStreamStep::kAfterTransQuery);
    }
    // Recheck the whole batch after MM processing, before queue admission.
    for (const auto& input : inputs) {
        if (input->request_deadline_ms <= currentTimeMs())
            return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "prefill batch request expired");
    }
    if (context->IsCancelled())
        return grpc::Status(grpc::StatusCode::CANCELLED, "batch cancelled by user");
    if (engine_->batchEnqueue(streams) != streams) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "batchEnqueue changed prepared stream identity or order");
    }
    for (auto& scope : scopes)
        scope->markStep(GenerateStreamStep::kAfterEngineEnqueue);
    auto status = pollBatchStreamOutput(context, streams, response);
    for (auto& item : contexts)
        item->error_status = status;
    for (auto& scope : scopes)
        scope->markStep(GenerateStreamStep::kAfterPollStream);
    return status;
}

::grpc::Status PrefillRpcServerNew2::StartLoad(::grpc::ServerContext*                context,
                                               const P2PConnectorStartLoadRequestPB* request,
                                               P2PConnectorStartLoadResponsePB*      response) {
    RTP_LLM_LOG_DEBUG(
        "StartLoad gRPC entry, unique_key=%s, peer=%s", request->unique_key().c_str(), context->peer().c_str());
    RTP_LLM_LOG_DEBUG("receive start load request from client: %s, request: [%s]",
                      context->peer().c_str(),
                      request->DebugString().c_str());
    if (!engine_) {
        RTP_LLM_LOG_WARNING("start load failed, engine is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "engine is null");
    }
    auto cache_manager = engine_->getCacheManager();
    if (!cache_manager) {
        RTP_LLM_LOG_WARNING("start load failed, cache manager is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "cache manager is null");
    }
    auto    is_cancelled      = [context]() { return context->IsCancelled(); };
    int64_t handle_read_start = currentTimeUs();
    cache_manager->handleRead(*request, *response, std::move(is_cancelled));
    int64_t handle_read_cost = currentTimeUs() - handle_read_start;
    if (handle_read_cost >= 2000000) {
        RTP_LLM_LOG_WARNING("[PD-DIAG] StartLoad slow handleRead cost_us=%ld, unique_key=%s",
                            handle_read_cost,
                            request->unique_key().c_str());
    }
    return grpc::Status::OK;
}

::grpc::Status PrefillRpcServerNew2::GetPeerInfo(::grpc::ServerContext*      context,
                                                 const GetPeerInfoRequestPB* request,
                                                 GetPeerInfoResponsePB*      response) {
    const auto& pc = maga_init_params_.parallelism_config;
    if (pc.tp_size <= 0) {
        return grpcStatusFromErrorInfo(
            ErrorInfo(ErrorCode::INVALID_PARAMS, "GetPeerInfo invalid tp_size=" + std::to_string(pc.tp_size)));
    }
    response->set_tp_size(static_cast<int32_t>(pc.tp_size));
    response->set_cp_size(static_cast<int32_t>(pc.prefill_cp_config.kv_cache_sharded ? pc.tp_size : 1));
    RTP_LLM_LOG_INFO("GetPeerInfo: tp_size=%ld, cp_size=%d", pc.tp_size, response->cp_size());
    return grpc::Status::OK;
}

}  // namespace rtp_llm
