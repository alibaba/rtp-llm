#include <unordered_set>
#include "autil/Scope.h"
#include <c10/core/InferenceMode.h>
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/RpcTimeoutUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <exception>
#include <tuple>
#include <utility>

namespace rtp_llm {

namespace {

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
    if (batch_context_cleanup_thread_) {
        batch_context_cleanup_thread_->stop();
        batch_context_cleanup_thread_.reset();
    }
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        batch_entries_.clear();
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

void PrefillRpcServer::registerBatchAttach(const std::string&       key,
                                           const GenerateStreamPtr& stream,
                                           int64_t                  timeout_ms,
                                           int64_t                  request_deadline_ms,
                                           int64_t                  now_ms) {
    constexpr int64_t kDefaultAttachTimeoutMs = 10 * 60 * 1000;
    const auto        ttl_ms                  = timeout_ms > 0 ? timeout_ms : kDefaultAttachTimeoutMs;
    // Subtract before adding so arbitrarily large wire values cannot overflow.
    const auto deadline_ms = now_ms + std::min(ttl_ms, std::max<int64_t>(0, request_deadline_ms - now_ms));
    std::lock_guard<std::mutex> lock(batch_mutex_);
    auto&                       entry = batch_entries_[key];
    entry.stream                      = stream;
    entry.deadline_ms                 = deadline_ms;
    entry.request_deadline_ms         = request_deadline_ms;
    entry.attach_pending              = true;
}

bool PrefillRpcServer::attachBatch(const std::string& key, int64_t now_ms) {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    auto                        it = batch_entries_.find(key);
    if (it == batch_entries_.end() || !it->second.attach_pending)
        return true;  // Ordinary GenerateStreamCall has no batch attachment deadline.
    if (it->second.expired || now_ms >= it->second.deadline_ms)
        return false;
    it->second.attach_pending = false;
    it->second.stream.reset();
    if (!it->second.reserved && !it->second.context)
        batch_entries_.erase(it);
    return true;
}

void PrefillRpcServer::expireBatchAttachments(int64_t now_ms) {
    struct ExpiredAttachment {
        std::string                   key;
        std::weak_ptr<GenerateStream> stream;
        int64_t                       request_deadline_ms;
    };
    std::vector<ExpiredAttachment> expired;
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        for (auto it = batch_entries_.begin(); it != batch_entries_.end();) {
            auto& state = it->second;
            if (state.attach_pending && !state.expired && now_ms >= state.deadline_ms) {
                state.expired = true;
                expired.push_back({it->first, state.stream, state.request_deadline_ms});
                state.stream.reset();
            }
            // Keep the rejection fence until the overall deadline. The resource
            // store retains its terminal fence after the control record is removed.
            if (state.attach_pending && now_ms >= state.request_deadline_ms && !state.reserved && !state.context)
                it = batch_entries_.erase(it);
            else
                ++it;
        }
    }
    for (const auto& state : expired) {
        if (auto stream = state.stream.lock())
            stream->reportError(ErrorCode::GENERATE_TIMEOUT, "PD batch result attachment deadline exceeded");
        if (engine_ && engine_->getCacheManager())
            engine_->getCacheManager()->cancelPrefillRequest(state.key, state.request_deadline_ms);
    }
}

void PrefillRpcServer::batchContextCleanupTick() {
    expireBatchAttachments(currentTimeMs());
    std::vector<std::unique_ptr<GenerateContext>> completed;
    std::vector<PDCancelRegistry::Handle>         finished_handles;
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        for (auto it = batch_entries_.begin(); it != batch_entries_.end();) {
            auto& entry = it->second;
            if (entry.context) {
                auto& stream = entry.context->getStream();
                if (stream && (stream->hasError() || stream->getStatus() == StreamState::FINISHED)) {
                    finished_handles.push_back(cancel_registry_->find(entry.context->request_id));
                    entry.reserved = false;
                    entry.context->markRpcHandlingCompleted();
                    completed.push_back(std::move(entry.context));
                }
            }
            if (!entry.reserved && !entry.context
                && (!entry.attach_pending || currentTimeMs() >= entry.request_deadline_ms)) {
                it = batch_entries_.erase(it);
            } else {
                ++it;
            }
        }
    }
    // GenerateContext destruction publishes the final RuntimeMeta snapshot.
    completed.clear();
    for (const auto& handle : finished_handles)
        if (handle)
            cancel_registry_->finishLocal(handle);
    cancelCleanupTick();
}

grpc::Status PrefillRpcServer::init(const EngineInitParams&                                maga_init_params,
                                    std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                    py::object                                             mm_process_engine) {
    auto ret = LocalRpcServer::init(maga_init_params, std::move(propose_params), mm_process_engine);
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
        RTP_LLM_LOG_WARNING("prefill rpc server new2 init failed, PD requires P2P connector");
        return grpc::Status(grpc::StatusCode::INTERNAL, "PD requires P2P connector");
    }

    cancel_registry_ = std::make_unique<PDCancelRegistry>(meta_);

    // Start [HANG-DIAG] watchdog: every 30s, scan onflight_trackers_ and WARN
    // about any GenerateStreamCall that has been alive longer than the
    // threshold (default 60s). Captures 5/22-style hangs where the thread
    // entered prefill but never returned and printed nothing in between.
    hang_diag_thread_ =
        autil::LoopThread::createLoopThread([this]() { hangDiagTick(); }, 30 * 1000 * 1000, "PrefillRpcServerHangDiag");
    if (!hang_diag_thread_) {
        RTP_LLM_LOG_WARNING("prefill rpc server new2: failed to start hang_diag_thread_, watchdog disabled");
    }
    batch_context_cleanup_thread_ =
        autil::LoopThread::createLoopThread([this]() { batchContextCleanupTick(); }, 10 * 1000, "P2PBatchCleanup");
    if (!batch_context_cleanup_thread_) {
        RTP_LLM_LOG_WARNING("prefill rpc server new2: failed to start batch cleanup thread");
        return grpc::Status(grpc::StatusCode::INTERNAL, "failed to start P2P batch cleanup thread");
    }
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServer::GenerateStreamCall(grpc::ServerContext*                   server_context,
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
    PDCancelRegistry::Handle cancel_handle;
    auto                     admission =
        cancel_registry_->admit(*request, request->generate_config().unique_key(), "", local_deadline, cancel_handle);
    if (!admission.ok())
        return serializeErrorMsg(std::to_string(request->request_id()), admission);
    PDCancelRegistry::CallGuard cancel_guard(*cancel_registry_, cancel_handle);
    AtomicGuard                 request_guard(onflight_requests_);
    auto                        request_id = request->request_id();
    // [HANG-DIAG] step 1/4: entry. Coupled with OnflightScope so the watchdog
    // can report which step a stuck request last reached even if no further
    // step log gets emitted. See 5/22 P1-B analysis where 4 prefill threads
    // entered RemoteRpcServiceImpl::GenerateStreamCall (entry log present) but
    // never produced any other log line for the remaining 3h36min.
    OnflightScope onflight_scope(this, request_id);
    auto          generate_context =
        GenerateContext(request_id, request->generate_config().timeout_ms(), server_context, metrics_reporter_, meta_);
    const int uncaught_exceptions = std::uncaught_exceptions();
    autil::ScopeGuard rpc_completion_guard([&generate_context, uncaught_exceptions] {
        if (std::uncaught_exceptions() == uncaught_exceptions) {
            generate_context.markRpcHandlingCompleted();
        }
    });
    auto input                 = QueryConverter::transQuery(request);
    input->request_deadline_ms = local_deadline;
    if (local_deadline <= currentTimeMs() || server_context->IsCancelled()) {
        return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "prefill request expired");
    }
    int64_t mm_cost_us = 0, enqueue_cost_us = 0, poll_cost_us = 0;
    int64_t phase_start = currentTimeUs();

    auto preprocess_status = preprocessForPD(input, mm_processor_.get(), engine_->isMTPEagle());
    mm_cost_us             = currentTimeUs() - phase_start;
    if (!preprocess_status.ok()) {
        generate_context.error_status = serializeErrorMsg(generate_context.request_key, preprocess_status);
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
        generate_context.error_status = serializeErrorMsg(std::to_string(request_id), generate_context.error_info);
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
}

grpc::Status PrefillRpcServer::EnqueueBatch(grpc::ServerContext*         context,
                                            const EnqueueBatchRequestPB* request,
                                            EnqueueBatchResponsePB*      response) {
    c10::InferenceMode inference_guard(true);
    RTP_LLM_PROFILE_FUNCTION();
    AtomicGuard request_guard(onflight_requests_);
    response->Clear();
    response->set_batch_id(request->batch_id());

    const auto& parallelism_config = maga_init_params_.parallelism_config;
    RTP_LLM_CHECK_WITH_INFO(parallelism_config.dp_size == 1,
                            "P2P EnqueueBatch only supports single-DP mode, dp_size=%ld",
                            parallelism_config.dp_size);
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

    std::vector<std::shared_ptr<GenerateInput>> inputs;
    std::vector<int64_t>                        admitted_request_ids;
    const auto                                  release_request_id = [this](int64_t request_id) {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        batch_entries_.erase(masterEnqueuedHandoffUniqueKey(request_id));
    };
    const auto release_admitted_request_ids = [this, &admitted_request_ids]() {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        for (const auto request_id : admitted_request_ids) {
            batch_entries_.erase(masterEnqueuedHandoffUniqueKey(request_id));
            if (auto handle = cancel_registry_->find(request_id))
                cancel_registry_->finishLocal(handle);
        }
    };
    inputs.reserve(input_count);
    admitted_request_ids.reserve(input_count);
    for (const auto& dp_slot : request->dp_slots()) {
        for (const auto& external_input : dp_slot.requests()) {
            if (!external_input.has_input()) {
                addBatchError(response, 0, ErrorCode::INVALID_PARAMS, "P2P EnqueueBatch input is missing");
                continue;
            }
            const auto request_id = external_input.input().request_id();
            if (dp_slot.dp_rank() != local_dp_rank) {
                addBatchError(response,
                              request_id,
                              ErrorCode::INVALID_PARAMS,
                              "P2P EnqueueBatch dp_rank mismatch, request dp_rank " + std::to_string(dp_slot.dp_rank())
                                  + ", local dp_rank " + std::to_string(local_dp_rank));
                continue;
            }
            if (context && context->IsCancelled()) {
                release_admitted_request_ids();
                return grpc::Status(grpc::StatusCode::CANCELLED, "P2P EnqueueBatch cancelled before admission");
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

            GenerateInputPB item;
            item.CopyFrom(external_input.input());
            item.set_group_size(input_count);
            item.mutable_group_id()->set_value(request->batch_id());
            auto* config = item.mutable_generate_config();
            config->set_unique_key(masterEnqueuedHandoffUniqueKey(request_id));
            config->set_timeout_ms(clampRpcTimeoutMsToInt32(
                normalizeRpcTimeoutMs(config->timeout_ms(), maga_init_params_.pd_sep_config.max_rpc_timeout_ms)));
            auto input = QueryConverter::transQuery(&item);
            input->request_deadline_ms =
                engine_->getCacheManager()->prefillRequestDeadline(config->unique_key(), config->timeout_ms());
            if (input->request_deadline_ms <= currentTimeMs()) {
                release_request_id(request_id);
                addBatchError(response,
                              request_id,
                              ErrorCode::GENERATE_TIMEOUT,
                              "P2P EnqueueBatch request expired before admission");
                continue;
            }
            PDCancelRegistry::Handle cancel_handle;
            const auto               admission =
                cancel_registry_->admit(item, config->unique_key(), "", input->request_deadline_ms, cancel_handle);
            if (!admission.ok()) {
                release_request_id(request_id);
                addBatchError(response, request_id, admission.code(), admission.ToString());
                continue;
            }
            const auto preprocess_status = preprocessForPD(input, mm_processor_.get(), engine_->isMTPEagle());
            if (!preprocess_status.ok()) {
                cancel_registry_->finishLocal(cancel_handle);
                cancel_registry_->finishDownstream(cancel_handle);
                release_request_id(request_id);
                addBatchError(response, request_id, preprocess_status.code(), preprocess_status.ToString());
                continue;
            }
            if (cancel_handle->canceled.load()) {
                cancel_registry_->finishLocal(cancel_handle);
                release_request_id(request_id);
                addBatchError(
                    response, request_id, cancel_handle->cancel_reason.code(), cancel_handle->cancel_reason.ToString());
                continue;
            }
            admitted_request_ids.push_back(request_id);
            inputs.push_back(std::move(input));
        }
    }
    if (inputs.empty()) {
        return grpc::Status::OK;
    }

    std::vector<bool>              enqueue_successes;
    std::vector<GenerateStreamPtr> streams;
    try {
        std::tie(enqueue_successes, streams) = engine_->enqueueMultiple(inputs);
    } catch (const std::exception& e) {
        release_admitted_request_ids();
        return grpc::Status(grpc::StatusCode::INTERNAL,
                            "P2P EnqueueBatch enqueueMultiple exception: " + std::string(e.what()));
    } catch (...) {
        release_admitted_request_ids();
        return grpc::Status(grpc::StatusCode::INTERNAL, "P2P EnqueueBatch enqueueMultiple unknown exception");
    }
    if (enqueue_successes.size() != inputs.size() || streams.size() != inputs.size()) {
        release_admitted_request_ids();
        return grpc::Status(grpc::StatusCode::INTERNAL, "P2P EnqueueBatch result size mismatch");
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (enqueue_successes[i]) {
            auto batch_context = std::make_unique<GenerateContext>(admitted_request_ids[i],
                                                                   inputs[i]->generate_config->timeout_ms,
                                                                   /*server_context=*/nullptr,
                                                                   metrics_reporter_,
                                                                   meta_);
            batch_context->setStream(streams[i]);
            cancel_registry_->attach(cancel_registry_->find(admitted_request_ids[i]), streams[i]);
            registerBatchAttach(inputs[i]->generate_config->unique_key,
                                streams[i],
                                request->fetch_attach_timeout_ms(),
                                inputs[i]->request_deadline_ms,
                                currentTimeMs());
            {
                std::lock_guard<std::mutex> lock(batch_mutex_);
                batch_entries_.at(inputs[i]->generate_config->unique_key).context = std::move(batch_context);
            }
            addBatchSuccess(response, admitted_request_ids[i]);
            continue;
        }
        auto error = streams[i] ? streams[i]->statusInfo() : ErrorInfo(ErrorCode::UNKNOWN_ERROR, "null stream");
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
                                           P2PConnectorStartLoadResponsePB*      response) {
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
}

grpc::Status
PrefillRpcServer::Cancel(grpc::ServerContext*, const CancelRequestPB* request, CancelResponsePB* response) {
    if (!request || request->request_id() <= 0 || !response)
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "cancel request missing request_id");
    if (!cancel_registry_)
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Prefill is not initialized");
    auto handle = cancel_registry_->find(request->request_id());
    if (handle && handle->terminal.load()) {
        response->set_status(CANCEL_STATUS_NOT_FOUND);
        return grpc::Status::OK;
    }
    // A completed Prefill may still have unconsumed KV in the resource store.
    if (handle && engine_ && engine_->getCacheManager())
        engine_->getCacheManager()->cancelPrefillRequest(handle->unique_key, handle->deadline_ms);
    response->set_status(cancel_registry_->cancel(
        request->request_id(),
        {request->cancel_error_code() == 0 ? ErrorCode::PRIORITY_PREEMPTED :
                                             static_cast<ErrorCode>(request->cancel_error_code()),
         request->cancel_error_message().empty() ? "PD request canceled" : request->cancel_error_message()}));
    return grpc::Status::OK;
}

void PrefillRpcServer::cancelCleanupTick() {
    if (currentTimeMs() - last_cancel_tick_ms_ < 100)
        return;
    last_cancel_tick_ms_ = currentTimeMs();
    for (const auto& handle : cancel_registry_->pending()) {
        engine_->getCacheManager()->cancelPrefillRequest(handle->unique_key, handle->deadline_ms);
        cancel_registry_->complete(handle);
    }
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
    response->set_cp_size(static_cast<int32_t>(pc.prefill_cp_config.kv_cache_sharded ? pc.tp_size : 1));
    RTP_LLM_LOG_INFO("GetPeerInfo: tp_size=%ld, cp_size=%d", pc.tp_size, response->cp_size());
    return grpc::Status::OK;
}

}  // namespace rtp_llm
