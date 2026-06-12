#include "rtp_llm/cpp/model_rpc/PrefillRpcServerNew2.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "autil/NetUtil.h"

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
                                        py::object                                             mm_process_engine,
                                        std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    // decode_entrance path uses P2P connector for KV cache transfer, not CacheStore.
    // Skip CacheStore init to avoid double-registering the same GPU memory region
    // (CacheStore registers the full KVCache area; P2P connector registers per-block).
    auto ret = RemoteRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params), false);
    if (!ret.ok()) {
        RTP_LLM_LOG_ERROR("prefill rpc server new2 init failed, err: %s", ret.error_message().c_str());
        return ret;
    }

    {
        pybind11::gil_scoped_acquire acquire;
        local_rpc_port_ = maga_init_params.server_config.attr("rpc_server_port").cast<int64_t>();
    }
    RTP_LLM_LOG_INFO("PrefillRpcServerNew2::init: captured local_rpc_port=%ld", local_rpc_port_);

    // Pre-compute dp_grpc_addrs_ from p2p_worker_addrs (format "ip:p2p_port:grpc_port").
    // For each DP group, the tp_rank=0 entry lives at index dp_rank * tp_size.
    // This replaces the fragile port-arithmetic in GetPeerInfo().
    {
        const auto& pc    = maga_init_params_.parallelism_config;
        const auto& addrs = maga_init_params_.runtime_config.p2p_worker_addrs;
        if (!addrs.empty() && pc.tp_size > 0 && pc.dp_size > 0) {
            for (int64_t dp = 0; dp < pc.dp_size; ++dp) {
                size_t idx = static_cast<size_t>(dp * pc.tp_size);
                if (idx >= addrs.size()) {
                    RTP_LLM_LOG_WARNING("PrefillRpcServerNew2::init: p2p_worker_addrs has %zu entries "
                                        "but need index %zu for dp_rank=%ld (tp_size=%ld, dp_size=%ld)",
                                        addrs.size(), idx, dp, pc.tp_size, pc.dp_size);
                    break;
                }
                // Parse "ip:p2p_port:grpc_port" -> extract "ip:grpc_port"
                const auto& entry      = addrs[idx];
                auto        first_col  = entry.find(':');
                auto        second_col = entry.find(':', first_col + 1);
                if (first_col == std::string::npos || second_col == std::string::npos) {
                    RTP_LLM_LOG_WARNING("PrefillRpcServerNew2::init: malformed p2p_worker_addrs[%zu]='%s', "
                                        "expected ip:p2p_port:grpc_port",
                                        idx, entry.c_str());
                    dp_grpc_addrs_.clear();
                    break;
                }
                std::string ip        = entry.substr(0, first_col);
                std::string grpc_port = entry.substr(second_col + 1);
                dp_grpc_addrs_.push_back(ip + ":" + grpc_port);
            }
            std::string addrs_str;
            for (size_t i = 0; i < dp_grpc_addrs_.size(); ++i) {
                if (i > 0)
                    addrs_str += ", ";
                addrs_str += dp_grpc_addrs_[i];
            }
            RTP_LLM_LOG_INFO("PrefillRpcServerNew2::init: built dp_grpc_addrs_ from p2p_worker_addrs: [%s]",
                             addrs_str.c_str());
        } else {
            RTP_LLM_LOG_INFO("PrefillRpcServerNew2::init: p2p_worker_addrs empty or parallelism not set, "
                             "dp_grpc_addrs_ will be computed from port arithmetic at GetPeerInfo time");
        }
    }

    auto kvcache_manager = engine_->getCacheManager();
    if (!kvcache_manager) {
        RTP_LLM_LOG_WARNING("prefill rpc server new2 init failed, kvcache manager is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "kvcache manager is null");
    }
    auto connector_coordinator = kvcache_manager->connectorCoordinator();
    if (!connector_coordinator) {
        RTP_LLM_LOG_WARNING("prefill rpc server new2 init failed, connector coordinator is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "connector coordinator is null");
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
    const bool pd_separation = decodeEntranceRequiresPrefill(*request);
    if (!pd_separation) {
        RTP_LLM_LOG_INFO("pd separation is disabled, call local rpc server");
        return LocalRpcServer::GenerateStreamCall(server_context, request, response_writer);
    }
    if (request->generate_config().unique_key().empty()) {
        RTP_LLM_LOG_WARNING("decode_entrance prefill handoff requires non-empty unique_key, request_id=%ld",
                            request->request_id());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "decode_entrance handoff requires non-empty unique_key");
    }

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
    input->generate_config->pd_separation = true;
    if (engine_->isMTPEagle()) {
        input->generate_config->force_disable_sp_run = false;
    } else {
        input->generate_config->force_disable_sp_run = true;
    }

    int64_t mm_cost_us = 0, enqueue_cost_us = 0, poll_cost_us = 0;
    int64_t phase_start = currentTimeUs();

    if (mm_processor_ != nullptr && input->multimodal_inputs) {
        auto mm_res = mm_processor_->updateMultimodalFeatures(input);
        mm_cost_us  = currentTimeUs() - phase_start;
        if (!mm_res.ok()) {
            generate_context.error_status = serializeErrorMsg(generate_context.request_key, mm_res);
        }
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

::grpc::Status PrefillRpcServerNew2::StartLoad(::grpc::ServerContext*                context,
                                               const P2PConnectorStartLoadRequestPB* request,
                                               P2PConnectorStartLoadResponsePB*      response) {
    RTP_LLM_LOG_DEBUG(
        "StartLoad gRPC entry, unique_key=%s, peer=%s", request->unique_key().c_str(), context->peer().c_str());
    RTP_LLM_LOG_DEBUG("receive start load request from client: %s, request: [%s]",
                      context->peer().c_str(),
                      request->DebugString().c_str());
    if (context->IsCancelled()) {
        RTP_LLM_LOG_WARNING("start load failed, request is cancelled");
        return grpc::Status(grpc::StatusCode::CANCELLED, "request is cancelled");
    }
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
    cache_manager->handleRead(*request, *response, is_cancelled);
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
    const auto& pc  = maga_init_params_.parallelism_config;
    const auto& pdc = maga_init_params_.pd_sep_config;
    response->set_tp_size(static_cast<int32_t>(pc.tp_size));
    response->set_dp_size(static_cast<int32_t>(pc.dp_size));

    if (!dp_grpc_addrs_.empty()) {
        // Preferred path: use pre-computed addresses from p2p_worker_addrs.
        for (const auto& addr : dp_grpc_addrs_) {
            response->add_dp_grpc_addrs(addr);
        }
    } else {
        // Fallback: port arithmetic (kept for backward compatibility when
        // p2p_worker_addrs is not populated).
        const int64_t rank_stride         = pdc.worker_port_offset;
        const int64_t dp_stride           = pc.tp_size * rank_stride;
        const int64_t current_rank_offset = (pc.dp_rank * pc.tp_size + pc.tp_rank) * rank_stride;
        const int64_t dp0_rpc_port        = local_rpc_port_ - current_rank_offset;
        std::string bind_ip      = autil::NetUtil::getBindIp();
        for (int64_t i = 0; i < pc.dp_size; ++i) {
            response->add_dp_grpc_addrs(bind_ip + ":" + std::to_string(dp0_rpc_port + i * dp_stride));
        }
    }

    RTP_LLM_LOG_INFO("GetPeerInfo: tp_size=%ld, dp_size=%ld, dp_addrs=[%s]",
                     pc.tp_size,
                     pc.dp_size,
                     [&]() {
                         std::string s;
                         for (int i = 0; i < response->dp_grpc_addrs_size(); ++i) {
                             if (i > 0)
                                 s += ", ";
                             s += response->dp_grpc_addrs(i);
                         }
                         return s;
                     }().c_str());
    return grpc::Status::OK;
}

}  // namespace rtp_llm
