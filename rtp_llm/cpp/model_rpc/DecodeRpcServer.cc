#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"
#include <algorithm>
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RpcTimeoutUtils.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/engine_base/Host.h"
#include "autil/NetUtil.h"
#include <chrono>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <utility>

namespace rtp_llm {

namespace {

constexpr int64_t kMinGrpcPort = 1;
constexpr int64_t kMaxGrpcPort = 65535;

struct ReuseLens {
    int64_t total  = 0;
    int64_t local  = 0;
    int64_t remote = 0;
    int64_t memory = 0;
    int64_t disk   = 0;
};

ReuseLens getPrefillReuseLens(std::shared_ptr<GenerateStream>& stream) {
    ReuseLens reuse_lens;
    reuse_lens.total  = stream->prefillTotalReuseLen();
    reuse_lens.local  = stream->prefillLocalReuseLen();
    reuse_lens.remote = stream->prefillRemoteReuseLen();
    reuse_lens.memory = stream->prefillMemoryReuseLen();
    reuse_lens.disk   = stream->prefillDiskReuseLen();
    return reuse_lens;
}

bool parseGrpcPort(const std::string& port_text, uint32_t* port) {
    if (!port || port_text.empty()) {
        return false;
    }
    char* end             = nullptr;
    errno                 = 0;
    auto       value      = std::strtoll(port_text.c_str(), &end, 10);
    const bool parsed_all = end == port_text.c_str() + port_text.size();
    if (errno != 0 || !parsed_all || value < kMinGrpcPort || value > kMaxGrpcPort) {
        return false;
    }
    *port = static_cast<uint32_t>(value);
    return true;
}

std::string prefillAddress(const GenerateInputPB& request) {
    for (const auto& addr : request.generate_config().role_addrs()) {
        if (addr.role() == RoleAddrPB::PREFILL) {
            return formatGrpcHostPort(addr.ip(), addr.grpc_port());
        }
    }
    return {};
}

}  // namespace

std::string makeDecodeEntranceUniqueKey(const std::string& bind_ip, int64_t unique_key_id, int64_t current_time_us) {
    return bind_ip + "_" + std::to_string(unique_key_id) + "_" + std::to_string(current_time_us);
}

GenerateInputPB makeDecodeEntranceHandoffRequest(const GenerateInputPB& request,
                                                 const std::string&     handoff_unique_key) {
    GenerateInputPB handoff_request;
    handoff_request.CopyFrom(request);
    handoff_request.mutable_generate_config()->set_unique_key(handoff_unique_key);
    return handoff_request;
}

grpc::Status DecodeRpcServer::parsePrefillDpAddr(const std::string& addr, std::string* ip, uint32_t* port) {
    if (!ip || !port || addr.empty()) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "malformed prefill dp_addr: " + addr);
    }

    std::string host;
    std::string port_text;
    if (addr.front() == '[') {
        const auto close_pos = addr.find(']');
        if (close_pos == std::string::npos || close_pos + 1 >= addr.size() || addr[close_pos + 1] != ':') {
            RTP_LLM_LOG_ERROR("malformed prefill dp_addr: %s", addr.c_str());
            return grpc::Status(grpc::StatusCode::INTERNAL, "malformed prefill dp_addr: " + addr);
        }
        host      = addr.substr(0, close_pos + 1);
        port_text = addr.substr(close_pos + 2);
    } else {
        const auto colon_pos = addr.rfind(':');
        if (colon_pos == std::string::npos || colon_pos == 0 || colon_pos + 1 >= addr.size()) {
            RTP_LLM_LOG_ERROR("malformed prefill dp_addr: %s", addr.c_str());
            return grpc::Status(grpc::StatusCode::INTERNAL, "malformed prefill dp_addr: " + addr);
        }
        host      = addr.substr(0, colon_pos);
        port_text = addr.substr(colon_pos + 1);
        if (host.find(':') != std::string::npos) {
            if (host.back() == ':' || host.find('.') != std::string::npos) {
                RTP_LLM_LOG_ERROR("malformed prefill dp_addr: %s", addr.c_str());
                return grpc::Status(grpc::StatusCode::INTERNAL, "malformed prefill dp_addr: " + addr);
            }
            host = "[" + host + "]";
        }
    }

    if (host.empty() || !parseGrpcPort(port_text, port)) {
        RTP_LLM_LOG_ERROR("invalid port in prefill dp_addr: %s", addr.c_str());
        return grpc::Status(grpc::StatusCode::INTERNAL, "invalid port in prefill dp_addr: " + addr);
    }
    *ip = std::move(host);
    return grpc::Status::OK;
}

DecodeRpcServer::~DecodeRpcServer() {
    if (cancel_cleanup_thread_)
        cancel_cleanup_thread_->stop();
}

grpc::Status DecodeRpcServer::Cancel(grpc::ServerContext*, const CancelRequestPB* request, CancelResponsePB* response) {
    if (!request || request->request_id() <= 0 || !response)
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "cancel request missing request_id");
    if (!cancel_registry_)
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Decode is not initialized");
    auto handle = cancel_registry_->find(request->request_id());
    if (!handle && !request->prefill_address().empty()) {
        // Create a control-only record so retries can finish Prefill cancellation
        // even when the frontend has not connected to Decode yet.
        GenerateInputPB pending_request;
        pending_request.set_request_id(request->request_id());
        const auto admission = cancel_registry_->admit(pending_request,
                                                       masterEnqueuedHandoffUniqueKey(request->request_id()),
                                                       request->prefill_address(),
                                                       currentTimeMs() + 10 * 60 * 1000,
                                                       handle);
        if (!admission.ok())
            handle = cancel_registry_->find(request->request_id());
        if (admission.ok())
            cancel_registry_->finishLocal(handle);
    }
    if (handle && handle->terminal.load()) {
        // Internal forwarding may retry after WorkerStatus's short delta window.
        // This retained terminal record proves both endpoint cleanups returned.
        response->set_status(CANCEL_STATUS_NOT_FOUND);
    } else {
        response->set_status(cancel_registry_->cancel(
            request->request_id(),
            {request->cancel_error_code() == 0 ? ErrorCode::PRIORITY_PREEMPTED :
                                                 static_cast<ErrorCode>(request->cancel_error_code()),
             request->cancel_error_message().empty() ? "PD request canceled" : request->cancel_error_message()}));
    }
    return grpc::Status::OK;
}

void DecodeRpcServer::cancelCleanupTick() {
    for (const auto& handle : cancel_registry_->pending()) {
        if (handle->downstream_address.empty()) {
            cancel_registry_->complete(handle);
            continue;
        }
        auto connection = cancel_pool_.getConnection(handle->downstream_address);
        if (!connection.ok())
            continue;
        CancelRequestPB request;
        request.set_request_id(handle->identity.request_id);
        request.set_cancel_error_code(static_cast<int64_t>(handle->cancel_reason.code()));
        request.set_cancel_error_message(handle->cancel_reason.ToString());
        CancelResponsePB    response;
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(200));
        if (!connection->stub->Cancel(&context, request, &response).ok())
            continue;
        if (response.status() == CANCEL_STATUS_TOMBSTONED || response.status() == CANCEL_STATUS_NOT_FOUND) {
            cancel_registry_->finishDownstream(handle);
        } else if (response.status() == CANCEL_STATUS_ACCEPTED) {
            StatusVersionPB version;
            version.set_latest_finished_version(0);
            WorkerStatusPB      status;
            grpc::ClientContext status_context;
            status_context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(200));
            if (connection->stub->GetWorkerStatus(&status_context, version, &status).ok()) {
                for (const auto& task : status.finished_task_list()) {
                    if (task.request_id() == handle->identity.request_id) {
                        cancel_registry_->finishDownstream(handle);
                        break;
                    }
                }
            }
        }
        cancel_registry_->complete(handle);
    }
}

grpc::Status DecodeRpcServer::init(const EngineInitParams&                                maga_init_params,
                                   std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                   py::object                                             mm_process_engine) {
    auto ret = LocalRpcServer::init(maga_init_params, std::move(propose_params), mm_process_engine);
    if (!ret.ok()) {
        RTP_LLM_LOG_ERROR("decode rpc server new2 init failed, err: %s", ret.error_message().c_str());
        return ret;
    }

    auto kvcache_manager = engine_->getCacheManager();
    if (!kvcache_manager) {
        RTP_LLM_LOG_WARNING("decode rpc server new2 init failed, kvcache manager is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "kvcache manager is null");
    }
    if (!kvcache_manager->hasP2PConnector()) {
        RTP_LLM_LOG_WARNING("decode rpc server new2 init failed, PD requires P2P connector");
        return grpc::Status(grpc::StatusCode::INTERNAL, "PD requires P2P connector");
    }

    prefill_server_caller_ = std::make_shared<PrefillServerCaller>();
    cancel_registry_       = std::make_unique<PDCancelRegistry>(meta_);
    cancel_cleanup_thread_ =
        autil::LoopThread::createLoopThread([this]() { cancelCleanupTick(); }, 10 * 1000, "P2PDecodeCancel");
    if (!cancel_cleanup_thread_)
        return grpc::Status(grpc::StatusCode::INTERNAL, "failed to start Decode cancel cleanup");

    RTP_LLM_LOG_INFO("decode rpc server new2 init");
    return grpc::Status::OK;
}

grpc::Status DecodeRpcServer::preparePDRequest(const GenerateInputPB&          request,
                                               int64_t                         deadline_ms,
                                               std::shared_ptr<GenerateInput>& input,
                                               GenerateInputPB&                prefill_request,
                                               PrefillPeerInfo&                peer_info) {
    input                      = QueryConverter::transQuery(&request);
    input->request_deadline_ms = deadline_ms;
    // Preserve the configured duration on the wire; D keeps its own deadline.
    input->generate_config->timeout_ms = request.generate_config().timeout_ms();
    auto status                        = preprocessForPD(input, mm_processor_.get(), engine_->isMTPEagle());
    if (!status.ok())
        return serializeErrorMsg(std::to_string(request.request_id()), status);
    prefill_request.CopyFrom(request);
    prefill_request.mutable_generate_config()->set_timeout_ms(input->generate_config->timeout_ms);
    const auto  address = prefillAddress(request);
    std::string ip;
    uint32_t    port         = 0;
    auto        parse_status = parsePrefillDpAddr(address, &ip, &port);
    if (!parse_status.ok())
        return parse_status;
    const auto remaining = deadline_ms - currentTimeMs();
    if (remaining <= 0)
        return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "PD preprocessing deadline exceeded");
    auto peer_result = prefill_server_caller_->getPrefillPeerInfo(ip, port, clampRpcTimeoutMsToInt32(remaining));
    if (!peer_result.ok())
        return grpcStatusFromErrorInfo(peer_result.status());
    peer_info = std::move(peer_result.value());
    if (peer_info.tp_size <= 0 || peer_info.cp_size <= 0) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "prefill peer info is not available");
    }
    return grpc::Status::OK;
}

grpc::Status DecodeRpcServer::GenerateStreamCall(grpc::ServerContext*                   server_context,
                                                 const GenerateInputPB*                 request,
                                                 grpc::ServerWriter<GenerateOutputsPB>* response_writer) {
    struct PrefillContextGuard {
        std::shared_ptr<PrefillServerCallerContext> context;

        ~PrefillContextGuard() {
            if (!context) {
                return;
            }
            if (!context->done()) {
                context->cancel();
            }
            context->wait();
        }
    };

    const int64_t request_entry_ms          = currentTimeMs();
    const auto    normalized_timeout_ms     = normalizeRpcTimeoutMs(request->generate_config().timeout_ms(),
                                                             maga_init_params_.pd_sep_config.max_rpc_timeout_ms);
    const auto    normalized_timeout_ms_i32 = clampRpcTimeoutMsToInt32(normalized_timeout_ms);

    // Always isolate decode-entrance P2P handoff from any caller-provided business unique_key.
    // The external unique_key remains on the original request object, while the internal handoff
    // request gets a per-request key used only by the prefill/decode P2P pipeline.
    GenerateInputPB request_with_handoff_key;
    const auto*     effective_request = request;

    // Check if pd separation should be used
    // A successful Master batch admission is authoritative: Prefill has
    // already entered the P2P path, so Decode must attach to that handoff.
    auto pd_separation = checkPDSupport(*request).supported || request->enqueued_by_master();
    if (pd_separation) {
        std::string handoff_unique_key;
        if (request->enqueued_by_master()) {
            handoff_unique_key = masterEnqueuedHandoffUniqueKey(request->request_id());
        } else {
            const auto handoff_id = unique_key_id_.fetch_add(1);
            handoff_unique_key = makeDecodeEntranceUniqueKey(autil::NetUtil::getBindIp(), handoff_id, currentTimeUs());
        }
        request_with_handoff_key = makeDecodeEntranceHandoffRequest(*request, handoff_unique_key);
        request_with_handoff_key.mutable_generate_config()->set_timeout_ms(normalized_timeout_ms_i32);
        effective_request = &request_with_handoff_key;
    }
    if (!pd_separation) {
        RTP_LLM_LOG_DEBUG("pd separation is disabled, call prefill server");
        GenerateInputPB prefill_forward_request;
        prefill_forward_request.CopyFrom(*effective_request);
        prefill_forward_request.mutable_generate_config()->set_timeout_ms(normalized_timeout_ms_i32);
        return prefill_server_caller_->callPrefill(server_context, &prefill_forward_request, response_writer);
    }

    const auto log_pd_error = [&](const char* stage, const grpc::Status& status) {
        const auto error = errorInfoFromGrpcStatus(status);
        if (status.error_code() == grpc::StatusCode::CANCELLED && error.code() == ErrorCode::CANCELLED) {
            RTP_LLM_LOG_INFO("P2P Decode request cancelled, stage=%s request_id=%ld unique_key=%s error=%s",
                             stage,
                             request->request_id(),
                             effective_request->generate_config().unique_key().c_str(),
                             error.ToString().c_str());
            return;
        }
        RTP_LLM_LOG_WARNING("P2P Decode request failed, stage=%s request_id=%ld business_unique_key=%s unique_key=%s "
                            "grpc_code=%d error_code=%d error_name=%s error=%s",
                            stage,
                            request->request_id(),
                            request->generate_config().unique_key().c_str(),
                            effective_request->generate_config().unique_key().c_str(),
                            static_cast<int>(status.error_code()),
                            static_cast<int>(error.code()),
                            ErrorCodeToString(error.code()).c_str(),
                            error.ToString().c_str());
    };

    AtomicGuard request_guard(onflight_requests_);
    auto        request_id = request->request_id();
    RTP_LLM_LOG_DEBUG("receive request %ld", request_id);
    PDCancelRegistry::Handle cancel_handle;
    auto                     admission = cancel_registry_->admit(*effective_request,
                                             effective_request->generate_config().unique_key(),
                                             prefillAddress(*effective_request),
                                             request_entry_ms + normalized_timeout_ms,
                                             cancel_handle);
    if (!admission.ok())
        return serializeErrorMsg(std::to_string(request_id), admission);
    PDCancelRegistry::CallGuard cancel_guard(*cancel_registry_, cancel_handle);
    auto                        generate_context =
        GenerateContext(request_id, normalized_timeout_ms, server_context, metrics_reporter_, meta_);
    struct PrefillCancelGuard {
        PDCancelRegistry&        registry;
        PDCancelRegistry::Handle handle;
        GenerateContext&         context;
        bool                     completed{false};
        void                     cancel() {
            auto reason = errorInfoFromGrpcStatus(context.error_status);
            if (!reason.hasError())
                reason = {ErrorCode::CANCELLED, "PD request abandoned before completion"};
            registry.cancel(handle->identity.request_id, reason);
        }
        ~PrefillCancelGuard() {
            if (!completed)
                cancel();
        }
    } prefill_cancel_guard{*cancel_registry_, cancel_handle, generate_context};
    std::shared_ptr<GenerateInput> input;
    GenerateInputPB                prefill_request;
    PrefillPeerInfo                peer_info;
    const auto                     prepare_start_us = currentTimeUs();
    auto                           prepare_status   = preparePDRequest(
        *effective_request, request_entry_ms + normalized_timeout_ms, input, prefill_request, peer_info);
    {
        RpcMetricsCollector collector;
        collector.min_response_done_time_us      = 0;
        collector.prepare_generate_context_rt_us = currentTimeUs() - prepare_start_us;
        generate_context.reportMetrics(collector);
    }
    if (!prepare_status.ok()) {
        generate_context.error_status = prepare_status;
        log_pd_error("prepare", generate_context.error_status);
        return generate_context.error_status;
    }
    if (cancel_handle->canceled.load()) {
        generate_context.error_info   = cancel_handle->cancel_reason;
        generate_context.error_status = serializeErrorMsg(std::to_string(request_id), generate_context.error_info);
        return generate_context.error_status;
    }
    auto stream = engine_->makeStream(input);
    stream->setPrefillTpSize(peer_info.tp_size);
    stream->setPrefillCpSize(peer_info.cp_size);
    generate_context.setStream(stream);
    cancel_registry_->attach(cancel_handle, stream);

    std::shared_ptr<PrefillServerCallerContext> prefill_caller_ctx;
    PrefillContextGuard                         prefill_context_guard;
    if (!request->enqueued_by_master()) {
        if (cancel_handle->canceled.load()) {
            engine_->enqueue(stream);  // Let the scheduler commit canceled lifecycle cleanup.
            generate_context.error_info   = cancel_handle->cancel_reason;
            generate_context.error_status = serializeErrorMsg(std::to_string(request_id), generate_context.error_info);
            return generate_context.error_status;
        }
        const auto& unique_key          = input->generate_config->unique_key;
        const auto  request_deadline_ms = input->request_deadline_ms;
        // The frontend owns Prefill selection. Use the same endpoint as the peer-info probe.
        std::string target_ip;
        uint32_t    target_port  = 0;
        auto        parse_status = parsePrefillDpAddr(prefillAddress(prefill_request), &target_ip, &target_port);
        if (!parse_status.ok()) {
            const auto status = parse_status;
            log_pd_error("prefill_address", status);
            return status;
        }
        const auto prefill_call_start_us = currentTimeUs();
        auto       started               = prefill_server_caller_->callPrefill(
            &prefill_request, target_ip, target_port, unique_key, request_deadline_ms);
        {
            RpcMetricsCollector collector;
            collector.min_response_done_time_us = 0;
            collector.remote_generate_rt_us     = currentTimeUs() - prefill_call_start_us;
            generate_context.reportMetrics(collector);
        }
        if (!started.ok()) {
            generate_context.error_info = started.status();
            generate_context.error_status =
                serializeErrorMsg(generate_context.request_key, generate_context.error_info);
            log_pd_error("call_prefill", generate_context.error_status);
            return generate_context.error_status;
        }
        prefill_caller_ctx            = std::move(started.value());
        prefill_context_guard.context = prefill_caller_ctx;
    }

    engine_->enqueue(stream);

    // Keep the async prefill RPC alive for the handoff lifecycle only. StartLoad applies the first token to the
    // decode stream, so client output must come exclusively from pollStreamOutput().
    if (prefill_caller_ctx) {
        generate_context.error_status = pollStreamOutput(server_context,
                                                         generate_context.request_key,
                                                         response_writer,
                                                         generate_context.getStream(),
                                                         [&](bool& done) {
                                                             done = prefill_caller_ctx->done();
                                                             return prefill_caller_ctx->firstError();
                                                         });
    } else {
        // Prefill was admitted by Master. Decode still enters the normal
        // engine path, which starts StartLoad/asyncRead for this handoff key.
        generate_context.error_status = pollStreamOutput(
            server_context, generate_context.request_key, response_writer, generate_context.getStream());
    }
    prefill_cancel_guard.completed = generate_context.error_status.ok() && !server_context->IsCancelled()
                                     && stream->getStatus() == StreamState::FINISHED;
    if (prefill_cancel_guard.completed)
        cancel_registry_->finishDownstream(cancel_handle);
    else
        prefill_cancel_guard.cancel();
    meta_->dequeue(generate_context.request_id, generate_context.getStream());

    if (prefill_caller_ctx && (!generate_context.error_status.ok() || server_context->IsCancelled())) {
        prefill_caller_ctx->cancel();
    }
    if (!generate_context.error_status.ok()) {
        log_pd_error("stream_output", generate_context.error_status);
    }
    return generate_context.error_status;
}

void DecodeRpcServer::updateAuxInfo(GenerateOutputsPB& outputs_pb, std::shared_ptr<GenerateStream>& stream) {
    auto       first_token_rt_us = stream->getTimeInfo().first_token_rt_us;
    auto       cost_time_us      = autil::TimeUtility::currentTimeInMicroSeconds() - stream->beginTimeUs();
    const auto reuse_lens        = getPrefillReuseLens(stream);

    for (size_t i = 0; i < outputs_pb.flatten_output().aux_info_size(); i++) {
        auto       aux_info                = outputs_pb.mutable_flatten_output()->mutable_aux_info(i);
        const auto decode_total_reuse_len  = aux_info->total_reuse_len();
        const auto decode_local_reuse_len  = aux_info->local_reuse_len();
        const auto decode_remote_reuse_len = aux_info->remote_reuse_len();
        const auto decode_memory_reuse_len = aux_info->memory_reuse_len();
        const auto decode_disk_reuse_len   = aux_info->disk_reuse_len();
        aux_info->set_first_token_cost_time_us(first_token_rt_us);
        aux_info->set_cost_time_us(cost_time_us);
        aux_info->set_pd_sep(true);

        const bool expose_decode_reuse =
            stream->prefillUsesIndependentBlockPools() && decode_total_reuse_len > reuse_lens.total;
        aux_info->set_total_reuse_len(expose_decode_reuse ? decode_total_reuse_len : reuse_lens.total);
        aux_info->set_local_reuse_len(expose_decode_reuse ? decode_local_reuse_len : reuse_lens.local);
        aux_info->set_remote_reuse_len(expose_decode_reuse ? decode_remote_reuse_len : reuse_lens.remote);
        aux_info->set_memory_reuse_len(expose_decode_reuse ? decode_memory_reuse_len : reuse_lens.memory);
        aux_info->set_disk_reuse_len(expose_decode_reuse ? decode_disk_reuse_len : reuse_lens.disk);

        aux_info->set_prefill_total_reuse_len(reuse_lens.total);
        aux_info->set_prefill_local_reuse_len(reuse_lens.local);
        aux_info->set_prefill_remote_reuse_len(reuse_lens.remote);
        aux_info->set_prefill_memory_reuse_len(static_cast<int32_t>(reuse_lens.memory));
        aux_info->set_prefill_disk_reuse_len(static_cast<int32_t>(reuse_lens.disk));

        aux_info->set_decode_total_reuse_len(decode_total_reuse_len);
        aux_info->set_decode_local_reuse_len(decode_local_reuse_len);
        aux_info->set_decode_remote_reuse_len(decode_remote_reuse_len);
        aux_info->set_decode_memory_reuse_len(decode_memory_reuse_len);
        aux_info->set_decode_disk_reuse_len(decode_disk_reuse_len);
    }
}

}  // namespace rtp_llm
