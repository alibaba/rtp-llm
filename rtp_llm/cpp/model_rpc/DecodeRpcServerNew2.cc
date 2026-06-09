#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.h"
#include "rtp_llm/cpp/model_rpc/RpcTimeoutUtils.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/engine_base/Host.h"
#include "autil/NetUtil.h"
#include <chrono>
#include <cstring>
#include <thread>

namespace rtp_llm {

namespace {

struct ReuseLens {
    int64_t total  = 0;
    int64_t local  = 0;
    int64_t remote = 0;
    int64_t memory = 0;

    bool hasValue() const {
        return total > 0 || local > 0 || remote > 0 || memory > 0;
    }
};

ReuseLens getPrefillReuseLens(std::shared_ptr<GenerateStream>&                   stream,
                              const std::shared_ptr<PrefillServerCallerContext>& prefill_ctx) {
    ReuseLens reuse_lens;
    reuse_lens.total  = stream->prefillTotalReuseLen();
    reuse_lens.local  = stream->prefillLocalReuseLen();
    reuse_lens.remote = stream->prefillRemoteReuseLen();
    reuse_lens.memory = stream->prefillMemoryReuseLen();
    if (reuse_lens.hasValue() || !prefill_ctx) {
        return reuse_lens;
    }

    PrefillServerCallerContext::ReuseLensSnapshot snapshot;
    if (!prefill_ctx->getPrefillReuseLensSnapshot(snapshot)) {
        return reuse_lens;
    }

    reuse_lens.total  = snapshot.total;
    reuse_lens.local  = snapshot.local;
    reuse_lens.remote = snapshot.remote;
    reuse_lens.memory = snapshot.memory;
    stream->setPrefillReuseLength(reuse_lens.total, reuse_lens.local, reuse_lens.remote, reuse_lens.memory);
    return reuse_lens;
}

}  // namespace

bool decodeEntranceRequiresPrefill(const GenerateInputPB& request) {
    return request.generate_config().max_new_tokens() > 1 && request.generate_config().num_beams() <= 1
           && request.generate_config().variable_num_beams().size() == 0
           && request.generate_config().num_return_sequences() <= 1
           && request.generate_config().can_use_pd_separation();
}

std::string makeDecodeEntranceUniqueKey(const std::string& bind_ip, int64_t unique_key_id, int64_t current_time_us) {
    return bind_ip + "_" + std::to_string(unique_key_id) + "_" + std::to_string(current_time_us);
}

DecodeEntranceKeys buildDecodeEntranceKeys(const GenerateInputPB& request,
                                           const std::string&     bind_ip,
                                           int64_t                unique_key_id,
                                           int64_t                current_time_us) {
    DecodeEntranceKeys keys;
    keys.business_unique_key = request.generate_config().unique_key();
    keys.handoff_unique_key  = makeDecodeEntranceUniqueKey(bind_ip, unique_key_id, current_time_us);
    return keys;
}

GenerateInputPB makeDecodeEntranceHandoffRequest(const GenerateInputPB& request,
                                                 const std::string&     handoff_unique_key) {
    GenerateInputPB handoff_request;
    handoff_request.CopyFrom(request);
    handoff_request.mutable_generate_config()->set_unique_key(handoff_unique_key);
    return handoff_request;
}

grpc::Status
DecodeRpcServerNew2::pollStreamOutputWithPrefill(grpc::ServerContext*                               context,
                                                 const std::string&                                 request_key,
                                                 WriterInterface*                                   writer,
                                                 std::shared_ptr<GenerateStream>&                   stream,
                                                 const std::shared_ptr<PrefillServerCallerContext>& prefill_ctx) {
    auto propagate_prefill_error = [&]() -> grpc::Status {
        if (!prefill_ctx || !prefill_ctx->failed()) {
            return grpc::Status::OK;
        }
        auto error_info = prefill_ctx->errorInfo();
        if (error_info.hasError() && !stream->hasError()) {
            stream->reportError(error_info.code(), error_info.ToString());
        }
        return serializeErrorMsg(request_key, error_info);
    };

    bool first_token_sent        = false;
    bool skip_next_decode_output = false;

    while (stream->isActive() || stream->hasOutput()) {
        auto prefill_status = propagate_prefill_error();
        if (!prefill_status.ok()) {
            return prefill_status;
        }
        if (context->IsCancelled()) {
            stream->reportError(ErrorCode::CANCELLED, "request cancelled by user");
            RTP_LLM_LOG_WARNING("request [%s] cancelled by user", request_key.c_str());
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
        }

        if (!first_token_sent && prefill_ctx) {
            GenerateOutputsPB prefill_output;
            if (prefill_ctx->takeFirstResponse(prefill_output)) {
                first_token_sent        = true;
                skip_next_decode_output = true;
                for (int i = 0; i < prefill_output.mutable_flatten_output()->finished_size(); ++i) {
                    prefill_output.mutable_flatten_output()->set_finished(i, false);
                }
                updateDecodeAuxInfo(prefill_output, stream, prefill_ctx);
                if (!writer->Write(prefill_output)) {
                    stream->reportError(ErrorCode::CANCELLED, "write prefill first token failed");
                    RTP_LLM_LOG_WARNING("request [%s] write prefill first token failed", request_key.c_str());
                    return grpc::Status(grpc::StatusCode::INTERNAL, "write prefill first token failed");
                }
                continue;
            }
        }

        if (!stream->hasOutput()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        const auto result = stream->nextOutput();
        if (!result.ok()) {
            if (result.status().code() != ErrorCode::FINISHED) {
                return serializeErrorMsg(request_key, result.status());
            } else {
                break;
            }
        }

        if (!first_token_sent) {
            first_token_sent = true;
        } else if (skip_next_decode_output) {
            skip_next_decode_output = false;
            continue;
        }

        RTP_LLM_LOG_DEBUG("request [%s] generate next output success", request_key.c_str());
        GenerateOutputsPB outputs_pb;

        QueryConverter::transResponse(&outputs_pb,
                                      &(result.value()),
                                      stream->generateConfig()->aux_info,
                                      maga_init_params_.misc_config.aux_string,
                                      stream->specialTokens().eos_token_id);
        updateDecodeAuxInfo(outputs_pb, stream, prefill_ctx);
        if (!writer->Write(outputs_pb)) {
            stream->reportError(ErrorCode::CANCELLED, "write outputs pb failed");
            RTP_LLM_LOG_WARNING("request [%s] write outputs pb failed", request_key.c_str());
            return grpc::Status(grpc::StatusCode::INTERNAL, "request write outputs pb failed");
        }
        if (stream->hasEvent(StreamEvents::NeedRemoteGenerate)) {
            break;
        }
    }

    auto prefill_status = propagate_prefill_error();
    if (!prefill_status.ok()) {
        return prefill_status;
    }
    if (stream->hasError()) {
        return serializeErrorMsg(request_key, stream->statusInfo());
    }
    RTP_LLM_LOG_DEBUG("request [%s] decode generate done", request_key.c_str());
    return grpc::Status::OK;
}

grpc::Status DecodeRpcServerNew2::init(const EngineInitParams&                                maga_init_params,
                                       py::object                                             mm_process_engine,
                                       std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    auto ret = RemoteRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params), false);
    if (!ret.ok()) {
        RTP_LLM_LOG_ERROR("decode rpc server new2 init failed, err: %s", ret.error_message().c_str());
        return ret;
    }

    // get memory connector from kvcache manager and set to connector coordinator
    auto kvcache_manager = engine_->getCacheManager();
    if (!kvcache_manager) {
        RTP_LLM_LOG_WARNING("decode rpc server new2 init failed, kvcache manager is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "kvcache manager is null");
    }
    // Validate P2P connector coordinator is initialized (required for PD separation cache transfer).
    // The coordinator itself is accessed later via KVCacheManager during cache operations.
    auto connector_coordinator = kvcache_manager->connectorCoordinator();
    if (!connector_coordinator) {
        RTP_LLM_LOG_WARNING("decode rpc server new2 init failed, connector coordinator is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "connector coordinator is null");
    }
    if (!kvcache_manager->hasP2PConnector()) {
        RTP_LLM_LOG_WARNING("decode rpc server new2 init failed, decode_entrance requires P2P connector");
        return grpc::Status(grpc::StatusCode::INTERNAL, "decode_entrance requires P2P connector");
    }

    prefill_server_caller_ = std::make_shared<PrefillServerCaller>(process_id_);

    RTP_LLM_LOG_INFO("decode rpc server new2 init");
    return grpc::Status::OK;
}

grpc::Status DecodeRpcServerNew2::GenerateStreamCall(grpc::ServerContext*                   server_context,
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

    const auto normalized_timeout_ms     = normalizeRpcTimeoutMs(request->generate_config().timeout_ms(),
                                                             maga_init_params_.pd_sep_config.max_rpc_timeout_ms);
    const auto normalized_timeout_ms_i32 = clampRpcTimeoutMsToInt32(normalized_timeout_ms);

    // Always isolate decode-entrance P2P handoff from any caller-provided business unique_key.
    // The external unique_key remains on the original request object, while the internal handoff
    // request gets a per-request key used only by the prefill/decode P2P pipeline.
    GenerateInputPB request_with_handoff_key;
    const auto*     effective_request = request;

    // Check if pd separation should be used
    auto pd_separation = decodeEntranceRequiresPrefill(*request);
    if (pd_separation) {
        auto decode_entrance_keys = buildDecodeEntranceKeys(
            *request, autil::NetUtil::getBindIp(), unique_key_id_.fetch_add(1), currentTimeUs());
        request_with_handoff_key = makeDecodeEntranceHandoffRequest(*request, decode_entrance_keys.handoff_unique_key);
        effective_request        = &request_with_handoff_key;
    }
    if (!pd_separation) {
        RTP_LLM_LOG_DEBUG("pd separation is disabled, call prefill server");
        GenerateInputPB prefill_forward_request;
        prefill_forward_request.CopyFrom(*effective_request);
        prefill_forward_request.mutable_generate_config()->set_timeout_ms(normalized_timeout_ms_i32);
        return prefill_server_caller_->callPrefill(server_context, &prefill_forward_request, response_writer);
    }

    AtomicGuard request_guard(onflight_requests_);
    auto        request_id = request->request_id();
    RTP_LLM_LOG_DEBUG("receive request %ld", request_id);
    auto generate_context =
        GenerateContext(request_id, normalized_timeout_ms, server_context, metrics_reporter_, meta_);
    auto input                         = QueryConverter::transQuery(effective_request);
    input->generate_config->timeout_ms = normalized_timeout_ms_i32;

    // need to check client has buffer at first
    if (mm_processor_ != nullptr && input->multimodal_inputs) {
        auto mm_res = mm_processor_->updateMultimodalFeatures(input);
        if (!mm_res.ok()) {
            generate_context.error_status = serializeErrorMsg(generate_context.request_key, mm_res);
        }
    }
    CHECK_ERROR_STATUS(generate_context);

    RTP_LLM_LOG_DEBUG("request [%ld] trans to stream success", request_id);
    input->generate_config->pd_separation        = true;
    input->generate_config->force_disable_sp_run = !engine_->isMTPEagle();
    auto stream                                  = engine_->makeStream(input);
    generate_context.setStream(stream);

    // 获取 prefill peer info（含 DP 地址列表），round-robin 选择目标 DP
    std::string prefill_ip;
    uint32_t    prefill_port = 0;
    for (const auto& role_addr : request->generate_config().role_addrs()) {
        if (role_addr.role() == RoleAddrPB::PREFILL) {
            prefill_ip   = role_addr.ip();
            prefill_port = role_addr.grpc_port();
            break;
        }
    }
    if (prefill_ip.empty() || prefill_port <= 0) {
        RTP_LLM_LOG_WARNING("decode rpc server new2 generate failed: prefill addr unavailable, request_id=%ld",
                            request_id);
        return grpc::Status(grpc::StatusCode::INTERNAL, "prefill_ip or prefill_port is not available");
    }

    auto peer_info = prefill_server_caller_->getPrefillPeerInfo(prefill_ip, prefill_port, normalized_timeout_ms_i32);
    if (peer_info.tp_size <= 0 || peer_info.dp_addrs.empty()) {
        RTP_LLM_LOG_WARNING("decode rpc server new2 generate failed: prefill peer info unavailable, "
                            "request_id=%ld, prefill_addr=%s:%u",
                            request_id,
                            prefill_ip.c_str(),
                            prefill_port);
        return grpc::Status(grpc::StatusCode::INTERNAL, "prefill peer info is not available");
    }
    stream->setPrefillTpSize(peer_info.tp_size);

    // Round-robin across prefill DPs using request_id
    const auto& dp_addrs      = peer_info.dp_addrs;
    size_t      dp_index      = static_cast<size_t>(request_id) % dp_addrs.size();
    const auto& selected_addr = dp_addrs[dp_index];
    auto colon_pos = selected_addr.rfind(':');
    if (colon_pos == std::string::npos || colon_pos + 1 >= selected_addr.size()) {
        RTP_LLM_LOG_ERROR("malformed prefill dp_addr: %s", selected_addr.c_str());
        return grpc::Status(grpc::StatusCode::INTERNAL, "malformed prefill dp_addr: " + selected_addr);
    }
    std::string target_ip   = selected_addr.substr(0, colon_pos);
    uint32_t    target_port = 0;
    try {
        target_port = static_cast<uint32_t>(std::stoul(selected_addr.substr(colon_pos + 1)));
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("failed to parse port from dp_addr '%s': %s", selected_addr.c_str(), e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, "invalid port in prefill dp_addr: " + selected_addr);
    }

    // Update the stream's PREFILL role_addr to point to the selected DP rank
    // so that StartLoad (P2P cache read) goes to the same prefill that runs
    // the GenerateStreamCall, not the initial DP-0 address from the router.
    for (auto& role_addr : input->generate_config->role_addrs) {
        if (role_addr.role == RoleType::PREFILL) {
            role_addr.ip        = target_ip;
            role_addr.grpc_port = static_cast<int>(target_port);
            break;
        }
    }

    std::shared_ptr<PrefillServerCallerContext> prefill_caller_ctx;
    PrefillContextGuard                         prefill_context_guard;
    const auto&                                 unique_key  = input->generate_config->unique_key;
    auto                                        deadline_us = normalized_timeout_ms * 1000;
    GenerateInputPB                             prefill_request;
    prefill_request.CopyFrom(*effective_request);
    prefill_request.mutable_generate_config()->set_timeout_ms(normalized_timeout_ms_i32);
    prefill_request.mutable_generate_config()->set_unique_key(unique_key);
    prefill_caller_ctx =
        prefill_server_caller_->callPrefill(&prefill_request, target_ip, target_port, unique_key, deadline_us);
    if (!prefill_caller_ctx) {
        generate_context.error_info   = ErrorInfo(ErrorCode::P2P_CONNECTOR_CALL_PREFILL_FAILED,
                                                "failed to start async prefill request to " + selected_addr);
        generate_context.error_status = serializeErrorMsg(generate_context.request_key, generate_context.error_info);
        return generate_context.error_status;
    }
    prefill_context_guard.context = prefill_caller_ctx;

    engine_->enqueue(stream);

    generate_context.error_status = pollStreamOutputWithPrefill(server_context,
                                                                generate_context.request_key,
                                                                response_writer,
                                                                generate_context.getStream(),
                                                                prefill_caller_ctx);
    meta_->dequeue(generate_context.request_id, generate_context.getStream());

    if (prefill_caller_ctx && (!generate_context.error_status.ok() || server_context->IsCancelled())) {
        prefill_caller_ctx->cancel();
    }
    return generate_context.error_status;
}

void updateDecodeAuxInfo(GenerateOutputsPB&                                 outputs_pb,
                         std::shared_ptr<GenerateStream>&                   stream,
                         const std::shared_ptr<PrefillServerCallerContext>& prefill_ctx) {
    auto       first_token_rt_us = stream->getTimeInfo().first_token_rt_us;
    auto       cost_time_us      = autil::TimeUtility::currentTimeInMicroSeconds() - stream->beginTimeUs();
    const auto reuse_lens        = getPrefillReuseLens(stream, prefill_ctx);

    for (size_t i = 0; i < outputs_pb.flatten_output().aux_info_size(); i++) {
        auto       aux_info                = outputs_pb.mutable_flatten_output()->mutable_aux_info(i);
        const auto decode_total_reuse_len  = aux_info->total_reuse_len();
        const auto decode_local_reuse_len  = aux_info->local_reuse_len();
        const auto decode_remote_reuse_len = aux_info->remote_reuse_len();
        const auto decode_memory_reuse_len = aux_info->memory_reuse_len();
        aux_info->set_first_token_cost_time_us(first_token_rt_us);
        aux_info->set_cost_time_us(cost_time_us);
        aux_info->set_pd_sep(true);

        // use prefill as
        aux_info->set_total_reuse_len(reuse_lens.total);
        aux_info->set_local_reuse_len(reuse_lens.local);
        aux_info->set_remote_reuse_len(reuse_lens.remote);
        aux_info->set_memory_reuse_len(static_cast<int32_t>(reuse_lens.memory));

        aux_info->set_prefill_total_reuse_len(reuse_lens.total);
        aux_info->set_prefill_local_reuse_len(reuse_lens.local);
        aux_info->set_prefill_remote_reuse_len(reuse_lens.remote);
        aux_info->set_prefill_memory_reuse_len(static_cast<int32_t>(reuse_lens.memory));

        aux_info->set_decode_total_reuse_len(decode_total_reuse_len);
        aux_info->set_decode_local_reuse_len(decode_local_reuse_len);
        aux_info->set_decode_remote_reuse_len(decode_remote_reuse_len);
        aux_info->set_decode_memory_reuse_len(decode_memory_reuse_len);
    }
}

}  // namespace rtp_llm
