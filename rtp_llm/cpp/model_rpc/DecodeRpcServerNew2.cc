#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.h"
#include "rtp_llm/cpp/model_rpc/RpcTimeoutUtils.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/engine_base/Host.h"
#include "autil/NetUtil.h"
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
    char* end   = nullptr;
    errno       = 0;
    auto value  = std::strtoll(port_text.c_str(), &end, 10);
    const bool parsed_all = end == port_text.c_str() + port_text.size();
    if (errno != 0 || !parsed_all || value < kMinGrpcPort || value > kMaxGrpcPort) {
        return false;
    }
    *port = static_cast<uint32_t>(value);
    return true;
}

}  // namespace

bool shouldUsePDSeparation(const GenerateInputPB& request) {
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

size_t selectDecodeEntranceDpIndex(size_t dp_count, int64_t handoff_id) {
    if (dp_count == 0) {
        return 0;
    }
    return static_cast<size_t>(handoff_id) % dp_count;
}

grpc::Status DecodeRpcServerNew2::parsePrefillDpAddr(const std::string& addr, std::string* ip, uint32_t* port) {
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

grpc::Status DecodeRpcServerNew2::init(const EngineInitParams&                                maga_init_params,
                                       py::object                                             mm_process_engine,
                                       std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    auto ret = RemoteRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params), false);
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

    const int64_t request_entry_ms = currentTimeMs();
    const auto normalized_timeout_ms     = normalizeRpcTimeoutMs(request->generate_config().timeout_ms(),
                                                             maga_init_params_.pd_sep_config.max_rpc_timeout_ms);
    const auto normalized_timeout_ms_i32 = clampRpcTimeoutMsToInt32(normalized_timeout_ms);

    // Always isolate decode-entrance P2P handoff from any caller-provided business unique_key.
    // The external unique_key remains on the original request object, while the internal handoff
    // request gets a per-request key used only by the prefill/decode P2P pipeline.
    GenerateInputPB request_with_handoff_key;
    const auto*     effective_request = request;
    int64_t         handoff_id        = 0;

    // Check if pd separation should be used
    auto pd_separation = shouldUsePDSeparation(*request);
    if (pd_separation) {
        handoff_id                = unique_key_id_.fetch_add(1);
        auto decode_entrance_keys =
            buildDecodeEntranceKeys(*request, autil::NetUtil::getBindIp(), handoff_id, currentTimeUs());
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
    input->request_deadline_ms = request_entry_ms + normalized_timeout_ms;

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
    if (peer_info.tp_size <= 0 || peer_info.cp_size <= 0 || peer_info.dp_addrs.empty()) {
        RTP_LLM_LOG_WARNING("decode rpc server new2 generate failed: prefill peer info unavailable, "
                            "request_id=%ld, prefill_addr=%s:%u",
                            request_id,
                            prefill_ip.c_str(),
                            prefill_port);
        return grpc::Status(grpc::StatusCode::INTERNAL, "prefill peer info is not available");
    }
    stream->setPrefillTpSize(peer_info.tp_size);
    stream->setPrefillCpSize(peer_info.cp_size);

    std::shared_ptr<PrefillServerCallerContext> prefill_caller_ctx;
    PrefillContextGuard                         prefill_context_guard;
    const auto&                                 unique_key  = input->generate_config->unique_key;
    const auto request_deadline_ms = input->request_deadline_ms;
    GenerateInputPB                             prefill_request;
    prefill_request.CopyFrom(*effective_request);
    prefill_request.set_request_deadline_ms(request_deadline_ms);
    prefill_request.mutable_generate_config()->set_timeout_ms(normalized_timeout_ms_i32);
    prefill_request.mutable_generate_config()->set_unique_key(unique_key);

    auto update_prefill_role_addr = [&](const std::string& target_ip, uint32_t target_port) {
        // Keep StartLoad (P2P cache read) on the same prefill DP rank that runs GenerateStreamCall.
        for (auto& role_addr : input->generate_config->role_addrs) {
            if (role_addr.role == RoleType::PREFILL) {
                role_addr.ip        = target_ip;
                role_addr.grpc_port = static_cast<int>(target_port);
                break;
            }
        }
        for (auto& role_addr : *prefill_request.mutable_generate_config()->mutable_role_addrs()) {
            if (role_addr.role() == RoleAddrPB::PREFILL) {
                role_addr.set_ip(target_ip);
                role_addr.set_grpc_port(static_cast<int>(target_port));
                break;
            }
        }
    };

    const auto& dp_addrs       = peer_info.dp_addrs;
    const auto  first_dp_index = selectDecodeEntranceDpIndex(dp_addrs.size(), handoff_id);
    std::string selected_addr;
    for (size_t attempt = 0; attempt < dp_addrs.size(); ++attempt) {
        const auto dp_index = (first_dp_index + attempt) % dp_addrs.size();
        selected_addr       = dp_addrs[dp_index];

        std::string target_ip;
        uint32_t    target_port = 0;
        auto        parse_status = parsePrefillDpAddr(selected_addr, &target_ip, &target_port);
        if (!parse_status.ok()) {
            return parse_status;
        }
        update_prefill_role_addr(target_ip, target_port);

        prefill_caller_ctx =
            prefill_server_caller_->callPrefill(&prefill_request, target_ip, target_port, unique_key, request_deadline_ms);
        if (prefill_caller_ctx) {
            if (attempt > 0) {
                RTP_LLM_LOG_WARNING("request [%ld] recovered async prefill by trying next DP, addr=%s",
                                    request_id,
                                    selected_addr.c_str());
            }
            break;
        }

        prefill_server_caller_->invalidatePrefillPeerInfo(prefill_ip, prefill_port);
        RTP_LLM_LOG_WARNING("request [%ld] async prefill start failed for DP %s, attempt %zu/%zu",
                            request_id,
                            selected_addr.c_str(),
                            attempt + 1,
                            dp_addrs.size());
    }
    if (!prefill_caller_ctx) {
        generate_context.error_info   = ErrorInfo(ErrorCode::P2P_CONNECTOR_CALL_PREFILL_FAILED,
                                                "failed to start async prefill request to cached DP addrs");
        generate_context.error_status = serializeErrorMsg(generate_context.request_key, generate_context.error_info);
        return generate_context.error_status;
    }
    prefill_context_guard.context = prefill_caller_ctx;

    engine_->enqueue(stream);

    // Keep the async prefill RPC alive for the handoff lifecycle only. StartLoad applies the first token to the
    // decode stream, so client output must come exclusively from pollStreamOutput().
    generate_context.error_status =
        pollStreamOutput(server_context, generate_context.request_key, response_writer, generate_context.getStream());
    meta_->dequeue(generate_context.request_id, generate_context.getStream());

    if (prefill_caller_ctx && prefill_caller_ctx->failed()) {
        prefill_server_caller_->invalidatePrefillPeerInfo(prefill_ip, prefill_port);
    }
    if (prefill_caller_ctx && (!generate_context.error_status.ok() || server_context->IsCancelled())) {
        prefill_caller_ctx->cancel();
    }
    return generate_context.error_status;
}

void DecodeRpcServerNew2::updateAuxInfo(GenerateOutputsPB& outputs_pb, std::shared_ptr<GenerateStream>& stream) {
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

        // use prefill as
        aux_info->set_total_reuse_len(reuse_lens.total);
        aux_info->set_local_reuse_len(reuse_lens.local);
        aux_info->set_remote_reuse_len(reuse_lens.remote);
        aux_info->set_memory_reuse_len(static_cast<int32_t>(reuse_lens.memory));
        aux_info->set_disk_reuse_len(static_cast<int32_t>(reuse_lens.disk));

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
