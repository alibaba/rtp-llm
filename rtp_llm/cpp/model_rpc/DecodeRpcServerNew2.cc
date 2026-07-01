#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.h"
#include "rtp_llm/cpp/model_rpc/RpcTimeoutUtils.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/engine_base/Host.h"
#include "autil/NetUtil.h"
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <thread>
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

bool DecodeRpcServerNew2::outputContainsFinished(const GenerateOutputsPB& output) {
    if (!output.has_flatten_output()) {
        return false;
    }
    for (int i = 0; i < output.flatten_output().finished_size(); ++i) {
        if (output.flatten_output().finished(i)) {
            return true;
        }
    }
    return false;
}

bool DecodeRpcServerNew2::refreshIdleStreamState(std::shared_ptr<GenerateStream>& stream) {
    if (!stream) {
        return false;
    }
    stream->checkTimeout();
    return stream->hasError();
}

bool DecodeRpcServerNew2::consumePrefillFirstResponse(
    const std::shared_ptr<PrefillServerCallerContext>& prefill_ctx,
    std::shared_ptr<GenerateStream>&                   stream,
    bool                                               client_first_chunk_sent,
    bool*                                              prefill_finished,
    int*                                               prefill_finished_size,
    bool*                                              skip_next_decode_output,
    GenerateOutputsPB*                                client_output) {
    if (!prefill_ctx) {
        return false;
    }

    GenerateOutputsPB prefill_output;
    if (!prefill_ctx->takeFirstResponse(prefill_output)) {
        return false;
    }

    const int finished_size = prefill_output.has_flatten_output() ? prefill_output.flatten_output().finished_size() : 0;
    bool      finished      = false;
    for (int i = 0; i < finished_size; ++i) {
        if (prefill_output.flatten_output().finished(i)) {
            finished = true;
            break;
        }
    }

    if (prefill_finished) {
        *prefill_finished = finished;
    }
    if (prefill_finished_size) {
        *prefill_finished_size = finished_size;
    }
    if (skip_next_decode_output) {
        // Once the client has consumed prefill's first token, the next decode
        // chunk is the same first token and must be suppressed even when that
        // token also terminates the stream. A terminal frame is emitted later
        // if decode does not deliver one itself.
        *skip_next_decode_output = !client_first_chunk_sent;
    }

    if (client_first_chunk_sent) {
        RTP_LLM_LOG_DEBUG("decode_entrance observed late prefill first response, finished=%d, unique_key=%s",
                          finished,
                          stream ? stream->uniqueKey().c_str() : "");
        return false;
    }

    for (int i = 0; i < finished_size; ++i) {
        prefill_output.mutable_flatten_output()->set_finished(i, false);
    }
    updateDecodeAuxInfo(prefill_output, stream, prefill_ctx);
    if (client_output) {
        client_output->Swap(&prefill_output);
    }
    return true;
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
    bool prefill_finished        = false;  // Track if prefill first response indicates termination
    bool client_finished_sent    = false;  // Track whether client has seen a terminal frame
    int  prefill_finished_size   = 0;      // batch/beam count from prefill first response (for termination frame)

    auto mark_client_finished = [&](const GenerateOutputsPB& output) {
        client_finished_sent = client_finished_sent || outputContainsFinished(output);
    };

    auto try_write_prefill_first_token = [&](bool* wrote) -> grpc::Status {
        if (wrote) {
            *wrote = false;
        }
        if (first_token_sent || !prefill_ctx) {
            return grpc::Status::OK;
        }

        GenerateOutputsPB prefill_output;
        if (!consumePrefillFirstResponse(prefill_ctx,
                                         stream,
                                         /*client_first_chunk_sent=*/false,
                                         &prefill_finished,
                                         &prefill_finished_size,
                                         &skip_next_decode_output,
                                         &prefill_output)) {
            return grpc::Status::OK;
        }

        first_token_sent = true;
        if (!writer->Write(prefill_output)) {
            stream->reportError(ErrorCode::CANCELLED, "write prefill first token failed");
            RTP_LLM_LOG_WARNING("request [%s] write prefill first token failed", request_key.c_str());
            return grpc::Status(grpc::StatusCode::INTERNAL, "write prefill first token failed");
        }
        mark_client_finished(prefill_output);
        if (wrote) {
            *wrote = true;
        }
        return grpc::Status::OK;
    };

    auto observe_prefill_first_token_as_late = [&]() {
        if (!prefill_ctx || prefill_finished) {
            return;
        }
        (void)consumePrefillFirstResponse(prefill_ctx,
                                          stream,
                                          /*client_first_chunk_sent=*/true,
                                          &prefill_finished,
                                          &prefill_finished_size,
                                          &skip_next_decode_output,
                                          /*client_output=*/nullptr);
    };

    auto drop_current_decode_output_if_duplicate = [&]() {
        if (!first_token_sent) {
            first_token_sent = true;
            return false;
        }
        if (skip_next_decode_output) {
            skip_next_decode_output = false;
            return true;
        }
        return false;
    };

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

        bool wrote_prefill_first = false;
        auto prefill_write_status = try_write_prefill_first_token(&wrote_prefill_first);
        if (!prefill_write_status.ok()) {
            return prefill_write_status;
        }
        if (wrote_prefill_first) {
            continue;
        }

        if (!stream->hasOutput()) {
            if (refreshIdleStreamState(stream)) {
                return serializeErrorMsg(request_key, stream->statusInfo());
            }
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

        wrote_prefill_first = false;
        prefill_write_status = try_write_prefill_first_token(&wrote_prefill_first);
        if (!prefill_write_status.ok()) {
            return prefill_write_status;
        }
        if (!wrote_prefill_first && first_token_sent) {
            observe_prefill_first_token_as_late();
        }

        if (drop_current_decode_output_if_duplicate()) {
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
        mark_client_finished(outputs_pb);
        if (stream->hasEvent(StreamEvents::NeedRemoteGenerate)) {
            break;
        }
    }

    observe_prefill_first_token_as_late();

    auto prefill_status = propagate_prefill_error();
    if (!prefill_status.ok()) {
        return prefill_status;
    }
    if (stream->hasError()) {
        return serializeErrorMsg(request_key, stream->statusInfo());
    }

    // If prefill first response indicated termination (e.g., max_new_tokens=1 or first token is EOS)
    // but the client has not yet seen a terminal frame, we must send one so it knows the request is complete.
    if (prefill_finished && !client_finished_sent && first_token_sent && stream->isFinished()) {
        GenerateOutputsPB term_output;
        auto*             flatten = term_output.mutable_flatten_output();
        // Set finished=true for all sequences using the saved batch/beam count from
        // the preserved prefill first response, even if later decode chunks arrived first.
        for (int i = 0; i < prefill_finished_size; ++i) {
            flatten->add_finished(true);
            flatten->add_aux_info();
        }
        updateDecodeAuxInfo(term_output, stream, prefill_ctx);
        if (!writer->Write(term_output)) {
            stream->reportError(ErrorCode::CANCELLED, "write termination frame failed");
            RTP_LLM_LOG_WARNING("request [%s] write termination frame failed", request_key.c_str());
            return grpc::Status(grpc::StatusCode::INTERNAL, "write termination frame failed");
        }
        client_finished_sent = true;
        RTP_LLM_LOG_DEBUG("request [%s] sent termination frame for first-token-EOS", request_key.c_str());
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
    int64_t         handoff_id        = 0;

    // Check if pd separation should be used
    auto pd_separation = decodeEntranceRequiresPrefill(*request);
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

    std::shared_ptr<PrefillServerCallerContext> prefill_caller_ctx;
    PrefillContextGuard                         prefill_context_guard;
    const auto&                                 unique_key  = input->generate_config->unique_key;
    auto                                        deadline_us = normalized_timeout_ms * 1000;
    GenerateInputPB                             prefill_request;
    prefill_request.CopyFrom(*effective_request);
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
            prefill_server_caller_->callPrefill(&prefill_request, target_ip, target_port, unique_key, deadline_us);
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

    generate_context.error_status = pollStreamOutputWithPrefill(server_context,
                                                                generate_context.request_key,
                                                                response_writer,
                                                                generate_context.getStream(),
                                                                prefill_caller_ctx);
    meta_->dequeue(generate_context.request_id, generate_context.getStream());

    if (prefill_caller_ctx && prefill_caller_ctx->failed()) {
        prefill_server_caller_->invalidatePrefillPeerInfo(prefill_ip, prefill_port);
    }
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
