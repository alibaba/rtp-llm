#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"
#include <algorithm>
#include <c10/core/InferenceMode.h>
#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.h"
#include "rtp_llm/cpp/model_rpc/RpcTimeoutUtils.h"
#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
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

void updatePrefillRoleAddr(GenerateInput& input, GenerateInputPB& request, const std::string& ip, uint32_t port) {
    for (auto& addr : input.generate_config->role_addrs) {
        if (addr.role == RoleType::PREFILL) {
            addr.ip        = ip;
            addr.grpc_port = port;
        }
    }
    for (auto& addr : *request.mutable_generate_config()->mutable_role_addrs()) {
        if (addr.role() == RoleAddrPB::PREFILL) {
            addr.set_ip(ip);
            addr.set_grpc_port(port);
        }
    }
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

grpc::Status DecodeRpcServerNew2::preparePDRequest(const GenerateInputPB&          request,
                                                   int64_t                         deadline_ms,
                                                   std::shared_ptr<GenerateInput>& input,
                                                   GenerateInputPB&                prefill_request,
                                                   PrefillPeerInfo&                peer_info) {
    input                              = QueryConverter::transQuery(&request);
    input->request_deadline_ms         = deadline_ms;
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
    if (peer_info.tp_size <= 0 || peer_info.cp_size <= 0 || peer_info.dp_addrs.empty()) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "prefill peer info is not available");
    }
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
    auto pd_separation = checkPDSupport(*request).supported;
    if (pd_separation) {
        handoff_id                = unique_key_id_.fetch_add(1);
        auto decode_entrance_keys =
            buildDecodeEntranceKeys(*request, autil::NetUtil::getBindIp(), handoff_id, currentTimeUs());
        request_with_handoff_key = makeDecodeEntranceHandoffRequest(*request, decode_entrance_keys.handoff_unique_key);
        request_with_handoff_key.mutable_generate_config()->set_timeout_ms(normalized_timeout_ms_i32);
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
    std::shared_ptr<GenerateInput> input;
    GenerateInputPB                prefill_request;
    PrefillPeerInfo                peer_info;
    const auto                     prepare_start_us = currentTimeUs();
    auto                           prepare_status = preparePDRequest(
        *effective_request, request_entry_ms + normalized_timeout_ms, input, prefill_request, peer_info);
    {
        RpcMetricsCollector collector;
        collector.min_response_done_time_us      = 0;
        collector.prepare_generate_context_rt_us = currentTimeUs() - prepare_start_us;
        generate_context.reportMetrics(collector);
    }
    if (!prepare_status.ok()) {
        generate_context.error_status = prepare_status;
        return prepare_status;
    }
    auto stream = engine_->makeStream(input);
    stream->setPrefillTpSize(peer_info.tp_size);
    stream->setPrefillCpSize(peer_info.cp_size);
    generate_context.setStream(stream);

    std::shared_ptr<PrefillServerCallerContext> prefill_caller_ctx;
    PrefillContextGuard                         prefill_context_guard;
    const auto&                                 unique_key  = input->generate_config->unique_key;
    const auto request_deadline_ms = input->request_deadline_ms;
    const auto& dp_addrs       = peer_info.dp_addrs;
    const auto  first_dp_index = selectDecodeEntranceDpIndex(dp_addrs.size(), handoff_id);
    std::string selected_addr;
    FirstError                                  startup_error;
    for (size_t attempt = 0; attempt < dp_addrs.size(); ++attempt) {
        const auto dp_index = (first_dp_index + attempt) % dp_addrs.size();
        selected_addr       = dp_addrs[dp_index];

        std::string target_ip;
        uint32_t    target_port = 0;
        auto        parse_status = parsePrefillDpAddr(selected_addr, &target_ip, &target_port);
        if (!parse_status.ok()) {
            startup_error.record(errorInfoFromGrpcStatus(parse_status));
            return grpcStatusFromErrorInfo(startup_error.snapshot().error);
        }
        updatePrefillRoleAddr(*input, prefill_request, target_ip, target_port);

        const auto prefill_call_start_us = currentTimeUs();
        auto started = prefill_server_caller_->callPrefill(
            &prefill_request, target_ip, target_port, unique_key, request_deadline_ms);
        {
            RpcMetricsCollector collector;
            collector.min_response_done_time_us = 0;
            collector.remote_generate_rt_us     = currentTimeUs() - prefill_call_start_us;
            generate_context.reportMetrics(collector);
        }
        if (started.ok())
            prefill_caller_ctx = std::move(started.value());
        else
            startup_error.record(started.status());
        if (prefill_caller_ctx) {
            if (attempt > 0) {
                RTP_LLM_LOG_WARNING("request [%ld] recovered async prefill by trying next DP, addr=%s",
                                    request_id,
                                    selected_addr.c_str());
            }
            break;
        }

        RTP_LLM_LOG_WARNING("request [%ld] async prefill start failed for DP %s, attempt %zu/%zu",
                            request_id,
                            selected_addr.c_str(),
                            attempt + 1,
                            dp_addrs.size());
    }
    if (!prefill_caller_ctx) {
        generate_context.error_info   = startup_error.snapshot().error;
        generate_context.error_status = serializeErrorMsg(generate_context.request_key, generate_context.error_info);
        return generate_context.error_status;
    }
    prefill_context_guard.context = prefill_caller_ctx;

    engine_->enqueue(stream);

    // Keep the async prefill RPC alive for the handoff lifecycle only. StartLoad applies the first token to the
    // decode stream, so client output must come exclusively from pollStreamOutput().
    generate_context.error_status = pollStreamOutput(
        server_context, generate_context.request_key, response_writer, generate_context.getStream(), [&](bool& done) {
            done = prefill_caller_ctx->done();
            return prefill_caller_ctx->firstError();
        });
    meta_->dequeue(generate_context.request_id, generate_context.getStream());

    if (prefill_caller_ctx && (!generate_context.error_status.ok() || server_context->IsCancelled())) {
        prefill_caller_ctx->cancel();
    }
    return generate_context.error_status;
}

grpc::Status DecodeRpcServerNew2::BatchGenerateCall(grpc::ServerContext*        context,
                                                    const BatchGenerateInputPB* request,
                                                    BatchGenerateOutputsPB*     response) {
    c10::InferenceMode inference_guard(true);
    response->Clear();
    if (request->inputs_size() == 0)
        return grpc::Status::OK;
    const int64_t entry_ms = currentTimeMs();
    bool          pd       = false;
    auto          support  = checkPDBatchSupport(*request, pd);
    if (!support.ok())
        return serializeErrorMsg("batch", support);
    AtomicGuard          request_guard(onflight_requests_);
    BatchGenerateInputPB forwarded;
    forwarded.CopyFrom(*request);
    const auto address = prefillAddress(request->inputs(0));
    if (address.empty())
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "batch requires a prefill address");
    int64_t rpc_deadline_ms = entry_ms;
    // Fill only missing roles. Explicit routing conflicts must never be silently overwritten.
    for (auto& item : *forwarded.mutable_inputs()) {
        auto* config = item.mutable_generate_config();
        for (const auto& first : request->inputs(0).generate_config().role_addrs()) {
            bool found = false;
            for (const auto& addr : config->role_addrs())
                found |= addr.role() == first.role();
            if (!found)
                config->add_role_addrs()->CopyFrom(first);
        }
        if (prefillAddress(item) != address) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "batch routed to conflicting prefill backends");
        }
        const auto timeout =
            normalizeRpcTimeoutMs(config->timeout_ms(), maga_init_params_.pd_sep_config.max_rpc_timeout_ms);
        config->set_timeout_ms(clampRpcTimeoutMsToInt32(timeout));
        rpc_deadline_ms = std::max(rpc_deadline_ms, entry_ms + timeout);
    }
    std::vector<std::unique_ptr<GenerateContext>> contexts;
    std::vector<GenerateStreamPtr>                streams;
    std::vector<std::shared_ptr<GenerateInput>>   inputs;
    PrefillPeerInfo                               batch_peer;
    const int64_t                                 batch_dp_id = pd ? batch_dp_id_.fetch_add(1) : 0;
    if (pd) {
        for (int i = 0; i < forwarded.inputs_size(); ++i) {
            if (context->IsCancelled())
                return grpc::Status(grpc::StatusCode::CANCELLED, "batch cancelled by user");
            auto&      item       = *forwarded.mutable_inputs(i);
            const auto handoff_id = unique_key_id_.fetch_add(1);
            const auto keys = buildDecodeEntranceKeys(item, autil::NetUtil::getBindIp(), handoff_id, currentTimeUs());
            auto       effective = makeDecodeEntranceHandoffRequest(item, keys.handoff_unique_key);
            std::shared_ptr<GenerateInput> input;
            PrefillPeerInfo                peer;
            auto status = preparePDRequest(effective, entry_ms + item.generate_config().timeout_ms(), input, item, peer);
            if (!status.ok())
                return grpc::Status(status.error_code(),
                                    "batch item " + std::to_string(i) + ": " + status.error_message(),
                                    status.error_details());
            if (i == 0)
                batch_peer = peer;
            if (peer.tp_size != batch_peer.tp_size || peer.cp_size != batch_peer.cp_size
                || peer.dp_addrs != batch_peer.dp_addrs) {
                return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                    "prefill topology changed while preparing batch");
            }
            inputs.push_back(std::move(input));
        }
        // All preprocessing succeeds before creating or scheduling any stream.
        for (const auto& input : inputs) {
            auto stream = engine_->makeStream(input);
            stream->setPrefillTpSize(batch_peer.tp_size);
            stream->setPrefillCpSize(batch_peer.cp_size);
            auto item_context = std::make_unique<GenerateContext>(
                input->request_id, input->generate_config->timeout_ms, context, metrics_reporter_, meta_);
            item_context->setStream(stream);
            contexts.push_back(std::move(item_context));
            streams.push_back(std::move(stream));
        }
    }
    std::unique_ptr<PrefillBatchCallerContext> caller;
    FirstError                                 startup_error;
    const auto&                                addresses = pd ? batch_peer.dp_addrs : std::vector<std::string>{address};
    const auto                                 first_dp  = selectDecodeEntranceDpIndex(addresses.size(), batch_dp_id);
    for (size_t attempt = 0; attempt < addresses.size(); ++attempt) {
        const auto& target = addresses[(first_dp + attempt) % addresses.size()];
        if (pd) {
            std::string ip;
            uint32_t    port   = 0;
            auto        status = parsePrefillDpAddr(target, &ip, &port);
            if (!status.ok()) {
                startup_error.record(errorInfoFromGrpcStatus(status));
                return grpcStatusFromErrorInfo(startup_error.snapshot().error);
            }
            for (size_t i = 0; i < inputs.size(); ++i) {
                updatePrefillRoleAddr(*inputs[i], *forwarded.mutable_inputs(i), ip, port);
            }
        }
        for (const auto& item : forwarded.inputs()) {
            if (entry_ms + item.generate_config().timeout_ms() <= currentTimeMs()) {
                startup_error.record(ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "batch preparation deadline exceeded"));
                return grpcStatusFromErrorInfo(startup_error.snapshot().error);
            }
        }
        if (context->IsCancelled()) {
            startup_error.record(ErrorInfo(ErrorCode::CANCELLED, "batch cancelled by user"));
            return grpcStatusFromErrorInfo(startup_error.snapshot().error);
        }
        auto started = prefill_server_caller_->callPrefillBatch(forwarded, target, rpc_deadline_ms);
        if (started.ok())
            caller = std::move(started.value());
        else
            startup_error.record(started.status());
        if (caller)
            break;
    }
    if (!caller)
        return grpcStatusFromErrorInfo(startup_error.snapshot().error);
    // Do not wait for the unary Prefill response: Decode must start loading KV first.
    if (pd && engine_->batchEnqueue(streams) != streams) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "batchEnqueue changed prepared stream identity or order");
    }
    auto status = pollBatchStreamOutput(context, streams, response, [&](bool& done) {
        done = caller->done();
        return caller->firstError();
    });
    for (auto& item : contexts)
        item->error_status = status;
    if (!status.ok()) {
        caller->cancel();
    } else if (!pd) {
        response->CopyFrom(caller->response());
    }
    return status;
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
