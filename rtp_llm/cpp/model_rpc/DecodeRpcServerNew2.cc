#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/engine_base/Host.h"
#include "autil/NetUtil.h"
#include <cstring>

namespace rtp_llm {

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

    prefill_server_caller_ = std::make_shared<PrefillServerCaller>(process_id_);

    RTP_LLM_LOG_INFO("decode rpc server new2 init");
    return grpc::Status::OK;
}

grpc::Status DecodeRpcServerNew2::GenerateStreamCall(grpc::ServerContext*                   server_context,
                                                     const GenerateInputPB*                 request,
                                                     grpc::ServerWriter<GenerateOutputsPB>* response_writer) {
    // Check if pd separation should be used
    auto pd_separation = request->generate_config().max_new_tokens() > 1 && request->generate_config().num_beams() <= 1
                         && request->generate_config().variable_num_beams().size() == 0
                         && request->generate_config().num_return_sequences() <= 1
                         && request->generate_config().can_use_pd_separation();
    if (!pd_separation) {
        RTP_LLM_LOG_DEBUG("pd separation is disabled, call prefill server");
        GenerateInputPB prefill_forward_request;
        prefill_forward_request.CopyFrom(*request);
        prefill_forward_request.mutable_generate_config()->set_unique_key("");
        return prefill_server_caller_->callPrefill(server_context, &prefill_forward_request, response_writer);
    }

    const bool need_prefill = request->generate_config().unique_key().empty();

    AtomicGuard request_guard(onflight_requests_);
    auto        request_id = request->request_id();
    RTP_LLM_LOG_DEBUG("receive request %ld", request_id);
    auto generate_context =
        GenerateContext(request_id, request->generate_config().timeout_ms(), server_context, metrics_reporter_, meta_);
    auto input = QueryConverter::transQuery(request);
    if (need_prefill) {
        input->generate_config->unique_key = autil::NetUtil::getBindIp() + "_"
                                             + std::to_string(unique_key_id_.fetch_add(1)) + "_"
                                             + std::to_string(currentTimeUs());
    }

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

    // 获取 prefill 地址并查询 prefill_tp_size（有缓存，首次后几乎无开销）
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

    int prefill_tp_size = prefill_server_caller_->getPrefillTpSize(
        prefill_ip, prefill_port, static_cast<int32_t>(request->generate_config().timeout_ms()));
    if (prefill_tp_size <= 0) {
        RTP_LLM_LOG_WARNING("decode rpc server new2 generate failed: prefill_tp_size unavailable, "
                            "request_id=%ld, prefill_addr=%s:%u",
                            request_id,
                            prefill_ip.c_str(),
                            prefill_port);
        return grpc::Status(grpc::StatusCode::INTERNAL, "prefill_tp_size is not available");
    }
    stream->setPrefillTpSize(prefill_tp_size);

    // 由 DecodeRpcServerNew2 负责发起异步 prefill 调用；保留局部 context 防止提前销毁导致 prefill 侧 stream 被 cancel
    std::shared_ptr<PrefillServerCallerContext> prefill_caller_ctx;
    if (need_prefill) {
        const auto& unique_key  = input->generate_config->unique_key;
        auto        deadline_us = static_cast<int64_t>(request->generate_config().timeout_ms()) * 1000;
        prefill_caller_ctx =
            prefill_server_caller_->callPrefill(request, prefill_ip, prefill_port, unique_key, deadline_us);
        if (!prefill_caller_ctx) {
            generate_context.error_info = ErrorInfo(ErrorCode::P2P_CONNECTOR_CALL_PREFILL_FAILED,
                                                    "failed to start async prefill request to " + prefill_ip + ":"
                                                        + std::to_string(prefill_port));
            generate_context.error_status =
                serializeErrorMsg(generate_context.request_key, generate_context.error_info);
            return generate_context.error_status;
        }
    }

    engine_->enqueue(stream);

    generate_context.error_status =
        pollStreamOutput(server_context, generate_context.request_key, response_writer, generate_context.getStream());
    meta_->dequeue(generate_context.request_id, generate_context.getStream());
    return generate_context.error_status;
}

void DecodeRpcServerNew2::updateAuxInfo(GenerateOutputsPB& outputs_pb, std::shared_ptr<GenerateStream>& stream) {
    auto first_token_rt_us = stream->getTimeInfo().first_token_rt_us;
    auto cost_time_us      = autil::TimeUtility::currentTimeInMicroSeconds() - stream->beginTimeUs();

    for (size_t i = 0; i < outputs_pb.flatten_output().aux_info_size(); i++) {
        auto aux_info = outputs_pb.mutable_flatten_output()->mutable_aux_info(i);
        aux_info->set_first_token_cost_time_us(first_token_rt_us);
        aux_info->set_cost_time_us(cost_time_us);
        aux_info->set_pd_sep(true);

        // use prefill as
        aux_info->set_total_reuse_len(stream->prefillTotalReuseLen());
        aux_info->set_local_reuse_len(stream->prefillLocalReuseLen());
        aux_info->set_remote_reuse_len(stream->prefillRemoteReuseLen());
        aux_info->set_memory_reuse_len(static_cast<int32_t>(stream->prefillMemoryReuseLen()));

        aux_info->set_prefill_total_reuse_len(stream->prefillTotalReuseLen());
        aux_info->set_prefill_local_reuse_len(stream->prefillLocalReuseLen());
        aux_info->set_prefill_remote_reuse_len(stream->prefillRemoteReuseLen());
        aux_info->set_prefill_memory_reuse_len(static_cast<int32_t>(stream->prefillMemoryReuseLen()));

        aux_info->set_decode_total_reuse_len(stream->reuseLength());
        aux_info->set_decode_local_reuse_len(stream->localReuseLength());
        aux_info->set_decode_remote_reuse_len(stream->remoteReuseLength());
        aux_info->set_decode_memory_reuse_len(stream->memoryReuseLength());
    }
}

}  // namespace rtp_llm
