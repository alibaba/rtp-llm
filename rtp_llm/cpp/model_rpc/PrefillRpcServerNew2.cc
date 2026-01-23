#include "rtp_llm/cpp/model_rpc/PrefillRpcServerNew2.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorCoordinator.h"

namespace rtp_llm {

grpc::Status PrefillRpcServerNew2::init(const EngineInitParams&                                maga_init_params,
                                        py::object                                             mm_process_engine,
                                        std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    auto ret = RemoteRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params), false);
    if (!ret.ok()) {
        RTP_LLM_LOG_ERROR("prefill rpc server new2 init failed, err: %s", ret.error_message().c_str());
        return ret;
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
    auto device = engine_->getDevice();
    if (!device) {
        RTP_LLM_LOG_WARNING("prefill rpc server new2 init failed, device is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "device is null");
    }
    device->setConnectorCoordinator(connector_coordinator);
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServerNew2::GenerateStreamCall(grpc::ServerContext*                   server_context,
                                                      const GenerateInputPB*                 request,
                                                      grpc::ServerWriter<GenerateOutputsPB>* response_writer) {
    auto pd_separation =
        request->generate_config().num_beams() <= 1 && request->generate_config().variable_num_beams().size() == 0
        && request->generate_config().num_return_sequences() <= 1 && request->generate_config().can_use_pd_separation()
        && !request->generate_config().unique_key().empty();
    if (!pd_separation) {
        RTP_LLM_LOG_INFO("pd separation is disabled, call local rpc server");
        return LocalRpcServer::GenerateStreamCall(server_context, request, response_writer);
    }

    // same as local rpc server generate stream call
    AtomicGuard request_guard(onflight_requests_);
    auto        request_id = request->request_id();
    RTP_LLM_LOG_DEBUG("receive request %ld", request_id);
    auto generate_context =
        GenerateContext(request_id, request->generate_config().timeout_ms(), server_context, metrics_reporter_, meta_);
    auto input = QueryConverter::transQuery(request);
    // TODO: diff is this
    input->generate_config->pd_separation = true;
    if (engine_->isMTPEagle()) {
        input->generate_config->force_disable_sp_run = false;
    } else {
        input->generate_config->force_disable_sp_run = true;
    }

    // need to check client has buffer at first
    if (mm_processor_ != nullptr && input->multimodal_inputs) {
        auto mm_res = mm_processor_->updateMultimodalFeatures(input);
        if (!mm_res.ok()) {
            generate_context.error_status = serializeErrorMsg(generate_context.request_key, mm_res);
        }
    }
    CHECK_ERROR_STATUS(generate_context);

    input->lora_id  = engine_->getLoraManager()->getLoraId(input->generate_config->adapter_name);
    auto lora_guard = lora::LoraResourceGuard(engine_->getLoraManager(), input->generate_config->adapter_name);
    RTP_LLM_LOG_DEBUG("request [%ld] trans to stream success", request_id);
    auto stream = engine_->enqueue(input);
    generate_context.setStream(stream);

    RTP_LLM_LOG_DEBUG("request [%ld] enqueue success", request_id);
    // same as local rpc server generate stream call
    generate_context.error_status =
        pollStreamOutput(server_context, generate_context.request_key, response_writer, generate_context.getStream());
    meta_->dequeue(generate_context.request_id, generate_context.getStream());
    return generate_context.error_status;
}

::grpc::Status PrefillRpcServerNew2::StartLoad(::grpc::ServerContext*                context,
                                               const P2PConnectorStartLoadRequestPB* request,
                                               P2PConnectorStartLoadResponsePB*      response) {
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
    if (!cache_manager->handleRead(*request, *response)) {
        RTP_LLM_LOG_WARNING("start load failed, request: [%s]", request->DebugString().c_str());
        const std::string error_msg = "start load failed, request: [" + request->DebugString() + "]";
        return grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
    }
    return grpc::Status::OK;
}

}  // namespace rtp_llm