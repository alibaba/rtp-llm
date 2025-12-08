#include "rtp_llm/cpp/model_rpc/PrefillRpcServerNew2.h"

namespace rtp_llm {

grpc::Status PrefillRpcServerNew2::init(const EngineInitParams&                                maga_init_params,
                                        py::object                                             mm_process_engine,
                                        std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    RTP_LLM_LOG_INFO("prefill rpc server new2 init start");
    auto ret = RemoteRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params));
    if (!ret.ok()) {
        RTP_LLM_LOG_ERROR("prefill rpc server new2 init failed, err: %s", ret.error_message().c_str());
        return ret;
    }

    RTP_LLM_LOG_INFO("prefill rpc server new2 init success");
    // init p2p connector prefill
    p2p_connector_prefill_ =
        std::make_shared<P2PConnectorPrefill>(maga_init_params.gpt_init_parameter,
                                              engine_->getDevice(),
                                              engine_->resourceContext().cache_manager->getAllocator());
    if (!p2p_connector_prefill_->init()) {
        RTP_LLM_LOG_ERROR("prefill rpc server new2 init failed, p2p_connector_prefill init failed");
        return grpc::Status(grpc::StatusCode::INTERNAL, "p2p_connector_prefill init failed");
    }

    auto callback = p2p_connector_prefill_->makeCallback();
    if (!callback) {
        RTP_LLM_LOG_ERROR("prefill rpc server new2 init failed, make callback failed");
        return grpc::Status(grpc::StatusCode::INTERNAL, "make callback failed");
    }
    tp_broadcast_service_->registerCallback(callback);
    engine_->getDevice()->setKVCacheConnector(p2p_connector_prefill_);

    RTP_LLM_LOG_INFO("prefill rpc server new2 init p2p_connector_prefill success");
    return grpc::Status::OK;
}

grpc::Status PrefillRpcServerNew2::GenerateStreamCall(grpc::ServerContext*                   server_context,
                                                      const GenerateInputPB*                 request,
                                                      grpc::ServerWriter<GenerateOutputsPB>* response_writer) {
    auto stream = initStream(request);
    if (!stream) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "init stream failed");
    }

    if (!stream->getPdSeparationUniqueKey().empty()) {
        p2p_connector_prefill_->addStream(stream->getPdSeparationUniqueKey(), stream);
    }

    return pollStreamOutput(server_context, std::to_string(request->request_id()), response_writer, stream, true);
}

std::shared_ptr<GenerateStream> PrefillRpcServerNew2::initStream(const GenerateInputPB* request) {
    AtomicGuard request_guard(onflight_requests_);
    auto        request_id = request->request_id();
    RTP_LLM_LOG_DEBUG("receive request %ld", request_id);
    auto input                            = QueryConverter::transQuery(request);
    input->generate_config->pd_separation = true;

    // need to check client has buffer at first
    if (mm_processor_ != nullptr && input->multimodal_inputs) {
        auto mm_res = mm_processor_->updateMultimodalFeatures(input);
        if (!mm_res.ok()) {
            RTP_LLM_LOG_WARNING("request [%ld] update multimodal features failed", request_id);
            return nullptr;
        }
    }

    input->lora_id  = engine_->getLoraManager()->getLoraId(input->generate_config->adapter_name);
    auto lora_guard = lora::LoraResourceGuard(engine_->getLoraManager(), input->generate_config->adapter_name);
    RTP_LLM_LOG_DEBUG("request [%ld] trans to stream success", request_id);
    return engine_->enqueue(input);
}

grpc::Status PrefillRpcServerNew2::StartLoad(grpc::ServerContext*                  context,
                                             const P2PConnectorStartLoadRequestPB* request,
                                             P2PConnectorStartLoadResponsePB*      response) {
    if (request->unique_key().empty()) {
        RTP_LLM_LOG_WARNING("StartLoad request pd_separation_unique_key is empty");
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "StartLoad request pd_separation_unique_key is empty");
    }
    RTP_LLM_LOG_INFO("StartLoad request unique_key: %s", request->unique_key().c_str());

    std::vector<std::pair<std::string, uint32_t>> decode_transfer_servers;
    for (const auto& worker : request->workers()) {
        decode_transfer_servers.push_back(std::make_pair(worker.ip(), worker.cache_store_port()));
    }

    if (decode_transfer_servers.empty()) {
        RTP_LLM_LOG_WARNING("StartLoad request decode_transfer_servers is empty");
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "decode_transfer_servers is empty");
    }
    auto status =
        p2p_connector_prefill_->handleWrite(request->unique_key(), decode_transfer_servers, request->deadline_ms());
    if (!status.ok()) {
        RTP_LLM_LOG_WARNING("StartLoad request failed, status: %s", status.error_message().c_str());
        return status;
    }
    response->set_success(true);
    RTP_LLM_LOG_INFO("StartLoad request success");
    return grpc::Status::OK;
}

}  // namespace rtp_llm