#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
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
    auto connector_coordinator = kvcache_manager->connectorCoordinator();
    if (!connector_coordinator) {
        RTP_LLM_LOG_WARNING("decode rpc server new2 init failed, connector coordinator is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "connector coordinator is null");
    }

    prefill_server_caller_ =
        std::make_shared<PrefillServerCaller>(process_id_, 10 /*maga_init_params.decode_polling_call_prefill_ms*/);

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

        RTP_LLM_LOG_INFO("pd separation is disabled, call prefill server");
        auto mutable_request = const_cast<GenerateInputPB*>(request);
        mutable_request->mutable_generate_config()->set_unique_key("");  // set no  unique key for prefill
        return prefill_server_caller_->callPrefill(server_context, mutable_request, response_writer);
    }

    bool need_prefill = false;
    if (request->generate_config().unique_key().empty()) {
        auto unique_key = autil::NetUtil::getBindIp() + "_" + std::to_string(unique_key_id_.fetch_add(1)) + "_"
                          + std::to_string(currentTimeUs());
        auto mutable_request = const_cast<GenerateInputPB*>(request);
        mutable_request->mutable_generate_config()->set_unique_key(unique_key);
        need_prefill = true;
    }

    AtomicGuard request_guard(onflight_requests_);
    auto        request_id = request->request_id();
    RTP_LLM_LOG_DEBUG("receive request %ld", request_id);
    auto generate_context =
        GenerateContext(request_id, request->generate_config().timeout_ms(), server_context, metrics_reporter_, meta_);
    auto input = QueryConverter::transQuery(request);
    // input->generate_config->pd_separation = true;

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

    std::shared_ptr<PrefillServerCallerContext> prefill_context;
    if (need_prefill) {
        auto [ip, port] = stream->prefillAddr();
        prefill_context =
            prefill_server_caller_->callPrefill(request, ip, port, stream->uniqueKey(), stream->deadlineMs() * 1000);
    }

    generate_context.error_status =
        pollStreamOutput(server_context, generate_context.request_key, response_writer, generate_context.getStream());
    meta_->dequeue(generate_context.request_id, generate_context.getStream());

    // likely prefill done before local generate done
    if (prefill_context) {
        prefill_context->waitPrefillDone();
    }
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
        aux_info->set_total_reuse_len(stream->reuseInfo().prefill_total_reuse_len);
        aux_info->set_local_reuse_len(stream->reuseInfo().prefill_local_reuse_len);
        aux_info->set_remote_reuse_len(stream->reuseInfo().prefill_remote_reuse_len);

        aux_info->set_prefill_total_reuse_len(stream->reuseInfo().prefill_total_reuse_len);
        aux_info->set_prefill_local_reuse_len(stream->reuseInfo().prefill_local_reuse_len);
        aux_info->set_prefill_remote_reuse_len(stream->reuseInfo().prefill_remote_reuse_len);

        aux_info->set_decode_total_reuse_len(stream->reuseInfo().reuse_length);
        aux_info->set_decode_local_reuse_len(stream->reuseInfo().local_reuse_length);
        aux_info->set_decode_remote_reuse_len(stream->reuseInfo().remote_reuse_length);
    }
}

}  // namespace rtp_llm
