#include "rtp_llm/cpp/model_rpc/PrefillGenerateContextNew.h"
#include "rtp_llm/cpp/model_rpc/RemoteServerResource.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"

namespace rtp_llm {

ErrorInfo PrefillGenerateContextNew::init(const std::shared_ptr<EngineBase>& engine) {
    RTP_LLM_LOG_DEBUG("request [%s] start to prepare generate context", request_key.c_str());

    generate_input                                        = QueryConverter::transQuery(&request->input());
    generate_input->generate_config->pd_separation        = true;
    generate_input->generate_config->force_disable_sp_run = true;

    // TODO: support MTP
    // if (engine->isMTPEagle()) {
    //     generate_input->generate_config->force_disable_sp_run = false;
    // } else {
    //     generate_input->generate_config->force_disable_sp_run = true;
    // }

    stream_            = engine->makeStream(generate_input);
    request_timeout_ms = stream_->getTimeoutMs();

    auto status = stream_->initKVBlock();
    if (!status.ok()) {
        RTP_LLM_LOG_WARNING("request [%s] init kv block failed, malloc kv cache block failed", request_key.c_str());
        error_info = ErrorInfo(ErrorCode::MALLOC_FAILED, "malloc kv cache block failed at prefill node");
        return error_info;
    }

    RTP_LLM_LOG_DEBUG("request [%s] prepare generate context done", request_key.c_str());
    return ErrorInfo::OkStatus();
}

void PrefillGenerateContextNew::stopStream() {
    if (stream_ != nullptr && stream_->running()) {
        stream_->setStop(error_info.code(), error_info.ToString());
    }

    // if (stream_) {
    //     // TODO: 如何安全的停止prefill generate
    //     stream_->cancelIfNotRunning();
    //     while (stream_->running()) {
    //         RTP_LLM_LOG_INFO("waiting prefill stream [%d] running done to cancel",
    //         stream_->generateInput()->request_id); usleep(1000);
    //     }
    // }
}

void PrefillGenerateContextNew::notifyRequestEndForAllRank() {
    for (int i = 0; i < resource->grpc_workers.size(); ++i) {
        notifyRequestEnd(i);
    }
}

void PrefillGenerateContextNew::notifyRequestEnd(int index) {
    auto& worker         = resource->grpc_workers[index];
    auto  connect_status = resource->rpc_pool.getConnection(worker);
    if (!connect_status.ok()) {
        RTP_LLM_LOG_WARNING(
            "request [%s] get grpc connection for rank:%d, addr:%s failed", request_key.c_str(), index, worker.c_str());
    }

    RemoteFinishRequestPB finish_request;
    finish_request.set_request_id(request_id);

    auto                stub = connect_status.value().stub.get();
    grpc::ClientContext client_context;
    EmptyPB             response;
    auto                status = stub->RemoteFinishNew(&client_context, finish_request, &response);

    if (!status.ok()) {
        std::string error_msg = "remote finish for rank:" + std::to_string(index) + " failed";
        RTP_LLM_LOG_ERROR("request [%s] %s", request_key.c_str(), error_msg.c_str());
    }
}

void PrefillGenerateContextNew::reportTime() {
    RpcMetricsCollector collector;
    collectBasicMetrics(collector);
    collector.notify_store_cache_rt_us   = notify_store_cache_done_time_us - request_begin_time_us;
    collector.generate_first_token_rt_us = generate_first_token_done_time_us - notify_store_cache_done_time_us;
    collector.wait_store_cache_rt_us     = currentTimeUs() - generate_first_token_done_time_us;
    collector.min_response_done_time_us  = min_response_done_time_us;
    collector.max_response_done_time_us  = max_response_done_time_us;
    reportMetrics(collector);
}

}  // namespace rtp_llm