#include "rtp_llm/cpp/model_rpc/PrefillServerCaller.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

PrefillServerCallerContext::PrefillServerCallerContext(const std::string& prefill_addr,
                                                       int64_t            decode_polling_call_prefill_ms,
                                                       const std::string& unique_key):
    prefill_addr(prefill_addr),
    decode_polling_call_prefill_ms_(decode_polling_call_prefill_ms),
    unique_key(unique_key),
    request_begin_time_us_(currentTimeUs()) {
    client_context   = std::make_shared<grpc::ClientContext>();
    completion_queue = std::make_shared<grpc::CompletionQueue>();
}

PrefillServerCallerContext::~PrefillServerCallerContext() {
    if (client_context) {
        client_context->TryCancel();
    }
    completion_queue->Shutdown();
}

PrefillServerCaller::PrefillServerCaller(const std::string& process_id,
                                         int64_t            decode_polling_call_prefill_ms,
                                         bool               is_mtp_eagle):
    rpc_pool_(std::make_shared<RPCPool>()),
    process_id_(process_id),
    decode_polling_call_prefill_ms_(decode_polling_call_prefill_ms),
    is_mtp_eagle_(is_mtp_eagle) {}

std::shared_ptr<PrefillServerCallerContext> PrefillServerCaller::callPrefill(const GenerateInputPB* request,
                                                                             const std::string&     ip,
                                                                             uint32_t               port,
                                                                             const std::string&     unique_key,
                                                                             int64_t                deadline_us) {
    auto connect_status = rpc_pool_->getConnection(ip + ":" + std::to_string(port));
    if (!connect_status.ok()) {
        RTP_LLM_LOG_WARNING("request [%lld] get grpc connection failed", request->request_id());
        return nullptr;
    }

    auto stub = connect_status.value().stub;

    auto context = std::make_shared<PrefillServerCallerContext>(
        ip + ":" + std::to_string(port), decode_polling_call_prefill_ms_, unique_key);
    context->stub = stub;

    // treat prefill server as normal server
    context->request.CopyFrom(*request);
    // should not set this, or we can not check if prefill is finished
    // context->request.mutable_generate_config()->set_max_new_tokens(1);
    context->request.set_client_id(process_id_);
    context->request.set_start_time(currentTimeUs());
    context->request.mutable_generate_config()->set_can_use_pd_separation(true);
    context->request.mutable_generate_config()->set_force_disable_sp_run(!is_mtp_eagle_);
    context->request.mutable_generate_config()->set_pd_sepration_unique_key(unique_key);

    context->client_context->set_deadline(std::chrono::system_clock::now() + std::chrono::microseconds(deadline_us));

    std::unique_ptr<grpc::ClientAsyncReader<GenerateOutputsPB>> reader(context->stub->AsyncGenerateStreamCall(
        context->client_context.get(), context->request, context->completion_queue.get(), reinterpret_cast<void*>(0)));

    context->reader = std::move(reader);

    return context;
}

grpc::Status PrefillServerCallerContext::waitPrefillDone(const std::shared_ptr<GenerateStream>& stream,
                                                         grpc::ServerContext*                   server_context,
                                                         grpc::ServerWriter<GenerateOutputsPB>* response_writer) {
    // 启动读取第一个响应
    reader->Read(&response, reinterpret_cast<void*>(1));

    bool response_received = false;
    while (!finished) {
        void* got_tag = nullptr;
        bool  ok      = false;
        auto  once_deadline =
            std::chrono::system_clock::now() + std::chrono::milliseconds(decode_polling_call_prefill_ms_);
        auto next_status = completion_queue->AsyncNext(&got_tag, &ok, once_deadline);
        if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
            continue;
        }
        if (!ok) {
            RTP_LLM_LOG_WARNING("request [%lld] async get next event from grpc completion queue failed, status %d(%s)",
                                request.request_id(),
                                status.error_code(),
                                status.error_message().c_str());
            return grpc::Status(grpc::StatusCode::INTERNAL, "async get next event from grpc completion queue failed");
        }

        // 处理 Read 事件
        if (got_tag == reinterpret_cast<void*>(1)) {
            // Read 完成，收到第一个响应
            response_received = true;
            // 启动 Finish 来获取最终状态
            reader->Finish(&status, reinterpret_cast<void*>(2));
        } else if (got_tag == reinterpret_cast<void*>(2)) {
            // Finish 完成
            finished           = true;
            auto error_code    = status.error_code();
            auto error_message = status.error_message();
            if (!status.ok()) {
                RTP_LLM_LOG_WARNING("request [%lld] prefill rpc failed, status %d(%s)",
                                    request.request_id(),
                                    error_code,
                                    error_message.c_str());
                return grpc::Status(grpc::StatusCode::INTERNAL, "prefill rpc failed");
            }
            break;
        }
    }

    if (!response_received) {
        RTP_LLM_LOG_WARNING("request [%lld] prefill rpc did not receive response", request.request_id());
        return grpc::Status(grpc::StatusCode::INTERNAL, "prefill rpc did not receive response");
    }

    return responseFirstToken(stream, server_context, response_writer);
}

grpc::Status PrefillServerCallerContext::responseFirstToken(const std::shared_ptr<GenerateStream>& stream,
                                                            grpc::ServerContext*                   server_context,
                                                            grpc::ServerWriter<GenerateOutputsPB>* response_writer) {
    if (server_context->IsCancelled()) {
        RTP_LLM_LOG_WARNING("request [%lld] is cancelled", request.request_id());
        return grpc::Status(grpc::StatusCode::CANCELLED, "request is cancelled");
    }

    auto decode_total_reuse_len  = stream->initialReuseLength();
    auto decode_local_reuse_len  = stream->localReuseLength();
    auto decode_remote_reuse_len = stream->remoteReuseLength();

    auto    first_token_rt_us = response.compute_done_time() - response.begin_compute_time();
    int64_t cost_time_us      = currentTimeUs() - request_begin_time_us_;

    for (size_t i = 0; i < response.generate_outputs_size(); i++) {
        auto generate_output = response.mutable_generate_outputs(i);
        auto aux_info        = generate_output->mutable_aux_info();
        aux_info->set_first_token_cost_time_us(first_token_rt_us);
        aux_info->set_cost_time_us(cost_time_us);
        aux_info->set_prefill_total_reuse_len(aux_info->total_reuse_len());
        aux_info->set_prefill_local_reuse_len(aux_info->local_reuse_len());
        aux_info->set_prefill_remote_reuse_len(aux_info->remote_reuse_len());
        aux_info->set_decode_total_reuse_len(decode_total_reuse_len);
        aux_info->set_decode_local_reuse_len(decode_local_reuse_len);
        aux_info->set_decode_remote_reuse_len(decode_remote_reuse_len);
    }

    if (!response_writer->Write(response)) {
        RTP_LLM_LOG_WARNING("request [%lld] write outputs pb failed", request.request_id());
        return grpc::Status(grpc::StatusCode::INTERNAL, "write outputs pb failed");
    }
    return grpc::Status::OK;
}

bool PrefillServerCallerContext::isGenerateFinished() const {
    return response.generate_outputs_size() == 0 || response.generate_outputs(0).finished()
           || response.error_info().error_code() != ErrorCodePB::NONE_ERROR;
}

}  // namespace rtp_llm
