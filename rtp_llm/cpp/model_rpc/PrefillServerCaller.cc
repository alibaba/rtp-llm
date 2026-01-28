#include "rtp_llm/cpp/model_rpc/PrefillServerCaller.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

PrefillServerCallerContext::PrefillServerCallerContext(const std::string& prefill_addr,
                                                       int64_t            decode_polling_call_prefill_ms,
                                                       const std::string& unique_key):
    prefill_addr(prefill_addr),
    unique_key(unique_key),
    request_begin_time_us_(currentTimeUs()),
    decode_polling_call_prefill_ms_(decode_polling_call_prefill_ms) {
    client_context   = std::make_shared<grpc::ClientContext>();
    completion_queue = std::make_shared<grpc::CompletionQueue>();
}

PrefillServerCallerContext::~PrefillServerCallerContext() {
    if (client_context) {
        client_context->TryCancel();
    }
    completion_queue->Shutdown();
}

PrefillServerCaller::PrefillServerCaller(const std::string& process_id, int64_t decode_polling_call_prefill_ms):
    rpc_pool_(std::make_shared<RPCPool>()),
    process_id_(process_id),
    decode_polling_call_prefill_ms_(decode_polling_call_prefill_ms) {}

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
    context->request.mutable_generate_config()->set_max_new_tokens(1);
    context->request.set_client_id(process_id_);
    context->request.set_start_time(currentTimeUs());
    context->request.mutable_generate_config()->set_can_use_pd_separation(true);
    context->request.mutable_generate_config()->set_unique_key(unique_key);

    context->client_context->set_deadline(std::chrono::system_clock::now() + std::chrono::microseconds(deadline_us));

    std::unique_ptr<grpc::ClientAsyncReader<GenerateOutputsPB>> reader(context->stub->AsyncGenerateStreamCall(
        context->client_context.get(), context->request, context->completion_queue.get(), reinterpret_cast<void*>(0)));

    context->reader = std::move(reader);

    // 启动读取第一个响应（非阻塞）
    context->reader->Read(&context->response, reinterpret_cast<void*>(1));

    return context;
}

grpc::Status PrefillServerCaller::callPrefill(grpc::ServerContext*                   server_context,
                                              const GenerateInputPB*                 request,
                                              grpc::ServerWriter<GenerateOutputsPB>* response_writer) {
    // 从 request 的 role_addrs 中获取 PREFILL 角色的地址
    std::string prefill_addr;
    for (const auto& role_addr : request->generate_config().role_addrs()) {
        if (role_addr.role() == RoleAddrPB::PREFILL) {
            prefill_addr = role_addr.ip() + ":" + std::to_string(role_addr.grpc_port());
            break;
        }
    }
    if (prefill_addr.empty()) {
        RTP_LLM_LOG_WARNING("request [%lld] no prefill server address found in role_addrs", request->request_id());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "no prefill server address found in role_addrs");
    }

    auto connect_status = rpc_pool_->getConnection(prefill_addr);
    if (!connect_status.ok()) {
        RTP_LLM_LOG_WARNING("request [%lld] get grpc connection to prefill server %s failed",
                            request->request_id(),
                            prefill_addr.c_str());
        return grpc::Status(grpc::StatusCode::INTERNAL, "get grpc connection to prefill server failed");
    }

    auto stub = connect_status.value().stub;

    // 创建客户端上下文
    grpc::ClientContext client_context;

    // 调用 PrefillServer 的 GenerateStreamCall
    auto reader = stub->GenerateStreamCall(&client_context, *request);
    if (!reader) {
        RTP_LLM_LOG_WARNING("request [%lld] create stream reader to prefill server %s failed",
                            request->request_id(),
                            prefill_addr.c_str());
        return grpc::Status(grpc::StatusCode::INTERNAL, "create stream reader to prefill server failed");
    }

    // 读取 PrefillServer 的响应并转发给客户端
    GenerateOutputsPB response;
    while (reader->Read(&response)) {
        if (!response_writer->Write(response)) {
            RTP_LLM_LOG_WARNING("request [%lld] write response to client failed", request->request_id());
            client_context.TryCancel();
            return grpc::Status(grpc::StatusCode::INTERNAL, "write response to client failed");
        }
    }

    // 获取最终状态
    auto status = reader->Finish();
    if (!status.ok()) {
        RTP_LLM_LOG_WARNING("request [%lld] prefill server rpc failed, status %d(%s)",
                            request->request_id(),
                            status.error_code(),
                            status.error_message().c_str());
    }

    return status;
}

void PrefillServerCallerContext::checkDone() {
    if (finished) {
        return;
    }

    if (!completion_queue) {
        RTP_LLM_LOG_WARNING("PrefillServerCallerContext::checkDone: completion_queue is null");
        finished = true;
        status   = grpc::Status(grpc::StatusCode::INTERNAL, "completion_queue is null");
        return;
    }

    void* got_tag = nullptr;
    bool  ok      = false;

    // 计算超时时间（非阻塞，使用较短的超时时间）
    auto once_deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(1);

    // 等待异步操作完成
    auto next_status = completion_queue->AsyncNext(&got_tag, &ok, once_deadline);
    if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
        // not finish yet
        return;
    }

    if (!ok) {
        finished = true;
        RTP_LLM_LOG_WARNING("PrefillServerCallerContext::checkDone: async next failed, unique_key: %s",
                            unique_key.c_str());
        status = grpc::Status(grpc::StatusCode::INTERNAL, "async get next event from grpc completion queue failed");
        return;
    }

    // 处理 Read 事件
    if (got_tag == reinterpret_cast<void*>(1)) {
        // Read 完成，收到第一个响应
        response_received_ = true;
        // 启动 Finish 来获取最终状态
        if (!finish_started_ && reader) {
            reader->Finish(&status, reinterpret_cast<void*>(2));
            finish_started_ = true;
        }
    } else if (got_tag == reinterpret_cast<void*>(2)) {
        // Finish 完成
        finished = true;
        if (!status.ok()) {
            RTP_LLM_LOG_WARNING(
                "PrefillServerCallerContext::checkDone: prefill rpc failed, unique_key: %s, status: %d(%s)",
                unique_key.c_str(),
                status.error_code(),
                status.error_message().c_str());
        } else {
            RTP_LLM_LOG_DEBUG("PrefillServerCallerContext::checkDone: prefill rpc success, unique_key: %s",
                              unique_key.c_str());
        }
    }
}

}  // namespace rtp_llm
