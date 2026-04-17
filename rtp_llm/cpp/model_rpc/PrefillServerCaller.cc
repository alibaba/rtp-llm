#include "rtp_llm/cpp/model_rpc/PrefillServerCaller.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

constexpr int32_t kPeerInfoProbeMaxMs = 500;

PrefillServerCaller::PrefillServerCaller(const std::string& process_id):
    rpc_pool_(std::make_shared<RPCPool>()), process_id_(process_id) {}

std::shared_ptr<PrefillServerCallerContext> PrefillServerCaller::callPrefill(const GenerateInputPB* request,
                                                                             const std::string&     ip,
                                                                             uint32_t               port,
                                                                             const std::string&     unique_key,
                                                                             int64_t                deadline_us) {
    std::string prefill_addr   = ip + ":" + std::to_string(port);
    auto        connect_status = rpc_pool_->getConnection(prefill_addr);
    if (!connect_status.ok()) {
        RTP_LLM_LOG_WARNING("request [%lld] get grpc connection to prefill failed, addr: %s",
                            request->request_id(),
                            prefill_addr.c_str());
        return nullptr;
    }

    auto stub = connect_status.value().stub;

    auto context   = std::make_shared<PrefillServerCallerContext>(ip + ":" + std::to_string(port), unique_key);
    context->stub_ = stub;

    // treat prefill server as normal server
    context->request_.CopyFrom(*request);
    // should not set this, or we can not check if prefill is finished
    context->request_.mutable_generate_config()->set_max_new_tokens(1);
    context->request_.set_client_id(process_id_);
    context->request_.set_start_time(currentTimeUs());
    context->request_.mutable_generate_config()->set_can_use_pd_separation(true);
    context->request_.mutable_generate_config()->set_unique_key(unique_key);

    context->client_context_->set_deadline(std::chrono::system_clock::now() + std::chrono::microseconds(deadline_us));

    std::unique_ptr<grpc::ClientAsyncReader<GenerateOutputsPB>> reader(
        context->stub_->AsyncGenerateStreamCall(context->client_context_.get(),
                                                context->request_,
                                                context->completion_queue_.get(),
                                                reinterpret_cast<void*>(0)));

    context->reader_ = std::move(reader);

    // Start reading first response (non-blocking)
    context->reader_->Read(&context->response_, reinterpret_cast<void*>(1));

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

int PrefillServerCaller::getPrefillTpSize(const std::string& ip, uint32_t port, int32_t request_timeout_ms) {
    std::string addr = ip + ":" + std::to_string(port);

    {
        std::shared_lock<std::shared_mutex> lock(prefill_tp_cache_mutex_);
        auto                                it = prefill_tp_cache_.find(addr);
        if (it != prefill_tp_cache_.end()) {
            return it->second;
        }
    }

    auto conn = rpc_pool_->getConnection(addr);
    if (!conn.ok()) {
        RTP_LLM_LOG_WARNING("getPrefillTpSize: getConnection failed for %s", addr.c_str());
        return -1;
    }

    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now()
                     + std::chrono::milliseconds(std::max(request_timeout_ms, kPeerInfoProbeMaxMs)));

    GetPeerInfoRequestPB  request;
    GetPeerInfoResponsePB response;
    auto                  status = conn.value().stub->GetPeerInfo(&ctx, request, &response);

    if (!status.ok()) {
        RTP_LLM_LOG_ERROR(
            "getPrefillTpSize: GetPeerInfo RPC failed for %s: %s", addr.c_str(), status.error_message().c_str());
        return -1;
    }

    int tp_size = response.tp_size();
    if (tp_size <= 0) {
        RTP_LLM_LOG_ERROR("getPrefillTpSize: invalid tp_size=%d from %s", tp_size, addr.c_str());
        return -1;
    }

    RTP_LLM_LOG_INFO("getPrefillTpSize: prefill %s tp_size=%d", addr.c_str(), tp_size);

    {
        std::unique_lock<std::shared_mutex> lock(prefill_tp_cache_mutex_);
        prefill_tp_cache_[addr] = tp_size;
    }
    return tp_size;
}

}  // namespace rtp_llm
