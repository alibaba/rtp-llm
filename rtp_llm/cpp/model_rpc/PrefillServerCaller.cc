#include "rtp_llm/cpp/model_rpc/PrefillServerCaller.h"
#include "rtp_llm/cpp/model_rpc/RpcTimeoutUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

namespace rtp_llm {

constexpr int32_t kPeerInfoProbeMaxMs = 500;
constexpr int32_t kPeerInfoProbeMinMs = 50;

namespace {

class SyncPrefillRpcCancelGuard {
public:
    SyncPrefillRpcCancelGuard(const std::function<bool()>&          is_cancelled,
                              grpc::ClientContext*                  client_context,
                              std::chrono::system_clock::time_point deadline):
        is_cancelled_(is_cancelled), client_context_(client_context), deadline_(deadline) {
        watcher_ = std::thread([this]() { watch(); });
    }

    ~SyncPrefillRpcCancelGuard() {
        stop();
    }

    void stop() {
        stop_.store(true);
        if (watcher_.joinable()) {
            watcher_.join();
        }
    }

    bool timedOut() const {
        return timed_out_.load();
    }

    bool clientCancelled() const {
        return client_cancelled_.load();
    }

private:
    void watch() {
        while (!stop_.load()) {
            if (is_cancelled_()) {
                client_cancelled_.store(true);
                client_context_->TryCancel();
                break;
            }
            if (std::chrono::system_clock::now() >= deadline_) {
                timed_out_.store(true);
                client_context_->TryCancel();
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

private:
    std::function<bool()>                 is_cancelled_;
    grpc::ClientContext*                  client_context_;
    std::chrono::system_clock::time_point deadline_;
    std::atomic<bool>                     stop_{false};
    std::atomic<bool>                     timed_out_{false};
    std::atomic<bool>                     client_cancelled_{false};
    std::thread                           watcher_;
};

}  // namespace

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

    // Preserve the original request shape for decode_entrance handoff so the
    // prefill side still recognizes it as a PD-separation request.
    context->request_.CopyFrom(*request);
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
    context->startPolling();

    return context;
}

grpc::Status PrefillServerCaller::callPrefill(grpc::ServerContext*                   server_context,
                                              const GenerateInputPB*                 request,
                                              grpc::ServerWriter<GenerateOutputsPB>* response_writer) {
    return callPrefill(server_context, request, response_writer, [server_context]() {
        return server_context && server_context->IsCancelled();
    });
}

grpc::Status PrefillServerCaller::callPrefill(grpc::ServerContext*                   server_context,
                                              const GenerateInputPB*                 request,
                                              grpc::ServerWriter<GenerateOutputsPB>* response_writer,
                                              const std::string&                     target_addr) {
    return callPrefillToAddr(server_context, request, response_writer, target_addr, [server_context]() {
        return server_context && server_context->IsCancelled();
    });
}

grpc::Status PrefillServerCaller::callPrefill(grpc::ServerContext*                   server_context,
                                              const GenerateInputPB*                 request,
                                              grpc::ServerWriter<GenerateOutputsPB>* response_writer,
                                              const std::function<bool()>&           is_cancelled) {
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
    return callPrefillToAddr(server_context, request, response_writer, prefill_addr, is_cancelled);
}

grpc::Status PrefillServerCaller::callPrefillToAddr(grpc::ServerContext*                   server_context,
                                                    const GenerateInputPB*                 request,
                                                    grpc::ServerWriter<GenerateOutputsPB>* response_writer,
                                                    const std::string&                     prefill_addr,
                                                    const std::function<bool()>&           is_cancelled) {
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
    const auto          rpc_timeout_ms = normalizeRpcTimeoutMs(request->generate_config().timeout_ms(), 0);
    const auto          deadline       = std::chrono::system_clock::now() + std::chrono::milliseconds(rpc_timeout_ms);
    client_context.set_deadline(deadline);
    SyncPrefillRpcCancelGuard cancel_guard(is_cancelled, &client_context, deadline);

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
        if (is_cancelled()) {
            RTP_LLM_LOG_WARNING("request [%lld] client cancelled while forwarding prefill response",
                                request->request_id());
            client_context.TryCancel();
            return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
        }
        if (!response_writer->Write(response)) {
            RTP_LLM_LOG_WARNING("request [%lld] write response to client failed", request->request_id());
            client_context.TryCancel();
            return grpc::Status(grpc::StatusCode::INTERNAL, "write response to client failed");
        }
    }

    // 获取最终状态
    auto status = reader->Finish();
    cancel_guard.stop();
    if (!status.ok() && cancel_guard.timedOut()) {
        client_context.TryCancel();
        return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "prefill server rpc timeout");
    }
    if (!status.ok() && cancel_guard.clientCancelled()) {
        client_context.TryCancel();
        return grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
    }
    if (!status.ok()) {
        RTP_LLM_LOG_WARNING("request [%lld] prefill server rpc failed, status %d(%s)",
                            request->request_id(),
                            status.error_code(),
                            status.error_message().c_str());
    }

    return status;
}

PrefillPeerInfo PrefillServerCaller::getPrefillPeerInfo(const std::string& ip, uint32_t port, int32_t request_timeout_ms) {
    std::string addr = ip + ":" + std::to_string(port);

    {
        std::shared_lock<std::shared_mutex> lock(prefill_peer_cache_mutex_);
        auto                                it = prefill_peer_cache_.find(addr);
        if (it != prefill_peer_cache_.end()) {
            return it->second;
        }
    }

    auto conn = rpc_pool_->getConnection(addr);
    if (!conn.ok()) {
        RTP_LLM_LOG_WARNING("getPrefillPeerInfo: getConnection failed for %s", addr.c_str());
        return {};
    }

    grpc::ClientContext ctx;
    ctx.set_deadline(
        std::chrono::system_clock::now()
        + std::chrono::milliseconds(std::clamp(request_timeout_ms, kPeerInfoProbeMinMs, kPeerInfoProbeMaxMs)));

    GetPeerInfoRequestPB  request;
    GetPeerInfoResponsePB response;
    auto                  status = conn.value().stub->GetPeerInfo(&ctx, request, &response);

    if (!status.ok()) {
        RTP_LLM_LOG_ERROR(
            "getPrefillPeerInfo: GetPeerInfo RPC failed for %s: %s", addr.c_str(), status.error_message().c_str());
        return {};
    }

    PrefillPeerInfo info;
    info.tp_size = response.tp_size();
    if (info.tp_size <= 0) {
        RTP_LLM_LOG_ERROR("getPrefillPeerInfo: invalid tp_size=%d from %s", info.tp_size, addr.c_str());
        return {};
    }

    for (const auto& dp_addr : response.dp_grpc_addrs()) {
        info.dp_addrs.push_back(dp_addr);
    }
    if (info.dp_addrs.empty()) {
        info.dp_addrs.push_back(addr);
    }

    RTP_LLM_LOG_INFO("getPrefillPeerInfo: prefill %s tp_size=%d, dp_addrs=[%s]",
                      addr.c_str(),
                      info.tp_size,
                      [&]() {
                          std::string s;
                          for (size_t i = 0; i < info.dp_addrs.size(); ++i) {
                              if (i > 0) s += ", ";
                              s += info.dp_addrs[i];
                          }
                          return s;
                      }().c_str());

    {
        std::unique_lock<std::shared_mutex> lock(prefill_peer_cache_mutex_);
        prefill_peer_cache_[addr] = info;
    }
    return info;
}

int PrefillServerCaller::getPrefillTpSize(const std::string& ip, uint32_t port, int32_t request_timeout_ms) {
    return getPrefillPeerInfo(ip, port, request_timeout_ms).tp_size;
}

}  // namespace rtp_llm
