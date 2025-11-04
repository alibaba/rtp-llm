#include "rtp_llm/cpp/cache_new/TpBroadcaster.h"

namespace rtp_llm {

void TpBroadcaster::registerHandler(const std::string& handler_name, TpBroadcastHandlerPtr handler) {
        handlers_[handler_name] = handler;
    }

bool TpBroadcaster::executeHandler(const std::string&           handler_name,
                                   const BroadcastAllTpRequestPB& request,
                                   BroadcastAllTpResponsePB&      response) {
    auto it = handlers_.find(handler_name);
    if (it == handlers_.end()) {
        RTP_LLM_LOG_WARNING("handler not found: %s", handler_name.c_str());
        return false;
    }
    return it->second->run(request, response);
}

bool TpBroadcaster::broadcast(const std::vector<BroadcastAllTpRequestPB>& requests,
                              std::vector<BroadcastAllTpResponsePB>&      responses,
                              const std::string&                        action,
                              int                                       timeout_ms) {
    if (requests.size() != peers_.size()) {
        RTP_LLM_LOG_WARNING("requests size mismatch, req=%zu peers=%zu", requests.size(), peers_.size());
        return false;
    }
    if (peers_.empty() || !rpc_pool_) {
        RTP_LLM_LOG_WARNING("broadcast failed, empty peers or null rpc_pool");
        return false;
    }

    struct RpcContext {
        std::shared_ptr<RpcService::Stub>    stub;
        std::string                          server_addr;
        BroadcastAllTpRequestPB                request;
        BroadcastAllTpResponsePB               response;
        std::unique_ptr<grpc::ClientContext> client_context = std::make_unique<grpc::ClientContext>();
        grpc::CompletionQueue                cq;
        grpc::Status                         status;
    };

    const int               worker_size = static_cast<int>(peers_.size());
    std::vector<RpcContext> worker(worker_size);

    auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (int rank = 0; rank < worker_size; ++rank) {
        const auto& addr = peers_[rank];
        auto        conn = rpc_pool_->getConnection(addr);
        if (!conn.ok()) {
            RTP_LLM_LOG_WARNING("broadcast: getConnection failed rank=%d addr=%s", rank, addr.c_str());
            return false;
        }
        auto& ctx       = worker[rank];
        ctx.stub        = conn.value().stub;
        ctx.server_addr = addr;
        ctx.request = requests[rank];
        ctx.client_context->set_deadline(deadline);
    }

    std::vector<std::unique_ptr<grpc::ClientAsyncResponseReader<BroadcastAllTpResponsePB>>> readers(worker_size);
    for (int rank = 0; rank < worker_size; ++rank) {
        auto& ctx     = worker[rank];
        readers[rank] = ctx.stub->AsyncBroadcastAllTp(ctx.client_context.get(), ctx.request, &ctx.cq);
        readers[rank]->Finish(&ctx.response, &ctx.status, reinterpret_cast<void*>(static_cast<intptr_t>(rank)));
    }

    std::vector<int> success(worker_size, 0);
    int              finished_count      = 0;
    bool             all_request_success = true;
    while (true) {
        if (finished_count == worker_size) {
            break;
        }
        const int  once_timeout_ms = 1;
        const auto once_deadline   = std::chrono::system_clock::now() + std::chrono::milliseconds(once_timeout_ms);
        for (int i = 0; i < worker_size; ++i) {
            if (success[i] == 1) {
                continue;
            }
            auto& ctx         = worker[i];
            void* got_tag     = nullptr;
            bool  ok          = false;
            auto  next_status = ctx.cq.AsyncNext(&got_tag, &ok, once_deadline);
            if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
                continue;
            }
            if (!ok) {
                RTP_LLM_FAIL("rpc cq failed, status=%d", static_cast<int>(next_status));
            }
            ++finished_count;
            const int rank = static_cast<int>(reinterpret_cast<intptr_t>(got_tag));
            success[rank]  = 1;
            const auto& status  = worker[rank].status;
            // if (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
            //     RTP_LLM_FAIL("rpc timeout rank=%d err=%d(%s) addr=%s",
            //                  rank,
            //                  status.error_code(),
            //                  status.error_message().c_str(),
            //                  worker[rank].server_addr.c_str());
            // }
            if (!status.ok()|| !ctx.response.success()) {
                RTP_LLM_LOG_WARNING("rpc failed rank=%d err=%d(%s) success=%d addr=%s",
                                    rank,
                                    status.error_code(),
                                    status.error_message().c_str(),
                                    ctx.response.success(),
                                    worker[rank].server_addr.c_str());
                all_request_success = false;
            }
        }
    }

    if (all_request_success) {
        responses.clear();
        responses.reserve(worker_size);
        for (auto& ctx : worker) {
            responses.emplace_back(std::move(ctx.response));
        }
    }
    return all_request_success;
}

}  // namespace rtp_llm
