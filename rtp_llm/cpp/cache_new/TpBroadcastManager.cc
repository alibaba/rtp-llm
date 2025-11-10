#include "rtp_llm/cpp/cache_new/TpBroadcastManager.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

// -------------------------------- TPBroadcastResult --------------------------------

void TPBroadcastResult::waitDone() {
    bool              all_request_success = true;
    const int         worker_size         = worker_rpc_contexts_.size();
    std::vector<bool> finished(worker_size, false);

    while (true) {
        if (finished_count_ == worker_size) {
            break;
        }
        const int  once_timeout_ms = 1;
        const auto once_deadline   = std::chrono::system_clock::now() + std::chrono::milliseconds(once_timeout_ms);
        for (int rank = 0; rank < worker_size; ++rank) {
            if (finished[rank]) {
                continue;
            }

            auto& ctx         = worker_rpc_contexts_.at(rank);
            void* got_tag     = nullptr;
            bool  ok          = false;
            auto  next_status = ctx->completion_queue.AsyncNext(&got_tag, &ok, once_deadline);
            if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
                continue;
            }
            if (!ok) {
                RTP_LLM_FAIL("tp broadcast rpc cq failed, rank=%d status=%d", rank, static_cast<int>(next_status));
            }
            ++finished_count_;
            finished[rank] = true;

            const auto& status = ctx->status;
            if (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
                RTP_LLM_FAIL("tp broadcast rpc timeout, timeout_ms=%d rank=%d err=%d(%s) addr=%s",
                             ctx->timeout_ms,
                             rank,
                             status.error_code(),
                             status.error_message().c_str(),
                             ctx->server_addr.c_str());
            }
            if (!status.ok()) {
                RTP_LLM_LOG_WARNING("tp broadcast rpc failed, rank=%d err=%d(%s) addr=%s",
                                    rank,
                                    status.error_code(),
                                    status.error_message().c_str(),
                                    ctx->server_addr.c_str());
                all_request_success = false;
                cancelAllRequests();
            }
        }
    }
    all_request_success_ = all_request_success;
}

bool TPBroadcastResult::success() const {
    return all_request_success_;
}

std::vector<BroadcastTpResponsePB> TPBroadcastResult::responses() const {
    std::vector<BroadcastTpResponsePB> responses;
    responses.reserve(worker_rpc_contexts_.size());
    for (const auto& worker_rpc_context : worker_rpc_contexts_) {
        responses.push_back(worker_rpc_context->response);
    }
    return responses;
}

void TPBroadcastResult::cancelAllRequests() const {
    for (const auto& context : worker_rpc_contexts_) {
        context->client_context->TryCancel();
    }
}

// -------------------------------- TpBroadcastManager --------------------------------

bool TpBroadcastManager::init() {
    if (worker_addrs_.empty()) {
        RTP_LLM_LOG_WARNING("init failed, worker_addrs is empty");
        return false;
    }

    rpc_pool_ = std::make_shared<RPCPool>();
    return true;
}

std::shared_ptr<TPBroadcastResult> TpBroadcastManager::broadcast(const std::vector<BroadcastTpRequestPB>& requests,
                                                                 int timeout_ms) const {
    const auto worker_size = worker_addrs_.size();
    if (requests.size() != worker_size) {
        RTP_LLM_LOG_WARNING(
            "broadcast failed, requests size mismatch, req: %zu, worker size: %zu", requests.size(), worker_size);
        return nullptr;
    }

    std::vector<std::shared_ptr<TPBroadcastResult::WorkerRpcContext>> contexts(worker_size);
    const auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);

    for (int rank = 0; rank < worker_size; ++rank) {
        const auto& addr        = worker_addrs_[rank];
        auto        conn_status = rpc_pool_->getConnection(addr);
        if (!conn_status.ok()) {
            RTP_LLM_LOG_WARNING("broadcast: getConnection failed rank=%d addr=%s", rank, addr.c_str());
            return nullptr;
        }

        contexts[rank]      = std::make_shared<TPBroadcastResult::WorkerRpcContext>();
        auto& ctx           = contexts.at(rank);
        ctx->stub           = conn_status.value().stub;
        ctx->request        = requests.at(rank);
        ctx->server_addr    = addr;
        ctx->timeout_ms     = timeout_ms;
        ctx->client_context = std::make_shared<grpc::ClientContext>();
        ctx->client_context->set_deadline(deadline);
    }

    for (int rank = 0; rank < worker_size; ++rank) {
        auto& ctx    = contexts.at(rank);
        auto  reader = ctx->stub->AsyncBroadcastTp(ctx->client_context.get(), ctx->request, &ctx->completion_queue);
        reader->Finish(&ctx->response, &ctx->status, reinterpret_cast<void*>(static_cast<intptr_t>(rank)));
    }

    return std::make_shared<TPBroadcastResult>(std::move(contexts));
}

}  // namespace rtp_llm
