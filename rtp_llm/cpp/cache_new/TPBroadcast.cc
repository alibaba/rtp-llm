#include "rtp_llm/cpp/cache_new/TPBroadcast.h"

namespace rtp_llm {

TPBroadcastResult::TPBroadcastResult(
    uint32_t once_timeout_ms, const std::vector<std::shared_ptr<TPBroadcastWorkerRpcContext>>& worker_rpc_contexts):
    once_timeout_ms_(once_timeout_ms), worker_rpc_contexts_(worker_rpc_contexts) {}

bool TPBroadcastResult::success() const {
    return all_request_success_;
}

const std::string& TPBroadcastResult::error_info() const {
    return error_info_;
}

void TPBroadcastResult::waitDone() {
    while (true) {
        if (finished_count_ == worker_rpc_contexts_.size()) {
            break;
        }
        const int  once_timeout_ms = 1;
        const auto once_deadline   = std::chrono::system_clock::now() + std::chrono::milliseconds(once_timeout_ms);
        for (int rank = 0; rank < worker_rpc_contexts_.size(); ++rank) {
            auto&      rpc_context = worker_rpc_contexts_[rank];
            void*      got_tag     = nullptr;
            bool       ok          = false;
            const auto next_status = rpc_context->completion_queue.AsyncNext(&got_tag, &ok, once_deadline);
            if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
                continue;
            }
            if (!ok) {
                // TODO: handle error, some abort, some continous
                RTP_LLM_LOG_WARNING(
                    "request failed, grpc completion queue failed, status: %d, rank: %d", next_status, rank);
                all_request_success_ = false;
                finished_count_++;
                error_info_ += "grpc completion queue failed, rank" + std::to_string(rank) + " failed; ";
                continue;
            }
            ++finished_count_;
            const auto& status = rpc_context->status;
            if (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
                RTP_LLM_LOG_WARNING("request failed, rank %d failed, error: %d(%s), addr: %s",
                                    rank,
                                    status.error_code(),
                                    status.error_message().c_str(),
                                    rpc_context->server_addr.c_str(),
                                    rank);
                all_request_success_ = false;
                error_info_ += "rank " + std::to_string(rank) + " failed, error: " + status.error_message() + "; ";
                continue;
            }

            if (!status.ok() || !rpc_context->response_pb.success()) {
                RTP_LLM_LOG_WARNING("request failed, rank %d failed, error: %d(%s), addr: %s",
                                    rank,
                                    status.error_code(),
                                    status.error_message().c_str(),
                                    rpc_context->server_addr.c_str(),
                                    rank);
                all_request_success_ = false;
                error_info_ += "rank " + std::to_string(rank) + " failed, error: " + status.error_message() + "; ";
                continue;
            }
        }
        if (finished_count_ == worker_rpc_contexts_.size()) {
            break;
        }
    }
}

TPBroadcast::TPBroadcast(const std::vector<std::string>& grpc_workers,
                         const std::shared_ptr<RPCPool>& rpc_pool,
                         uint32_t                        once_timeout_ms):
    grpc_workers_(grpc_workers), rpc_pool_(rpc_pool), once_timeout_ms_(once_timeout_ms) {}

template<typename BroadcastInfoPB>
std::shared_ptr<TPBroadcastResult> TPBroadcast::broadcast(const BroadcastInfoPB& broadcast_info_pb, int timeout_ms) {
    std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);
    const int                                                 worker_size = static_cast<int>(grpc_workers_.size());
    std::vector<std::shared_ptr<TPBroadcastWorkerRpcContext>> worker_rpc_contexts(worker_size);

    for (int rank = 0; rank < worker_size; ++rank) {
        const auto& worker_addr    = grpc_workers_[rank];
        auto        connect_status = rpc_pool_->getConnection(worker_addr);
        if (!connect_status.ok()) {
            RTP_LLM_LOG_WARNING(
                "broadcast failed, get grpc connection failed for rank: %d, addr: %s", rank, worker_addr.c_str());
            return nullptr;
        }
        auto& rpc_context        = worker_rpc_contexts[rank];
        rpc_context->stub        = connect_status.value().stub;
        rpc_context->server_addr = worker_addr;
        rpc_context->request_pb.mutable_p2p_info()->CopyFrom(broadcast_info_pb);

        rpc_context->client_context = std::make_shared<grpc::ClientContext>();
        rpc_context->client_context->set_deadline(deadline);
    }

    for (int rank = 0; rank < worker_size; ++rank) {
        auto& rpc_context = worker_rpc_contexts[rank];
        auto  reader      = rpc_context->stub->AsyncBroadcastAllTp(
            rpc_context->client_context.get(), (rpc_context->request_pb), &(rpc_context->completion_queue));
        reader->Finish(&(rpc_context->response_pb), &(rpc_context->status), reinterpret_cast<void*>(rank));
    }

    return std::make_shared<TPBroadcastResult>(once_timeout_ms_, worker_rpc_contexts);
}

}  // namespace rtp_llm