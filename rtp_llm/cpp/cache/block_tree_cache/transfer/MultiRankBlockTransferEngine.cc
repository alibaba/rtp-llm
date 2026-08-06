#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"

#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

MultiRankBlockTransferEngine::MultiRankBlockTransferEngine(std::vector<GroupSetPtr>          group_sets,
                                                           std::shared_ptr<BroadcastManager> broadcast_manager):
    group_sets_(std::move(group_sets)), broadcast_manager_(std::move(broadcast_manager)) {}

bool MultiRankBlockTransferEngine::execute(const std::vector<TransferDescriptor>& descriptors, int timeout_ms) const {
    if (descriptors.empty() || timeout_ms <= 0) {
        RTP_LLM_LOG_WARNING("invalid batch, item_count=%zu, timeout_ms=%d", descriptors.size(), timeout_ms);
        return false;
    }

    MemoryOperationRequestPB request;
    if (!BlockTransferRequestConverter::encodeTransfer(request, descriptors, group_sets_)) {
        RTP_LLM_LOG_WARNING("failed to encode transfer batch, item_count=%zu", descriptors.size());
        return false;
    }
    const size_t              worker_count = broadcast_manager_->workerNum();
    FunctionRequestPB         function_request;
    function_request.mutable_mem_request()->CopyFrom(request);
    std::vector<FunctionRequestPB> requests(worker_count, function_request);
    auto broadcast_result = broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
        requests,
        timeout_ms,
        [](const std::shared_ptr<RpcService::Stub>&    stub,
           const std::shared_ptr<grpc::ClientContext>& context,
           const FunctionRequestPB&                    rpc_request,
           grpc::CompletionQueue*                      completion_queue) {
            return stub->AsyncExecuteFunction(context.get(), rpc_request, completion_queue);
        });
    if (broadcast_result == nullptr) {
        RTP_LLM_LOG_WARNING("failed to start broadcast");
        return false;
    }

    broadcast_result->waitDone();
    if (!broadcast_result->success()) {
        RTP_LLM_FAIL("multi-rank transfer aborted, at least one worker RPC status is not OK; worker copy state is "
                     "unknown, item_count=%zu worker_count=%zu timeout_ms=%d",
                     descriptors.size(),
                     worker_count,
                     timeout_ms);
    }

    const std::vector<FunctionResponsePB> responses = broadcast_result->responses();
    if (responses.size() != worker_count) {
        RTP_LLM_FAIL("multi-rank transfer aborted, response count mismatch, expected=%zu actual=%zu",
                     worker_count,
                     responses.size());
    }

    bool transfer_success = true;
    for (size_t rank = 0; rank < responses.size(); ++rank) {
        const FunctionResponsePB& response = responses[rank];
        if (!response.has_mem_response()) {
            RTP_LLM_FAIL("multi-rank transfer aborted, response has no mem_response, rank=%zu", rank);
        }
        switch (response.mem_response().code()) {
            case MemoryOperationResponsePB::OK:
                break;
            case MemoryOperationResponsePB::FAILED:
                RTP_LLM_LOG_WARNING("worker transfer failed, rank=%zu", rank);
                transfer_success = false;
                break;
            default:
                RTP_LLM_FAIL("multi-rank transfer aborted, unexpected mem_response code, rank=%zu code=%d",
                             rank,
                             static_cast<int>(response.mem_response().code()));
        }
    }
    return transfer_success;
}

}  // namespace rtp_llm
