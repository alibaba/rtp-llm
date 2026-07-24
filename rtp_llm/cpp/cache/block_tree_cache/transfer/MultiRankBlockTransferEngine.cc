#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"

#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferRequestConverter.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

MultiRankBlockTransferEngine::MultiRankBlockTransferEngine(std::vector<ComponentGroupPtr>    component_groups,
                                                           std::shared_ptr<BroadcastManager> broadcast_manager):
    component_groups_(std::move(component_groups)), broadcast_manager_(std::move(broadcast_manager)) {}

bool MultiRankBlockTransferEngine::execute(const std::vector<TransferDescriptor>& descriptors,
                                                int                                    timeout_ms) const {
    if (broadcast_manager_ == nullptr) {
        RTP_LLM_LOG_WARNING("broadcast manager is not initialized");
        return false;
    }
    if (descriptors.empty() || timeout_ms <= 0) {
        RTP_LLM_LOG_WARNING("invalid batch, item_count=%zu, timeout_ms=%d",
                            descriptors.size(),
                            timeout_ms);
        return false;
    }

    MemoryOperationRequestPB request;
    for (const TransferDescriptor& descriptor : descriptors) {
        if (!BlockTransferRequestConverter::appendTransfer(descriptor, component_groups_, request)) {
            RTP_LLM_LOG_WARNING("failed to encode transfer, "
                                "group=%d source=%s target=%s",
                                descriptor.component_group_id,
                                tierName(descriptor.source_tier),
                                tierName(descriptor.target_tier));
            return false;
        }
    }

    const size_t worker_count = broadcast_manager_->workerNum();
    if (worker_count == 0) {
        RTP_LLM_LOG_WARNING("no worker configured");
        return false;
    }

    FunctionRequestPB         function_request;
    MemoryOperationRequestPB* memory_request = function_request.mutable_mem_request();
    if (memory_request == nullptr) {
        RTP_LLM_LOG_WARNING("failed to create memory request");
        return false;
    }
    memory_request->CopyFrom(request);
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
        RTP_LLM_LOG_WARNING("worker RPC failed");
        return false;
    }

    const std::vector<FunctionResponsePB> responses = broadcast_result->responses();
    if (responses.size() != worker_count) {
        RTP_LLM_LOG_WARNING("response count mismatch, expected=%zu, actual=%zu",
                            worker_count,
                            responses.size());
        return false;
    }
    for (size_t rank = 0; rank < responses.size(); ++rank) {
        const FunctionResponsePB& response = responses[rank];
        if (!response.has_mem_response() || !response.mem_response().success()) {
            RTP_LLM_LOG_WARNING("worker transfer failed, rank=%zu", rank);
            return false;
        }
    }
    return true;
}

}  // namespace rtp_llm
