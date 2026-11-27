#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

namespace rtp_llm {

BlockTransferDispatcher::BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine> per_rank_engine,
                                                 std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine):
    BlockTransferDispatcher(
        multi_rank_engine ? multi_rank_engine->groupSets() :
        per_rank_engine   ? per_rank_engine->groupSets() :
                            std::vector<GroupSetPtr>{},
        TransferTransport{
            [this](TransferTask task) -> TransferTransport::Context {
                if (per_rank_engine_) {
                    return per_rank_engine_->execute(std::move(task));
                }
                if (const auto& proof = task.completionProof()) {
                    proof->report(true);
                }
                auto result = TransferResult::executionException(true, "Local block transfer executor unavailable");
                result.copy_error = CacheCopyError::NOT_EXECUTED;
                return std::make_shared<CompletedTransferAsyncContext>(std::move(result), task.descriptors().size());
            },
            multi_rank_engine ? std::function<TransferTransport::Context(TransferTask, const char*)>{
                                    [this](TransferTask task, const char* operation) {
                                        return multi_rank_engine_->execute(std::move(task), operation);
                                    }} : nullptr,
            multi_rank_engine ? multi_rank_engine->participantAddresses() : std::vector<std::string>{},
            [this] {
                if (per_rank_engine_)
                    per_rank_engine_->cancelPendingStagingTransfers();
            },
            [this] {
                if (per_rank_engine_)
                    per_rank_engine_->shutdown();
            },
            [this] { return per_rank_engine_ ? per_rank_engine_->queueSizes() : BlockTreeQueueSizes{}; }}) {
    per_rank_engine_ = std::move(per_rank_engine);
    multi_rank_engine_ = std::move(multi_rank_engine);
}

}  // namespace rtp_llm
