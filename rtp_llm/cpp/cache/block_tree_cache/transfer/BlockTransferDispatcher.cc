#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <string>
#include <utility>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

BlockTransferDispatcher::BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                                                 std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine,
                                                 size_t max_descriptors_per_batch):
    per_rank_engine_(std::move(per_rank_engine)),
    multi_rank_engine_(std::move(multi_rank_engine)),
    max_descriptors_per_batch_(max_descriptors_per_batch) {
    RTP_LLM_CHECK(max_descriptors_per_batch_ > 0);
}

std::shared_ptr<AsyncContext>
BlockTransferDispatcher::executePerRank(const std::vector<TransferDescriptor>& descriptors) const {
    return per_rank_engine_->submit(descriptors);
}

std::shared_ptr<AsyncContext>
BlockTransferDispatcher::executeMultiRank(const std::vector<TransferDescriptor>& descriptors, int timeout_ms) const {
    if (descriptors.empty()) {
        return std::make_shared<CompletedAsyncContext>(ErrorInfo::OkStatus());
    }
    if (multi_rank_engine_ != nullptr) {
        return multi_rank_engine_->execute(descriptors, timeout_ms);
    }
    return executePerRank(descriptors);
}

bool BlockTransferDispatcher::runTransfer(const std::vector<TransferDescriptor>& descriptors, int timeout_ms) const {
    for (const auto& descriptor : descriptors) {
        auto context = executeMultiRank({descriptor}, timeout_ms);
        context->waitDone();
        if (!context->success()) {
            return false;
        }
    }
    return true;
}

void BlockTransferDispatcher::runTransfer(const std::vector<TransferDescriptor>& descriptors,
                                          int                                    timeout_ms,
                                          TransferDoneCallback                   callback) const {
    struct DescriptorGroup {
        Tier                            source{Tier::NONE};
        Tier                            target{Tier::NONE};
        size_t                          group_set_id{0};
        std::vector<TransferDescriptor> descriptors;
    };

    std::vector<DescriptorGroup> groups;
    for (const auto& descriptor : descriptors) {
        const auto group = std::find_if(groups.begin(), groups.end(), [&descriptor](const DescriptorGroup& item) {
            return item.source == descriptor.source_tier && item.target == descriptor.target_tier
                   && item.group_set_id == descriptor.group_set_id;
        });
        if (group != groups.end()) {
            group->descriptors.push_back(descriptor);
        } else {
            groups.push_back(DescriptorGroup{descriptor.source_tier,
                                             descriptor.target_tier,
                                             descriptor.group_set_id,
                                             {descriptor}});
        }
    }

    auto stage_state = std::make_shared<TransferStageState>(std::move(callback));
    for (const auto& group : groups) {
        for (size_t begin = 0; begin < group.descriptors.size(); begin += max_descriptors_per_batch_) {
            const size_t end = std::min(begin + max_descriptors_per_batch_, group.descriptors.size());
            std::vector<TransferDescriptor> batch(group.descriptors.begin() + begin,
                                                   group.descriptors.begin() + end);
            stage_state->addBatch();
            try {
                auto context = executeMultiRank(batch, timeout_ms);
                if (context == nullptr) {
                    stage_state->completeBatch(
                        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "transfer engine returned a null context"));
                    continue;
                }
                context->onDone([stage_state](ErrorInfo error) { stage_state->completeBatch(std::move(error)); });
            } catch (const std::exception& error) {
                stage_state->completeBatch(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
            } catch (...) {
                stage_state->completeBatch(
                    ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown transfer submission exception"));
            }
        }
    }
    stage_state->finishSubmitting();
}

void BlockTransferDispatcher::cancelPendingStagingTransfers() const {
    per_rank_engine_->cancelPendingStagingTransfers();
}

void BlockTransferDispatcher::shutdown() const {
    per_rank_engine_->shutdown();
}

}  // namespace rtp_llm
