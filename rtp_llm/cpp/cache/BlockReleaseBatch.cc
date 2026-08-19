#include "rtp_llm/cpp/cache/BlockReleaseBatch.h"

#include <utility>

namespace rtp_llm {

void BlockReleaseBatch::append(size_t group_id, const std::vector<BlockRefTransition>& transitions) {
    for (const BlockRefTransition& transition : transitions) {
        if (isNullBlockIdx(transition.block_id)) {
            continue;
        }
        if (transition.ref_type != BlockRefType::REQUEST && transition.ref_type != BlockRefType::STORAGE_BACKEND) {
            continue;
        }

        const ReceiptKey key{group_id, transition.block_id};
        const auto       it = receipt_index_map_.find(key);
        if (it == receipt_index_map_.end()) {
            receipt_index_map_.emplace(key, receipts_.size());
            receipts_.push_back(BlockReleaseReceipt{group_id,
                                                    transition.block_id,
                                                    transition.ref_type,
                                                    transition.old_total_ref_count,
                                                    transition.new_total_ref_count,
                                                    transition.block_released});
            continue;
        }

        BlockReleaseReceipt& receipt = receipts_[it->second];
        receipt.released_ref_type    = transition.ref_type;
        receipt.new_total_ref_count  = transition.new_total_ref_count;
        receipt.block_released       = transition.block_released;
    }
}

std::vector<BlockReleaseReceipt> BlockReleaseBatch::finish() {
    std::vector<BlockReleaseReceipt> receipts = std::move(receipts_);
    receipts_.clear();
    receipt_index_map_.clear();
    return receipts;
}

}  // namespace rtp_llm
