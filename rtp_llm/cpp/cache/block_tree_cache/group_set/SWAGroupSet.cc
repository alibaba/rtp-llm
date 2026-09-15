#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"

namespace rtp_llm {

SWAGroupSet::SWAGroupSet(size_t                          sliding_window_size,
                         size_t                          seq_size_per_block,
                         std::vector<DeviceBlockPoolPtr> device_pools,
                         std::shared_ptr<HostBlockPool>  host_pool,
                         BlockTreeDiskBlockPoolPtr       disk_pool):
    GroupSet(std::move(device_pools), std::move(host_pool), std::move(disk_pool)),
    sliding_window_size_(sliding_window_size),
    seq_size_per_block_(seq_size_per_block) {}

std::unique_ptr<MatchValidator> SWAGroupSet::createMatchValidator() {
    return std::make_unique<SWAMatchValidator>(sliding_window_size_, seq_size_per_block_);
}

size_t SWAGroupSet::computeReuseBlockCount(size_t matched_block_count) const {
    return groupAt(0).reuseBlockCount(matched_block_count);
}

// SWAMatchValidator
SWAMatchValidator::SWAMatchValidator(size_t sliding_window_size, size_t seq_size_per_block):
    sliding_window_size_(sliding_window_size), seq_size_per_block_(seq_size_per_block) {}

bool SWAMatchValidator::validate(const GroupSetResource& resource) {
    // A busy resource (DEMOTING/LOAD_PENDING) counts as a hole: it cannot serve
    // this match, so the window accumulation restarts behind it.
    const bool has_swa_data = resource.isMatchUsable() && !resource.is_empty();

    if (!has_swa_data) {
        connected_to_root_  = false;
        accumulated_length_ = 0;
        return false;
    }

    accumulated_length_ += seq_size_per_block_;
    return connected_to_root_
           || (sliding_window_size_ > 0 && accumulated_length_ >= sliding_window_size_);
}

}  // namespace rtp_llm
