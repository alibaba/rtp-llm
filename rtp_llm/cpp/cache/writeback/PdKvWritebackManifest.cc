#include "rtp_llm/cpp/cache/writeback/PdKvWritebackManifest.h"

#include <algorithm>
#include <string>

#include "absl/status/status.h"

namespace rtp_llm {
namespace {

int64_t firstBlockedByMultimodal(const PdKvWritebackSnapshot& snapshot, int64_t full_blocks) {
    int64_t usable_blocks = full_blocks;
    for (const auto& interval : snapshot.mm_intervals) {
        if (interval.size() < 2) {
            continue;
        }

        const int64_t start = interval[0];
        const int64_t end   = interval[1];
        if (start < 0 || end <= start) {
            continue;
        }

        const int64_t start_block = start / snapshot.seq_size_per_block;
        if (start_block < usable_blocks) {
            usable_blocks = start_block;
        }
    }
    return usable_blocks;
}

}  // namespace

absl::StatusOr<PdKvWritebackManifest> buildPdKvWritebackManifest(const PdKvWritebackSnapshot& snapshot) {
    if (snapshot.seq_size_per_block <= 0) {
        return absl::InvalidArgumentError("seq_size_per_block must be positive");
    }
    if (snapshot.final_token_count < 0) {
        return absl::InvalidArgumentError("final_token_count must be non-negative");
    }
    if (snapshot.prefill_token_count < 0) {
        return absl::InvalidArgumentError("prefill_token_count must be non-negative");
    }

    const int64_t full_blocks = snapshot.final_token_count / snapshot.seq_size_per_block;
    const int64_t limited_blocks =
        std::min<int64_t>(firstBlockedByMultimodal(snapshot, full_blocks), snapshot.cache_keys.size());
    const int64_t prefill_full_blocks = snapshot.prefill_token_count / snapshot.seq_size_per_block;
    const int64_t start_block         = std::min(prefill_full_blocks, limited_blocks);
    const int64_t reusable_blocks     = limited_blocks - start_block;

    PdKvWritebackManifest manifest;
    manifest.request_id = snapshot.request_id;
    manifest.request_key =
        snapshot.request_key.empty() ? "pd_kv_writeback_" + std::to_string(snapshot.request_id) : snapshot.request_key;
    manifest.seq_size_per_block   = snapshot.seq_size_per_block;
    manifest.final_token_count    = snapshot.final_token_count;
    manifest.reusable_block_count = reusable_blocks;
    manifest.cache_keys.assign(snapshot.cache_keys.begin() + start_block, snapshot.cache_keys.begin() + limited_blocks);

    manifest.group_block_ids.reserve(snapshot.group_block_ids.size());
    for (const auto& group_blocks : snapshot.group_block_ids) {
        if (static_cast<int64_t>(group_blocks.size()) < limited_blocks) {
            return absl::InvalidArgumentError("group block count is smaller than reusable block count");
        }
        manifest.group_block_ids.emplace_back(group_blocks.begin() + start_block,
                                              group_blocks.begin() + limited_blocks);
    }

    return manifest;
}

}  // namespace rtp_llm
