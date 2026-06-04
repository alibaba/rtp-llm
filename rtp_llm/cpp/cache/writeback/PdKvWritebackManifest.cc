#include "rtp_llm/cpp/cache/writeback/PdKvWritebackManifest.h"

#include <algorithm>
#include <string>

#include "absl/status/status.h"
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

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

bool pdKvWritebackCacheDebugEnabled() {
    static const bool enabled = autil::EnvUtil::getEnv("PD_KV_WRITEBACK_DEBUG_CACHE", false);
    return enabled;
}

const char* blockKind(int64_t token_begin, int64_t token_end, int64_t prefill_token_count) {
    if (token_end <= prefill_token_count) {
        return "prefill";
    }
    if (token_begin >= prefill_token_count) {
        return "decode";
    }
    return "mixed";
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
    const int64_t start_block = std::min(snapshot.prefill_token_count / snapshot.seq_size_per_block, limited_blocks);
    const int64_t reusable_blocks = limited_blocks - start_block;

    if (pdKvWritebackCacheDebugEnabled()) {
        RTP_LLM_LOG_INFO(
            "PD KV writeback manifest summary, request_id=%ld, seq_size_per_block=%d, prefill_token_count=%ld, final_token_count=%ld, full_blocks=%ld, start_block=%ld, limited_blocks=%ld, reusable_blocks=%ld, cache_keys=%zu",
            snapshot.request_id,
            snapshot.seq_size_per_block,
            snapshot.prefill_token_count,
            snapshot.final_token_count,
            full_blocks,
            start_block,
            limited_blocks,
            reusable_blocks,
            snapshot.cache_keys.size());
        for (int64_t block = start_block; block < limited_blocks; ++block) {
            const int64_t token_begin = block * snapshot.seq_size_per_block;
            const int64_t token_end =
                std::min<int64_t>(token_begin + snapshot.seq_size_per_block, snapshot.final_token_count);
            const int64_t expected_valid_tokens = std::max<int64_t>(0, token_end - token_begin);
            RTP_LLM_LOG_INFO(
                "PD KV writeback manifest block, request_id=%ld, manifest_index=%ld, original_block=%ld, cache_key=%ld, token_begin=%ld, token_end=%ld, expected_valid_tokens=%ld, block_kind=%s",
                snapshot.request_id,
                block - start_block,
                block,
                snapshot.cache_keys[block],
                token_begin,
                token_end,
                expected_valid_tokens,
                blockKind(token_begin, token_end, snapshot.prefill_token_count));
        }
    }

    PdKvWritebackManifest manifest;
    manifest.request_id = snapshot.request_id;
    manifest.request_key =
        snapshot.request_key.empty() ? "pd_kv_writeback_" + std::to_string(snapshot.request_id) : snapshot.request_key;
    manifest.seq_size_per_block   = snapshot.seq_size_per_block;
    manifest.final_token_count    = snapshot.final_token_count;
    manifest.start_block_index    = start_block;
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
