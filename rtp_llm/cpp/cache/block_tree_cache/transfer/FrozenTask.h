#pragma once
#include <algorithm>
#include <memory>
#include <vector>
#include <unordered_set>
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {
struct FrozenTask {
    enum class QueryKind { TRANSFER, KVCM };
    struct Block {
        std::shared_ptr<IBlockPool> pool;
        BlockIdxType block;
    };
    explicit FrozenTask(const TransferResult& result, QueryKind query_kind = QueryKind::TRANSFER):
        finalFrozen(result.status.code() != ErrorCode::TRANSFER_COMPLETION_UNKNOWN || result.recovery_task_ids.empty()),
        recovery_task_ids(result.recovery_task_ids), query_kind(query_kind) {}

    void freeze(std::shared_ptr<IBlockPool> pool, BlockIdxType block) {
        if (std::any_of(blocks.begin(), blocks.end(), [&](const Block& b) {
                return b.pool == pool && b.block == block;
            })) return;
        blocks.push_back({std::move(pool), block});
        blocks.back().pool->freezeBlock(block);
    }

    bool readyToRelease(const std::unordered_set<std::string>& completed_ids) const {
        return !finalFrozen && !recovery_task_ids.empty()
            && std::all_of(recovery_task_ids.begin(), recovery_task_ids.end(),
                           [&](const std::string& id) { return completed_ids.count(id) != 0; });
    }
    bool unfreeze(const std::unordered_set<std::string>& completed_ids) {
        if (!readyToRelease(completed_ids)) return false;
        for (const auto& block : blocks) block.pool->unfreezeBlock(block.block);
        blocks.clear();
        return true;
    }

    bool finalFrozen;
    std::vector<std::string> recovery_task_ids;
    QueryKind query_kind;
    std::vector<Block> blocks;
};
}  // namespace rtp_llm
