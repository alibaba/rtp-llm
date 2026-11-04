#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>
#include <utility>
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
struct FrozenTask {
    enum class QueryKind { TRANSFER, KVCM };
    struct Block {
        std::shared_ptr<IBlockPool> pool;
        BlockIdxType block;
    };
    explicit FrozenTask(const TransferResult& result,
                        QueryKind             query_kind = QueryKind::TRANSFER,
                        const char*           operation  = "unknown",
                        uint64_t              transfer_id = 0,
                        int64_t               request_id  = -1):
        finalFrozen(result.status.code() != ErrorCode::TRANSFER_COMPLETION_UNKNOWN || result.recovery_task_ids.empty()),
        recovery_task_ids(result.recovery_task_ids), query_kind(query_kind) {
        diagnostic_context = "operation=" + std::string(operation) + " request_id=" + std::to_string(request_id)
                             + " transfer_id=" + std::to_string(transfer_id) + " recovery_task_ids=[";
        for (size_t index = 0; index < recovery_task_ids.size(); ++index) {
            diagnostic_context += (index == 0 ? "" : ",") + recovery_task_ids[index];
        }
        diagnostic_context += std::string("] query_kind=") + (query_kind == QueryKind::KVCM ? "kvcm" : "transfer")
                              + " final_frozen=" + (finalFrozen ? "1" : "0");
    }

    // A task may contain descriptors with different directions (e.g. HOST/DISK loads).
    void freeze(std::shared_ptr<IBlockPool> pool,
                BlockIdxType                block,
                Tier                        source_tier = Tier::NONE,
                Tier                        target_tier = Tier::NONE) {
        if (std::any_of(blocks.begin(), blocks.end(), [&](const Block& b) {
                return b.pool == pool && b.block == block;
            })) return;
        blocks.push_back({std::move(pool), block});
        // Log before the pool mutation: exceeding the frozen-block limit may abort.
        RTP_LLM_LOG_WARNING("cache block freeze %s source=%s target=%s pool=%s block=%d",
                            diagnostic_context.c_str(),
                            tierName(source_tier),
                            tierName(target_tier),
                            blocks.back().pool->poolName().c_str(),
                            block);
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
        // Recovery accounting belongs to the operation, not to each pool hold.
        // Consume the callback so repeated completion observations cannot settle it twice.
        if (auto callback = std::exchange(on_unfreeze, {})) callback();
        return true;
    }

    bool finalFrozen;
    std::vector<std::string> recovery_task_ids;
    QueryKind query_kind;
    std::vector<Block> blocks;
    std::string diagnostic_context;
    std::function<void()> on_unfreeze;
};
}  // namespace rtp_llm
