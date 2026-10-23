#include "rtp_llm/cpp/cache/connector/p2p/P2PWritebackResourceUtil.h"
#include <map>
#include <set>

namespace rtp_llm {
namespace {

// Groups can share one physical pool; each pool is checked once against its own capacity.
using ReleasesByPool = std::map<DeviceBlockPool*, BlockIdList>;

ReleasesByPool distinctPools(const std::vector<SingleTypeCacheManagerPtr>& groups) {
    ReleasesByPool pools;
    for (const auto& group : groups) {
        pools[group->blockPool().get()];
    }
    return pools;
}

bool poolsKeepReserve(const ReleasesByPool& pools, double reserve_ratio) {
    for (const auto& [pool, releasing] : pools) {
        if (pool->availableBlocksAfterRelease(releasing)
            < static_cast<size_t>(reserve_ratio * pool->totalBlocksNum())) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool writebackKeepsKvReserve(const std::vector<SingleTypeCacheManagerPtr>& groups, double reserve_ratio) {
    return poolsKeepReserve(distinctPools(groups), reserve_ratio);
}

bool writebackKeepsKvReserveAfterRelease(const std::vector<SingleTypeCacheManagerPtr>& groups,
                                         const KVCacheResource&              source,
                                         const KVCacheResource&              retained,
                                         double                              reserve_ratio) {
    std::map<DeviceBlockPool*, std::set<BlockIdxType>> retained_by_pool;
    for (const auto& group : groups) {
        auto& kept = retained_by_pool[group->blockPool().get()];
        for (const auto block : retained.blocks(group->tag())) {
            if (!isNullBlockIdx(block)) {
                kept.insert(block);
            }
        }
    }
    auto pools = distinctPools(groups);
    for (const auto& group : groups) {
        const auto  pool      = group->blockPool().get();
        const auto& kept      = retained_by_pool[pool];
        auto&       releasing = pools[pool];
        for (const auto block : source.blocks(group->tag())) {
            if (!isNullBlockIdx(block) && !kept.count(block)) {
                releasing.push_back(block);
            }
        }
    }
    return poolsKeepReserve(pools, reserve_ratio);
}

}  // namespace rtp_llm
