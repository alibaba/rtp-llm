#pragma once

#include <memory>

#include "rtp_llm/cpp/cache/test/mock/MockCoordinatorCacheManager.h"

namespace rtp_llm::test {

// Forward cache operations to the real allocator while callers control allocation readiness.
inline auto makeDelegatingTestAllocator(const CacheConfig& config, const CoordinatorCacheManagerPtr& real_allocator) {
    auto allocator = std::make_shared<testing::NiceMock<MockCoordinatorCacheManager>>(config);
    // Unmocked accounting and failure classification need the delegated allocator's topology.
    allocator->group_block_pools_ = real_allocator->groupBlockPools();
    allocator->kv_cache_groups_   = real_allocator->cacheGroups();
    allocator->setReserveBlocksNum(real_allocator->reserveBlocksNum());
    ON_CALL(*allocator, incrMalloc(testing::_)).WillByDefault(testing::Invoke([real_allocator](const MallocInfo& info) {
        return real_allocator->incrMalloc(info);
    }));
    ON_CALL(*allocator, free(testing::_)).WillByDefault(testing::Invoke([real_allocator](const FreeInfo& info) {
        real_allocator->free(info);
    }));
    ON_CALL(*allocator, insertIntoCache(testing::_, testing::_))
        .WillByDefault(testing::Invoke([real_allocator](const InsertInfo& info, size_t& resident_prefix_length) {
            real_allocator->insertIntoCache(info, resident_prefix_length);
        }));
    ON_CALL(*allocator, availableBlocksNum()).WillByDefault(testing::Invoke([real_allocator] {
        return real_allocator->availableBlocksNum();
    }));
    ON_CALL(*allocator, totalBlocksNum()).WillByDefault(testing::Invoke([real_allocator] {
        return real_allocator->totalBlocksNum();
    }));
    ON_CALL(*allocator, maxAvailableTokensNum()).WillByDefault(testing::Invoke([real_allocator] {
        return real_allocator->maxAvailableTokensNum();
    }));
    ON_CALL(*allocator, seqSizePerBlock()).WillByDefault(testing::Invoke([real_allocator] {
        return real_allocator->seqSizePerBlock();
    }));
    ON_CALL(*allocator, getNeedBlocks(testing::_))
        .WillByDefault(
            testing::Invoke([real_allocator](const MallocInfo& info) { return real_allocator->getNeedBlocks(info); }));
    ON_CALL(*allocator, singleBatchNeedBlocks(testing::_, testing::_, testing::_))
        .WillByDefault(
            testing::Invoke([real_allocator](const BatchKVCacheResourcePtr& resource, int seq_len, int reserve_step) {
                return real_allocator->singleBatchNeedBlocks(resource, seq_len, reserve_step);
            }));
    ON_CALL(*allocator, estimatePeakNeedBlocks(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillByDefault(testing::Invoke([real_allocator](const KVCacheResource& resource,
                                                        int                    seq_len,
                                                        int                    remaining_tokens,
                                                        int                    reserve_step,
                                                        bool                   reuse_cache,
                                                        LinearCachePolicy      linear_cache_policy) {
            return real_allocator->estimatePeakNeedBlocks(
                resource, seq_len, remaining_tokens, reserve_step, reuse_cache, linear_cache_policy);
        }));
    ON_CALL(*allocator,
            estimateInitialBatchPeakNeedBlocks(
                testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillByDefault(testing::Invoke([real_allocator](int               seq_len,
                                                        int               common_seq_len,
                                                        int               remaining_tokens,
                                                        int               reserve_step,
                                                        bool              reuse_cache,
                                                        int               target_batch_size,
                                                        LinearCachePolicy linear_cache_policy) {
            return real_allocator->estimateInitialBatchPeakNeedBlocks(seq_len,
                                                                      common_seq_len,
                                                                      remaining_tokens,
                                                                      reserve_step,
                                                                      reuse_cache,
                                                                      target_batch_size,
                                                                      linear_cache_policy);
        }));
    return allocator;
}

}  // namespace rtp_llm::test
