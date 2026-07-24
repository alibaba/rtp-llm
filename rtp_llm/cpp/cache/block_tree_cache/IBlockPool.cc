#include "rtp_llm/cpp/cache/block_tree_cache/IBlockPool.h"

#include <algorithm>
#include <sstream>

namespace rtp_llm {

IBlockPool::IBlockPool(std::shared_ptr<const BlockPoolConfigBase> config): config_(std::move(config)) {
    RTP_LLM_CHECK(config_ != nullptr);
    RTP_LLM_CHECK(config_->physical_block_count > 1);
    allocated_.assign(config_->physical_block_count, 0);
    refcounts_.assign(config_->physical_block_count, 0);
    for (std::vector<uint32_t>& typed_refcounts : metric_refcounts_by_type_) {
        typed_refcounts.assign(config_->physical_block_count, 0);
    }
    free_blocks_.reserve(config_->physical_block_count - 1);
    for (BlockIdxType block = 1; block < static_cast<BlockIdxType>(config_->physical_block_count); ++block) {
        free_blocks_.push_back(block);
    }
}

const std::string& IBlockPool::poolName() const {
    return config_->pool_name;
}

std::string IBlockPool::debugString() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream          oss;
    const size_t                free_blocks = availableFreeBlocksNoLock();
    const size_t                used_blocks = totalBlocksNumNoLock() - free_blocks;
    oss << "IBlockPool{name=" << config_->pool_name << ", total=" << totalBlocksNumNoLock() << ", used=" << used_blocks
        << ", free=" << free_blocks << ", active_tree_cached=" << active_tree_cached_blocks_num_
        << ", request_refs=" << metric_total_ref_counts_[refTypeIndex(BlockRefType::REQUEST)]
        << ", connector_refs=" << metric_total_ref_counts_[refTypeIndex(BlockRefType::CONNECTOR)]
        << ", block_cache_refs=" << metric_total_ref_counts_[refTypeIndex(BlockRefType::BLOCK_CACHE)]
        << ", eviction_refs=" << metric_total_ref_counts_[refTypeIndex(BlockRefType::EVICTION)] << "}";
    return oss.str();
}

std::optional<BlockIdxType> IBlockPool::malloc() {
    auto blocks = malloc(1);
    if (!blocks.has_value()) {
        return std::nullopt;
    }
    return (*blocks)[0];
}

std::optional<BlockIdList> IBlockPool::malloc(size_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    if (n == 0) {
        return BlockIdList{};
    }
    if (availableFreeBlocksNoLock() < n) {
        return std::nullopt;
    }
    if (free_blocks_.size() - free_head_ < n) {
        refillAscendingFreeBlocksNoLock();
    }

    BlockIdList result;
    result.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        result.push_back(popFreeBlockNoLock());
    }

    for (const auto block : result) {
        allocated_[block] = 1;
        refcounts_[block] = 0;
        for (std::vector<uint32_t>& typed_refcounts : metric_refcounts_by_type_) {
            typed_refcounts[block] = 0;
        }
    }
    return result;
}

void IBlockPool::free(BlockIdxType block) {
    free(BlockIdList{block});
}

void IBlockPool::free(const BlockIdList& blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    if (blocks.empty()) {
        return;
    }
    checkUniqueBlocksNoLock(blocks);
    for (const auto block : blocks) {
        checkAllocatedNoLock(block);
        RTP_LLM_CHECK_WITH_INFO(refcounts_[block] <= 1,
                                "cannot free block [%d] of pool [%s] with refcount [%u]",
                                block,
                                config_->pool_name.c_str(),
                                refcounts_[block]);
    }

    for (const auto block : blocks) {
        freeAllocatedBlockNoLock(block);
    }
}

void IBlockPool::incRef(BlockIdxType block, BlockRefType ref_type) {
    incRef(BlockIdList{block}, ref_type);
}

void IBlockPool::incRef(const BlockIdList& blocks, BlockRefType ref_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    if (blocks.empty()) {
        return;
    }
    checkUniqueBlocksNoLock(blocks);
    for (const auto block : blocks) {
        checkAllocatedNoLock(block);
    }
    const size_t ref_type_index = refTypeIndex(ref_type);
    for (const auto block : blocks) {
        const uint32_t old_rc = refcounts_[block];
        const uint32_t new_rc = old_rc + 1;
        refcounts_[block]     = new_rc;
        metric_refcounts_by_type_[ref_type_index][block] += 1;
        metric_total_ref_counts_[ref_type_index] += 1;
        adjustActiveTreeCachedBlocksNoLock(old_rc, new_rc);
    }
}

void IBlockPool::decRef(BlockIdxType block, BlockRefType ref_type) {
    decRef(BlockIdList{block}, ref_type);
}

void IBlockPool::decRef(const BlockIdList& blocks, BlockRefType ref_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    if (blocks.empty()) {
        return;
    }
    checkUniqueBlocksNoLock(blocks);
    const size_t ref_type_index = refTypeIndex(ref_type);
    for (const auto block : blocks) {
        checkAllocatedNoLock(block);
        RTP_LLM_CHECK_WITH_INFO(refcounts_[block] > 0,
                                "cannot decRef block [%d] of pool [%s] with refcount 0",
                                block,
                                config_->pool_name.c_str());
    }
    for (const auto block : blocks) {
        decRefOneNoLock(block, ref_type_index);
        if (refcounts_[block] == 0) {
            freeAllocatedBlockNoLock(block);
        }
    }
}

uint32_t IBlockPool::refCount(BlockIdxType block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    checkAllocatedNoLock(block);
    return refcounts_[block];
}

size_t IBlockPool::totalRefCount(BlockRefType ref_type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    return metric_total_ref_counts_[refTypeIndex(ref_type)];
}

bool IBlockPool::validBlock(BlockIdxType block) const {
    return validBlockNoLock(block);
}

bool IBlockPool::isAllocated(BlockIdxType block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!validBlockNoLock(block)) {
        return false;
    }
    return allocated_[block] != 0;
}

size_t IBlockPool::totalBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return totalBlocksNumNoLock();
}

size_t IBlockPool::freeBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return availableFreeBlocksNoLock();
}

size_t IBlockPool::usedBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return totalBlocksNumNoLock() - availableFreeBlocksNoLock();
}

size_t IBlockPool::activeTreeCachedBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_tree_cached_blocks_num_;
}

size_t IBlockPool::TEST_unreferencedBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      count = 0;
    for (size_t block = 1; block < allocated_.size(); ++block) {
        if (allocated_[block] != 0 && refcounts_[block] == 0) {
            ++count;
        }
    }
    return count;
}

size_t IBlockPool::TEST_treeCachedBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      count = 0;
    for (size_t block = 1; block < allocated_.size(); ++block) {
        if (allocated_[block] != 0 && refcounts_[block] > 0) {
            ++count;
        }
    }
    return count;
}

void IBlockPool::markInitialized() {
    std::lock_guard<std::mutex> lock(mutex_);
    initialized_ = true;
}

bool IBlockPool::initialized() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initialized_;
}

bool IBlockPool::validBlockNoLock(BlockIdxType block) const {
    return block > 0 && static_cast<size_t>(block) < config_->physical_block_count && !isNullBlockIdx(block);
}

void IBlockPool::checkInitializedNoLock() const {
    RTP_LLM_CHECK_WITH_INFO(initialized_, "block pool [%s] is not initialized", config_->pool_name.c_str());
}

void IBlockPool::checkAllocatedNoLock(BlockIdxType block) const {
    RTP_LLM_CHECK_WITH_INFO(
        validBlockNoLock(block), "invalid block id [%d] for pool [%s]", block, config_->pool_name.c_str());
    RTP_LLM_CHECK_WITH_INFO(
        allocated_[block] != 0, "block [%d] of pool [%s] is not allocated", block, config_->pool_name.c_str());
}

void IBlockPool::checkUniqueBlocksNoLock(const BlockIdList& blocks) const {
    BlockIdList sorted_blocks(blocks.begin(), blocks.end());
    std::sort(sorted_blocks.begin(), sorted_blocks.end());
    RTP_LLM_CHECK_WITH_INFO(std::adjacent_find(sorted_blocks.begin(), sorted_blocks.end()) == sorted_blocks.end(),
                            "duplicate block id in batch operation for pool [%s]",
                            config_->pool_name.c_str());
}

size_t IBlockPool::refTypeIndex(BlockRefType ref_type) {
    const size_t ref_type_index = static_cast<size_t>(ref_type);
    RTP_LLM_CHECK_WITH_INFO(ref_type_index < kBlockRefTypeCount, "invalid block ref type [%zu]", ref_type_index);
    return ref_type_index;
}

size_t IBlockPool::totalBlocksNumNoLock() const {
    return config_->physical_block_count - 1;
}

size_t IBlockPool::availableFreeBlocksNoLock() const {
    RTP_LLM_CHECK(free_head_ <= free_blocks_.size());
    const size_t available_blocks = free_blocks_.size() - free_head_ + released_blocks_.size();
    RTP_LLM_CHECK(available_blocks <= totalBlocksNumNoLock());
    return available_blocks;
}

void IBlockPool::refillAscendingFreeBlocksNoLock() {
    BlockIdList merged(free_blocks_.begin() + free_head_, free_blocks_.end());
    merged.insert(merged.end(), released_blocks_.begin(), released_blocks_.end());
    std::sort(merged.begin(), merged.end());
    merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
    free_blocks_ = std::move(merged);
    free_head_   = 0;
    released_blocks_.clear();
}

BlockIdxType IBlockPool::popFreeBlockNoLock() {
    RTP_LLM_CHECK_WITH_INFO(
        free_head_ < free_blocks_.size(), "no free block available in pool [%s]", config_->pool_name.c_str());
    const BlockIdxType block = free_blocks_[free_head_];
    ++free_head_;
    return block;
}

void IBlockPool::pushFreeBlockNoLock(BlockIdxType block) {
    released_blocks_.push_back(block);
}

void IBlockPool::decRefOneNoLock(BlockIdxType block, size_t ref_type_index) {
    const uint32_t old_rc = refcounts_[block];
    const uint32_t new_rc = old_rc - 1;
    refcounts_[block]     = new_rc;
    adjustActiveTreeCachedBlocksNoLock(old_rc, new_rc);
    if (metric_refcounts_by_type_[ref_type_index][block] > 0 && metric_total_ref_counts_[ref_type_index] > 0) {
        metric_refcounts_by_type_[ref_type_index][block] -= 1;
        metric_total_ref_counts_[ref_type_index] -= 1;
    }
}

void IBlockPool::adjustActiveTreeCachedBlocksNoLock(uint32_t old_ref_count, uint32_t new_ref_count) {
    if (old_ref_count < 2 && new_ref_count >= 2) {
        active_tree_cached_blocks_num_ += 1;
    }
    if (old_ref_count >= 2 && new_ref_count < 2) {
        active_tree_cached_blocks_num_ -= 1;
    }
}

void IBlockPool::freeAllocatedBlockNoLock(BlockIdxType block) {
    for (size_t ref_type_index = 0; ref_type_index < kBlockRefTypeCount; ++ref_type_index) {
        const uint32_t metric_refcount                   = metric_refcounts_by_type_[ref_type_index][block];
        metric_total_ref_counts_[ref_type_index]         = metric_total_ref_counts_[ref_type_index] >= metric_refcount ?
                                                               metric_total_ref_counts_[ref_type_index] - metric_refcount :
                                                               0;
        metric_refcounts_by_type_[ref_type_index][block] = 0;
    }
    allocated_[block] = 0;
    refcounts_[block] = 0;
    pushFreeBlockNoLock(block);
}

}  // namespace rtp_llm
