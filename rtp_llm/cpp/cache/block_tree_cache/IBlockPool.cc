#include "rtp_llm/cpp/cache/block_tree_cache/IBlockPool.h"

#include <algorithm>
#include <sstream>

namespace rtp_llm {

namespace {

// old_rc/new_rc always differ by exactly 1 (a single incRef/decRef/tryIncRefIf step),
// so it is enough to detect the two threshold crossings independently:
//   - refcount 0  <-> refcount >= 1 : unreferenced/tree-cached membership
//   - refcount 1  <-> refcount >= 2 : active-tree-cached membership
void adjustRefCountMetricsNoLock(uint32_t old_rc,
                                  uint32_t new_rc,
                                  size_t&  unreferenced_blocks_num,
                                  size_t&  tree_cached_blocks_num,
                                  size_t&  active_tree_cached_blocks_num) {
    if (old_rc == 0 && new_rc >= 1) {
        unreferenced_blocks_num -= 1;
        tree_cached_blocks_num += 1;
    }
    if (old_rc >= 1 && new_rc == 0) {
        unreferenced_blocks_num += 1;
        tree_cached_blocks_num -= 1;
    }
    if (old_rc < 2 && new_rc >= 2) {
        active_tree_cached_blocks_num += 1;
    }
    if (old_rc >= 2 && new_rc < 2) {
        active_tree_cached_blocks_num -= 1;
    }
}

}  // namespace

IBlockPool::IBlockPool(std::shared_ptr<const BlockPoolConfigBase> config): config_(std::move(config)) {
    RTP_LLM_CHECK(config_ != nullptr);
    RTP_LLM_CHECK(config_->physical_block_count > 1);
    allocated_.assign(config_->physical_block_count, 0);
    refcounts_.assign(config_->physical_block_count, 0);
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
    oss << "IBlockPool{name=" << config_->pool_name << ", total=" << totalBlocksNum()
        << ", used=" << used_blocks_num_ << ", free=" << availableFreeBlocksNoLock()
        << ", unreferenced=" << unreferenced_blocks_num_ << ", tree_cached=" << tree_cached_blocks_num_
        << ", active_tree_cached=" << active_tree_cached_blocks_num_ << "}";
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
    if (config_->free_block_order_policy == FreeBlockOrderPolicy::ASCENDING_ORDER
        && free_blocks_.size() - free_head_ < n) {
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
    }
    used_blocks_num_ += n;
    unreferenced_blocks_num_ += n;
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
        if (refcounts_[block] == 0) {
            unreferenced_blocks_num_ -= 1;
        } else {
            tree_cached_blocks_num_ -= 1;
        }
        allocated_[block] = 0;
        refcounts_[block] = 0;
        pushFreeBlockNoLock(block);
    }
    used_blocks_num_ -= blocks.size();
}

void IBlockPool::incRef(BlockIdxType block) {
    incRef(BlockIdList{block});
}

void IBlockPool::incRef(const BlockIdList& blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    if (blocks.empty()) {
        return;
    }
    checkUniqueBlocksNoLock(blocks);
    for (const auto block : blocks) {
        checkAllocatedNoLock(block);
    }
    for (const auto block : blocks) {
        const uint32_t old_rc = refcounts_[block];
        const uint32_t new_rc = old_rc + 1;
        refcounts_[block]     = new_rc;
        adjustRefCountMetricsNoLock(
            old_rc, new_rc, unreferenced_blocks_num_, tree_cached_blocks_num_, active_tree_cached_blocks_num_);
    }
}

bool IBlockPool::tryIncRefIf(BlockIdxType block, uint32_t expected_refcount) {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    checkAllocatedNoLock(block);
    if (refcounts_[block] != expected_refcount) {
        return false;
    }
    const uint32_t old_rc = refcounts_[block];
    const uint32_t new_rc = old_rc + 1;
    refcounts_[block]     = new_rc;
    adjustRefCountMetricsNoLock(
        old_rc, new_rc, unreferenced_blocks_num_, tree_cached_blocks_num_, active_tree_cached_blocks_num_);
    return true;
}

void IBlockPool::decRef(BlockIdxType block) {
    decRef(BlockIdList{block});
}

void IBlockPool::decRef(const BlockIdList& blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    if (blocks.empty()) {
        return;
    }
    checkUniqueBlocksNoLock(blocks);
    for (const auto block : blocks) {
        checkAllocatedNoLock(block);
        RTP_LLM_CHECK_WITH_INFO(refcounts_[block] > 0,
                                 "cannot decRef block [%d] of pool [%s] with refcount 0",
                                 block,
                                 config_->pool_name.c_str());
    }
    for (const auto block : blocks) {
        const uint32_t old_rc = refcounts_[block];
        const uint32_t new_rc = old_rc - 1;
        refcounts_[block]     = new_rc;
        adjustRefCountMetricsNoLock(
            old_rc, new_rc, unreferenced_blocks_num_, tree_cached_blocks_num_, active_tree_cached_blocks_num_);
    }
}

uint32_t IBlockPool::refCount(BlockIdxType block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    checkAllocatedNoLock(block);
    return refcounts_[block];
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
    return config_->physical_block_count - 1;
}

size_t IBlockPool::freeBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return availableFreeBlocksNoLock();
}

size_t IBlockPool::usedBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return used_blocks_num_;
}

size_t IBlockPool::unreferencedBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return unreferenced_blocks_num_;
}

size_t IBlockPool::treeCachedBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tree_cached_blocks_num_;
}

size_t IBlockPool::activeTreeCachedBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_tree_cached_blocks_num_;
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
    RTP_LLM_CHECK_WITH_INFO(allocated_[block] != 0,
                             "block [%d] of pool [%s] is not allocated",
                             block,
                             config_->pool_name.c_str());
}

void IBlockPool::checkUniqueBlocksNoLock(const BlockIdList& blocks) const {
    BlockIdList sorted_blocks(blocks.begin(), blocks.end());
    std::sort(sorted_blocks.begin(), sorted_blocks.end());
    RTP_LLM_CHECK_WITH_INFO(std::adjacent_find(sorted_blocks.begin(), sorted_blocks.end()) == sorted_blocks.end(),
                             "duplicate block id in batch operation for pool [%s]",
                             config_->pool_name.c_str());
}

size_t IBlockPool::availableFreeBlocksNoLock() const {
    return totalBlocksNum() - used_blocks_num_;
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
    if (config_->free_block_order_policy == FreeBlockOrderPolicy::ASCENDING_ORDER) {
        RTP_LLM_CHECK_WITH_INFO(
            free_head_ < free_blocks_.size(), "no free block available in pool [%s]", config_->pool_name.c_str());
        const BlockIdxType block = free_blocks_[free_head_];
        ++free_head_;
        return block;
    }
    RTP_LLM_CHECK_WITH_INFO(!free_blocks_.empty(), "no free block available in pool [%s]", config_->pool_name.c_str());
    const BlockIdxType block = free_blocks_.back();
    free_blocks_.pop_back();
    return block;
}

void IBlockPool::pushFreeBlockNoLock(BlockIdxType block) {
    if (config_->free_block_order_policy == FreeBlockOrderPolicy::ASCENDING_ORDER) {
        released_blocks_.push_back(block);
    } else {
        free_blocks_.push_back(block);
    }
}

}  // namespace rtp_llm
