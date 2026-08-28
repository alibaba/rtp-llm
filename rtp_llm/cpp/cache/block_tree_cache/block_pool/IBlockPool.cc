#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"

#include <algorithm>
#include <cassert>
#include <sstream>

namespace rtp_llm {

IBlockPool::IBlockPool(std::shared_ptr<const BlockPoolConfigBase> config): config_(std::move(config)) {
    RTP_LLM_CHECK(config_ != nullptr);
    RTP_LLM_CHECK(config_->physical_block_count > 1);
    allocated_.assign(config_->physical_block_count, 0);
    tree_refcounts_.assign(config_->physical_block_count, 0);
    for (std::vector<uint32_t>& typed_refcounts : tree_refcounts_by_type_) {
        typed_refcounts.assign(config_->physical_block_count, 0);
    }
    free_blocks_.reserve(config_->physical_block_count - 1);
    for (BlockIdxType block = 1; block < static_cast<BlockIdxType>(config_->physical_block_count); ++block) {
        free_blocks_.push_back(block);
    }
    available_blocks_num_ = free_blocks_.size();
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
        << ", free=" << free_blocks << ", available=" << available_blocks_num_
        << ", cache_ref_blocks=" << tree_referenced_block_counts_[treeRefTypeIndex(BlockTreeRefType::CACHE)]
        << ", load_ref_blocks=" << tree_referenced_block_counts_[treeRefTypeIndex(BlockTreeRefType::LOAD)]
        << ", eviction_ref_blocks=" << tree_referenced_block_counts_[treeRefTypeIndex(BlockTreeRefType::EVICTION)]
        << ", store_ref_blocks=" << tree_referenced_block_counts_[treeRefTypeIndex(BlockTreeRefType::STORE)] << "}";
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
        const bool was_available = isAvailableNoLock(block);
        assert(was_available);
        allocated_[block]      = 1;
        tree_refcounts_[block] = 0;
        for (std::vector<uint32_t>& typed_refcounts : tree_refcounts_by_type_) {
            typed_refcounts[block] = 0;
        }
        updateAvailableBlocksNumNoLock(block, was_available);
    }
    return result;
}

void IBlockPool::incTreeRef(BlockIdxType block, BlockTreeRefType ref_type) {
    incTreeRef(BlockIdList{block}, ref_type);
}

void IBlockPool::incTreeRef(const BlockIdList& blocks, BlockTreeRefType ref_type) {
    const size_t ref_type_index = treeRefTypeIndex(ref_type);
    mutateAllocatedBlocks(
        blocks,
        [](BlockIdxType) {},
        [this, ref_type_index](BlockIdxType block) {
            const bool was_active    = isActiveNoLock(block);
            const bool was_available = isAvailableNoLock(block);
            if (tree_refcounts_[block] == 0) {
                onFirstTreeRefNoLock(block);
            }
            ++tree_refcounts_[block];

            uint32_t& typed_refcount = tree_refcounts_by_type_[ref_type_index][block];
            if (typed_refcount == 0) {
                ++tree_referenced_block_counts_[ref_type_index];
            }
            ++typed_refcount;
            updateActiveBlocksNumNoLock(block, was_active);
            updateAvailableBlocksNumNoLock(block, was_available);
        });
}

void IBlockPool::decTreeRef(BlockIdxType block, BlockTreeRefType ref_type) {
    decTreeRef(BlockIdList{block}, ref_type);
}

void IBlockPool::decTreeRef(const BlockIdList& blocks, BlockTreeRefType ref_type) {
    const size_t ref_type_index = treeRefTypeIndex(ref_type);
    mutateAllocatedBlocks(
        blocks,
        [this, ref_type_index](BlockIdxType block) {
            RTP_LLM_CHECK_WITH_INFO(tree_refcounts_by_type_[ref_type_index][block] > 0,
                                    "cannot decTreeRef block [%d] of pool [%s] with ref type [%zu] at 0",
                                    block,
                                    poolName().c_str(),
                                    ref_type_index);
        },
        [this, ref_type_index](BlockIdxType block) {
            const bool was_active     = isActiveNoLock(block);
            const bool was_available  = isAvailableNoLock(block);
            uint32_t&  typed_refcount = tree_refcounts_by_type_[ref_type_index][block];
            --typed_refcount;
            if (typed_refcount == 0) {
                assert(tree_referenced_block_counts_[ref_type_index] > 0);
                --tree_referenced_block_counts_[ref_type_index];
            }

            assert(tree_refcounts_[block] > 0);
            --tree_refcounts_[block];
            if (tree_refcounts_[block] == 0 && onLastTreeRefNoLock(block)) {
                freeAllocatedBlockNoLock(block);
            }
            updateActiveBlocksNumNoLock(block, was_active);
            updateAvailableBlocksNumNoLock(block, was_available);
        });
}

uint32_t IBlockPool::treeRefCount(BlockIdxType block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    checkAllocatedNoLock(block);
    return tree_refcounts_[block];
}

size_t IBlockPool::referencedBlocksNum(BlockTreeRefType ref_type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    return tree_referenced_block_counts_[treeRefTypeIndex(ref_type)];
}

size_t IBlockPool::activeBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    return active_blocks_num_;
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

size_t IBlockPool::availableBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    return available_blocks_num_;
}

size_t IBlockPool::usedBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return totalBlocksNumNoLock() - availableFreeBlocksNoLock();
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

size_t IBlockPool::treeRefTypeIndex(BlockTreeRefType ref_type) {
    const size_t ref_type_index = static_cast<size_t>(ref_type);
    RTP_LLM_CHECK_WITH_INFO(
        ref_type_index < kBlockTreeRefTypeCount, "invalid tree block ref type [%zu]", ref_type_index);
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

uint32_t IBlockPool::treeRefCountNoLock(BlockIdxType block) const {
    return tree_refcounts_[block];
}

uint32_t IBlockPool::treeRefCountNoLock(BlockIdxType block, BlockTreeRefType ref_type) const {
    return tree_refcounts_by_type_[treeRefTypeIndex(ref_type)][block];
}

bool IBlockPool::isActiveNoLock(BlockIdxType block) const {
    if (hasExternalRefNoLock(block)) {
        return true;
    }
    for (size_t ref_type_index = treeRefTypeIndex(BlockTreeRefType::LOAD); ref_type_index < kBlockTreeRefTypeCount;
         ++ref_type_index) {
        if (tree_refcounts_by_type_[ref_type_index][block] > 0) {
            return true;
        }
    }
    return false;
}

bool IBlockPool::isAvailableNoLock(BlockIdxType block) const {
    if (allocated_[block] == 0) {
        return true;
    }
    return tree_refcounts_by_type_[treeRefTypeIndex(BlockTreeRefType::CACHE)][block] > 0 && !isActiveNoLock(block);
}

void IBlockPool::updateActiveBlocksNumNoLock(BlockIdxType block, bool was_active) {
    const bool is_active = isActiveNoLock(block);
    if (was_active != is_active) {
        if (is_active) {
            ++active_blocks_num_;
        } else {
            assert(active_blocks_num_ > 0);
            --active_blocks_num_;
        }
    }
}

void IBlockPool::updateAvailableBlocksNumNoLock(BlockIdxType block, bool was_available) {
    const bool is_available = isAvailableNoLock(block);
    if (was_available != is_available) {
        if (is_available) {
            ++available_blocks_num_;
        } else {
            assert(available_blocks_num_ > 0);
            --available_blocks_num_;
        }
        assert(available_blocks_num_ <= totalBlocksNumNoLock());
    }
}

void IBlockPool::freeAllocatedBlockNoLock(BlockIdxType block) {
    assert(tree_refcounts_[block] == 0);
    for (size_t ref_type_index = 0; ref_type_index < kBlockTreeRefTypeCount; ++ref_type_index) {
        assert(tree_refcounts_by_type_[ref_type_index][block] == 0);
    }
    allocated_[block]      = 0;
    tree_refcounts_[block] = 0;
    pushFreeBlockNoLock(block);
}

}  // namespace rtp_llm
