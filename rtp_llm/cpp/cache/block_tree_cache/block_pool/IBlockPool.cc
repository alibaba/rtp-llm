#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"

#include <algorithm>
#include <cassert>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace rtp_llm {

IBlockPool::IBlockPool(std::shared_ptr<const BlockPoolConfigBase> config): config_(std::move(config)) {
    if (!config_ || config_->physical_block_count < (config_->growable ? 1 : 2)
        || config_->physical_block_count > static_cast<size_t>(std::numeric_limits<BlockIdxType>::max())) {
        throw std::invalid_argument("Block pool capacity must leave room for an allocatable slot and the null slot");
    }
    slots_.assign(config_->physical_block_count, SlotState::FREE);
    generations_.assign(config_->physical_block_count, 0);
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

void IBlockPool::setCapacityChangeCallback(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_change_callback_ = std::move(callback);
}

void IBlockPool::setBlockRetirementCallback(std::function<void(BlockPoolPage)> callback) {
    std::lock_guard lock(mutex_);
    if (!callback || config_->growable || retirement_callback_
        || (config_->pool_type != BlockPoolType::HOST && config_->pool_type != BlockPoolType::DISK)
        || free_head_ != 0) {
        throw std::invalid_argument("Block retirement requires a fresh fixed Host or Disk pool and a callback");
    }
    reserveAllReleasedBlocksNoLock();
    retirement_callback_ = std::move(callback);
}

bool IBlockPool::onLastTreeRefNoLock(BlockIdxType block) {
    if (!retirement_callback_) {
        return true;
    }
    assert(slots_[block] == SlotState::ALLOCATED && !tree_refcounts_[block]);
    slots_[block] = SlotState::RETIRED;
    ++retired_blocks_num_;
    try {
        retirement_callback_({this, block, generations_[block]});
    } catch (...) {
        // Reference counts have already changed. The owner must accept the
        // retired slot without throwing, including on a final FROZEN release.
        std::terminate();
    }
    return false;
}

void IBlockPool::releaseRetiredBlock(BlockPoolPage page) {
    std::function<void()> notify;
    {
        std::lock_guard lock(mutex_);
        const auto block = page.slot;
        if (!retirement_callback_ || page.pool != this || !validBlockNoLock(block)
            || slots_[block] != SlotState::RETIRED || generations_[block] != page.generation
            || tree_refcounts_[block] != 0) {
            throw std::invalid_argument("Invalid retired block acknowledgment for pool " + poolName());
        }
        notify = capacity_change_callback_;
        const bool was_available = isAvailableNoLock(block);
        freeAllocatedBlockNoLock(block);
        --retired_blocks_num_;
        updateAvailableBlocksNumNoLock(block, was_available);
    }
    if (notify) {
        try {
            notify();
        } catch (...) {
            // An observer failure cannot undo the slot's acknowledged release.
        }
    }
}

const std::string& IBlockPool::poolName() const {
    return config_->pool_name;
}

std::string IBlockPool::debugString() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream          oss;
    const size_t free_blocks = availableFreeBlocksNoLock();
    const size_t used_blocks = totalBlocksNumNoLock() - free_blocks;
    oss << "IBlockPool{name=" << config_->pool_name << ", total=" << totalBlocksNumNoLock() << ", used=" << used_blocks
        << ", free=" << free_blocks << ", available=" << available_blocks_num_
        << ", cache_ref_blocks=" << tree_referenced_block_counts_[treeRefTypeIndex(BlockTreeRefType::CACHE)]
        << ", load_ref_blocks=" << tree_referenced_block_counts_[treeRefTypeIndex(BlockTreeRefType::LOAD)]
        << ", eviction_ref_blocks="
        << tree_referenced_block_counts_[treeRefTypeIndex(BlockTreeRefType::EVICT)]
        << ", frozen_ref_blocks=" << tree_referenced_block_counts_[treeRefTypeIndex(BlockTreeRefType::FROZEN)] << "}";
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
    return mallocNoLock(n);
}

void IBlockPool::appendCapacity(size_t additional) {
    if (!config_->growable) {
        throw std::logic_error("Cannot grow fixed block pool " + poolName());
    }
    if (!additional) {
        return;
    }
    std::function<void()> notify;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        checkInitializedNoLock();
        const auto before = slots_.size();
        const auto max = static_cast<size_t>(std::numeric_limits<BlockIdxType>::max());
        if (additional > max - before) {
            throw std::overflow_error("Block pool index range exhausted");
        }
        const auto after = before + additional;
        // Allocate while all visible sizes and allocation lists still agree.
        const auto reserve = [=](auto& slots) {
            if (slots.capacity() < after) {
                slots.reserve(std::max(after, std::min(max, slots.capacity() * 2)));
            }
        };
        reserve(slots_);
        reserve(generations_);
        reserve(tree_refcounts_);
        for (auto& counts : tree_refcounts_by_type_) {
            reserve(counts);
        }
        reserve(free_blocks_);
        notify = capacity_change_callback_;
        slots_.resize(after, SlotState::FREE);
        generations_.resize(after);
        tree_refcounts_.resize(after);
        for (auto& counts : tree_refcounts_by_type_) {
            counts.resize(after);
        }
        for (size_t block = before; block < after; ++block) {
            free_blocks_.push_back(static_cast<BlockIdxType>(block));
        }
        available_blocks_num_ += additional;
    }
    if (notify) {
        try {
            notify();
        } catch (...) {}
    }
}

std::optional<BlockIdList> IBlockPool::mallocNoLock(size_t n) {
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
        slots_[block]          = SlotState::ALLOCATED;
        ++generations_[block];
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
    if (ref_type == BlockTreeRefType::FROZEN) {
        throw std::invalid_argument("Frozen references require freezeBlock");
    }
    const size_t ref_type_index = treeRefTypeIndex(ref_type);
    mutateAllocatedBlocks(
        blocks,
        [](BlockIdxType) {},
        [this, ref_type_index](BlockIdxType block) { incTreeRefNoLock(block, ref_type_index); });
}

void IBlockPool::incTreeRefNoLock(BlockIdxType block, size_t ref_type_index) {
    if (tree_refcounts_[block] == std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("Too many references to a block of pool " + poolName());
    }
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
}

void IBlockPool::incTreeRefLocked(BlockIdxType block, BlockTreeRefType type) {
    checkAllocatedNoLock(block);
    incTreeRefNoLock(block, treeRefTypeIndex(type));
}

void IBlockPool::freezeBlock(BlockIdxType block) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    checkAllocatedNoLock(block);
    incTreeRefNoLock(block, treeRefTypeIndex(BlockTreeRefType::FROZEN));
}

bool IBlockPool::isFrozen(BlockIdxType block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    if (!validBlockNoLock(block)) {
        throw std::out_of_range("Invalid frozen block index in pool " + poolName());
    }
    return treeRefCountNoLock(block, BlockTreeRefType::FROZEN) != 0;
}

std::function<void()> IBlockPool::prepareFrozenReferenceRelease(BlockIdList blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    checkUniqueBlocksNoLock(blocks);
    for (const auto block : blocks) {
        checkAllocatedNoLock(block);
    }
    // All live allocations can eventually be returned without growing this
    // vector, including when ordinary ownership disappears before unfreeze.
    reserveAllReleasedBlocksNoLock();
    auto notify = capacity_change_callback_;
    return [this, blocks = std::move(blocks), notify = std::move(notify)]() noexcept {
        releaseFrozenReferences(blocks, notify);
    };
}

void IBlockPool::releaseFrozenReferences(const BlockIdList& blocks, const std::function<void()>& notify) noexcept {
    bool capacity_grew = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  type = treeRefTypeIndex(BlockTreeRefType::FROZEN);
        // Preparation already checked uniqueness. Validate the whole balance
        // before changing any block; a violated ownership contract is fatal.
        for (const auto block : blocks) {
            checkAllocatedNoLock(block);
            if (tree_refcounts_by_type_[type][block] == 0) {
                std::terminate();
            }
        }
        const auto free_before      = availableFreeBlocksNoLock();
        const auto available_before = available_blocks_num_;
        for (const auto block : blocks) {
            const bool was_active    = isActiveNoLock(block);
            const bool was_available = isAvailableNoLock(block);
            if (--tree_refcounts_by_type_[type][block] == 0) {
                --tree_referenced_block_counts_[type];
            }
            if (--tree_refcounts_[block] == 0 && onLastTreeRefNoLock(block)) {
                freeAllocatedBlockNoLock(block);
            }
            updateActiveBlocksNumNoLock(block, was_active);
            updateAvailableBlocksNumNoLock(block, was_available);
        }
        capacity_grew = availableFreeBlocksNoLock() > free_before || available_blocks_num_ > available_before;
    }
    if (capacity_grew && notify) {
        try {
            notify();
        } catch (...) {}
    }
}

size_t IBlockPool::decTreeRef(BlockIdxType block, BlockTreeRefType ref_type) {
    return decTreeRef(BlockIdList{block}, ref_type);
}

size_t IBlockPool::decTreeRef(const BlockIdList& blocks, BlockTreeRefType ref_type) {
    if (ref_type == BlockTreeRefType::FROZEN) {
        throw std::invalid_argument("Frozen references require the prepared release");
    }
    const size_t ref_type_index = treeRefTypeIndex(ref_type);
    size_t freed = 0;
    mutateAllocatedBlocks(
        blocks,
        [this, ref_type_index](BlockIdxType block) {
            if (tree_refcounts_by_type_[ref_type_index][block] == 0) {
                throw std::underflow_error("No tree reference to block " + std::to_string(block)
                                           + " in pool " + poolName());
            }
        },
        [this, ref_type_index, &freed](BlockIdxType block) {
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
                ++freed;
            }
            updateActiveBlocksNumNoLock(block, was_active);
            updateAvailableBlocksNumNoLock(block, was_available);
        });
    return freed;
}

uint32_t IBlockPool::treeRefCount(BlockIdxType block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    checkAllocatedNoLock(block, /*allow_retired=*/true);
    return tree_refcounts_[block];
}

bool IBlockPool::canReclaimCacheBlock(BlockIdxType block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    checkInitializedNoLock();
    checkAllocatedNoLock(block, /*allow_retired=*/true);
    return tree_refcounts_[block] == 1 && treeRefCountNoLock(block, BlockTreeRefType::CACHE) == 1
           && !hasExternalRefNoLock(block);
}

size_t IBlockPool::unpinnedCacheReferences(BlockIdxType block) const {
    std::lock_guard lock(mutex_);
    if (!validBlockNoLock(block) || slots_[block] != SlotState::ALLOCATED || isActiveNoLock(block)) {
        return 0;
    }
    return treeRefCountNoLock(block, BlockTreeRefType::CACHE);
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
    if (config_->growable) {
        std::lock_guard<std::mutex> lock(mutex_);
        return validBlockNoLock(block);
    }
    return validBlockNoLock(block);
}

bool IBlockPool::isAllocated(BlockIdxType block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!validBlockNoLock(block)) {
        return false;
    }
    return slots_[block] != SlotState::FREE;
}

std::optional<BlockPoolPage> IBlockPool::pageIdentity(BlockIdxType block) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pageIdentityNoLock(block);
}

std::optional<BlockPoolPage> IBlockPool::pageIdentityNoLock(BlockIdxType block) const {
    if (!validBlockNoLock(block) || slots_[block] == SlotState::FREE) {
        return std::nullopt;
    }
    return BlockPoolPage{this, block, generations_[block]};
}

bool IBlockPool::isCurrentPage(BlockPoolPage page) const {
    if (page.pool != this) {
        return false;
    }
    const auto current = pageIdentity(page.slot);
    return current && current->generation == page.generation;
}

IBlockPool::ReferenceStats IBlockPool::referenceStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {treeReferenceStatsNoLock()};
}

size_t IBlockPool::totalBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return totalBlocksNumNoLock();
}

size_t IBlockPool::freeBlocksNum() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return availableFreeBlocksNoLock();
}

size_t IBlockPool::retiredBlocksNum() const {
    std::lock_guard lock(mutex_);
    return retired_blocks_num_;
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
    return block > 0 && static_cast<size_t>(block) < slots_.size() && !isNullBlockIdx(block);
}

void IBlockPool::checkInitializedNoLock() const {
    if (!initialized_) {
        throw std::logic_error("Block pool is not initialized: " + poolName());
    }
}

void IBlockPool::checkAllocatedNoLock(BlockIdxType block, bool allow_retired) const {
    if (!validBlockNoLock(block)
        || (allow_retired ? slots_[block] == SlotState::FREE : slots_[block] != SlotState::ALLOCATED)) {
        throw std::runtime_error("Block " + std::to_string(block) + " is not allocated in pool " + poolName());
    }
}

void IBlockPool::checkUniqueBlocksNoLock(const BlockIdList& blocks) const {
    BlockIdList sorted_blocks(blocks.begin(), blocks.end());
    std::sort(sorted_blocks.begin(), sorted_blocks.end());
    if (std::adjacent_find(sorted_blocks.begin(), sorted_blocks.end()) != sorted_blocks.end()) {
        throw std::invalid_argument("Duplicate block ID in batch operation for pool " + poolName());
    }
}

size_t IBlockPool::treeRefTypeIndex(BlockTreeRefType ref_type) {
    const size_t ref_type_index = static_cast<size_t>(ref_type);
    if (ref_type_index >= kBlockTreeRefTypeCount) {
        throw std::invalid_argument("Invalid tree reference type");
    }
    return ref_type_index;
}

size_t IBlockPool::totalBlocksNumNoLock() const {
    return slots_.size() - 1;
}

size_t IBlockPool::availableFreeBlocksNoLock() const {
    assert(free_head_ <= free_blocks_.size() && "Block pool free cursor is invalid");
    const size_t available_blocks = free_blocks_.size() - free_head_ + released_blocks_.size();
    assert(available_blocks <= totalBlocksNumNoLock() && "Block pool free lists overlap");
    return available_blocks;
}

IBlockPool::TreeReferenceStats IBlockPool::treeReferenceStatsNoLock() const {
    checkInitializedNoLock();
    const size_t free_blocks = availableFreeBlocksNoLock();
    return {totalBlocksNumNoLock(),
            free_blocks,
            available_blocks_num_ - free_blocks,
            active_blocks_num_,
            tree_referenced_block_counts_};
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
    assert(free_head_ < free_blocks_.size() && "Empty block pool free list after capacity check");
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
    if (slots_[block] == SlotState::FREE) {
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
    slots_[block]          = SlotState::FREE;
    tree_refcounts_[block] = 0;
    pushFreeBlockNoLock(block);
}

}  // namespace rtp_llm
