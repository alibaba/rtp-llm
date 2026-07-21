#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

using BlockIdList = std::vector<BlockIdxType>;

enum class BlockPoolType {
    DEVICE,
    HOST,
    DISK,
};

struct BlockPoolConfigBase {
    virtual ~BlockPoolConfigBase() = default;

    BlockPoolType pool_type{BlockPoolType::DEVICE};
    std::string   pool_name;
    size_t        physical_block_count{0};
};

// IBlockPool owns the common lifecycle for device / host / disk pools. Lifecycle
// operations are intentionally non-virtual so every tier uses one atomic free-list
// and reference-count implementation; subclasses only add medium-specific backing.
class IBlockPool {
public:
    virtual ~IBlockPool() = default;

    const std::string&  poolName() const;
    virtual std::string debugString() const;

    std::optional<BlockIdxType> malloc();
    std::optional<BlockIdList>  malloc(size_t n);

    void free(BlockIdxType block);
    void free(const BlockIdList& blocks);

    void incRef(BlockIdxType block);
    void incRef(const BlockIdList& blocks);

    // Release one ownership reference and, only when the refcount reaches 0, return the
    // block's capacity to the free list. Requires refcount > 0.
    void     decRef(BlockIdxType block);
    void     decRef(const BlockIdList& blocks);
    uint32_t refCount(BlockIdxType block) const;

    bool validBlock(BlockIdxType block) const;
    bool isAllocated(BlockIdxType block) const;

    size_t totalBlocksNum() const;
    size_t freeBlocksNum() const;
    size_t usedBlocksNum() const;
    size_t unreferencedBlocksNum() const;
    size_t treeCachedBlocksNum() const;
    size_t activeTreeCachedBlocksNum() const;

protected:
    explicit IBlockPool(std::shared_ptr<const BlockPoolConfigBase> config);

    void markInitialized();
    bool initialized() const;

    template<typename ConfigT>
    const ConfigT& configAs(BlockPoolType expected_type) const {
        RTP_LLM_CHECK(config_->pool_type == expected_type);
        return static_cast<const ConfigT&>(*config_);
    }

private:
    bool validBlockNoLock(BlockIdxType block) const;
    void checkInitializedNoLock() const;
    void checkAllocatedNoLock(BlockIdxType block) const;
    void checkUniqueBlocksNoLock(const BlockIdList& blocks) const;

    size_t       availableFreeBlocksNoLock() const;
    void         refillAscendingFreeBlocksNoLock();
    BlockIdxType popFreeBlockNoLock();
    void         pushFreeBlockNoLock(BlockIdxType block);

    // Single-block primitives shared by decRef/free (one source of truth for refcount and
    // metrics). Callers must hold mutex_ and have validated the block is allocated (and, for
    // decRefOneNoLock, that refcount > 0).
    void decRefOneNoLock(BlockIdxType block);
    void freeAllocatedBlockNoLock(BlockIdxType block);

private:
    std::shared_ptr<const BlockPoolConfigBase> config_;
    mutable std::mutex                         mutex_;
    bool                                       initialized_{false};
    std::vector<uint8_t>                       allocated_;
    std::vector<uint32_t>                      refcounts_;
    std::vector<BlockIdxType>                  free_blocks_;
    std::vector<BlockIdxType>                  released_blocks_;
    size_t                                     free_head_{0};
    size_t                                     used_blocks_num_{0};
    size_t                                     unreferenced_blocks_num_{0};
    size_t                                     tree_cached_blocks_num_{0};
    size_t                                     active_tree_cached_blocks_num_{0};
};

}  // namespace rtp_llm
