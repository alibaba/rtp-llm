#pragma once

#include <array>
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

enum class BlockRefType : uint8_t {
    REQUEST = 0,
    CONNECTOR,
    BLOCK_CACHE,
    COUNT,
};

constexpr size_t kBlockRefTypeCount = static_cast<size_t>(BlockRefType::COUNT);

struct BlockPoolConfigBase {
    virtual ~BlockPoolConfigBase() = default;

    BlockPoolType pool_type{BlockPoolType::DEVICE};
    std::string   pool_name;
    size_t        physical_block_count{0};
};

// IBlockPool is the abstract lifecycle base class shared by the device / host / disk
// block pool implementations under rtp_llm::block_tree_cache. It owns the
// BlockPoolConfigBase and implements every non-virtual malloc/free/refcount/metrics
// API so that subclasses cannot intercept or override lifecycle behavior. Subclasses
// only add medium-specific init() (calling the protected markInitialized()) and
// medium-specific accessors.
class IBlockPool {
public:
    virtual ~IBlockPool() = default;

    const std::string&  poolName() const;
    virtual std::string debugString() const;
    virtual size_t      blockSizeBytes() const = 0;

    std::optional<BlockIdxType> malloc();
    std::optional<BlockIdList>  malloc(size_t n);

    void free(BlockIdxType block);
    void free(const BlockIdList& blocks);

    void incRef(BlockIdxType block, BlockRefType ref_type);
    void incRef(const BlockIdList& blocks, BlockRefType ref_type);

    // Release one holder: decrement one reference and, only when the refcount reaches 0,
    // return the block's capacity to the free list. Category releases must use this rather
    // than free() directly. Requires refcount > 0.
    void     decRef(BlockIdxType block, BlockRefType ref_type);
    void     decRef(const BlockIdList& blocks, BlockRefType ref_type);
    uint32_t refCount(BlockIdxType block) const;
    size_t   totalRefCount(BlockRefType ref_type) const;

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
    bool          validBlockNoLock(BlockIdxType block) const;
    void          checkInitializedNoLock() const;
    void          checkAllocatedNoLock(BlockIdxType block) const;
    void          checkUniqueBlocksNoLock(const BlockIdList& blocks) const;
    static size_t refTypeIndex(BlockRefType ref_type);

    size_t       availableFreeBlocksNoLock() const;
    void         refillAscendingFreeBlocksNoLock();
    BlockIdxType popFreeBlockNoLock();
    void         pushFreeBlockNoLock(BlockIdxType block);

    // Single-block primitives shared by decRef/free (one source of truth for refcount and
    // metrics). Callers must hold mutex_ and have validated the block is allocated (and, for
    // decRefOneNoLock, that refcount > 0).
    void decRefOneNoLock(BlockIdxType block, size_t ref_type_index);
    void freeAllocatedBlockNoLock(BlockIdxType block);

private:
    std::shared_ptr<const BlockPoolConfigBase>            config_;
    mutable std::mutex                                    mutex_;
    bool                                                  initialized_{false};
    std::vector<uint8_t>                                  allocated_;
    std::vector<uint32_t>                                 refcounts_;
    std::array<std::vector<uint32_t>, kBlockRefTypeCount> refcounts_by_type_;
    std::array<size_t, kBlockRefTypeCount>                total_ref_counts_{};
    std::vector<BlockIdxType>                             free_blocks_;
    std::vector<BlockIdxType>                             released_blocks_;
    size_t                                                free_head_{0};
    size_t                                                used_blocks_num_{0};
    size_t                                                unreferenced_blocks_num_{0};
    size_t                                                tree_cached_blocks_num_{0};
    size_t                                                active_tree_cached_blocks_num_{0};
};

}  // namespace rtp_llm
