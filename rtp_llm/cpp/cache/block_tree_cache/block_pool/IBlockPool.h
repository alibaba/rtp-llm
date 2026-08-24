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

enum class BlockTreeRefType : uint8_t {
    CACHE = 0,
    LOAD,
    EVICTION,
    STORE,
    COUNT,
};

constexpr size_t kBlockTreeRefTypeCount = static_cast<size_t>(BlockTreeRefType::COUNT);

struct BlockPoolConfigBase {
    virtual ~BlockPoolConfigBase() = default;

    BlockPoolType pool_type{BlockPoolType::DEVICE};
    std::string   pool_name;
    size_t        physical_block_count{0};
};

// IBlockPool owns allocation and Tree-internal lifetime shared by Device, Host, and Disk.
// DeviceBlockPool adds request-like outer ownership through the protected Tree edge hooks.
class IBlockPool {
public:
    virtual ~IBlockPool() = default;

    const std::string&  poolName() const;
    virtual std::string debugString() const;
    virtual size_t      blockSizeBytes() const = 0;

    std::optional<BlockIdxType> malloc();
    std::optional<BlockIdList>  malloc(size_t n);

    void     incTreeRef(BlockIdxType block, BlockTreeRefType ref_type);
    void     incTreeRef(const BlockIdList& blocks, BlockTreeRefType ref_type);
    void     decTreeRef(BlockIdxType block, BlockTreeRefType ref_type);
    void     decTreeRef(const BlockIdList& blocks, BlockTreeRefType ref_type);
    uint32_t treeRefCount(BlockIdxType block) const;

    // Number of distinct blocks carrying at least one tree reference of this type.
    size_t referencedBlocksNum(BlockTreeRefType ref_type) const;

    bool validBlock(BlockIdxType block) const;
    bool isAllocated(BlockIdxType block) const;

    size_t totalBlocksNum() const;
    size_t freeBlocksNum() const;
    size_t usedBlocksNum() const;
    size_t activeBlocksNum() const;

protected:
    explicit IBlockPool(std::shared_ptr<const BlockPoolConfigBase> config);

    void markInitialized();
    bool initialized() const;

    virtual void onFirstTreeRefNoLock(BlockIdxType) {}
    virtual bool onLastTreeRefNoLock(BlockIdxType) {
        return true;
    }
    virtual void onCacheRefChangedNoLock(BlockIdxType, bool) {}
    virtual bool hasExternalRefNoLock(BlockIdxType) const {
        return false;
    }

    template<typename Validator, typename Mutator>
    void mutateAllocatedBlocks(const BlockIdList& blocks, Validator&& validate, Mutator&& mutate) {
        std::lock_guard<std::mutex> lock(mutex_);
        checkInitializedNoLock();
        if (blocks.empty()) {
            return;
        }
        checkUniqueBlocksNoLock(blocks);
        for (const auto block : blocks) {
            checkAllocatedNoLock(block);
            validate(block);
        }
        for (const auto block : blocks) {
            mutate(block);
        }
    }

    void     checkInitializedNoLock() const;
    void     checkAllocatedNoLock(BlockIdxType block) const;
    uint32_t treeRefCountNoLock(BlockIdxType block) const;
    uint32_t treeRefCountNoLock(BlockIdxType block, BlockTreeRefType ref_type) const;
    bool     isActiveNoLock(BlockIdxType block) const;
    void     updateActiveBlocksNumNoLock(BlockIdxType block, bool was_active);
    void     freeAllocatedBlockNoLock(BlockIdxType block);

    mutable std::mutex mutex_;

    template<typename ConfigT>
    const ConfigT& configAs(BlockPoolType expected_type) const {
        RTP_LLM_CHECK(config_->pool_type == expected_type);
        return static_cast<const ConfigT&>(*config_);
    }

private:
    bool          validBlockNoLock(BlockIdxType block) const;
    void          checkUniqueBlocksNoLock(const BlockIdList& blocks) const;
    static size_t treeRefTypeIndex(BlockTreeRefType ref_type);

    size_t       totalBlocksNumNoLock() const;
    size_t       availableFreeBlocksNoLock() const;
    void         refillAscendingFreeBlocksNoLock();
    BlockIdxType popFreeBlockNoLock();
    void         pushFreeBlockNoLock(BlockIdxType block);

private:
    std::shared_ptr<const BlockPoolConfigBase>                config_;
    bool                                                      initialized_{false};
    std::vector<uint8_t>                                      allocated_;
    std::vector<uint32_t>                                     tree_refcounts_;
    std::array<std::vector<uint32_t>, kBlockTreeRefTypeCount> tree_refcounts_by_type_;
    std::array<size_t, kBlockTreeRefTypeCount>                tree_referenced_block_counts_{};
    std::vector<BlockIdxType>                                 free_blocks_;
    std::vector<BlockIdxType>                                 released_blocks_;
    size_t                                                    free_head_{0};
    size_t                                                    active_blocks_num_{0};
};

}  // namespace rtp_llm
