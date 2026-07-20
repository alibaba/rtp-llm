#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/IBlockPool.h"

namespace rtp_llm {

class CacheStore;
class KVCacheGroup;

// Non-owning tier adapter over the declarative cache layer's BlockPool. It never
// allocates a second device backing. Its references are deliberately isolated in
// BlockPool's block-cache category, leaving request and connector ownership intact.
class DeviceBlockPool;
using DeviceBlockPoolPtr = std::shared_ptr<DeviceBlockPool>;

class DeviceBlockPool final: public IBlockPool {
public:
    explicit DeviceBlockPool(BlockPoolPtr block_pool, std::shared_ptr<KVCacheGroup> address_view = nullptr);
    ~DeviceBlockPool() override = default;

    bool init();

    const std::string& poolName() const override;
    std::string        debugString() const override;

    std::optional<BlockIdxType> malloc() override;
    std::optional<BlockIdList>  malloc(size_t n) override;
    void                        free(BlockIdxType block) override;
    void                        free(const BlockIdList& blocks) override;
    void                        incRef(BlockIdxType block) override;
    void                        incRef(const BlockIdList& blocks) override;
    void                        decRef(BlockIdxType block) override;
    void                        decRef(const BlockIdList& blocks) override;
    uint32_t                    refCount(BlockIdxType block) const override;
    bool                        validBlock(BlockIdxType block) const override;
    bool                        isAllocated(BlockIdxType block) const override;
    bool                        hasRequestHold(BlockIdxType block) const;
    bool                        hasExternalHolder(BlockIdxType block) const;
    bool                        isCacheOnly(BlockIdxType block) const;
    size_t                      totalBlocksNum() const override;
    size_t                      freeBlocksNum() const override;
    size_t                      usedBlocksNum() const override;
    size_t                      unreferencedBlocksNum() const override;
    size_t                      treeCachedBlocksNum() const override;
    size_t                      activeTreeCachedBlocksNum() const override;

    int                        deviceIndex() const;
    MemoryType                 where() const;
    std::vector<torch::Tensor> allLayerCacheBase() const;
    std::vector<torch::Tensor> allLayerScaleCacheBase() const;

    void    regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store = nullptr);
    void    deregUserMr();
    int64_t getMrCostTimeMs() const;

    BlockAddrInfo          convertIndexToAddr(int layer_id, BlockIdxType block) const;
    std::vector<BlockInfo> convertIndexToBuffer(int layer_id, BlockIdxType block) const;
    std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, BlockIdxType block, int partition_count, int partition_id) const;

    void*  getBaseAddress() const;
    size_t getTotalSizeBytes() const;

    BlockPoolPtr backingPool() const {
        return block_pool_;
    }

    std::shared_ptr<KVCacheGroup> addressView() const {
        return address_view_;
    }

private:
    void   checkInitializedNoLock() const;
    void   checkValidNoLock(BlockIdxType block) const;
    void   checkUniqueNoLock(const BlockIdList& blocks) const;
    bool   hasExternalHolderNoLock(BlockIdxType block, uint32_t tree_refs) const;
    size_t activeTreeCachedBlocksNumNoLock() const;

private:
    BlockPoolPtr                               block_pool_;
    std::shared_ptr<KVCacheGroup>              address_view_;
    mutable std::mutex                         mutex_;
    bool                                       initialized_{false};
    std::unordered_map<BlockIdxType, uint32_t> tree_refs_;
};

}  // namespace rtp_llm
