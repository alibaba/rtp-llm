#pragma once

#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include "rtp_llm/cpp/cache/BlockRefCounter.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/cache_new/types.h"


// 结合 BlockCache 作为辅助判断， BlockCache 不持有 Block;

namespace rtp_llm {

class BlockPool {
public:
    struct KVCacheBuffer {
        rtp_llm::BufferPtr kv_blocks;
        rtp_llm::BufferPtr kv_scales;
    };

    struct BlockAddrInfo {
        void* k_addr       = nullptr;
        void* v_addr       = nullptr;
        void* k_scale_addr = nullptr;
        void* v_scale_addr = nullptr;
    };

    BlockPool(const CacheConfig& config, rtp_llm::DeviceBase* device, AllocationType atype = AllocationType::DEVICE);

    bool init();

    // size_t totalBlocks() const;
    // size_t freeBlockNums() const;

    std::vector<BufferPtr> layerCacheBase() const;

    std::vector<int> alloc(int num_blocks);
    void free(vector<int> block_ids);
    void reference(vector<int> block_ids);

    void regUserMr(size_t model_id);
    BlockAddrInfo convertIndexToAddr(int block_index, int layer_id) const;

private:
    void initKvCacheNormal();
    void initKvCacheMla();
    void initKvCacheScale();
    void initLinearCache();

    void incrBlockRefCounter(const std::vector<int>& blocks);
    void decrBlockRefCounter(const std::vector<int>& blocks);

private:
    CacheConfig config_;
    std::set<int> free_block_ids;
    std::unordered_map<int, BufferPtr> kv_addresses;        // global_layer_id -> kv cache addresses
    KVCacheBuffer kv_cache_;
    BlockRefCounter block_ref_counter_;
    rtp_llm::DeviceBase* device_;
    AllocationType atype_;
};

using BlockPoolPtr = std::shared_ptr<BlockPool>;

}  // namespace rtp_llm
