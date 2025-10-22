#pragma once

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>

#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"
#include "rtp_llm/cpp/cache_new/types.h"

namespace rtp_llm {

class KVCacheAllocator {
public:
    bool init();
    // struct PoolUsage {
    //     size_t free_blocks_num;
    //     size_t available_blocks_num;
    //     size_t total_blocks_num;
    // }
    
    // return the number of blocks in the pool
    // std::vector<PoolUsage> poolSize() const;
    MallocResult malloc(const MallocInfo& malloc_info) {
        if malloc_info.enable_reuse_cache {
            return mallocWithCache(malloc_info);
        } else {
            return mallocSimple(malloc_info);
        }
    }

    FreeResult free(const FreeInfo& free_info);
    InsertResult insertIntoCache(const InsertInfo& insert_info);
    std::vector<BufferPtr> layerCacheBase() const;    

private:
    // void incrQueryRefCounter(int pool_idx, const std::vector<int>& blocks);
    // void decrQueryRefCounter(int pool_idx, const std::vector<int>& blocks);

    std::vector<std::shared_ptr<KVCacheGroup>> kv_cache_groups_;
    
    MallocResult mallocWithCache(const MallocInfo& malloc_info) {

    }

    MallocResult mallocSimple(const MallocInfo& malloc_info) {
    }

    // global_layer_id -> group_pools
    // std::unordered_map<int, BlockPoolPtr>>pool_map_;
    // std::vector<BlockPoolPtr> block_pools_;

    // std::vector<BlockRefCounter> query_ref_counters_;
    // std::vector<size_t> available_blocks_;
};

}  // namespace rtp_llm
