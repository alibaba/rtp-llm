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
        // if malloc_info.enable_reuse_cache {
        //     return mallocWithCache(malloc_info);
        // } else {
        //     return mallocSimple(malloc_info);
        // }
    }

    FreeResult free(const FreeInfo& free_info) {
        // for(int i = 0; i < grouped_kv_cache_resources.size(); i++) {
        //     auto grouped_kv_cache_resource = free_info.grouped_kv_cache_resources[i];
        //     kv_cache_groups_[i]->free(grouped_kv_cache_resource->block_indices);
        // }
    }
    InsertResult insertIntoCache(const InsertInfo& insert_info){
        // for(int i = 0; i < grouped_kv_cache_resources.size(); i++) {
        //     auto grouped_kv_cache_resource = insert_info.grouped_kv_cache_resources[i];
        //     kv_cache_groups_[i]->insertIntoCache(grouped_kv_cache_resource->cache_keys, grouped_kv_cache_resource->block_indices);
        // }
    };
    CacheLayerLayout layerCacheBase() const {
        
    };

private:
    // void incrQueryRefCounter(int pool_idx, const std::vector<int>& blocks);
    // void decrQueryRefCounter(int pool_idx, const std::vector<int>& blocks);

    std::vector<std::shared_ptr<KVCacheGroup>> kv_cache_groups_;
    int global_block_stride; // gcd; 
    
    MallocResult mallocWithCache(const MallocInfo& malloc_info) {
        // MallocResult malloc_result;
        // int reuse_len = 0;  // block-wise

        // for (auto& kv_cache_group : kv_cache_groups_) {
        //     auto grouped_kv_cache_resource = std::make_shared<GroupedKVCacheResource>();

        //     auto match_result = kv_cache_group->match(malloc_info.cache_keys);
        //     grouped_kv_cache_resource->cache_keys = malloc_info.cache_keys;
        //     grouped_kv_cache_resource->block_indices = match_result.block_indices;

        //     malloc_result.grouped_kv_cache_resources.push_back(grouped_kv_cache_resource);
        //     aligned_reuse_length = match_result.reuse_length / global_block_stride * global_block_stride;

        //     if (aligned_reuse_length < reuse_len) {
        //         reuse_len = match_result.reuse_length;
        //     }
        // }

        // for (int i = 0; i< kv_cache_groups_.size(); i++) {
        //     auto grouped_kv_cache_resource = malloc_result.grouped_kv_cache_resources[i];
        //     int block_len = aligned_reuse_length / kv_cache_groups_[i]->get_block_stride();

        //     vector<int> aligned_block_indices(grouped_kv_cache_resource->block_indices.begin(), grouped_kv_cache_resource->block_indices.begin() + block_len);
        //     grouped_kv_cache_resource->block_indices = aligned_block_indices;

        //     vector<int> alloc_block_indices = kv_cache_groups_[i]->alloc(grouped_kv_cache_resource->cache_keys, block_len);
        //     grouped_kv_cache_resource->block_indices.append(alloc_block_indices);
        // }

        return malloc_result;
    }

    MallocResult mallocSimple(const MallocInfo& malloc_info) {
        // for(int i = 0; i < kv_cache_groups_.size(); i++) {
        //     auto grouped_kv_cache_resource = std::make_shared<GroupedKVCacheResource>();
        //     grouped_kv_cache_resource->cache_keys = malloc_info.cache_keys;
        //     grouped_kv_cache_resource->block_indices = kv_cache_groups_[i]->alloc(malloc_info.cache_keys, block_len);
        //     malloc_result.grouped_kv_cache_resources.push_back(grouped_kv_cache_resource);
        // }

        // return malloc_result;
    }

    // global_layer_id -> group_pools
    // std::unordered_map<int, BlockPoolPtr>>pool_map_;
    // std::vector<BlockPoolPtr> block_pools_;

    // std::vector<BlockRefCounter> query_ref_counters_;
    // std::vector<size_t> available_blocks_;
};

}  // namespace rtp_llm
