MallocResult KVCacheManager::malloc(const MallocInfo& malloc_info) {
    auto malloc_result = allocator_->malloc(malloc_info);
    if (!malloc_result.success) {
        return {false, {}};
    }

    // match from memory block cache
    if (enable_memory_block_cache) {
        auto matched_block_num = malloc_result.match_result.reuse_length;
        auto total_block_num = malloc_info.stream->streamCacheResource()->kvCache().batch_block_id[0].size();
        if (matched_block_num < total_block_num) {
            auto need_block_num = total_block_num - matched_block_num;
            auto mem_match_result = memory_block_cache_->match(malloc_info.stream->streamCacheResource()->kvCache().cache_keys[0]);
            if (mem_match_result.matched_len > 0) {

            }
        }
    }
}