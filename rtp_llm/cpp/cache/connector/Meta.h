#pragma once

#include <utility>

namespace rtp_llm {

class Meta {
public:
    Meta()  = default;
    ~Meta() = default;

public:
    // <start_block_index, block_num>
    std::pair<int, int> blockRange() const {
        return {start_block_index_, block_num_};
    }
    void setBlockRange(int start_block_index, int block_num) {
        start_block_index_ = start_block_index;
        block_num_         = block_num;
    }

    bool enableMemoryCache() const {
        return enable_memory_cache_;
    }
    void setEnableMemoryCache(bool enable_memory_cache) {
        enable_memory_cache_ = enable_memory_cache;
    }

    bool skipLastCacheKey() const {
        return skip_last_cache_key_;
    }
    void setSkipLastCacheKey(bool skip_last_cache_key) {
        skip_last_cache_key_ = skip_last_cache_key;
    }

private:
    int  start_block_index_{0};
    int  block_num_{0};
    bool enable_memory_cache_{true};
    bool skip_last_cache_key_{true};
};

}  // namespace rtp_llm
