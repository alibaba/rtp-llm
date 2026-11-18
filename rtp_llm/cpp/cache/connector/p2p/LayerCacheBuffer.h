#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

class LayerCacheBuffer {
public:
    LayerCacheBuffer(int layer_id, std::string cache_tag, int chunk_id = 0);
    ~LayerCacheBuffer() = default;

public:
    /// @brief 记录 cache_key 对应的 block_id
    void addBlockId(int64_t cache_key, int block_id);
    /// @brief 查询 cache_key 对应的 block_id，未找到返回 -1
    int getBlockId(int64_t cache_key) const;
    int getLayerId() const {
        return layer_id_;
    }
    const std::string& cacheTag() const {
        return cache_tag_;
    }
    int chunkId() const {
        return chunk_id_;
    }
    std::string bufferKey() const {
        return std::to_string(layer_id_) + ":" + cache_tag_;
    }
    const std::map<int64_t, int>& blockIdMap() const {
        return block_id_map_;
    }
    void setKVCacheResource(const KVCacheResourcePtr& resource) {
        resource_ = resource;
    }
private:
    int                    layer_id_;
    std::string            cache_tag_;
    int                    chunk_id_;
    std::map<int64_t, int> block_id_map_;  // [cache_key, block_id]
    KVCacheResourcePtr     resource_;  // Rank-0 source hold for the sender's synchronous copy.
};

}  // namespace rtp_llm
