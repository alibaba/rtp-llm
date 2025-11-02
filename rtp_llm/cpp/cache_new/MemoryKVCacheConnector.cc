#include "rtp_llm/cpp/cache_new/MemoryKVCacheConnector.h"

#include <cstring>
#include <optional>
#include <unordered_map>
#include <shared_mutex>

namespace rtp_llm {

MemoryKVCacheConnector::MemoryKVCacheConnector(BlockPoolPtr         block_pool,
                                               BlockCachePtr        block_cache,
                                               rtp_llm::DeviceBase* device):
    block_pool_(block_pool), block_cache_(block_cache), device_(device) {}

MemoryKVCacheConnector::~MemoryKVCacheConnector() {
    // TODO(LXQ): implement
}

bool MemoryKVCacheConnector::init() {
    // TODO(LXQ): implement
    return true;
}

void MemoryKVCacheConnector::asyncPrefixPut(const Buffers& buffers, const Meta& meta, const CallBack& callback) {
    asyncPut(buffers, meta, callback);
}

void MemoryKVCacheConnector::asyncPut(const Buffers& buffers, const Meta& /*meta*/, const CallBack& callback) {
    for (const auto& buffer : buffers) {
        if (block_cache_->isExistKey(buffer.key2)) {
            continue;
        }

        auto new_blocks = block_pool_->alloc(1);
        if (new_blocks.empty()) {
            callback(false);
            return;
        }

        const auto dst_buffer = getBuffer(new_blocks[0]);
        copyBuffer(dst_buffer, buffer.buffer);

        BlockCacheV1::CacheItem cache_item;
        cache_item.cache_key   = buffer.key2;
        cache_item.block_index = new_blocks[0];
        cache_item.loss        = buffer.loss.value();
        cache_item.is_resident = false;
        if (!block_cache_->put(cache_item)) {
            callback(false);
            return;
        }
    }
    callback(true);
}

void MemoryKVCacheConnector::asyncPrefixGet(const Buffers& buffers, const Meta& meta, const CallBack& callback) {
    asyncGet(buffers, meta, callback);
}

void MemoryKVCacheConnector::asyncGet(const Buffers& buffers, const Meta& /*meta*/, const CallBack& callback) {
    for (const auto& buffer : buffers) {
        if (!block_cache_->isExistKey(buffer.key2)) {
            callback(false);
            return;
        }
        auto match_result = block_cache_->match(buffer.key2);
        if (match_result.matched_index == NULL_BLOCK_IDX) {
            callback(false);
            return;
        }
        const auto dst_buffer = getBuffer(match_result.matched_index);
        copyBuffer(dst_buffer, buffer.buffer);
    }
    callback(true);
}

std::vector<bool> MemoryKVCacheConnector::match(const std::vector<int64_t>& keys) {
    return block_cache_->match(keys);
}

int32_t MemoryKVCacheConnector::prefixMatch(const std::vector<int64_t>& keys) {
    return block_cache_->prefixMatch(keys);
}

Buffer MemoryKVCacheConnector::getBuffer(const BlockIdxType& block_index) {
    return block_pool_->convertIndexToBuffer(block_index);
}

// Buffer MemoryKVCacheConnector::getBuffers(const int32_t& layer_idx, const BlockIdxType& block_index, int64_t key2) {
//     Buffers buffers;
//     auto buffer_info = block_pool_->convertIndexToBuffer(layer_id, block_index);
//     if (buffer_info.k_addr) {
//         buffers.push_back({group_id_, key2, buffer_info.k_addr, std::nullopt});
//     }
//     if (buffer_info.v_addr) {
//         buffers.push_back({group_id_, key2, buffer_info.v_addr, std::nullopt});
//     }
//     return buffers;
// }

void MemoryKVCacheConnector::copyBuffer(const Buffer& dst, const Buffer& src) {
    device_->noBlockCopy({dst, src});
}

void MemoryKVCacheConnector::copyBuffers(const Buffers& dst, const Buffers& src) {
    device_->noBlockCopy({dst, src});
}
}  // namespace rtp_llm
