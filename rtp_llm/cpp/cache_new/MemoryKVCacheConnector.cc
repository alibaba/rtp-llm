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
    return true;
}

void MemoryKVCacheConnector::asyncPrefixPut(const Buffers& buffers, const Meta& meta, const CallBack& callback) {
    //
}

void MemoryKVCacheConnector::asyncPut(const Buffers& buffers, const Meta& /*meta*/, const CallBack& callback) {
    //
}

void MemoryKVCacheConnector::asyncPrefixGet(const Buffers& buffers, const Meta& meta, const CallBack& callback) {
    //
}

void MemoryKVCacheConnector::asyncGet(const Buffers& buffers, const Meta& /*meta*/, const CallBack& callback) {
    //
}

bool MemoryKVCacheConnector::match(int64_t key) {
    //
}

int32_t MemoryKVCacheConnector::prefixMatch(const std::vector<int64_t>& keys) {
    //
}

bool MemoryKVCacheConnector::copyBufferData(const BufferPtr& dst, const BufferPtr& src) {
    // if (!dst || !src) {
    //     return false;
    // }
    // if (dst->type() != src->type() || dst->sizeBytes() != src->sizeBytes()) {
    //     return false;
    // }
    // // Only handle host memory for now; extend as needed.
    // if (dst->where() != src->where()) {
    //     return false;
    // }
    // if (dst->where() != MemoryType::MEMORY_CPU) {
    //     // Non-CPU copies are not supported in this simple memory connector.
    //     return false;
    // }
    // std::memcpy(dst->data(), src->data(), dst->sizeBytes());
    // return true;
}

}  // namespace rtp_llm
