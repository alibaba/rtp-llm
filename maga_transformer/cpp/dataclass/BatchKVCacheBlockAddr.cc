#pragma once

#include "maga_transformer/cpp/dataclass/BatchKVCacheBlockAddr.h"
#include <memory>
#include <sstream>
#include <cassert>

namespace rtp_llm {

int BatchKVCacheBlockAddr::batchSize() const {
    return batch_offset.size();
}

int BatchKVCacheBlockAddr::blockSize(int batch_id) const {
    return batch_offset[batch_id].size();
}

void BatchKVCacheBlockAddr::resize(size_t batch_size) {
    batch_offset.resize(batch_size);
}

void BatchKVCacheBlockAddr::resize(size_t batch_id, int reserver_blocks) {
    FT_CHECK(batch_offset.size() > batch_id && batch_offset[batch_id].size() >= reserver_blocks);
    batch_offset[batch_id].resize(reserver_blocks);
}

void BatchKVCacheBlockAddr::pushBack(const KVCacheBlockAddr& addr) {
    batch_offset.push_back(addr.offset);
}

void BatchKVCacheBlockAddr::appendClone(const KVCacheBlockAddr& addr, std::shared_ptr<CacheManager>& cache_manager) {
    append(0, addr);
    for (uint32_t i = 1; i < batch_offset.size(); i++) {
        // clone increased block reference count
        append(i, addr.clone(cache_manager));
    }
}

void BatchKVCacheBlockAddr::append(const std::vector<KVCacheBlockAddr>& resource) {
    for (int i = 0; i < batch_offset.size(); i++) {
        append(i, resource[i]);
    }
}

void BatchKVCacheBlockAddr::append(size_t batch_id, const KVCacheBlockAddr& addr) {
    FT_CHECK(batch_offset.size() > batch_id);
    batch_offset[batch_id].insert(batch_offset[batch_id].end(), addr.offset.begin(), addr.offset.end());
}

int BatchKVCacheBlockAddr::maxBlockSize() const {
    size_t max_block_size = 0;
    for (const auto& blocks : batch_offset) {
        max_block_size = std::max(max_block_size, blocks.size());
    }
    return max_block_size;
}

const std::vector<int>& BatchKVCacheBlockAddr::blocks(int batch_id) const {
    FT_CHECK(batch_offset.size() > batch_id);
    return batch_offset[batch_id];
}

void BatchKVCacheBlockAddr::clear() {
    batch_offset.clear();
}

std::string BatchKVCacheBlockAddr::debugString() const {
    std::stringstream debug_string, batch_offset_string;
    for (int i = 0; i < batch_offset.size(); i++) {
        batch_offset_string << "batch: " << i << " , block_id: ";
        for (auto &v: batch_offset[i]) {
            batch_offset_string << v << ", ";
        }
    }

    debug_string << "BatchKVCacheBlockAddr {" << batch_offset_string.str()
                    << "}";
    return debug_string.str();
}

}  // namespace rtp_llm
