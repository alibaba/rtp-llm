#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheManager.h"
#include <memory>
#include <sstream>
#include <cassert>

namespace rtp_llm {

int BatchKVCacheResource::batchSize() const {
    return batch_block_id.size();
}

int BatchKVCacheResource::blockSize(int batch_id) const {
    return batch_block_id[batch_id].size();
}

void BatchKVCacheResource::resize(size_t batch_size) {
    batch_block_id.resize(batch_size);

    batch_resource.resize(batch_size);
}

// TODO, fix name
void BatchKVCacheResource::resize(size_t batch_id, int reserver_blocks, int value) {
    RTP_LLM_CHECK(batch_block_id.size() > batch_id);
    batch_block_id[batch_id].resize(reserver_blocks, value);

    batch_resource[batch_id].resizeBlocks(reserver_blocks, value);
}

void BatchKVCacheResource::shrink(size_t batch_id, int reserver_blocks) {
    RTP_LLM_CHECK(batch_block_id.size() > batch_id && batch_block_id[batch_id].size() >= size_t(reserver_blocks));
    batch_block_id[batch_id].resize(reserver_blocks);
}

void BatchKVCacheResource::pushBack(const KVCacheResource& addr) {
    batch_block_id.push_back(addr.block_id);
}

void BatchKVCacheResource::appendClone(const KVCacheResource& addr, std::shared_ptr<CacheManager>& cache_manager) {
    append(0, addr);
    for (uint32_t i = 1; i < batch_block_id.size(); i++) {
        // clone increased block reference count
        append(i, addr.clone(cache_manager));
    }
}

void BatchKVCacheResource::append(const std::vector<KVCacheResource>& resource) {
    for (int i = 0; i < batch_block_id.size(); i++) {
        append(i, resource[i]);
    }
}

void BatchKVCacheResource::append(size_t batch_id, const KVCacheResource& addr) {
    RTP_LLM_CHECK(batch_block_id.size() > batch_id);
    batch_block_id[batch_id].insert(batch_block_id[batch_id].end(), addr.block_id.begin(), addr.block_id.end());
}

int BatchKVCacheResource::maxBlockSize() const {
    size_t max_block_size = 0;
    for (const auto& blocks : batch_block_id) {
        max_block_size = std::max(max_block_size, blocks.size());
    }
    return max_block_size;
}

const std::vector<int>& BatchKVCacheResource::blocks(int batch_id) const {
    RTP_LLM_CHECK(batch_block_id.size() > batch_id);
    return batch_block_id[batch_id];
}

void BatchKVCacheResource::clear() {
    batch_block_id.clear();
}

void BatchKVCacheResource::check() const {
    RTP_LLM_CHECK(!batch_block_id.empty());
    size_t block_size = batch_block_id[0].size();
    for (const auto& blocks : batch_block_id) {
        RTP_LLM_CHECK(blocks.size() == block_size);
    }
}

std::string BatchKVCacheResource::debugString() const {
    std::stringstream debug_string, batch_offset_string;
    for (size_t i = 0; i < batch_block_id.size(); i++) {
        batch_offset_string << "batch:[" << i << "] , block:[";
        for (auto& v : batch_block_id[i]) {
            batch_offset_string << v << ", ";
        }
        batch_offset_string << "], ";
    }

    debug_string << "BatchKVCacheResource {" << batch_offset_string.str() << "}";
    return debug_string.str();
}

}  // namespace rtp_llm