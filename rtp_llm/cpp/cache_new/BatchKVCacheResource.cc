#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache_new/KVCacheManager.h"
#include <memory>
#include <sstream>
#include <cassert>

namespace rtp_llm {

int BatchKVCacheResource::batchSize() const {
    return static_cast<int>(batch_resource.size());
}

int BatchKVCacheResource::blockSize(int batch_id) const {
    // return group 0 block size
    return batch_resource[batch_id].blocks();
}

void BatchKVCacheResource::resize(size_t batch_size) {
    batch_resource.resize(batch_size);
}

void BatchKVCacheResource::initGroups(int group_nums) {
    for (auto& br : batch_resource) {
        br.initGroups(group_nums);
    }
}

void BatchKVCacheResource::resize(size_t batch_id, int reserver_blocks, int value) {
    RTP_LLM_CHECK(batch_resource.size() > batch_id);
    batch_resource[batch_id].resizeBlocks(reserver_blocks, value);
}

int BatchKVCacheResource::maxBlockSize() const {
    size_t max_block_size = 0;
    for (const auto& br : batch_resource) {
        if (!br.group_block_ids.empty()) {
            max_block_size = std::max(max_block_size, br.group_block_ids[0]->size());
        }
    }
    return static_cast<int>(max_block_size);
}

const std::vector<int>& BatchKVCacheResource::blocks(int batch_id, int group_id) const {
    RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
    RTP_LLM_CHECK(batch_resource[batch_id].group_block_ids.size() > static_cast<size_t>(group_id));
    return batch_resource[batch_id].group_block_ids[group_id]->block_indices;
}

void BatchKVCacheResource::clear() {
    batch_resource.clear();
}

void BatchKVCacheResource::check() const {
    RTP_LLM_CHECK(!batch_resource.empty());
    size_t block_size = batch_resource[0].group_block_ids[0]->size();
    for (const auto& br : batch_resource) {
        RTP_LLM_CHECK(!br.group_block_ids.empty());
        RTP_LLM_CHECK(br.group_block_ids[0]->size() == block_size);
    }
}

std::string BatchKVCacheResource::debugString() const {
    std::stringstream debug_string, batch_offset_string;
    for (size_t i = 0; i < batch_resource.size(); i++) {
        for (int j = 0; j < batch_resource[i].group_block_ids.size(); j++) {
            batch_offset_string << "batch:[" << i << "], group:[" << j << "], block:[";
            for (auto& v : batch_resource[i].group_block_ids[j]->block_indices) {
                batch_offset_string << v << ", ";
            }
            batch_offset_string << "], ";
        }
    }

    debug_string << "BatchKVCacheResource {" << batch_offset_string.str() << "}";
    return debug_string.str();
}

KVCacheResourceV1& BatchKVCacheResource::resource(int batch_id) {
    RTP_LLM_CHECK(batch_id >= 0 && static_cast<size_t>(batch_id) < batch_resource.size());
    return batch_resource[batch_id];
}

}  // namespace rtp_llm