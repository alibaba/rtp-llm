#include "maga_transformer/cpp/dataclass/StreamCacheResource.h"
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include <atomic>
#include "maga_transformer/cpp/utils/StringUtil.h"

using namespace std;

namespace rtp_llm {

void StreamCacheResource::freeBatchBlocks(size_t batch_id, vector<void*>& blocks) {
    if (blocks.size() == kv_cache_block_addr_.k_ptr[batch_id][0].size() && resource_context_.reuse_cache) {
        auto tokens_id = stream_->completeTokenIdsVec(batch_id);
        resource_context_.cache_manager->freeWithCache(blocks, tokens_id);
    } else {
        resource_context_.cache_manager->free(blocks);
    }
}

void StreamCacheResource::releaseResource() {
    if (!need_release_resource_ || !resource_context_.cache_manager) {
        return ;
    }
    tryReleaseKVBlock(maxBlockSize());
    kv_cache_block_addr_.clear();
}

// TODO(xinfei.sxf) add ut
int StreamCacheResource::tryReleaseKVBlock(size_t nums) {
    FT_LOG_DEBUG("stream [%ld] try release [%lu] blocks", stream_->streamId(), nums);
    size_t release_blocks_num = 0;
    for (size_t batch_id = 0; batch_id < kv_cache_block_addr_.k_ptr.size(); batch_id++) {
        for (size_t layer_id = 0; layer_id < kv_cache_block_addr_.k_ptr[batch_id].size(); layer_id++) {
            auto& k_blocks = kv_cache_block_addr_.k_ptr[batch_id][layer_id];
            size_t reserver_blocks = std::max(0, int(k_blocks.size()) - int(nums));
            // TODO(xinfei.sxf) release_blocks_num select min?
            release_blocks_num = k_blocks.size() - reserver_blocks;
            if (layer_id == 0) {
                vector<void*> release_blocks(k_blocks.begin() + reserver_blocks, k_blocks.end());
                freeBatchBlocks(batch_id, release_blocks);
            }
            kv_cache_block_addr_.resize(batch_id, layer_id, reserver_blocks);
        }
    }
    return release_blocks_num;
}

bool StreamCacheResource::initKVBlock() {
    auto             block_num = needKVCacheBlockNums();
    KVCacheBlockAddr kv_cache_block_addr;
    int              reuse_length;
    bool             success;
    if (resource_context_.reuse_cache) {
        std::tie(success, kv_cache_block_addr, reuse_length) =
            resource_context_.cache_manager->mallocWithCache(block_num, stream_->completeTokenIdsVec());
    } else {
        std::tie(success, kv_cache_block_addr) = resource_context_.cache_manager->malloc(block_num);
        reuse_length                           = 0;
    }

    if (success) {
        int                   tile_num = stream_->tileNum();
        BatchKVCacheBlockAddr batch_block;
        batch_block.pushBack(kv_cache_block_addr);
        for (uint32_t i = 1; i < tile_num; i++) {
            // clone increased block reference count
            batch_block.pushBack(kv_cache_block_addr.clone(resource_context_.cache_manager));
        }
        setKVCache(batch_block);
        stream_->setReuseLength(reuse_length);
    }
    return success;
}

int StreamCacheResource::singleBatchNeedBlocks() const {
    return std::max((stream_->seqLength() + seqSizePerBlock() - 1) / seqSizePerBlock() - maxBlockSize(), 0);
}

int StreamCacheResource::needKVCacheBlockNums() const {
    int block_batch = 1;
    if (stream_->isContextStream()) {
        block_batch = 1;
    } else {
        block_batch = stream_->tileNum();
    }
    return singleBatchNeedBlocks() * block_batch;
}

bool StreamCacheResource::incrKVBlock() {
    auto blocks_num = singleBatchNeedBlocks();
    if (blocks_num <= 0) {
        return true;
    }
    auto                          batch_size  = stream_->tileNum();
    bool                          all_success = true;
    std::vector<KVCacheBlockAddr> resource;
    for (uint32_t i = 0; i < batch_size; i++) {
        auto [success, kv_cache_block_addr] = resource_context_.cache_manager->malloc(blocks_num);
        if (success) {
            resource.push_back(kv_cache_block_addr);
        } else {
            all_success = false;
            break;
        }
    }
    if (!all_success) {
        resource_context_.cache_manager->free(resource);
        return false;
    }

    int resource_index = 0;
    for (int i = 0; i < batch_size; i++) {
        kv_cache_block_addr_.append(i, resource[resource_index]);
        resource_index++;
    }
    return true;
}

int StreamCacheResource::maxBlockSize() const {
    size_t max_block_size = 0;
    for (auto& batch_blocks : kv_cache_block_addr_.k_ptr) {
        max_block_size = std::max(max_block_size, batch_blocks[0].size());
    }
    return max_block_size;
}

const BatchKVCacheBlockAddr& StreamCacheResource::kvCache() const {
    return kv_cache_block_addr_;
}

void StreamCacheResource::setKVCache(const BatchKVCacheBlockAddr& kv_cache_block_addr) {
    kv_cache_block_addr_ = kv_cache_block_addr;
}

}  // namespace rtp_llm
