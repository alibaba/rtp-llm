#include "maga_transformer/cpp/dataclass/StreamCacheResource.h"
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include <atomic>
#include "maga_transformer/cpp/utils/StringUtil.h"

using namespace std;

namespace rtp_llm {

void StreamCacheResource::freeBatchBlocks(size_t batch_id, vector<int>& blocks) {
    if (blocks.size() == batch_block_addr_.batch_offset[batch_id].size() && resource_context_.reuse_cache) {
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
    batch_block_addr_.clear();
}

int StreamCacheResource::tryReleaseKVBlock(size_t nums) {
    FT_LOG_DEBUG("stream [%ld] try release [%lu] blocks", stream_->streamId(), nums);
    size_t release_blocks_num = 0;

    for (size_t batch_id = 0; batch_id < batch_block_addr_.batch_offset.size(); batch_id++) {
        auto& blocks = batch_block_addr_.batch_offset[batch_id];
        size_t reserver_blocks = std::max(0, int(blocks.size()) - int(nums));
        // NOTE: all batch has same number of blocks
        release_blocks_num = blocks.size() - reserver_blocks;
        vector<int> release_blocks(blocks.begin() + reserver_blocks, blocks.end());
        freeBatchBlocks(batch_id, release_blocks);
        batch_block_addr_.resize(batch_id, reserver_blocks);
    }
    return release_blocks_num;
}

bool StreamCacheResource::initKVBlock() {
    auto             align_block_num = singleBatchNeedBlocks(stream_->seqLength() / seqSizePerBlock() * seqSizePerBlock());
    KVCacheBlockAddr kv_cache_block_addr;
    int              reuse_length;
    bool             success;
    auto current_block_size = maxBlockSize();
    if (current_block_size) {
        // 说明是部分回退的场景
        auto fallback_prefix_length = current_block_size * seqSizePerBlock();
        stream_->setFallbackPrefixLength(fallback_prefix_length);
        return incrKVBlock();
    }
    if (resource_context_.reuse_cache) {
        std::tie(success, kv_cache_block_addr, reuse_length) =
            resource_context_.cache_manager->mallocWithCache(align_block_num, stream_->completeTokenIdsVec());
    } else {
        std::tie(success, kv_cache_block_addr) = resource_context_.cache_manager->malloc(align_block_num);
        reuse_length                           = 0;
    }

    if (success) {
        batch_block_addr_.clear();
        int                   tile_num = stream_->tileNum();
        auto kv_cache = resource_context_.cache_manager->kvCacheBuffer();
        BatchKVCacheBlockAddr batch_blocks;
        batch_block_addr_.pushBack(kv_cache_block_addr);
        for (uint32_t i = 1; i < tile_num; i++) {
            // clone increased block reference count
            batch_block_addr_.pushBack(kv_cache_block_addr.clone(resource_context_.cache_manager));
        }
        stream_->setReuseLength(reuse_length);
    }
    return success && incrKVBlock();
}

int StreamCacheResource::singleBatchNeedBlocks(int seq_len) const {
    return std::max((seq_len + seqSizePerBlock() - 1) / seqSizePerBlock() - maxBlockSize(), 0);
}

bool StreamCacheResource::incrKVBlock() {
    auto blocks_num = singleBatchNeedBlocks(stream_->seqLength());
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
        batch_block_addr_.append(i, resource[resource_index]);
        resource_index++;
    }

    return true;
}

int StreamCacheResource::maxBlockSize() const {
    size_t max_block_size = 0;
    for (auto& blocks : batch_block_addr_.batch_offset) {
        max_block_size = std::max(max_block_size, blocks.size());
    }
    return max_block_size;
}

const BatchKVCacheBlockAddr& StreamCacheResource::kvCache() const {
    return batch_block_addr_;
}

void StreamCacheResource::setKVCache(const BatchKVCacheBlockAddr& kv_cache_block_addr) {
    batch_block_addr_ = kv_cache_block_addr;
}

}  // namespace rtp_llm
