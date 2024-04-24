#include "maga_transformer/cpp/dataclass/StreamCacheResource.h"
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/utils/BufferUtils.h"
#include <atomic>

using namespace std;

namespace rtp_llm {

void StreamCacheResource::releaseResource() {
    if (!need_release_resource_) {
        return ;
    }

    FT_LOG_DEBUG("stream [%ld] release resource", stream_->streamId());

    // for test
    if (!cache_manager_) {
        return;
    }
    if (!kv_cache_block_addr_.k_ptr.empty()) {
        for (auto& batch : kv_cache_block_addr_.k_ptr) {
            const auto& blocks = batch[0];
            if (reuse_cache_) {
                // TODO(xinfei.sxf) batch token
                auto tokens_id = stream_->completeTokenIdsVec();
                cache_manager_->freeWithCache(blocks, tokens_id);
            } else {
                cache_manager_->free(blocks);
            }
        }
        kv_cache_block_addr_.clear();
    }
}

int StreamCacheResource::tryReleaseKVBlock(size_t nums) {
    FT_LOG_DEBUG("stream [%ld] try release [%lu] blocks", stream_->streamId());
        
    if (kv_cache_block_addr_.k_ptr.empty() || kv_cache_block_addr_.k_ptr[0].empty()) {
        return 0;
    }
    int           release_blocks_num;
    vector<void*> release_blocks;
    // TODO(xinfei.sxf) deal with v, scale etc
    for (auto& batch : kv_cache_block_addr_.k_ptr) {
        for (size_t layer_id = 0; layer_id < batch.size(); layer_id++) {
            auto& blocks = batch[layer_id];
            auto reserver_blocks = std::max(0lu, blocks.size() - nums);
            release_blocks_num   = blocks.size() - reserver_blocks;
            if (layer_id == 0) {
                release_blocks.insert(release_blocks.end(), blocks.begin() + reserver_blocks, blocks.end());
            }
            blocks.resize(reserver_blocks);
        }
    }
    // TODO(xinfei.sxf) call free with cache if all blocks is released
    cache_manager_->free(release_blocks);
    return release_blocks_num;
}

bool StreamCacheResource::initKVBlock() {
    auto             block_num = initalKVCacheCount();
    KVCacheBlockAddr kv_cache_block_addr;
    int              reuse_length;
    bool             success;
    if (reuse_cache_) {
        std::tie(success, kv_cache_block_addr, reuse_length) =
            cache_manager_->mallocWithCache(block_num, stream_->completeTokenIdsVec());
    } else {
        std::tie(success, kv_cache_block_addr) = cache_manager_->malloc(block_num);
        reuse_length                           = 0;
    }

    if (success) {
        int                   tile_num = stream_->tileNum();
        BatchKVCacheBlockAddr batch_block;
        batch_block.pushBack(kv_cache_block_addr);
        for (uint32_t i = 1; i < tile_num; i++) {
            batch_block.pushBack(kv_cache_block_addr.clone(cache_manager_));
        }
        setKVCache(batch_block);
        stream_->setReuseLength(reuse_length);
    }
    return success;
}

// TODO(xinfei.sxf) fix this to reduce waste
int StreamCacheResource::initalKVCacheCount() const {
    return (stream_->seqLength() - 2 + gen_num_per_circle_) / cache_manager_->cacheConfig().seq_size_per_block + 1;
}

// TODO(xinfei.sxf) fix this to reduce waste
int StreamCacheResource::nextNeedBlockNums() const {
    auto next_length = stream_->seqLength() + gen_num_per_circle_;
    // TODO(xinfei.sxf) deal with ptuning
    auto current_block_length = maxBlockSize() * cache_manager_->cacheConfig().seq_size_per_block;
    return ((next_length - current_block_length - 1) / cache_manager_->cacheConfig().seq_size_per_block) + 1;
}

bool StreamCacheResource::incrKVBlock() {
    auto blocks_num = nextNeedBlockNums();
    if (blocks_num <= 0) {
        return true;
    }
    auto                          batch_size  = stream_->batchSize();
    bool                          all_success = true;
    std::vector<KVCacheBlockAddr> resource;
    for (uint32_t i = 0; i < batch_size; i++) {
        auto [success, kv_cache_block_addr] = cache_manager_->malloc(blocks_num);
        if (success) {
            resource.push_back(kv_cache_block_addr);
        } else {
            all_success = false;
            break;
        }
    }
    if (!all_success) {
        cache_manager_->free(resource);
        return false;
    }

    int tile_num       = stream_->tileNum();
    int resource_index = 0;
    for (int i = 0; i < tile_num; i++) {
        if (stream_->sub_generate_status_[i].status == GenerateState::RUNNING) {
            kv_cache_block_addr_.append(i, resource[resource_index]);
            resource_index++;
        }
    }
    return true;
}

size_t StreamCacheResource::maxBlockSize() const {
    size_t max_block_size = 0;
    if (!kv_cache_block_addr_.k_ptr.empty()) {
        for (auto& batch_blocks : kv_cache_block_addr_.k_ptr) {
            max_block_size = std::max(max_block_size, batch_blocks[0].size());
        }
    }
    return max_block_size;
}

const BatchKVCacheBlockAddr& StreamCacheResource::kvCache() const {
    return kv_cache_block_addr_;
}

void StreamCacheResource::setKVCache(const BatchKVCacheBlockAddr& kv_cache_block_addr) {
    kv_cache_block_addr_ = kv_cache_block_addr;
}

void StreamCacheResource::setCacheManager(const std::shared_ptr<CacheManager>& cache_manager) {
    cache_manager_ = cache_manager;
}

void StreamCacheResource::setPtuning(const std::shared_ptr<PtuningBase>& ptuning) {
    ptuning_ = ptuning;
}

void StreamCacheResource::setReuseCache(bool reuse_cache) {
    reuse_cache_ = reuse_cache;
}

void StreamCacheResource::setNeedReleaseResource(bool need_release_resource) {
    need_release_resource_ = need_release_resource;
}

}  // namespace rtp_llm
