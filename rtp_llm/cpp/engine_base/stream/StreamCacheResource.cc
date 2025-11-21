#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/HashUtil.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

using namespace std;

namespace rtp_llm {

void StreamCacheResource::init(int batch_size) {
    batch_resource_->resize(batch_size);
    batch_resource_->initGroups(1);
    // constructCacheKey();
}

// void StreamCacheResource::freeBatchBlocks(size_t batch_id, vector<int>& blocks) {
//     if (blocks.empty()) {
//         return;
//     }
//     // only reuse the cache of batch 0 of finished beam search streams
//     // to avoid corrupted states and excessive block cache entries
//     bool should_reuse_cache =
//         reuseCache() && (!stream_->hasNumBeams() || (!stream_->stoppedWithoutLock() && batch_id == 0));
//     // TODO(zhangjianning.zjn) cache all beams of beam search
//     if (blocks.size() == batch_resource_.blockSize(batch_id) && should_reuse_cache) {
//         reConstructCacheKeys();
//         auto          tokens_id  = stream_->completeTokenIdsVec(batch_id);
//         const auto&   cache_keys = stream_->cacheKeys(batch_id);
//         vector<float> loss;
//         if (stream_->getLoss()) {
//             loss = rtp_llm::buffer2vector<float>(*(stream_->getLoss()));
//         }
//         // TODO(xinfei.sxf) 一些场景调用了cancel的地方，是否应该free with cache
//         CacheManager::FreeInfo free_info(stream_->streamId(),
//                                          tokens_id,
//                                          cache_keys,
//                                          blocks,
//                                          loss,
//                                          adapter_name_,
//                                          enable3FS(),
//                                          enableMemoryBlockCache());
//         resource_context_.cache_manager->freeWithCache(free_info);
//     } else {
//         resource_context_.cache_manager->free(blocks);
//     }
// }

void StreamCacheResource::releaseResource() {
    if (!resource_context_.cache_manager) {
        return;
    }
    // do not reuse cache from stopped beam search streams, whose states are likely corrupted
    if (!need_release_resource_ && (!stream_->hasNumBeams() || !stream_->stoppedWithoutLock())) {
        // reConstructCacheKeys();
        return;
    }
    tryReleaseKVBlock(maxBlockSize());
    batch_resource_->clear();
}

int StreamCacheResource::tryReleaseKVBlock(size_t nums) {
    RTP_LLM_LOG_DEBUG("stream [%ld] try release [%lu] blocks", stream_->streamId(), nums);

    if (fake_inited_) {
        int max_block_size = maxBlockSize();
        int batch_size     = batch_resource_->batchSize();
        batch_resource_->clear();
        batch_resource_->resize(batch_size);
        fake_inited_ = false;
        return max_block_size;
    }

    // NOTE: Currently only support releasing all blocks
    // Partial release (shrink) is not supported yet
    int release_blocks_num = maxBlockSize();

    if (release_blocks_num > 0 && batch_resource_->batchSize() > 0) {
        // Free all blocks using KVCacheManager::free
        FreeInfo free_info(batch_resource_, stream_->completeTokenIdsPtr());
        free_info.request_id = stream_->streamId();

        // TODO(chanyin): Handle cache insertion for reuse_cache case

        resource_context_.cache_manager->free(free_info);

        // batch_resource_ is modified directly by KVCacheManager::free
    }

    // After releasing all blocks, reserved_blocks = 0
    stream_->setFallbackPrefixLength(0);
    if (stream_->enable_fast_gen_) {
        stream_->resetChunkLen(0, stream_->seqLength());
    }
    return release_blocks_num;
}

absl::Status StreamCacheResource::releaseSequenceKVCache(size_t total_seq_len, size_t release_seq_len) {
    RTP_LLM_LOG_DEBUG("stream [%ld] max block size is [%lu] total seq_len is [%lu], release [%lu] seq_len KVCache",
                      stream_->streamId(),
                      maxBlockSize(),
                      total_seq_len,
                      release_seq_len);
    size_t last_block_occupied_seq_len =
        seqSizePerBlock() == 1 ? 1 : ((total_seq_len + seqSizePerBlock() - 2) % seqSizePerBlock() + 1);
    if (release_seq_len < last_block_occupied_seq_len) {
        return absl::OkStatus();
    }
    size_t release_block_num       = release_seq_len / seqSizePerBlock() + 1;
    size_t succ_release_blocks_num = tryReleaseKVBlock(release_block_num);
    if (release_block_num != succ_release_blocks_num) {
        return absl::InternalError("Release KVCache failed");
    }
    return absl::OkStatus();
}

int StreamCacheResource::singleBatchNeedBlocks(int seq_len) const {
    return std::max((seq_len + seqSizePerBlock() - 1) / seqSizePerBlock() - maxBlockSize(), 0);
}

// TODO(xinfei.sxf) 保证这个函数的原子性
absl::StatusOr<int> StreamCacheResource::initKVBlock(int token_capacity, size_t reserve_step) {
    return incrKVBlock(token_capacity, reserve_step);
}

absl::StatusOr<int> StreamCacheResource::incrKVBlock(int token_capacity, size_t reserve_step) {
    // TODO(xinfei.sxf) rollback token_capacity
    // TODO(xinfei.sxf) add reserver_blocks
    if (fake_inited_) {
        return absl::InternalError("fake inited not allow to incr block");
    }
    int real_occupy = 0;
    if (stream_->enable_fast_gen_) {
        if (stream_->isChunkStream() || !stream_->isContextStream()) {
            auto result = stream_->acquireCapacity(token_capacity);
            if (!result.ok()) {
                return result;
            }
            real_occupy = result.value();
        }
    }

    auto seq_len = stream_->isChunkStream() ? stream_->currentChunkLen() : (stream_->seqLength() + (int)reserve_step);
    auto common_seq_len = std::min(seq_len, stream_->adjustedCommonLen());

    // Prepare MallocInfo for KVCacheManager
    MallocInfo malloc_info(batch_resource_, stream_->completeTokenIdsPtr());
    malloc_info.request_id     = stream_->streamId();
    malloc_info.verbose        = malloc_failed_times_ >= 10 ? malloc_failed_times_ % 100 == 0 : true;
    malloc_info.common_seq_len = common_seq_len;
    malloc_info.total_seq_len  = seq_len;

    // Call KVCacheManager::malloc which will handle batch_resource_ updates internally
    auto result = resource_context_.cache_manager->malloc(malloc_info);
    if (!result.success) {
        malloc_failed_times_++;
        return absl::InternalError("malloc failed");
    }

    return real_occupy;
}

int StreamCacheResource::maxBlockSize() const {
    return batch_resource_->maxBlockSize();
}

const BatchKVCacheResource& StreamCacheResource::kvCache() const {
    batch_resource_->check();
    return *batch_resource_;
}

BatchKVCacheResource& StreamCacheResource::kvCacheMutable() {
    batch_resource_->check();
    return *batch_resource_;
}

void StreamCacheResource::setKVCache(const BatchKVCacheResource& kv_cache_resource) {
    *batch_resource_ = kv_cache_resource;
}

// TODO(chanyin): move  kv blocks update for beam search to kv cache manager
bool StreamCacheResource::updateKVBlock(const std::vector<int>& block_src_batch, bool copy_last_block) {
    // return true;
    return resource_context_.cache_manager->updateKVBlock(
        batch_resource_, block_src_batch, copy_last_block, block_update_mapping_);
}

bool StreamCacheResource::hasCacheKeys() const {
    if (batch_resource_->batch_resource.empty()) {
        return false;
    }
    for (const auto& br : batch_resource_->batch_resource) {
        if (!br.cache_keys.empty()) {
            return true;
        }
    }
    return false;
}

const CacheKeysType& StreamCacheResource::cacheKeys(int32_t batch_id) const {
    RTP_LLM_CHECK_WITH_INFO(batch_id >= 0 && batch_id < batch_resource_->batchSize(), "invalid batch_id");
    return batch_resource_->batch_resource[batch_id].cache_keys;
}

void StreamCacheResource::fakeInitKVBlock() {
    fake_inited_ = true;
    batch_resource_->resize(stream_->maxBatchSize());
    for (size_t i = 0; i < stream_->maxBatchSize(); i++) {
        batch_resource_->resize(i, stream_->seqLength(), true);
    }

    // cache keys will be constructed lazily per batch_resource[i].cache_keys
}

int StreamCacheResource::mallocFailedTimes() const {
    return malloc_failed_times_;
}

bool StreamCacheResource::reuseCache() const {
    return resource_context_.reuse_cache && stream_->reuseCache();
}

bool StreamCacheResource::enable3FS() const {
    return resource_context_.enable_3fs && stream_->enable3FS();
}

bool StreamCacheResource::enableMemoryBlockCache() const {
    return resource_context_.enable_memory_block_cache && stream_->enableMemoryBlockCache();
}

}  // namespace rtp_llm
