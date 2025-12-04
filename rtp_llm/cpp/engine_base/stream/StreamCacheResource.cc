#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/HashUtil.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

using namespace std;

namespace rtp_llm {

void StreamCacheResource::init(int batch_size) {
    batch_resource_->resetBatchSize(batch_size);
    batch_resource_->initGroups(1);
    batch_resource_->enable_reuse_cache = reuseCache();
}

void StreamCacheResource::releaseResource() {
    if (!resource_context_.cache_manager) {
        return;
    }
    // do not reuse cache from stopped beam search streams, whose states are likely corrupted
    if (!need_release_resource_ && (!stream_->hasNumBeams() || !stream_->stoppedWithoutLock())) {
        return;
    }
    tryReleaseKVBlock(maxBlocksNum());
    batch_resource_->clearBlocks();
}

int StreamCacheResource::tryReleaseKVBlock(size_t nums) {
    RTP_LLM_LOG_DEBUG("stream [%ld] try release [%lu] blocks", stream_->streamId(), nums);

    if (fake_inited_) {
        int max_blocks_num = maxBlocksNum();
        int batch_size     = batch_resource_->batchSize();
        batch_resource_->clearBlocks();
        batch_resource_->resetBatchSize(batch_size);
        fake_inited_ = false;
        return max_blocks_num;
    }

    // NOTE: Currently only support releasing all blocks
    // Partial release (shrink) is not supported yet
    int total_blocks = maxBlocksNum();
    RTP_LLM_CHECK(nums == total_blocks);

    if (total_blocks > 0) {
        // TODO(xinfei.sxf) fix it, after finshed and remote running commit.
        if (reuseCache()) {
            InsertInfo insert_info{batch_resource_, stream_->completeTokenIdsPtr(), false};
            resource_context_.cache_manager->insertIntoCache(insert_info);
        }

        FreeInfo free_info{batch_resource_, stream_->completeTokenIdsPtr()};
        free_info.request_id = stream_->streamId();

        resource_context_.cache_manager->free(free_info);
    }

    stream_->setFallbackPrefixLength(0);
    if (stream_->enable_fast_gen_) {
        stream_->resetChunkLen(0, stream_->seqLength());
    }
    return total_blocks;
}

// TODO, 等待删除。
int StreamCacheResource::singleBatchNeedBlocks(int seq_len) const {
    return std::max((seq_len + seqSizePerBlock() - 1) / seqSizePerBlock() - maxBlocksNum(), 0);
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

    MallocInfo malloc_info;
    malloc_info.batch_kv_cache_resource = batch_resource_;
    malloc_info.complete_token_ids      = stream_->completeTokenIdsPtr();
    malloc_info.request_id              = stream_->streamId();
    malloc_info.verbose                 = malloc_failed_times_ >= 10 ? malloc_failed_times_ % 100 == 0 : true;
    malloc_info.common_seq_len          = common_seq_len;
    malloc_info.total_seq_len           = seq_len;

    auto result = resource_context_.cache_manager->malloc(malloc_info);
    if (!result.success) {
        malloc_failed_times_++;
        return absl::InternalError("malloc failed");
    }

    if (result.reuse_len > 0) {
        stream_->setReuseLength(result.reuse_len);
        stream_->setMtpTokenIndex(result.reuse_len);
        stream_->setInitialReuseLength(result.reuse_len);
    }

    return real_occupy;
}

// TODO, delete it soon
int StreamCacheResource::maxBlocksNum() const {
    return batch_resource_->maxBlocksNum();
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

bool StreamCacheResource::updateKVBlock(const std::vector<int>& block_src_batch, bool copy_last_block) {
    return resource_context_.cache_manager->updateKVBlock(
        batch_resource_, block_src_batch, copy_last_block, block_update_mapping_);
}

bool StreamCacheResource::hasCacheKeys() const {
    if (batch_resource_->batch_resource.empty()) {
        return false;
    }
    for (auto& resource : batch_resource_->batch_resource) {
        if (!resource.cacheKeys().empty()) {
            return true;
        }
    }
    return false;
}

const CacheKeysType& StreamCacheResource::cacheKeys(int32_t batch_id) const {
    return batch_resource_->cacheKeys(batch_id);
}

void StreamCacheResource::fakeInitKVBlock() {
    fake_inited_ = true;
    batch_resource_->resetBatchSize(stream_->maxBatchSize());
    batch_resource_->resizeBlocks(stream_->seqLength(), 0);
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
