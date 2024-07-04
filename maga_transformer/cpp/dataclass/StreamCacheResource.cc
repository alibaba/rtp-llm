#include "maga_transformer/cpp/dataclass/StreamCacheResource.h"
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include <atomic>
#include "maga_transformer/cpp/utils/StringUtil.h"

using namespace std;

namespace rtp_llm {

void StreamCacheResource::init(int batch_size) {
    batch_block_addr_.resize(batch_size);
}

void StreamCacheResource::freeBatchBlocks(size_t batch_id, vector<int>& blocks) {
    if (blocks.size() == batch_block_addr_.blockSize(batch_id) && resource_context_.reuse_cache) {
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
    size_t reserved_blocks = 0;

    // NOTE: all batch has same number of blocks
    for (size_t batch_id = 0; batch_id < batch_block_addr_.batchSize(); batch_id++) {
        const auto& blocks = batch_block_addr_.blocks(batch_id);
        reserved_blocks = std::max(0, int(blocks.size()) - int(nums));
        release_blocks_num = blocks.size() - reserved_blocks;
        vector<int> release_blocks(blocks.begin() + reserved_blocks, blocks.end());
        freeBatchBlocks(batch_id, release_blocks);
        batch_block_addr_.resize(batch_id, reserved_blocks);
    }
    stream_->setFallbackPrefixLength(reserved_blocks * seqSizePerBlock());
    if (stream_->enable_fast_gen_) {
        stream_->resetChunkLen(reserved_blocks * seqSizePerBlock(), stream_->seqLength());
    }
    return release_blocks_num;
}

int StreamCacheResource::singleBatchNeedBlocks(int seq_len) const {
    return std::max((seq_len + seqSizePerBlock() - 1) / seqSizePerBlock() - maxBlockSize(), 0);
}

// TODO(xinfei.sxf) 保证这个函数的原子性
absl::StatusOr<int> StreamCacheResource::initKVBlock(int token_capacity) {
    auto current_block_size = maxBlockSize();
    if (current_block_size) {
        // partial fallback
        return incrKVBlock(token_capacity);
    }

    if (resource_context_.reuse_cache) {
        KVCacheBlockAddr kv_cache_block_addr;
        int              reuse_length;
        bool             success = true;

        auto common_tokens_vec = stream_->commonCompleteTokenIdsVec();
        reuse_length = resource_context_.cache_manager->match(common_tokens_vec);
        stream_->setReuseLength(reuse_length);

        auto reuse_block_num = reuse_length / seqSizePerBlock();
        std::tie(success, kv_cache_block_addr, reuse_length) =
            resource_context_.cache_manager->mallocWithCache(reuse_block_num, common_tokens_vec);
        if (success) {
            batch_block_addr_.appendClone(kv_cache_block_addr, resource_context_.cache_manager);
        } else {
            return absl::InternalError("malloc with cache failed");
        }
    }

    return incrKVBlock(token_capacity);
}

absl::StatusOr<int> StreamCacheResource::incrKVBlock(int token_capacity) {
    // TODO(xinfei.sxf) rollback token_capacity
    // TODO(xinfei.sxf) add reserver_blocks
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
    auto seq_len = stream_->isChunkStream() ? stream_->currentChunkLen() : stream_->seqLength();
    auto common_seq_len = std::min(seq_len, stream_->adjustedCommonLen());
    auto common_blocks_nums = singleBatchNeedBlocks(common_seq_len);

    auto [success, kv_cache_block_addr] = resource_context_.cache_manager->malloc(common_blocks_nums);
    if (!success) {
        return absl::InternalError("malloc failed");
    }
    batch_block_addr_.appendClone(kv_cache_block_addr, resource_context_.cache_manager);

    auto extra_blocks_num = singleBatchNeedBlocks(seq_len);
    if (extra_blocks_num <= 0) {
        return real_occupy;
    }
    if (extra_blocks_num) {
        auto                          batch_size  = stream_->tileNum();
        bool                          all_success = true;
        std::vector<KVCacheBlockAddr> resource;
        // TODO(xinfei.sxf) optimize code -> call malloc only once
        for (uint32_t i = 0; i < batch_size; i++) {
            auto [success, kv_cache_block_addr] = resource_context_.cache_manager->malloc(extra_blocks_num);
            if (success) {
                resource.push_back(kv_cache_block_addr);
            } else {
                all_success = false;
                break;
            }
        }

        if (!all_success) {
            resource_context_.cache_manager->free(resource);
            return absl::InternalError("malloc failed");
        }

        batch_block_addr_.append(resource);
    }

    return real_occupy;
}

int StreamCacheResource::maxBlockSize() const {
    return batch_block_addr_.maxBlockSize();
}

const BatchKVCacheBlockAddr& StreamCacheResource::kvCache() const {
    return batch_block_addr_;
}

void StreamCacheResource::setKVCache(const BatchKVCacheBlockAddr& kv_cache_block_addr) {
    batch_block_addr_ = kv_cache_block_addr;
}

}  // namespace rtp_llm
