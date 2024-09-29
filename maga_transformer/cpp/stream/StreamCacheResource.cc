#include "maga_transformer/cpp/stream/StreamCacheResource.h"
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "src/fastertransformer/core/BufferHelper.h"

using namespace std;
namespace ft = fastertransformer;

namespace rtp_llm {

void StreamCacheResource::init(int batch_size) {
    batch_block_addr_.resize(batch_size);
}

void StreamCacheResource::freeBatchBlocks(size_t batch_id, vector<int>& blocks) {
    if (blocks.size() == batch_block_addr_.blockSize(batch_id) && resource_context_.reuse_cache) {
        // TODO(xinfei.sxf) 一些场景调用了cancel的地方，其实并没有错误，也应该free with cache
        if (stream_->finished()) {
            auto tokens_id = stream_->completeTokenIdsVec(batch_id);
            vector<float> loss;
            if (stream_->getLoss()) {
                loss = ft::buffer2vector<float>(*(stream_->getLoss()));
            }
            resource_context_.cache_manager->freeWithCache(blocks, tokens_id, loss);
        }
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

absl::Status StreamCacheResource::releaseSequenceKVCache(size_t total_seq_len, size_t release_seq_len) {
    FT_LOG_DEBUG("stream [%ld] max block size is [%lu] total seq_len is [%lu], release [%lu] seq_len KVCache", stream_->streamId(), maxBlockSize(), total_seq_len, release_seq_len);
    size_t last_block_occupied_seq_len = (total_seq_len - 1) % seqSizePerBlock();
    if (release_seq_len < last_block_occupied_seq_len) {
        return absl::OkStatus();
    }
    size_t release_block_num = release_seq_len / seqSizePerBlock() + 1;
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
    auto current_block_size = maxBlockSize();
    if (current_block_size) {
        // partial fallback
        return incrKVBlock(token_capacity, reserve_step);
    }

    if (resource_context_.reuse_cache) {
        auto common_tokens_vec = stream_->commonCompleteTokenIdsVec();
        auto mm_bounds_vec = stream_->multimodalIntervals();
        auto match_info = resource_context_.cache_manager->mallocWithCache(common_tokens_vec, mm_bounds_vec);
        if (stream_->calculateLoss() && match_info.loss.empty()) {
            match_info = CacheManager::MatchInfo{0, {}, {}};
        }
        stream_->setReuseLength(match_info.reuse_length);
        if (!match_info.loss.empty()) {
            auto loss = ft::vector2Buffer<float>(match_info.loss);
            stream_->setLoss(*loss);
        }
        if (match_info.reuse_length) {
            batch_block_addr_.appendClone({match_info.cache_blocks}, resource_context_.cache_manager);
        }
    }

    return incrKVBlock(token_capacity, reserve_step);
}

absl::StatusOr<int> StreamCacheResource::incrKVBlock(int token_capacity, size_t reserve_step) {
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
    auto seq_len = stream_->isChunkStream() ? stream_->currentChunkLen() : (stream_->seqLength() + (int)reserve_step);
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
    batch_block_addr_.check();
    return batch_block_addr_;
}

void StreamCacheResource::setKVCache(const BatchKVCacheBlockAddr& kv_cache_block_addr) {
    batch_block_addr_ = kv_cache_block_addr;
}

void StreamCacheResource::beamSearchKvCacheUpdate(const std::vector<int>& beam_index) {
    auto kv_cache = kvCache();
    int batch_size = kv_cache.batchSize();
    int block_size = kv_cache.blocks(0).size();

    std::vector<int> src_block_offset(batch_size * block_size);
    std::vector<int> target_block_offset(batch_size * block_size);
    // check all batch has same block num
    for (int i = 0; i < batch_size; i ++) {
        FT_CHECK(block_size == kv_cache.blocks(i).size());
        for (int j = 0; j < block_size; j++) {
            src_block_offset[i * block_size + j] = kv_cache.blocks(i)[j];
            target_block_offset[i * block_size + j] = kv_cache.blocks(beam_index[i])[j];
        }
    }

    resource_context_.cache_manager->beamSearchKvUpdate(ft::vector2Buffer(src_block_offset),
                                                        ft::vector2Buffer(target_block_offset));
}

}  // namespace rtp_llm
