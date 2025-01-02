#include "maga_transformer/cpp/stream/StreamCacheResource.h"
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/utils/HashUtil.h"
#include "src/fastertransformer/core/BufferHelper.h"

using namespace std;
namespace ft = fastertransformer;

namespace rtp_llm {

void StreamCacheResource::init(int batch_size) {
    batch_resource_.resize(batch_size);
    constructCacheKey();
}

void StreamCacheResource::freeBatchBlocks(size_t batch_id, vector<int>& blocks) {
    if (blocks.empty()) {
        return;
    }
    if (blocks.size() == batch_resource_.blockSize(batch_id) && resource_context_.reuse_cache) {
        reConstructCacheKeys();
        auto tokens_id = stream_->completeTokenIdsVec(batch_id);
        const auto& cache_keys = stream_->cacheKeys(batch_id);
        vector<float> loss;
        if (stream_->getLoss()) {
            loss = ft::buffer2vector<float>(*(stream_->getLoss()));
        }
        // TODO(xinfei.sxf) 一些场景调用了cancel的地方，是否应该free with cache
        CacheManager::FreeInfo free_info(stream_->streamId(), tokens_id, cache_keys, blocks, loss);
        resource_context_.cache_manager->freeWithCache(free_info);
    } else {
        resource_context_.cache_manager->free(blocks);
    }
}

void StreamCacheResource::releaseResource() {
    if (!resource_context_.cache_manager) {
        return;
    }
    if (!need_release_resource_) {
        reConstructCacheKeys();
        return ;
    }
    tryReleaseKVBlock(maxBlockSize());
    batch_resource_.clear();
}

int StreamCacheResource::tryReleaseKVBlock(size_t nums) {
    FT_LOG_DEBUG("stream [%ld] try release [%lu] blocks", stream_->streamId(), nums);
    size_t release_blocks_num = 0;
    size_t reserved_blocks = 0;

    // NOTE: all batch has same number of blocks
    for (size_t batch_id = 0; batch_id < batch_resource_.batchSize(); batch_id++) {
        const auto& blocks = batch_resource_.blocks(batch_id);
        reserved_blocks = std::max(0, int(blocks.size()) - int(nums));
        release_blocks_num = blocks.size() - reserved_blocks;
        vector<int> release_blocks(blocks.begin() + reserved_blocks, blocks.end());
        freeBatchBlocks(batch_id, release_blocks);
        batch_resource_.resize(batch_id, reserved_blocks);
    }
    stream_->setFallbackPrefixLength(reserved_blocks * seqSizePerBlock());
    if (stream_->enable_fast_gen_) {
        stream_->resetChunkLen(reserved_blocks * seqSizePerBlock(), stream_->seqLength());
    }
    return release_blocks_num;
}

absl::Status StreamCacheResource::releaseSequenceKVCache(size_t total_seq_len, size_t release_seq_len) {
    FT_LOG_DEBUG("stream [%ld] max block size is [%lu] total seq_len is [%lu], release [%lu] seq_len KVCache", stream_->streamId(), maxBlockSize(), total_seq_len, release_seq_len);
    size_t last_block_occupied_seq_len = seqSizePerBlock() == 1 ? 1 : ((total_seq_len + seqSizePerBlock() - 2) % seqSizePerBlock() + 1);
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
        // TODO(xinfei.sxf) fix cache keys in fallback case
        auto common_cache_keys = stream_->cacheKeys(0);
        auto mm_bounds_vec = stream_->multimodalIntervals();
        // TODO(xinfei.sxf) fix need loss param
        CacheManager::MallocInfo malloc_info(stream_->streamId(), common_tokens_vec, common_cache_keys, mm_bounds_vec);
        auto match_info = resource_context_.cache_manager->mallocWithCache(malloc_info);
        if (stream_->calculateLoss() && match_info.loss.empty()) {
            match_info = CacheManager::MatchInfo{0, {}, {}};
        }
        stream_->setReuseLength(match_info.reuse_length);
        if (!match_info.loss.empty()) {
            auto loss = ft::vector2Buffer<float>(match_info.loss);
            stream_->setLoss(*loss);
        }
        if (match_info.reuse_length) {
            batch_resource_.appendClone({match_info.cache_blocks}, resource_context_.cache_manager);
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

    auto [success, kv_cache_resource] = resource_context_.cache_manager->malloc(stream_->streamId(), common_blocks_nums);
    if (!success) {
        return absl::InternalError("malloc failed");
    }
    batch_resource_.appendClone(kv_cache_resource, resource_context_.cache_manager);

    auto extra_blocks_num = singleBatchNeedBlocks(seq_len);
    if (extra_blocks_num <= 0) {
        return real_occupy;
    }

    auto batch_size  = stream_->tileNum();
    auto total_blocks = batch_size * extra_blocks_num;
    std::tie(success, kv_cache_resource) = resource_context_.cache_manager->malloc(stream_->streamId(), total_blocks);
    if (success) {
        const auto& all_blocks = kv_cache_resource.block_id;
        std::vector<KVCacheResource> resource;
        for (uint32_t i = 0; i < batch_size; i++) {
            auto blocks = std::vector<int32_t>(all_blocks.begin() + i * extra_blocks_num,
                                                all_blocks.begin() + (i + 1) * extra_blocks_num);
            resource.push_back(KVCacheResource(blocks));
        }
        batch_resource_.append(resource);
    } else {
        return absl::InternalError("malloc failed");
    }

    return real_occupy;
}

int StreamCacheResource::maxBlockSize() const {
    return batch_resource_.maxBlockSize();
}

const BatchKVCacheResource& StreamCacheResource::kvCache() const {
    batch_resource_.check();
    return batch_resource_;
}

void StreamCacheResource::setKVCache(const BatchKVCacheResource& kv_cache_resource) {
    batch_resource_ = kv_cache_resource;
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

// TODO(xinfei.sxf) move code to batch resource class
void StreamCacheResource::constructCacheKey() {
    batch_resource_.cache_keys.resize(stream_->tileNum());
    if (!resource_context_.cache_manager) {
        return;
    }
    if (!resource_context_.reuse_cache && !resource_context_.use_cache_store) {
        return;
    }
    for (size_t i = 0; i < stream_->tileNum(); i++) {
        batch_resource_.cache_keys[i].reserve(singleBatchNeedBlocks(stream_->max_seq_len_));
    }
    auto seq_size_per_block = seqSizePerBlock();
    auto token_ids = stream_->generate_input_->input_ids->data<int32_t>();
    int64_t hash = 0;
    last_block_aligned_ = stream_->seqLength() % seq_size_per_block == 0 ? true : false;
    int32_t end_block_index = singleBatchNeedBlocks(stream_->seqLength());
    for (int index = 0; index < end_block_index; index++) {
        auto pos = index * seq_size_per_block;
        hash = hashInt64Array(hash, token_ids + pos, token_ids + std::min(pos + seq_size_per_block, stream_->seqLength()));
        batch_resource_.cache_keys[0].push_back(hash);
    }
    for (size_t i = 1; i < stream_->tileNum(); i++) {
        batch_resource_.cache_keys[i] = batch_resource_.cache_keys[0];
    }
}

void StreamCacheResource::reConstructCacheKeys() {
    if (!resource_context_.cache_manager) {
        return;
    }
    if (!resource_context_.reuse_cache && !resource_context_.use_cache_store) {
        return;
    }
    auto seq_size_per_block = seqSizePerBlock();
    auto total_blocks = stream_->seqLength() / seq_size_per_block;
    for (size_t i = 0; i < stream_->tileNum(); ++i) {
        if (!last_block_aligned_ && !batch_resource_.cache_keys[i].empty()) {
            batch_resource_.cache_keys[i].pop_back();
        }
        auto token_ids = stream_->complete_token_ids_->data(i);
        int64_t hash = batch_resource_.cache_keys[i].empty() ? 0 : batch_resource_.cache_keys[i].back();
        for (int index = batch_resource_.cache_keys[i].size(); index < total_blocks; index++) {
            auto pos = index * seq_size_per_block;
            hash = hashInt64Array(hash, token_ids + pos, token_ids + pos + seq_size_per_block);
            batch_resource_.cache_keys[i].push_back(hash);
        }
    }

    last_block_aligned_ = true;
}

bool StreamCacheResource::hasCacheKeys() const {
    return !batch_resource_.cache_keys.empty();
}

const std::vector<int64_t>& StreamCacheResource::cacheKeys(int32_t batch_id) const {
    FT_CHECK_WITH_INFO(batch_resource_.cache_keys.size() > batch_id, "cache_keys size is <= batch_id");
    return batch_resource_.cache_keys[batch_id];
}

}  // namespace rtp_llm
