#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/HashUtil.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

using namespace std;

namespace rtp_llm {

void StreamCacheResource::init(int batch_size) {
    batch_resource_.resize(batch_size);
    constructCacheKey();
}

void StreamCacheResource::freeBatchBlocks(size_t batch_id, vector<int>& blocks) {
    if (blocks.empty()) {
        return;
    }
    // only reuse the cache of batch 0 of finished beam search streams
    // to avoid corrupted states and excessive block cache entries
    bool should_reuse_cache =
        reuseCache() && (!stream_->hasNumBeams() || (!stream_->stoppedWithoutLock() && batch_id == 0));
    // TODO(zhangjianning.zjn) cache all beams of beam search
    if (blocks.size() == batch_resource_.blockSize(batch_id) && should_reuse_cache
        && (stream_->finishedWithoutLock() || stream_->isRemoteRunningWithoutLock())) {
        reConstructCacheKeys();
        auto          tokens_id  = stream_->completeTokenIdsVec(batch_id);
        const auto&   cache_keys = stream_->cacheKeys(batch_id);
        vector<float> loss;
        if (stream_->getLoss()) {
            loss = rtp_llm::buffer2vector<float>(*(stream_->getLoss()));
        }
        CacheManager::FreeInfo free_info(stream_->streamId(),
                                         tokens_id,
                                         cache_keys,
                                         blocks,
                                         loss,
                                         adapter_name_,
                                         enable3FS(),
                                         enableMemoryBlockCache());
        resource_context_.cache_manager->freeWithCache(free_info);
    } else {
        resource_context_.cache_manager->free(blocks);
    }
}

void StreamCacheResource::releaseResource() {
    if (!resource_context_.cache_manager) {
        return;
    }
    // do not reuse cache from stopped beam search streams, whose states are likely corrupted
    if (!need_release_resource_ && (!stream_->hasNumBeams() || !stream_->stoppedWithoutLock())) {
        reConstructCacheKeys();
        return;
    }
    tryReleaseKVBlock(maxBlockSize());
    batch_resource_.clear();
}

int StreamCacheResource::tryReleaseKVBlock(size_t nums) {
    RTP_LLM_LOG_DEBUG("stream [%ld] try release [%lu] blocks", stream_->streamId(), nums);
    size_t release_blocks_num = 0;
    size_t reserved_blocks    = 0;
    if (fake_inited_) {
        int max_block_size = maxBlockSize();
        int batch_size     = batch_resource_.batchSize();
        batch_resource_.clear();
        batch_resource_.resize(batch_size);
        fake_inited_ = false;
        return max_block_size;
    }
    // NOTE: all batch has same number of blocks
    for (size_t batch_id = 0; batch_id < batch_resource_.batchSize(); batch_id++) {
        const auto& blocks = batch_resource_.blocks(batch_id);
        reserved_blocks    = std::max(0, int(blocks.size()) - int(nums));
        release_blocks_num = blocks.size() - reserved_blocks;
        vector<int> release_blocks(blocks.begin() + reserved_blocks, blocks.end());
        freeBatchBlocks(batch_id, release_blocks);
        batch_resource_.shrink(batch_id, reserved_blocks);
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
absl::Status StreamCacheResource::initKVBlock(size_t reserve_step) {
    auto current_block_size = maxBlockSize();
    if (current_block_size) {
        return incrKVBlock(reserve_step);
    }

    if (reuseCache()) {
        auto common_tokens_vec = stream_->commonCompleteTokenIdsVec();
        auto common_cache_keys = stream_->cacheKeys(0);
        auto mm_bounds_vec     = stream_->multimodalIntervals();
        // TODO(xinfei.sxf) fix need loss param
        CacheManager::AdvancedMallocInfo malloc_info(stream_->streamId(),
                                                     common_tokens_vec,
                                                     common_cache_keys,
                                                     mm_bounds_vec,
                                                     false,
                                                     false,
                                                     adapter_name_,
                                                     enable3FS(),
                                                     enableMemoryBlockCache());
        auto                             match_info = resource_context_.cache_manager->mallocWithCache(malloc_info);
        if (stream_->calculateLoss() && match_info.loss.empty()) {
            match_info = CacheManager::MatchInfo{0, {}, {}};
        }
        stream_->setReuseLength(match_info.reuse_length);
        stream_->setLocalReuseLength(match_info.local_reuse_length);
        stream_->setRemoteReuseLength(match_info.remote_reuse_length);
        stream_->setMtpTokenIndex(match_info.reuse_length);
        stream_->setInitialReuseLength(match_info.reuse_length);
        if (!match_info.loss.empty()) {
            auto loss = rtp_llm::vector2Buffer<float>(match_info.loss);
            stream_->setLoss(*loss);
        }
        if (match_info.reuse_length) {
            batch_resource_.appendClone({match_info.cache_blocks}, resource_context_.cache_manager);
        }
    }

    return incrKVBlock(reserve_step);
}

absl::Status StreamCacheResource::incrKVBlock(size_t reserve_step) {
    // TODO(xinfei.sxf) add reserver_blocks
    if (fake_inited_) {
        return absl::InternalError("fake inited not allow to incr block");
    }
    auto seq_len            = stream_->seqLength() + (int)reserve_step;
    auto common_seq_len     = std::min(seq_len, stream_->adjustedCommonLen());
    auto common_blocks_nums = singleBatchNeedBlocks(common_seq_len);

    bool verbose = malloc_failed_times_ >= 10 ? malloc_failed_times_ % 100 == 0 : true;
    auto [success, kv_cache_resource] =
        resource_context_.cache_manager->malloc({stream_->streamId(), (uint32_t)common_blocks_nums, verbose});
    if (!success) {
        malloc_failed_times_++;
        return absl::InternalError("malloc failed");
    }
    batch_resource_.appendClone(kv_cache_resource, resource_context_.cache_manager);

    auto extra_blocks_num = singleBatchNeedBlocks(seq_len);
    if (extra_blocks_num <= 0) {
        return absl::OkStatus();
    }

    auto batch_size   = stream_->currentBatchSize();
    auto total_blocks = batch_size * extra_blocks_num;
    std::tie(success, kv_cache_resource) =
        resource_context_.cache_manager->malloc({stream_->streamId(), (uint32_t)total_blocks, verbose});
    if (success) {
        const auto&                  all_blocks = kv_cache_resource.block_id;
        std::vector<KVCacheResource> resource;
        for (uint32_t i = 0; i < batch_size; i++) {
            auto blocks = std::vector<int32_t>(all_blocks.begin() + i * extra_blocks_num,
                                               all_blocks.begin() + (i + 1) * extra_blocks_num);
            resource.push_back(KVCacheResource(blocks));
        }
        batch_resource_.append(resource);
    } else {
        malloc_failed_times_++;
        return absl::InternalError("malloc failed");
    }

    return absl::OkStatus();
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

bool StreamCacheResource::updateKVBlock(const std::vector<int>& block_src_batch, bool copy_last_block) {
    block_update_mapping_.clear();

    if (!resource_context_.cache_manager || block_src_batch.size() == 0) {
        return true;
    }

    // collect forking count for all old batches
    auto        old_batch_size = batch_resource_.batchSize();
    auto        new_batch_size = block_src_batch.size();
    vector<int> batch_fork_count(old_batch_size, 0);
    for (const auto& old_batch_idx : block_src_batch) {
        RTP_LLM_CHECK_WITH_INFO(old_batch_idx < old_batch_size,
                                "try to reuse an old batch %d that out of range %d",
                                old_batch_idx,
                                old_batch_size);
        ++batch_fork_count[old_batch_idx];
    }

    // collect free and malloc infos of kv cache blocks
    // TODO(zhangjianning.zjn): might be possible to repurpose disused blocks for new batches?
    vector<int> disused_kv_blocks;
    uint32_t    num_new_blocks = 0;
    for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
        int fork_count = batch_fork_count[old_batch_idx];

        if (fork_count == 0) {
            const auto& blocks = batch_resource_.batch_block_id[old_batch_idx];
            disused_kv_blocks.insert(disused_kv_blocks.end(), blocks.begin(), blocks.end());
        } else if (fork_count > 1 && copy_last_block) {
            num_new_blocks += fork_count - 1;
        }
    }

    // check kv cache capacity
    size_t available_blocks = resource_context_.cache_manager->availableBlockNums()
                              + resource_context_.cache_manager->newFreeBlocks(disused_kv_blocks);
    if (available_blocks < num_new_blocks) {
        RTP_LLM_LOG_WARNING(
            "no enough available blocks for kv cache update of stream %lld, need %llu, but only %llu remained",
            stream_->streamId(),
            num_new_blocks,
            available_blocks);
        return false;
    }

    // do free and malloc
    if (disused_kv_blocks.size() > 0) {
        resource_context_.cache_manager->free(disused_kv_blocks);
    }
    std::vector<int> new_blocks;
    if (num_new_blocks > 0) {
        auto [malloc_status, cache_resource] =
            resource_context_.cache_manager->malloc({stream_->streamId(), num_new_blocks});
        RTP_LLM_CHECK_WITH_INFO(malloc_status,
                                "failed to malloc %u new blocks during kv cache update of stream %lld",
                                num_new_blocks,
                                stream_->streamId());
        new_blocks = std::move(cache_resource.block_id);
    }

    // increase ref count of shared blocks
    for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
        if (batch_fork_count[old_batch_idx] <= 1) {
            // no shared blocks
            continue;
        }

        auto& batch_blocks = batch_resource_.batch_block_id[old_batch_idx];

        // need to exclude last block if it is not shared
        const bool exclude_last_block = copy_last_block && batch_blocks.size() > 0;

        int last_block;
        if (exclude_last_block) {
            last_block = batch_blocks.back();
            batch_blocks.pop_back();
        }

        // TODO(zhangjianning.zjn): would be better to pass the ref increment directly
        if (batch_blocks.size() > 0) {
            for (int i = 1; i < batch_fork_count[old_batch_idx]; ++i) {
                resource_context_.cache_manager->incrRefCounter(batch_blocks);
            }
        }

        if (exclude_last_block) {
            batch_blocks.push_back(last_block);
        }
    }

    // generate update mapping for block ids
    vector<vector<int32_t>> old_block_ids = std::move(batch_resource_.batch_block_id);
    batch_resource_.batch_block_id.reserve(new_batch_size);
    vector<vector<int64_t>> old_cache_keys = std::move(batch_resource_.cache_keys);
    batch_resource_.cache_keys.reserve(new_batch_size);
    block_update_mapping_.reserve(num_new_blocks);
    for (int new_batch_idx = 0; new_batch_idx < new_batch_size; ++new_batch_idx) {
        int   old_batch_idx = block_src_batch[new_batch_idx];
        auto& fork_count    = batch_fork_count[old_batch_idx];
        RTP_LLM_CHECK_WITH_INFO(fork_count > 0, "old batch %d has been forked too many times", old_batch_idx);
        if (fork_count == 1) {
            // move from old blocks directly
            batch_resource_.batch_block_id.emplace_back(std::move(old_block_ids[old_batch_idx]));
            batch_resource_.cache_keys.emplace_back(std::move(old_cache_keys[old_batch_idx]));
        } else {
            // copy from old blocks
            batch_resource_.batch_block_id.emplace_back(old_block_ids[old_batch_idx]);
            batch_resource_.cache_keys.emplace_back(old_cache_keys[old_batch_idx]);
            auto& blocks = batch_resource_.batch_block_id.back();
            if (copy_last_block && blocks.size() > 0) {
                int old_block = blocks.back();
                blocks.pop_back();

                int new_block = new_blocks.back();
                new_blocks.pop_back();
                blocks.push_back(new_block);

                block_update_mapping_.push_back(BlockIdPair{old_block, new_block});
            }
        }
        --fork_count;
    }

    return true;
}

// TODO(xinfei.sxf) move code to batch resource class
void StreamCacheResource::constructCacheKey() {
    batch_resource_.cache_keys.resize(stream_->currentBatchSize());
    if (!resource_context_.cache_manager) {
        return;
    }
    if (!reuseCache() && !resource_context_.use_cache_store) {
        return;
    }
    for (size_t i = 0; i < stream_->currentBatchSize(); i++) {
        batch_resource_.cache_keys[i].reserve(singleBatchNeedBlocks(stream_->max_seq_len_));
    }
    auto    seq_size_per_block = seqSizePerBlock();
    auto    token_ids          = stream_->generate_input_->input_ids->data<int32_t>();
    int64_t hash               = 0;
    last_block_aligned_        = stream_->seqLength() % seq_size_per_block == 0 ? true : false;
    int32_t end_block_index    = singleBatchNeedBlocks(stream_->seqLength());
    for (int index = 0; index < end_block_index; index++) {
        auto pos = index * seq_size_per_block;
        hash =
            hashInt64Array(hash, token_ids + pos, token_ids + std::min(pos + seq_size_per_block, stream_->seqLength()));
        batch_resource_.cache_keys[0].push_back(hash);
    }
    for (size_t i = 1; i < stream_->currentBatchSize(); i++) {
        batch_resource_.cache_keys[i] = batch_resource_.cache_keys[0];
    }
}

void StreamCacheResource::reConstructCacheKeys() {
    if (!resource_context_.cache_manager) {
        return;
    }
    if (!reuseCache() && !resource_context_.use_cache_store) {
        return;
    }
    auto seq_size_per_block = seqSizePerBlock();
    auto total_blocks       = stream_->seqLength() / seq_size_per_block;
    for (size_t i = 0; i < stream_->currentBatchSize(); ++i) {
        if (!last_block_aligned_ && !batch_resource_.cache_keys[i].empty()) {
            batch_resource_.cache_keys[i].pop_back();
        }
        auto    token_ids = stream_->complete_token_ids_->data(i);
        int64_t hash      = batch_resource_.cache_keys[i].empty() ? 0 : batch_resource_.cache_keys[i].back();
        for (int index = batch_resource_.cache_keys[i].size(); index < total_blocks; index++) {
            auto pos = index * seq_size_per_block;
            hash     = hashInt64Array(hash, token_ids + pos, token_ids + pos + seq_size_per_block);
            batch_resource_.cache_keys[i].push_back(hash);
        }
    }

    last_block_aligned_ = true;
}

bool StreamCacheResource::hasCacheKeys() const {
    return !batch_resource_.cache_keys.empty();
}

const std::vector<int64_t>& StreamCacheResource::cacheKeys(int32_t batch_id) const {
    RTP_LLM_CHECK_WITH_INFO(batch_resource_.cache_keys.size() > batch_id, "cache_keys size is <= batch_id");
    return batch_resource_.cache_keys[batch_id];
}

void StreamCacheResource::fakeInitKVBlock() {
    fake_inited_ = true;
    batch_resource_.resize(stream_->maxBatchSize());
    for (size_t i = 0; i < stream_->maxBatchSize(); i++) {
        batch_resource_.resize(i, stream_->seqLength(), true);
    }
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
