#include "rtp_llm/cpp/cache_new/FullKVCacheGroup.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool FullKVCacheGroup::init() {
    auto layer_tensors = block_pool_->layerCacheBase();

    // TODO(chanyin): layer_ids might not be set in sequence, move to basic class
    for (int i = 0; i < layer_ids_.size(); ++i) {
        gloabl_layer_to_kv_tensors[layer_ids_[i]]  = layer_tensors[i];
        gloabl_layer_to_local_layer[layer_ids_[i]] = i;
    }

    return true;
}

int FullKVCacheGroup::needBlocksNum(int seq_len, int current_blocks) const {
    return std::max((seq_len + seq_size_per_block_ - 1) / seq_size_per_block_ - current_blocks, 0);
}

void FullKVCacheGroup::malloc(CacheKeysType& cache_keys, BlockIndicesType& block_indices, int seq_len) {
    int new_blocks = needBlocksNum(seq_len, block_indices.size());

    auto result = block_pool_->malloc(new_blocks);
    // check aloc error
    block_indices.insert(block_indices.end(), result.begin(), result.end());

    // insert to cache right row, for reuse in batch query
}

MatchResult FullKVCacheGroup::match(CacheKeysType& cache_keys) {
    MatchResult final_result;
    final_result.reuse_length = 0;

    for (auto& cache_key : cache_keys) {
        auto result = block_cache_->match(cache_key);
        if (!isNullBlockIdx(result.matched_index)) {
            final_result.reuse_length++;
            final_result.block_indices.push_back(result.matched_index);
        }
    }

    final_result.reuse_length *= seqSizePerBlock();

    return final_result;
}

void FullKVCacheGroup::free(const BlockIndicesType& block_indices) {
    if (!block_pool_) {
        RTP_LLM_LOG_ERROR("Block pool is not initialized");
        return;
    }

    if (block_indices.empty()) {
        return;
    }

    block_pool_->free(block_indices);
    RTP_LLM_LOG_DEBUG("Freed %zu blocks", block_indices.size());
}

void FullKVCacheGroup::insertIntoCache(CacheKeysType& cache_keys, BlockIndicesType& block_indices, bool is_resident) {
    if (!block_cache_) {
        RTP_LLM_LOG_DEBUG("Block cache is not initialized, skip insertion");
        return;
    }

    if (cache_keys.empty()) {
        return;
    }

    if (cache_keys.size() != block_indices.size()) {
        RTP_LLM_LOG_ERROR(
            "Cache keys size (%zu) doesn't match block indices size (%zu)", cache_keys.size(), block_indices.size());
        return;
    }

    for (size_t i = 0; i < cache_keys.size(); ++i) {
        BlockCacheV1::CacheItem item;
        item.cache_key   = cache_keys[i];
        item.block_index = block_indices[i];
        item.is_resident = is_resident;
        block_cache_->put(item);
    }

    RTP_LLM_LOG_DEBUG("Inserted %zu blocks into cache", block_indices.size());
}

void FullKVCacheGroup::removeSkippedBlocks(BlockIndicesType& block_indices) {}

size_t FullKVCacheGroup::freeBlockNums() const {
    return block_pool_->freeBlockNums();
}

std::unordered_map<int, torch::Tensor> FullKVCacheGroup::layerCacheBase() const {
    return gloabl_layer_to_kv_tensors;
}

BlockAddrInfo FullKVCacheGroup::convertIndexToAddr(int layer_id, int block_id) const {
    int local_layer_id = gloabl_layer_to_local_layer.at(layer_id);
    return block_pool_->convertIndexToAddr(local_layer_id, block_id);
}

BlockBufferInfo FullKVCacheGroup::convertIndexToBuffer(int layer_id, int block_id) const {
    int local_layer_id = gloabl_layer_to_local_layer.at(layer_id);
    return block_pool_->convertIndexToBuffer(local_layer_id, block_id);
}

}  // namespace rtp_llm
