#include <algorithm>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache_new/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"

namespace rtp_llm {

MallocResult KVCacheAllocator::initMalloc(const MallocInfo& malloc_info) {
    auto init_result = initMallocForCommonLen(malloc_info);
    if (!init_result.success) {
        return init_result;
    }

    auto incr_result = incrMalloc(malloc_info);
    if (!incr_result.success) {
        return incr_result;
    } else {
        return init_result;
    }
}

MallocResult KVCacheAllocator::malloc(const MallocInfo& malloc_info) {
    if (!malloc_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return {false, 0};
    }

    if (!malloc_info.complete_token_ids) {
        RTP_LLM_LOG_ERROR("CompleteTokenIds is null");
        return {false, 0};
    }

    if (malloc_info.batch_kv_cache_resource->maxBlocksNum() == 0) {
        return initMalloc(malloc_info);
    } else {
        return incrMalloc(malloc_info);
    }
}

void KVCacheAllocator::blockCopy(int src_block_index, int dest_block_index) {
    BlockIdPair copy_mapping{src_block_index, dest_block_index};
    blockBatchCopy(&copy_mapping, &copy_mapping + 1);
}

void KVCacheAllocator::blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) {
    blockBatchCopy(copy_mapping.data(), copy_mapping.data() + copy_mapping.size());
}

void KVCacheAllocator::blockBatchCopy(const Buffer& copy_mapping) {
    RTP_LLM_CHECK(copy_mapping.dim() == 2 && copy_mapping.shape()[1] == 2);
    const auto* begin_ptr = (const BlockIdPair*)copy_mapping.data();
    size_t      copy_num  = copy_mapping.shape()[0];
    blockBatchCopy(begin_ptr, begin_ptr + copy_num);
}

void KVCacheAllocator::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
    using CopyType = BatchCopyParams::CopyType;

    if (end_ptr == begin_ptr) {
        return;
    }

    BatchCopyParams copy_params;

    const size_t copy_num = (end_ptr - begin_ptr) * config_.layer_num;

    size_t copy_nums[CopyType::TYPE_SIZE] = {};
    auto   copy_type                      = BatchCopyParams::get_copy_type(
        allocation_type_ == AllocationType::DEVICE ? rtp_llm::MEMORY_GPU : rtp_llm::MEMORY_CPU,
        allocation_type_ == AllocationType::DEVICE ? rtp_llm::MEMORY_GPU : rtp_llm::MEMORY_CPU);
    copy_nums[copy_type] += copy_num * 2;  // for k and v

    for (size_t i = 0; i < CopyType::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<CopyType>(i), copy_nums[i]);
    }

    auto&  spec         = config_.cache_specs[0];
    size_t k_block_size = spec->k_block_size() + spec->v_block_size();

    for (auto it = begin_ptr; it != end_ptr; ++it) {
        auto [src_block_index, dest_block_index] = *it;

        for (int layer_id = 0; layer_id < config_.layer_num; layer_id++) {
            auto src_addr_info = convertIndexToAddr(layer_id, src_block_index);
            auto dst_addr_info = convertIndexToAddr(layer_id, dest_block_index);

            if (!src_addr_info.k_addr || !dst_addr_info.k_addr) {
                RTP_LLM_LOG_ERROR("Failed to get block address for layer %d, src_block %d, dst_block %d",
                                  layer_id,
                                  src_block_index,
                                  dest_block_index);
                continue;
            }

            copy_params.add(dst_addr_info.k_addr, src_addr_info.k_addr, k_block_size, copy_type);
        }
    }

    device_->batchCopy(copy_params);
}

size_t KVCacheAllocator::freeBlocksNum() const {
    return block_pool_->freeBlocksNum();
}

int64_t KVCacheAllocator::getMrCostTimeMs() const {
    return block_pool_ ? block_pool_->getMrCostTimeMs() : 0;
}

size_t KVCacheAllocator::availableBlocksNum() const {
    return block_pool_->availableBlocksNum();
}

size_t KVCacheAllocator::availableTokensNum() const {
    return block_pool_->availableBlocksNum() * seqSizePerBlock();
}

size_t KVCacheAllocator::totalBlocksNum() const {
    return block_pool_->totalBlocksNum();
}

size_t KVCacheAllocator::maxAvailableTokensNum() const {
    return block_pool_->totalBlocksNum() * seqSizePerBlock();
}

KVCacheBuffer KVCacheAllocator::kvCacheBuffer() const {
    return block_pool_->kvCacheBuffer();
}

void KVCacheAllocator::regUserMr(size_t model_id) {
    if (block_pool_) {
        block_pool_->regUserMr(model_id);
    }
}

std::vector<std::pair<BufferPtr, size_t>> KVCacheAllocator::getAllBuffers() const {
    std::vector<std::pair<BufferPtr, size_t>> results;

    CacheLayerLayout layout = layerCacheBase();
    results.reserve(layout.layers_to_buffer_ptrs.size());

    for (const auto& buf : layout.layers_to_buffer_ptrs) {
        if (!buf || buf->sizeBytes() == 0) {
            continue;
        }
        size_t block_stride_bytes = config_.block_stride;
        results.emplace_back(buf, block_stride_bytes);
    }

    return results;
}

void KVCacheAllocator::clearCache() {
    if (block_pool_) {
        block_pool_->clearCache();
    }
}

}  // namespace rtp_llm
