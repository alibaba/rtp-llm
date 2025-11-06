#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
BlockPool::BlockPool(const BlockPoolConfig& config, rtp_llm::DeviceBase* device, AllocationType atype):
    config_(config), device_(device), atype_(atype) {}

BlockPool::~BlockPool() {
    cache_aligned_buffer_.reset();
}

bool BlockPool::init() {
    cache_aligned_buffer_ = device_->allocateBuffer({rtp_llm::TYPE_INT8, {config_.total_size}, atype_});
    cache_base_ptr_       = cache_aligned_buffer_->data();
    if (cache_aligned_buffer_ == nullptr || cache_base_ptr_ == nullptr) {
        RTP_LLM_LOG_ERROR("block pool allocate cache aligned buffer is null");
        return false;
    }
    kv_cache_.kv_blocks = Buffer2torchTensor(cache_aligned_buffer_, false);

    // 创建布局策略
    layout_strategy_ = MemoryLayoutStrategyFactory::create(config_.layout);
    if (!layout_strategy_) {
        RTP_LLM_LOG_ERROR("Failed to create memory layout strategy");
        return false;
    }

    // 初始化布局策略
    if (!layout_strategy_->init(config_, kv_cache_.kv_blocks, cache_base_ptr_)) {
        RTP_LLM_LOG_ERROR("Failed to initialize memory layout strategy");
        return false;
    }

    initFreeBlocks();

    RTP_LLM_LOG_INFO("block pool init success with layout: %s",
                     config_.layout == LAYER_FIRST ? "LAYER_FIRST" : "KV_FIRST");
    return true;
}

// 初始化空闲块列表
void BlockPool::initFreeBlocks() {
    for (BlockIdxType i = 0; i < static_cast<BlockIdxType>(config_.block_num); ++i) {
        free_block_ids.insert(i);
    }
    block_ref_counter_.init(config_.block_num);
}

std::vector<torch::Tensor> BlockPool::layerCacheBase() const {
    if (!layout_strategy_) {
        RTP_LLM_LOG_ERROR("Layout strategy not initialized");
        return {};
    }
    return layout_strategy_->getLayerCacheTensors();
}

std::vector<BlockIdxType> BlockPool::alloc(int num_blocks) {
    std::vector<BlockIdxType> block_ids;
    block_ids.reserve(num_blocks);
    if (free_block_ids.size() < static_cast<size_t>(num_blocks)) {
        RTP_LLM_LOG_DEBUG(
            "Block pool only has %zu free blocks, cannot allocate %d blocks", free_block_ids.size(), num_blocks);
        return {};
    }
    for (int i = 0; i < num_blocks; ++i) {
        auto it = free_block_ids.begin();
        if (it == free_block_ids.end())
            break;
        block_ids.push_back(*it);
        free_block_ids.erase(it);
    }
    return block_ids;
}

void BlockPool::free(const std::vector<BlockIdxType>& block_ids) {
    block_ref_counter_.decrementRefCounter(block_ids);
    for (auto& block_id : block_ids) {
        if (block_ref_counter_.getRefCounter(block_id) == 0) {
            free_block_ids.insert(block_id);
        }
    }
}

void BlockPool::reference(const std::vector<BlockIdxType>& block_ids) {
    block_ref_counter_.incrementRefCounter(block_ids);
}

size_t BlockPool::freeBlockNums() const {
    return free_block_ids.size();
}

BlockAddrInfo BlockPool::convertIndexToAddr(int layer_id, int block_id) const {
    if (!layout_strategy_) {
        RTP_LLM_LOG_ERROR("Layout strategy not initialized");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    return layout_strategy_->convertIndexToAddr(layer_id, block_id);
}

void* BlockPool::getKCacheAddr(int layer_id, int block_id) const {
    if (!layout_strategy_) {
        RTP_LLM_LOG_ERROR("Layout strategy not initialized");
        return nullptr;
    }
    return layout_strategy_->getKCacheAddr(layer_id, block_id);
}

void* BlockPool::getVCacheAddr(int layer_id, int block_id) const {
    if (!layout_strategy_) {
        RTP_LLM_LOG_ERROR("Layout strategy not initialized");
        return nullptr;
    }
    return layout_strategy_->getVCacheAddr(layer_id, block_id);
}

BlockBufferInfo BlockPool::convertIndexToBuffer(int layer_id, int block_id) const {
    if (!layout_strategy_) {
        RTP_LLM_LOG_ERROR("Layout strategy not initialized");
        return {nullptr, nullptr};
    }
    return layout_strategy_->convertIndexToBuffer(layer_id, block_id);
}

BufferPtr BlockPool::cacheBuffer() const {
    return cache_aligned_buffer_;
}

}  // namespace rtp_llm