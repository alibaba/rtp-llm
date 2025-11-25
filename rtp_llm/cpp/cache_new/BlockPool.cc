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
    torch::Tensor kv_cache_tensor = Buffer2torchTensor(cache_aligned_buffer_, false);

    layout_strategy_ = MemoryLayoutStrategyFactory::create(config_.layout);
    if (!layout_strategy_) {
        RTP_LLM_LOG_ERROR("Failed to create memory layout strategy");
        return false;
    }

    if (!layout_strategy_->init(config_, kv_cache_tensor, cache_base_ptr_, config_.dtype)) {
        RTP_LLM_LOG_ERROR("Failed to initialize memory layout strategy");
        return false;
    }

    initFreeBlocks();

    block_cache_ = std::make_shared<BlockCacheV1>();

    RTP_LLM_LOG_INFO("block pool init success with layout: %s",
                     config_.layout == LAYER_FIRST ? "LAYER_FIRST" : "KV_FIRST");
    return true;
}

BlockCacheV1Ptr BlockPool::blockCache() {
    return block_cache_;
}

void BlockPool::initFreeBlocks() {
    // block 0 is reserved
    for (BlockIdxType i = 1; i < static_cast<BlockIdxType>(config_.block_num); ++i) {
        free_block_ids_.insert(i);
    }
    all_ref_counter_.init(config_.block_num);
    request_ref_counter_.init(config_.block_num);
}

std::vector<torch::Tensor> BlockPool::layerCacheBase() const {
    if (!layout_strategy_) {
        RTP_LLM_LOG_ERROR("Layout strategy not initialized");
        return {};
    }
    return layout_strategy_->getLayerCacheTensors();
}

BlockIndicesType BlockPool::malloc(int num_blocks) {
    BlockIndicesType block_ids;
    block_ids.reserve(num_blocks);
    if (free_block_ids_.size() < static_cast<size_t>(num_blocks)) {
        RTP_LLM_LOG_WARNING(
            "Block pool only has %zu free blocks, cannot allocate %d blocks", free_block_ids_.size(), num_blocks);
        return {};
    }
    auto first = free_block_ids_.begin();
    auto last  = first;
    std::advance(last, num_blocks);
    for (auto it = first; it != last; ++it) {
        block_ids.push_back(*it);
    }
    free_block_ids_.erase(first, last);
    requestReference(block_ids);
    return block_ids;
}

void BlockPool::requestFree(BlockIdxType block_idx) {
    auto block_ids = {block_idx};
    freeImpl(block_ids);
}

void BlockPool::requestFree(const BlockIndicesType& block_ids) {
    freeImpl(block_ids);
    request_ref_counter_.decrementRefCounter(block_ids);
}

void BlockPool::blockCacheFree(BlockIdxType block_idx) {
    auto block_ids = {block_idx};
    freeImpl(block_ids);
}

void BlockPool::blockCacheFree(const BlockIndicesType& block_ids) {
    freeImpl(block_ids);
}

void BlockPool::freeImpl(const BlockIndicesType& block_ids) {
    all_ref_counter_.decrementRefCounter(block_ids);
    for (auto& block_id : block_ids) {
        if (all_ref_counter_.getRefCounter(block_id) == 0) {
            free_block_ids_.insert(block_id);
        }
    }
}

void BlockPool::requestReference(BlockIdxType block_idx) {
    BlockIndicesType block_ids = {block_idx};
    requestReference(block_ids);
}

void BlockPool::requestReference(const BlockIndicesType& block_ids) {
    request_ref_counter_.incrementRefCounter(block_ids);
    all_ref_counter_.incrementRefCounter(block_ids);
}

void BlockPool::blockCacheReference(BlockIdxType block_idx) {
    BlockIndicesType block_ids = {block_idx};
    blockCacheReference(block_ids);
}

void BlockPool::blockCacheReference(const BlockIndicesType& block_ids) {
    all_ref_counter_.incrementRefCounter(block_ids);
}

void BlockPool::regUserMr(size_t model_id) {
    if (device_->cacheStore() && !kvcache_reg_mr_) {
        RTP_LLM_LOG_INFO("start to register user mr");
        auto memory_util = std::static_pointer_cast<NormalCacheStore>(device_->cacheStore())->getMemoryUtil();

        auto   start_time_us     = currentTimeUs();
        size_t total_memory_size = config_.layer_num * config_.block_num * config_.block_size;

        if (!memory_util->regUserMr(cache_base_ptr_, config_.total_size, true, config_.block_size)) {
            RTP_LLM_FAIL("register user mr for block pool cache buffer failed");
        }

        auto cost_time_ms = (currentTimeUs() - start_time_us) / 1000;
        RTP_LLM_LOG_INFO(
            "register user mr for block pool cache buffer success: cost %ld ms, cache base address %p, len %lu",
            cost_time_ms,
            cache_base_ptr_,
            total_memory_size);
        mr_cost_time_ms_ += cost_time_ms;
        kvcache_reg_mr_ = true;
    }
}

void BlockPool::deregUserMr() {
    if (device_->cacheStore() && kvcache_reg_mr_) {
        RTP_LLM_LOG_INFO("start to deregister user mr");
        auto memory_util = std::static_pointer_cast<NormalCacheStore>(device_->cacheStore())->getMemoryUtil();
        if (!memory_util->deregUserMr(cache_base_ptr_, true)) {
            RTP_LLM_FAIL("deregister user mr for block pool cache buffer failed");
        }
        RTP_LLM_LOG_INFO("deregister user mr for block pool cache buffer success");
        kvcache_reg_mr_ = false;
    }
}

size_t BlockPool::freeBlocksNum() const {
    return free_block_ids_.size();
}

size_t BlockPool::totalBlocksNum() const {
    // reserve block 0 for internal use
    return config_.block_num - 1;
}

// Blocks not referenced by a request are free.
size_t BlockPool::availableBlocksNum() const {
    return request_ref_counter_.freeBlockNum();
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

BlockBufferPtrInfo BlockPool::convertIndexToBuffer(int layer_id, int block_id) const {
    if (!layout_strategy_) {
        RTP_LLM_LOG_ERROR("Layout strategy not initialized");
        return {nullptr, nullptr};
    }
    return layout_strategy_->convertIndexToBuffer(layer_id, block_id);
}

KVCacheBuffer BlockPool::kvCacheBuffer() const {
    if (!layout_strategy_) {
        RTP_LLM_LOG_ERROR("Layout strategy not initialized for kvCacheBuffer");
        return KVCacheBuffer{nullptr, nullptr, nullptr, nullptr};
    }
    return layout_strategy_->kvCacheBuffer();
}

MemoryType BlockPool::where() const {
    return cache_aligned_buffer_->where();
}

}  // namespace rtp_llm
