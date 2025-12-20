#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

BlockPool::BlockPool(const BlockPoolConfig& config, rtp_llm::DeviceBase* device, AllocationType allocation_type):
    config_(config), device_(device), allocation_type_(allocation_type) {}

BlockPool::~BlockPool() {
    cache_aligned_buffer_.reset();
}

bool BlockPool::init() {
    cache_aligned_buffer_ = device_->allocateBuffer({rtp_llm::TYPE_INT8, {config_.total_size_bytes}, allocation_type_});
    cache_base_ptr_       = cache_aligned_buffer_->data();
    if (cache_aligned_buffer_ == nullptr || cache_base_ptr_ == nullptr) {
        RTP_LLM_LOG_ERROR("block pool allocate cache aligned buffer is null");
        return false;
    }
    torch::Tensor full_tensor   = Buffer2torchTensor(cache_aligned_buffer_, false);
    const int64_t kv_pool_bytes = static_cast<int64_t>(
        config_.kv_block_pool_size_bytes > 0 ? config_.kv_block_pool_size_bytes : config_.total_size_bytes);
    torch::Tensor kv_cache_tensor =
        (kv_pool_bytes < full_tensor.numel()) ? full_tensor.narrow(0, 0, kv_pool_bytes) : full_tensor;

    torch::Tensor kv_scale_tensor;
    if (config_.enable_kv_scale && config_.kv_scale_pool_size_bytes > 0) {
        const int64_t scale_off   = static_cast<int64_t>(config_.kv_scale_offset_bytes);
        const int64_t scale_bytes = static_cast<int64_t>(config_.kv_scale_pool_size_bytes);
        RTP_LLM_CHECK_WITH_INFO(scale_off + scale_bytes <= full_tensor.numel(),
                                "kv_scale tensor out of range: off=%ld bytes=%ld full=%ld",
                                scale_off,
                                scale_bytes,
                                full_tensor.numel());
        kv_scale_tensor = full_tensor.narrow(0, scale_off, scale_bytes);
    }

    layout_strategy_ = MemoryLayoutStrategyFactory::create(config_.layout);
    RTP_LLM_CHECK_WITH_INFO(layout_strategy_ != nullptr, "Failed to create memory layout strategy");

    RTP_LLM_CHECK_WITH_INFO(
        layout_strategy_->init(config_, kv_cache_tensor, kv_scale_tensor, cache_base_ptr_, config_.dtype),
        "Failed to initialize memory layout strategy");

    initFreeBlocks();

    auto kv_cache = kvCacheBuffer();

#if (defined(USING_ROCM) && USING_ROCM) || (defined(USING_CUDA) && USING_CUDA)
    if (kv_cache.kv_blocks) {
        device_->bufMemset(*kv_cache.kv_blocks, 0);
    }
#endif

    block_cache_ = std::make_shared<BlockCache>();

    RTP_LLM_LOG_INFO("block pool init success");
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
    return layout_strategy_->getLayerCacheTensors();
}

std::vector<torch::Tensor> BlockPool::layerScaleCacheBase() const {
    return layout_strategy_->getLayerScaleCacheTensors();
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
    auto last  = std::next(first, num_blocks);
    block_ids.assign(first, last);
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

        auto start_time_us = currentTimeUs();

        // aligned_size should match kv block stride bytes (not including separated scale pool).
        if (!memory_util->regUserMr(cache_base_ptr_, config_.total_size_bytes, true, config_.kv_block_stride_bytes)) {
            RTP_LLM_FAIL("register user mr for block pool cache buffer failed");
        }

        auto cost_time_ms = (currentTimeUs() - start_time_us) / 1000;
        RTP_LLM_LOG_INFO(
            "register user mr for block pool cache buffer success: cost %ld ms, cache base address %p, len %lu",
            cost_time_ms,
            cache_base_ptr_,
            config_.total_size);
        mr_cost_time_ms_ += cost_time_ms;

        // Register scale pool mr separately when enabled.
        if (config_.enable_kv_scale && config_.kv_scale_pool_size_bytes > 0 && config_.kv_scale_block_bytes > 0) {
            void* scale_base_ptr      = static_cast<void*>(static_cast<char*>(cache_base_ptr_)
                                                      + static_cast<ptrdiff_t>(config_.kv_scale_offset_bytes));
            auto  start_scale_time_us = currentTimeUs();
            // aligned_size should match one scale block bytes (ONE of {K,V}).
            if (!memory_util->regUserMr(
                    scale_base_ptr, config_.kv_scale_pool_size_bytes, true, config_.kv_scale_block_bytes)) {
                RTP_LLM_FAIL("register user mr for block pool kv scale buffer failed");
            }
            auto scale_cost_time_ms = (currentTimeUs() - start_scale_time_us) / 1000;
            RTP_LLM_LOG_INFO(
                "register user mr for block pool kv scale buffer success: cost %ld ms, scale base address %p, len %lu",
                scale_cost_time_ms,
                scale_base_ptr,
                config_.kv_scale_pool_size_bytes);
            mr_cost_time_ms_ += scale_cost_time_ms;
        }

        kvcache_reg_mr_ = true;
    }
}

void BlockPool::deregUserMr() {
    if (device_->cacheStore() && kvcache_reg_mr_) {
        RTP_LLM_LOG_INFO("start to deregister user mr");
        auto memory_util = std::static_pointer_cast<NormalCacheStore>(device_->cacheStore())->getMemoryUtil();
        if (config_.enable_kv_scale && config_.kv_scale_pool_size_bytes > 0 && config_.kv_scale_block_bytes > 0) {
            void* scale_base_ptr = static_cast<void*>(static_cast<char*>(cache_base_ptr_)
                                                      + static_cast<ptrdiff_t>(config_.kv_scale_offset_bytes));
            if (!memory_util->deregUserMr(scale_base_ptr, true)) {
                RTP_LLM_FAIL("deregister user mr for block pool kv scale buffer failed");
            }
        }
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
    return layout_strategy_->convertIndexToAddr(layer_id, block_id);
}

void* BlockPool::getKCacheAddr(int layer_id, int block_id) const {
    return layout_strategy_->getKCacheAddr(layer_id, block_id);
}

void* BlockPool::getVCacheAddr(int layer_id, int block_id) const {
    return layout_strategy_->getVCacheAddr(layer_id, block_id);
}

BlockBufferPtrInfo BlockPool::convertIndexToBuffer(int layer_id, int block_id) const {
    return layout_strategy_->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BufferPtr>
BlockPool::convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const {
    return layout_strategy_->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

KVCacheBuffer BlockPool::kvCacheBuffer() const {
    return layout_strategy_->kvCacheBuffer();
}

MemoryType BlockPool::where() const {
    return cache_aligned_buffer_->where();
}

}  // namespace rtp_llm
