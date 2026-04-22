#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/core/ExecOps.h"
#include "rtp_llm/cpp/cache/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

BlockPool::BlockPool(const BlockPoolConfig& config, AllocationType allocation_type):
    config_(config), allocation_type_(allocation_type) {}

BlockPool::~BlockPool() {
    cache_aligned_buffer_ = torch::Tensor();
}

void BlockPool::validateConfig() const {
    RTP_LLM_CHECK_WITH_INFO(!config_.memory_layouts.empty(), "BlockPoolConfig.memory_layouts must not be empty");
    RTP_LLM_CHECK_WITH_INFO(config_.block_num > 0, "BlockPoolConfig.block_num must be > 0");

    for (size_t layout_idx = 0; layout_idx < config_.memory_layouts.size(); ++layout_idx) {
        const auto& layout_cfg = config_.memory_layouts[layout_idx];

        RTP_LLM_CHECK_WITH_INFO(layout_cfg.block_num == config_.block_num,
                                "MemoryLayoutConfig.block_num mismatch: layout[%zu].block_num=%u, pool.block_num=%u",
                                layout_idx,
                                layout_cfg.block_num,
                                config_.block_num);
        RTP_LLM_CHECK_WITH_INFO(
            layout_cfg.layer_num > 0, "MemoryLayoutConfig.layer_num must be > 0 (layout=%zu)", layout_idx);
        RTP_LLM_CHECK_WITH_INFO(layout_cfg.kv_block_pool_size_bytes > 0,
                                "MemoryLayoutConfig.kv_block_pool_size_bytes must be > 0 (layout=%zu)",
                                layout_idx);
    }
}

void BlockPool::initializeCacheBuffer() {
    if (allocation_type_ == AllocationType::HOST) {
        cache_aligned_buffer_ = torch::empty({static_cast<int64_t>(config_.total_size_bytes)},
                                             torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                                    .pin_memory();
    } else {
        cache_aligned_buffer_ = torch::empty({static_cast<int64_t>(config_.total_size_bytes)},
                                             torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    }
    cache_base_ptr_ = cache_aligned_buffer_.data_ptr();
    RTP_LLM_CHECK_WITH_INFO(cache_base_ptr_ != nullptr, "block pool allocate cache aligned buffer is null");
}

void BlockPool::initializeLayoutStrategies() {
    layout_strategies_.resize(config_.memory_layouts.size());
    torch::Tensor full_tensor = cache_aligned_buffer_;

    for (size_t layout_idx = 0; layout_idx < config_.memory_layouts.size(); ++layout_idx) {
        processMemoryLayout(layout_idx, full_tensor);
    }
}

void BlockPool::processMemoryLayout(size_t layout_idx, const torch::Tensor& full_tensor) {
    const auto& layout_cfg = config_.memory_layouts[layout_idx];

    // 创建 KV 缓存张量
    torch::Tensor kv_cache_tensor = createTensor(full_tensor,
                                                 static_cast<int64_t>(layout_cfg.kv_cache_offset_bytes),
                                                 static_cast<int64_t>(layout_cfg.kv_block_pool_size_bytes),
                                                 layout_idx,
                                                 "kv");
    // 创建缩放张量（如果需要）
    torch::Tensor kv_scale_tensor;
    if (layout_cfg.hasScale()) {
        kv_scale_tensor = createTensor(full_tensor,
                                       static_cast<int64_t>(layout_cfg.kv_scale_offset_bytes),
                                       static_cast<int64_t>(layout_cfg.kv_scale_pool_size_bytes),
                                       layout_idx,
                                       "kv_scale");
    }

    // 初始化内存布局策略
    initializeLayoutStrategy(layout_idx, layout_cfg, kv_cache_tensor, kv_scale_tensor);

    // 记录初始化信息
    RTP_LLM_LOG_INFO(
        "MemoryLayout[%zu] initialized: layer_num=%u block_num=%u kv_off=%zu kv_bytes=%zu scale_off=%zu scale_bytes=%zu",
        layout_idx,
        layout_cfg.layer_num,
        layout_cfg.block_num,
        layout_cfg.kv_cache_offset_bytes,
        layout_cfg.kv_block_pool_size_bytes,
        layout_cfg.kv_scale_offset_bytes,
        layout_cfg.kv_scale_pool_size_bytes);
}

torch::Tensor BlockPool::createTensor(
    const torch::Tensor& full_tensor, int64_t offset, int64_t size, size_t layout_idx, const std::string& tensor_type) {
    RTP_LLM_CHECK_WITH_INFO(offset >= 0 && size >= 0 && offset + size <= full_tensor.numel(),
                            "layout[%zu] %s tensor out of range: off=%ld bytes=%ld full=%ld",
                            layout_idx,
                            tensor_type.c_str(),
                            offset,
                            size,
                            full_tensor.numel());
    return full_tensor.narrow(0, offset, size);
}

void BlockPool::initializeLayoutStrategy(size_t                    layout_idx,
                                         const MemoryLayoutConfig& layout_cfg,
                                         torch::Tensor&            kv_cache_tensor,
                                         torch::Tensor&            kv_scale_tensor) {
    void* layout_cache_base_ptr =
        static_cast<void*>(static_cast<char*>(cache_base_ptr_) + layout_cfg.kv_cache_offset_bytes);

    layout_strategies_[layout_idx] = std::make_unique<MemoryLayoutStrategy>();
    RTP_LLM_CHECK_WITH_INFO(layout_strategies_[layout_idx] != nullptr,
                            "Failed to create memory layout strategy for layout[%zu]",
                            layout_idx);

    RTP_LLM_CHECK_WITH_INFO(
        layout_strategies_[layout_idx]->init(layout_cfg, kv_cache_tensor, kv_scale_tensor, layout_cache_base_ptr),
        "Failed to initialize memory layout strategy for layout[%zu]",
        layout_idx);
}

bool BlockPool::init() {
    validateConfig();
    initializeCacheBuffer();
    initializeLayoutStrategies();
    initFreeBlocks();

    const size_t total_layers = layout_strategies_.empty() ? 0 : layout_strategies_[0]->getLayerCacheTensors().size();
    RTP_LLM_LOG_INFO(
        "BlockPool init success: layer_num=%zu, total_size=%zu bytes", total_layers, config_.total_size_bytes);
    return true;
}

void BlockPool::initFreeBlocks() {
    // block 0 is reserved
    for (BlockIdxType i = 1; i < static_cast<BlockIdxType>(config_.block_num); ++i) {
        free_block_ids_.insert(i);
    }
    request_ref_counter_.init(config_.block_num);
    connector_ref_counter_.init(config_.block_num);
    req_con_ref_counter_.init(config_.block_num);
    block_cache_ref_counter_.init(config_.block_num);
    req_cache_ref_counter_.init(config_.block_num);
}

std::vector<torch::Tensor> BlockPool::allLayerCacheBase() const {
    if (layout_strategies_.empty() || !layout_strategies_[0]) {
        return {};
    }
    return layout_strategies_[0]->getLayerCacheTensors();
}

std::vector<torch::Tensor> BlockPool::allLayerScaleCacheBase() const {
    if (layout_strategies_.empty() || !layout_strategies_[0]) {
        return {};
    }
    return layout_strategies_[0]->getLayerScaleCacheTensors();
}

BlockIndicesType BlockPool::malloc(int num_blocks) {
    RTP_LLM_PROFILE_FUNCTION();
    if (num_blocks <= 0) {
        return {};
    }
    BlockIndicesType block_ids;
    block_ids.reserve(num_blocks);

    {
        std::scoped_lock lock(ref_mu_, free_mu_);
        if (free_block_ids_.size() < static_cast<size_t>(num_blocks)) {
            RTP_LLM_LOG_WARNING(
                "Block pool only has %zu free blocks, cannot allocate %d blocks", free_block_ids_.size(), num_blocks);
            return {};
        }
        auto first = free_block_ids_.begin();
        auto last  = std::next(first, num_blocks);
        block_ids.assign(first, last);
        free_block_ids_.erase(first, last);
        request_ref_counter_.incrementRefCounter(block_ids);
        req_con_ref_counter_.incrementRefCounter(block_ids);
        req_cache_ref_counter_.incrementRefCounter(block_ids);
    }

    return block_ids;
}

void BlockPool::requestFree(BlockIdxType block_idx) {
    auto block_ids = {block_idx};
    requestFree(block_ids);
}

void BlockPool::requestFree(const BlockIndicesType& block_ids) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    request_ref_counter_.decrementRefCounter(block_ids);
    req_con_ref_counter_.decrementRefCounter(block_ids);
    req_cache_ref_counter_.decrementRefCounter(block_ids);
    tryFreeBlocks(block_ids);
}

void BlockPool::connectorFree(BlockIdxType block_idx) {
    auto block_ids = {block_idx};
    connectorFree(block_ids);
}

void BlockPool::connectorFree(const BlockIndicesType& block_indices) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    connector_ref_counter_.decrementRefCounter(block_indices);
    req_con_ref_counter_.decrementRefCounter(block_indices);
    tryFreeBlocks(block_indices);
}

void BlockPool::blockCacheFree(BlockIdxType block_idx) {
    auto block_ids = {block_idx};
    blockCacheFree(block_ids);
}

void BlockPool::blockCacheFree(const BlockIndicesType& block_ids) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    block_cache_ref_counter_.decrementRefCounter(block_ids);
    req_cache_ref_counter_.decrementRefCounter(block_ids);
    tryFreeBlocks(block_ids);
}

// Must be called with ref_mu_ and free_mu_ held.
void BlockPool::tryFreeBlocks(const BlockIndicesType& block_ids) {
    RTP_LLM_PROFILE_FUNCTION();
    for (const auto& block_id : block_ids) {
        if (req_con_ref_counter_.getRefCounter(block_id) == 0
            && block_cache_ref_counter_.getRefCounter(block_id) == 0) {
            free_block_ids_.insert(block_id);
        }
    }
}

void BlockPool::requestReference(BlockIdxType block_idx) {
    BlockIndicesType block_ids = {block_idx};
    requestReference(block_ids);
}

void BlockPool::requestReference(const BlockIndicesType& block_ids) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    request_ref_counter_.incrementRefCounter(block_ids);
    req_con_ref_counter_.incrementRefCounter(block_ids);
    req_cache_ref_counter_.incrementRefCounter(block_ids);
    for (const auto& block_id : block_ids) {
        free_block_ids_.erase(block_id);
    }
}

void BlockPool::connectorReference(BlockIdxType block_idx) {
    BlockIndicesType block_ids = {block_idx};
    connectorReference(block_ids);
}

void BlockPool::connectorReference(const BlockIndicesType& block_indices) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    connector_ref_counter_.incrementRefCounter(block_indices);
    req_con_ref_counter_.incrementRefCounter(block_indices);
    for (const auto& block_id : block_indices) {
        free_block_ids_.erase(block_id);
    }
}

void BlockPool::blockCacheReference(BlockIdxType block_idx) {
    BlockIndicesType block_ids = {block_idx};
    blockCacheReference(block_ids);
}

void BlockPool::blockCacheReference(const BlockIndicesType& block_ids) {
    RTP_LLM_PROFILE_FUNCTION();
    std::scoped_lock lock(ref_mu_, free_mu_);
    block_cache_ref_counter_.incrementRefCounter(block_ids);
    req_cache_ref_counter_.incrementRefCounter(block_ids);
    for (const auto& block_id : block_ids) {
        free_block_ids_.erase(block_id);
    }
}

void BlockPool::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    if (cache_store) {
        cache_store_ = std::move(cache_store);
    }
    if (cache_store_ && !kvcache_reg_mr_) {
        RTP_LLM_LOG_INFO("start to register user mr");
        auto memory_util = std::static_pointer_cast<NormalCacheStore>(cache_store_)->getMemoryUtil();

        for (size_t layout_idx = 0; layout_idx < config_.memory_layouts.size(); ++layout_idx) {
            const auto& layout_cfg = config_.memory_layouts[layout_idx];

            // Register KV buffer
            registerUserMrForBuffer(memory_util,
                                    layout_idx,
                                    layout_cfg.kv_cache_offset_bytes,
                                    layout_cfg.kv_block_pool_size_bytes,
                                    layout_cfg.kv_block_stride_bytes,
                                    "kv");

            // Register scale buffer if present
            if (layout_cfg.hasScale()) {
                registerUserMrForBuffer(memory_util,
                                        layout_idx,
                                        layout_cfg.kv_scale_offset_bytes,
                                        layout_cfg.kv_scale_pool_size_bytes,
                                        layout_cfg.kv_scale_stride_bytes,
                                        "scale");
            }
        }

        kvcache_reg_mr_ = true;
    }
}

void BlockPool::deregUserMr() {
    if (kvcache_reg_mr_ && cache_store_) {
        RTP_LLM_LOG_INFO("start to deregister user mr");
        auto memory_util = std::static_pointer_cast<NormalCacheStore>(cache_store_)->getMemoryUtil();

        for (size_t layout_idx = 0; layout_idx < config_.memory_layouts.size(); ++layout_idx) {
            const auto& layout_cfg = config_.memory_layouts[layout_idx];

            // Deregister KV buffer
            deregisterUserMrForBuffer(memory_util, layout_idx, layout_cfg.kv_cache_offset_bytes, "kv");

            // Deregister scale buffer if present
            if (layout_cfg.hasScale()) {
                deregisterUserMrForBuffer(memory_util, layout_idx, layout_cfg.kv_scale_offset_bytes, "scale");
            }
        }

        RTP_LLM_LOG_INFO("deregister user mr for block pool success");
        kvcache_reg_mr_ = false;
    }
}

void BlockPool::registerUserMrForBuffer(std::shared_ptr<rtp_llm::MemoryUtil> memory_util,
                                        size_t                               layout_idx,
                                        size_t                               offset_bytes,
                                        size_t                               bytes,
                                        size_t                               stride_bytes,
                                        const std::string&                   buffer_type) {
    void* base_ptr = static_cast<void*>(static_cast<char*>(cache_base_ptr_) + static_cast<ptrdiff_t>(offset_bytes));
    auto  start_us = currentTimeUs();

    if (!memory_util->regUserMr(base_ptr, bytes, true, stride_bytes)) {
        RTP_LLM_FAIL("register user mr for block pool layout[%zu] %s buffer failed", layout_idx, buffer_type.c_str());
    }

    auto cost_ms = (currentTimeUs() - start_us) / 1000;
    mr_cost_time_ms_ += cost_ms;

    RTP_LLM_LOG_INFO("register user mr success: layout[%zu] %s base=%p len=%zu aligned=%zu cost=%ld ms",
                     layout_idx,
                     buffer_type.c_str(),
                     base_ptr,
                     bytes,
                     stride_bytes,
                     cost_ms);
}

void BlockPool::deregisterUserMrForBuffer(std::shared_ptr<rtp_llm::MemoryUtil> memory_util,
                                          size_t                               layout_idx,
                                          size_t                               offset_bytes,
                                          const std::string&                   buffer_type) {
    void* base_ptr = static_cast<void*>(static_cast<char*>(cache_base_ptr_) + static_cast<ptrdiff_t>(offset_bytes));

    if (!memory_util->deregUserMr(base_ptr, true)) {
        RTP_LLM_FAIL("deregister user mr for block pool layout[%zu] %s buffer failed", layout_idx, buffer_type.c_str());
    }
}

size_t BlockPool::freeBlocksNum() const {
    std::lock_guard<std::mutex> free_lock(free_mu_);
    return free_block_ids_.size();
}

size_t BlockPool::totalBlocksNum() const {
    // reserve block 0 for internal use
    return config_.block_num - 1;
}

// Available blocks need to satisfy two conditions:
// 1. not referenced by a request
// 2. not referenced by connector(read or write)
size_t BlockPool::availableBlocksNum() const {
    std::lock_guard<std::mutex> lock(ref_mu_);
    return req_con_ref_counter_.freeBlockNum();
}

size_t BlockPool::requestRefBlocksNum() const {
    std::lock_guard<std::mutex> lock(ref_mu_);
    return request_ref_counter_.busyBlockNum();
}

size_t BlockPool::connectorRefBlocksNum() const {
    std::lock_guard<std::mutex> lock(ref_mu_);
    return connector_ref_counter_.busyBlockNum();
}

size_t BlockPool::blockCacheRefBlocksNum() const {
    std::lock_guard<std::mutex> lock(ref_mu_);
    return block_cache_ref_counter_.busyBlockNum();
}

size_t BlockPool::notInUseBlocksNum() const {
    std::lock_guard<std::mutex> lock(ref_mu_);
    return req_cache_ref_counter_.freeBlockNum();
}

BlockAddrInfo BlockPool::convertIndexToAddr(int layer_id, int block_id) const {
    RTP_LLM_CHECK_WITH_INFO(!layout_strategies_.empty() && layout_strategies_[0],
                            "BlockPool not initialized (no layout strategy)");
    return layout_strategies_[0]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> BlockPool::convertIndexToBuffer(int layer_id, int block_id) const {
    RTP_LLM_CHECK_WITH_INFO(!layout_strategies_.empty() && layout_strategies_[0],
                            "BlockPool not initialized (no layout strategy)");
    return layout_strategies_[0]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo>
BlockPool::convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const {
    RTP_LLM_CHECK_WITH_INFO(!layout_strategies_.empty() && layout_strategies_[0],
                            "BlockPool not initialized (no layout strategy)");
    return layout_strategies_[0]->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

MemoryType BlockPool::where() const {
    return cache_aligned_buffer_.is_cuda() ? MemoryType::MEMORY_GPU : MemoryType::MEMORY_CPU;
}

}  // namespace rtp_llm
