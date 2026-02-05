#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

BlockPool::BlockPool(const BlockPoolConfig& config, rtp_llm::DeviceBase* device, AllocationType allocation_type):
    config_(config), device_(device), allocation_type_(allocation_type) {}

BlockPool::~BlockPool() {
    cache_aligned_buffer_.reset();
}

bool BlockPool::init() {
    RTP_LLM_CHECK_WITH_INFO(!config_.memory_layouts.empty(), "BlockPoolConfig.memory_layouts must not be empty");
    RTP_LLM_CHECK_WITH_INFO(config_.block_num > 0, "BlockPoolConfig.block_num must be > 0");

    cache_aligned_buffer_ = device_->allocateBuffer({rtp_llm::TYPE_INT8, {config_.total_size_bytes}, allocation_type_});
    cache_base_ptr_       = cache_aligned_buffer_->data();
    RTP_LLM_CHECK_WITH_INFO(cache_aligned_buffer_ != nullptr && cache_base_ptr_ != nullptr,
                            "block pool allocate cache aligned buffer is null");

    torch::Tensor full_tensor = Buffer2torchTensor(cache_aligned_buffer_, false);

    size_t total_layers = 0;
    for (const auto& layout_cfg : config_.memory_layouts) {
        total_layers += static_cast<size_t>(layout_cfg.layer_num);
    }
    global_layer_to_local_.assign(total_layers, {-1, -1});
    global_layer_kv_tensors_.assign(total_layers, torch::Tensor());
    global_layer_kv_scale_tensors_.assign(total_layers, torch::Tensor());

    layout_strategies_.resize(config_.memory_layouts.size());

    bool   has_any_scale      = false;
    size_t global_layer_begin = 0;

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

        const int64_t kv_off   = static_cast<int64_t>(layout_cfg.kv_cache_offset_bytes);
        const int64_t kv_bytes = static_cast<int64_t>(layout_cfg.kv_block_pool_size_bytes);
        RTP_LLM_CHECK_WITH_INFO(kv_off >= 0 && kv_bytes >= 0 && kv_off + kv_bytes <= full_tensor.numel(),
                                "layout[%zu] kv tensor out of range: off=%ld bytes=%ld full=%ld",
                                layout_idx,
                                kv_off,
                                kv_bytes,
                                full_tensor.numel());

        torch::Tensor kv_cache_tensor = full_tensor.narrow(0, kv_off, kv_bytes);

        torch::Tensor kv_scale_tensor;
        if (layout_cfg.hasScale()) {
            const int64_t scale_off   = static_cast<int64_t>(layout_cfg.kv_scale_offset_bytes);
            const int64_t scale_bytes = static_cast<int64_t>(layout_cfg.kv_scale_pool_size_bytes);
            RTP_LLM_CHECK_WITH_INFO(scale_off >= 0 && scale_bytes >= 0
                                        && scale_off + scale_bytes <= full_tensor.numel(),
                                    "layout[%zu] kv_scale tensor out of range: off=%ld bytes=%ld full=%ld",
                                    layout_idx,
                                    scale_off,
                                    scale_bytes,
                                    full_tensor.numel());
            kv_scale_tensor = full_tensor.narrow(0, scale_off, scale_bytes);
        }

        void* layout_cache_base_ptr =
            static_cast<void*>(static_cast<char*>(cache_base_ptr_) + layout_cfg.kv_cache_offset_bytes);

        layout_strategies_[layout_idx] = MemoryLayoutStrategyFactory::create(layout_cfg.layout);
        RTP_LLM_CHECK_WITH_INFO(layout_strategies_[layout_idx] != nullptr,
                                "Failed to create memory layout strategy for layout[%zu]",
                                layout_idx);

        const auto dtype = layout_cfg.dtype;
        RTP_LLM_CHECK_WITH_INFO(layout_strategies_[layout_idx]->init(
                                    layout_cfg, kv_cache_tensor, kv_scale_tensor, layout_cache_base_ptr, dtype),
                                "Failed to initialize memory layout strategy for layout[%zu]",
                                layout_idx);

        auto layer_tensors = layout_strategies_[layout_idx]->getLayerCacheTensors();
        RTP_LLM_CHECK_WITH_INFO(layer_tensors.size() == static_cast<size_t>(layout_cfg.layer_num),
                                "layout[%zu] layer tensors size mismatch: got=%zu expect=%u",
                                layout_idx,
                                layer_tensors.size(),
                                layout_cfg.layer_num);

        for (size_t local_layer = 0; local_layer < static_cast<size_t>(layout_cfg.layer_num); ++local_layer) {
            const size_t global_layer = global_layer_begin + local_layer;
            RTP_LLM_CHECK_WITH_INFO(global_layer < global_layer_to_local_.size(), "global layer index out of range");
            global_layer_to_local_[global_layer]   = {static_cast<int>(layout_idx), static_cast<int>(local_layer)};
            global_layer_kv_tensors_[global_layer] = layer_tensors[local_layer];
        }

        auto scale_tensors = layout_strategies_[layout_idx]->getLayerScaleCacheTensors();
        if (!scale_tensors.empty()) {
            RTP_LLM_CHECK_WITH_INFO(scale_tensors.size() == static_cast<size_t>(layout_cfg.layer_num),
                                    "layout[%zu] scale tensors size mismatch: got=%zu expect=%u",
                                    layout_idx,
                                    scale_tensors.size(),
                                    layout_cfg.layer_num);
            has_any_scale = true;
            for (size_t local_layer = 0; local_layer < static_cast<size_t>(layout_cfg.layer_num); ++local_layer) {
                const size_t global_layer                    = global_layer_begin + local_layer;
                global_layer_kv_scale_tensors_[global_layer] = scale_tensors[local_layer];
            }
        }

        RTP_LLM_LOG_INFO(
            "MemoryLayout[%zu] initialized: layer_num=%u block_num=%u kv_off=%zu kv_bytes=%zu scale_off=%zu scale_bytes=%zu",
            layout_idx,
            layout_cfg.layer_num,
            layout_cfg.block_num,
            layout_cfg.kv_cache_offset_bytes,
            layout_cfg.kv_block_pool_size_bytes,
            layout_cfg.kv_scale_offset_bytes,
            layout_cfg.kv_scale_pool_size_bytes);

        global_layer_begin += static_cast<size_t>(layout_cfg.layer_num);
    }

    if (!has_any_scale) {
        global_layer_kv_scale_tensors_.clear();
    }

    initFreeBlocks();
    block_cache_ = std::make_shared<BlockCache>();

    RTP_LLM_LOG_INFO("BlockPool init success: memory_layouts=%zu, total_layers=%zu, total_size=%zu bytes",
                     config_.memory_layouts.size(),
                     total_layers,
                     config_.total_size_bytes);
    return true;
}

BlockCachePtr BlockPool::blockCache() {
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

std::vector<torch::Tensor> BlockPool::allLayerCacheBase() const {
    return global_layer_kv_tensors_;
}

std::vector<torch::Tensor> BlockPool::allLayerScaleCacheBase() const {
    return global_layer_kv_scale_tensors_;
}

BlockIndicesType BlockPool::malloc(int num_blocks) {
    if (num_blocks <= 0) {
        return {};
    }
    BlockIndicesType block_ids;
    block_ids.reserve(num_blocks);

    {
        std::lock_guard<std::mutex> free_lock(free_mu_);
        if (free_block_ids_.size() < static_cast<size_t>(num_blocks)) {
            RTP_LLM_LOG_WARNING(
                "Block pool only has %zu free blocks, cannot allocate %d blocks", free_block_ids_.size(), num_blocks);
            return {};
        }
        auto first = free_block_ids_.begin();
        auto last  = std::next(first, num_blocks);
        block_ids.assign(first, last);
        free_block_ids_.erase(first, last);
    }
    requestReference(block_ids);

    return block_ids;
}

void BlockPool::requestFree(BlockIdxType block_idx) {
    auto block_ids = {block_idx};
    requestFree(block_ids);
}

void BlockPool::requestFree(const BlockIndicesType& block_ids) {
    freeImpl(block_ids);
    std::lock_guard<std::mutex> ref_lock(ref_mu_);
    request_ref_counter_.decrementRefCounter(block_ids);
}

void BlockPool::blockCacheFree(BlockIdxType block_idx) {
    auto block_ids = {block_idx};
    blockCacheFree(block_ids);
}

void BlockPool::blockCacheFree(const BlockIndicesType& block_ids) {
    freeImpl(block_ids);
}

void BlockPool::freeImpl(const BlockIndicesType& block_ids) {
    std::scoped_lock lock(ref_mu_, free_mu_);
    auto             new_free_block_ids = all_ref_counter_.decrementRefCounterWithFreeInfo(block_ids);
    free_block_ids_.insert(new_free_block_ids.begin(), new_free_block_ids.end());
}

void BlockPool::requestReference(BlockIdxType block_idx) {
    BlockIndicesType block_ids = {block_idx};
    requestReference(block_ids);
}

void BlockPool::requestReference(const BlockIndicesType& block_ids) {
    std::lock_guard<std::mutex> ref_lock(ref_mu_);
    request_ref_counter_.incrementRefCounter(block_ids);
    all_ref_counter_.incrementRefCounter(block_ids);
}

void BlockPool::blockCacheReference(BlockIdxType block_idx) {
    BlockIndicesType block_ids = {block_idx};
    blockCacheReference(block_ids);
}

void BlockPool::blockCacheReference(const BlockIndicesType& block_ids) {
    std::lock_guard<std::mutex> ref_lock(ref_mu_);
    all_ref_counter_.incrementRefCounter(block_ids);
}

void BlockPool::regUserMr(size_t model_id) {
    if (device_->cacheStore() && !kvcache_reg_mr_) {
        RTP_LLM_LOG_INFO("start to register user mr");
        auto memory_util = std::static_pointer_cast<NormalCacheStore>(device_->cacheStore())->getMemoryUtil();

        for (size_t layout_idx = 0; layout_idx < config_.memory_layouts.size(); ++layout_idx) {
            const auto& layout_cfg = config_.memory_layouts[layout_idx];

            void*  kv_base_ptr = static_cast<void*>(static_cast<char*>(cache_base_ptr_)
                                                   + static_cast<ptrdiff_t>(layout_cfg.kv_cache_offset_bytes));
            auto   start_kv_us = currentTimeUs();
            size_t kv_bytes    = layout_cfg.kv_block_pool_size_bytes;

            if (!memory_util->regUserMr(kv_base_ptr, kv_bytes, true, layout_cfg.kv_block_stride_bytes)) {
                RTP_LLM_FAIL("register user mr for block pool layout[%zu] kv buffer failed", layout_idx);
            }
            auto kv_cost_ms = (currentTimeUs() - start_kv_us) / 1000;
            mr_cost_time_ms_ += kv_cost_ms;
            RTP_LLM_LOG_INFO("register user mr success: layout[%zu] kv base=%p len=%zu aligned=%zu cost=%ld ms",
                             layout_idx,
                             kv_base_ptr,
                             kv_bytes,
                             layout_cfg.kv_block_stride_bytes,
                             kv_cost_ms);

            if (layout_cfg.hasScale()) {
                void*  scale_base_ptr = static_cast<void*>(static_cast<char*>(cache_base_ptr_)
                                                          + static_cast<ptrdiff_t>(layout_cfg.kv_scale_offset_bytes));
                auto   start_scale_us = currentTimeUs();
                size_t scale_bytes    = layout_cfg.kv_scale_pool_size_bytes;

                if (!memory_util->regUserMr(scale_base_ptr, scale_bytes, true, layout_cfg.kv_scale_stride_bytes)) {
                    RTP_LLM_FAIL("register user mr for block pool layout[%zu] kv scale buffer failed", layout_idx);
                }
                auto scale_cost_ms = (currentTimeUs() - start_scale_us) / 1000;
                mr_cost_time_ms_ += scale_cost_ms;
                RTP_LLM_LOG_INFO("register user mr success: layout[%zu] scale base=%p len=%zu aligned=%zu cost=%ld ms",
                                 layout_idx,
                                 scale_base_ptr,
                                 scale_bytes,
                                 layout_cfg.k_scale_stride_bytes,
                                 scale_cost_ms);
            }
        }

        kvcache_reg_mr_ = true;
    }
}

void BlockPool::deregUserMr() {
    if (kvcache_reg_mr_) {
        RTP_LLM_LOG_INFO("start to deregister user mr");
        auto memory_util = std::static_pointer_cast<NormalCacheStore>(device_->cacheStore())->getMemoryUtil();

        for (size_t layout_idx = 0; layout_idx < config_.memory_layouts.size(); ++layout_idx) {
            const auto& layout_cfg  = config_.memory_layouts[layout_idx];
            void*       kv_base_ptr = static_cast<void*>(static_cast<char*>(cache_base_ptr_)
                                                   + static_cast<ptrdiff_t>(layout_cfg.kv_cache_offset_bytes));
            if (!memory_util->deregUserMr(kv_base_ptr, true)) {
                RTP_LLM_FAIL("deregister user mr for block pool layout[%zu] kv buffer failed", layout_idx);
            }
        }

        for (size_t layout_idx = 0; layout_idx < config_.memory_layouts.size(); ++layout_idx) {
            const auto& layout_cfg = config_.memory_layouts[layout_idx];
            if (layout_cfg.hasScale()) {
                void* scale_base_ptr = static_cast<void*>(static_cast<char*>(cache_base_ptr_)
                                                          + static_cast<ptrdiff_t>(layout_cfg.kv_scale_offset_bytes));
                if (!memory_util->deregUserMr(scale_base_ptr, true)) {
                    RTP_LLM_FAIL("deregister user mr for block pool layout[%zu] kv scale buffer failed", layout_idx);
                }
            }
        }

        RTP_LLM_LOG_INFO("deregister user mr for block pool success");
        kvcache_reg_mr_ = false;
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

// Blocks not referenced by a request are free.
size_t BlockPool::availableBlocksNum() const {
    std::lock_guard<std::mutex> ref_lock(ref_mu_);
    return request_ref_counter_.freeBlockNum();
}

void BlockPool::debugString() const {
    std::scoped_lock lock(ref_mu_, free_mu_);
    size_t           total_blocks        = totalBlocksNum();
    size_t           free_blocks         = free_block_ids_.size();
    size_t           available_blocks    = request_ref_counter_.freeBlockNum();
    size_t           busy_blocks         = all_ref_counter_.busyBlockNum();
    size_t           request_busy_blocks = request_ref_counter_.busyBlockNum();

    RTP_LLM_LOG_INFO("BlockPool state: total_blocks = %zu, free_blocks = %zu, available_blocks = %zu, "
                     "busy_blocks = %zu, request_busy_blocks = %zu, used_blocks = %zu",
                     total_blocks,
                     free_blocks,
                     available_blocks,
                     busy_blocks,
                     request_busy_blocks,
                     total_blocks - available_blocks);

    // Print reference count for each block
    // Block 0 is reserved, so we start from block 1
    size_t       printed_count   = 0;
    const size_t max_print_count = 100;  // Limit output to avoid log flooding

    for (BlockIdxType block_id = 1; block_id < static_cast<BlockIdxType>(config_.block_num); ++block_id) {
        int  all_ref_count     = all_ref_counter_.getRefCounter(block_id);
        int  request_ref_count = request_ref_counter_.getRefCounter(block_id);
        bool is_free           = free_block_ids_.find(block_id) != free_block_ids_.end();

        // Print all blocks or only non-zero reference count blocks (to reduce log size)
        // Uncomment the condition below to print only blocks with non-zero reference count
        // if (all_ref_count > 0 || request_ref_count > 0) {
        if (printed_count < max_print_count) {
            RTP_LLM_LOG_INFO("BlockPool block[%d]: all_ref_count = %d, request_ref_count = %d, is_free = %s",
                             block_id,
                             all_ref_count,
                             request_ref_count,
                             is_free ? "true" : "false");
            printed_count++;
        } else if (printed_count == max_print_count) {
            RTP_LLM_LOG_INFO(
                "BlockPool: ... (showing first %zu blocks, total %zu blocks)", max_print_count, total_blocks);
            printed_count++;
            break;
        }
        // }
    }
}

// MTP support: Map global_layer_id to (model_index, local_layer_id).
// Returns {layout_index, local_layer_id}. layout_index is the index in BlockPoolConfig.memory_layouts.
std::pair<int, int> BlockPool::mapGlobalLayerIdToLocal(int global_layer_id) const {
    if (global_layer_id < 0 || static_cast<size_t>(global_layer_id) >= global_layer_to_local_.size()) {
        RTP_LLM_LOG_ERROR(
            "Global layer_id %d out of range (total layers: %zu)", global_layer_id, global_layer_to_local_.size());
        return {-1, -1};
    }

    return global_layer_to_local_[static_cast<size_t>(global_layer_id)];
}

BlockAddrInfo BlockPool::convertIndexToAddr(int layer_id, int block_id) const {
    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);
    return layout_strategies_[static_cast<size_t>(layout_index)]->convertIndexToAddr(local_layer_id, block_id);
}

void* BlockPool::getKCacheAddr(int layer_id, int block_id) const {
    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);
    return layout_strategies_[static_cast<size_t>(layout_index)]->getKCacheAddr(local_layer_id, block_id);
}

void* BlockPool::getVCacheAddr(int layer_id, int block_id) const {
    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);
    return layout_strategies_[static_cast<size_t>(layout_index)]->getVCacheAddr(local_layer_id, block_id);
}

std::vector<BlockInfo> BlockPool::convertIndexToBuffer(int layer_id, int block_id) const {
    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);
    return layout_strategies_[static_cast<size_t>(layout_index)]->convertIndexToBuffer(local_layer_id, block_id);
}

std::vector<BlockInfo>
BlockPool::convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const {
    auto [layout_index, local_layer_id] = mapGlobalLayerIdToLocal(layer_id);
    checkLayoutValidity(layout_index);

    return layout_strategies_[static_cast<size_t>(layout_index)]->convertIndexToBuffer(
        local_layer_id, block_id, partition_count, partition_id);
}

KVCacheBuffer BlockPool::kvCacheBuffer() const {
    return getMemoryLayoutKVCacheBuffer(0);
}

MemoryType BlockPool::where() const {
    return cache_aligned_buffer_->where();
}

KVCacheBuffer BlockPool::getMemoryLayoutKVCacheBuffer(int layout_id) const {
    if (layout_id < 0 || static_cast<size_t>(layout_id) >= layout_strategies_.size()) {
        RTP_LLM_LOG_ERROR("Memory layout ID %d out of range (max: %zu)", layout_id, layout_strategies_.size());
        return KVCacheBuffer{};
    }
    return layout_strategies_[static_cast<size_t>(layout_id)]->kvCacheBuffer();
}

void BlockPool::checkLayoutValidity(int layout_id) const {
    RTP_LLM_CHECK_WITH_INFO(layout_id >= 0 && static_cast<size_t>(layout_id) < layout_strategies_.size(),
                            "Memory layout ID %d out of range (max: %zu)",
                            layout_id,
                            layout_strategies_.size());
}

}  // namespace rtp_llm
