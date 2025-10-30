#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
BlockPool::BlockPool(const BlockPoolConfig& config, rtp_llm::DeviceBase* device, AllocationType atype = AllocationType::DEVICE):
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
    for(int i = 0; i < config_.block_num; ++i) {
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

std::vector<int> BlockPool::alloc(int num_blocks) {
    std::vector<int> block_ids;
    block_ids.reserve(num_blocks);
    if (free_block_ids.size() < num_blocks) {
        RTP_LLM_LOG_DEBUG("Block pool only has %d free blocks, cannot allocate %d blocks", free_block_ids.size(), num_blocks);
        return {};
    }
    for(int i = 0; i < num_blocks; ++i) {
        block_ids.push_back(free_block_ids.top());
        free_block_ids.pop();
    }
    return block_ids;
}


void BlockPool::free(const std::vector<int>& block_ids) {
    block_ref_counter_.decrementRefCounter(block_ids);
    for(auto& block_id : block_ids) {
        if (block_ref_counter_.getRefCounter(block_id) == 0) {
            block_ref_counter_.incrementRefCounter({block_id});
        }
    }
}

void BlockPool::reference(const std::vector<int>& block_ids) {
    block_ref_counter_.incrementRefCounter(block_ids);
}

void BlockPool::regUserMr(size_t model_id) {
    if (device_->cacheStore() && !kvcache_reg_mr_) {
        RTP_LLM_LOG_INFO("start to register user mr");
        auto memory_util = std::static_pointer_cast<NormalCacheStore>(device_->cacheStore())->getMemoryUtil();
        
        auto start_time_us = currentTimeUs();
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
        
        // // register user buffer by block_index for decode entrance scenario
        // if(config_.layout == LAYER_FIRST) {
        //     std::vector<std::shared_ptr<BlockBuffer>> buffers;
        //     for (int block_index = 0; block_index < config_.block_num; ++block_index) {
        //         for (int layer_index = 0; layer_index < config_.layer_num; ++layer_index) {
        //             auto block_key = makeCacheKey(model_id, std::to_string(layer_index), block_index);
        //             auto addr_info = convertIndexToAddr(layer_index, block_index);
        //             auto kv_buffer = std::make_shared<BlockBuffer>(
        //                 "kv_" + block_key, 
        //                 std::shared_ptr<void>(addr_info.k_addr, [](void*) {}), 
        //                 config_.block_size, 
        //                 true, 
        //                 true);
        //             buffers.push_back(kv_buffer);
        //         }
        //     }
        //     device_->cacheStore()->regUserBuffers(buffers);
        //  } else if (config_.layout == KV_FIRST) {
        //      // TODO: implement kv first layout
        //  }
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
        return {nullptr, nullptr, nullptr, nullptr};
    }
    return layout_strategy_->convertIndexToBuffer(layer_id, block_id);
}

}  // namespace rtp_llm
