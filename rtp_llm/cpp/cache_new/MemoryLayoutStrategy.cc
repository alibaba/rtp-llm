#include "rtp_llm/cpp/cache_new/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {


// LayerFirstLayoutStrategy 

bool LayerFirstLayoutStrategy::init(const BlockPoolConfig& config, 
                                   torch::Tensor& cache_buffer,
                                   void* cache_base_ptr) {
    config_ = config;
    cache_base_ptr_ = cache_base_ptr;
    
    if (cache_buffer.numel() == 0) {
        RTP_LLM_LOG_ERROR("Cache buffer tensor is empty, cannot split by layers");
        return false;
    }
    
    torch::Tensor reshaped_tensor = cache_buffer.reshape({
        static_cast<int64_t>(config_.layer_num),
        static_cast<int64_t>(config_.block_num),
        static_cast<int64_t>(config_.block_size)
    });
    
    layer_kv_tensors_.clear();
    layer_kv_tensors_.reserve(config_.layer_num);
    
    for (uint32_t layer_id = 0; layer_id < config_.layer_num; ++layer_id) {
        torch::Tensor layer_tensor = reshaped_tensor[layer_id];
        layer_kv_tensors_.push_back(layer_tensor);
        
        RTP_LLM_LOG_DEBUG("Layer %d tensor shape: [%s], elements: %ld", 
                         layer_id, 
                         torch::str(layer_tensor.sizes()).c_str(),
                         layer_tensor.numel());
    }
    
    RTP_LLM_LOG_INFO("LayerFirstLayoutStrategy initialized successfully");
    return true;
}

std::vector<torch::Tensor> LayerFirstLayoutStrategy::getLayerCacheTensors() const {
    return layer_kv_tensors_;
}

BlockAddrInfo LayerFirstLayoutStrategy::convertIndexToAddr(int layer_id, int block_id) const {
    if (layer_id >= layer_kv_tensors_.size()) {
        RTP_LLM_LOG_ERROR("Layer ID %d out of range (max: %zu)", layer_id, layer_kv_tensors_.size());
        return {nullptr, nullptr, nullptr, nullptr};
    }
    
    torch::Tensor tensor = layer_kv_tensors_[layer_id][block_id];
    return {tensor.data_ptr(), tensor.data_ptr(), nullptr, nullptr};
}

BlockBufferInfo LayerFirstLayoutStrategy::convertIndexToBuffer(int layer_id, int block_id) const {
    if (layer_id >= layer_kv_tensors_.size()) {
        RTP_LLM_LOG_ERROR("Layer ID %d out of range (max: %zu)", layer_id, layer_kv_tensors_.size());
        return {nullptr, nullptr, nullptr, nullptr};
    }
    
    torch::Tensor tensor = layer_kv_tensors_[layer_id][block_id];
    BufferPtr buffer = torchTensor2Buffer(tensor);
    return {buffer, buffer, nullptr, nullptr};
}

void* LayerFirstLayoutStrategy::getKCacheAddr(int layer_id, int block_id) const {
    auto addr_info = convertIndexToAddr(layer_id, block_id);
    return addr_info.k_addr ? addr_info.k_addr->data() : nullptr;
}

void* LayerFirstLayoutStrategy::getVCacheAddr(int layer_id, int block_id) const {
    auto addr_info = convertIndexToAddr(layer_id, block_id);
    return addr_info.v_addr ? addr_info.v_addr->data() : nullptr;
}


// KVFirstLayoutStrategy 

bool KVFirstLayoutStrategy::init(const BlockPoolConfig& config, 
                                 torch::Tensor& cache_buffer,
                                 void* cache_base_ptr) {
    config_ = config;
    cache_base_ptr_ = cache_base_ptr;
    
    if (cache_buffer.numel() == 0) {
        RTP_LLM_LOG_ERROR("Cache buffer tensor is empty, cannot split by KV layout");
        return false;
    }
    
    torch::Tensor reshaped_tensor = cache_buffer.reshape({
        2,
        static_cast<int64_t>(config_.layer_num),
        static_cast<int64_t>(config_.block_num),
        static_cast<int64_t>(config_.k_block_size)  // assume k and v block size are the same
    });
    

    k_cache_tensor_ = reshaped_tensor[0];  
    v_cache_tensor_ = reshaped_tensor[1];  // [layer_num, block_num, kv_block_size]
    
    layer_kv_tensors_.clear();
    layer_kv_tensors_.reserve(config_.layer_num);
    
    for (uint32_t layer_id = 0; layer_id < config_.layer_num; ++layer_id) {
        torch::Tensor k_layer_tensor = k_cache_tensor_[layer_id];
        layer_kv_tensors_.push_back(k_layer_tensor);  // return k cache tensor
        
        RTP_LLM_LOG_DEBUG("Layer %d K tensor shape: [%s], elements: %ld", 
                         layer_id, 
                         torch::str(k_layer_tensor.sizes()).c_str(),
                         k_layer_tensor.numel());
    }
    
    RTP_LLM_LOG_INFO("KVFirstLayoutStrategy initialized successfully");
    return true;
}

std::vector<torch::Tensor> KVFirstLayoutStrategy::getLayerCacheTensors() const {
    return layer_kv_tensors_;
}

BlockAddrInfo KVFirstLayoutStrategy::convertIndexToAddr(int layer_id, int block_id) const {
    // return k address and v address 
    if (layer_id >= config_.layer_num) {
        RTP_LLM_LOG_ERROR("Layer ID %d out of range (max: %d)", layer_id, config_.layer_num);
        return {nullptr, nullptr, nullptr, nullptr};
    }
    
    return {k_cache_tensor_[layer_id][block_id].data_ptr(), v_cache_tensor_[layer_id][block_id].data_ptr(), nullptr, nullptr};
}

BlockBufferInfo KVFirstLayoutStrategy::convertIndexToBuffer(int layer_id, int block_id) const {
    if (layer_id >= config_.layer_num) {
        RTP_LLM_LOG_ERROR("Layer ID %d out of range (max: %d)", layer_id, config_.layer_num);
        return {nullptr, nullptr, nullptr, nullptr};
    }
    
    torch::Tensor k_tensor = k_cache_tensor_[layer_id][block_id];
    BufferPtr k_buffer = torchTensor2Buffer(k_tensor);
    torch::Tensor v_tensor = v_cache_tensor_[layer_id][block_id];
    BufferPtr v_buffer = torchTensor2Buffer(v_tensor);
    return {k_buffer, v_buffer, nullptr, nullptr};
}

void* KVFirstLayoutStrategy::getKCacheAddr(int layer_id, int block_id) const {
    if (layer_id >= config_.layer_num) {
        RTP_LLM_LOG_ERROR("Layer ID %d out of range (max: %d)", layer_id, config_.layer_num);
        return nullptr;
    }
    
    torch::Tensor k_tensor = k_cache_tensor_[layer_id][block_id];
    return k_tensor.data_ptr();
}

void* KVFirstLayoutStrategy::getVCacheAddr(int layer_id, int block_id) const {
    if (layer_id >= config_.layer_num) {
        RTP_LLM_LOG_ERROR("Layer ID %d out of range (max: %d)", layer_id, config_.layer_num);
        return nullptr;
    }
    
    torch::Tensor v_tensor = v_cache_tensor_[layer_id][block_id];
    return v_tensor.data_ptr();
}

std::unique_ptr<MemoryLayoutStrategy> MemoryLayoutStrategyFactory::create(MemoryLayout layout) {
    switch (layout) {
        case LAYER_FIRST:
            return std::make_unique<LayerFirstLayoutStrategy>();
        case KV_FIRST:
            return std::make_unique<KVFirstLayoutStrategy>();
        default:
            RTP_LLM_LOG_ERROR("Unknown memory layout type: %d", static_cast<int>(layout));
            return nullptr;
    }
}

}  // namespace rtp_llm
