#include "rtp_llm/cpp/cache_new/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {


// LayerFirstLayoutStrategy 
bool LayerFirstLayoutStrategy::init(const BlockPoolConfig& config, 
                                   torch::Tensor& cache_buffer,
                                   void* cache_base_ptr,
                                   rtp_llm::DataType data_type) {
    config_ = config;
    cache_base_ptr_ = cache_base_ptr;
    data_type_ = data_type;
    
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
        return {nullptr, nullptr};
    }
    
    torch::Tensor tensor = layer_kv_tensors_[layer_id][block_id];
    BufferPtr buffer = torchTensor2Buffer(tensor);
    return {buffer, buffer};
}

void* LayerFirstLayoutStrategy::getKCacheAddr(int layer_id, int block_id) const {
    auto addr_info = convertIndexToAddr(layer_id, block_id);
    return addr_info.k_addr;
}

void* LayerFirstLayoutStrategy::getVCacheAddr(int layer_id, int block_id) const {
    auto addr_info = convertIndexToAddr(layer_id, block_id);
    return addr_info.v_addr;
}

KVCacheBuffer LayerFirstLayoutStrategy::kvCacheBuffer() const {
    // LayerFirst layout does not support K/V separation
    // Return empty buffer with all fields as nullptr
    KVCacheBuffer buffer;
    buffer.k_blocks = nullptr;
    buffer.v_blocks = nullptr;
    buffer.k_scale = nullptr;
    buffer.v_scale = nullptr;
    RTP_LLM_LOG_DEBUG("LayerFirstLayoutStrategy does not support kvCacheBuffer, returning empty buffer");
    return buffer;
}


// KVFirstLayoutStrategy 
bool KVFirstLayoutStrategy::init(const BlockPoolConfig& config, 
                                 torch::Tensor& cache_buffer,
                                 void* cache_base_ptr,
                                 rtp_llm::DataType data_type) {
    config_ = config;
    cache_base_ptr_ = cache_base_ptr;
    data_type_ = data_type;
    
    if (cache_buffer.numel() == 0) {
        RTP_LLM_LOG_ERROR("Cache buffer tensor is empty, cannot split by KV layout");
        return false;
    }
    
    size_t k_total_size = config_.layer_num * config_.block_num * config_.k_block_size;
    size_t v_total_size = config_.layer_num * config_.block_num * config_.v_block_size;
    size_t expected_total = k_total_size + v_total_size;
    
    if (cache_buffer.numel() != static_cast<int64_t>(expected_total)) {
        RTP_LLM_LOG_ERROR("Cache buffer size mismatch: expected %zu, got %ld", 
                         expected_total, cache_buffer.numel());
        return false;
    }

    // Layout: [K cache: layer_num * block_num * k_block_size][V cache: layer_num * block_num * v_block_size]
    torch::Tensor k_buffer = cache_buffer.narrow(0, 0, k_total_size);
    torch::Tensor v_buffer = cache_buffer.narrow(0, k_total_size, v_total_size);
    
    k_cache_tensor_ = k_buffer.reshape({
        static_cast<int64_t>(config_.layer_num),
        static_cast<int64_t>(config_.block_num),
        static_cast<int64_t>(config_.k_block_size)
    });
    
    v_cache_tensor_ = v_buffer.reshape({
        static_cast<int64_t>(config_.layer_num),
        static_cast<int64_t>(config_.block_num),
        static_cast<int64_t>(config_.v_block_size)
    });
    
    layer_kv_tensors_.clear();
    layer_kv_tensors_.reserve(config_.layer_num);
    
    for (uint32_t layer_id = 0; layer_id < config_.layer_num; ++layer_id) {
        torch::Tensor k_layer_tensor = k_cache_tensor_[layer_id];
        layer_kv_tensors_.push_back(k_layer_tensor);
        
        RTP_LLM_LOG_DEBUG("Layer %d K tensor shape: [%s], elements: %ld, V tensor shape: [%s], elements: %ld", 
                         layer_id, 
                         torch::str(k_layer_tensor.sizes()).c_str(),
                         k_layer_tensor.numel(),
                         torch::str(v_cache_tensor_[layer_id].sizes()).c_str(),
                         v_cache_tensor_[layer_id].numel());
    }
    
    RTP_LLM_LOG_INFO("KVFirstLayoutStrategy initialized successfully with k_block_size=%zu, v_block_size=%zu",
                     config_.k_block_size, config_.v_block_size);
    return true;
}

std::vector<torch::Tensor> KVFirstLayoutStrategy::getLayerCacheTensors() const {
    return layer_kv_tensors_;
}

BlockAddrInfo KVFirstLayoutStrategy::convertIndexToAddr(int layer_id, int block_id) const {
    if (layer_id >= config_.layer_num) {
        RTP_LLM_LOG_ERROR("Layer ID %d out of range (max: %d)", layer_id, config_.layer_num);
        return {nullptr, nullptr, nullptr, nullptr};
    }
    
    return {k_cache_tensor_[layer_id][block_id].data_ptr(), v_cache_tensor_[layer_id][block_id].data_ptr(), nullptr, nullptr};
}

BlockBufferInfo KVFirstLayoutStrategy::convertIndexToBuffer(int layer_id, int block_id) const {
    if (layer_id >= config_.layer_num) {
        RTP_LLM_LOG_ERROR("Layer ID %d out of range (max: %d)", layer_id, config_.layer_num);
        return {nullptr, nullptr};
    }
    
    torch::Tensor k_tensor = k_cache_tensor_[layer_id][block_id];
    BufferPtr k_buffer = torchTensor2Buffer(k_tensor);
    torch::Tensor v_tensor = v_cache_tensor_[layer_id][block_id];
    BufferPtr v_buffer = torchTensor2Buffer(v_tensor);
    return {k_buffer, v_buffer};
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

KVCacheBuffer KVFirstLayoutStrategy::kvCacheBuffer() const {
    KVCacheBuffer buffer;
    
    if (!k_cache_tensor_.defined() || !v_cache_tensor_.defined() || 
        k_cache_tensor_.numel() == 0 || v_cache_tensor_.numel() == 0) {
        RTP_LLM_LOG_WARNING("KV cache tensors are not initialized");
        return buffer;
    }
    
    std::vector<size_t> k_shape;
    for (int i = 0; i < k_cache_tensor_.dim(); ++i) {
        k_shape.push_back(k_cache_tensor_.size(i));
    }
    
    std::vector<size_t> v_shape;
    for (int i = 0; i < v_cache_tensor_.dim(); ++i) {
        v_shape.push_back(v_cache_tensor_.size(i));
    }
    
    auto memory_type = k_cache_tensor_.is_cuda() ? rtp_llm::MEMORY_GPU : rtp_llm::MEMORY_CPU;
    
    if (data_type_ == rtp_llm::TYPE_INVALID) {
        RTP_LLM_LOG_WARNING("DataType not initialized in KVFirstLayoutStrategy");
        return buffer;
    }
    
    buffer.k_blocks = std::make_shared<rtp_llm::Buffer>(
        memory_type, data_type_, k_shape, k_cache_tensor_.data_ptr());
    buffer.v_blocks = std::make_shared<rtp_llm::Buffer>(
        memory_type, data_type_, v_shape, v_cache_tensor_.data_ptr());
    
    buffer.k_scale = nullptr;
    buffer.v_scale = nullptr;
    
    RTP_LLM_LOG_DEBUG("KVFirstLayoutStrategy created kvCacheBuffer: K shape [%zu, %zu, %zu], V shape [%zu, %zu, %zu], dtype=%d",
                     k_shape.size() > 0 ? k_shape[0] : 0,
                     k_shape.size() > 1 ? k_shape[1] : 0,
                     k_shape.size() > 2 ? k_shape[2] : 0,
                     v_shape.size() > 0 ? v_shape[0] : 0,
                     v_shape.size() > 1 ? v_shape[1] : 0,
                     v_shape.size() > 2 ? v_shape[2] : 0,
                     static_cast<int>(data_type_));
    
    return buffer;
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
