#include "rtp_llm/cpp/cache_new/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// LayerFirstLayoutStrategy
bool LayerFirstLayoutStrategy::init(const BlockPoolConfig& config,
                                    torch::Tensor&         cache_buffer,
                                    void*                  cache_base_ptr,
                                    rtp_llm::DataType      data_type) {
    config_         = config;
    cache_base_ptr_ = cache_base_ptr;
    data_type_      = data_type;

    if (cache_buffer.numel() == 0) {
        RTP_LLM_LOG_ERROR("Cache buffer tensor is empty, cannot split by layers");
        return false;
    }

    torch::Tensor reshaped_tensor = cache_buffer.reshape({static_cast<int64_t>(config_.layer_num),
                                                          static_cast<int64_t>(config_.block_num),
                                                          static_cast<int64_t>(config_.block_size)});

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

    const auto layer_num          = static_cast<size_t>(config_.layer_num);
    const auto block_num          = static_cast<size_t>(config_.block_num);
    const auto seq_size_per_block = static_cast<size_t>(config_.seq_size_per_block);
    const auto k_token_size       = static_cast<size_t>(config_.k_token_size);
    const auto v_token_size       = static_cast<size_t>(config_.v_token_size);
    const auto local_head_num_kv  = static_cast<size_t>(config_.local_head_num_kv);

    // for adaption use k_blocks as base ptr
    std::vector<size_t> k_shape;
    std::vector<size_t> v_shape;

    if (config_.is_mla) {
        k_shape = {layer_num, block_num, seq_size_per_block, k_token_size + v_token_size};
        v_shape = {layer_num, block_num, seq_size_per_block, (size_t)0};
    } else {
        // check k_token_size and v_token_size are equal
        if (config_.k_token_size != config_.v_token_size) {
            RTP_LLM_LOG_ERROR("k_token_size and v_token_size are not equal");
            return false;
        }
        k_shape = {layer_num, block_num, 2, local_head_num_kv, seq_size_per_block, k_token_size};
        v_shape = {layer_num, block_num, 2, local_head_num_kv, seq_size_per_block, (size_t)0};
    }

    auto memory_type = cache_buffer.is_cuda() ? rtp_llm::MEMORY_GPU : rtp_llm::MEMORY_CPU;

    kv_cache_buffer_.k_blocks = std::make_shared<rtp_llm::Buffer>(memory_type, data_type_, k_shape, cache_base_ptr_);
    kv_cache_buffer_.v_blocks = std::make_shared<rtp_llm::Buffer>(
        memory_type, data_type_, v_shape, (void*)((char*)cache_base_ptr_ + kv_cache_buffer_.k_blocks->size()));

    RTP_LLM_LOG_INFO("LayerFirstLayoutStrategy initialized successfully");
    return true;
}

std::vector<torch::Tensor> LayerFirstLayoutStrategy::getLayerCacheTensors() const {
    return layer_kv_tensors_;
}

BlockAddrInfo LayerFirstLayoutStrategy::convertIndexToAddr(int layer_id, int block_id) const {
    if (layer_id >= layer_kv_tensors_.size()) {
        RTP_LLM_LOG_ERROR("Layer ID %d out of range (max: %zu)", layer_id, layer_kv_tensors_.size());
        return {nullptr, nullptr};
    }

    torch::Tensor tensor = layer_kv_tensors_[layer_id][block_id];
    return {tensor.data_ptr(), tensor.data_ptr()};
}

BlockBufferPtrInfo LayerFirstLayoutStrategy::convertIndexToBuffer(int layer_id, int block_id) const {
    if (layer_id >= layer_kv_tensors_.size()) {
        RTP_LLM_LOG_ERROR("Layer ID %d out of range (max: %zu)", layer_id, layer_kv_tensors_.size());
        return {nullptr, nullptr};
    }

    torch::Tensor tensor = layer_kv_tensors_[layer_id][block_id];
    BufferPtr     buffer = torchTensor2Buffer(tensor);
    return {buffer, buffer};
}

std::vector<BufferPtr> LayerFirstLayoutStrategy::convertIndexToBuffer(int layer_id,
                                                                      int block_id,
                                                                      int partition_count,
                                                                      int partition_id) const {
    if (layer_id >= layer_kv_tensors_.size()) {
        RTP_LLM_LOG_ERROR("Layer ID %d out of range (max: %zu)", layer_id, layer_kv_tensors_.size());
        return {};
    }

    torch::Tensor tensor = layer_kv_tensors_[layer_id][block_id];

    const size_t k_total_bytes = static_cast<size_t>(config_.k_block_size);
    const size_t v_total_bytes = static_cast<size_t>(config_.v_block_size);
    const int    heads         = static_cast<int>(config_.local_head_num_kv);

    RTP_LLM_CHECK_WITH_INFO(partition_count > 0, "partition_count must be > 0");
    RTP_LLM_CHECK_WITH_INFO(partition_id >= 0 && partition_id < partition_count,
                            "partition_id out of range: %d / %d",
                            partition_id,
                            partition_count);
    RTP_LLM_CHECK_WITH_INFO(tensor.defined(), "tensor is not defined");
    RTP_LLM_CHECK_WITH_INFO(tensor.dim() == 1, "tensor must be 1-D, got dim=%ld", tensor.dim());

    const size_t full_bytes = static_cast<size_t>(tensor.numel());
    RTP_LLM_CHECK_WITH_INFO(k_total_bytes + v_total_bytes == full_bytes,
                            "block bytes mismatch: full=%zu k=%zu v=%zu",
                            full_bytes,
                            k_total_bytes,
                            v_total_bytes);

    const size_t k_bytes_per_head = k_total_bytes / static_cast<size_t>(heads);
    const size_t v_bytes_per_head = v_total_bytes / static_cast<size_t>(heads);

    RTP_LLM_CHECK_WITH_INFO(heads % partition_count == 0,
                            "heads must be divisible by partition_count, heads=%d partition_count=%d",
                            heads,
                            partition_count);

    // Compute [head_begin, head_cnt] for this partition_id (equal split).
    const int head_cnt   = heads / partition_count;
    const int head_begin = partition_id * head_cnt;

    const size_t k_off = static_cast<size_t>(head_begin) * k_bytes_per_head;
    const size_t v_off = k_total_bytes + static_cast<size_t>(head_begin) * v_bytes_per_head;
    const size_t k_sz  = static_cast<size_t>(head_cnt) * k_bytes_per_head;
    const size_t v_sz  = static_cast<size_t>(head_cnt) * v_bytes_per_head;

    auto k_part = tensor.narrow(0, static_cast<int64_t>(k_off), static_cast<int64_t>(k_sz));
    auto v_part = tensor.narrow(0, static_cast<int64_t>(v_off), static_cast<int64_t>(v_sz));
    return {torchTensor2Buffer(k_part), torchTensor2Buffer(v_part)};
}

void* LayerFirstLayoutStrategy::getKCacheAddr(int layer_id, int block_id) const {
    auto addr_info = convertIndexToAddr(layer_id, block_id);
    return addr_info.k_addr;
}

void* LayerFirstLayoutStrategy::getVCacheAddr(int layer_id, int block_id) const {
    auto addr_info = convertIndexToAddr(layer_id, block_id);
    return addr_info.v_addr;
}

const KVCacheBuffer& LayerFirstLayoutStrategy::kvCacheBuffer() const {
    return kv_cache_buffer_;
}

std::unique_ptr<MemoryLayoutStrategy> MemoryLayoutStrategyFactory::create(MemoryLayout layout) {
    switch (layout) {
        case LAYER_FIRST:
            return std::make_unique<LayerFirstLayoutStrategy>();
        default:
            RTP_LLM_LOG_ERROR("Unknown memory layout type: %d", static_cast<int>(layout));
            return nullptr;
    }
}

}  // namespace rtp_llm
