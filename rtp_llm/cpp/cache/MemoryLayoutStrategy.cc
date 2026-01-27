#include "rtp_llm/cpp/cache/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

// for p2p connector when TP settings of prefill & decode are different.
struct KVPartition {
    torch::Tensor k_partition;
    torch::Tensor v_partition;
};

inline KVPartition splitKVPartition(const torch::Tensor& tensor,
                                    size_t               k_block_stride_bytes,
                                    size_t               v_block_stride_bytes,
                                    int                  heads,
                                    int                  partition_count,
                                    int                  partition_id,
                                    const char*          debug_name) {
    RTP_LLM_CHECK_WITH_INFO(partition_count > 0, "partition_count must be > 0");
    RTP_LLM_CHECK_WITH_INFO(partition_id >= 0 && partition_id < partition_count,
                            "partition_id out of range: %d / %d",
                            partition_id,
                            partition_count);
    RTP_LLM_CHECK_WITH_INFO(heads > 0, "heads must be > 0, got=%d (%s)", heads, debug_name);
    RTP_LLM_CHECK_WITH_INFO(tensor.defined(), "tensor is not defined (%s)", debug_name);
    RTP_LLM_CHECK_WITH_INFO(tensor.dim() == 1, "tensor must be 1-D, got dim=%ld (%s)", tensor.dim(), debug_name);

    const size_t full_bytes = static_cast<size_t>(tensor.numel());
    RTP_LLM_CHECK_WITH_INFO(k_block_stride_bytes + v_block_stride_bytes == full_bytes,
                            "block bytes mismatch (%s): full=%zu k_partition=%zu v_partition=%zu",
                            debug_name,
                            full_bytes,
                            k_block_stride_bytes,
                            v_block_stride_bytes);
    RTP_LLM_CHECK_WITH_INFO(k_block_stride_bytes % static_cast<size_t>(heads) == 0,
                            "k_block_stride_bytes must be divisible by heads (%s): k_partition=%zu heads=%d",
                            debug_name,
                            k_block_stride_bytes,
                            heads);
    RTP_LLM_CHECK_WITH_INFO(v_block_stride_bytes % static_cast<size_t>(heads) == 0,
                            "v_block_stride_bytes must be divisible by heads (%s): v_partition=%zu heads=%d",
                            debug_name,
                            v_block_stride_bytes,
                            heads);
    RTP_LLM_CHECK_WITH_INFO(heads % partition_count == 0,
                            "heads must be divisible by partition_count (%s): heads=%d partition_count=%d",
                            debug_name,
                            heads,
                            partition_count);

    const size_t k_partition_bytes_per_head = k_block_stride_bytes / static_cast<size_t>(heads);
    const size_t v_partition_bytes_per_head = v_block_stride_bytes / static_cast<size_t>(heads);

    // Compute [head_begin, head_cnt] for this partition_id (equal split).
    const int head_cnt   = heads / partition_count;
    const int head_begin = partition_id * head_cnt;

    const size_t k_partition_off = static_cast<size_t>(head_begin) * k_partition_bytes_per_head;
    const size_t v_partition_off = k_block_stride_bytes + static_cast<size_t>(head_begin) * v_partition_bytes_per_head;
    const size_t k_partition_sz  = static_cast<size_t>(head_cnt) * k_partition_bytes_per_head;
    const size_t v_partition_sz  = static_cast<size_t>(head_cnt) * v_partition_bytes_per_head;

    auto k_partition_part =
        tensor.narrow(0, static_cast<int64_t>(k_partition_off), static_cast<int64_t>(k_partition_sz));
    auto v_partition_part =
        tensor.narrow(0, static_cast<int64_t>(v_partition_off), static_cast<int64_t>(v_partition_sz));
    return {k_partition_part, v_partition_part};
}

}  // namespace

// LayerFirstLayoutStrategy
bool LayerFirstLayoutStrategy::init(const MemoryLayoutConfig& config,
                                    torch::Tensor&            kv_cache_buffer,
                                    torch::Tensor&            kv_scale_buffer,
                                    void*                     cache_base_ptr,
                                    rtp_llm::DataType         data_type) {
    config_         = config;
    cache_base_ptr_ = cache_base_ptr;
    data_type_      = data_type;

    if (kv_cache_buffer.numel() == 0) {
        RTP_LLM_LOG_ERROR("Cache buffer tensor is empty, cannot split by layers");
        return false;
    }

    torch::Tensor reshaped_tensor = kv_cache_buffer.reshape({static_cast<int64_t>(config_.layer_num),
                                                             static_cast<int64_t>(config_.block_num),
                                                             static_cast<int64_t>(config_.kv_block_stride_bytes)});
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

    // for adaption use kv_blocks as base ptr
    std::vector<size_t> kv_shape;

    if (config_.is_mla) {
        kv_shape = {layer_num, block_num, seq_size_per_block, k_token_size + v_token_size};
    } else {
        // check k_token_size and v_token_size are equal
        if (config_.k_token_size != config_.v_token_size) {
            RTP_LLM_LOG_ERROR("k_token_size and v_token_size are not equal");
            return false;
        }
        kv_shape = {layer_num, block_num, 2, local_head_num_kv, seq_size_per_block, k_token_size};
    }

    auto memory_type = kv_cache_buffer.is_cuda() ? rtp_llm::MEMORY_GPU : rtp_llm::MEMORY_CPU;

    kv_cache_buffer_.kv_blocks = std::make_shared<rtp_llm::Buffer>(memory_type, data_type_, kv_shape, cache_base_ptr_);

#if (defined(USING_ROCM) && USING_ROCM) || (defined(USING_CUDA) && USING_CUDA)
    Buffer2torchTensor(kv_cache_buffer_.kv_blocks, false).fill_(0);
#endif

    if (config_.enable_kv_scale && config_.kv_scale_pool_size_bytes > 0 && config_.k_scale_stride_bytes > 0
        && config_.v_scale_stride_bytes > 0) {
        RTP_LLM_CHECK_WITH_INFO(kv_scale_buffer.defined() && kv_scale_buffer.numel() > 0,
                                "kv_scale_buffer must be provided when enable_kv_scale is true");
        RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(kv_scale_buffer.numel()) == config_.kv_scale_pool_size_bytes,
                                "kv_scale_buffer bytes mismatch: got=%ld expect=%zu",
                                kv_scale_buffer.numel(),
                                config_.kv_scale_pool_size_bytes);

        kv_scale_base_ptr_ = kv_scale_buffer.data_ptr();

        // Keep a buffer view for kernels / model (FP32 scale blocks).
        std::vector<size_t> scale_shape  = {layer_num, block_num, 2, local_head_num_kv, seq_size_per_block};
        kv_cache_buffer_.kv_scale_blocks = std::make_shared<rtp_llm::Buffer>(
            memory_type, rtp_llm::DataType::TYPE_FP32, scale_shape, kv_scale_base_ptr_);

        torch::Tensor reshaped_scale_tensor =
            kv_scale_buffer.reshape({static_cast<int64_t>(config_.layer_num),
                                     static_cast<int64_t>(config_.block_num),
                                     static_cast<int64_t>(config_.kv_scale_stride_bytes)});

        layer_kv_scale_tensors_.clear();
        layer_kv_scale_tensors_.reserve(config_.layer_num);

        for (uint32_t layer_id = 0; layer_id < config_.layer_num; ++layer_id) {
            torch::Tensor layer_tensor = reshaped_scale_tensor[layer_id];
            layer_kv_scale_tensors_.push_back(layer_tensor);

            RTP_LLM_LOG_DEBUG("Layer %d scale tensor shape: [%s], elements: %ld",
                              layer_id,
                              torch::str(layer_tensor.sizes()).c_str(),
                              layer_tensor.numel());
        }

        if (data_type_ == rtp_llm::TYPE_FP8_E4M3) {
            Buffer2torchTensor(kv_cache_buffer_.kv_scale_blocks, false).fill_(1.0);
        }
    }

    RTP_LLM_LOG_INFO("LayerFirstLayoutStrategy initialized successfully");
    return true;
}

std::vector<torch::Tensor> LayerFirstLayoutStrategy::getLayerCacheTensors() const {
    return layer_kv_tensors_;
}

std::vector<torch::Tensor> LayerFirstLayoutStrategy::getLayerScaleCacheTensors() const {
    return layer_kv_scale_tensors_;
}

BlockAddrInfo LayerFirstLayoutStrategy::convertIndexToAddr(int layer_id, int block_id) const {
    checkLayerIdValidity(layer_id);
    auto& layer_tensor = layer_kv_tensors_[layer_id];
    void* kv_addr =
        static_cast<char*>(layer_tensor.data_ptr()) + block_id * layer_tensor.stride(0) * layer_tensor.element_size();

    if (config_.enable_kv_scale) {
        auto& layer_scale_tensor = layer_kv_scale_tensors_[layer_id];
        void* kv_scale_addr      = static_cast<char*>(layer_scale_tensor.data_ptr())
                              + block_id * layer_scale_tensor.stride(0) * layer_scale_tensor.element_size();
        return {kv_addr, kv_scale_addr};
    }

    return {kv_addr, nullptr};
}

BlockBufferPtrInfo LayerFirstLayoutStrategy::convertIndexToBuffer(int layer_id, int block_id) const {
    checkLayerIdValidity(layer_id);
    torch::Tensor tensor = layer_kv_tensors_[layer_id][block_id];
    BufferPtr     buffer = torchTensor2Buffer(tensor);

    if (config_.enable_kv_scale) {
        torch::Tensor scale_tensor = layer_kv_scale_tensors_[layer_id][block_id];
        BufferPtr     scale_buffer = torchTensor2Buffer(scale_tensor);
        return {buffer, scale_buffer};
    }

    return {buffer, nullptr};
}

std::vector<BufferPtr> LayerFirstLayoutStrategy::convertIndexToBuffer(int layer_id,
                                                                      int block_id,
                                                                      int partition_count,
                                                                      int partition_id) const {
    checkLayerIdValidity(layer_id);
    torch::Tensor tensor = layer_kv_tensors_[layer_id][block_id];

    if (config_.is_mla) {
        if (config_.enable_kv_scale) {
            torch::Tensor scale_tensor = layer_kv_scale_tensors_[layer_id][block_id];
            return {torchTensor2Buffer(tensor), torchTensor2Buffer(scale_tensor)};
        }
        return {torchTensor2Buffer(tensor)};
    }

    const size_t k_total_bytes = static_cast<size_t>(config_.k_block_stride_bytes);
    const size_t v_total_bytes = static_cast<size_t>(config_.v_block_stride_bytes);
    const int    heads         = static_cast<int>(config_.local_head_num_kv);

    auto kv_parts =
        splitKVPartition(tensor, k_total_bytes, v_total_bytes, heads, partition_count, partition_id, "kv_cache");
    std::vector<BufferPtr> out = {torchTensor2Buffer(kv_parts.k_partition), torchTensor2Buffer(kv_parts.v_partition)};

    if (config_.enable_kv_scale) {
        torch::Tensor scale_tensor  = layer_kv_scale_tensors_[layer_id][block_id];
        const size_t  k_scale_bytes = static_cast<size_t>(config_.k_scale_stride_bytes);
        const size_t  v_scale_bytes = static_cast<size_t>(config_.v_scale_stride_bytes);
        auto          sc_parts      = splitKVPartition(
            scale_tensor, k_scale_bytes, v_scale_bytes, heads, partition_count, partition_id, "kv_cache_scale");
        out.push_back(torchTensor2Buffer(sc_parts.k_partition));
        out.push_back(torchTensor2Buffer(sc_parts.v_partition));
    }

    return out;
}

void* LayerFirstLayoutStrategy::getKCacheAddr(int layer_id, int block_id) const {
    auto addr_info = convertIndexToAddr(layer_id, block_id);
    return addr_info.kv_addr;
}

void* LayerFirstLayoutStrategy::getVCacheAddr(int layer_id, int block_id) const {
    auto addr_info = convertIndexToAddr(layer_id, block_id);
    return addr_info.kv_addr;
}

const KVCacheBuffer& LayerFirstLayoutStrategy::kvCacheBuffer() const {
    return kv_cache_buffer_;
}

void LayerFirstLayoutStrategy::checkLayerIdValidity(int layer_id) const {
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layer_kv_tensors_.size(),
                            "Layer ID %d out of range (max: %zu)",
                            layer_id,
                            layer_kv_tensors_.size());
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
