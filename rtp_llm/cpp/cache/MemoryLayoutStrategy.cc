#include "rtp_llm/cpp/cache/MemoryLayoutStrategy.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"

namespace rtp_llm {

// Initialization function
bool MemoryLayoutStrategy::init(const MemoryLayoutConfig& config,
                                torch::Tensor&            kv_cache_tensor,
                                torch::Tensor&            kv_scale_tensor,
                                void*                     cache_base_ptr) {
    config_         = config;
    cache_base_ptr_ = cache_base_ptr;
    data_type_      = config_.dtype;

    RTP_LLM_CHECK_WITH_INFO(data_type_ != rtp_llm::TYPE_INVALID, "dtype must be set");
    RTP_LLM_CHECK_WITH_INFO(kv_cache_tensor.numel() > 0, "Cache tensor is empty, cannot split by layers");

    clearTensor(kv_cache_tensor, kv_scale_tensor);
    processKVTensor(kv_cache_tensor);
    processScaleTensor(kv_scale_tensor);
    initializeCacheBuffers(kv_cache_tensor, kv_scale_tensor, cache_base_ptr);

    RTP_LLM_LOG_INFO("MemoryLayoutStrategy initialized successfully");
    return true;
}

// Clear tensor function
void MemoryLayoutStrategy::clearTensor(torch::Tensor& kv_cache_tensor, torch::Tensor& kv_scale_tensor) {
    // Fill the cache buffers with appropriate values based on data type
    kv_cache_tensor.fill_(0);
    if (config_.hasScale()) {
        if (config_.dtype == rtp_llm::TYPE_FP8_E4M3) {
            kv_scale_tensor.fill_(1.0);
        } else {
            kv_scale_tensor.fill_(0);
        }
    }
}

// Shape computation functions
std::vector<size_t> MemoryLayoutStrategy::computeKvShape() const {
    if (config_.enable_hybrid_attention) {
        // For hybrid attention, the shape is [layer_num, block_num, kv_block_stride_elems]
        return {static_cast<size_t>(config_.layer_num),
                static_cast<size_t>(config_.block_num),
                static_cast<size_t>(config_.kv_block_stride_bytes / rtp_llm::getTypeSize(data_type_))};
    }

    // For non-hybrid attention, calculate based on model configuration
    const auto layer_num          = static_cast<size_t>(config_.layer_num);
    const auto block_num          = static_cast<size_t>(config_.block_num);
    const auto seq_size_per_block = static_cast<size_t>(config_.seq_size_per_block);
    const auto k_dim              = static_cast<size_t>(config_.k_dim);
    const auto v_dim              = static_cast<size_t>(config_.v_dim);
    const auto local_head_num_kv  = static_cast<size_t>(config_.local_head_num_kv);

    if (config_.is_mla) {
        return {layer_num, block_num, seq_size_per_block, k_dim + v_dim};
    } else {
        // check k_dim and v_dim are equal
        if (config_.k_dim != config_.v_dim) {
            RTP_LLM_LOG_ERROR("k_dim and v_dim are not equal");
            return {};  // Return empty vector to indicate error
        }
        return {layer_num, block_num, 2, local_head_num_kv, seq_size_per_block, k_dim};
    }
}

std::vector<size_t> MemoryLayoutStrategy::computeScaleShape() const {
    if (config_.enable_hybrid_attention) {
        // For hybrid attention, scale shape is [layer_num, block_num, kv_scale_stride_elems]
        return {static_cast<size_t>(config_.layer_num),
                static_cast<size_t>(config_.block_num),
                static_cast<size_t>(config_.kv_scale_stride_bytes / sizeof(float))};
    }

    // For non-hybrid attention, scale shape depends on the model type
    const auto layer_num          = static_cast<size_t>(config_.layer_num);
    const auto block_num          = static_cast<size_t>(config_.block_num);
    const auto seq_size_per_block = static_cast<size_t>(config_.seq_size_per_block);
    const auto local_head_num_kv  = static_cast<size_t>(config_.local_head_num_kv);

    if (config_.is_mla) {
        return {layer_num, block_num, seq_size_per_block, 1};  // MLA uses different scale shape
    } else {
        return {layer_num, block_num, 2, local_head_num_kv, seq_size_per_block};
    }
}

// Tensor processing functions
void MemoryLayoutStrategy::processKVTensor(torch::Tensor& kv_cache_tensor) {
    const size_t kv_elem_size          = rtp_llm::getTypeSize(data_type_);
    const size_t kv_block_stride_elems = config_.kv_block_stride_bytes / kv_elem_size;

    auto kv_options = torch::TensorOptions()
                          .dtype(dataTypeToTorchType(data_type_))
                          .device(kv_cache_tensor.device())
                          .requires_grad(false);
    const int64_t kv_total_bytes  = static_cast<int64_t>(kv_cache_tensor.nbytes());
    const int64_t kv_typed_numel  = static_cast<int64_t>(static_cast<size_t>(kv_total_bytes) / kv_elem_size);
    torch::Tensor kv_cache_typed  = torch::from_blob(kv_cache_tensor.data_ptr(), {kv_typed_numel}, kv_options);
    torch::Tensor reshaped_tensor = kv_cache_typed.reshape({static_cast<int64_t>(config_.layer_num),
                                                            static_cast<int64_t>(config_.block_num),
                                                            static_cast<int64_t>(kv_block_stride_elems)});

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
}

bool MemoryLayoutStrategy::processScaleTensor(torch::Tensor& kv_scale_tensor) {
    if (!config_.hasScale()) {
        // No scale processing needed
        layer_kv_scale_tensors_.clear();
        return true;
    }

    RTP_LLM_CHECK_WITH_INFO(kv_scale_tensor.defined() && kv_scale_tensor.numel() > 0,
                            "kv_scale_tensor must be provided when kv scale is enabled");
    RTP_LLM_CHECK_WITH_INFO(
        kv_scale_tensor.dim() == 1, "kv_scale_tensor must be 1-D, got dim=%ld", kv_scale_tensor.dim());
    RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(kv_scale_tensor.numel()) % sizeof(float) == 0,
                            "kv_scale_tensor bytes must be divisible by sizeof(float): bytes=%ld",
                            kv_scale_tensor.numel());
    RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(kv_scale_tensor.numel()) == config_.kv_scale_pool_size_bytes,
                            "kv_scale_tensor bytes mismatch: got=%ld expect=%zu",
                            kv_scale_tensor.numel(),
                            config_.kv_scale_pool_size_bytes);
    RTP_LLM_CHECK_WITH_INFO(config_.kv_scale_stride_bytes % sizeof(float) == 0,
                            "kv_scale_stride_bytes must be divisible by sizeof(float): stride_bytes=%zu",
                            config_.kv_scale_stride_bytes);

    const size_t scale_stride_elems = config_.kv_scale_stride_bytes / sizeof(float);
    auto         scale_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(kv_scale_tensor.device()).requires_grad(false);
    const int64_t scale_total_bytes = static_cast<int64_t>(kv_scale_tensor.nbytes());
    const int64_t scale_typed_numel = static_cast<int64_t>(static_cast<size_t>(scale_total_bytes) / sizeof(float));
    torch::Tensor kv_scale_typed    = torch::from_blob(kv_scale_tensor.data_ptr(), {scale_typed_numel}, scale_options);
    torch::Tensor reshaped_scale_tensor = kv_scale_typed.reshape({static_cast<int64_t>(config_.layer_num),
                                                                  static_cast<int64_t>(config_.block_num),
                                                                  static_cast<int64_t>(scale_stride_elems)});
    layer_kv_scale_tensors_.clear();
    layer_kv_scale_tensors_.reserve(config_.layer_num);
    for (uint32_t layer_id = 0; layer_id < config_.layer_num; ++layer_id) {
        layer_kv_scale_tensors_.push_back(reshaped_scale_tensor[layer_id]);

        RTP_LLM_LOG_DEBUG("Layer %d scale tensor shape: [%s], elements: %ld",
                          layer_id,
                          torch::str(layer_kv_scale_tensors_[layer_id].sizes()).c_str(),
                          layer_kv_scale_tensors_[layer_id].numel());
    }

    return true;
}

// Cache buffer initialization functions
void MemoryLayoutStrategy::initializeCacheBuffers(torch::Tensor& kv_cache_tensor,
                                                  torch::Tensor& kv_scale_tensor,
                                                  void*          cache_base_ptr) {
    std::vector<size_t> kv_shape = computeKvShape();
    RTP_LLM_CHECK_WITH_INFO(!kv_shape.empty(), "Failed to compute KV shape");

    std::vector<size_t> scale_shape = computeScaleShape();
    RTP_LLM_CHECK_WITH_INFO(!scale_shape.empty(), "Failed to compute Scale shape");

    initializeKvCacheBuffer(config_, kv_cache_tensor, kv_scale_tensor, cache_base_ptr, kv_shape, scale_shape);
}

void MemoryLayoutStrategy::initializeKvCacheBuffer(const MemoryLayoutConfig&  config,
                                                   torch::Tensor&             kv_cache_tensor,
                                                   torch::Tensor&             kv_scale_tensor,
                                                   void*                      cache_base_ptr,
                                                   const std::vector<size_t>& kv_shape,
                                                   const std::vector<size_t>& scale_shape) {
    // Initialize the main KV cache buffer
    auto memory_type           = kv_cache_tensor.is_cuda() ? rtp_llm::MEMORY_GPU : rtp_llm::MEMORY_CPU;
    kv_cache_buffer_.kv_blocks = std::make_shared<rtp_llm::Buffer>(memory_type, config.dtype, kv_shape, cache_base_ptr);

    // Handle scale buffer if needed
    if (config.hasScale()) {
        kv_scale_base_ptr_ = kv_scale_tensor.data_ptr();

        kv_cache_buffer_.kv_scale_blocks = std::make_shared<rtp_llm::Buffer>(
            memory_type, rtp_llm::DataType::TYPE_FP32, scale_shape, kv_scale_base_ptr_);
    }
}

// Address and buffer conversion functions
BlockAddrInfo MemoryLayoutStrategy::convertIndexToAddr(int layer_id, int block_id) const {
    auto  blocks        = convertIndexToBuffer(layer_id, block_id);
    void* kv_addr       = blocks[0].addr;
    void* kv_scale_addr = nullptr;

    if (config_.hasScale() && blocks.size() > 1) {
        kv_scale_addr = blocks[1].addr;
    }

    return {kv_addr, kv_scale_addr};
}

std::vector<BlockInfo> MemoryLayoutStrategy::convertIndexToBuffer(int layer_id, int block_id) const {
    return createBasicBlockInfo(layer_id, block_id);
}

std::vector<BlockInfo>
MemoryLayoutStrategy::convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const {
    // Hybrid attention models are not support asymmetric TP, thus transfer the whole kvache blocks
    if (config_.is_mla || config_.enable_hybrid_attention) {
        // For MLA models and hybrid attention models, use the same logic as the simpler convertIndexToBuffer function
        return createBasicBlockInfo(layer_id, block_id);
    }

    // TODO(xinfei.sxf) deal with linear attention

    // For non-MLA models with partitioning
    return createPartitionedBlockInfo(layer_id, block_id, partition_count, partition_id);
}

// Helper functions for creating block info
std::vector<BlockInfo> MemoryLayoutStrategy::createBasicBlockInfo(int layer_id, int block_id) const {
    checkLayerIdValidity(layer_id);
    auto& layer_tensor = layer_kv_tensors_[layer_id];
    void* kv_addr      = layer_tensor[block_id].data_ptr();
    auto  kv_info = makeBlockInfo(layer_tensor[block_id], kv_addr, static_cast<size_t>(config_.kv_block_stride_bytes));

    if (config_.hasScale()) {
        auto& layer_scale_tensor = layer_kv_scale_tensors_[layer_id];
        void* kv_scale_addr      = layer_scale_tensor[block_id].data_ptr();
        auto  scale_info         = makeBlockInfo(
            layer_scale_tensor[block_id], kv_scale_addr, static_cast<size_t>(config_.kv_scale_stride_bytes));
        return {kv_info, scale_info};
    }

    return {kv_info};
}

std::vector<BlockInfo> MemoryLayoutStrategy::createPartitionedBlockInfo(int layer_id,
                                                                        int block_id,
                                                                        int partition_count,
                                                                        int partition_id) const {
    checkLayerIdValidity(layer_id);
    auto& layer_tensor = layer_kv_tensors_[layer_id];
    void* kv_addr      = layer_tensor[block_id].data_ptr();

    const int heads = static_cast<int>(config_.local_head_num_kv);

    auto kv_parts = MHAKVCacheSpec::splitKVPartitionBytes(static_cast<size_t>(config_.kv_block_stride_bytes),
                                                          static_cast<size_t>(config_.k_block_stride_bytes),
                                                          static_cast<size_t>(config_.v_block_stride_bytes),
                                                          heads,
                                                          partition_count,
                                                          partition_id,
                                                          "kv_cache");

    std::vector<BlockInfo> out = createPartitionedSubBlocks(layer_tensor[block_id], kv_addr, kv_parts);

    if (config_.hasScale()) {
        auto& layer_scale_tensor = layer_kv_scale_tensors_[layer_id];
        void* scale_addr         = layer_scale_tensor[block_id].data_ptr();
        auto  sc_parts     = MHAKVCacheSpec::splitKVPartitionBytes(static_cast<size_t>(config_.kv_scale_stride_bytes),
                                                              static_cast<size_t>(config_.k_scale_stride_bytes),
                                                              static_cast<size_t>(config_.v_scale_stride_bytes),
                                                              heads,
                                                              partition_count,
                                                              partition_id,
                                                              "kv_cache_scale");
        auto  scale_blocks = createPartitionedSubBlocks(layer_scale_tensor[block_id], scale_addr, sc_parts);
        out.insert(out.end(), scale_blocks.begin(), scale_blocks.end());
    }

    return out;
}

std::vector<BlockInfo> MemoryLayoutStrategy::createPartitionedSubBlocks(const torch::Tensor&    layer_tensor,
                                                                        void*                   base_addr,
                                                                        const KVPartitionBytes& parts) const {
    void* k_ptr   = static_cast<char*>(base_addr) + parts.k_off;
    void* v_ptr   = static_cast<char*>(base_addr) + parts.v_off;
    auto  k_block = makeBlockInfo(layer_tensor, k_ptr, parts.k_sz);
    auto  v_block = makeBlockInfo(layer_tensor, v_ptr, parts.v_sz);
    return {k_block, v_block};
}

BlockInfo MemoryLayoutStrategy::makeBlockInfo(const torch::Tensor& tensor, void* addr, size_t size_bytes) const {
    auto      dev = tensor.device();
    BlockInfo info;
    info.is_cuda      = dev.is_cuda();
    info.device_index = dev.index();
    info.scalar_type  = static_cast<int32_t>(tensor.scalar_type());
    info.addr         = addr;
    info.size_bytes   = size_bytes;
    return info;
}

// Getter functions
std::vector<torch::Tensor> MemoryLayoutStrategy::getLayerCacheTensors() const {
    return layer_kv_tensors_;
}

std::vector<torch::Tensor> MemoryLayoutStrategy::getLayerScaleCacheTensors() const {
    return layer_kv_scale_tensors_;
}

void* MemoryLayoutStrategy::getKCacheAddr(int layer_id, int block_id) const {
    auto blocks = convertIndexToBuffer(layer_id, block_id);
    return blocks[0].addr;
}

void* MemoryLayoutStrategy::getVCacheAddr(int layer_id, int block_id) const {
    auto blocks = convertIndexToBuffer(layer_id, block_id);
    return blocks[0].addr;
}

const KVCacheBuffer& MemoryLayoutStrategy::kvCacheBuffer() const {
    return kv_cache_buffer_;
}

// Utility functions
void MemoryLayoutStrategy::checkLayerIdValidity(int layer_id) const {
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layer_kv_tensors_.size(),
                            "Layer ID %d out of range (max: %zu)",
                            layer_id,
                            layer_kv_tensors_.size());
}

}  // namespace rtp_llm