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
    RTP_LLM_CHECK_WITH_INFO(kv_cache_tensor.numel() > 0, "kv cache tensor is empty, cannot split by layers");

    processKVTensor(kv_cache_tensor);
    processScaleTensor(kv_scale_tensor);

    RTP_LLM_LOG_INFO("MemoryLayoutStrategy initialized successfully");
    return true;
}

// Clear tensor function
void MemoryLayoutStrategy::clearKVTensor(torch::Tensor& kv_cache_tensor) {
    kv_cache_tensor.fill_(0);
}

void MemoryLayoutStrategy::clearScaleTensor(torch::Tensor& kv_scale_tensor) {
    if (config_.hasScale()) {
        if (config_.dtype == rtp_llm::TYPE_FP8_E4M3) {
            kv_scale_tensor.fill_(1.0);
        } else {
            kv_scale_tensor.fill_(0);
        }
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
    const int64_t kv_total_bytes = static_cast<int64_t>(kv_cache_tensor.nbytes());
    const int64_t kv_typed_numel = static_cast<int64_t>(static_cast<size_t>(kv_total_bytes) / kv_elem_size);
    torch::Tensor kv_cache_typed = torch::from_blob(kv_cache_tensor.data_ptr(), {kv_typed_numel}, kv_options);

    layer_kv_tensors_.clear();
    layer_kv_tensors_.reserve(config_.layer_num);

    if (config_.use_mla && config_.seq_size_per_block > 0) {
        // MLA: concat_and_cache_mla expects [num_blocks, block_size, stride] per layer
        RTP_LLM_CHECK_WITH_INFO(kv_block_stride_elems % config_.seq_size_per_block == 0,
                                "kv_block_stride_elems=%zu must be divisible by seq_size_per_block=%zu for MLA",
                                kv_block_stride_elems,
                                config_.seq_size_per_block);
        const size_t  stride_elems    = kv_block_stride_elems / config_.seq_size_per_block;
        torch::Tensor reshaped_tensor = kv_cache_typed.reshape({static_cast<int64_t>(config_.layer_num),
                                                                static_cast<int64_t>(config_.block_num),
                                                                static_cast<int64_t>(config_.seq_size_per_block),
                                                                static_cast<int64_t>(stride_elems)});
        clearKVTensor(reshaped_tensor);
        for (uint32_t layer_id = 0; layer_id < config_.layer_num; ++layer_id) {
            layer_kv_tensors_.push_back(reshaped_tensor[layer_id]);
            RTP_LLM_LOG_DEBUG("Layer %d KV tensor shape: [%s] (MLA 3D)",
                              layer_id,
                              torch::str(layer_kv_tensors_[layer_id].sizes()).c_str());
        }
    } else {
        // MHA: [layer_num, block_num, kv_block_stride_elems], per layer 2D
        torch::Tensor reshaped_tensor = kv_cache_typed.reshape({static_cast<int64_t>(config_.layer_num),
                                                                static_cast<int64_t>(config_.block_num),
                                                                static_cast<int64_t>(kv_block_stride_elems)});
        clearKVTensor(reshaped_tensor);
        for (uint32_t layer_id = 0; layer_id < config_.layer_num; ++layer_id) {
            layer_kv_tensors_.push_back(reshaped_tensor[layer_id]);
            RTP_LLM_LOG_DEBUG("Layer %d tensor shape: [%s], elements: %ld",
                              layer_id,
                              torch::str(layer_kv_tensors_[layer_id].sizes()).c_str(),
                              layer_kv_tensors_[layer_id].numel());
        }
    }
}

bool MemoryLayoutStrategy::processScaleTensor(torch::Tensor& kv_scale_tensor) {
    if (!config_.hasScale()) {
        return true;
    }

    RTP_LLM_CHECK_WITH_INFO(kv_scale_tensor.defined() && kv_scale_tensor.numel() > 0,
                            "kv_scale_tensor must be provided when kv scale is enabled");
    RTP_LLM_CHECK_WITH_INFO(
        kv_scale_tensor.dim() == 1, "kv_scale_tensor must be 1-D, got dim=%ld", kv_scale_tensor.dim());
    RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(kv_scale_tensor.nbytes()) == config_.kv_scale_pool_size_bytes,
                            "kv_scale_tensor bytes mismatch: got=%zu expect=%zu",
                            static_cast<size_t>(kv_scale_tensor.nbytes()),
                            config_.kv_scale_pool_size_bytes);

    if (config_.is_mla) {
        // MLA: scale is byte-packed (UINT8), shape [layer_num, block_num, seq_size_per_block, bytes_per_token]
        RTP_LLM_CHECK_WITH_INFO(config_.seq_size_per_block > 0, "seq_size_per_block must be > 0 for MLA scale");
        RTP_LLM_CHECK_WITH_INFO(config_.kv_scale_stride_bytes % config_.seq_size_per_block == 0,
                                "kv_scale_stride_bytes=%zu must be divisible by seq_size_per_block=%zu",
                                config_.kv_scale_stride_bytes,
                                config_.seq_size_per_block);

        const size_t scale_bytes_per_token = config_.kv_scale_stride_bytes / config_.seq_size_per_block;
        auto         scale_options =
            torch::TensorOptions().dtype(torch::kUInt8).device(kv_scale_tensor.device()).requires_grad(false);
        torch::Tensor kv_scale_typed = torch::from_blob(
            kv_scale_tensor.data_ptr(), {static_cast<int64_t>(config_.kv_scale_pool_size_bytes)}, scale_options);
        torch::Tensor reshaped_scale_tensor = kv_scale_typed.reshape({static_cast<int64_t>(config_.layer_num),
                                                                      static_cast<int64_t>(config_.block_num),
                                                                      static_cast<int64_t>(config_.seq_size_per_block),
                                                                      static_cast<int64_t>(scale_bytes_per_token)});
        reshaped_scale_tensor.fill_(0);

        layer_kv_scale_tensors_.clear();
        layer_kv_scale_tensors_.reserve(config_.layer_num);
        for (uint32_t layer_id = 0; layer_id < config_.layer_num; ++layer_id) {
            layer_kv_scale_tensors_.push_back(reshaped_scale_tensor[layer_id]);

            RTP_LLM_LOG_DEBUG("Layer %d scale tensor shape: [%s], elements: %ld (MLA)",
                              layer_id,
                              torch::str(layer_kv_scale_tensors_[layer_id].sizes()).c_str(),
                              layer_kv_scale_tensors_[layer_id].numel());
        }
    } else {
        // MHA: scale is FP32, shape [layer_num, block_num, scale_stride_elems] for kernel/model
        RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(kv_scale_tensor.numel()) % sizeof(float) == 0,
                                "kv_scale_tensor bytes must be divisible by sizeof(float): bytes=%ld",
                                kv_scale_tensor.numel());
        RTP_LLM_CHECK_WITH_INFO(config_.kv_scale_stride_bytes % sizeof(float) == 0,
                                "kv_scale_stride_bytes must be divisible by sizeof(float): stride_bytes=%zu",
                                config_.kv_scale_stride_bytes);

        const size_t scale_stride_elems = config_.kv_scale_stride_bytes / sizeof(float);
        auto         scale_options =
            torch::TensorOptions().dtype(torch::kFloat32).device(kv_scale_tensor.device()).requires_grad(false);
        const int64_t scale_total_bytes = static_cast<int64_t>(kv_scale_tensor.nbytes());
        const int64_t scale_typed_numel = static_cast<int64_t>(static_cast<size_t>(scale_total_bytes) / sizeof(float));
        torch::Tensor kv_scale_typed = torch::from_blob(kv_scale_tensor.data_ptr(), {scale_typed_numel}, scale_options);
        torch::Tensor reshaped_scale_tensor = kv_scale_typed.reshape({static_cast<int64_t>(config_.layer_num),
                                                                      static_cast<int64_t>(config_.block_num),
                                                                      static_cast<int64_t>(scale_stride_elems)});
        clearScaleTensor(reshaped_scale_tensor);

        layer_kv_scale_tensors_.clear();
        layer_kv_scale_tensors_.reserve(config_.layer_num);
        for (uint32_t layer_id = 0; layer_id < config_.layer_num; ++layer_id) {
            layer_kv_scale_tensors_.push_back(reshaped_scale_tensor[layer_id]);

            RTP_LLM_LOG_DEBUG("Layer %d scale tensor shape: [%s], elements: %ld",
                              layer_id,
                              torch::str(layer_kv_scale_tensors_[layer_id].sizes()).c_str(),
                              layer_kv_scale_tensors_[layer_id].numel());
        }
    }

    return true;
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
                                                          static_cast<size_t>(config_.kv_block_stride_bytes / 2),
                                                          static_cast<size_t>(config_.kv_block_stride_bytes / 2),
                                                          heads,
                                                          partition_count,
                                                          partition_id,
                                                          "kv_cache");

    std::vector<BlockInfo> out = createPartitionedSubBlocks(layer_tensor[block_id], kv_addr, kv_parts);

    if (config_.hasScale()) {
        auto& layer_scale_tensor = layer_kv_scale_tensors_[layer_id];
        void* scale_addr         = layer_scale_tensor[block_id].data_ptr();
        auto  sc_parts     = MHAKVCacheSpec::splitKVPartitionBytes(static_cast<size_t>(config_.kv_scale_stride_bytes),
                                                              static_cast<size_t>(config_.kv_scale_stride_bytes / 2),
                                                              static_cast<size_t>(config_.kv_scale_stride_bytes / 2),
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

// Utility functions
void MemoryLayoutStrategy::checkLayerIdValidity(int layer_id) const {
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layer_kv_tensors_.size(),
                            "Layer ID %d out of range (max: %zu)",
                            layer_id,
                            layer_kv_tensors_.size());
}

}  // namespace rtp_llm