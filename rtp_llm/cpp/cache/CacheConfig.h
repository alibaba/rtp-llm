#pragma once

#include "rtp_llm/cpp/core/Types.h"
#include <sstream>
#include <string>

namespace rtp_llm {

struct KVCacheParam {
    uint              layer_num;
    uint              block_nums;
    uint              local_head_num_kv;
    uint              size_per_head;
    uint              seq_size_per_block = 1;
    rtp_llm::DataType dtype;
};

struct MlaCacheParam {
    uint              layer_num;
    uint              block_nums;
    uint              kv_lora_rank;
    uint              rope_head_dim;
    uint              seq_size_per_block = 1;
    rtp_llm::DataType dtype;
};

struct CacheConfig {
    uint32_t          layer_num          = 0;
    uint32_t          block_nums         = 0;
    uint32_t          local_head_num_kv  = 0;
    uint32_t          size_per_head      = 0;
    uint32_t          seq_size_per_block = 1;
    rtp_llm::DataType dtype              = rtp_llm::TYPE_INVALID;

    size_t block_size   = 0;
    size_t k_block_size = 0;
    size_t v_block_size = 0;

    size_t kv_block_size         = 0;
    size_t kv_scale_block_size   = 0;
    size_t k_block_stride        = 0;
    size_t v_block_stride        = 0;
    size_t kv_scale_block_stride = 0;
    size_t total_size            = 0;
    size_t k_total_size          = 0;
    size_t v_total_size          = 0;
    size_t scale_size            = 0;

    bool        use_mla        = false;
    uint32_t    kv_lora_rank   = 0;
    uint32_t    rope_head_dim  = 0;
    std::string mtp_model_type = "default_model";

    CacheConfig() {}

    CacheConfig(const KVCacheParam& param):
        layer_num(param.layer_num),
        block_nums(param.block_nums),
        local_head_num_kv(param.local_head_num_kv),
        size_per_head(param.size_per_head),
        seq_size_per_block(param.seq_size_per_block),
        dtype(param.dtype) {

        auto dtype_size = rtp_llm::getTypeSize(dtype);
        if (dtype == rtp_llm::TYPE_INT8 || dtype == rtp_llm::TYPE_FP8_E4M3) {
            scale_size = 4;
        }

        block_size = layer_num * local_head_num_kv * (size_per_head + scale_size) * seq_size_per_block * dtype_size * 2;
        kv_block_size       = layer_num * local_head_num_kv * size_per_head * seq_size_per_block * dtype_size * 2;
        kv_scale_block_size = layer_num * local_head_num_kv * scale_size * seq_size_per_block * dtype_size;

        // k_block_stride/v_block_stride is the size of a single block in a single layer
        k_block_stride        = kv_block_size / layer_num;
        v_block_stride        = 0;
        kv_scale_block_stride = kv_scale_block_size / layer_num;

        k_block_size = kv_block_size;
        v_block_size = 0;

        refresh();
    }

    CacheConfig(const MlaCacheParam& param):
        CacheConfig(KVCacheParam{
            param.layer_num, param.block_nums, 1, param.kv_lora_rank, param.seq_size_per_block, param.dtype}) {
        use_mla       = true;
        kv_lora_rank  = param.kv_lora_rank;
        rope_head_dim = param.rope_head_dim;

        auto dtype_size = rtp_llm::getTypeSize(dtype);
        block_size = layer_num * local_head_num_kv * (kv_lora_rank + rope_head_dim) * seq_size_per_block * dtype_size;

        k_block_stride =
            local_head_num_kv * (kv_lora_rank + rope_head_dim + scale_size) * seq_size_per_block * dtype_size;
        v_block_stride = 0;

        k_block_size = layer_num * k_block_stride;
        v_block_size = layer_num * v_block_stride;

        refresh();
    }

    void refresh() {
        total_size   = block_size * block_nums;
        k_total_size = k_block_size * block_nums;
        v_total_size = v_block_size * block_nums;
    }

    virtual size_t getKeyBlockStride() const {
        return k_block_stride;
    }

    virtual size_t getValueBlockStride() const {
        return v_block_stride;
    }

    virtual size_t getKVScaleBlockStride() const {
        return kv_scale_block_stride;
    }

    size_t getKeyLayerStride() const {
        return block_nums * getKeyBlockStride();
    }

    size_t getValueLayerStride() const {
        return block_nums * getValueBlockStride();
    }

    size_t getKVScaleLayerStride() const {
        return block_nums * getKVScaleBlockStride();
    }

    size_t getKeyOffset(int block_index, int layer_id) const {
        auto const block_stride = getKeyBlockStride();
        auto const layer_stride = getKeyLayerStride();
        return layer_id * layer_stride + block_index * block_stride;
    }

    size_t getValueOffset(int block_index, int layer_id) const {
        auto const block_stride = getValueBlockStride();
        auto const layer_stride = getValueLayerStride();
        return layer_id * layer_stride + block_index * block_stride;
    }

    size_t getKVScaleOffset(int block_index, int layer_id) const {
        auto const block_stride = getKVScaleBlockStride();
        auto const layer_stride = getKVScaleLayerStride();
        return layer_id * layer_stride + block_index * block_stride;
    }

    size_t getKeyShape() const {
        return getKeyBlockStride() / rtp_llm::getTypeSize(dtype);
    }

    size_t getValueShape() const {
        return getValueBlockStride() / rtp_llm::getTypeSize(dtype);
    }

    size_t getKVScaleShape() const {
        return getKVScaleBlockStride() / rtp_llm::getTypeSize(dtype);
    }

    size_t getKBlockSize() const {
        if (use_mla) {
            return rtp_llm::getTypeSize(dtype) * (size_t)layer_num * (size_t)block_nums * (size_t)seq_size_per_block
                   * (size_t)kv_lora_rank;
        } else {
            return rtp_llm::getTypeSize(dtype) * (size_t)layer_num * (size_t)block_nums * (size_t)local_head_num_kv
                   * (size_t)seq_size_per_block * (size_t)size_per_head;
        }
    }

    size_t getVBlockSize() const {
        if (use_mla) {
            return rtp_llm::getTypeSize(dtype) * (size_t)layer_num * (size_t)block_nums * (size_t)seq_size_per_block
                   * (size_t)rope_head_dim;
        } else {
            return rtp_llm::getTypeSize(dtype) * (size_t)layer_num * (size_t)block_nums * (size_t)local_head_num_kv
                   * (size_t)seq_size_per_block * (size_t)size_per_head;
        }
    }

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "CacheConfig { "
                     << "layer_num: " << layer_num << ", block_nums: " << block_nums << ", block_size: " << block_size
                     << ", local_head_num_kv: " << local_head_num_kv << ", size_per_head: " << size_per_head
                     << ", seq_size_per_block: " << seq_size_per_block << ", dtype: " << int(dtype)
                     << ", k_block_stride: " << k_block_stride << ", v_block_stride: " << v_block_stride
                     << ", k_block_size: " << k_block_size << ", v_block_size: " << v_block_size
                     << ", kv_scale_block_stride: " << kv_scale_block_stride << ", k_total_size: " << k_total_size
                     << ", v_total_size: " << v_total_size << ", total_size: " << total_size << "}";
        return debug_string.str();
    }
};

}  // namespace rtp_llm
