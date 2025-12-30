#pragma once

#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/cache/types.h"
#include <sstream>
#include <string>
#include <memory>
#include <vector>

namespace rtp_llm {

enum KVCacheType {
    MultiHeadAttention,
    MultiHeadLatentAttention,
    LinearAttention,
};

enum MemoryLayout {
    LAYER_FIRST,  // [layer_num, num_blocks, block_size] -> hybrid attention
};

struct MemoryLayoutConfig {
    uint32_t layer_num = 0;
    uint32_t block_num = 0;

    MemoryLayout      layout = LAYER_FIRST;
    rtp_llm::DataType dtype  = rtp_llm::TYPE_INVALID;

    // ---- Offsets within BlockPool global buffer ----
    // kv cache pool base offset
    size_t kv_cache_offset_bytes = 0;
    // kv scale pool base offset (valid only when enable_kv_scale == true)
    size_t kv_scale_offset_bytes = 0;

    // ---- Pool sizes ----
    size_t kv_block_pool_size_bytes = 0;
    size_t kv_scale_pool_size_bytes = 0;
    size_t total_size_bytes         = 0;

    // ---- Per-block sizes (all layers) ----
    size_t kv_block_size       = 0;
    size_t kv_block_size_bytes = 0;
    size_t kv_scale_size       = 0;
    size_t kv_scale_size_bytes = 0;
    size_t block_size          = 0;
    size_t block_size_bytes    = 0;

    // ---- Per-block strides (one layer) ----
    size_t kv_block_stride       = 0;
    size_t kv_block_stride_bytes = 0;
    size_t kv_scale_stride       = 0;
    size_t kv_scale_stride_bytes = 0;
    size_t block_stride          = 0;
    size_t block_stride_bytes    = 0;

    // For partitioning / kernels (KV separation info)
    size_t k_block_size         = 0;
    size_t v_block_size         = 0;
    size_t k_block_stride       = 0;
    size_t v_block_stride       = 0;
    size_t k_block_size_bytes   = 0;
    size_t v_block_size_bytes   = 0;
    size_t k_block_stride_bytes = 0;
    size_t v_block_stride_bytes = 0;
    size_t k_scale_stride_bytes = 0;
    size_t v_scale_stride_bytes = 0;
    size_t k_token_size         = 0;
    size_t v_token_size         = 0;

    bool   is_mla             = false;
    size_t local_head_num_kv  = 0;
    size_t seq_size_per_block = 0;

    bool enable_kv_scale = false;
};

struct KVCacheSpec {
    uint32_t layer_num;
    uint32_t local_head_num_kv;
    uint32_t seq_size_per_block = 1;

    KVCacheType       type;
    rtp_llm::DataType dtype;

    virtual size_t block_size() const   = 0;
    virtual size_t k_block_size() const = 0;
    virtual size_t v_block_size() const = 0;
    virtual size_t k_token_size() const = 0;
    virtual size_t v_token_size() const = 0;

    virtual size_t block_size_bytes() const   = 0;
    virtual size_t k_block_size_bytes() const = 0;
    virtual size_t v_block_size_bytes() const = 0;
};

typedef std::shared_ptr<KVCacheSpec> KVCacheSpecPtr;

struct MHAKVCacheSpec: public KVCacheSpec {
    uint32_t size_per_head;

    size_t block_size() const override {
        return 2 * local_head_num_kv * size_per_head * seq_size_per_block;
    }
    size_t k_block_size() const override {
        return local_head_num_kv * size_per_head * seq_size_per_block;
    }
    size_t v_block_size() const override {
        return local_head_num_kv * size_per_head * seq_size_per_block;
    }

    size_t block_size_bytes() const {
        return block_size() * rtp_llm::getTypeSize(dtype);
    }
    size_t k_block_size_bytes() const {
        return k_block_size() * rtp_llm::getTypeSize(dtype);
    }
    size_t v_block_size_bytes() const {
        return v_block_size() * rtp_llm::getTypeSize(dtype);
    }

    size_t k_token_size() const override {
        return size_per_head;
    }
    size_t v_token_size() const override {
        return size_per_head;
    }
};

struct MLAKVCacheSpec: public KVCacheSpec {
    uint32_t kv_lora_rank;
    uint32_t rope_head_dim;

    size_t block_size() const {
        return local_head_num_kv * (kv_lora_rank + rope_head_dim) * seq_size_per_block;
    }
    size_t k_block_size() const {
        return local_head_num_kv * kv_lora_rank * seq_size_per_block;
    }
    size_t v_block_size() const {
        return local_head_num_kv * rope_head_dim * seq_size_per_block;
    }

    size_t block_size_bytes() const override {
        return block_size() * rtp_llm::getTypeSize(dtype);
    }
    size_t k_block_size_bytes() const override {
        return k_block_size() * rtp_llm::getTypeSize(dtype);
    }
    size_t v_block_size_bytes() const override {
        return v_block_size() * rtp_llm::getTypeSize(dtype);
    }

    size_t k_token_size() const override {
        return kv_lora_rank;
    }
    size_t v_token_size() const override {
        return rope_head_dim;
    }
};

struct LinearKVCacheSpec: public KVCacheSpec {
    uint32_t conv_state_size;
    uint32_t temporal_state_size;

    size_t block_size() const override {
        return (conv_state_size + temporal_state_size) * seq_size_per_block;
    }
    size_t k_block_size() const override {
        return conv_state_size * seq_size_per_block;
    }
    size_t v_block_size() const override {
        return temporal_state_size * seq_size_per_block;
    }

    size_t block_size_bytes() const override {
        return block_size() * rtp_llm::getTypeSize(dtype);
    }
    size_t k_block_size_bytes() const override {
        return k_block_size() * rtp_llm::getTypeSize(dtype);
    }
    size_t v_block_size_bytes() const override {
        return v_block_size() * rtp_llm::getTypeSize(dtype);
    }

    size_t k_token_size() const override {
        return conv_state_size;
    }
    size_t v_token_size() const override {
        return temporal_state_size;
    }
};

struct CacheConfig {
    std::vector<KVCacheSpecPtr>   cache_specs;
    std::vector<std::vector<int>> global_layer_ids;  // including mtp module layers
    std::vector<std::vector<int>> layer_ids;

    rtp_llm::DataType dtype;
    uint32_t          layer_num;      // the number of main model layers
    uint32_t          layer_all_num;  // the number of all layers including mtp modules

    uint32_t block_num;

    // ---- Per-block sizes (all layers) ----
    // kv_block_*: kv cache only
    size_t kv_block_size       = 0;
    size_t kv_block_size_bytes = 0;
    // kv_scale_*: kv cache scale only (int8/fp8) (K+V together).
    size_t kv_scale_size       = 0;
    size_t kv_scale_size_bytes = 0;
    // block_*: kv cache + scale, for one logical "block" across all layers. (K+V scales together).
    size_t block_size       = 0;
    size_t block_size_bytes = 0;

    size_t seq_size_per_block = 1;  // for cache_keys generation

    // for adpation to MLA
    bool use_mla = false;

    // mtp
    std::string mtp_model_type = "default_model";

    std::vector<std::shared_ptr<CacheConfig>> mtp_sub_configs;

    // ---- Per-block strides (one layer) ----
    // kv_block_stride_*: one-layer kv cache block stride (K+V together).
    size_t kv_block_stride       = 0;
    size_t kv_block_stride_bytes = 0;
    // kv_scale_stride_*: one-layer kv cache scale stride for one logical block (K+V scales together).
    size_t kv_scale_stride       = 0;
    size_t kv_scale_stride_bytes = 0;
    // block_stride_*: one-layer total stride (kv + scale)
    size_t block_stride       = 0;
    size_t block_stride_bytes = 0;

    CacheConfig() {}

    std::string debugString(size_t indent = 0) const {
        const auto indent_str = std::string(indent, ' ');

        auto dataTypeToString = [](rtp_llm::DataType t) -> const char* {
            switch (t) {
                case rtp_llm::TYPE_INVALID:
                    return "invalid";
                case rtp_llm::TYPE_BOOL:
                    return "bool";
                case rtp_llm::TYPE_UINT8:
                    return "uint8";
                case rtp_llm::TYPE_UINT16:
                    return "uint16";
                case rtp_llm::TYPE_UINT32:
                    return "uint32";
                case rtp_llm::TYPE_UINT64:
                    return "uint64";
                case rtp_llm::TYPE_INT8:
                    return "int8";
                case rtp_llm::TYPE_INT16:
                    return "int16";
                case rtp_llm::TYPE_INT32:
                    return "int32";
                case rtp_llm::TYPE_INT64:
                    return "int64";
                case rtp_llm::TYPE_FP16:
                    return "fp16";
                case rtp_llm::TYPE_FP32:
                    return "fp32";
                case rtp_llm::TYPE_FP64:
                    return "fp64";
                case rtp_llm::TYPE_BYTES:
                    return "bytes";
                case rtp_llm::TYPE_BF16:
                    return "bf16";
                case rtp_llm::TYPE_FP8_E4M3:
                    return "fp8_e4m3";
                case rtp_llm::TYPE_STR:
                    return "str";
                case rtp_llm::TYPE_VOID:
                    return "void";
                case rtp_llm::TYPE_QINT8:
                    return "qint8";
                case rtp_llm::TYPE_INT4X2:
                    return "int4x2";
                case rtp_llm::TYPE_QINT4X2:
                    return "qint4x2";
                case rtp_llm::TYPE_QFP8_E4M3:
                    return "qfp8_e4m3";
                default:
                    return "unknown";
            }
        };

        auto kvCacheTypeToString = [](KVCacheType t) -> const char* {
            switch (t) {
                case KVCacheType::MultiHeadAttention:
                    return "mha";
                case KVCacheType::MultiHeadLatentAttention:
                    return "mla";
                case KVCacheType::LinearAttention:
                    return "linear";
                default:
                    return "unknown";
            }
        };

        auto vecToString = [](const std::vector<int>& v) -> std::string {
            std::ostringstream os;
            os << "[";
            for (size_t i = 0; i < v.size(); ++i) {
                if (i) {
                    os << ",";
                }
                os << v[i];
            }
            os << "]";
            return os.str();
        };

        auto vecVecToString = [&](const std::vector<std::vector<int>>& vv) -> std::string {
            std::ostringstream os;
            os << "[";
            for (size_t i = 0; i < vv.size(); ++i) {
                if (i) {
                    os << ",";
                }
                os << vecToString(vv[i]);
            }
            os << "]";
            return os.str();
        };

        std::ostringstream os;
        os << indent_str << "CacheConfig{\n";
        os << indent_str << "  dtype=" << dataTypeToString(dtype) << "(" << static_cast<int>(dtype) << ")\n";
        os << indent_str << "  layer_num=" << layer_num << "\n";
        os << indent_str << "  layer_all_num=" << layer_all_num << "\n";
        os << indent_str << "  block_num=" << block_num << "\n";

        os << indent_str << "  seq_size_per_block=" << seq_size_per_block << "\n";
        os << indent_str << "  use_mla=" << (use_mla ? "true" : "false") << "\n";

        os << indent_str << "  kv_block_size=" << kv_block_size << "\n";
        os << indent_str << "  kv_block_size_bytes=" << kv_block_size_bytes << "\n";
        os << indent_str << "  kv_scale_size=" << kv_scale_size << "\n";
        os << indent_str << "  kv_scale_size_bytes=" << kv_scale_size_bytes << "\n";
        os << indent_str << "  block_size=" << block_size << "\n";
        os << indent_str << "  block_size_bytes=" << block_size_bytes << "\n";

        os << indent_str << "  kv_block_stride=" << kv_block_stride << "\n";
        os << indent_str << "  kv_block_stride_bytes=" << kv_block_stride_bytes << "\n";
        os << indent_str << "  kv_scale_stride=" << kv_scale_stride << "\n";
        os << indent_str << "  kv_scale_stride_bytes=" << kv_scale_stride_bytes << "\n";
        os << indent_str << "  block_stride=" << block_stride << "\n";
        os << indent_str << "  block_stride_bytes=" << block_stride_bytes << "\n";

        os << indent_str << "  cache_specs.size=" << cache_specs.size() << "\n";
        for (size_t i = 0; i < cache_specs.size(); ++i) {
            const auto& spec = cache_specs[i];
            if (!spec) {
                os << indent_str << "  cache_specs[" << i << "]=null\n";
                continue;
            }

            os << indent_str << "  cache_specs[" << i << "] {\n";
            os << indent_str << "    type=" << kvCacheTypeToString(spec->type) << "\n";
            os << indent_str << "    dtype=" << dataTypeToString(spec->dtype) << "(" << static_cast<int>(spec->dtype)
               << ")\n";
            os << indent_str << "    layer_num=" << spec->layer_num << "\n";
            os << indent_str << "    local_head_num_kv=" << spec->local_head_num_kv << "\n";
            os << indent_str << "    seq_size_per_block=" << spec->seq_size_per_block << "\n";
            os << indent_str << "    block_size=" << spec->block_size() << "\n";
            os << indent_str << "    k_block_size=" << spec->k_block_size() << "\n";
            os << indent_str << "    v_block_size=" << spec->v_block_size() << "\n";
            os << indent_str << "    k_token_size=" << spec->k_token_size() << "\n";
            os << indent_str << "    v_token_size=" << spec->v_token_size() << "\n";
            os << indent_str << "    block_size_bytes=" << spec->block_size_bytes() << "\n";
            os << indent_str << "    k_block_size_bytes=" << spec->k_block_size_bytes() << "\n";
            os << indent_str << "    v_block_size_bytes=" << spec->v_block_size_bytes() << "\n";

            if (spec->type == KVCacheType::MultiHeadAttention) {
                if (auto mha = std::dynamic_pointer_cast<MHAKVCacheSpec>(spec); mha) {
                    os << indent_str << "    size_per_head=" << mha->size_per_head << "\n";
                }
            } else if (spec->type == KVCacheType::MultiHeadLatentAttention) {
                if (auto mla = std::dynamic_pointer_cast<MLAKVCacheSpec>(spec); mla) {
                    os << indent_str << "    kv_lora_rank=" << mla->kv_lora_rank << "\n";
                    os << indent_str << "    rope_head_dim=" << mla->rope_head_dim << "\n";
                }
            } else if (spec->type == KVCacheType::LinearAttention) {
                if (auto linear = std::dynamic_pointer_cast<LinearKVCacheSpec>(spec); linear) {
                    os << indent_str << "    conv_state_size=" << linear->conv_state_size << "\n";
                    os << indent_str << "    temporal_state_size=" << linear->temporal_state_size << "\n";
                }
            }
            os << indent_str << "  }\n";
        }

        os << indent_str << "  global_layer_ids=" << vecVecToString(global_layer_ids) << "\n";
        os << indent_str << "  layer_ids=" << vecVecToString(layer_ids) << "\n";

        os << indent_str << "  mtp_model_type=" << mtp_model_type << "\n";
        os << indent_str << "  mtp_sub_configs.size=" << mtp_sub_configs.size() << "\n";
        for (size_t i = 0; i < mtp_sub_configs.size(); ++i) {
            const auto& sub = mtp_sub_configs[i];
            if (!sub) {
                os << indent_str << "  mtp_sub_configs[" << i << "]=null\n";
                continue;
            }
            os << indent_str << "  mtp_sub_configs[" << i << "]:\n";
            os << sub->debugString(indent + 4);
        }

        os << indent_str << "}\n";
        return os.str();
    }
};

struct BlockPoolConfig {
    // all memory layouts share the same block id space
    uint32_t block_num = 0;

    size_t total_size       = 0;
    size_t total_size_bytes = 0;

    std::vector<MemoryLayoutConfig> memory_layouts;
};

}  // namespace rtp_llm
