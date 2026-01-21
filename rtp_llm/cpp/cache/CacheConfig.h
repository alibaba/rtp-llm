#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

enum KVCacheType {
    MultiHeadAttention,
    MultiHeadLatentAttention,
    LinearAttention,
};

enum MemoryLayout {
    LAYER_FIRST,  // [layer_num, num_blocks, block_size]
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

    bool hasScale() const {
        return enable_kv_scale && kv_scale_pool_size_bytes > 0;
    }
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
        auto is_fp8      = (dtype == DataType::TYPE_FP8_E4M3 || dtype == DataType::TYPE_FP8_E8M0);
        auto single_size = local_head_num_kv * (kv_lora_rank + rope_head_dim);
        if (is_fp8) {
            // First 512 bytes: The "quantized NoPE" part, containing 512 float8_e4m3 values.
            // Next 16 bytes: Scale factors, containing 4 float32 values. The first float32 is the scale for the first
            // 128 float8_e4m3 values, the second for the next 128, and so on. Last 128 bytes: The "RoPE" part,
            // containing 64 bfloat16 values. This part is not quantized for accuracy.
            single_size = local_head_num_kv * (kv_lora_rank + kv_lora_rank / 128 * 4 + rope_head_dim * 2);
        }
        return single_size * seq_size_per_block;
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
        return block_size() / seq_size_per_block;
    }
    size_t v_token_size() const override {
        return 0;
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
    bool use_mla   = false;
    bool is_sparse = false;

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
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";
        const std::string indent2    = indent_str + "    ";

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

        std::ostringstream os;
        os << indent_str << "CacheConfig{\n";
        os << indent1 << "dtype=" << static_cast<int>(dtype) << "\n";
        os << indent1 << "layer_num=" << layer_num << "\n";
        os << indent1 << "layer_all_num=" << layer_all_num << "\n";
        os << indent1 << "block_num=" << block_num << "\n";
        os << indent1 << "seq_size_per_block=" << seq_size_per_block << "\n";
        os << indent1 << "use_mla=" << (use_mla ? "true" : "false") << "\n";

        os << indent1 << "kv_block_size=" << kv_block_size << "\n";
        os << indent1 << "kv_block_size_bytes=" << kv_block_size_bytes << "\n";
        os << indent1 << "kv_scale_size=" << kv_scale_size << "\n";
        os << indent1 << "kv_scale_size_bytes=" << kv_scale_size_bytes << "\n";
        os << indent1 << "block_size=" << block_size << "\n";
        os << indent1 << "block_size_bytes=" << block_size_bytes << "\n";

        os << indent1 << "kv_block_stride=" << kv_block_stride << "\n";
        os << indent1 << "kv_block_stride_bytes=" << kv_block_stride_bytes << "\n";
        os << indent1 << "kv_scale_stride=" << kv_scale_stride << "\n";
        os << indent1 << "kv_scale_stride_bytes=" << kv_scale_stride_bytes << "\n";
        os << indent1 << "block_stride=" << block_stride << "\n";
        os << indent1 << "block_stride_bytes=" << block_stride_bytes << "\n";

        os << indent1 << "cache_specs.size=" << cache_specs.size() << "\n";
        for (size_t i = 0; i < cache_specs.size(); ++i) {
            const auto& spec = cache_specs[i];
            if (!spec) {
                os << indent1 << "cache_specs[" << i << "]=null\n";
                continue;
            }

            os << indent1 << "cache_specs[" << i << "] {\n";
            os << indent2 << "type=" << kvCacheTypeToString(spec->type) << "(" << static_cast<int>(spec->type) << ")\n";
            os << indent2 << "dtype=" << static_cast<int>(spec->dtype) << "\n";
            os << indent2 << "layer_num=" << spec->layer_num << "\n";
            os << indent2 << "local_head_num_kv=" << spec->local_head_num_kv << "\n";
            os << indent2 << "seq_size_per_block=" << spec->seq_size_per_block << "\n";
            os << indent2 << "block_size=" << spec->block_size() << "\n";
            os << indent2 << "k_block_size=" << spec->k_block_size() << "\n";
            os << indent2 << "v_block_size=" << spec->v_block_size() << "\n";
            os << indent2 << "k_token_size=" << spec->k_token_size() << "\n";
            os << indent2 << "v_token_size=" << spec->v_token_size() << "\n";
            os << indent2 << "block_size_bytes=" << spec->block_size_bytes() << "\n";
            os << indent2 << "k_block_size_bytes=" << spec->k_block_size_bytes() << "\n";
            os << indent2 << "v_block_size_bytes=" << spec->v_block_size_bytes() << "\n";

            if (spec->type == KVCacheType::MultiHeadAttention) {
                if (auto mha = std::dynamic_pointer_cast<MHAKVCacheSpec>(spec); mha) {
                    os << indent2 << "size_per_head=" << mha->size_per_head << "\n";
                }
            } else if (spec->type == KVCacheType::MultiHeadLatentAttention) {
                if (auto mla = std::dynamic_pointer_cast<MLAKVCacheSpec>(spec); mla) {
                    os << indent2 << "kv_lora_rank=" << mla->kv_lora_rank << "\n";
                    os << indent2 << "rope_head_dim=" << mla->rope_head_dim << "\n";
                }
            } else if (spec->type == KVCacheType::LinearAttention) {
                if (auto linear = std::dynamic_pointer_cast<LinearKVCacheSpec>(spec); linear) {
                    os << indent2 << "conv_state_size=" << linear->conv_state_size << "\n";
                    os << indent2 << "temporal_state_size=" << linear->temporal_state_size << "\n";
                }
            }
            os << indent1 << "}\n";
        }

        os << indent1 << "global_layer_ids.size=" << global_layer_ids.size() << "\n";
        os << indent1 << "global_layer_ids=" << rtp_llm::vectorsToString(global_layer_ids) << "\n";
        os << indent1 << "layer_ids.size=" << layer_ids.size() << "\n";
        os << indent1 << "layer_ids=" << rtp_llm::vectorsToString(layer_ids) << "\n";

        os << indent1 << "mtp_sub_configs.size=" << mtp_sub_configs.size() << "\n";
        for (size_t i = 0; i < mtp_sub_configs.size(); ++i) {
            const auto& sub = mtp_sub_configs[i];
            if (!sub) {
                os << indent1 << "mtp_sub_configs[" << i << "]=null\n";
                continue;
            }
            os << indent1 << "mtp_sub_configs[" << i << "]:\n";
            os << sub->debugString(indent + 4);
        }

        os << indent_str << "}\n";
        return os.str();
    }
};

struct BlockPoolConfig {
    // all memory layouts share the same block id space
    uint32_t block_num = 0;

    size_t total_size_bytes = 0;

    std::vector<MemoryLayoutConfig> memory_layouts;
};

}  // namespace rtp_llm
