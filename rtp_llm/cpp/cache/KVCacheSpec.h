#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

enum KVCacheSpecType {
    MultiHeadAttention,
    MultiHeadLatentAttention,
    LinearAttention,
};

inline std::string kvCacheSpecTypeToString(KVCacheSpecType t) {
    switch (t) {
        case KVCacheSpecType::MultiHeadAttention:
            return "mha";
        case KVCacheSpecType::MultiHeadLatentAttention:
            return "mla";
        case KVCacheSpecType::LinearAttention:
            return "linear";
        default:
            return "unknown";
    }
}

struct KVCacheSpec {
    uint32_t local_head_num_kv;
    uint32_t seq_size_per_block = 1;
    uint32_t layer_num          = 0;

    KVCacheSpecType   type;
    rtp_llm::DataType dtype;

    // Size related members
    size_t block_size_val       = 0;
    size_t block_size_bytes_val = 0;

    // Scale related members
    bool   enable_kv_scale       = false;
    size_t k_scale_stride_bytes  = 0;
    size_t v_scale_stride_bytes  = 0;
    size_t kv_scale_stride_bytes = 0;
    size_t kv_scale_stride       = 0;

    virtual size_t block_size() const   = 0;
    virtual size_t k_block_size() const = 0;
    virtual size_t v_block_size() const = 0;
    virtual size_t k_dim() const        = 0;
    virtual size_t v_dim() const        = 0;

    virtual size_t block_size_bytes() const   = 0;
    virtual size_t k_block_size_bytes() const = 0;
    virtual size_t v_block_size_bytes() const = 0;

    virtual void initializeFromConfigs(const AttentionConfigs&  attn_config,
                                       const ParallelismConfig& parallelism_config) {
        seq_size_per_block = static_cast<uint32_t>(attn_config.tokens_per_block);

        // Calculate block size
        block_size_val       = block_size();
        block_size_bytes_val = block_size_bytes();

        enable_kv_scale = false;
    }

    virtual std::string debugString(size_t indent = 0) const {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";

        std::ostringstream os;
        os << indent_str << "KVCacheSpec{\n";
        os << indent1 << "type=" << kvCacheSpecTypeToString(type) << "(" << static_cast<int>(type) << ")\n";
        os << indent1 << "dtype=" << static_cast<int>(dtype) << "\n";

        os << indent1 << "local_head_num_kv=" << local_head_num_kv << "\n";
        os << indent1 << "seq_size_per_block=" << seq_size_per_block << "\n";
        os << indent1 << "enable_kv_scale=" << (enable_kv_scale ? "true" : "false") << "\n";
        os << indent1 << "k_scale_stride_bytes=" << k_scale_stride_bytes << "\n";
        os << indent1 << "v_scale_stride_bytes=" << v_scale_stride_bytes << "\n";
        os << indent1 << "kv_scale_stride_bytes=" << kv_scale_stride_bytes << "\n";
        os << indent1 << "kv_scale_stride=" << kv_scale_stride << "\n";
        os << indent1 << "block_size=" << block_size() << "\n";
        os << indent1 << "k_block_size=" << k_block_size() << "\n";
        os << indent1 << "v_block_size=" << v_block_size() << "\n";
        os << indent1 << "k_dim=" << k_dim() << "\n";
        os << indent1 << "v_dim=" << v_dim() << "\n";
        os << indent1 << "block_size_bytes=" << block_size_bytes() << "\n";
        os << indent1 << "k_block_size_bytes=" << k_block_size_bytes() << "\n";
        os << indent1 << "v_block_size_bytes=" << v_block_size_bytes() << "\n";

        // 输出类型特定的信息
        os << typeSpecificDebugString(indent + 2);

        os << indent_str << "}\n";
        return os.str();
    }

    virtual std::string typeSpecificDebugString(size_t indent = 0) const {
        // 默认实现返回空字符串
        return "";
    }

protected:
    // Protected helper methods to calculate scale strides
    void calculateScaleStrides() {
        if (enable_kv_scale) {
            const size_t local_head_num_kv_val    = static_cast<size_t>(this->local_head_num_kv);
            const size_t seq_size_per_block_val   = static_cast<size_t>(this->seq_size_per_block);
            const size_t kv_scale_kv_stride       = local_head_num_kv_val * seq_size_per_block_val;
            const size_t kv_scale_kv_stride_bytes = kv_scale_kv_stride * sizeof(float);

            k_scale_stride_bytes  = kv_scale_kv_stride_bytes;
            v_scale_stride_bytes  = kv_scale_kv_stride_bytes;
            kv_scale_stride_bytes = 2 * kv_scale_kv_stride_bytes;
            kv_scale_stride       = 2 * local_head_num_kv_val * seq_size_per_block_val;
        }
    }

    // Common initialization logic for block sizes
    void initializeBlockSizes() {
        block_size_val       = block_size();
        block_size_bytes_val = block_size_bytes();
    }
};

typedef std::shared_ptr<KVCacheSpec> KVCacheSpecPtr;

struct MHAKVCacheSpec: public KVCacheSpec {
    uint32_t size_per_head;

    MHAKVCacheSpec() {
        type = KVCacheSpecType::MultiHeadAttention;
    }

    MHAKVCacheSpec(const AttentionConfigs&  attn_config,
                   const ParallelismConfig& parallelism_config,
                   rtp_llm::DataType        dtype_val) {
        type  = KVCacheSpecType::MultiHeadAttention;
        dtype = dtype_val;

        // Calculate local head num
        local_head_num_kv =
            static_cast<uint32_t>(std::max(1u, (uint32_t)(attn_config.kv_head_num / parallelism_config.tp_size)));
        size_per_head = static_cast<uint32_t>(attn_config.size_per_head);

        initializeFromConfigs(attn_config, parallelism_config);
    }

    void initializeFromConfigs(const AttentionConfigs&  attn_config,
                               const ParallelismConfig& parallelism_config) override {
        seq_size_per_block = static_cast<uint32_t>(attn_config.tokens_per_block);

        // Enable scale if dtype is INT8 or FP8
        enable_kv_scale = (dtype == rtp_llm::TYPE_INT8 || dtype == rtp_llm::TYPE_FP8_E4M3);

        // Initialize block sizes using base class helper
        initializeBlockSizes();

        // Calculate scale strides if enabled
        if (enable_kv_scale) {
            calculateScaleStrides();
        }
    }

    size_t block_size() const override {
        return 2 * local_head_num_kv * size_per_head * seq_size_per_block;
    }
    size_t k_block_size() const override {
        return local_head_num_kv * size_per_head * seq_size_per_block;
    }
    size_t v_block_size() const override {
        return local_head_num_kv * size_per_head * seq_size_per_block;
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

    size_t k_dim() const override {
        return size_per_head;
    }
    size_t v_dim() const override {
        return size_per_head;
    }

    std::string typeSpecificDebugString(size_t indent = 0) const override {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";

        std::ostringstream os;
        os << indent1 << "size_per_head=" << size_per_head << "\n";
        return os.str();
    }
};

struct MLAKVCacheSpec: public KVCacheSpec {
    uint32_t kv_lora_rank;
    uint32_t rope_head_dim;

    MLAKVCacheSpec() {
        type              = KVCacheSpecType::MultiHeadLatentAttention;
        local_head_num_kv = 1;  // MLA sets local_head_num_kv to 1
    }

    MLAKVCacheSpec(const AttentionConfigs&  attn_config,
                   const ParallelismConfig& parallelism_config,
                   rtp_llm::DataType        dtype_val) {
        type              = KVCacheSpecType::MultiHeadLatentAttention;
        dtype             = dtype_val;
        local_head_num_kv = 1;  // MLA sets local_head_num_kv to 1
        kv_lora_rank      = static_cast<uint32_t>(attn_config.kv_lora_rank);
        rope_head_dim     = static_cast<uint32_t>(attn_config.rope_head_dim);

        initializeFromConfigs(attn_config, parallelism_config);
    }

    void initializeFromConfigs(const AttentionConfigs&  attn_config,
                               const ParallelismConfig& parallelism_config) override {
        seq_size_per_block = static_cast<uint32_t>(attn_config.tokens_per_block);

        // Initialize block sizes using base class helper
        initializeBlockSizes();

        enable_kv_scale = false;
    }

    size_t block_size() const override {
        return local_head_num_kv * (kv_lora_rank + rope_head_dim) * seq_size_per_block;
    }
    size_t k_block_size() const override {
        return local_head_num_kv * kv_lora_rank * seq_size_per_block;
    }
    size_t v_block_size() const override {
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

    size_t k_dim() const override {
        return kv_lora_rank;
    }
    size_t v_dim() const override {
        return rope_head_dim;
    }

    std::string typeSpecificDebugString(size_t indent = 0) const override {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";

        std::ostringstream os;
        os << indent1 << "kv_lora_rank=" << kv_lora_rank << "\n";
        os << indent1 << "rope_head_dim=" << rope_head_dim << "\n";
        return os.str();
    }
};

struct LinearKVCacheSpec: public KVCacheSpec {
    uint32_t conv_state_size     = 0;
    uint32_t temporal_state_size = 0;

    LinearKVCacheSpec() {
        type = KVCacheSpecType::LinearAttention;
    }

    LinearKVCacheSpec(const AttentionConfigs&  attn_config,
                      const ParallelismConfig& parallelism_config,
                      rtp_llm::DataType        dtype_val) {
        type  = KVCacheSpecType::LinearAttention;
        dtype = dtype_val;

        initializeFromConfigs(attn_config, parallelism_config);
    }

    void initializeFromConfigs(const AttentionConfigs&  attn_config,
                               const ParallelismConfig& parallelism_config) override {
        seq_size_per_block = static_cast<uint32_t>(attn_config.tokens_per_block);
        local_head_num_kv =
            static_cast<uint32_t>(std::max(1u, (uint32_t)(attn_config.kv_head_num / parallelism_config.tp_size)));

        // Initialize block sizes using base class helper
        initializeBlockSizes();

        enable_kv_scale = false;
    }

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

    size_t k_dim() const override {
        return conv_state_size;
    }
    size_t v_dim() const override {
        return temporal_state_size;
    }

    std::string typeSpecificDebugString(size_t indent = 0) const override {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";

        std::ostringstream os;
        os << indent1 << "conv_state_size=" << conv_state_size << "\n";
        os << indent1 << "temporal_state_size=" << temporal_state_size << "\n";
        return os.str();
    }
};

}  // namespace rtp_llm
