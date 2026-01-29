#pragma once

#include <memory>
#include <sstream>
#include <string>

#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"

namespace rtp_llm {

enum KVCacheSpecType {
    MultiHeadAttention,
    MultiHeadLatentAttention,
    LinearAttention,
};

inline const char* KVCacheSpecTypeToString(KVCacheSpecType t) {
    switch (t) {
        case KVCacheSpecType::MultiHeadAttention:
            return "MultiHeadAttention";
        case KVCacheSpecType::MultiHeadLatentAttention:
            return "MultiHeadLatentAttention";
        case KVCacheSpecType::LinearAttention:
            return "LinearAttention";
        default:
            return "Unknown";
    }
}

struct KVCacheSpec {
    uint32_t layer_num;
    uint32_t local_head_num_kv;
    uint32_t seq_size_per_block = 1;

    KVCacheSpecType   type;
    rtp_llm::DataType dtype;

    virtual size_t block_size() const   = 0;
    virtual size_t k_block_size() const = 0;
    virtual size_t v_block_size() const = 0;
    virtual size_t k_dim() const        = 0;
    virtual size_t v_dim() const        = 0;

    virtual size_t block_size_bytes() const   = 0;
    virtual size_t k_block_size_bytes() const = 0;
    virtual size_t v_block_size_bytes() const = 0;

    // Scale-related methods
    virtual size_t scale_size_per_block() const {
        return 0;
    }  // Default: no scale
    virtual size_t scale_size_bytes_per_block() const {
        return scale_size_per_block() * sizeof(float);
    }
    virtual size_t scale_stride_bytes() const {
        return scale_size_bytes_per_block();
    }

    virtual std::string debugString(size_t indent = 0) const = 0;

protected:
    // Helper method to generate common parts of debug string
    std::string commonDebugString(size_t indent = 0) const {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";

        std::ostringstream os;
        os << indent1 << "type=" << KVCacheSpecTypeToString(type) << "(" << static_cast<int>(type) << ")\n";
        os << indent1 << "dtype=" << static_cast<int>(dtype) << "\n";
        os << indent1 << "layer_num=" << layer_num << "\n";
        os << indent1 << "local_head_num_kv=" << local_head_num_kv << "\n";
        os << indent1 << "seq_size_per_block=" << seq_size_per_block << "\n";
        os << indent1 << "block_size=" << block_size() << "\n";
        os << indent1 << "k_block_size=" << k_block_size() << "\n";
        os << indent1 << "v_block_size=" << v_block_size() << "\n";
        os << indent1 << "k_dim=" << k_dim() << "\n";
        os << indent1 << "v_dim=" << v_dim() << "\n";
        os << indent1 << "block_size_bytes=" << block_size_bytes() << "\n";
        os << indent1 << "k_block_size_bytes=" << k_block_size_bytes() << "\n";
        os << indent1 << "v_block_size_bytes=" << v_block_size_bytes() << "\n";
        return os.str();
    }
};

typedef std::shared_ptr<KVCacheSpec> KVCacheSpecPtr;

struct MHAKVCacheSpec: public KVCacheSpec {
    uint32_t size_per_head;

    MHAKVCacheSpec() = default;

    MHAKVCacheSpec(const AttentionConfigs& attn_config, const ParallelismConfig& parallelism_config) {
        type               = KVCacheSpecType::MultiHeadAttention;
        layer_num          = 1;  // Will be set by caller
        local_head_num_kv  = static_cast<uint32_t>(std::max(
            1,
            (attn_config.kv_head_num > 1) ? static_cast<int>(attn_config.kv_head_num / parallelism_config.tp_size) :
                                             static_cast<int>(attn_config.kv_head_num)));
        seq_size_per_block = static_cast<uint32_t>(attn_config.tokens_per_block);
        size_per_head      = static_cast<uint32_t>(attn_config.size_per_head);
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

    // Scale-related methods for MHA (only MHA supports scales for now)
    size_t scale_size_per_block() const override {
        // For INT8 or FP8, we need scales for both K and V
        if (dtype == rtp_llm::TYPE_INT8 || dtype == rtp_llm::TYPE_FP8_E4M3) {
            return 2 * local_head_num_kv * seq_size_per_block;  // K and V scales
        }
        return 0;  // No scales for other data types
    }

    size_t scale_size_bytes_per_block() const override {
        return scale_size_per_block() * sizeof(float);
    }

    size_t scale_stride_bytes() const override {
        return scale_size_bytes_per_block();
    }

    size_t k_dim() const override {
        return size_per_head;
    }
    size_t v_dim() const override {
        return size_per_head;
    }

    std::string debugString(size_t indent = 0) const override {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";

        std::ostringstream os;
        os << commonDebugString(indent);
        os << indent1 << "size_per_head=" << size_per_head << "\n";
        os << indent1 << "scale_size_per_block=" << scale_size_per_block() << "\n";
        os << indent1 << "scale_size_bytes_per_block=" << scale_size_bytes_per_block() << "\n";
        os << indent1 << "scale_stride_bytes=" << scale_stride_bytes() << "\n";
        return os.str();
    }
};

struct MLAKVCacheSpec: public KVCacheSpec {
    uint32_t kv_lora_rank;
    uint32_t rope_head_dim;

    MLAKVCacheSpec() = default;

    MLAKVCacheSpec(const AttentionConfigs& attn_config, const ParallelismConfig& parallelism_config) {
        type               = KVCacheSpecType::MultiHeadLatentAttention;
        layer_num          = 1;  // Will be set by caller
        local_head_num_kv  = 1;  // mla set local_head_num_kv to 1
        seq_size_per_block = static_cast<uint32_t>(attn_config.tokens_per_block);
        kv_lora_rank       = static_cast<uint32_t>(attn_config.kv_lora_rank);
        rope_head_dim      = static_cast<uint32_t>(attn_config.rope_head_dim);
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

    std::string debugString(size_t indent = 0) const override {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";

        std::ostringstream os;
        os << commonDebugString(indent);
        os << indent1 << "kv_lora_rank=" << kv_lora_rank << "\n";
        os << indent1 << "rope_head_dim=" << rope_head_dim << "\n";
        return os.str();
    }
};

struct LinearKVCacheSpec: public KVCacheSpec {
    // Linear attention has explicit "head" concept as well (Qwen3Next):
    // - local_num_k_heads / local_num_v_heads are sharded by TP.
    // - head_k_dim / head_v_dim are per-head dims (currently equal in Python impl).
    // - conv_kernel_dim controls conv_state_size = (kernel_dim - 1) * qkv_size.
    uint32_t local_num_k_heads = 0;
    uint32_t local_num_v_heads = 0;
    uint32_t head_k_dim        = 0;
    uint32_t head_v_dim        = 0;
    uint32_t conv_kernel_dim   = 0;

    LinearKVCacheSpec() = default;

    LinearKVCacheSpec(const AttentionConfigs&      attn_config,
                      const ParallelismConfig&     parallelism_config,
                      const LinearAttentionConfig& linear_config) {
        // Validation checks that were in HybridConfigCreator
        RTP_LLM_CHECK_WITH_INFO(linear_config.linear_key_head_dim > 0 && linear_config.linear_value_head_dim > 0,
                                "invalid linear head dim");
        RTP_LLM_CHECK_WITH_INFO(linear_config.linear_conv_kernel_dim > 1,
                                "invalid linear_conv_kernel_dim=%d",
                                linear_config.linear_conv_kernel_dim);
        RTP_LLM_CHECK_WITH_INFO(linear_config.linear_num_key_heads > 0 && linear_config.linear_num_value_heads > 0,
                                "invalid linear heads");

        const int tp      = std::max(1, static_cast<int>(parallelism_config.tp_size));
        local_num_k_heads = static_cast<uint32_t>(linear_config.linear_num_key_heads / tp);
        local_num_v_heads = static_cast<uint32_t>(linear_config.linear_num_value_heads / tp);
        RTP_LLM_CHECK_WITH_INFO(local_num_k_heads > 0 && local_num_v_heads > 0,
                                "invalid local heads for linear attention: k=%d v=%d tp=%d",
                                local_num_k_heads,
                                local_num_v_heads,
                                tp);
        RTP_LLM_CHECK_WITH_INFO(linear_config.linear_key_head_dim == linear_config.linear_value_head_dim,
                                "linear head dims must match (current impl): k=%d v=%d",
                                linear_config.linear_key_head_dim,
                                linear_config.linear_value_head_dim);

        type              = KVCacheSpecType::LinearAttention;
        layer_num         = 1;  // Will be set by caller
        local_head_num_kv = static_cast<uint32_t>(
            std::max(1,
                     (linear_config.linear_num_value_heads > 1) ?
                         static_cast<int>(linear_config.linear_num_value_heads / parallelism_config.tp_size) :
                         static_cast<int>(linear_config.linear_num_value_heads)));
        seq_size_per_block = static_cast<uint32_t>(attn_config.tokens_per_block);
        head_k_dim         = static_cast<uint32_t>(linear_config.linear_key_head_dim);
        head_v_dim         = static_cast<uint32_t>(linear_config.linear_value_head_dim);
        conv_kernel_dim    = static_cast<uint32_t>(linear_config.linear_conv_kernel_dim);
    }

    size_t ssm_state_size() const {
        // Python: ssm_state_size = local_num_v_heads * head_k_dim * head_v_dim
        return static_cast<size_t>(local_num_v_heads) * static_cast<size_t>(head_k_dim)
               * static_cast<size_t>(head_v_dim);
    }

    size_t qkv_size() const {
        // Python: qkv_size = head_k_dim * local_num_k_heads * 2 + head_v_dim * local_num_v_heads
        return static_cast<size_t>(head_k_dim) * static_cast<size_t>(local_num_k_heads) * 2
               + static_cast<size_t>(head_v_dim) * static_cast<size_t>(local_num_v_heads);
    }

    size_t conv_state_size() const {
        // Python: conv_state_size = (kernel_dim - 1) * qkv_size
        const size_t kernel = static_cast<size_t>(conv_kernel_dim);
        if (kernel <= 1) {
            return 0;
        }
        return (kernel - 1) * qkv_size();
    }

    size_t block_size() const override {
        return ssm_state_size() + conv_state_size();
    }
    size_t k_block_size() const override {
        // Keep the same physical order as Python Qwen3Next: [ssm_state][conv_state].
        return ssm_state_size();
    }
    size_t v_block_size() const override {
        return conv_state_size();
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
        return head_k_dim;
    }
    size_t v_dim() const override {
        return head_v_dim;
    }

    std::string debugString(size_t indent = 0) const override {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";

        std::ostringstream os;
        os << commonDebugString(indent);
        os << indent1 << "local_num_k_heads=" << local_num_k_heads << "\n";
        os << indent1 << "local_num_v_heads=" << local_num_v_heads << "\n";
        os << indent1 << "head_k_dim=" << head_k_dim << "\n";
        os << indent1 << "head_v_dim=" << head_v_dim << "\n";
        os << indent1 << "conv_kernel_dim=" << conv_kernel_dim << "\n";
        os << indent1 << "ssm_state_size=" << ssm_state_size() << "\n";
        os << indent1 << "qkv_size=" << qkv_size() << "\n";
        os << indent1 << "conv_state_size=" << conv_state_size() << "\n";
        return os.str();
    }
};

}  // namespace rtp_llm