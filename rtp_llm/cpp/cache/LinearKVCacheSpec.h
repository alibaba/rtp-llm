#pragma once

#include <memory>
#include <sstream>
#include <string>

#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"

namespace rtp_llm {

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

    // Static helper function for Linear attention - no head partitioning
    static KVPartitionBytes splitKVPartitionBytes(size_t      full_block_bytes,
                                                  size_t      k_block_bytes,
                                                  size_t      v_block_bytes,
                                                  int         heads,
                                                  int         partition_count,
                                                  int         partition_id,
                                                  const char* debug_name) {
        // Validate basic parameters
        RTP_LLM_CHECK_WITH_INFO(partition_count > 0, "partition_count must be > 0");
        RTP_LLM_CHECK_WITH_INFO(partition_id >= 0 && partition_id < partition_count,
                                "partition_id out of range: %d / %d",
                                partition_id,
                                partition_count);
        RTP_LLM_CHECK_WITH_INFO(heads > 0, "heads must be > 0, got=%d (%s)", heads, debug_name);
        RTP_LLM_CHECK_WITH_INFO(k_block_bytes + v_block_bytes == full_block_bytes,
                                "block bytes mismatch (%s): full=%zu k_partition=%zu v_partition=%zu",
                                debug_name,
                                full_block_bytes,
                                k_block_bytes,
                                v_block_bytes);

        // For Linear attention implementation, return the full blocks without any head-based partitioning
        return {0, k_block_bytes, k_block_bytes, v_block_bytes};
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