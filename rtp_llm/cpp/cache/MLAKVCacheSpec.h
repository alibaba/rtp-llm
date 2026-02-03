#pragma once

#include <memory>
#include <sstream>
#include <string>

#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"

namespace rtp_llm {

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

    // Static helper function for MLA - no head partitioning
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

        // For MLA implementation, return the full blocks without any head-based partitioning
        return {0, k_block_bytes, k_block_bytes, v_block_bytes};
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

}  // namespace rtp_llm