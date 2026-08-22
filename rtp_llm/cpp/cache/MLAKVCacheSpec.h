#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>

#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct MLAKVCacheSpec: public KVCacheSpec {
    uint32_t kv_lora_rank;
    uint32_t rope_head_dim;
    bool     cache_tp_flat_shard = false;

    MLAKVCacheSpec() = default;

    MLAKVCacheSpec(const AttentionConfigs& attn_config, const ParallelismConfig& parallelism_config) {
        type               = KVCacheSpecType::MultiHeadLatentAttention;
        layer_num          = 1;  // Will be set by caller
        local_head_num_kv  = 1;  // mla set local_head_num_kv to 1
        seq_size_per_block = static_cast<uint32_t>(attn_config.tokens_per_block);
        kv_lora_rank       = static_cast<uint32_t>(attn_config.kv_lora_rank);
        rope_head_dim      = static_cast<uint32_t>(attn_config.rope_head_dim);
        const char* cache_tp = std::getenv("KIMI_K3_MLA_CACHE_TP");
        if (cache_tp != nullptr && std::strcmp(cache_tp, "0") != 0 && std::strcmp(cache_tp, "1") != 0) {
            throw std::runtime_error("KIMI_K3_MLA_CACHE_TP must be 0 or 1");
        }
        if (cache_tp != nullptr && std::strcmp(cache_tp, "1") == 0
            && parallelism_config.role_type == RoleType::PREFILL) {
            const auto tp = static_cast<uint32_t>(parallelism_config.get_attn_tp_size());
            RTP_LLM_CHECK_WITH_INFO(kv_lora_rank == 512 && rope_head_dim == 64,
                                    "KIMI_K3_MLA_CACHE_TP requires K3 MLA 512+64, got %u+%u",
                                    kv_lora_rank,
                                    rope_head_dim);
            const auto full_width = kv_lora_rank + rope_head_dim;
            RTP_LLM_CHECK_WITH_INFO(tp > 1 && full_width % tp == 0,
                                    "KIMI_K3_MLA_CACHE_TP requires TP>1 dividing packed width 576, got TP=%u",
                                    tp);
            const auto local_width = full_width / tp;
            // The physical cache is one opaque flat shard.  Keeping all 72
            // values in the first component also guarantees the generic BF16
            // concat kernel always launches with a non-zero primary width.
            kv_lora_rank       = local_width;
            rope_head_dim      = 0;
            cache_tp_flat_shard = true;
        }
    }

    size_t block_size() const override {
        auto is_fp8      = (dtype == DataType::TYPE_FP8_E4M3 || dtype == DataType::TYPE_FP8_E8M0);
        RTP_LLM_CHECK_WITH_INFO(!(cache_tp_flat_shard && is_fp8),
                                "KIMI_K3_MLA_CACHE_TP flat-72 ABI currently supports BF16 only");
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
