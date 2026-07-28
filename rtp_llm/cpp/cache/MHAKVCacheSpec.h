#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>

#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"

namespace rtp_llm {

struct MHAKVCacheSpec: public KVCacheSpec {
    MHAKVCacheSpec() {
        type = KVCacheSpecType::MultiHeadAttention;
    }

    static KVCacheSpecPtr build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        RTP_LLM_CHECK_WITH_INFO(ctx.attn_config != nullptr,
                                "KVCacheSpecDesc tag=%s cache_type=%d requires SpecBuildContext.attn_config",
                                desc.tag.c_str(),
                                static_cast<int>(desc.cache_type));
        RTP_LLM_CHECK_WITH_INFO(ctx.parallelism_config != nullptr,
                                "KVCacheSpecDesc tag=%s cache_type=%d requires SpecBuildContext.parallelism_config",
                                desc.tag.c_str(),
                                static_cast<int>(desc.cache_type));

        const auto& attn = *ctx.attn_config;
        RTP_LLM_CHECK_WITH_INFO(attn.kv_head_num > 0,
                                "MHA KVCacheSpecDesc tag=%s requires positive attn_config.kv_head_num",
                                desc.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(attn.size_per_head > 0,
                                "MHA KVCacheSpecDesc tag=%s requires positive attn_config.size_per_head",
                                desc.tag.c_str());

        auto spec                = std::make_shared<MHAKVCacheSpec>();
        spec->tag                = desc.tag;
        spec->seq_size_per_block = ctx.seq_size_per_block == 0 ? 1 : ctx.seq_size_per_block;
        spec->dtype_             = desc.dtype != DataType::TYPE_INVALID ? desc.dtype : ctx.dtype;
        RTP_LLM_CHECK_WITH_INFO(spec->dtype_ != DataType::TYPE_INVALID,
                                "KVCacheSpecDesc tag=%s cache_type=%d requires valid dtype",
                                desc.tag.c_str(),
                                static_cast<int>(desc.cache_type));

        const auto     attn_tp        = std::max<int64_t>(1, ctx.parallelism_config->get_attn_tp_size());
        const uint32_t tp             = static_cast<uint32_t>(attn_tp);
        const uint32_t kv             = static_cast<uint32_t>(attn.kv_head_num);
        const uint32_t local_kv_heads = (kv % tp == 0) ? kv / tp : kv / std::gcd(kv, tp);

        spec->per_token_k_elems       = static_cast<size_t>(local_kv_heads) * attn.size_per_head;
        if (spec->dtype_ == DataType::TYPE_INT8 || spec->dtype_ == DataType::TYPE_FP8_E4M3) {
            spec->per_token_k_scale_bytes = static_cast<size_t>(local_kv_heads) * sizeof(float);
        }
        return spec;
    }

    size_t block_size() const override {
        return k_block_size() + v_block_size();
    }

    size_t k_block_size() const override {
        return per_token_k_elems * seq_size_per_block;
    }

    size_t v_block_size() const override {
        return per_token_k_elems * seq_size_per_block;
    }

    size_t block_size_bytes() const override {
        return k_block_size_bytes() + v_block_size_bytes();
    }

    size_t k_block_size_bytes() const override {
        return k_block_size() * getTypeSize(dtype_);
    }

    size_t v_block_size_bytes() const override {
        return v_block_size() * getTypeSize(dtype_);
    }

    size_t scale_block_size_bytes() const override {
        return k_scale_block_size_bytes() + v_scale_block_size_bytes();
    }

    size_t k_scale_block_size_bytes() const override {
        return per_token_k_scale_bytes * seq_size_per_block;
    }

    size_t v_scale_block_size_bytes() const override {
        return per_token_k_scale_bytes * seq_size_per_block;
    }

    rtp_llm::DataType memoryLayoutDType() const override {
        return dtype_;
    }

    KVCacheSpecPtr clone() const override {
        return std::make_shared<MHAKVCacheSpec>(*this);
    }

    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "MHAKVCacheSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }

private:
    DataType dtype_ = DataType::TYPE_INVALID;

    size_t per_token_k_elems       = 0;
    size_t per_token_k_scale_bytes = 0;
};

}  // namespace rtp_llm