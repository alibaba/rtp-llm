#pragma once

#include <memory>
#include <sstream>
#include <string>

#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"

namespace rtp_llm {

struct MLAKVCacheSpec: public KVCacheSpec {
    MLAKVCacheSpec() {
        type = KVCacheSpecType::MultiHeadLatentAttention;
    }

    static KVCacheSpecPtr build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        RTP_LLM_CHECK_WITH_INFO(ctx.attn_config != nullptr,
                                "KVCacheSpecDesc tag=%s cache_type=%d requires SpecBuildContext.attn_config",
                                desc.tag.c_str(),
                                static_cast<int>(desc.cache_type));

        const auto& attn = *ctx.attn_config;
        RTP_LLM_CHECK_WITH_INFO(attn.kv_lora_rank > 0,
                                "MLA KVCacheSpecDesc tag=%s requires positive attn_config.kv_lora_rank",
                                desc.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(attn.rope_head_dim > 0,
                                "MLA KVCacheSpecDesc tag=%s requires positive attn_config.rope_head_dim",
                                desc.tag.c_str());

        auto spec                = std::make_shared<MLAKVCacheSpec>();
        spec->tag                = desc.tag;
        spec->seq_size_per_block = ctx.seq_size_per_block == 0 ? 1 : ctx.seq_size_per_block;
        spec->dtype_             = desc.dtype != DataType::TYPE_INVALID ? desc.dtype : ctx.dtype;
        RTP_LLM_CHECK_WITH_INFO(spec->dtype_ != DataType::TYPE_INVALID,
                                "KVCacheSpecDesc tag=%s cache_type=%d requires valid dtype",
                                desc.tag.c_str(),
                                static_cast<int>(desc.cache_type));

        const bool   is_fp8     = spec->dtype_ == DataType::TYPE_FP8_E4M3 || spec->dtype_ == DataType::TYPE_FP8_E8M0;
        const size_t no_pe      = static_cast<size_t>(attn.kv_lora_rank);
        const size_t rope       = static_cast<size_t>(attn.rope_head_dim);
        spec->nope_per_token = no_pe;
        spec->rope_per_token = rope;
        spec->elems_per_token = is_fp8 ? no_pe + no_pe / 128 * 4 + rope * 2 : no_pe + rope;

        return spec;
    }

    size_t block_size() const override {
        return elems_per_token * seq_size_per_block;
    }

    size_t k_block_size() const override {
        return nope_per_token * seq_size_per_block;
    }

    size_t v_block_size() const override {
        return rope_per_token * seq_size_per_block;
    }

    size_t block_size_bytes() const override {
        return block_size() * getTypeSize(dtype_);
    }

    size_t k_block_size_bytes() const override {
        return k_block_size() * getTypeSize(dtype_);
    }

    size_t v_block_size_bytes() const override {
        return v_block_size() * getTypeSize(dtype_);
    }

    rtp_llm::DataType memoryLayoutDType() const override {
        return dtype_;
    }

    KVCacheSpecPtr clone() const override {
        return std::make_shared<MLAKVCacheSpec>(*this);
    }

    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "MLAKVCacheSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }

private:
    DataType dtype_ = DataType::TYPE_INVALID;

    size_t elems_per_token = 0;
    size_t nope_per_token  = 0;
    size_t rope_per_token  = 0;
};

}  // namespace rtp_llm