#pragma once

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"

namespace rtp_llm {

struct LinearKVCacheSpec: public KVCacheSpec {
    LinearKVCacheSpec() {
        type = KVCacheSpecType::LinearAttention;
    }

    static KVCacheSpecPtr build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        RTP_LLM_CHECK_WITH_INFO(
            ctx.linear_attention_config != nullptr,
            "KVCacheSpecDesc tag=%s cache_type=%d requires SpecBuildContext.linear_attention_config",
            desc.tag.c_str(),
            static_cast<int>(desc.cache_type));
        RTP_LLM_CHECK_WITH_INFO(ctx.parallelism_config != nullptr,
                                "KVCacheSpecDesc tag=%s cache_type=%d requires SpecBuildContext.parallelism_config",
                                desc.tag.c_str(),
                                static_cast<int>(desc.cache_type));

        const auto& linear = *ctx.linear_attention_config;
        RTP_LLM_CHECK_WITH_INFO(linear.linear_key_head_dim > 0 && linear.linear_value_head_dim > 0,
                                "LINEAR KVCacheSpecDesc tag=%s requires positive linear head dims",
                                desc.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(linear.linear_conv_kernel_dim > 1,
                                "LINEAR KVCacheSpecDesc tag=%s requires linear_conv_kernel_dim > 1, got %d",
                                desc.tag.c_str(),
                                linear.linear_conv_kernel_dim);
        RTP_LLM_CHECK_WITH_INFO(linear.linear_num_key_heads > 0 && linear.linear_num_value_heads > 0,
                                "LINEAR KVCacheSpecDesc tag=%s requires positive linear head counts",
                                desc.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(linear.linear_key_head_dim == linear.linear_value_head_dim,
                                "LINEAR KVCacheSpecDesc tag=%s requires matching head dims: k=%d v=%d",
                                desc.tag.c_str(),
                                linear.linear_key_head_dim,
                                linear.linear_value_head_dim);

        auto spec                  = std::make_shared<LinearKVCacheSpec>();
        spec->tag                  = desc.tag;
        spec->seq_size_per_block   = ctx.seq_size_per_block == 0 ? 1 : ctx.seq_size_per_block;
        spec->memory_layout_dtype_ = desc.dtype != DataType::TYPE_INVALID ? desc.dtype : ctx.dtype;
        RTP_LLM_CHECK_WITH_INFO(spec->memory_layout_dtype_ != DataType::TYPE_INVALID,
                                "KVCacheSpecDesc tag=%s cache_type=%d requires valid dtype",
                                desc.tag.c_str(),
                                static_cast<int>(desc.cache_type));

        const auto     attn_tp     = std::max<int64_t>(1, ctx.parallelism_config->get_attn_tp_size());
        const uint32_t tp          = static_cast<uint32_t>(attn_tp);
        const uint32_t key_heads   = static_cast<uint32_t>(linear.linear_num_key_heads);
        const uint32_t value_heads = static_cast<uint32_t>(linear.linear_num_value_heads);
        RTP_LLM_CHECK_WITH_INFO(key_heads % tp == 0 && value_heads % tp == 0,
                                "LINEAR KVCacheSpecDesc tag=%s requires key/value heads divisible by attention TP: "
                                "key=%u value=%u tp=%u",
                                desc.tag.c_str(),
                                key_heads,
                                value_heads,
                                tp);
        RTP_LLM_CHECK_WITH_INFO(value_heads >= key_heads && value_heads % key_heads == 0,
                                "LINEAR KVCacheSpecDesc tag=%s requires value heads to be a multiple of key heads: "
                                "key=%u value=%u tp=%u",
                                desc.tag.c_str(),
                                key_heads,
                                value_heads,
                                tp);

        const uint32_t local_k_heads = key_heads / tp;
        const uint32_t local_v_heads = value_heads / tp;
        RTP_LLM_CHECK_WITH_INFO(local_k_heads > 0,
                                "LINEAR KVCacheSpecDesc tag=%s has invalid local key heads: global=%d tp=%u",
                                desc.tag.c_str(),
                                linear.linear_num_key_heads,
                                tp);
        RTP_LLM_CHECK_WITH_INFO(local_v_heads > 0,
                                "LINEAR KVCacheSpecDesc tag=%s has invalid local value heads: global=%d tp=%u",
                                desc.tag.c_str(),
                                linear.linear_num_value_heads,
                                tp);

        spec->ssm_elems =
            static_cast<size_t>(local_v_heads) * linear.linear_key_head_dim * linear.linear_value_head_dim;
        const size_t qkv = static_cast<size_t>(linear.linear_key_head_dim) * local_k_heads * 2
                           + static_cast<size_t>(linear.linear_value_head_dim) * local_v_heads;
        spec->conv_elems       = static_cast<size_t>(linear.linear_conv_kernel_dim - 1) * qkv;
        spec->ssm_state_dtype  = linear.ssm_state_dtype;
        spec->conv_state_dtype = linear.conv_state_dtype;
        RTP_LLM_CHECK_WITH_INFO(spec->ssm_state_dtype != DataType::TYPE_INVALID,
                                "LINEAR KVCacheSpecDesc tag=%s requires valid ssm_state_dtype",
                                desc.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(spec->conv_state_dtype != DataType::TYPE_INVALID,
                                "LINEAR KVCacheSpecDesc tag=%s requires valid conv_state_dtype",
                                desc.tag.c_str());
        return spec;
    }

    size_t block_size() const override {
        return k_block_size() + v_block_size();
    }

    size_t k_block_size() const override {
        return ssm_elems;
    }

    size_t v_block_size() const override {
        return conv_elems;
    }

    size_t block_size_bytes() const override {
        return k_block_size_bytes() + v_block_size_bytes();
    }

    size_t k_block_size_bytes() const override {
        return ssm_elems * getTypeSize(ssm_state_dtype);
    }

    size_t v_block_size_bytes() const override {
        return conv_elems * getTypeSize(conv_state_dtype);
    }

    rtp_llm::DataType memoryLayoutDType() const override {
        return memory_layout_dtype_;
    }

    KVCacheSpecPtr clone() const override {
        return std::make_shared<LinearKVCacheSpec>(*this);
    }

    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "LinearKVCacheSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }

private:
    DataType memory_layout_dtype_ = DataType::TYPE_INVALID;

    size_t ssm_elems  = 0;
    size_t conv_elems = 0;

    DataType ssm_state_dtype  = DataType::TYPE_INVALID;
    DataType conv_state_dtype = DataType::TYPE_INVALID;
};

}  // namespace rtp_llm
