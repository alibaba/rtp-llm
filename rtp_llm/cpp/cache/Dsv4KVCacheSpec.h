#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"

namespace rtp_llm {

struct PackedEntryKVCacheSpec: public KVCacheSpec {
    PackedEntryKVCacheSpec() {
        type = KVCacheSpecType::CompressedKVCache;
    }

    size_t block_size() const override {
        return payload_elems_;
    }

    size_t k_block_size() const override {
        return payload_elems_;
    }

    size_t v_block_size() const override {
        return 0;
    }

    size_t block_size_bytes() const override {
        return block_stride_bytes_;
    }

    size_t k_block_size_bytes() const override {
        return block_stride_bytes_;
    }

    size_t v_block_size_bytes() const override {
        return 0;
    }

    size_t block_payload_bytes() const override {
        return payload_bytes_;
    }

    size_t k_block_payload_bytes() const override {
        return payload_bytes_;
    }

    size_t v_block_payload_bytes() const override {
        return 0;
    }

    rtp_llm::DataType memoryLayoutDType() const override {
        return entry_dtype_;
    }

    // These values describe the complete CP block. A PREFILL spec may expose a
    // rank-local entry count/stride while these accessors intentionally remain
    // in full-block coordinates.
    size_t fullBlockEntryPaddingCount() const {
        return entry_padding_count_;
    }

    size_t fullBlockStridePaddingBytes() const {
        return stride_padding_bytes_;
    }

    bool isCpLocalSliced() const {
        return cp_local_sliced_;
    }

    size_t fullBlockStrideBytes() const {
        return full_block_stride_bytes_;
    }

    KVCacheSpecPtr clone() const override {
        return std::make_shared<PackedEntryKVCacheSpec>(*this);
    }

    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "PackedEntryKVCacheSpec{\n";
        os << commonDebugString(indent);
        const std::string indent1 = std::string(indent, ' ') + "  ";
        os << indent1 << "entry_size_bytes=" << static_cast<size_t>(entry_elems_) * getTypeSize(entry_dtype_) << "\n";
        os << indent1 << "entries_per_block=" << entry_count_ << "\n";
        os << indent1 << "entry_padding_count=" << entry_padding_count_ << "\n";
        os << indent1 << "stride_padding_bytes=" << stride_padding_bytes_ << "\n";
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }

protected:
    struct ResolvedLayout {
        uint32_t entry_count;
        size_t   payload_bytes;
        size_t   block_stride_bytes;
        size_t   entry_padding_count;
        size_t   stride_padding_bytes;
        size_t   full_block_stride_bytes;
        bool     cp_local_sliced;
    };

    void setLayout(uint32_t entry_elems, const ResolvedLayout& layout) {
        RTP_LLM_CHECK_WITH_INFO(entry_elems > 0, "packed-entry layout requires positive entry_elems");
        RTP_LLM_CHECK_WITH_INFO(layout.entry_count > 0, "packed-entry layout requires positive entry_count");
        RTP_LLM_CHECK_WITH_INFO(layout.payload_bytes > 0, "packed-entry layout requires positive payload bytes");
        RTP_LLM_CHECK_WITH_INFO(layout.block_stride_bytes >= layout.payload_bytes,
                                "packed-entry block stride %zu must cover payload %zu",
                                layout.block_stride_bytes,
                                layout.payload_bytes);
        entry_elems_          = entry_elems;
        entry_count_          = layout.entry_count;
        payload_elems_        = static_cast<size_t>(entry_elems) * layout.entry_count;
        payload_bytes_        = layout.payload_bytes;
        block_stride_bytes_   = layout.block_stride_bytes;
        entry_padding_count_  = layout.entry_padding_count;
        stride_padding_bytes_ = layout.stride_padding_bytes;
        full_block_stride_bytes_ = layout.full_block_stride_bytes;
        cp_local_sliced_          = layout.cp_local_sliced;
    }

    static bool cpScaleSeqSize(const KVCacheSpecDesc& desc) {
        return desc.cp.has_value() && desc.cp->scale_seq_size.value_or(false);
    }

    static bool cpSlice(const KVCacheSpecDesc& desc) {
        return desc.cp.has_value() && desc.cp->slice.value_or(false);
    }

    static uint32_t fixedRegionCpSize(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        const bool needs_cp_size = cpScaleSeqSize(desc) || cpSlice(desc);
        if (!needs_cp_size) {
            return 1;
        }
        RTP_LLM_CHECK_WITH_INFO(ctx.parallelism_config != nullptr,
                                "KVCacheSpecDesc tag=%s cache_type=%d requires SpecBuildContext.parallelism_config",
                                desc.tag.c_str(),
                                static_cast<int>(desc.cache_type));
        const auto& parallelism_config = *ctx.parallelism_config;
        if (!parallelism_config.prefill_cp_config.kv_cache_sharded) {
            return 1;
        }
        if (parallelism_config.role_type == RoleType::PREFILL && parallelism_config.tp_size > 1) {
            if (parallelism_config.prefill_cp_config.is_prefill_enabled()) {
                RTP_LLM_CHECK_WITH_INFO(
                    parallelism_config.tp_size == parallelism_config.prefill_cp_config.prefill_cp_size,
                    "PREFILL CP sharding requires tp_size (%d) == prefill_cp_size (%d)",
                    parallelism_config.tp_size,
                    parallelism_config.prefill_cp_config.prefill_cp_size);
            }
            return static_cast<uint32_t>(parallelism_config.tp_size);
        }
        if (parallelism_config.role_type == RoleType::DECODE
            && parallelism_config.prefill_cp_config.is_prefill_enabled()) {
            RTP_LLM_CHECK_WITH_INFO(
                parallelism_config.prefill_cp_config.prefill_cp_size > 1,
                "fixed/SWA CP sharding decode requires explicit prefill_cp_size when PREFILL_CP and kv_cache_sharded are enabled");
            return static_cast<uint32_t>(parallelism_config.prefill_cp_config.prefill_cp_size);
        }
        return 1;
    }

    static bool isPrefillCpSliced(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        if (!cpSlice(desc)) {
            return false;
        }
        const auto cp_size = fixedRegionCpSize(desc, ctx);
        if (cp_size <= 1) {
            return false;
        }
        RTP_LLM_CHECK_WITH_INFO(ctx.parallelism_config != nullptr,
                                "KVCacheSpecDesc tag=%s cache_type=%d requires SpecBuildContext.parallelism_config",
                                desc.tag.c_str(),
                                static_cast<int>(desc.cache_type));
        return ctx.parallelism_config->role_type == RoleType::PREFILL;
    }

    static uint32_t seqSizePerBlock(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        const uint32_t ctx_seq_size = ctx.seq_size_per_block == 0 ? 1 : ctx.seq_size_per_block;
        const uint32_t cp_size      = fixedRegionCpSize(desc, ctx);
        if (cpScaleSeqSize(desc) && cp_size > 1) {
            return ctx_seq_size * cp_size;
        }
        return ctx_seq_size;
    }

    static uint32_t alignUpToMultiple(uint32_t value, uint32_t multiple) {
        RTP_LLM_CHECK_WITH_INFO(multiple > 0, "align multiple must be > 0");
        return ((value + multiple - 1) / multiple) * multiple;
    }

    static uint32_t stateRingEntries(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        RTP_LLM_CHECK_WITH_INFO(
            desc.compression_ratio > 0, "state ring desc tag=%s requires positive compression_ratio", desc.tag.c_str());
        const uint32_t window = (1 + desc.state_ring_overlap) * desc.compression_ratio;
        const uint32_t raw    = window + (desc.state_ring_include_gen_num_per_cycle ? ctx.gen_num_per_cycle : 0);
        return (raw + 1) & ~1U;
    }

    static uint32_t logicalEntryCount(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        uint32_t entries = 0;
        switch (desc.entry_count_mode) {
            case BlockEntryCountMode::KERNEL_BLOCK_COMPRESSED:
                RTP_LLM_CHECK_WITH_INFO(
                    desc.compression_ratio > 0,
                    "desc tag=%s derives entries from kernel block but has invalid compression_ratio=%u",
                    desc.tag.c_str(),
                    desc.compression_ratio);
                RTP_LLM_CHECK_WITH_INFO(
                    ctx.kernel_tokens_per_block > 0,
                    "desc tag=%s derives entries from kernel block but kernel_tokens_per_block is 0",
                    desc.tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(ctx.kernel_tokens_per_block % desc.compression_ratio == 0,
                                        "desc tag=%s compression_ratio=%u must divide kernel block %u",
                                        desc.tag.c_str(),
                                        desc.compression_ratio,
                                        ctx.kernel_tokens_per_block);
                entries = ctx.kernel_tokens_per_block / desc.compression_ratio;
                break;
            case BlockEntryCountMode::STATE_RING:
                entries = stateRingEntries(desc, ctx);
                break;
            case BlockEntryCountMode::EXPLICIT:
                entries = desc.explicit_entry_count;
                break;
        }

        return entries;
    }

    static size_t payloadBytes(uint32_t entry_elems, uint32_t entry_count, DataType entry_dtype) {
        RTP_LLM_CHECK_WITH_INFO(entry_elems > 0, "packed-entry layout requires positive entry_elems");
        RTP_LLM_CHECK_WITH_INFO(entry_count > 0, "packed-entry layout requires positive entry_count");
        RTP_LLM_CHECK_WITH_INFO(entry_dtype != DataType::TYPE_INVALID,
                                "packed-entry layout requires valid entry_dtype");
        return static_cast<size_t>(entry_count) * entry_elems * getTypeSize(entry_dtype);
    }

    static size_t blockStrideBytes(const KVCacheSpecDesc& desc, size_t payload_bytes) {
        if (desc.block_stride_bytes_alignment > 0) {
            return ((payload_bytes + desc.block_stride_bytes_alignment - 1) / desc.block_stride_bytes_alignment)
                   * desc.block_stride_bytes_alignment;
        }
        return payload_bytes;
    }

    static ResolvedLayout resolveLayout(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        const uint32_t logical_entries = logicalEntryCount(desc, ctx);
        const uint32_t cp_size         = fixedRegionCpSize(desc, ctx);
        const bool     sliced          = cpSlice(desc) && cp_size > 1;
        const uint32_t padded_entries  = sliced ? alignUpToMultiple(logical_entries, cp_size) : logical_entries;
        const size_t   full_payload    = payloadBytes(desc.entry_elems, padded_entries, desc.entry_dtype);

        size_t full_stride = blockStrideBytes(desc, full_payload);
        if (sliced) {
            const size_t stride_alignment =
                desc.block_stride_bytes_alignment > 0 ?
                    std::lcm(desc.block_stride_bytes_alignment, static_cast<size_t>(cp_size)) :
                    static_cast<size_t>(cp_size);
            full_stride = ((full_stride + stride_alignment - 1) / stride_alignment) * stride_alignment;
        }

        RTP_LLM_CHECK_WITH_INFO(!sliced || full_stride % cp_size == 0,
                                "CP sliced layout tag=%s full stride %zu must be divisible by cp_size %u",
                                desc.tag.c_str(),
                                full_stride,
                                cp_size);

        const bool     local          = sliced && isPrefillCpSliced(desc, ctx);
        const uint32_t entry_count    = local ? padded_entries / cp_size : padded_entries;
        const size_t   payload_bytes  = local ? full_payload / cp_size : full_payload;
        const size_t   stride_bytes   = local ? full_stride / cp_size : full_stride;
        const size_t   entry_padding  = static_cast<size_t>(padded_entries - logical_entries);
        const size_t   stride_padding = full_stride - full_payload;
        return {entry_count,
                payload_bytes,
                stride_bytes,
                entry_padding,
                stride_padding,
                full_stride,
                local};
    }

protected:
    DataType entry_dtype_ = DataType::TYPE_INVALID;

private:
    uint32_t entry_elems_ = 0;
    uint32_t entry_count_ = 0;

    size_t payload_elems_        = 0;
    size_t payload_bytes_        = 0;
    size_t block_stride_bytes_   = 0;
    size_t entry_padding_count_  = 0;
    size_t stride_padding_bytes_ = 0;
    size_t full_block_stride_bytes_ = 0;
    bool   cp_local_sliced_          = false;
};

struct CompressedKVCacheSpec: public PackedEntryKVCacheSpec {
    CompressedKVCacheSpec() {
        type = KVCacheSpecType::CompressedKVCache;
    }

    static KVCacheSpecPtr build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        RTP_LLM_CHECK_WITH_INFO(desc.entry_elems > 0,
                                "COMPRESSED_KV KVCacheSpecDesc tag=%s requires positive entry_elems",
                                desc.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(desc.entry_dtype != DataType::TYPE_INVALID,
                                "COMPRESSED_KV KVCacheSpecDesc tag=%s requires valid entry_dtype",
                                desc.tag.c_str());

        auto spec                = std::make_shared<CompressedKVCacheSpec>();
        spec->tag                = desc.tag;
        spec->seq_size_per_block = seqSizePerBlock(desc, ctx);
        spec->entry_dtype_       = desc.entry_dtype;
        const auto layout        = resolveLayout(desc, ctx);
        spec->setLayout(desc.entry_elems, layout);
        return spec;
    }

    KVCacheSpecPtr clone() const override {
        return std::make_shared<CompressedKVCacheSpec>(*this);
    }

    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "CompressedKVCacheSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }
};

struct SWAStateCacheSpec: public PackedEntryKVCacheSpec {
    SWAStateCacheSpec() {
        type = KVCacheSpecType::SWAState;
    }

    static KVCacheSpecPtr build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        RTP_LLM_CHECK_WITH_INFO(
            desc.entry_elems > 0, "SWA_STATE KVCacheSpecDesc tag=%s requires positive entry_elems", desc.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(desc.entry_dtype != DataType::TYPE_INVALID,
                                "SWA_STATE KVCacheSpecDesc tag=%s requires valid entry_dtype",
                                desc.tag.c_str());

        auto spec                = std::make_shared<SWAStateCacheSpec>();
        spec->tag                = desc.tag;
        spec->seq_size_per_block = seqSizePerBlock(desc, ctx);
        spec->entry_dtype_       = desc.entry_dtype;
        const auto layout        = resolveLayout(desc, ctx);
        spec->setLayout(desc.entry_elems, layout);
        return spec;
    }

    KVCacheSpecPtr clone() const override {
        return std::make_shared<SWAStateCacheSpec>(*this);
    }

    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "SWAStateCacheSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }
};

}  // namespace rtp_llm
