#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>

#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"

namespace rtp_llm {

struct OpaqueKVCacheSpec: public KVCacheSpec {
    explicit OpaqueKVCacheSpec(uint32_t seq_size_per_block = 1): KVCacheSpec(seq_size_per_block) {
        type = KVCacheSpecType::OpaqueKV;
    }

    OpaqueKVCacheSpec(uint32_t seq_size_per_block, uint32_t kernel_seq_size_per_block):
        KVCacheSpec(seq_size_per_block, kernel_seq_size_per_block) {
        type = KVCacheSpecType::OpaqueKV;
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

    KVCacheSpecPtr clone() const override {
        return std::make_shared<OpaqueKVCacheSpec>(*this);
    }

    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "OpaqueKVCacheSpec{\n";
        os << commonDebugString(indent);
        const std::string indent1 = std::string(indent, ' ') + "  ";
        os << indent1 << "entry_size_bytes=" << static_cast<size_t>(entry_elems_) * getTypeSize(entry_dtype_) << "\n";
        os << indent1 << "entries_per_block=" << entry_count_ << "\n";
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }

protected:
    void setLayout(uint32_t entry_elems,
                   uint32_t entry_count,
                   size_t   payload_bytes,
                   size_t   block_stride_bytes,
                   bool     allow_partial_block_stride = false) {
        RTP_LLM_CHECK_WITH_INFO(entry_elems > 0, "opaque KV layout requires positive entry_elems");
        RTP_LLM_CHECK_WITH_INFO(entry_count > 0, "opaque KV layout requires positive entry_count");
        RTP_LLM_CHECK_WITH_INFO(payload_bytes > 0, "opaque KV layout requires positive payload bytes");
        if (!allow_partial_block_stride) {
            RTP_LLM_CHECK_WITH_INFO(block_stride_bytes >= payload_bytes,
                                    "opaque KV block stride %zu must cover payload %zu",
                                    block_stride_bytes,
                                    payload_bytes);
        }
        entry_elems_        = entry_elems;
        entry_count_        = entry_count;
        payload_elems_      = static_cast<size_t>(entry_elems) * entry_count;
        payload_bytes_      = payload_bytes;
        block_stride_bytes_ = block_stride_bytes;
    }

    static bool cpAlignPayload(const KVCacheSpecDesc& desc) {
        return desc.cp.has_value() && desc.cp->align_payload.value_or(false);
    }

    static CpPrefillSliceLayout cpPrefillSliceLayout(const KVCacheSpecDesc& desc) {
        if (!desc.cp.has_value() || !desc.cp->prefill_slice_layout.has_value()) {
            return CpPrefillSliceLayout::NONE;
        }
        return *desc.cp->prefill_slice_layout;
    }

    static bool cpPrefillSlicePayload(const KVCacheSpecDesc& desc) {
        return cpPrefillSliceLayout(desc) == CpPrefillSliceLayout::PAYLOAD;
    }

    static bool cpPrefillSliceBlockStride(const KVCacheSpecDesc& desc) {
        return cpPrefillSliceLayout(desc) == CpPrefillSliceLayout::BLOCK_STRIDE;
    }

    static uint32_t fixedRegionCpSize(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        const bool needs_cp_size =
            cpAlignPayload(desc) || cpPrefillSlicePayload(desc) || cpPrefillSliceBlockStride(desc);
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

    static uint32_t alignUpToMultiple(uint32_t value, uint32_t multiple) {
        RTP_LLM_CHECK_WITH_INFO(multiple > 0, "align multiple must be > 0");
        RTP_LLM_CHECK_WITH_INFO(value <= std::numeric_limits<uint32_t>::max() - (multiple - 1),
                                "align overflow: value=%u multiple=%u",
                                value,
                                multiple);
        return ((value + multiple - 1) / multiple) * multiple;
    }

    static uint32_t stateRingEntries(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        RTP_LLM_CHECK_WITH_INFO(
            desc.compression_ratio > 0, "state ring desc tag=%s requires positive compression_ratio", desc.tag.c_str());
        const uint64_t window = (uint64_t{1} + desc.state_ring_overlap) * desc.compression_ratio;
        const uint64_t raw    = window + (desc.state_ring_include_gen_num_per_cycle ? ctx.gen_num_per_cycle : 0);
        RTP_LLM_CHECK_WITH_INFO(raw < std::numeric_limits<uint32_t>::max(),
                                "state ring entry count overflow for desc tag=%s",
                                desc.tag.c_str());
        return static_cast<uint32_t>((raw + 1) & ~uint64_t{1});
    }

    static uint32_t entryCount(const KVCacheSpecDesc&  desc,
                               const SpecBuildContext& ctx,
                               uint32_t                seq_size_per_block,
                               uint32_t                kernel_seq_size_per_block) {
        uint32_t entries = 0;
        switch (desc.entry_count_mode) {
            case OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED:
                RTP_LLM_CHECK_WITH_INFO(
                    desc.compression_ratio > 0,
                    "desc tag=%s derives entries from kernel block but has invalid compression_ratio=%u",
                    desc.tag.c_str(),
                    desc.compression_ratio);
                RTP_LLM_CHECK_WITH_INFO(seq_size_per_block % desc.compression_ratio == 0,
                                        "desc tag=%s compression_ratio=%u must divide physical block span %u",
                                        desc.tag.c_str(),
                                        desc.compression_ratio,
                                        seq_size_per_block);
                RTP_LLM_CHECK_WITH_INFO(kernel_seq_size_per_block % desc.compression_ratio == 0,
                                        "desc tag=%s compression_ratio=%u must divide kernel block span %u",
                                        desc.tag.c_str(),
                                        desc.compression_ratio,
                                        kernel_seq_size_per_block);
                entries = seq_size_per_block / desc.compression_ratio;
                break;
            case OpaqueBlockEntryCountMode::STATE_RING:
                entries = stateRingEntries(desc, ctx);
                break;
            case OpaqueBlockEntryCountMode::EXPLICIT:
                entries = desc.explicit_entry_count;
                break;
        }

        const auto cp_size = fixedRegionCpSize(desc, ctx);
        if (cp_size > 1 && (cpAlignPayload(desc) || cpPrefillSlicePayload(desc))) {
            entries = alignUpToMultiple(entries, cp_size);
            if (cpPrefillSlicePayload(desc) && isPrefillCpSliced(desc, ctx)) {
                entries /= cp_size;
            }
        }
        return entries;
    }

    static size_t payloadBytes(uint32_t entry_elems, uint32_t entry_count, DataType entry_dtype) {
        RTP_LLM_CHECK_WITH_INFO(entry_elems > 0, "opaque KV layout requires positive entry_elems");
        RTP_LLM_CHECK_WITH_INFO(entry_count > 0, "opaque KV layout requires positive entry_count");
        RTP_LLM_CHECK_WITH_INFO(entry_dtype != DataType::TYPE_INVALID, "opaque KV layout requires valid entry_dtype");
        const auto elem_count = static_cast<size_t>(entry_count) * entry_elems;
        const auto type_size  = getTypeSize(entry_dtype);
        RTP_LLM_CHECK_WITH_INFO(elem_count <= std::numeric_limits<size_t>::max() / type_size,
                                "opaque KV payload bytes overflow: entries=%u entry_elems=%u dtype_size=%zu",
                                entry_count,
                                entry_elems,
                                type_size);
        return elem_count * type_size;
    }

    static size_t blockStrideBytes(const KVCacheSpecDesc& desc, size_t payload_bytes, uint32_t entry_count) {
        if (desc.block_stride_bytes_override > 0) {
            return desc.block_stride_bytes_override;
        }
        if (desc.block_stride_bytes_alignment > 0 && entry_count >= desc.block_stride_alignment_min_entries) {
            return ((payload_bytes + desc.block_stride_bytes_alignment - 1) / desc.block_stride_bytes_alignment)
                   * desc.block_stride_bytes_alignment;
        }
        return payload_bytes;
    }

    static size_t fixedStateBlockStrideBytes(const KVCacheSpecDesc&  desc,
                                             size_t                  payload_bytes,
                                             uint32_t                entry_count,
                                             const SpecBuildContext& ctx) {
        const auto cp_size = fixedRegionCpSize(desc, ctx);
        if (cp_size <= 1 || !isPrefillCpSliced(desc, ctx) || !cpPrefillSliceBlockStride(desc)) {
            return blockStrideBytes(desc, payload_bytes, entry_count);
        }
        const size_t align             = desc.block_stride_bytes_alignment > 0 ?
                                             std::lcm(desc.block_stride_bytes_alignment, static_cast<size_t>(cp_size)) :
                                             static_cast<size_t>(cp_size);
        const size_t full_stride_bytes = ((payload_bytes + align - 1) / align) * align;
        RTP_LLM_CHECK_WITH_INFO(full_stride_bytes % cp_size == 0,
                                "CP prefill byte slicing tag=%s full stride %zu must be divisible by cp_size %u",
                                desc.tag.c_str(),
                                full_stride_bytes,
                                cp_size);
        return full_stride_bytes / cp_size;
    }

protected:
    DataType entry_dtype_ = DataType::TYPE_INVALID;

private:
    uint32_t entry_elems_ = 0;
    uint32_t entry_count_ = 0;

    size_t payload_elems_      = 0;
    size_t payload_bytes_      = 0;
    size_t block_stride_bytes_ = 0;
};

struct CompressedKVCacheSpec: public OpaqueKVCacheSpec {
    explicit CompressedKVCacheSpec(uint32_t seq_size_per_block = 1): OpaqueKVCacheSpec(seq_size_per_block) {
        type = KVCacheSpecType::OpaqueKV;
    }

    CompressedKVCacheSpec(uint32_t seq_size_per_block, uint32_t kernel_seq_size_per_block):
        OpaqueKVCacheSpec(seq_size_per_block, kernel_seq_size_per_block) {
        type = KVCacheSpecType::OpaqueKV;
    }

    static KVCacheSpecPtr build(const KVCacheSpecDesc&  desc,
                                const SpecBuildContext& ctx,
                                uint32_t                seq_size_per_block,
                                uint32_t                kernel_seq_size_per_block) {
        RTP_LLM_CHECK_WITH_INFO(desc.entry_elems > 0,
                                "COMPRESSED_KV KVCacheSpecDesc tag=%s requires positive entry_elems",
                                desc.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(desc.entry_dtype != DataType::TYPE_INVALID,
                                "COMPRESSED_KV KVCacheSpecDesc tag=%s requires valid entry_dtype",
                                desc.tag.c_str());

        auto spec              = std::make_shared<CompressedKVCacheSpec>(seq_size_per_block, kernel_seq_size_per_block);
        spec->tag              = desc.tag;
        spec->entry_dtype_     = desc.entry_dtype;
        const uint32_t entries = entryCount(desc, ctx, seq_size_per_block, kernel_seq_size_per_block);
        const size_t   payload = payloadBytes(desc.entry_elems, entries, desc.entry_dtype);
        const size_t   stride  = blockStrideBytes(desc, payload, entries);
        spec->setLayout(desc.entry_elems, entries, payload, stride);
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

struct FixedStateCacheSpec: public OpaqueKVCacheSpec {
    explicit FixedStateCacheSpec(uint32_t seq_size_per_block = 1): OpaqueKVCacheSpec(seq_size_per_block) {
        type = KVCacheSpecType::OpaqueState;
    }

    FixedStateCacheSpec(uint32_t seq_size_per_block, uint32_t kernel_seq_size_per_block):
        OpaqueKVCacheSpec(seq_size_per_block, kernel_seq_size_per_block) {
        type = KVCacheSpecType::OpaqueState;
    }

    static KVCacheSpecPtr build(const KVCacheSpecDesc&  desc,
                                const SpecBuildContext& ctx,
                                uint32_t                seq_size_per_block,
                                uint32_t                kernel_seq_size_per_block) {
        RTP_LLM_CHECK_WITH_INFO(
            desc.entry_elems > 0, "FIXED_STATE KVCacheSpecDesc tag=%s requires positive entry_elems", desc.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(desc.entry_dtype != DataType::TYPE_INVALID,
                                "FIXED_STATE KVCacheSpecDesc tag=%s requires valid entry_dtype",
                                desc.tag.c_str());

        auto spec              = std::make_shared<FixedStateCacheSpec>(seq_size_per_block, kernel_seq_size_per_block);
        spec->tag              = desc.tag;
        spec->entry_dtype_     = desc.entry_dtype;
        const uint32_t entries = entryCount(desc, ctx, seq_size_per_block, kernel_seq_size_per_block);
        const size_t   payload = payloadBytes(desc.entry_elems, entries, desc.entry_dtype);
        const size_t   stride  = fixedStateBlockStrideBytes(desc, payload, entries, ctx);
        spec->setLayout(desc.entry_elems,
                        entries,
                        payload,
                        stride,
                        isPrefillCpSliced(desc, ctx) && cpPrefillSliceBlockStride(desc));
        return spec;
    }

    KVCacheSpecPtr clone() const override {
        return std::make_shared<FixedStateCacheSpec>(*this);
    }

    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "FixedStateCacheSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }
};

}  // namespace rtp_llm
