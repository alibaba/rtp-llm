#pragma once

#include <algorithm>
#include <memory>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"

namespace rtp_llm {

struct OpaqueKVCacheSpec: public KVCacheSpec {
    OpaqueKVCacheSpec(std::string tag                       = {},
                      uint32_t    seq_size_per_block        = 1,
                      uint32_t    kernel_seq_size_per_block = 1,
                      uint32_t    local_kv_head_num         = 1):
        KVCacheSpec(std::move(tag), seq_size_per_block, kernel_seq_size_per_block, local_kv_head_num) {
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

    // Logical payload entries, excluding byte alignment. For BLOCK_STRIDE CP
    // slices this describes the complete logical ring, before byte slicing.
    uint32_t entriesPerBlock() const {
        return entry_count_;
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

    static uint32_t seqSizePerBlock(const SpecBuildContext& ctx) {
        return ctx.seq_size_per_block == 0 ? 1 : ctx.seq_size_per_block;
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

    static uint32_t entryCount(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        uint32_t entries = 0;
        switch (desc.entry_count_mode) {
            case OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED:
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
            case OpaqueBlockEntryCountMode::STATE_RING:
                entries = stateRingEntries(desc, ctx);
                break;
            case OpaqueBlockEntryCountMode::EXPLICIT:
                entries = desc.explicit_entry_count;
                break;
        }

        return entries;
    }

    static size_t payloadBytes(uint32_t entry_elems, uint32_t entry_count, DataType entry_dtype) {
        RTP_LLM_CHECK_WITH_INFO(entry_elems > 0, "opaque KV layout requires positive entry_elems");
        RTP_LLM_CHECK_WITH_INFO(entry_count > 0, "opaque KV layout requires positive entry_count");
        RTP_LLM_CHECK_WITH_INFO(entry_dtype != DataType::TYPE_INVALID, "opaque KV layout requires valid entry_dtype");
        return static_cast<size_t>(entry_count) * entry_elems * getTypeSize(entry_dtype);
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
    CompressedKVCacheSpec(std::string tag                       = {},
                          uint32_t    seq_size_per_block        = 1,
                          uint32_t    kernel_seq_size_per_block = 1,
                          uint32_t    local_kv_head_num         = 1):
        OpaqueKVCacheSpec(std::move(tag), seq_size_per_block, kernel_seq_size_per_block, local_kv_head_num) {
        type = KVCacheSpecType::OpaqueKV;
    }

    static ResolvedCompressedLayouts resolveLayouts(const std::vector<std::vector<KVCacheSpecDesc>>& descs,
                                                    const ParallelismConfig&                         parallelism,
                                                    uint32_t                                         base_block_tokens,
                                                    uint32_t kernel_block_tokens,
                                                    uint32_t gen_num_per_cycle);

    static KVCacheSpecPtr build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        RTP_LLM_CHECK_WITH_INFO(desc.entry_elems > 0,
                                "COMPRESSED_KV KVCacheSpecDesc tag=%s requires positive entry_elems",
                                desc.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(desc.entry_dtype != DataType::TYPE_INVALID,
                                "COMPRESSED_KV KVCacheSpecDesc tag=%s requires valid entry_dtype",
                                desc.tag.c_str());

        uint32_t seq = seqSizePerBlock(ctx);
        const ResolvedCompressedLayout* resolved = nullptr;
        if (ctx.compressed_layouts != nullptr) {
            const auto it = ctx.compressed_layouts->find(desc.tag);
            if (it != ctx.compressed_layouts->end()) {
                resolved = &it->second;
                seq      = resolved->seq_size_per_block;
            }
        }
        const auto kernel                = SpecBuilder::kernelSeqSizePerBlock(desc, ctx, seq);
        auto       spec                  = std::make_shared<CompressedKVCacheSpec>(desc.tag, seq, kernel, 1);
        spec->entry_dtype_               = desc.entry_dtype;
        auto page_ctx                    = ctx;
        page_ctx.kernel_tokens_per_block = kernel;
        const uint32_t entries           = resolved != nullptr ? resolved->entries_per_kernel_block :
                                                                 entryCount(desc, page_ctx);
        spec->entries_per_kernel_block_  = entries;
        const size_t   payload           = payloadBytes(desc.entry_elems, entries, desc.entry_dtype);
        const size_t   stride            = blockStrideBytes(desc, payload, entries);
        const size_t   bpk =
            desc.entry_count_mode == OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED ? seq / kernel : 1;
        RTP_LLM_CHECK_WITH_INFO(entries <= std::numeric_limits<uint32_t>::max() / bpk
                                    && payload <= std::numeric_limits<size_t>::max() / bpk
                                    && stride <= std::numeric_limits<size_t>::max() / bpk,
                                "opaque physical block layout overflows for tag=%s",
                                desc.tag.c_str());
        // Preserve each kernel page's padding; all public block sizes describe a physical block.
        spec->setLayout(desc.entry_elems, static_cast<uint32_t>(entries * bpk), payload * bpk, stride * bpk);
        return spec;
    }

    KVCacheSpecPtr clone() const override {
        return std::make_shared<CompressedKVCacheSpec>(*this);
    }

    uint32_t entriesPerKernelBlock() const {
        return entries_per_kernel_block_;
    }

private:
    uint32_t entries_per_kernel_block_ = 0;

public:
    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "CompressedKVCacheSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }
};

struct FixedStateCacheSpec: public OpaqueKVCacheSpec {
    FixedStateCacheSpec(std::string tag                       = {},
                        uint32_t    seq_size_per_block        = 1,
                        uint32_t    kernel_seq_size_per_block = 1,
                        uint32_t    local_kv_head_num         = 1):
        OpaqueKVCacheSpec(std::move(tag), seq_size_per_block, kernel_seq_size_per_block, local_kv_head_num) {
        type = KVCacheSpecType::OpaqueState;
    }

    static ResolvedStateEntryLayouts resolveEntryLayouts(const std::vector<std::vector<KVCacheSpecDesc>>& descs,
                                                         const ParallelismConfig&                         parallelism,
                                                         uint32_t base_block_tokens,
                                                         uint32_t gen_num_per_cycle,
                                                         uint32_t kernel_block_tokens);
    static ResolvedSwaByteLayouts    resolveSwaByteLayouts(const std::vector<std::vector<KVCacheSpecDesc>>& descs,
                                                           const ParallelismConfig&                         parallelism,
                                                           uint32_t base_block_tokens,
                                                           uint32_t gen_num_per_cycle,
                                                           uint32_t kernel_block_tokens);

    static KVCacheSpecPtr build(const KVCacheSpecDesc& desc, const SpecBuildContext& ctx) {
        RTP_LLM_CHECK_WITH_INFO(
            desc.entry_elems > 0, "FIXED_STATE KVCacheSpecDesc tag=%s requires positive entry_elems", desc.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(desc.entry_dtype != DataType::TYPE_INVALID,
                                "FIXED_STATE KVCacheSpecDesc tag=%s requires valid entry_dtype",
                                desc.tag.c_str());

        const ResolvedStateEntryLayout* resolved = nullptr;
        if (ctx.state_entry_layouts != nullptr) {
            const auto it = ctx.state_entry_layouts->find(desc.tag);
            if (it != ctx.state_entry_layouts->end()) {
                resolved = &it->second;
            }
        }
        const ResolvedSwaByteLayout* resolved_swa = nullptr;
        if (ctx.swa_byte_layouts != nullptr) {
            const auto it = ctx.swa_byte_layouts->find(desc.tag);
            if (it != ctx.swa_byte_layouts->end()) {
                resolved_swa = &it->second;
            }
        }
        RTP_LLM_CHECK_WITH_INFO(resolved == nullptr || resolved_swa == nullptr,
                                "fixed region tag=%s cannot use entry and packed-byte projections together",
                                desc.tag.c_str());
        const auto seq = resolved ? resolved->seq_size_per_block :
                         resolved_swa ? resolved_swa->seq_size_per_block : seqSizePerBlock(ctx);
        const auto kernel      = SpecBuilder::kernelSeqSizePerBlock(desc, ctx, seq);
        auto       spec        = std::make_shared<FixedStateCacheSpec>(desc.tag, seq, kernel, 1);
        spec->entry_dtype_     = desc.entry_dtype;
        if (resolved != nullptr) {
            spec->state_entry_layout_ = *resolved;
        }
        if (resolved_swa != nullptr) {
            spec->swa_byte_layout_ = *resolved_swa;
        }
        const uint32_t entries = resolved ? resolved->local_entries :
                                 resolved_swa ? resolved_swa->full_entries : entryCount(desc, ctx);
        const size_t payload = resolved_swa ? resolved_swa->full_payload_bytes :
                                            payloadBytes(desc.entry_elems, entries, desc.entry_dtype);
        const size_t stride = resolved_swa ? resolved_swa->local_stride_bytes :
                                              blockStrideBytes(desc, payload, entries);
        spec->setLayout(desc.entry_elems,
                        entries,
                        payload,
                        stride,
                        resolved_swa != nullptr && stride < payload);
        return spec;
    }

    KVCacheSpecPtr clone() const override {
        return std::make_shared<FixedStateCacheSpec>(*this);
    }

    const ResolvedStateEntryLayout* stateEntryLayout() const {
        return state_entry_layout_ ? &*state_entry_layout_ : nullptr;
    }

    const ResolvedSwaByteLayout* swaByteLayout() const {
        return swa_byte_layout_ ? &*swa_byte_layout_ : nullptr;
    }

private:
    std::optional<ResolvedStateEntryLayout> state_entry_layout_;
    std::optional<ResolvedSwaByteLayout>    swa_byte_layout_;

public:
    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "FixedStateCacheSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }
};

}  // namespace rtp_llm
