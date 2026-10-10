#include "rtp_llm/cpp/cache/OpaqueKVCacheSpec.h"

#include <limits>
#include <numeric>
#include <stdexcept>

namespace rtp_llm {

namespace {
uint32_t fixedRegionCpSize(const KVCacheSpecDesc& desc, const ParallelismConfig& parallelism) {
    const auto& cp       = *desc.cp;
    const bool  needs_cp = cp.scale_seq_size.value_or(false) || cp.align_payload.value_or(false)
                          || cp.prefill_slice_layout == CpPrefillSliceLayout::PAYLOAD
                          || cp.prefill_slice_layout == CpPrefillSliceLayout::BLOCK_STRIDE;
    if (!needs_cp || !parallelism.prefill_cp_config.kv_cache_sharded) {
        return 1;
    }
    if (parallelism.role_type == RoleType::PREFILL && parallelism.tp_size > 1) {
        return static_cast<uint32_t>(parallelism.tp_size);
    }
    if (parallelism.role_type == RoleType::DECODE && parallelism.prefill_cp_config.is_prefill_enabled()) {
        if (parallelism.prefill_cp_config.prefill_cp_size <= 1) {
            throw std::invalid_argument("fixed region decode CP layout requires explicit prefill_cp_size");
        }
        return static_cast<uint32_t>(parallelism.prefill_cp_config.prefill_cp_size);
    }
    return 1;
}
}  // namespace

ResolvedStateEntryLayouts
FixedStateCacheSpec::resolveEntryLayouts(const std::vector<std::vector<KVCacheSpecDesc>>& descs,
                                         const ParallelismConfig&                         parallelism,
                                         uint32_t                                         base_block_tokens,
                                         uint32_t                                         gen_num_per_cycle,
                                         uint32_t                                         kernel_block_tokens) {
    ResolvedStateEntryLayouts result;
    if (base_block_tokens == 0) {
        throw std::invalid_argument("STATE layout requires positive base block tokens");
    }
    for (const auto& layer : descs) {
        for (const auto& desc : layer) {
            if (desc.cache_type != KVCacheSpecType::OpaqueState || !desc.cp.has_value()
                || desc.cp->prefill_slice_layout == CpPrefillSliceLayout::BLOCK_STRIDE
                || !(desc.cp->scale_seq_size.value_or(false) || desc.cp->align_payload.value_or(false)
                     || desc.cp->prefill_slice_layout == CpPrefillSliceLayout::PAYLOAD)) {
                continue;
            }
            if (desc.entry_elems == 0 || desc.entry_dtype == DataType::TYPE_INVALID
                || (desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING && desc.compression_ratio == 0)) {
                throw std::invalid_argument("fixed entry projection requires a valid entry format");
            }
            const uint32_t cp_size = fixedRegionCpSize(desc, parallelism);
            uint64_t       raw     = desc.explicit_entry_count;
            if (desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING) {
                raw = (uint64_t{1} + desc.state_ring_overlap) * desc.compression_ratio
                      + (desc.state_ring_include_gen_num_per_cycle ? gen_num_per_cycle : 0);
                raw = (raw + 1) & ~uint64_t{1};
            } else if (desc.entry_count_mode == OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED) {
                if (desc.compression_ratio == 0 || kernel_block_tokens == 0
                    || kernel_block_tokens % desc.compression_ratio != 0) {
                    throw std::invalid_argument("fixed entry CP projection requires K divisible by ratio");
                }
                raw = kernel_block_tokens / desc.compression_ratio;
            }
            if (raw == 0) {
                throw std::invalid_argument("fixed entry projection requires positive entries");
            }
            const bool align = desc.cp->align_payload.value_or(false)
                               || desc.cp->prefill_slice_layout == CpPrefillSliceLayout::PAYLOAD;
            const uint64_t aligned       = align ? (raw + cp_size - 1) / cp_size * cp_size : raw;
            const bool     prefill_slice = desc.cp->prefill_slice_layout == CpPrefillSliceLayout::PAYLOAD && cp_size > 1
                                       && parallelism.role_type == RoleType::PREFILL;
            const uint64_t local = prefill_slice ? aligned / cp_size : aligned;
            const uint64_t seq =
                static_cast<uint64_t>(base_block_tokens) * (desc.cp->scale_seq_size.value_or(false) ? cp_size : 1);
            if (aligned > std::numeric_limits<uint32_t>::max() || local > std::numeric_limits<uint32_t>::max()
                || seq > std::numeric_limits<uint32_t>::max()) {
                throw std::invalid_argument("STATE entry projection exceeds uint32 range");
            }
            const size_t   entry_bytes = static_cast<size_t>(desc.entry_elems) * getTypeSize(desc.entry_dtype);
            const uint32_t rank        = prefill_slice ? static_cast<uint32_t>(parallelism.tp_rank) : 0;
            if (rank >= cp_size) {
                throw std::invalid_argument("STATE entry projection rank exceeds CP size");
            }
            ResolvedStateEntryLayout value{static_cast<uint32_t>(seq),
                                           static_cast<uint32_t>(aligned),
                                           static_cast<uint32_t>(rank * local),
                                           static_cast<uint32_t>(local),
                                           entry_bytes};
            const auto [it, inserted] = result.emplace(desc.tag, value);
            if (!inserted
                && (it->second.seq_size_per_block != value.seq_size_per_block
                    || it->second.full_entries != value.full_entries
                    || it->second.local_entry_begin != value.local_entry_begin
                    || it->second.local_entries != value.local_entries
                    || it->second.entry_bytes != value.entry_bytes)) {
                throw std::invalid_argument("STATE tag has inconsistent entry projection across layers");
            }
        }
    }
    return result;
}

ResolvedCompressedLayouts CompressedKVCacheSpec::resolveLayouts(const std::vector<std::vector<KVCacheSpecDesc>>& descs,
                                                                const ParallelismConfig& parallelism,
                                                                uint32_t                 base_block_tokens,
                                                                uint32_t                 kernel_block_tokens,
                                                                uint32_t                 gen_num_per_cycle) {
    if (base_block_tokens == 0) {
        throw std::invalid_argument("compressed layout requires positive base block tokens");
    }
    ResolvedCompressedLayouts result;
    for (const auto& layer : descs) {
        for (const auto& desc : layer) {
            if (desc.cache_type != KVCacheSpecType::OpaqueKV || !desc.cp.has_value()
                || !(desc.cp->scale_seq_size.value_or(false) || desc.cp->align_payload.value_or(false)
                     || desc.cp->prefill_slice_layout == CpPrefillSliceLayout::PAYLOAD
                     || desc.cp->prefill_slice_layout == CpPrefillSliceLayout::BLOCK_STRIDE)) {
                continue;
            }
            const uint32_t cp_size = fixedRegionCpSize(desc, parallelism);
            const uint64_t seq =
                static_cast<uint64_t>(base_block_tokens) * (desc.cp->scale_seq_size.value_or(false) ? cp_size : 1);
            if (seq > std::numeric_limits<uint32_t>::max()) {
                throw std::invalid_argument("compressed CP block token span exceeds uint32 range");
            }
            const uint32_t kernel  = SpecBuilder::groupType(desc) != CacheGroupType::FULL || kernel_block_tokens == 0 ?
                                         static_cast<uint32_t>(seq) :
                                         std::min(kernel_block_tokens, static_cast<uint32_t>(seq));
            uint64_t       entries = desc.explicit_entry_count;
            if (desc.entry_count_mode == OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED) {
                if (desc.compression_ratio == 0 || kernel == 0 || kernel % desc.compression_ratio != 0) {
                    throw std::invalid_argument("compressed CP projection requires K divisible by ratio");
                }
                entries = kernel / desc.compression_ratio;
            } else if (desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING) {
                if (desc.compression_ratio == 0) {
                    throw std::invalid_argument("compressed CP projection requires positive ratio");
                }
                entries = (uint64_t{1} + desc.state_ring_overlap) * desc.compression_ratio
                          + (desc.state_ring_include_gen_num_per_cycle ? gen_num_per_cycle : 0);
                entries = (entries + 1) & ~uint64_t{1};
            }
            const bool align = cp_size > 1
                               && (desc.cp->align_payload.value_or(false)
                                   || desc.cp->prefill_slice_layout == CpPrefillSliceLayout::PAYLOAD);
            if (align) {
                entries = (entries + cp_size - 1) / cp_size * cp_size;
                if (desc.cp->prefill_slice_layout == CpPrefillSliceLayout::PAYLOAD
                    && parallelism.role_type == RoleType::PREFILL) {
                    entries /= cp_size;
                }
            }
            if (entries == 0 || entries > std::numeric_limits<uint32_t>::max()) {
                throw std::invalid_argument("compressed CP entry count exceeds uint32 range");
            }
            const ResolvedCompressedLayout value{static_cast<uint32_t>(seq), static_cast<uint32_t>(entries)};
            const auto [it, inserted] = result.emplace(desc.tag, value);
            if (!inserted
                && (it->second.seq_size_per_block != value.seq_size_per_block
                    || it->second.entries_per_kernel_block != value.entries_per_kernel_block)) {
                throw std::invalid_argument("compressed tag has inconsistent CP shape across layers");
            }
        }
    }
    return result;
}

ResolvedSwaByteLayouts
FixedStateCacheSpec::resolveSwaByteLayouts(const std::vector<std::vector<KVCacheSpecDesc>>& descs,
                                           const ParallelismConfig&                         parallelism,
                                           uint32_t                                         base_block_tokens,
                                           uint32_t                                         gen_num_per_cycle,
                                           uint32_t                                         kernel_block_tokens) {
    ResolvedSwaByteLayouts result;
    if (base_block_tokens == 0) {
        throw std::invalid_argument("SWA layout requires positive base block tokens");
    }
    for (const auto& layer : descs) {
        for (const auto& desc : layer) {
            if (desc.cache_type != KVCacheSpecType::OpaqueState || !desc.cp.has_value()
                || desc.cp->prefill_slice_layout != CpPrefillSliceLayout::BLOCK_STRIDE) {
                continue;
            }
            if (desc.entry_elems == 0 || desc.entry_dtype == DataType::TYPE_INVALID
                || (desc.entry_count_mode != OpaqueBlockEntryCountMode::EXPLICIT && desc.compression_ratio == 0)) {
                throw std::invalid_argument("SWA packed-byte projection requires a valid entry format");
            }
            const uint32_t cp_size     = fixedRegionCpSize(desc, parallelism);
            uint64_t       raw_entries = desc.explicit_entry_count;
            if (desc.entry_count_mode == OpaqueBlockEntryCountMode::STATE_RING) {
                raw_entries = (uint64_t{1} + desc.state_ring_overlap) * desc.compression_ratio
                              + (desc.state_ring_include_gen_num_per_cycle ? gen_num_per_cycle : 0);
                raw_entries = (raw_entries + 1) & ~uint64_t{1};
            } else if (desc.entry_count_mode == OpaqueBlockEntryCountMode::KERNEL_BLOCK_COMPRESSED) {
                if (kernel_block_tokens == 0 || kernel_block_tokens % desc.compression_ratio != 0) {
                    throw std::invalid_argument("SWA packed-byte projection requires K divisible by ratio");
                }
                raw_entries = kernel_block_tokens / desc.compression_ratio;
            }
            if (raw_entries == 0 || raw_entries > std::numeric_limits<uint32_t>::max()) {
                throw std::invalid_argument("SWA packed-byte entry count is out of range");
            }
            const uint64_t aligned_entries =
                desc.cp->align_payload.value_or(false) ? (raw_entries + cp_size - 1) / cp_size * cp_size : raw_entries;
            const uint64_t seq =
                static_cast<uint64_t>(base_block_tokens) * (desc.cp->scale_seq_size.value_or(false) ? cp_size : 1);
            const size_t entry_bytes = static_cast<size_t>(desc.entry_elems) * getTypeSize(desc.entry_dtype);
            if (aligned_entries > std::numeric_limits<uint32_t>::max() || seq > std::numeric_limits<uint32_t>::max()
                || aligned_entries > std::numeric_limits<size_t>::max() / entry_bytes) {
                throw std::invalid_argument("SWA packed-byte shape exceeds supported range");
            }
            const size_t payload = static_cast<size_t>(aligned_entries) * entry_bytes;
            const bool   sliced  = cp_size > 1 && parallelism.role_type == RoleType::PREFILL;
            size_t       stride  = payload;
            if (sliced) {
                const size_t align = desc.block_stride_bytes_alignment > 0 ?
                                         std::lcm(desc.block_stride_bytes_alignment, static_cast<size_t>(cp_size)) :
                                         static_cast<size_t>(cp_size);
                if (payload > std::numeric_limits<size_t>::max() - (align - 1)) {
                    throw std::invalid_argument("SWA packed-byte stride overflows size_t");
                }
                stride = (payload + align - 1) / align * align;
            } else if (desc.block_stride_bytes_override > 0) {
                stride = desc.block_stride_bytes_override;
            } else if (desc.block_stride_bytes_alignment > 0
                       && aligned_entries >= desc.block_stride_alignment_min_entries) {
                const size_t align = desc.block_stride_bytes_alignment;
                if (payload > std::numeric_limits<size_t>::max() - (align - 1)) {
                    throw std::invalid_argument("SWA packed-byte stride overflows size_t");
                }
                stride = (payload + align - 1) / align * align;
            }
            if (sliced && stride % cp_size != 0) {
                throw std::invalid_argument("SWA packed-byte stride is not divisible by CP size");
            }
            const size_t   local_stride = sliced ? stride / cp_size : stride;
            const uint32_t rank         = sliced ? static_cast<uint32_t>(parallelism.tp_rank) : 0;
            if (rank >= cp_size) {
                throw std::invalid_argument("SWA packed-byte projection rank exceeds CP size");
            }
            const ResolvedSwaByteLayout value{static_cast<uint32_t>(seq),
                                              static_cast<uint32_t>(aligned_entries),
                                              payload,
                                              stride,
                                              static_cast<size_t>(rank) * local_stride,
                                              local_stride};
            const auto [it, inserted] = result.emplace(desc.tag, value);
            if (!inserted
                && (it->second.seq_size_per_block != value.seq_size_per_block
                    || it->second.full_entries != value.full_entries
                    || it->second.full_payload_bytes != value.full_payload_bytes
                    || it->second.full_stride_bytes != value.full_stride_bytes
                    || it->second.local_byte_begin != value.local_byte_begin
                    || it->second.local_stride_bytes != value.local_stride_bytes)) {
                throw std::invalid_argument("SWA tag has inconsistent packed-byte projection across layers");
            }
        }
    }
    return result;
}

}  // namespace rtp_llm
