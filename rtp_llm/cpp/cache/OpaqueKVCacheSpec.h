#pragma once

#include <utility>

#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct OpaqueKVCacheSpec: public KVCacheSpec {
    uint32_t entry_elems                       = 0;
    uint32_t entries_per_block                 = 0;
    DataType store_dtype                       = DataType::TYPE_INVALID;
    size_t   block_size_bytes_override         = 0;
    size_t   block_size_bytes_alignment        = 0;
    uint32_t block_size_alignment_min_entries  = 0;

    OpaqueKVCacheSpec() = default;

    OpaqueKVCacheSpec(KVCacheSpecType spec_type,
                      CacheGroupType  lifecycle_type,
                      uint32_t        entry_elements,
                      uint32_t        block_entries,
                      DataType        storage_dtype,
                      uint32_t        seq_size_per_blk,
                      size_t          block_size_bytes_override_value = 0,
                      size_t          block_size_alignment            = 0,
                      uint32_t        block_alignment_min_entries     = 0) {
        type                             = spec_type;
        lifecycle                        = lifecycle_type;
        entry_elems                      = entry_elements;
        entries_per_block                = block_entries;
        store_dtype                      = storage_dtype;
        block_size_bytes_override        = block_size_bytes_override_value;
        block_size_bytes_alignment       = block_size_alignment;
        block_size_alignment_min_entries = block_alignment_min_entries;

        local_head_num_kv  = 1;
        seq_size_per_block = seq_size_per_blk;
        dtype              = store_dtype;
    }

    size_t block_size() const override {
        return static_cast<size_t>(entries_per_block) * entry_elems;
    }

    size_t k_block_size() const override {
        return block_size() / 2;
    }

    size_t v_block_size() const override {
        return block_size() / 2;
    }

    size_t natural_block_size_bytes() const {
        return static_cast<size_t>(entries_per_block) * entry_elems * getTypeSize(store_dtype);
    }

    size_t block_size_bytes() const override {
        if (block_size_bytes_override > 0) {
            return block_size_bytes_override;
        }
        const size_t natural = natural_block_size_bytes();
        if (block_size_bytes_alignment > 0 && entries_per_block >= block_size_alignment_min_entries) {
            return ((natural + block_size_bytes_alignment - 1) / block_size_bytes_alignment)
                   * block_size_bytes_alignment;
        }
        return natural;
    }

    size_t k_block_size_bytes() const override {
        return block_size_bytes() / 2;
    }

    size_t v_block_size_bytes() const override {
        return block_size_bytes() / 2;
    }

    KVCacheSpecPtr clone() const override {
        return std::make_shared<OpaqueKVCacheSpec>(*this);
    }

protected:
    std::string opaqueFingerprintExtra(const std::string& prefix) const {
        std::ostringstream os;
        os << ";" << prefix << ".entry_elems=" << entry_elems
           << ";" << prefix << ".entries_per_block=" << entries_per_block
           << ";" << prefix << ".store_dtype=" << static_cast<int>(store_dtype)
           << ";" << prefix << ".block_size_bytes_override=" << block_size_bytes_override
           << ";" << prefix << ".block_size_bytes_alignment=" << block_size_bytes_alignment
           << ";" << prefix << ".block_size_alignment_min_entries=" << block_size_alignment_min_entries;
        return os.str();
    }

    std::string fingerprintExtra() const override {
        return opaqueFingerprintExtra("opaque");
    }

public:
    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "OpaqueKVCacheSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent + 2, ' ') << "entry_elems=" << entry_elems << "\n";
        os << std::string(indent + 2, ' ') << "entries_per_block=" << entries_per_block << "\n";
        os << std::string(indent + 2, ' ') << "block_size_bytes_override=" << block_size_bytes_override << "\n";
        os << std::string(indent + 2, ' ') << "block_size_bytes_alignment=" << block_size_bytes_alignment << "\n";
        os << std::string(indent + 2, ' ')
           << "block_size_alignment_min_entries=" << block_size_alignment_min_entries << "\n";
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }
};

struct CompressedKVCacheSpec: public OpaqueKVCacheSpec {
    uint32_t compression_ratio = 1;

    CompressedKVCacheSpec() {
        type      = KVCacheSpecType::OpaqueKV;
        lifecycle = CacheGroupType::FULL;
    }

    CompressedKVCacheSpec(std::string cache_tag,
                          uint32_t    entry_elements,
                          uint32_t    block_entries,
                          DataType    storage_dtype,
                          uint32_t    seq_size_per_blk,
                          uint32_t    cache_compression_ratio = 1,
                          size_t      block_size_alignment    = 0)
        : CompressedKVCacheSpec() {
        tag                        = std::move(cache_tag);
        entry_elems                = entry_elements;
        entries_per_block          = block_entries;
        compression_ratio          = cache_compression_ratio;
        store_dtype                = storage_dtype;
        block_size_bytes_alignment = block_size_alignment;

        local_head_num_kv  = 1;
        seq_size_per_block = seq_size_per_blk;
        dtype              = store_dtype;
    }

    KVCacheSpecPtr clone() const override {
        return std::make_shared<CompressedKVCacheSpec>(*this);
    }

protected:
    std::string fingerprintExtra() const override {
        std::ostringstream os;
        os << ";compressed_kv.compression_ratio=" << compression_ratio
           << opaqueFingerprintExtra("compressed_kv");
        return os.str();
    }

public:
    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "CompressedKVCacheSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent + 2, ' ') << "entry_elems=" << entry_elems << "\n";
        os << std::string(indent + 2, ' ') << "entries_per_block=" << entries_per_block << "\n";
        os << std::string(indent + 2, ' ') << "compression_ratio=" << compression_ratio << "\n";
        os << std::string(indent + 2, ' ') << "block_size_bytes_alignment=" << block_size_bytes_alignment << "\n";
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }
};

struct FixedStateCacheSpec: public OpaqueKVCacheSpec {
    uint32_t& state_dim;

    FixedStateCacheSpec(): state_dim(entry_elems) {
        type      = KVCacheSpecType::OpaqueState;
        lifecycle = CacheGroupType::SWA;
    }

    FixedStateCacheSpec(const FixedStateCacheSpec& other): OpaqueKVCacheSpec(other), state_dim(entry_elems) {}

    FixedStateCacheSpec& operator=(const FixedStateCacheSpec& other) {
        if (this != &other) {
            OpaqueKVCacheSpec::operator=(other);
        }
        return *this;
    }

    FixedStateCacheSpec(std::string cache_tag,
                        uint32_t    state_elements,
                        uint32_t    block_entries,
                        DataType    storage_dtype,
                        uint32_t    seq_size_per_blk,
                        size_t      block_size_bytes_override_value = 0,
                        size_t      block_size_alignment            = 0,
                        uint32_t    block_alignment_min_entries     = 0,
                        bool        state_cache                     = true,
                        bool        skip_reuse                      = false)
        : FixedStateCacheSpec() {
        tag                              = std::move(cache_tag);
        state_dim                        = state_elements;
        entries_per_block                = block_entries;
        store_dtype                      = storage_dtype;
        block_size_bytes_override        = block_size_bytes_override_value;
        block_size_bytes_alignment       = block_size_alignment;
        block_size_alignment_min_entries = block_alignment_min_entries;

        local_head_num_kv  = 1;
        seq_size_per_block = seq_size_per_blk;
        dtype              = store_dtype;
        is_state_cache     = state_cache;
        skip_prefix_reuse  = skip_reuse;
    }

    std::vector<BlockInfo> sliceBlockForPeer(std::vector<BlockInfo> parts,
                                             size_t                 cp_size,
                                             size_t                 peer_idx) const override {
        if (cp_size <= 1) {
            return parts;
        }
        RTP_LLM_CHECK_WITH_INFO(parts.size() == 1,
                                "FixedStateCacheSpec CP byte slicing expects one block part, got %zu",
                                parts.size());
        auto& block = parts[0];
        RTP_LLM_CHECK_WITH_INFO(block.addr != nullptr, "FixedStateCacheSpec CP byte slicing got null block addr");

        size_t slice_bytes = 0;
        if (block_size_bytes_override > 0 || block.size_bytes == block_size_bytes()) {
            RTP_LLM_CHECK_WITH_INFO(block.size_bytes % cp_size == 0,
                                    "FixedStateCacheSpec block bytes %zu not divisible by cp_size %zu",
                                    block.size_bytes,
                                    cp_size);
            slice_bytes = block.size_bytes / cp_size;
        } else {
            RTP_LLM_CHECK_WITH_INFO(entries_per_block % cp_size == 0,
                                    "FixedStateCacheSpec entries %u not divisible by cp_size %zu",
                                    entries_per_block,
                                    cp_size);
            const size_t local_entries = entries_per_block / cp_size;
            slice_bytes = local_entries * static_cast<size_t>(state_dim) * getTypeSize(store_dtype);
        }

        const size_t slice_offset = slice_bytes * peer_idx;
        RTP_LLM_CHECK_WITH_INFO(slice_offset + slice_bytes <= block.size_bytes,
                                "FixedStateCacheSpec CP slice [%zu, %zu) exceeds block bytes %zu",
                                slice_offset,
                                slice_offset + slice_bytes,
                                block.size_bytes);
        block.addr       = static_cast<void*>(static_cast<char*>(block.addr) + slice_offset);
        block.size_bytes = slice_bytes;
        return parts;
    }

    KVCacheSpecPtr clone() const override {
        return std::make_shared<FixedStateCacheSpec>(*this);
    }

protected:
    std::string fingerprintExtra() const override {
        return opaqueFingerprintExtra("fixed_state");
    }

public:
    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "FixedStateCacheSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent + 2, ' ') << "state_dim=" << state_dim << "\n";
        os << std::string(indent + 2, ' ') << "entries_per_block=" << entries_per_block << "\n";
        os << std::string(indent + 2, ' ') << "block_size_bytes_override=" << block_size_bytes_override << "\n";
        os << std::string(indent + 2, ' ') << "block_size_bytes_alignment=" << block_size_bytes_alignment << "\n";
        os << std::string(indent + 2, ' ')
           << "block_size_alignment_min_entries=" << block_size_alignment_min_entries << "\n";
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }
};

}  // namespace rtp_llm
