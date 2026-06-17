#pragma once

#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"

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

}  // namespace rtp_llm
