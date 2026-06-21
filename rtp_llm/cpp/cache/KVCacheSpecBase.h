#pragma once

#include <memory>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"

namespace rtp_llm {

// Physical signature used to determine whether two KVCacheSpec instances can
// share the same KVCacheGroup and BlockPool. Two specs with identical tags AND
// identical SpecPhysicalSignature are merged into a single group; different tags
// always produce different groups regardless of physical equality.
struct SpecPhysicalSignature {
    size_t         block_size_bytes       = 0;
    size_t         scale_block_size_bytes = 0;
    CacheGroupType lifecycle_type         = CacheGroupType::FULL;
    rtp_llm::DataType dtype               = rtp_llm::DataType::TYPE_INVALID;

    bool operator==(const SpecPhysicalSignature& other) const {
        return block_size_bytes       == other.block_size_bytes
            && scale_block_size_bytes == other.scale_block_size_bytes
            && lifecycle_type         == other.lifecycle_type
            && dtype                  == other.dtype;
    }
    bool operator!=(const SpecPhysicalSignature& other) const {
        return !(*this == other);
    }
};

enum KVCacheSpecType {
    MultiHeadAttention,        // MHAKVCacheSpec: standard multi-head attention KV cache
    MultiHeadLatentAttention,  // MLAKVCacheSpec: MLA compressed latent KV cache
    LinearAttention,           // LinearKVCacheSpec: linear / SSM attention state cache
    OpaqueKV,                  // Byte-addressed opaque paged KV pool
    OpaqueState,               // Fixed-allocation opaque state / SWA-like pool
};

inline const char* KVCacheSpecTypeToString(KVCacheSpecType t) {
    switch (t) {
        case KVCacheSpecType::MultiHeadAttention:
            return "MultiHeadAttention";
        case KVCacheSpecType::MultiHeadLatentAttention:
            return "MultiHeadLatentAttention";
        case KVCacheSpecType::LinearAttention:
            return "LinearAttention";
        case KVCacheSpecType::OpaqueKV:
            return "OpaqueKV";
        case KVCacheSpecType::OpaqueState:
            return "OpaqueState";
        default:
            return "Unknown";
    }
}

struct KVCacheSpec;
using KVCacheSpecPtr = std::shared_ptr<KVCacheSpec>;

struct KVCacheSpec {
    std::string tag;
    std::vector<int> layers;
    uint32_t local_head_num_kv = 1;
    uint32_t seq_size_per_block = 1;
    bool     is_state_cache = false;
    bool     skip_prefix_reuse = false;

    // Lifecycle governs the allocation strategy for this cache group.
    // Each concrete spec subclass sets this in its constructor; do NOT set it
    // manually from outside the spec class hierarchy.
    //   FULL    - standard paged allocation (MHA, MLA, OpaqueKV)
    //   LINEAR  - fixed-capacity ring buffer (LinearAttention / SSM state)
    //   SWA     - fixed-size tail-allocation pool (DSV4 state / SWA_KV)
    CacheGroupType   lifecycle = CacheGroupType::FULL;

    KVCacheSpecType   type = KVCacheSpecType::MultiHeadAttention;
    rtp_llm::DataType dtype = rtp_llm::DataType::TYPE_INVALID;

    // Derived from lifecycle; true when this spec uses SWA-style fixed allocation.
    bool isFixedCache() const { return lifecycle == CacheGroupType::SWA; }

    virtual size_t block_size() const   = 0;
    virtual size_t k_block_size() const = 0;
    virtual size_t v_block_size() const = 0;

    virtual size_t block_size_bytes() const   = 0;
    virtual size_t k_block_size_bytes() const = 0;
    virtual size_t v_block_size_bytes() const = 0;

    virtual size_t scale_block_size_bytes() const {
        return 0;
    }
    virtual size_t k_scale_block_size_bytes() const {
        return 0;
    }
    virtual size_t v_scale_block_size_bytes() const {
        return 0;
    }

    virtual KVCacheSpecPtr clone() const = 0;

    virtual std::vector<BlockInfo> sliceBlockForPeer(std::vector<BlockInfo> parts,
                                                     size_t                 cp_size,
                                                     size_t                 peer_idx) const {
        (void)cp_size;
        (void)peer_idx;
        return parts;
    }

    std::string fingerprint() const {
        std::ostringstream os;
        os << "tag=" << tag << ";type=" << static_cast<int>(type) << ";dtype=" << static_cast<int>(dtype)
           << ";local_head_num_kv=" << local_head_num_kv << ";seq_size_per_block=" << seq_size_per_block;
        os << fingerprintExtra();
        return os.str();
    }

    virtual std::string debugString(size_t indent = 0) const = 0;

    // Returns the physical signature used for spec grouping.
    // Two specs with the same (tag, physicalSignature()) are merged into one
    // KVCacheGroup. lifecycle is a direct field — no switch needed.
    // LinearKVCacheSpec overrides to encode its dual-dtype block layout.
    virtual SpecPhysicalSignature physicalSignature() const {
        return {block_size_bytes(), scale_block_size_bytes(), lifecycle, dtype};
    }

protected:
    virtual std::string fingerprintExtra() const {
        return "";
    }

    // Helper method to generate common parts of debug string
    std::string commonDebugString(size_t indent = 0) const {
        const std::string indent_str = std::string(indent, ' ');
        const std::string indent1    = indent_str + "  ";

        std::ostringstream os;
        os << indent1 << "tag=" << tag << "\n";
        os << indent1 << "type=" << KVCacheSpecTypeToString(type) << "(" << static_cast<int>(type) << ")\n";
        os << indent1 << "dtype=" << static_cast<int>(dtype) << "\n";
        os << indent1 << "layers.size=" << layers.size() << "\n";
        os << indent1 << "local_head_num_kv=" << local_head_num_kv << "\n";
        os << indent1 << "seq_size_per_block=" << seq_size_per_block << "\n";
        os << indent1 << "is_state_cache=" << (is_state_cache ? "true" : "false") << "\n";
        os << indent1 << "is_fixed_cache=" << (isFixedCache() ? "true" : "false") << "\n";
        os << indent1 << "skip_prefix_reuse=" << (skip_prefix_reuse ? "true" : "false") << "\n";
        os << indent1 << "block_size=" << block_size() << "\n";
        os << indent1 << "k_block_size=" << k_block_size() << "\n";
        os << indent1 << "v_block_size=" << v_block_size() << "\n";
        os << indent1 << "block_size_bytes=" << block_size_bytes() << "\n";
        os << indent1 << "k_block_size_bytes=" << k_block_size_bytes() << "\n";
        os << indent1 << "v_block_size_bytes=" << v_block_size_bytes() << "\n";
        return os.str();
    }
};

}  // namespace rtp_llm
