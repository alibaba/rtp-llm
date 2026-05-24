#include "rtp_llm/cpp/cache/connector/KVCacheHandshake.h"

#include <algorithm>
#include <sstream>

#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"

namespace rtp_llm {

namespace {

// FNV-1a 64-bit constants.
constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime       = 0x100000001b3ULL;

inline void fnvMix64(uint64_t& h, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        h ^= (v & 0xffULL);
        h *= kFnvPrime;
        v >>= 8;
    }
}

inline void fnvMix32(uint64_t& h, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        h ^= (v & 0xffu);
        h *= kFnvPrime;
        v >>= 8;
    }
}

}  // namespace

std::string HandshakeInfo::toString() const {
    std::ostringstream os;
    os << "HandshakeInfo{protocol_magic=" << protocol_magic << ", pool_descriptor_hash=0x" << std::hex
       << pool_descriptor_hash << std::dec << ", hash_salt_version=" << hash_salt_version
       << ", hash_salt_nonzero_bitmap=0x" << std::hex << hash_salt_nonzero_bitmap << std::dec << "}";
    return os.str();
}

std::vector<PoolDescriptorHashInput> poolDescriptorHashInputsFor(const CacheConfig& cache_config) {
    std::vector<PoolDescriptorHashInput> out;
    const size_t                         pool_count = cache_config.group_types.size();
    out.reserve(pool_count);
    for (size_t p = 0; p < pool_count; ++p) {
        PoolDescriptorHashInput in{};
        in.pool_id    = static_cast<uint32_t>(p);
        in.group_type = static_cast<uint32_t>(cache_config.group_types[p]);
        if (p < cache_config.group_region_names.size()) {
            in.region_name = static_cast<uint32_t>(cache_config.group_region_names[p]);
        } else {
            in.region_name = static_cast<uint32_t>(KVCacheRegionName::DEFAULT);
        }
        if (p < cache_config.group_block_size_bytes.size()) {
            in.group_block_size_bytes = static_cast<uint64_t>(cache_config.group_block_size_bytes[p]);
        }
        in.layer_count    = cache_config.layer_all_num;
        in.tma_alignment  = static_cast<uint32_t>(DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES);
        out.push_back(in);
    }
    return out;
}

uint64_t hashPoolDescriptorInputs(const std::vector<PoolDescriptorHashInput>& inputs) {
    uint64_t h = kFnvOffsetBasis;
    // Seed with pool count so 0-pool vs 1-pool-empty differ.
    fnvMix32(h, static_cast<uint32_t>(inputs.size()));
    for (const auto& in : inputs) {
        fnvMix32(h, in.pool_id);
        fnvMix32(h, in.group_type);
        fnvMix32(h, in.region_name);
        fnvMix64(h, in.group_block_size_bytes);
        fnvMix32(h, in.layer_count);
        fnvMix32(h, in.tma_alignment);
    }
    return h;
}

HandshakeInfo computeLocalHandshakeInfo(const CacheConfig& cache_config,
                                        uint32_t           hash_salt_version,
                                        uint32_t           hash_salt_nonzero_bitmap) {
    HandshakeInfo info{};
    // Legacy path: protocol_magic = 0; salt/hash forced to zero so legacy
    // peers hand-shake trivially regardless of CacheConfig contents.
    if (!cache_config.super_block_layout.enabled) {
        return info;
    }
    info.protocol_magic           = 1;  // unified-aware (bps=1 wire schema)
    info.pool_descriptor_hash     = hashPoolDescriptorInputs(poolDescriptorHashInputsFor(cache_config));
    info.hash_salt_version        = hash_salt_version;
    info.hash_salt_nonzero_bitmap = hash_salt_nonzero_bitmap;
    return info;
}

bool validateHandshake(const HandshakeInfo& local, const HandshakeInfo& peer, std::string* error_message) {
    auto fail = [&](const std::string& reason) {
        if (error_message) {
            std::ostringstream os;
            os << "PD pair handshake REFUSED: " << reason << "; local=" << local.toString()
               << ", peer=" << peer.toString();
            *error_message = os.str();
        }
        return false;
    };

    // legacy↔legacy: both magic=0; insist all fields zero to catch a
    // misconfigured legacy peer that leaked a nonzero salt by accident.
    if (local.protocol_magic == 0 && peer.protocol_magic == 0) {
        if (local.pool_descriptor_hash != 0 || peer.pool_descriptor_hash != 0 || local.hash_salt_version != 0
            || peer.hash_salt_version != 0 || local.hash_salt_nonzero_bitmap != 0
            || peer.hash_salt_nonzero_bitmap != 0) {
            return fail("legacy↔legacy pair carries nonzero handshake fields (unexpected leak)");
        }
        return true;
    }

    // legacy↔unified: REFUSE outright.  The receiver-side per-request magic-
    // byte negotiation (§4.3) does NOT protect against the silent reuse-miss
    // hazard at the cache_key layer (Risk 9.6) because RDMA routes by opaque
    // key string only.  Force the operator to land both sides at the same
    // unified version.
    if (local.protocol_magic == 0 || peer.protocol_magic == 0) {
        return fail("mixed-mode pair (one side legacy, one side unified) — refuse to prevent silent reuse-miss");
    }

    // unified↔unified: insist on byte-for-byte agreement on the pinned hash
    // and salt schema.  Per REQ-D2 the pinned hash inputs exclude additive
    // PR-4+ fields, so additive evolution does not force a fleet-wide deploy.
    if (local.pool_descriptor_hash != peer.pool_descriptor_hash) {
        return fail("pool_descriptor_hash mismatch (REQ-D2)");
    }
    if (local.hash_salt_version != peer.hash_salt_version) {
        return fail("hash_salt_version mismatch (REQ-D1)");
    }
    if (local.hash_salt_nonzero_bitmap != peer.hash_salt_nonzero_bitmap) {
        return fail("hash_salt_nonzero_bitmap mismatch (REQ-D1)");
    }
    // protocol_magic intentionally allowed to differ if both sides nonzero —
    // future PR-5 may add magic=2 etc., handshake downgrades to common
    // subset.  Today both must be 1; downgrade is a no-op.
    return true;
}

}  // namespace rtp_llm
