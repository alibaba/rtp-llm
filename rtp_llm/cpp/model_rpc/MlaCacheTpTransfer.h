#pragma once

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/all.h>

namespace rtp_llm {

// Wire/layout contract for K3 Prefill MLA cache dimension sharding.  The
// physical Prefill block is token-major [tokens, 576 / TP].  It must therefore
// be loaded into rank-local staging buffers and repacked into the Decode
// owner's token-major [tokens, 576] block; the two component ranges are not
// contiguous across tokens and cannot be represented by two pointer offsets.
struct K3MlaCacheTpLayout {
    static constexpr int kFullLatent   = 512;
    static constexpr int kFullSuffix   = 64;
    static constexpr int kFullWidth    = kFullLatent + kFullSuffix;
    static constexpr int kLayoutVersion = 1;

    explicit K3MlaCacheTpLayout(int shard_count): shard_count(shard_count) {
        if (shard_count <= 1 || kFullLatent % shard_count != 0 || kFullSuffix % shard_count != 0) {
            throw std::invalid_argument("K3 MLA cache TP shard_count must divide 512 and 64");
        }
    }

    int localLatent() const {
        return kFullLatent / shard_count;
    }
    int localSuffix() const {
        return kFullSuffix / shard_count;
    }
    int localWidth() const {
        return localLatent() + localSuffix();
    }

    void validateShard(const torch::Tensor& shard, int64_t token_count) const {
        if (!shard.defined() || shard.dim() != 2 || shard.size(0) != token_count
            || shard.size(1) != localWidth()) {
            throw std::invalid_argument("K3 MLA cache TP shard must be [tokens, 576 / TP]");
        }
        if (shard.scalar_type() != torch::kBFloat16) {
            throw std::invalid_argument("K3 MLA cache TP P->D fan-in currently supports BF16 only");
        }
        if (!shard.is_contiguous()) {
            throw std::invalid_argument("K3 MLA cache TP wire shard must be contiguous");
        }
    }

    void reconstruct(const std::vector<torch::Tensor>& rank_major_shards, torch::Tensor destination) const {
        if (static_cast<int>(rank_major_shards.size()) != shard_count) {
            throw std::invalid_argument("K3 MLA cache TP fan-in is missing one or more rank shards");
        }
        if (!destination.defined() || destination.dim() != 2 || destination.size(1) != kFullWidth
            || destination.scalar_type() != torch::kBFloat16 || !destination.is_contiguous()) {
            throw std::invalid_argument("K3 MLA cache TP destination must be contiguous BF16 [tokens, 576]");
        }
        const int64_t token_count = destination.size(0);
        for (int rank = 0; rank < shard_count; ++rank) {
            const auto& shard = rank_major_shards[rank];
            validateShard(shard, token_count);
            destination.narrow(1, rank * localLatent(), localLatent())
                .copy_(shard.narrow(1, 0, localLatent()));
            destination.narrow(1, kFullLatent + rank * localSuffix(), localSuffix())
                .copy_(shard.narrow(1, localLatent(), localSuffix()));
        }
    }

    int shard_count;
};

struct K3MlaCacheTpPeerPlan {
    std::vector<std::string> kda_peer_addrs;
    std::vector<std::string> mla_peer_addrs;
};

inline K3MlaCacheTpPeerPlan makeK3MlaCacheTpPeerPlan(const std::vector<std::string>& ordered_prefill_peers,
                                                      int                             decode_worker_rank,
                                                      int                             owner_rank) {
    if (ordered_prefill_peers.size() <= 1) {
        throw std::invalid_argument("K3 MLA cache TP fan-in requires more than one ordered Prefill peer");
    }
    const int world_size = static_cast<int>(ordered_prefill_peers.size());
    if (decode_worker_rank < 0 || decode_worker_rank >= world_size || owner_rank < 0 || owner_rank >= world_size) {
        throw std::invalid_argument("K3 MLA cache TP worker/owner rank is outside the ordered peer list");
    }
    K3MlaCacheTpPeerPlan plan;
    // KDA ownership is unchanged: Prefill rank i -> Decode rank i.
    plan.kda_peer_addrs.push_back(ordered_prefill_peers[decode_worker_rank]);
    // Only the stable request owner receives the eight MLA dimension shards.
    if (decode_worker_rank == owner_rank) {
        plan.mla_peer_addrs = ordered_prefill_peers;
    }
    return plan;
}

inline bool k3MlaCacheTpEnvEnabled() {
    const char* raw = std::getenv("KIMI_K3_MLA_CACHE_TP");
    if (raw == nullptr || std::strcmp(raw, "0") == 0) {
        return false;
    }
    if (std::strcmp(raw, "1") == 0) {
        return true;
    }
    throw std::invalid_argument("KIMI_K3_MLA_CACHE_TP must be 0 or 1");
}

}  // namespace rtp_llm
