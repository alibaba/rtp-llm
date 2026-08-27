#pragma once

#include <cstddef>
#include <stdexcept>

namespace rtp_llm {

struct K3PdPartitionPlan {
    int prefill_peer_index;
    // Select a slice of the Prefill rank's 12-head KDA source block. The
    // Decode destination is already sized for its local KTP heads and must
    // therefore remain an unpartitioned destination buffer.
    int remote_kda_partition_count;
    int remote_kda_partition_id;
    int local_kda_heads;
};

struct K3PdDestinationPlan {
    int partition_count;
    int partition_id;
};

// The wire partition selects a slice at the Prefill source. Decode storage is
// already allocated with 96 / KTP local heads, so the receive destination is
// always the whole local block.
inline K3PdDestinationPlan makeK3PdDestinationPlan(int remote_partition_count, int remote_partition_id) {
    if ((remote_partition_count != 1 && remote_partition_count != 2) || remote_partition_id < 0
        || remote_partition_id >= remote_partition_count) {
        throw std::invalid_argument("invalid K3 PD remote KDA partition");
    }
    return {1, 0};
}

// K3 Prefill publishes eight ordered TP peers. MLA is replicated on every
// peer, while the 96 KDA heads are partitioned by Decode KTP. Decode KTP may
// equal Prefill TP (8->8) or be its only supported expansion (8->16).
inline K3PdPartitionPlan makeK3PdPartitionPlan(int    prefill_attention_tp,
                                               int    decode_attention_tp,
                                               int    decode_kda_tp,
                                               size_t ordered_prefill_peer_count,
                                               int    decode_worker_rank,
                                               int    total_kda_heads = 96) {
    constexpr int kK3PrefillTp     = 8;
    if (prefill_attention_tp != kK3PrefillTp) {
        throw std::invalid_argument("K3 PD requires Prefill attention TP8");
    }
    if (static_cast<int>(ordered_prefill_peer_count) != prefill_attention_tp) {
        throw std::invalid_argument("K3 PD ordered Prefill peer count must equal Prefill attention TP");
    }
    if (decode_attention_tp != 1 && decode_attention_tp != prefill_attention_tp) {
        throw std::invalid_argument("K3 PD Decode attention TP must be 1 or equal Prefill TP");
    }
    if (decode_kda_tp != 8 && decode_kda_tp != 16) {
        throw std::invalid_argument("K3 PD formally supports Decode KTP8 or KTP16 only");
    }
    if (decode_kda_tp < prefill_attention_tp || decode_kda_tp % prefill_attention_tp != 0) {
        throw std::invalid_argument("K3 PD Decode KTP must be an integer multiple of Prefill TP");
    }
    if (total_kda_heads <= 0 || total_kda_heads % decode_kda_tp != 0) {
        throw std::invalid_argument("K3 PD KDA heads are not divisible by Decode KTP");
    }
    if (decode_worker_rank < 0 || decode_worker_rank >= decode_kda_tp) {
        throw std::invalid_argument("K3 PD Decode worker rank is outside the KTP group");
    }
    const int partition_count = decode_kda_tp / prefill_attention_tp;
    return {decode_worker_rank / partition_count,
            partition_count,
            decode_worker_rank % partition_count,
            total_kda_heads / decode_kda_tp};
}

inline bool isK3PdTopologySupported(int    prefill_attention_tp,
                                    int    decode_attention_tp,
                                    int    decode_kda_tp,
                                    size_t ordered_prefill_peer_count) {
    try {
        (void)makeK3PdPartitionPlan(prefill_attention_tp,
                                    decode_attention_tp,
                                    decode_kda_tp,
                                    ordered_prefill_peer_count,
                                    0);
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

}  // namespace rtp_llm
