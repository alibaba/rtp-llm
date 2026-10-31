#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

struct LoadTransferPlan {
    // A participating peer receives an empty RequestBlockBuffer when no page
    // is selected. Nonparticipating peers do not receive a group buffer.
    bool                peer_participates = true;
    std::vector<size_t> logical_pages;
};

struct LoadPeerRoute {
    int                 partition_count = 1;
    int                 partition_id    = 0;
    std::vector<size_t> peer_indices;
};

LoadPeerRoute projectLoadPeerRoute(int    producer_cp_size,
                                   size_t decode_worker_count,
                                   size_t prefill_peer_count,
                                   size_t decode_worker_index,
                                   bool   prefill_enabled,
                                   bool   mla_request);

// Content selection only. Cache maps each selected logical page to a local
// slot, key range, and physical parts. compact_tail is a cache-provided fact.
std::vector<size_t> selectProducerTransferPages(const CacheGroupPolicy& policy,
                                                size_t                  logical_page_count,
                                                bool                    use_hybrid,
                                                bool                    compact_tail);

// Pages before this boundary are outside the SWA history needed by the first
// decode step. Unlike a missing destination, they impose no transfer obligation.
size_t firstRequiredSwaHandoffPage(const CacheGroupPolicy& policy,
                                   size_t                  tokens_per_block,
                                   size_t                  input_token_count);

LoadTransferPlan selectLoadTransferPages(const CacheGroupPolicy& policy,
                                         size_t                  logical_page_count,
                                         size_t                  max_destination_slots,
                                         size_t                  first_page_after_reuse,
                                         bool                    use_hybrid,
                                         bool                    compact_tail,
                                         int                     producer_cp_size,
                                         int                     decode_peer_count,
                                         int                     peer_index);

// Reader routing remains a transfer policy. Selection is independent of the
// cache's local physical mapping of that page.
bool transferPageReadFromPeer(const CacheGroupPolicy& policy,
                              size_t                  logical_page,
                              bool                    use_hybrid,
                              int                     producer_cp_size,
                              int                     decode_peer_count,
                              int                     peer_index);

bool nonFullTransferPeerParticipates(
    const CacheGroupPolicy& policy, bool use_hybrid, int producer_cp_size, int decode_peer_count, int peer_index);

// Shared wire order and part-count contract for producer and decode.
const char* cacheTransferPartPrefix(bool combined_kv, size_t part_index, size_t part_count);

}  // namespace rtp_llm
