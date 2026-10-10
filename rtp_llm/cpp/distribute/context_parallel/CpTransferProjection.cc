#include "rtp_llm/cpp/distribute/context_parallel/CpTransferProjection.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace rtp_llm {

LoadPeerRoute projectLoadPeerRoute(int    producer_cp_size,
                                   size_t decode_worker_count,
                                   size_t prefill_peer_count,
                                   size_t decode_worker_index,
                                   bool   prefill_enabled,
                                   bool   mla_request) {
    if (producer_cp_size <= 0 || decode_worker_count == 0 || prefill_peer_count == 0
        || decode_worker_index >= decode_worker_count
        || decode_worker_count > static_cast<size_t>(std::numeric_limits<int>::max())
        || prefill_peer_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("decode load route has invalid worker or peer topology");
    }
    LoadPeerRoute route;
    if (producer_cp_size > 1) {
        for (size_t peer = 0; peer < prefill_peer_count; ++peer) {
            route.peer_indices.push_back(peer);
        }
    } else if (mla_request) {
        if (decode_worker_count % prefill_peer_count == 0) {
            route.peer_indices.push_back(decode_worker_index / (decode_worker_count / prefill_peer_count));
        } else {
            route.peer_indices.push_back(decode_worker_index * (prefill_peer_count / decode_worker_count));
        }
    } else if (prefill_enabled) {
        route.partition_count = static_cast<int>(decode_worker_count);
        route.partition_id    = static_cast<int>(decode_worker_index % decode_worker_count);
        route.peer_indices.push_back(decode_worker_index % prefill_peer_count);
    } else if (decode_worker_count % prefill_peer_count == 0) {
        const size_t partition_count = decode_worker_count / prefill_peer_count;
        route.partition_count        = static_cast<int>(partition_count);
        route.partition_id           = static_cast<int>(decode_worker_index % partition_count);
        route.peer_indices.push_back(decode_worker_index / partition_count);
    } else {
        const size_t group_count = prefill_peer_count / decode_worker_count;
        for (size_t peer = 0; peer < group_count; ++peer) {
            route.peer_indices.push_back(decode_worker_index * group_count + peer);
        }
    }
    return route;
}

bool nonFullTransferPeerParticipates(
    const CacheGroupPolicy& policy, bool use_hybrid, int producer_cp_size, int decode_peer_count, int peer_index) {
    if (policy.group_type == CacheGroupType::FULL || producer_cp_size <= 0 || decode_peer_count <= 0 || peer_index < 0
        || peer_index >= decode_peer_count) {
        throw std::invalid_argument("non-FULL peer participation has invalid group or peer topology");
    }
    const bool page_level_rr = producer_cp_size > 1 && decode_peer_count == producer_cp_size;
    return !page_level_rr || !use_hybrid || policy.cp_slice != CpBlockSliceMode::NONE || peer_index == 0;
}

std::vector<size_t> selectProducerTransferPages(const CacheGroupPolicy& policy,
                                                size_t                  logical_page_count,
                                                bool                    use_hybrid,
                                                bool                    compact_tail) {
    size_t first_page = 0;
    if (policy.group_type != CacheGroupType::FULL && use_hybrid && (compact_tail || policy.active_tail_blocks > 0)) {
        const size_t tail_count = std::max<size_t>(1, policy.active_tail_blocks);
        if (logical_page_count > tail_count) {
            first_page = logical_page_count - tail_count;
        }
    }
    std::vector<size_t> pages(logical_page_count - first_page);
    std::iota(pages.begin(), pages.end(), first_page);
    return pages;
}

LoadTransferPlan selectLoadTransferPages(const CacheGroupPolicy& policy,
                                         size_t                  logical_page_count,
                                         size_t                  max_destination_slots,
                                         size_t                  first_page_after_reuse,
                                         bool                    use_hybrid,
                                         bool                    compact_tail,
                                         int                     producer_cp_size,
                                         int                     decode_peer_count,
                                         int                     peer_index) {
    if (producer_cp_size <= 0 || decode_peer_count <= 0 || peer_index < 0 || peer_index >= decode_peer_count) {
        throw std::invalid_argument("decode transfer has invalid peer topology");
    }
    LoadTransferPlan plan;
    if (policy.group_type != CacheGroupType::FULL) {
        plan.peer_participates =
            nonFullTransferPeerParticipates(policy, use_hybrid, producer_cp_size, decode_peer_count, peer_index);
        if (!plan.peer_participates) {
            return plan;
        }
    }
    size_t first_page = use_hybrid ? 0 : std::min(first_page_after_reuse, logical_page_count);
    if (policy.group_type != CacheGroupType::FULL && use_hybrid && (compact_tail || policy.active_tail_blocks > 0)) {
        const size_t tail_count = std::max<size_t>(1, policy.active_tail_blocks);
        if (logical_page_count > tail_count) {
            first_page = logical_page_count - tail_count;
        }
    }
    const size_t page_count = std::min(logical_page_count, max_destination_slots);
    if (first_page < page_count) {
        plan.logical_pages.resize(page_count - first_page);
        std::iota(plan.logical_pages.begin(), plan.logical_pages.end(), first_page);
    }
    return plan;
}

bool transferPageReadFromPeer(const CacheGroupPolicy& policy,
                              size_t                  logical_page,
                              bool                    use_hybrid,
                              int                     producer_cp_size,
                              int                     decode_peer_count,
                              int                     peer_index) {
    if (producer_cp_size <= 0 || decode_peer_count <= 0 || peer_index < 0 || peer_index >= decode_peer_count) {
        throw std::invalid_argument("transfer reader has invalid peer topology");
    }
    const bool page_level_rr = producer_cp_size > 1 && decode_peer_count == producer_cp_size;
    if (page_level_rr && (policy.group_type == CacheGroupType::FULL || !use_hybrid)) {
        return logical_page % static_cast<size_t>(producer_cp_size) == static_cast<size_t>(peer_index);
    }
    return true;
}

const char* cacheTransferPartPrefix(bool combined_kv, size_t part_index, size_t part_count) {
    if (combined_kv) {
        if ((part_count != 1 && part_count != 2) || part_index >= part_count) {
            throw std::invalid_argument("combined KV transfer requires one KV part and optional scale part");
        }
        return part_index == 0 ? "kv_" : "kv_scale_";
    }
    if ((part_count != 2 && part_count != 4) || part_index >= part_count) {
        throw std::invalid_argument("split KV transfer requires K/V parts and optional K/V scales");
    }
    static constexpr const char* prefixes[] = {"k_", "v_", "k_scale_", "v_scale_"};
    return prefixes[part_index];
}

}  // namespace rtp_llm
