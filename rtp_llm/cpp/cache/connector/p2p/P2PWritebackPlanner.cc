#include "rtp_llm/cpp/cache/connector/p2p/P2PWritebackPlanner.h"

#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/PipelineLayoutCodec.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/RouteCodec.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h"

#include <algorithm>

namespace rtp_llm {

std::shared_ptr<const PlanResult> P2PWritebackPlanner::planFor(
    int tp, int cp, const std::string& layout, int workers, const std::string& unique_key) {
    const int    source_tp = role_ == RoleType::DECODE ? config_.parallelism_config.tp_size : tp;
    const size_t offset    = KVCacheTransferPlanner::sourceReplicaOffset(unique_key, source_tp);
    auto         key       = std::make_tuple(tp, cp, offset, layout, workers);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
    }
    auto result = std::make_shared<const PlanResult>(buildPlan(tp, cp, layout, workers, unique_key, offset));
    if (!result->ok()) {
        return result;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    constexpr size_t            kMaxPlanCacheEntries = 64;
    if (cache_.size() >= kMaxPlanCacheEntries && cache_.find(key) == cache_.end()) {
        cache_.erase(cache_.begin());
    }
    return cache_.emplace(std::move(key), result).first->second;
}

PlanResult P2PWritebackPlanner::buildPlan(int                peer_tp,
                                          int                peer_cp,
                                          const std::string& peer_layout,
                                          int                peer_workers,
                                          const std::string& unique_key,
                                          size_t             source_replica_offset) const {
    const auto& pc        = config_.parallelism_config;
    const auto  peer_role = role_ == RoleType::DECODE ? RoleType::PREFILL : RoleType::DECODE;
    if (pc.tp_size <= 0 || peer_tp <= 0 || (peer_cp != 1 && peer_cp != peer_tp)) {
        return {{}, ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid writeback TP or CP width")};
    }
    std::vector<std::string> reusable_tags;
    std::vector<std::string> excluded_tags;
    for (const auto& group : config_.topology->groups()) {
        (cacheGroupParticipatesInBlockTree(group.policy) ? reusable_tags : excluded_tags).push_back(group.tag);
    }
    if (reusable_tags.empty()) {
        return {{}, ErrorInfo(ErrorCode::INVALID_PARAMS, "no reusable cache groups for writeback")};
    }
    if (!peer_layout.empty()) {
        auto local = decodeP2PPipelineLayout(config_.pipeline_layout, role_);
        auto peer  = decodeP2PPipelineLayout(peer_layout, peer_role);
        if (!local || !peer || (peer_workers && peer->worker_count != static_cast<size_t>(peer_workers))
            || peer->stages.front().main_shards.rankCount() != peer_tp
            || (peer_role == RoleType::PREFILL && peer->stages.front().main_shards.cpSize() != peer_cp)) {
            return {{}, ErrorInfo(ErrorCode::INVALID_PARAMS, "inconsistent writeback pipeline layout")};
        }
        // Both roles use the same Decode -> Prefill layer intersections and replica selection.
        const auto& source = role_ == RoleType::DECODE ? *local : *peer;
        const auto& target = role_ == RoleType::DECODE ? *peer : *local;
        auto        result = KVCacheTransferPlanner::planStages(source, target, unique_key, excluded_tags);
        if (!result.ok()) {
            return result;
        }
        if (result.plan.routes.empty()) {
            return {{}, ErrorInfo(ErrorCode::INVALID_PARAMS, "no reusable layer routes for writeback")};
        }
        // A stage-local group needs a root block table before its target KV can be published.
        for (const auto& route : result.plan.routes) {
            if (std::find(reusable_tags.begin(), reusable_tags.end(), route.cache_tag) == reusable_tags.end()) {
                return {
                    {},
                    ErrorInfo(ErrorCode::INVALID_PARAMS, "missing writeback root block table for " + route.cache_tag)};
            }
        }
        return result;
    }
    if (pc.pp_size != 1 || (peer_workers && peer_workers != peer_tp)) {
        return {{}, ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback requires peer stage metadata for PP")};
    }
    // TP-only peers from before the pipeline handshake advertise just their physical CP width.
    auto local = ShardLayoutFactory::fromTopology(*config_.topology, pc, role_, config_.cache_key_tokens_per_block);
    local.use_group_cache_transfer_policy = config_.use_group_cache_transfer_policy;
    local.use_mla                         = config_.use_mla;
    auto peer                             = ShardLayoutFactory::peerOf(
        local, peer_tp, role_ == RoleType::DECODE ? peer_cp > 1 : local.cpSize() > 1, peer_role);
    if (peer_role == RoleType::PREFILL && peer_cp == 1 && pc.prefill_cp_config.is_prefill_enabled()) {
        auto peer_pc                     = peer.pc;
        peer_pc.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
        peer                             = ShardLayout::forPeer(local, peer_pc, true);
    }
    return role_ == RoleType::DECODE ?
               KVCacheTransferPlanner::plan(local, peer, reusable_tags, source_replica_offset) :
               KVCacheTransferPlanner::plan(peer, local, reusable_tags, source_replica_offset);
}

ErrorInfo P2PWritebackPlanner::buildRankRoutes(const TransferPlan&             plan,
                                               KVCacheResource&                resource,
                                               size_t                          start,
                                               size_t                          count,
                                               int                             cp_size,
                                               P2PBroadcastClient::RankRoutes& routes,
                                               int64_t*                        bytes) const {
    if (bytes) {
        *bytes = 0;
    }
    const size_t key_count = resource.cacheKeys().size();
    if (start > key_count || count > key_count - start || cp_size <= 0) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid writeback key range or CP size");
    }
    const bool sending = role_ == RoleType::DECODE;
    for (const auto& route : plan.routes) {
        const int rank = sending ? route.src_rank : route.dst_rank;
        if (rank < 0 || static_cast<size_t>(rank) >= routes.size()) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback worker index out of range");
        }
        auto positions = KVCacheTransferPlanner::resolveKeys(route.src_keys, key_count);
        positions.erase(std::remove_if(positions.begin(),
                                       positions.end(),
                                       [&](size_t pos) { return pos < start || pos >= start + count; }),
                        positions.end());
        if (positions.empty()) {
            continue;
        }
        const auto&                              group  = config_.topology->group(route.cache_tag);
        const bool                               staged = !route.layers.empty();
        std::vector<TransferRoute::LayerMapping> tp_layers;
        if (!staged) {
            for (int id : config_.topology->layerIdsForGroup(group.tag)) {
                tp_layers.push_back({id, id, id});
            }
        }
        const auto&     layers = staged ? route.layers : tp_layers;
        TransferRoutePB pb;
        if (sending) {
            RouteCodec::encodeForPrefill(route, route.dst_rank, &pb);
        } else {
            RouteCodec::encodeForDecode(route, &pb);
        }
        for (const auto& layer : layers) {
            auto buffer =
                LayerCacheBufferUtil::convertLayerTagForRoute(resource,
                                                              group,
                                                              sending ? layer.src_local_id : layer.dst_local_id,
                                                              positions,
                                                              rank % cp_size,
                                                              cp_size,
                                                              config_.cache_key_tokens_per_block,
                                                              staged);
            if (!buffer.ok()) {
                return buffer.status();
            }
            if (!buffer.value() || buffer.value()->blockIdMap().size() != positions.size()) {
                return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback key projection incomplete");
            }
            auto* block = pb.add_layer_blocks();
            block->set_layer_id(layer.global_id);
            block->set_cache_tag(group.tag);
            for (const auto& [key, id] : buffer.value()->blockIdMap()) {
                block->add_cache_keys(key);
                block->add_block_ids(id);
            }
        }
        routes[rank].push_back(std::move(pb));
        if (bytes) {
            *bytes += (sending ? route.src_bytes : route.dst_bytes) * positions.size() * layers.size();
        }
    }
    return ErrorInfo::OkStatus();
}

}  // namespace rtp_llm
