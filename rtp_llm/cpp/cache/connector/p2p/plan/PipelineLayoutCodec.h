#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

#include <optional>
#include <utility>

namespace rtp_llm {

inline void encodeP2PShardLayout(const ShardLayout& layout, P2PShardLayoutPB* pb) {
    const auto& cp = layout.pc.prefill_cp_config;
    pb->set_tp_size(layout.pc.tp_size);
    pb->set_cp_method(static_cast<int>(cp.method));
    pb->set_kv_cache_sharded(cp.kv_cache_sharded);
    pb->set_prefill_cp_size(cp.prefill_cp_size);
    pb->set_cache_key_tokens_per_block(layout.cache_key_tokens_per_block);
    pb->set_use_group_cache_transfer_policy(layout.use_group_cache_transfer_policy);
    pb->set_use_mla(layout.use_mla);
    for (const auto& [tag, group] : layout.groups) {
        auto* g = pb->add_groups();
        g->set_tag(tag);
        g->set_group_type(static_cast<int>(group.policy.group_type));
        g->set_spec_type(group.spec_type);
        g->set_cp_mapping(static_cast<int>(group.policy.cp_mapping));
        g->set_cp_slice(static_cast<int>(group.policy.cp_slice));
        g->set_active_tail_blocks(group.policy.active_tail_blocks);
        g->set_kv_block_stride_bytes(group.kv_block_stride_bytes);
        g->set_kv_scale_stride_bytes(group.kv_scale_stride_bytes);
        g->set_k_block_payload_bytes(group.k_block_payload_bytes);
        g->set_seq_size_per_block(group.seq_size_per_block);
        g->set_pre_sliced(group.pre_sliced);
        g->set_global_kv_heads(group.global_kv_heads);
    }
}

inline bool decodeP2PShardLayout(const P2PShardLayoutPB& pb, RoleType role, ShardLayout& layout) {
    if (pb.tp_size() <= 0 || pb.cp_method() < 0 || pb.cp_method() > static_cast<int>(CPRotateMethod::UNKNOWN)
        || pb.prefill_cp_size() < 0 || pb.cache_key_tokens_per_block() == 0) {
        return false;
    }
    layout.pc.tp_size                            = pb.tp_size();
    layout.pc.role_type                          = role;
    layout.pc.prefill_cp_config.method           = static_cast<CPRotateMethod>(pb.cp_method());
    layout.pc.prefill_cp_config.kv_cache_sharded = pb.kv_cache_sharded();
    layout.pc.prefill_cp_config.prefill_cp_size  = pb.prefill_cp_size();
    layout.cache_key_tokens_per_block            = pb.cache_key_tokens_per_block();
    layout.use_group_cache_transfer_policy       = pb.use_group_cache_transfer_policy();
    layout.use_mla                               = pb.use_mla();
    for (const auto& g : pb.groups()) {
        if (g.tag().empty() || g.group_type() < 0 || g.group_type() > static_cast<int>(CacheGroupType::SWA)
            || g.spec_type() < 0 || g.spec_type() > KVCacheSpecType::OpaqueState || g.cp_mapping() < 0
            || g.cp_mapping() > static_cast<int>(CpBlockMappingMode::COMPACT_LAST_RANK) || g.cp_slice() < 0
            || g.cp_slice() > static_cast<int>(CpBlockSliceMode::PAYLOAD_BYTES)) {
            return false;
        }
        ShardLayout::GroupLayout group;
        group.policy.group_type         = static_cast<CacheGroupType>(g.group_type());
        group.spec_type                 = static_cast<KVCacheSpecType>(g.spec_type());
        group.policy.cp_mapping         = static_cast<CpBlockMappingMode>(g.cp_mapping());
        group.policy.cp_slice           = static_cast<CpBlockSliceMode>(g.cp_slice());
        group.policy.active_tail_blocks = g.active_tail_blocks();
        group.kv_block_stride_bytes     = g.kv_block_stride_bytes();
        group.kv_scale_stride_bytes     = g.kv_scale_stride_bytes();
        group.k_block_payload_bytes     = g.k_block_payload_bytes();
        group.seq_size_per_block        = g.seq_size_per_block();
        group.pre_sliced                = g.pre_sliced();
        group.global_kv_heads           = g.global_kv_heads();
        if (!layout.groups.emplace(g.tag(), std::move(group)).second) {
            return false;
        }
    }
    layout.deriveHeadShardCounts();
    return true;
}

inline std::optional<PipelineTransferLayout> decodeP2PPipelineLayout(const std::string& payload, RoleType role) {
    P2PPipelineLayoutPB pb;
    if (!pb.ParseFromString(payload) || pb.stages().empty()) {
        return std::nullopt;
    }
    PipelineTransferLayout layout;
    layout.worker_count = pb.worker_count();
    for (const auto& metadata : pb.stages()) {
        auto& stage            = layout.stages.emplace_back();
        stage.main_layer_begin = metadata.main_layer_begin();
        stage.main_layer_count = metadata.main_layer_count();
        stage.worker_indices.assign(metadata.worker_indices().begin(), metadata.worker_indices().end());
        for (const auto& layer : metadata.layers()) {
            stage.layer_tags.emplace_back(layer.tags().begin(), layer.tags().end());
        }
        if (!metadata.has_main_shards() || !decodeP2PShardLayout(metadata.main_shards(), role, stage.main_shards)
            || (metadata.has_draft_shards() && !decodeP2PShardLayout(metadata.draft_shards(), role, stage.draft_shards))
            || (metadata.layers_size() > stage.main_layer_count && !metadata.has_draft_shards())) {
            return std::nullopt;
        }
    }
    return layout;
}

inline PlanResult planP2PPipelineTransfer(const std::string& prefill_payload,
                                          const std::string& decode_payload,
                                          const std::string& unique_key) {
    const auto prefill = decodeP2PPipelineLayout(prefill_payload, RoleType::PREFILL);
    const auto decode  = decodeP2PPipelineLayout(decode_payload, RoleType::DECODE);
    if (!prefill || !decode) {
        return {{},
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                          "missing or invalid P2P PP peer layout")};
    }
    return KVCacheTransferPlanner::planStages(*prefill, *decode, unique_key);
}

}  // namespace rtp_llm
