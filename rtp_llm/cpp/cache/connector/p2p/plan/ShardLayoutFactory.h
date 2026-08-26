#pragma once

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayout.h"
#include "rtp_llm/cpp/config/RoleTypes.h"

#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

/// @brief 从真实的 CacheTopology + ParallelismConfig 构造 ShardLayout。
///
/// 这是 planner 与线上配置之间唯一的接缝：把 CacheTopology 里每个 GroupBase 的 policy /
/// spec 类型 / 字节尺寸抄进 ShardLayout::GroupLayout，并按 spec 类型和角色补出
/// head_shard_count 与 pre_sliced。纯函数、无 IO。
class ShardLayoutFactory {
public:
    /// @brief 构造本端布局。
    /// @param role 决定 pre_sliced —— OpaqueKVCacheSpec::isPrefillCpSliced 只对 PREFILL 为真，
    ///        即只有 prefill 的 spec 本身被切成 1/cp_size，decode 持整块。
    static ShardLayout fromTopology(const CacheTopology&     topology,
                                    const ParallelismConfig& pc,
                                    RoleType                 role) {
        ShardLayout layout;
        layout.pc = pc;

        for (const auto& group : topology.groups()) {
            ShardLayout::GroupLayout g;
            g.policy                = group.policy;
            g.kv_block_stride_bytes = group.kv_block_stride_bytes;
            g.kv_scale_stride_bytes = group.kv_scale_stride_bytes;
            g.seq_size_per_block    = group.seq_size_per_block;
            if (group.spec) {
                g.spec_type             = group.spec->type;
                g.k_block_payload_bytes = group.spec->k_block_payload_bytes();
            }
            layout.groups[group.tag] = g;
        }

        // head_shard_count 依赖 pc + spec_type，必须在 groups 填好之后算。
        layout.deriveHeadShardCounts();

        // pre_sliced 是角色属性：仅 PREFILL 角色、且该 group 的 CP 字节切分生效时，
        // spec 自身已被切成 1/cp_size（OpaqueKVCacheSpec 里的 `entries /= cp_size`）。
        const bool is_prefill_role = role == RoleType::PREFILL;
        for (auto& [tag, g] : layout.groups) {
            g.pre_sliced = is_prefill_role && layout.effectiveSlice(tag) != CpBlockSliceMode::NONE;
        }
        return layout;
    }

    /// @brief 由本端布局推导对端布局（跨端协议零改动的落点，见设计文档 §3.2）。
    /// @param peer_tp_size 对端 tp_size。decode 侧取自 Meta::P2PRoutingContext::prefill_tp_size，
    ///        prefill 侧取自 decode_transfer_servers.size()。
    /// @param peer_kv_cache_sharded 对端是否开 KV cache CP 分片。这是部署级同配开关
    ///        （OpaqueKVCacheSpec::fixedRegionCpSize 的 DECODE 分支已依赖此前提）。
    static ShardLayout peerOf(const ShardLayout& self,
                              int                peer_tp_size,
                              bool               peer_kv_cache_sharded,
                              RoleType           peer_role) {
        ParallelismConfig peer_pc                     = self.pc;
        peer_pc.tp_size                               = peer_tp_size;
        peer_pc.prefill_cp_config.kv_cache_sharded    = peer_kv_cache_sharded;
        peer_pc.prefill_cp_config.prefill_cp_size     = peer_kv_cache_sharded ? peer_tp_size : 0;
        peer_pc.role_type                             = peer_role;
        return ShardLayout::forPeer(self, peer_pc, peer_role == RoleType::PREFILL);
    }

    /// @brief 该拓扑下需要编排的 tag 列表（顺序稳定，保证两侧 route_id 一致）。
    static std::vector<std::string> tagsOf(const CacheTopology& topology) {
        std::vector<std::string> tags;
        tags.reserve(topology.groups().size());
        for (const auto& group : topology.groups()) {
            tags.push_back(group.tag);
        }
        return tags;
    }
};

}  // namespace rtp_llm
