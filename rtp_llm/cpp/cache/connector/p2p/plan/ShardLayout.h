#pragma once

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace rtp_llm {

/// @brief 一侧（Prefill 或 Decode）的 KV cache 分片布局。
///
/// 全部字段都可以从本端的 ParallelismConfig + CacheTopology 推导，对端布局则由
/// forPeer() 以替换后的 ParallelismConfig 重算 —— 因此 GetPeerInfo / StartLoad
/// 无需新增字段。参见设计文档 §3.2。
struct ShardLayout {
    /// @brief 单个 cache group（以 tag 标识）在本侧的布局。
    struct GroupLayout {
        CacheGroupPolicy policy;
        KVCacheSpecType  spec_type = KVCacheSpecType::MultiHeadAttention;
        IndexerScaleFormat indexer_scale_format = IndexerScaleFormat::NONE;

        size_t kv_block_stride_bytes = 0;
        size_t kv_scale_stride_bytes = 0;
        size_t k_block_payload_bytes = 0;
        size_t seq_size_per_block    = 0;

        /// spec 自身已按 CP 切成 1/cp_size（仅 Prefill 角色的 cp_slice group），
        /// 见 OpaqueKVCacheSpec::isPrefillCpSliced。
        bool pre_sliced = false;

        // MHA may replicate KV heads across adjacent attention ranks.
        uint32_t global_kv_heads  = 0;
        int      head_shard_count = 1;

        // Local convolution history geometry for models that opt in to LINEAR_QKV.
        // The SSM bytes are carried in k_block_payload_bytes.
        size_t linear_q_row_bytes  = 0;
        size_t linear_k_row_bytes  = 0;
        size_t linear_v_row_bytes  = 0;
        size_t linear_history_rows = 0;
    };

    ParallelismConfig            pc;
    size_t                             cache_key_tokens_per_block = 1;
    bool                               use_group_cache_transfer_policy = false;
    bool                               use_mla                         = false;
    bool                               local_full_page_rr              = false;
    std::map<std::string, GroupLayout> groups;

    // ---- rank 坐标：CP 轴与 head 轴相互独立，不做除法分解（设计文档 §3.3）----

    int cpSize() const {
        if (local_full_page_rr) {
            return static_cast<int>(pc.tp_size);
        }
        // Ordinary Decode's flag describes its Prefill peer.
        return pc.role_type != RoleType::DECODE && pc.prefill_cp_config.is_enabled()
                       && pc.prefill_cp_config.kv_cache_sharded ?
                   static_cast<int>(pc.tp_size) :
                   1;
    }

    // Compact state pages retain Prefill's token geometry on Decode, without
    // distributing ownership across Decode ranks as FULL pages would do.
    int physicalCpSize(const std::string& tag) const {
        if (pc.role_type == RoleType::DECODE && pc.prefill_cp_config.kv_cache_sharded
            && pc.prefill_cp_config.is_prefill_enabled() && hasGroup(tag)
            && group(tag).policy.cp_mapping == CpBlockMappingMode::COMPACT_LAST_RANK) {
            return std::max(1, static_cast<int>(pc.prefill_cp_config.prefill_cp_size));
        }
        return cpSize();
    }

    int cpRank(int rank) const {
        const int n = cpSize();
        return n > 1 ? rank % n : 0;
    }

    int routeCpSize(const std::string& tag) const {
        const auto& layout = group(tag);
        // Head-only LINEAR state has no page ownership axis under local PageRR.
        const bool head_only_linear = local_full_page_rr && layout.spec_type == KVCacheSpecType::LinearAttention
                                      && effectiveMapping(tag) == CpBlockMappingMode::NONE
                                      && effectiveSlice(tag) == CpBlockSliceMode::NONE;
        return layout.linear_history_rows != 0 || head_only_linear ? 1 : cpSize();
    }

    int routeCpRank(int rank, const std::string& tag) const {
        return rank % std::max(1, routeCpSize(tag));
    }

    int rankCount() const {
        return static_cast<int>(pc.tp_size);
    }

    /// per-group，不是 per-side。旧版曾写成 pc.get_attn_tp_size() 的全局标量，
    /// 对 MLA 会错误地返回 tp_size。
    int headShardCount(const std::string& tag) const {
        auto it = groups.find(tag);
        return it == groups.end() ? 1 : it->second.head_shard_count;
    }

    int headShard(int rank, const std::string& tag) const {
        const int n = headShardCount(tag);
        if (n <= 1) {
            return 0;
        }
        if (group(tag).global_kv_heads == 0) {
            return rank % n;
        }
        const int attn_tp = static_cast<int>(pc.get_attn_tp_size());
        // Weight loading puts replicas of a KV shard on consecutive ranks.
        return (rank % attn_tp) / (attn_tp / n);
    }

    bool hasGroup(const std::string& tag) const {
        return groups.find(tag) != groups.end();
    }

    const GroupLayout& group(const std::string& tag) const {
        return groups.at(tag);
    }

    // ---- 生效的 CP 语义：镜像 CPSlotMapper::layoutForGroup ----

    /// FULL follows local CP ownership; compact state retains its physical CP geometry.
    CpBlockMappingMode effectiveMapping(const std::string& tag) const {
        if (!hasGroup(tag) || physicalCpSize(tag) <= 1) {
            return CpBlockMappingMode::NONE;
        }
        return group(tag).policy.cp_mapping;
    }

    /// FULL group 走 page/block 级 CP 映射，字节切分只对 state/SWA 类 group 有效。
    CpBlockSliceMode effectiveSlice(const std::string& tag) const {
        if (cpSize() <= 1 || !hasGroup(tag)) {
            return CpBlockSliceMode::NONE;
        }
        const auto& policy = group(tag).policy;
        return policy.group_type == CacheGroupType::FULL ? CpBlockSliceMode::NONE : policy.cp_slice;
    }

    /// @brief 还原出「不分片时」的全局 block 字节数，用于两侧一致性校验。
    size_t effectiveGlobalBlockBytes(const std::string& tag) const {
        const auto& g = group(tag);
        size_t      bytes = g.kv_block_stride_bytes * static_cast<size_t>(headShardCount(tag));
        if (g.pre_sliced) {
            bytes *= static_cast<size_t>(cpSize());
        }
        return bytes;
    }

    // ---- 构造辅助 ----

    static bool specShardsHeads(KVCacheSpecType spec_type) {
        return spec_type == KVCacheSpecType::MultiHeadAttention || spec_type == KVCacheSpecType::LinearAttention;
    }

    /// 依据 pc 与各 group 的 spec_type 填充 head_shard_count。构造 ShardLayout 后必须调用一次。
    void deriveHeadShardCounts() {
        const int attn_tp = static_cast<int>(pc.get_attn_tp_size() > 0 ? pc.get_attn_tp_size() : 1);
        for (auto& [tag, g] : groups) {
            (void)tag;
            g.head_shard_count = specShardsHeads(g.spec_type) ? attn_tp : 1;
            if (g.spec_type == KVCacheSpecType::MultiHeadAttention && g.global_kv_heads > 0) {
                g.head_shard_count = std::gcd(g.global_kv_heads, attn_tp);
            }
        }
    }

    /// @brief 由本端布局推导对端布局。
    ///
    /// 依据 same-build 契约：两侧的 group policy / spec 类型 / 全局 block 字节数相同，
    /// 只有并行度不同。等价于「拷一份 ParallelismConfig、替换 tp_size 与 prefill_cp_config，
    /// 再调 localKvHeadNumForSpec」，但这里改用「还原全局再除对端因子」，避免依赖 config creator。
    static ShardLayout forPeer(const ShardLayout& self, const ParallelismConfig& peer_pc, bool peer_is_prefill_role);
};

}  // namespace rtp_llm
