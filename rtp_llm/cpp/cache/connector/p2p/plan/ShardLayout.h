#pragma once

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

#include <cstddef>
#include <map>
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

        size_t kv_block_stride_bytes = 0;
        size_t kv_scale_stride_bytes = 0;
        size_t k_block_payload_bytes = 0;
        size_t seq_size_per_block    = 0;

        /// spec 自身已按 CP 切成 1/cp_size（仅 Prefill 角色的 cp_slice group），
        /// 见 OpaqueKVCacheSpec::isPrefillCpSliced。
        bool pre_sliced = false;

        /// head 分片数是 **spec 类型的属性**，不是这一侧的属性：只有会除
        /// get_attn_tp_size() 的 spec 才被 head 切分。由 deriveHeadShardCounts() 填充。
        int head_shard_count = 1;
    };

    ParallelismConfig            pc;
    std::map<std::string, GroupLayout> groups;

    // ---- rank 坐标：CP 轴与 head 轴相互独立，不做除法分解（设计文档 §3.3）----

    int cpSize() const {
        return pc.prefill_cp_config.kv_cache_sharded ? static_cast<int>(pc.tp_size) : 1;
    }

    int cpRank(int rank) const {
        const int n = cpSize();
        return n > 1 ? rank % n : 0;
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
        return n > 1 ? rank % n : 0;
    }

    bool hasGroup(const std::string& tag) const {
        return groups.find(tag) != groups.end();
    }

    const GroupLayout& group(const std::string& tag) const {
        return groups.at(tag);
    }

    // ---- 生效的 CP 语义：镜像 CPSlotMapper::layoutForGroup ----

    /// cpSize() <= 1 时 layoutForGroup 提前返回，mapping 退化为 NONE。
    CpBlockMappingMode effectiveMapping(const std::string& tag) const {
        if (cpSize() <= 1 || !hasGroup(tag)) {
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

    /// 只有会除 get_attn_tp_size() 的 spec 才按 attention TP 切 head。
    /// MHAKVCacheSpec / LinearKVCacheSpec: 会除；MLAKVCacheSpec / Opaque: 不会。
    static bool specShardsHeads(KVCacheSpecType spec_type) {
        return spec_type == KVCacheSpecType::MultiHeadAttention || spec_type == KVCacheSpecType::LinearAttention;
    }

    /// 依据 pc 与各 group 的 spec_type 填充 head_shard_count。构造 ShardLayout 后必须调用一次。
    void deriveHeadShardCounts() {
        const int attn_tp = static_cast<int>(pc.get_attn_tp_size() > 0 ? pc.get_attn_tp_size() : 1);
        for (auto& [tag, g] : groups) {
            (void)tag;
            g.head_shard_count = specShardsHeads(g.spec_type) ? attn_tp : 1;
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
