#pragma once

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rtp_llm {

/// @brief head 维切分参数，直接喂给 LayerBlockConverter::convertIndexToBuffer。
struct PartitionSpec {
    int count = 1;
    int id    = 0;

    bool operator==(const PartitionSpec& o) const {
        return count == o.count && id == o.id;
    }
    bool operator!=(const PartitionSpec& o) const {
        return !(*this == o);
    }
};

/// @brief CP 字节切分参数，直接喂给 CPSlotMapper::sliceBlockForPeer。
/// 只施加在「持整块」的那一侧，且用对端的 CP 几何（设计文档 Step 4）。
struct SliceSpec {
    CpBlockSliceMode mode  = CpBlockSliceMode::NONE;
    int              count = 1;
    int              index = 0;

    bool operator==(const SliceSpec& o) const {
        return mode == o.mode && count == o.count && index == o.index;
    }
    bool operator!=(const SliceSpec& o) const {
        return !(*this == o);
    }
};

/// @brief 键选择规则：一个「模 lcm(src.cpSize, dst.cpSize) 的剩余类」，外加 COMPACT 的尾键例外。
///
/// 不能用 (cp_size, cp_rank) 表达 —— 两侧都分片且 dst 更细时，需要的是两个剩余类的交集，
/// 由中国剩余定理它是模 lcm 的剩余类。参见设计文档 Step 3 与用例 A11。
struct KeyShardSpec {
    int  modulus           = 1;
    int  residue           = 0;
    bool include_final_key = false;

    /// 只保留解析结果里最后 tail_count 个位置；0 表示不限制。
    ///
    /// 来源是 CacheGroupPolicy::active_tail_blocks（LINEAR=1、SWA=2、FULL=0）。**必须由编排层
    /// 折进规则、不能让两侧各自筛**：`buildCacheStorePlan` 的 `start = total - tail_count` 里的
    /// `total` 是**本侧**的块数，而两侧 compact 程度可能不同（prefill compact 4→1、decode 不
    /// compact），"最后 N 个"在两侧会指向不同的 key，破坏键集包含契约。
    int tail_count = 0;

    // 副本均分（设计文档 Step 3b，默认关闭）：在剩余类之上再筛一层。
    int replica_split_count = 1;
    int replica_split_index = 0;

    bool operator==(const KeyShardSpec& o) const {
        return modulus == o.modulus && residue == o.residue && include_final_key == o.include_final_key
               && tail_count == o.tail_count && replica_split_count == o.replica_split_count
               && replica_split_index == o.replica_split_index;
    }
    bool operator!=(const KeyShardSpec& o) const {
        return !(*this == o);
    }
};

/// @brief 一条传输路径 = 一个 (src_rank, dst_rank, cache_tag) 组合。
///
/// route 与 layer 无关：同一条 route 的描述对该 tag 覆盖的所有 layer 都成立，
/// 执行期按层展开成传输单元，key 为 <unique_key>_<layer_id>_<tag>_r<route_id>。
struct TransferRoute {
    int         route_id = 0;
    int         src_rank = 0;  // prefill worker index
    int         dst_rank = 0;  // decode worker index
    std::string cache_tag;

    KeyShardSpec  src_keys;
    PartitionSpec src_partition;
    PartitionSpec dst_partition;
    SliceSpec     src_slice;
    SliceSpec     dst_slice;

    // Step 5 的形状校验留痕，便于断言与排障。
    size_t src_bytes = 0;
    size_t dst_bytes = 0;
};

struct TransferPlan {
    std::vector<TransferRoute> routes;

    std::vector<const TransferRoute*> forDecodeRank(int dst_rank) const {
        std::vector<const TransferRoute*> out;
        for (const auto& r : routes) {
            if (r.dst_rank == dst_rank) {
                out.push_back(&r);
            }
        }
        return out;
    }

    std::vector<const TransferRoute*> forPrefillRank(int src_rank) const {
        std::vector<const TransferRoute*> out;
        for (const auto& r : routes) {
            if (r.src_rank == src_rank) {
                out.push_back(&r);
            }
        }
        return out;
    }

    std::vector<const TransferRoute*> forTag(const std::string& tag) const {
        std::vector<const TransferRoute*> out;
        for (const auto& r : routes) {
            if (r.cache_tag == tag) {
                out.push_back(&r);
            }
        }
        return out;
    }

    std::vector<const TransferRoute*> forDecodeRankAndTag(int dst_rank, const std::string& tag) const {
        std::vector<const TransferRoute*> out;
        for (const auto& r : routes) {
            if (r.dst_rank == dst_rank && r.cache_tag == tag) {
                out.push_back(&r);
            }
        }
        return out;
    }

    /// 稳定摘要，仅用于日志 / metric / 一致性自检，不上协议。
    uint64_t digest() const;
};

struct PlanResult {
    TransferPlan plan;
    ErrorInfo    error;

    bool ok() const {
        return error.ok();
    }
};

}  // namespace rtp_llm
