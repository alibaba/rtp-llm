#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayout.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/TransferPlan.h"

#include <cstddef>
#include <string>
#include <vector>

namespace rtp_llm {

/// @brief P2P KV cache 传输编排器。
///
/// 把 TP 不对称、RR CP、非 RR CP、CP 字节切分统一成一份 TransferPlan。由两侧 rank0
/// scheduler（P2PConnectorSchedulerDecode / P2PConnectorSchedulerPrefill）各调用一次，
/// worker 只执行下发的 route。
///
/// plan() 是纯函数：不含 cache_keys、不含 block id、不含时间/随机源，因此可按
/// (src.pc, dst.pc, topology) 缓存 —— 同一部署下每个 tag 只算一次。
class KVCacheTransferPlanner {
public:
    /// @brief 布局级编排：产出与请求无关的 route 集。
    static PlanResult plan(const ShardLayout& src, const ShardLayout& dst, const std::vector<std::string>& tags);

    /// @brief 请求级展开：把 route 上的键规则解析成具体逻辑位置。两侧执行期各调一次。
    ///
    /// plan() 不能吃 logical_count（否则退化成每请求重算、无法缓存），而 COMPACT 的尾键与
    /// active_tail_blocks 都依赖 count，故拆成两层。某条 route 在特定 count 下解析为空是预期
    /// 行为：两侧规则相同故一致判空，decode 不注册、prefill 不发送。
    ///
    /// @param logical_count **必须是全序列的 cache_keys 数量，不是 block_range 窗口长度。**
    ///        这与 LayerCacheBufferUtil::convertLayerTag 的既有约定一致——它把
    ///        `cache_keys.size()` 传给 physicalBlockPosition，而 `[start_block_idx, +block_count)`
    ///        只是**额外**的窗口裁剪。prefill 侧不知道 decode 的 block_range（prefix 部分命中的
    ///        结果），若两侧用不同的 count，`include_final_key` 与 `tail_count` 会算出不同的键，
    ///        破坏键集包含契约。窗口裁剪只在 decode 侧叠加在本函数结果之上。
    static std::vector<size_t> resolveKeys(const KeyShardSpec& spec, size_t logical_count);


private:
    struct HeadPair {
        int           src_head = 0;
        int           dst_head = 0;
        PartitionSpec src_partition;
        PartitionSpec dst_partition;
    };

    struct CpPair {
        int          src_cp      = 0;
        int          dst_cp      = 0;
        int          slice_index = 0;
        KeyShardSpec keys;
    };

    static ErrorInfo validateTag(const ShardLayout& src, const ShardLayout& dst, const std::string& tag);

    static bool ownsPosition(const ShardLayout& side, const std::string& tag, size_t pos, size_t count, int cp_rank);

    static ErrorInfo buildHeadPairs(const ShardLayout&     src,
                                    const ShardLayout&     dst,
                                    const std::string&     tag,
                                    std::vector<HeadPair>& out);

    static ErrorInfo
    buildCpPairs(const ShardLayout& src, const ShardLayout& dst, const std::string& tag, std::vector<CpPair>& out);

    /// 把一组剩余值折叠成尽可能粗的剩余类；无法用单一剩余类表达时返回 false。
    static bool collapseResidues(const std::vector<int>& residues, int modulus, int& out_modulus, int& out_residue);

    static void assignSlices(const ShardLayout& src,
                             const ShardLayout& dst,
                             const std::string& tag,
                             const CpPair&      cp_pair,
                             SliceSpec&         src_slice,
                             SliceSpec&         dst_slice);

    static std::vector<int>
    ranksWithCoord(const ShardLayout& side, const std::string& tag, int cp_rank, int head_shard);
};

}  // namespace rtp_llm
