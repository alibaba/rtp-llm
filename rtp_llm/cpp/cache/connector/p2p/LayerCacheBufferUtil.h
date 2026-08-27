#pragma once

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/TransferPlan.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/Types.h"
#include <vector>
#include <memory>

namespace rtp_llm {

/// @brief LayerCacheBuffer 转换工具类
/// 提供 KVCacheResource 到 LayerCacheBuffer 的转换功能
class LayerCacheBufferUtil {
public:
    static std::vector<std::shared_ptr<LayerCacheBuffer>> convert(KVCacheResource&    resource,
                                                                  const CacheTopology& topology,
                                                                  int                   start_block_idx = 0,
                                                                  int                   block_count     = -1,
                                                                  int                   cp_rank         = 0,
                                                                  int                   cp_size         = 1);

    static std::vector<std::shared_ptr<LayerCacheBuffer>> convertLayer(KVCacheResource&    resource,
                                                                       const CacheTopology& topology,
                                                                       int                   layer_id,
                                                                       int                   start_block_idx,
                                                                       int                   block_count,
                                                                       int                   cp_rank = 0,
                                                                       int                   cp_size = 1);

    static std::shared_ptr<LayerCacheBuffer> convertLayerTag(KVCacheResource&      resource,
                                                             const GroupBase&       group,
                                                             int                    layer_id,
                                                             int                    start_block_idx,
                                                             int                    block_count,
                                                             int                    cp_rank = 0,
                                                             int                    cp_size = 1);

    /// @brief Route 驱动的投影：只取 logical_positions 指定的逻辑位置。
    ///
    /// 与 convertLayerTag 的关键区别：**本函数不施加 group.policy.active_tail_blocks**。
    /// 尾部裁剪必须由编排层统一算进 KeyShardSpec::tail_count —— convertLayerTag 里那段
    /// `logical_end - tail_count` 用的是**本侧**的 logical_end，两侧 compact 程度不同时
    /// （prefill compact cp_size→1、decode 不 compact）「最后 N 个」会指向不同的 key，
    /// 破坏 executeCopy 的键集包含契约。logical_positions 由
    /// KVCacheTransferPlanner::resolveKeys 解析，已经含了 tail_count。
    static std::shared_ptr<LayerCacheBuffer> convertLayerTagForRoute(KVCacheResource&           resource,
                                                                     const GroupBase&           group,
                                                                     int                        layer_id,
                                                                     const std::vector<size_t>& logical_positions,
                                                                     int                        cp_rank,
                                                                     int                        cp_size);

    /// @brief 对某个 tag 覆盖的每一层各产出一个 buffer（route 与 layer 无关，故在此展开）。
    static std::vector<std::shared_ptr<LayerCacheBuffer>>
    convertTagForRoute(KVCacheResource&           resource,
                       const CacheTopology&       topology,
                       const std::string&         cache_tag,
                       const std::vector<size_t>& logical_positions,
                       int                        cp_rank,
                       int                        cp_size);

    /// @brief 将 LayerCacheBuffer 转换为 transfer 层需要的 KeyBlockInfoMap
    static transfer::KeyBlockInfoMap buildKeyBlockInfos(const std::shared_ptr<LayerBlockConverter>& converter,
                                                        const std::shared_ptr<LayerCacheBuffer>&    layer_cache_buffer,
                                                        int                                         partition_count = 1,
                                                        int                                         partition_id = 0);

    /// @brief 带 CP 字节切分的版本。切分语义与 CPSlotMapper::sliceBlockForPeer 保持一致：
    ///   EQUAL_BYTES   -> 分母是 block.size_bytes（整个 stride）
    ///   PAYLOAD_BYTES -> 分母是 k_block_payload_bytes，只覆盖 payload 区间，
    ///                    stride 的对齐填充尾部不参与传输
    /// 且要求被切的 block 只有一个子块（sliceBlockForPeer 的 parts.size() == 1 前提），
    /// 即与 head 维切分互斥 —— planner 的 Step 1 已保证这一点。
    static transfer::KeyBlockInfoMap buildKeyBlockInfosSliced(const std::shared_ptr<LayerBlockConverter>& converter,
                                                             const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                                                             int                                      partition_count,
                                                             int                                      partition_id,
                                                             const SliceSpec&                         slice,
                                                             size_t k_block_payload_bytes);
};

}  // namespace rtp_llm
