#pragma once

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/TransferPlan.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/Types.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <vector>
#include <memory>

namespace rtp_llm {

/// @brief LayerCacheBuffer 转换工具类
/// 提供 KVCacheResource 到 LayerCacheBuffer 的转换功能
class LayerCacheBufferUtil {
public:
    [[nodiscard]] static ErrorResult<std::vector<std::shared_ptr<LayerCacheBuffer>>>
    convert(KVCacheResource&     resource,
            const CacheTopology& topology,
            int                  start_block_idx            = 0,
            int                  block_count                = -1,
            int                  cp_rank                    = 0,
            int                  cp_size                    = 1,
            size_t               cache_key_tokens_per_block = 1);

    [[nodiscard]] static ErrorResult<std::vector<std::shared_ptr<LayerCacheBuffer>>>
    convertLayer(KVCacheResource&     resource,
                 const CacheTopology& topology,
                 int                  layer_id,
                 int                  start_block_idx,
                 int                  block_count,
                 int                  cp_rank                    = 0,
                 int                  cp_size                    = 1,
                 size_t               cache_key_tokens_per_block = 1);

    [[nodiscard]] static ErrorResult<std::shared_ptr<LayerCacheBuffer>>
    convertLayerTag(KVCacheResource& resource,
                    const GroupBase& group,
                    int              layer_id,
                    int              start_block_idx,
                    int              block_count,
                    int              cp_rank                    = 0,
                    int              cp_size                    = 1,
                    size_t           cache_key_tokens_per_block = 1);

    /// @brief Route 驱动的投影：只取 logical_positions 指定的逻辑位置。
    ///
    /// 不重新裁剪 logical_positions；state/SWA 位置必须命中与发布端相同的物理尾块投影。
    /// cache_key_tokens_per_block 是基础 key 粒度，不能用传输 CP 数替代 group 的块跨度。
    [[nodiscard]] static ErrorResult<std::shared_ptr<LayerCacheBuffer>>
    convertLayerTagForRoute(KVCacheResource&           resource,
                            const GroupBase&           group,
                            int                        layer_id,
                            const std::vector<size_t>& logical_positions,
                            int                        cp_rank,
                            int                        cp_size,
                            size_t                     cache_key_tokens_per_block = 1);

    /// @brief 对某个 tag 覆盖的每一层各产出一个 buffer（route 与 layer 无关，故在此展开）。
    [[nodiscard]] static ErrorResult<std::vector<std::shared_ptr<LayerCacheBuffer>>>
    convertTagForRoute(KVCacheResource&           resource,
                       const CacheTopology&       topology,
                       const std::string&         cache_tag,
                       const std::vector<size_t>& logical_positions,
                       int                        cp_rank,
                       int                        cp_size,
                       size_t                     cache_key_tokens_per_block = 1);

    /// @brief 将 LayerCacheBuffer 转换为 transfer 层需要的 KeyBlockInfoMap
    [[nodiscard]] static ErrorResult<transfer::KeyBlockInfoMap>
    buildKeyBlockInfos(const std::shared_ptr<LayerBlockConverter>& converter,
                       const std::shared_ptr<LayerCacheBuffer>&    layer_cache_buffer,
                       int                                         partition_count = 1,
                       int                                         partition_id    = 0);

    /// @brief 带 CP 字节切分的版本。切分语义与 CPSlotMapper::sliceBlockForPeer 保持一致：
    ///   EQUAL_BYTES   -> 分母是 block.size_bytes（整个 stride）
    ///   PAYLOAD_BYTES -> 分母是 k_block_payload_bytes，只覆盖 payload 区间，
    ///                    stride 的对齐填充尾部不参与传输
    /// 且要求被切的 block 只有一个子块（sliceBlockForPeer 的 parts.size() == 1 前提），
    /// 即与 head 维切分互斥 —— planner 的 Step 1 已保证这一点。
    [[nodiscard]] static ErrorResult<transfer::KeyBlockInfoMap>
    buildKeyBlockInfosSliced(const std::shared_ptr<LayerBlockConverter>& converter,
                             const std::shared_ptr<LayerCacheBuffer>&    layer_cache_buffer,
                             int                                         partition_count,
                             int                                         partition_id,
                             const SliceSpec&                            slice,
                             size_t                                      k_block_payload_bytes);
};

}  // namespace rtp_llm
