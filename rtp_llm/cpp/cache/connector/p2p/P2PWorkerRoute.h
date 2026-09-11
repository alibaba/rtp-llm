#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/TransferPlan.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

/// @brief worker 执行一条 route 所需的全部信息，**已投影到本侧**。
///
/// 由 P2PConnector 从 TransferRoutePB 解出后交给 worker。worker 不再推导任何映射：
/// 它只负责按 partition / slice 取出字节，用 route_id 命名 key，发到 / 收自指定端点。
struct P2PWorkerRoute {
    int         route_id = 0;
    std::string cache_tag;

    /// 本侧的 head 维切分与 CP 字节切分。
    PartitionSpec partition;
    SliceSpec     slice;

    /// 本 route 的本侧每层 buffer（cache_key -> block_id 已由 rank0 解析好）。
    /// Read 的 Decode 接收侧、Write 的 Decode 发送侧和 Prefill 接收侧均显式提供。
    /// Read 的 Prefill 发送侧为空，使用 writeByLayer 产出的本地投影，与该 route 键集对应。
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_buffers;

    /// 发送侧目的端点（由 peer_index 在 peer_workers 里解析）：Read 的 Prefill / Write 的 Decode。
    /// 接收侧不使用这两个字段。
    std::string dst_ip;
    uint32_t    dst_port = 0;
};

/// @brief 一个 worker 收到的完整指令集。
struct P2PWorkerRoutePlan {
    std::vector<P2PWorkerRoute> routes;
    /// 计划摘要，进入传输 key。两侧 plan 分歧时 key 不匹配 ⇒ 退化为 recv 超时而非拷错字节。
    uint64_t plan_digest = 0;

    bool empty() const {
        return routes.empty();
    }

    /// 某个 tag 上有多少条 route —— Read 的 Prefill 发送侧用它把 outstanding 阈值按「层」而非
    /// 「传输次数」计量，否则一层的 route 会填满窗口、per-layer overlap 塌成 1 层。
    int routeCountForTag(const std::string& tag) const {
        int n = 0;
        for (const auto& r : routes) {
            if (r.cache_tag == tag) {
                ++n;
            }
        }
        return n;
    }

    int maxRoutesPerTag() const {
        int n = 0;
        for (const auto& r : routes) {
            n = std::max(n, routeCountForTag(r.cache_tag));
        }
        return std::max(1, n);
    }
};

}  // namespace rtp_llm
