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

    /// Decode 方向：本 route 覆盖的每层 buffer（cache_key -> block_id 已由 rank0 解析好）。
    /// Prefill 方向为空 —— prefill worker 用自己 writeByLayer 产出的本地投影，
    /// 在所有被允许的 CP 形态下它恰好等于该 route 的键集。
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_buffers;

    /// Prefill 方向：目的端点（由 peer_index 在 peer_workers 里解析而来）。
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

    /// 某个 tag 上有多少条 route —— prefill 侧用它把 outstanding 阈值按「层」而非
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
