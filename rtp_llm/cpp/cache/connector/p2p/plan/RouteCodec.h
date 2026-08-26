#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/plan/TransferPlan.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rtp_llm {

/// @brief TransferRoute <-> TransferRoutePB 的编解码。
///
/// 每个 worker 只收到**它自己那一侧**的 partition / slice —— 对端的那一半永不上线。
/// 因此编码是「有方向的」：encodeForPrefill 写 src_*，encodeForDecode 写 dst_*。
class RouteCodec {
public:
    /// @brief 编码给 prefill worker：它需要 route_id（命名 key）、目的端点下标、以及自己的
    ///        src_partition / src_slice。不需要键规则 —— 在所有被允许的 CP 形态下，
    ///        (src_rank, dst_rank, tag) 唯一确定一条 route，而 prefill worker 的本地投影
    ///        恰好等于该 route 的键集（详见设计文档 Step 3 的 CP 白名单）。
    static void encodeForPrefill(const TransferRoute& route, int peer_index, TransferRoutePB* pb) {
        pb->set_route_id(route.route_id);
        pb->set_cache_tag(route.cache_tag);
        pb->set_peer_index(peer_index);
        pb->set_partition_count(route.src_partition.count);
        pb->set_partition_id(route.src_partition.id);
        pb->set_slice_mode(static_cast<int32_t>(route.src_slice.mode));
        pb->set_slice_count(route.src_slice.count);
        pb->set_slice_index(route.src_slice.index);
    }

    /// @brief 编码给 decode worker：它需要 route_id、自己的 dst_partition / dst_slice，
    ///        以及本 route 覆盖的具体 (cache_key, block_id)。键规则由 rank0 自己解析完，
    ///        decode worker 因此是纯执行器、不重新推导键集。
    static void encodeForDecode(const TransferRoute& route, TransferRoutePB* pb) {
        pb->set_route_id(route.route_id);
        pb->set_cache_tag(route.cache_tag);
        pb->set_partition_count(route.dst_partition.count);
        pb->set_partition_id(route.dst_partition.id);
        pb->set_slice_mode(static_cast<int32_t>(route.dst_slice.mode));
        pb->set_slice_count(route.dst_slice.count);
        pb->set_slice_index(route.dst_slice.index);
    }

    /// @brief 解码出「本侧」的执行参数。方向已在编码时确定，故这里无需再区分。
    struct LocalRoute {
        int           route_id = 0;
        std::string   cache_tag;
        int           peer_index = 0;
        PartitionSpec partition;
        SliceSpec     slice;
    };

    static LocalRoute decode(const TransferRoutePB& pb) {
        LocalRoute out;
        out.route_id    = pb.route_id();
        out.cache_tag   = pb.cache_tag();
        out.peer_index  = pb.peer_index();
        out.partition   = PartitionSpec{pb.partition_count() > 0 ? pb.partition_count() : 1, pb.partition_id()};
        out.slice       = SliceSpec{toSliceMode(pb.slice_mode()),
                                    pb.slice_count() > 0 ? pb.slice_count() : 1,
                                    pb.slice_index()};
        return out;
    }

    static CpBlockSliceMode toSliceMode(int32_t raw) {
        switch (raw) {
            case static_cast<int32_t>(CpBlockSliceMode::EQUAL_BYTES):
                return CpBlockSliceMode::EQUAL_BYTES;
            case static_cast<int32_t>(CpBlockSliceMode::PAYLOAD_BYTES):
                return CpBlockSliceMode::PAYLOAD_BYTES;
            default:
                return CpBlockSliceMode::NONE;
        }
    }
};

}  // namespace rtp_llm
