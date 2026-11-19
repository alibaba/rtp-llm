#pragma once

#include <cstdint>
#include <string>

namespace rtp_llm {

class P2PKeyUtil {
public:
    /// @brief 传输 key：由编排层签发的 route_id 命名，而非两侧各自推导的 partition_id。
    ///
    /// 这是「双端约定」被彻底消灭的落点：send_req.unique_key 与 recv_req.unique_key 必须逐字节
    /// 相同，传输层靠它在 TransferTaskStore::task_map_ 里做 rendezvous。旧的
    /// makePartitionLayerKey 用 partition_id 命名，而 partition_id 在源端由
    /// 源端的不对称 TP 工具算、在目的端由接收循环下标算 —— 两个独立公式碰巧相等，
    /// 是全链路最脆的耦合点。
    ///
    /// @param plan_digest 计划摘要。两侧 plan 若因配置漂移而分歧，key 不匹配 ⇒ 退化为
    ///        "no matching recv task within deadline"（TIMEOUT），而不是拷进错误的字节区间。
    ///        失败模式严格变好，且零协议成本。
    static std::string makeRouteLayerKey(const std::string& base_key,
                                         int                layer_id,
                                         const std::string& cache_tag,
                                         int                route_id,
                                         uint64_t           plan_digest) {
        return base_key + "_" + std::to_string(layer_id) + "_" + cache_tag + "_r" + std::to_string(route_id) + "_p"
               + shortDigest(plan_digest);
    }

    /// 把 64 位摘要的高低位折叠成 32 位十六进制，避免只保留高位时漏掉低位差异，
    /// 同时不让 key 过长（key 会进 std::map<std::string, ...> 的比较路径）。
    static std::string shortDigest(uint64_t plan_digest) {
        static constexpr char kHex[] = "0123456789abcdef";
        const uint32_t        v      = static_cast<uint32_t>(plan_digest >> 32) ^ static_cast<uint32_t>(plan_digest);
        std::string           out(8, '0');
        for (int i = 7; i >= 0; --i) {
            out[static_cast<size_t>(i)] = kHex[(v >> ((7 - i) * 4)) & 0xfu];
        }
        return out;
    }
};

}  // namespace rtp_llm
