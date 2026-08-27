#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/PrefillLoadCaller.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace autil {
class LockFreeThreadPool;
}

namespace rtp_llm {

class P2PConnectorSchedulerDecode {
public:
    struct AsyncReadResult {
        std::shared_ptr<P2PConnectorAsyncReadContext> context;
        ErrorInfo                                     error_info;

        bool ok() const {
            return error_info.ok();
        }
    };

    P2PConnectorSchedulerDecode(P2PConnectorSchedulerConfig                config,
                                const kmonitor::MetricsReporterPtr&        metrics_reporter,
                                const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client);
    ~P2PConnectorSchedulerDecode();

public:
    bool init(const std::string& process_id);
    void stopChecker();

    // asyncRead from Meta (extracts routing from Meta::p2pRouting())
    AsyncReadResult asyncRead(const KVCacheResourcePtr&       resource,
                              const std::shared_ptr<Meta>&    meta,
                              const std::pair<int, int>&      block_range,
                              bool                           no_transfer = false);
    void cancel(const std::shared_ptr<P2PConnectorAsyncReadContext>& context);

private:
    struct AsyncReadCallResults {
        std::shared_ptr<PrefillLoadCaller::Result>  server_call_result;
        std::shared_ptr<P2PBroadcastClient::Result> tp_sync_result;
    };

    std::optional<AsyncReadCallResults>
    startAsyncReadCalls(int64_t                                                  request_id,
                        const std::string&                                       prefill_ip,
                        uint32_t                                                 prefill_port,
                        const std::string&                                       unique_key,
                        int64_t                                                  request_deadline_ms,
                        int64_t                                                  transfer_deadline_ms,
                        const P2PBroadcastClient::RankLayerCacheBuffers&         rank_layer_cache_buffers,
                        const std::shared_ptr<DecodeSchedulerMetricsCollector>& collector,
                        ErrorInfo&                                               out_error,
                        int                                                      prefill_tp_size = 0,
                        bool                                                     no_transfer = false,
                        const P2PBroadcastClient::RankRoutes&                    rank_routes = {},
                        uint64_t                                                 plan_digest = 0);

    /// @brief 校验对端上报的 CP 片数与本端配置推导出的值一致（设计文档 §3.2.4）。
    ///
    /// 「对端 layout 由本端推导」是跨端协议零改动的前提，这条断言是它唯一的保护：
    /// 把配置漂移 / 版本不一致从传输期的疑难杂症变成首个请求上的确定性报错。
    ErrorInfo checkPeerCpLayout(int prefill_tp_size, int prefill_cp_size) const;

    /// @brief 取（并缓存）本请求形态对应的传输计划。plan 与请求无关，只依赖两侧布局，
    /// 故按 (prefill_tp_size, prefill_cp_size) 缓存 —— 同一部署下通常只算一次。
    std::shared_ptr<const PlanResult> planFor(int prefill_tp_size, int prefill_cp_size);

    /// @brief 把 plan 投影成「每个 decode worker 的 route 列表 + 具体 block」。
    ///
    /// rank0 自己解析键规则（resolveKeys），因此 decode worker 是纯执行器、不重新推导键集。
    /// logical_count 恒取全序列 cache_keys 数量；block_range 只作为额外的窗口裁剪叠加在
    /// resolveKeys 结果之上 —— prefill 侧不知道 block_range，若两侧用不同的 count，
    /// include_final_key 与 tail_count 会算出不同的键。
    P2PBroadcastClient::RankRoutes buildDecodeRankRoutes(const TransferPlan&        plan,
                                                        KVCacheResource&           resource,
                                                        const std::pair<int, int>& block_range,
                                                        size_t                     worker_num) const;

private:
    const P2PConnectorSchedulerConfig                    config_;
    // plan 缓存：key = (prefill_tp_size, prefill_cp_size)。plan() 是纯函数，
    // 不含 cache_keys / block id / 时间源，因此可跨请求复用。
    mutable std::mutex                                                       plan_cache_mutex_;
    std::map<std::pair<int, int>, std::shared_ptr<const PlanResult>>         plan_cache_;
    kmonitor::MetricsReporterPtr                         metrics_reporter_;
    std::shared_ptr<P2PBroadcastClient>                  tp_broadcast_client_;
    std::shared_ptr<PrefillLoadCaller>                   server_caller_;
    std::shared_ptr<P2PConnectorAsyncReadContextChecker> checker_;
    std::shared_ptr<autil::LockFreeThreadPool>           async_read_pool_;
};

}  // namespace rtp_llm
