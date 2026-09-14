#pragma once

#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include <memory>
#include <atomic>
#include <string>
#include <vector>

namespace rtp_llm {

/// @brief P2PBroadcastClient 在 rank0 上向所有 TP worker 广播 P2P 传输请求
class P2PBroadcastClient {
public:
    using TpBroadcastResult = ::rtp_llm::BroadcastResult<FunctionRequestPB, FunctionResponsePB>;
    /// 每个 worker 一份 route 列表，顺序与 worker_addrs 一致。空表示该 worker 无任务。
    using RankRoutes = std::vector<std::vector<TransferRoutePB>>;

    explicit P2PBroadcastClient(const std::vector<std::string>& worker_addrs,
                                int64_t                         cancel_broadcast_timeout_ms = 1000);
    ~P2PBroadcastClient() = default;

public:
    /// @brief 初始化 RPC 连接池和 BroadcastManager
    bool init();

    struct Result {
        Result(const std::string& unique_key, const std::shared_ptr<TpBroadcastResult>& tp_broadcast_result):
            unique_key_(unique_key), tp_broadcast_result_(tp_broadcast_result), start_time_us_(currentTimeUs()) {}
        explicit Result(const std::string& unique_key):
            unique_key_(unique_key), start_time_us_(currentTimeUs()), locally_completed_(true) {}
        ~Result() {}

        std::string uniqueKey() const {
            return unique_key_;
        }

        bool done() const {
            const bool completed = locally_completed_ || (tp_broadcast_result_ && tp_broadcast_result_->done());
            if (completed) {
                int64_t expected = 0;
                total_cost_time_us_.compare_exchange_strong(expected, currentTimeUs() - start_time_us_);
            }
            return completed;
        }
        bool success() const;
        void checkDone();
        void setDoneCallback(std::function<void()> callback) {
            if (tp_broadcast_result_) {
                tp_broadcast_result_->setProgressCallback(std::move(callback));
            } else if (callback) {
                callback();
            }
        }

        int64_t totalCostTimeUs() const {
            return total_cost_time_us_.load();
        }

        FirstError::Snapshot firstError() const {
            return tp_broadcast_result_ ? tp_broadcast_result_->firstError() : FirstError::Snapshot{};
        }
        ErrorCode   errorCode() const;
        std::string errorMessage() const;

    private:
        std::string                        unique_key_;
        std::shared_ptr<TpBroadcastResult> tp_broadcast_result_;
        int64_t                            start_time_us_;
        mutable std::atomic<int64_t>       total_cost_time_us_{0};
        bool                               locally_completed_{false};
    };

    /// @brief 向所有 TP worker 广播一次 P2P 传输请求。
    ///
    /// routes 为空 ⇒ 所有 worker 收到同一份不带传输计划的请求；
    /// routes 非空 ⇒ 必须与 worker 数等长，第 i 项即第 i 个 worker 的那份计划。
    /// 「该 worker 无任务」由它自己的 routes[i] 为空表达，worker 侧以此为准。
    struct BroadcastParams {
        int64_t     request_id{0};
        std::string unique_key;
        /// 物理传输 deadline，同时作为本次广播的 gRPC deadline。
        int64_t deadline_ms{0};
        /// 原始用户请求 deadline，必须 > 0 且不早于 deadline_ms。
        int64_t                   request_deadline_ms{0};
        P2PConnectorBroadcastType type;
        /// 传输端点索引表，route 里的 peer_index 指向它。
        std::vector<std::pair<std::string, uint32_t>> peer_workers;
        RankRoutes                                    routes;
        uint64_t                                      plan_digest{0};
    };

    std::shared_ptr<Result> broadcast(BroadcastParams params);

    /// @brief 向所有 TP worker 广播 cancel 请求
    std::shared_ptr<Result>
    cancel(const std::string&        unique_key,
           P2PConnectorBroadcastType type,
           int64_t                   request_deadline_ms,
           int64_t                   request_id,
           int64_t                   deadline_ms);

    struct LeaseStatusResult {
        bool success{false};
        // Per-rank lease status. Empty on broadcast failure.
        struct RankStatus {
            bool valid{false};
            bool sealed{true};
            int  started_ops{0};
            int  finished_ops{0};
            bool stopped{false};
        };
        std::vector<RankStatus> ranks;

        bool allStopped() const {
            if (ranks.empty()) {
                return false;
            }
            for (const auto& r : ranks) {
                if (!r.valid || !r.stopped) {
                    return false;
                }
            }
            return true;
        }
    };

    /// @brief 向所有 TP worker 查询指定 unique_key 的 lease 状态（仅读，不修改状态）
    /// @param poll_timeout_ms 单次 broadcast 的 gRPC 超时（毫秒）
    LeaseStatusResult queryLeaseStatus(const std::string& unique_key, int64_t poll_timeout_ms);

private:
    std::shared_ptr<Result> broadcastRequests(std::vector<FunctionRequestPB> requests,
                                              const std::string&             unique_key,
                                              int64_t                        deadline_ms);

    void genBroadcastRequest(FunctionRequestPB&                  request,
                             const BroadcastParams&              params,
                             const std::vector<TransferRoutePB>* routes_of_worker);

private:
    std::vector<std::string>          worker_addrs_;
    int64_t                           cancel_broadcast_timeout_ms_;
    std::shared_ptr<RPCPool>          rpc_pool_;
    std::shared_ptr<BroadcastManager> tp_broadcast_manager_;
};

}  // namespace rtp_llm
