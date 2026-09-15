#pragma once

#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

/// @brief P2PBroadcastClient 在 rank0 上向所有 TP worker 广播 P2P 传输请求
class P2PBroadcastClient {
public:
    using TpBroadcastResult = ::rtp_llm::BroadcastResult<FunctionRequestPB, FunctionResponsePB>;

    explicit P2PBroadcastClient(const std::vector<std::string>& worker_addrs,
                                int64_t                         cancel_broadcast_timeout_ms = 1000);
    ~P2PBroadcastClient() = default;

public:
    /// @brief 初始化 RPC 连接池和 BroadcastManager
    bool init();

    struct Result {
        Result(const std::string& unique_key, const std::shared_ptr<TpBroadcastResult>& tp_broadcast_result):
            unique_key_(unique_key), tp_broadcast_result_(tp_broadcast_result), start_time_us_(currentTimeUs()) {}
        ~Result() {}

        std::string uniqueKey() const {
            return unique_key_;
        }

        bool done() const {
            return tp_broadcast_result_->done();
        }
        bool success() const;
        void checkDone();

        int64_t totalCostTimeUs() const {
            return total_cost_time_us_;
        }

        ErrorCode   errorCode() const;
        std::string errorMessage() const;

    private:
        std::string                        unique_key_;
        std::shared_ptr<TpBroadcastResult> tp_broadcast_result_;
        int64_t                            start_time_us_;
        int64_t                            total_cost_time_us_{0};
    };

    /// @brief 向所有 TP worker 广播 KV cache 传输请求
    std::shared_ptr<Result> broadcast(int64_t                                               request_id,
                                      const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                                      const std::vector<std::pair<std::string, uint32_t>>&  decode_transfer_servers,
                                      const std::string&                                    unique_key,
                                      int64_t                                               deadline_ms,
                                      P2PConnectorBroadcastType                             type,
                                      int                                                   remote_tp_size = 0);

    /// @brief 向所有 TP worker 广播 cancel 请求
    std::shared_ptr<Result> cancel(const std::string& unique_key, P2PConnectorBroadcastType type);

private:
    void genBroadcastRequest(FunctionRequestPB&                                    request,
                             int64_t                                               request_id,
                             const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                             const std::vector<std::pair<std::string, uint32_t>>&  decode_transfer_servers,
                             const std::string&                                    unique_key,
                             int64_t                                               deadline_ms,
                             P2PConnectorBroadcastType                             type,
                             int                                                   remote_tp_size);

private:
    std::vector<std::string>          worker_addrs_;
    int64_t                           cancel_broadcast_timeout_ms_;
    std::shared_ptr<RPCPool>          rpc_pool_;
    std::shared_ptr<BroadcastManager> tp_broadcast_manager_;
};

}  // namespace rtp_llm
