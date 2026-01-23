#pragma once

#include "rtp_llm/cpp/model_rpc/TpBroadcastManager.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

/// @brief TPBroadcastClient 在 rank0 上调用 broadcast
class TPBroadcastClient {
public:
    using BroadcastResult = TPBroadcastResult<FunctionRequestPB, FunctionResponsePB>;

    TPBroadcastClient(const std::vector<std::string>& worker_addrs, int64_t extra_wait_time_ms = 10 * 1000);
    ~TPBroadcastClient() = default;

public:
    bool init();

    struct Result {
        Result(const std::string& unique_key, const std::shared_ptr<BroadcastResult>& tp_broadcast_result):
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

    private:
        std::string                      unique_key_;
        std::shared_ptr<BroadcastResult> tp_broadcast_result_;
        int64_t                          start_time_us_;
        int64_t                          total_cost_time_us_{0};
    };

    std::shared_ptr<Result> broadcast(int64_t                                               request_id,
                                      const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                                      const std::vector<std::pair<std::string, uint32_t>>&  decode_transfer_servers,
                                      const std::string&                                    unique_key,
                                      int64_t                                               deadline_ms,
                                      P2PConnectorBroadcastType                             type);

    void setExtraWaitTimeMs(int64_t extra_wait_time_ms);

private:
    void genBroadcastRequest(FunctionRequestPB&                                    request,
                             int64_t                                               request_id,
                             const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                             const std::vector<std::pair<std::string, uint32_t>>&  decode_transfer_servers,
                             const std::string&                                    unique_key,
                             int64_t                                               deadline_ms,
                             P2PConnectorBroadcastType                             type);

private:
    std::vector<std::string>            worker_addrs_;
    int64_t                             extra_wait_time_ms_{10 * 1000};
    std::shared_ptr<RPCPool>            rpc_pool_;
    std::shared_ptr<TpBroadcastManager> tp_broadcast_manager_;
};

}  // namespace rtp_llm
