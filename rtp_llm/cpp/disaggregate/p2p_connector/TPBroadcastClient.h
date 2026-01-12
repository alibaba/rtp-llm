#pragma once

#include "rtp_llm/cpp/model_rpc/TpBroadcastManager.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

/// @brief TPBroadcastClient 在 rank0 上调用 broadcast 和 cancel
class TPBroadcastClient {
public:
    TPBroadcastClient(const std::vector<std::string>& worker_addrs);
    ~TPBroadcastClient() = default;

public:
    bool init();

    struct Result {
        Result(const std::string& unique_key, const std::shared_ptr<TPBroadcastResult>& tp_broadcast_result):
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
        std::string                        unique_key_;
        std::shared_ptr<TPBroadcastResult> tp_broadcast_result_;
        int64_t                            start_time_us_;
        int64_t                            total_cost_time_us_;
    };

    std::shared_ptr<Result> broadcast(int64_t                                               request_id,
                                      const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                                      const std::vector<std::pair<std::string, uint32_t>>&  decode_transfer_servers,
                                      const std::string&                                    unique_key,
                                      int64_t                                               deadline_ms,
                                      P2PConnectorBroadcastType                             type);

    void cancel(const std::shared_ptr<Result>& result, int64_t timeout_ms = 10 * 1000);

private:
    void genBroadcastRequest(BroadcastTpRequestPB&                                 request,
                             int64_t                                               request_id,
                             const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                             const std::vector<std::pair<std::string, uint32_t>>&  decode_transfer_servers,
                             const std::string&                                    unique_key,
                             int64_t                                               deadline_ms,
                             P2PConnectorBroadcastType                             type);

    void genCancelRequest(BroadcastTpRequestPB& request, const std::string& unique_key);
    void setExtraWaitTimeMs(int64_t extra_wait_time_ms);

private:
    std::vector<std::string>            worker_addrs_;
    int64_t                             extra_wait_time_ms_{10 * 1000};
    std::shared_ptr<RPCPool>            rpc_pool_;
    std::shared_ptr<TpBroadcastManager> tp_broadcast_manager_;
};

}  // namespace rtp_llm
