#pragma once

#include "rtp_llm/cpp/cache_new/TpBroadcastManager.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/disaggregate/transfer/CommonDefs.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

/// @brief TPBroadcastClient 在 rank0 上调用 broadcast 和 cancel
class TPBroadcastClient {
public:
    TPBroadcastClient(const GptInitParameter& gpt_init_parameter);
    ~TPBroadcastClient() = default;

public:
    bool init();

    struct Result {
        std::string                        unique_key;
        std::shared_ptr<TPBroadcastResult> result;
        Result(const std::string& unique_key, const std::shared_ptr<TPBroadcastResult>& result):
            unique_key(unique_key), result(result) {}

        void waitDone();
        bool success() const;
    };

    /// @brief 广播接口
    /// @param request_id 请求ID
    /// @param layer_cache_buffers LayerCacheBuffer 列表（只有blockid）
    /// @param decode_transfer_servers 解码传输服务器列表
    /// @param unique_key 唯一键
    /// @param deadline_ms 截止时间（毫秒时间戳）
    /// @return Result，包含 unique_key 和 broadcast result
    std::shared_ptr<Result> broadcast(int64_t                                               request_id,
                                      const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                                      const std::vector<std::pair<std::string, uint32_t>>&  decode_transfer_servers,
                                      const std::string&                                    unique_key,
                                      int64_t                                               deadline_ms,
                                      int64_t                                               timeout_ms = 0);

    /// @brief 取消任务, 一直要等到取消任务完成才返回
    /// @param result 之前 broadcast 返回的 Result
    void cancel(const std::shared_ptr<Result>& result, int64_t timeout_ms = 10 * 1000);

private:
    void genBroadcastRequest(BroadcastTpRequestPB&                                 request,
                             int64_t                                               request_id,
                             const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                             const std::vector<std::pair<std::string, uint32_t>>&  decode_transfer_servers,
                             const std::string&                                    unique_key,
                             int64_t                                               deadline_ms);

    void genCancelRequest(BroadcastTpRequestPB& request, const std::string& unique_key);

private:
    const GptInitParameter&             gpt_init_parameter_;
    int64_t                             extra_wait_time_ms_{10 * 1000};
    std::shared_ptr<RPCPool>            rpc_pool_;
    std::shared_ptr<TpBroadcastManager> tp_broadcast_manager_;
};

}  // namespace rtp_llm
