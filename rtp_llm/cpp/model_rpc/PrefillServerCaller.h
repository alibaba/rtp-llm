#pragma once

#include <cstdint>
#include "rtp_llm/cpp/model_rpc/PrefillServerCallerContext.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include <shared_mutex>
#include <unordered_map>

namespace rtp_llm {

class PrefillServerCaller {
public:
    explicit PrefillServerCaller(const std::string& process_id);
    ~PrefillServerCaller() = default;

    // 调用 Prefill 服务器
    std::shared_ptr<PrefillServerCallerContext> callPrefill(const GenerateInputPB* request,
                                                            const std::string&     ip,
                                                            uint32_t               port,
                                                            const std::string&     unique_key,
                                                            int64_t                deadline_us);

    grpc::Status callPrefill(grpc::ServerContext*                   server_context,
                             const GenerateInputPB*                 request,
                             grpc::ServerWriter<GenerateOutputsPB>* response_writer);

    /// @brief Get prefill's tp_size via GetPeerInfo RPC, with caching per endpoint.
    /// @param request_timeout_ms Stream timeout (generate_config.timeout_ms, ms). GetPeerInfo uses a deadline
    ///        derived from this value with a bounded min/max so unreachable prefill does not block for the full
    ///        generation budget.
    /// @return tp_size on success, -1 on failure.
    int getPrefillTpSize(const std::string& ip, uint32_t port, int32_t request_timeout_ms);

private:
    std::shared_ptr<RPCPool> rpc_pool_;
    std::string              process_id_;

    mutable std::shared_mutex            prefill_tp_cache_mutex_;
    std::unordered_map<std::string, int> prefill_tp_cache_;
};

}  // namespace rtp_llm
