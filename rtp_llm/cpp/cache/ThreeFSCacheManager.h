#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/th_op/GptInitParameter.h"

#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {
class Buffer;

namespace threefs {
class ThreeFSBlockCache;

struct WorkerRpcContext {
    WorkerRpcContext() {
        client_context = std::make_shared<grpc::ClientContext>();
    }
    grpc::Status                         status;
    std::shared_ptr<RpcService::Stub>    stub;
    std::shared_ptr<grpc::ClientContext> client_context;
    std::string                          server_addr;
};

class ThreeFSCacheManager final {
public:
    ThreeFSCacheManager(const std::shared_ptr<Buffer>&      k_cache,
                        const std::shared_ptr<Buffer>&      v_cache,
                        const CacheConfig&                  cache_config,
                        const GptInitParameter&             params,
                        const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr):
        k_cache_(k_cache),
        v_cache_(v_cache),
        cache_config_(cache_config),
        params_(params),
        metrics_reporter_(metrics_reporter) {}
    ~ThreeFSCacheManager();

public:
    bool    init();
    int32_t matchCache(const std::vector<int64_t>& cache_keys) const;
    bool    getCacheForAllRank(const std::vector<int64_t>& cache_keys,
                               const std::vector<int32_t>& block_indices,
                               int32_t                     input_len,
                               int64_t                     request_id);
    bool    getCacheForRank(const std::vector<int64_t>& cache_keys,
                            const std::vector<int32_t>& block_indices,
                            int64_t                     request_id) const;
    bool    putCacheForAllRank(const std::vector<int64_t>& cache_keys,
                               const std::vector<int32_t>& block_indices,
                               int64_t                     request_id) const;
    bool    putCacheForRank(const std::vector<int64_t>& cache_keys,
                            const std::vector<int32_t>& block_indices,
                            int64_t                     request_id) const;

private:
    std::string constructKvCacheKey(int64_t last_cache_key, int32_t rank = -1) const;
    bool        checkCacheIn3FS(const std::vector<int64_t>& cache_keys, bool for_put = false) const;
    bool        rpcGetCacheForAllRank(const std::vector<int64_t>& cache_keys,
                                      const std::vector<int32_t>& block_indices,
                                      int64_t                     request_id) const;
    bool        rpcPutCacheForAllRank(const std::vector<int64_t>& cache_keys,
                                      const std::vector<int32_t>& block_indices,
                                      int64_t                     request_id) const;
    bool        waitAllReuqestDone(const std::vector<WorkerRpcContext>&  worker_rpc_contexts,
                                   std::vector<::grpc::CompletionQueue>& completion_queues,
                                   std::vector<int>&                     unfinished_count_per_queue,
                                   int64_t                               request_id) const;

private:
    const std::shared_ptr<Buffer> k_cache_;
    const std::shared_ptr<Buffer> v_cache_;
    const CacheConfig             cache_config_;
    const GptInitParameter        params_;
    kmonitor::MetricsReporterPtr  metrics_reporter_;

    std::shared_ptr<ThreeFSBlockCache> threefs_block_cache_;
    std::shared_ptr<RPCPool>           rpc_pool_;
    // <total_matched_len, total_input_len>
    std::pair<int64_t, int64_t> cache_key_num_{0, 0};
    std::mutex                  cache_key_num_mutex_;
};

}  // namespace threefs
}  // namespace rtp_llm