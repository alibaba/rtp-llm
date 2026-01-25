#pragma once

#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/core/Event.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorScheduler.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorStreamStore.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.h"
#include <grpc++/grpc++.h>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

/**
 * Q: 如何保证kvcache不被写坏
 * A: scheduler 应该在 worker 执行完成之前都持有kv_cache资源, 如果在worker执行过程中, scheduler 因为等待 worker 超时 /
 * RPC失败等原因退出等待，释放worker资源，那么应该abort 目前这部分是在 scheduler 对 worker
 * 的调用中实现，如果调用超时或rpc失败，则scheduler会abort进程
 * Q: 超时处理
 * A:
 * 每个stream都会有自己的超时，worker的实现逻辑中会尽量保证在超时后尽快终止后续的可能操作，以尽快完成资源释放，但是不保证一定能在deadline之前完成操作.
 */
class P2PConnector: public KVCacheConnector {
public:
    P2PConnector(const KVCacheConfig&                        cache_config,
                 const RuntimeConfig&                        runtime_config,
                 const CacheStoreConfig&                     cache_store_config,
                 const ParallelismConfig&                    parallelism_config,
                 const PDSepConfig&                          pd_sep_config,
                 const ModelConfig&                          model_config,
                 uint32_t                                    layer_all_num,
                 const std::shared_ptr<LayerBlockConvertor>& layer_block_convertor,
                 const kmonitor::MetricsReporterPtr&         metrics_reporter);
    ~P2PConnector() override;

public:
    bool init();

public:
    // KVCacheConnector interface
    std::shared_ptr<AsyncMatchContext> asyncMatch(const KVCacheResourcePtr&    resource,
                                                  const std::shared_ptr<Meta>& meta) override;

    std::shared_ptr<AsyncContext> asyncRead(const KVCacheResourcePtr&                 resource,
                                            const std::shared_ptr<Meta>&              meta,
                                            const std::shared_ptr<AsyncMatchContext>& match_context) override;

    std::shared_ptr<AsyncContext> asyncWrite(const KVCacheResourcePtr&    resource,
                                             const std::shared_ptr<Meta>& meta) override;

    std::shared_ptr<AsyncContext>
    asyncWriteByLayer(int layer_id, const KVCacheResourcePtr& resource, const std::shared_ptr<Meta>& meta) override;

public:
    // Prefill side: handle load request from decode side (StartLoad RPC)
    // is_cancelled: optional callback to check if the request is cancelled by client
    grpc::Status handleRead(const P2PConnectorStartLoadRequestPB& request,
                            P2PConnectorStartLoadResponsePB&      response,
                            std::function<bool()>                 is_cancelled = nullptr);

    bool executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response);

private:
    // Wait for resource entry with cancellation check every 1ms
    // Returns grpc::Status::OK if resource found, CANCELLED if cancelled, INTERNAL if timeout
    grpc::Status waitForResourceEntry(const std::string&                          unique_key,
                                      int64_t                                     deadline_ms,
                                      std::function<bool()>                       is_cancelled,
                                      std::shared_ptr<P2PConnectorResourceEntry>& resource_entry);

    // Fill response with stream information (first token, position ids, reuse length, propose info)
    // Returns grpc::Status::OK if successful, INTERNAL if first token not found
    grpc::Status fillResponseWithStreamInfo(const std::shared_ptr<P2PConnectorResourceEntry>& resource_entry,
                                            P2PConnectorStartLoadResponsePB&                  response);

private:
    const KVCacheConfig&                 cache_config_;
    const RuntimeConfig&                 runtime_config_;
    const CacheStoreConfig&              cache_store_config_;
    const ParallelismConfig&             parallelism_config_;
    const PDSepConfig&                   pd_sep_config_;
    const ModelConfig&                   model_config_;
    const uint32_t                       layer_all_num_;
    std::shared_ptr<LayerBlockConvertor> layer_block_convertor_;
    kmonitor::MetricsReporterPtr         metrics_reporter_;

    std::shared_ptr<P2PConnectorScheduler>   scheduler_;
    std::shared_ptr<P2PConnectorWorker>      worker_;
    std::shared_ptr<P2PConnectorStreamStore> stream_store_;
};

}  // namespace rtp_llm
