#pragma once

#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/core/Event.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorScheduler.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorStreamStore.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorWorker.h"
#include <grpc++/grpc++.h>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class P2PConnector: public KVCacheConnector {
public:
    P2PConnector(const KVCacheConfig&                        cache_config,
                 const RuntimeConfig&                        runtime_config,
                 const CacheStoreConfig&                     cache_store_config,
                 const ParallelismConfig&                    parallelism_config,
                 const PDSepConfig&                          pd_sep_config,
                 const ModelConfig&                          model_config,
                 const std::shared_ptr<LayerBlockConvertor>& layer_block_convertor,
                 const kmonitor::MetricsReporterPtr&         metrics_reporter);
    ~P2PConnector() override;

public:
    bool init() override;

public:
    // KVCacheConnector interface
    std::shared_ptr<AsyncMatchContext> asyncMatch(const KVCacheResourcePtr&                    resource,
                                                  const std::shared_ptr<KVCacheConnectorMeta>& meta) override;

    std::shared_ptr<AsyncContext> asyncRead(const KVCacheResourcePtr&                    resource,
                                            const std::shared_ptr<KVCacheConnectorMeta>& meta,
                                            const std::shared_ptr<AsyncMatchContext>&    match_context,
                                            const std::pair<int, int>&                   block_range) override;

    std::shared_ptr<AsyncContext> asyncWrite(const KVCacheResourcePtr&                    resource,
                                             const std::shared_ptr<KVCacheConnectorMeta>& meta) override;

    std::shared_ptr<AsyncContext> asyncWriteByLayer(int                                          layer_id,
                                                    const KVCacheResourcePtr&                    resource,
                                                    const std::shared_ptr<KVCacheConnectorMeta>& meta) override;

public:
    // Prefill side: handle load request from decode side (StartLoad RPC)
    grpc::Status handleRead(const P2PConnectorStartLoadRequestPB& request, P2PConnectorStartLoadResponsePB& response);

    bool handleTpBroadcast(const BroadcastTpRequestPB request, BroadcastTpResponsePB& response);

    // Prefill side: reserve resource for P2P transfer
    void addResource(const std::string&        unique_key,
                     int64_t                   request_id,
                     const IGenerateStreamPtr& generate_stream,
                     const KVCacheResourcePtr& kv_cache_resource,
                     int64_t                   deadline_ms);

private:
    const KVCacheConfig&                 cache_config_;
    const RuntimeConfig&                 runtime_config_;
    const CacheStoreConfig&              cache_store_config_;
    const ParallelismConfig&             parallelism_config_;
    const PDSepConfig&                   pd_sep_config_;
    const ModelConfig&                   model_config_;
    std::shared_ptr<LayerBlockConvertor> layer_block_convertor_;
    kmonitor::MetricsReporterPtr         metrics_reporter_;

    std::shared_ptr<P2PConnectorScheduler>   scheduler_;
    std::shared_ptr<P2PConnectorWorker>      worker_;
    std::shared_ptr<P2PConnectorStreamStore> stream_store_;
};

}  // namespace rtp_llm
