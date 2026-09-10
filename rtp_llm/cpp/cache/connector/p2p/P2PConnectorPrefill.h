#pragma once

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorResourceStore.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include <c10/core/Event.h>
#include <grpc++/grpc++.h>
#include <functional>
#include <memory>
#include <string>

namespace rtp_llm {

class KVCacheConnectorLayerContext;
class Meta;
class P2PBroadcastClient;
class P2PConnectorSchedulerPrefill;
class LayerBlockConverter;
class P2PConnectorWorkerPrefill;

class P2PConnectorPrefill {
public:
    P2PConnectorPrefill(P2PConnectorConfig                          config,
                        const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                        const kmonitor::MetricsReporterPtr&         metrics_reporter);
    ~P2PConnectorPrefill();

    bool init();

    std::shared_ptr<AsyncContext> registerResource(const KVCacheResourcePtr&    resource,
                                                   const std::shared_ptr<Meta>& meta);

    std::shared_ptr<AsyncContext>
    asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context);

    bool writeByLayerTag(int                                   layer_id,
                         const std::string&                    tag,
                         const KVCacheResourcePtr&             resource,
                         int64_t                               request_id,
                         const std::shared_ptr<c10::Event>& event,
                         int64_t                               deadline_ms);

    void processRead(const P2PConnectorStartLoadRequestPB& request,
                     P2PConnectorStartLoadResponsePB&      response,
                     std::function<bool()>                 is_cancelled = nullptr);

    bool processReadPerRank(int64_t                                 request_id,
                            const std::string&                      unique_key,
                            int64_t                                 deadline_ms,
                            const P2PConnectorBroadcastTpRequestPB& p2p_request,
                            FunctionResponsePB&                     response);

    bool processNoTransferPerRank(int64_t                                 request_id,
                                  const std::string&                      unique_key,
                                  int64_t                                 deadline_ms,
                                  const P2PConnectorBroadcastTpRequestPB& p2p_request,
                                  FunctionResponsePB&                     response);

    bool cancelProcessReadPerRank(int64_t             request_id,
                                  const std::string&  unique_key,
                                  int64_t             deadline_ms,
                                  int64_t             request_deadline_ms,
                                  FunctionResponsePB& response);

    std::shared_ptr<P2PConnectorResourceStore> resourceStore() const {
        return stream_store_;
    }

private:
    grpc::Status waitForResourceEntry(const std::string&                          unique_key,
                                      int64_t                                     request_deadline_ms,
                                      int64_t                                     transfer_deadline_ms,
                                      std::function<bool()>                       is_cancelled,
                                      std::shared_ptr<P2PConnectorResourceEntry>& resource_entry);

    void waitAndFillResponse(const std::shared_ptr<P2PConnectorResourceEntry>& resource_entry,
                             P2PConnectorStartLoadResponsePB&                  response,
                             std::function<bool()>                             is_cancelled = nullptr);

    grpc::Status fillResponseWithStreamInfo(const P2PConnectorResourceEntry::SideChannelData& data,
                                            P2PConnectorStartLoadResponsePB&                  response);

private:
    const P2PConnectorConfig                       config_;
    std::shared_ptr<LayerBlockConverter>           layer_block_converter_;
    kmonitor::MetricsReporterPtr                   metrics_reporter_;
    std::shared_ptr<P2PBroadcastClient>            tp_broadcast_client_;
    std::unique_ptr<P2PConnectorSchedulerPrefill> scheduler_;
    std::shared_ptr<P2PConnectorWorkerPrefill>    worker_;
    std::shared_ptr<P2PConnectorResourceStore>    stream_store_;
};

}  // namespace rtp_llm
