#pragma once

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class LayerBlockConverter;
class P2PBroadcastClient;
class P2PSchedulerDecodeRead;
class P2PSchedulerDecodeWrite;
class P2PWorkerDecodeRead;
class P2PWorkerDecodeWrite;

class P2PConnectorDecode {
public:
    P2PConnectorDecode(P2PConnectorConfig                          config,
                       const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                       const kmonitor::MetricsReporterPtr&         metrics_reporter);
    ~P2PConnectorDecode();

    bool init();
    void stopWriteback();

    std::shared_ptr<AsyncContext> read(const KVCacheResourcePtr&    resource,
                                       const std::shared_ptr<Meta>& meta,
                                       int                          start_read_block_index,
                                       int                          read_block_num);

    void cancelRead(const std::shared_ptr<AsyncContext>& context);
    std::shared_ptr<AsyncContext> write(KVCacheResourcePtr      resource,
                                        std::vector<int>        token_ids,
                                        int                     input_length,
                                        size_t                  kv_ready_token_count,
                                        Meta::P2PRoutingContext routing);

    bool readPerRank(int64_t                                 request_id,
                     const std::string&                      unique_key,
                     int64_t                                 deadline_ms,
                     const P2PConnectorBroadcastTpRequestPB& p2p_request,
                     FunctionResponsePB&                     response);

    bool cancelReadPerRank(const std::string& unique_key,
                           int64_t            request_deadline_ms,
                           FunctionResponsePB& response);

    bool queryLeaseStatusPerRank(const std::string& unique_key, FunctionResponsePB& response);
    bool writePerRank(const P2PConnectorBroadcastTpRequestPB& request, FunctionResponsePB& response);

private:
    const P2PConnectorConfig                      config_;
    std::shared_ptr<LayerBlockConverter>          layer_block_converter_;
    kmonitor::MetricsReporterPtr                  metrics_reporter_;
    std::shared_ptr<P2PBroadcastClient>           tp_broadcast_client_;
    std::unique_ptr<P2PSchedulerDecodeRead>       read_scheduler_;
    std::unique_ptr<P2PWorkerDecodeRead>          read_worker_;
    std::unique_ptr<P2PWorkerDecodeWrite>         write_worker_;
    std::unique_ptr<P2PSchedulerDecodeWrite>      write_scheduler_;
};

}  // namespace rtp_llm
