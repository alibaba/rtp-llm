#pragma once

#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/core/Event.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include <grpc++/grpc++.h>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class P2PConnectorScheduler;
class P2PConnectorWorker;
class P2PConnectorResourceStore;
struct P2PConnectorResourceEntry;

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
    P2PConnector(P2PConnectorConfig                          config,
                 const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                 const kmonitor::MetricsReporterPtr&         metrics_reporter);
    ~P2PConnector() override;

public:
    bool init();

public:
    std::shared_ptr<AsyncMatchContext> asyncMatch(const KVCacheResourcePtr&    resource,
                                                  const std::shared_ptr<Meta>& meta) override;

    std::shared_ptr<AsyncContext> asyncRead(const KVCacheResourcePtr&                 resource,
                                            const std::shared_ptr<Meta>&              meta,
                                            const std::shared_ptr<AsyncMatchContext>& match_context,
                                            int                                       start_read_block_index,
                                            int                                       read_block_num) override;

    std::shared_ptr<AsyncContext> asyncWrite(const KVCacheResourcePtr&    resource,
                                             const std::shared_ptr<Meta>& meta) override;

    std::shared_ptr<AsyncContext>
    asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) override;

public:
    void handleRead(const P2PConnectorStartLoadRequestPB& request,
                    P2PConnectorStartLoadResponsePB&      response,
                    std::function<bool()>                 is_cancelled = nullptr);

    bool executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response);

private:
    grpc::Status waitForResourceEntry(const std::string&                          unique_key,
                                      int64_t                                     deadline_ms,
                                      std::function<bool()>                       is_cancelled,
                                      std::shared_ptr<P2PConnectorResourceEntry>& resource_entry);

    void waitAndFillResponse(const std::shared_ptr<P2PConnectorResourceEntry>& resource_entry,
                             P2PConnectorStartLoadResponsePB&                  response);

    grpc::Status fillResponseWithStreamInfo(const std::shared_ptr<P2PConnectorResourceEntry>& resource_entry,
                                            P2PConnectorStartLoadResponsePB&                  response);

    bool executeHandleRead(int64_t                                 request_id,
                           const std::string&                      unique_key,
                           int64_t                                 deadline_ms,
                           const P2PConnectorBroadcastTpRequestPB& p2p_request,
                           FunctionResponsePB&                     response);

    bool executeRead(int64_t                                 request_id,
                     const std::string&                      unique_key,
                     int64_t                                 deadline_ms,
                     const P2PConnectorBroadcastTpRequestPB& p2p_request,
                     FunctionResponsePB&                     response);

    bool executeCancelRead(const std::string& unique_key, FunctionResponsePB& response);

    bool executeCancelHandleRead(const std::string& unique_key, FunctionResponsePB& response);

private:
    const P2PConnectorConfig             config_;
    std::shared_ptr<LayerBlockConverter> layer_block_converter_;
    kmonitor::MetricsReporterPtr         metrics_reporter_;

    std::shared_ptr<P2PConnectorScheduler>     scheduler_;
    std::shared_ptr<P2PConnectorWorker>        worker_;
    std::shared_ptr<P2PConnectorResourceStore> stream_store_;
};

}  // namespace rtp_llm
