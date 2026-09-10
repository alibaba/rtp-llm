#pragma once

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnectorLayerContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include <c10/core/Event.h>
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include <grpc++/grpc++.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class Meta;
class P2PConnectorPrefill;
class P2PConnectorDecode;
class P2PConnectorResourceStore;
class PrefillResultStore;

/**
 * Q: 如何保证kvcache不被写坏
 * A: scheduler 应该在 worker 执行完成之前都持有kv_cache资源, 如果在worker执行过程中, scheduler 因为等待 worker 超时 /
 * RPC失败等原因退出等待，释放worker资源，那么应该abort 目前这部分是在 scheduler 对 worker
 * 的调用中实现，如果调用超时或rpc失败，则scheduler会abort进程
 * Q: 超时处理
 * A:
 * 每个stream都会有自己的超时，worker的实现逻辑中会尽量保证在超时后尽快终止后续的可能操作，以尽快完成资源释放，但是不保证一定能在deadline之前完成操作.
 */
class P2PConnector {
public:
    P2PConnector(P2PConnectorConfig                          config,
                 const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                 const kmonitor::MetricsReporterPtr&         metrics_reporter);
    ~P2PConnector();

public:
    bool init();

    // Expose the Prefill resource store for integration and testing.
    std::shared_ptr<P2PConnectorResourceStore> streamStore() const;
    std::shared_ptr<PrefillResultStore> resultStore() const;

public:
    std::shared_ptr<AsyncContext> asyncRead(const KVCacheResourcePtr&    resource,
                                            const std::shared_ptr<Meta>& meta,
                                            int                          start_read_block_index,
                                            int                          read_block_num);

    std::shared_ptr<AsyncContext>
    asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context);

    bool writeByLayerTag(int                                   layer_id,
                         const std::string&                    tag,
                         const KVCacheResourcePtr&             resource,
                         int64_t                               request_id,
                         const std::shared_ptr<c10::Event>& event,
                         int64_t                               deadline_ms);

public:
    void handleRead(const P2PConnectorStartLoadRequestPB& request,
                    P2PConnectorStartLoadResponsePB&      response,
                    std::function<bool()>                 is_cancelled = nullptr);

    bool executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response);
    void cancelRead(const std::shared_ptr<AsyncContext>& context);

private:
    const P2PConnectorConfig             config_;
    std::shared_ptr<LayerBlockConverter> layer_block_converter_;
    kmonitor::MetricsReporterPtr         metrics_reporter_;

    std::unique_ptr<P2PConnectorPrefill> prefill_;
    std::unique_ptr<P2PConnectorDecode>  decode_;
};

}  // namespace rtp_llm
