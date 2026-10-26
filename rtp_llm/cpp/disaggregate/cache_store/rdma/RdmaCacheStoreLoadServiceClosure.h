#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClient.h"

namespace rtp_llm {

class RdmaCacheStoreLoadServiceClosure: public RPCClosure {
public:
    RdmaCacheStoreLoadServiceClosure(const std::shared_ptr<RdmaClient>&                           rdma_client,
                                     const std::shared_ptr<RdmaConnection>&                       rdma_connection,
                                     const std::shared_ptr<MemoryUtil>&                           memory_util,
                                     const std::shared_ptr<RequestBlockBuffer>                    request_block_buffer,
                                     arpc::ANetRPCController*                                     controller,
                                     CacheLoadRequest*                                            request,
                                     CacheLoadResponse*                                           response,
                                     CacheStoreLoadDoneCallback                                   callback,
                                     const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector);
    virtual ~RdmaCacheStoreLoadServiceClosure();

public:
    void Run() override;

private:
    void recycleRdmaConnection(bool success);
    void end(bool success, CacheStoreErrorCode ec);

private:
    std::weak_ptr<RdmaClient>                             rdma_client_;
    std::shared_ptr<RdmaConnection>                       rdma_connection_;
    std::shared_ptr<MemoryUtil>                           memory_util_;
    std::shared_ptr<RequestBlockBuffer>                   request_block_buffer_;
    arpc::ANetRPCController*                              controller_{nullptr};
    CacheLoadRequest*                                     request_{nullptr};
    CacheLoadResponse*                                    response_{nullptr};
    CacheStoreLoadDoneCallback                            callback_{nullptr};
    std::shared_ptr<CacheStoreClientLoadMetricsCollector> collector_;
};

}  // namespace rtp_llm