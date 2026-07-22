#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/LoadCopyFence.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/arpc/arpc/ANetRPCController.h"

namespace rtp_llm {

class TcpCacheStoreLoadServiceClosure: public RPCClosure {
public:
    TcpCacheStoreLoadServiceClosure(const std::shared_ptr<MemoryUtil>&                           memory_util,
                                    const std::shared_ptr<RequestBlockBuffer>                    request_block_buffer,
                                    arpc::ANetRPCController*                                     controller,
                                    CacheLoadRequest*                                            request,
                                    CacheLoadResponse*                                           response,
                                    CacheStoreLoadDoneCallback                                   callback,
                                    const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector,
                                    int                                                          device_id,
                                    const std::shared_ptr<LoadCopyFence>&                         copy_fence = nullptr):
        memory_util_(memory_util),
        request_block_buffer_(request_block_buffer),
        controller_(controller),
        request_(request),
        response_(response),
        copy_fence_(copy_fence),
        callback_(callback),
        collector_(collector),
        device_id_(device_id) {}

    ~TcpCacheStoreLoadServiceClosure();

public:
    void Run() override;

private:
    void end(bool success, CacheStoreErrorCode ec);
    CacheStoreErrorCode copyResponseBlocks();

private:
    std::shared_ptr<MemoryUtil>                           memory_util_;
    std::shared_ptr<RequestBlockBuffer>                   request_block_buffer_;
    arpc::ANetRPCController*                              controller_{nullptr};
    CacheLoadRequest*                                     request_{nullptr};
    CacheLoadResponse*                                    response_{nullptr};
    std::shared_ptr<LoadCopyFence>                         copy_fence_;
    CacheStoreLoadDoneCallback                            callback_{nullptr};
    std::shared_ptr<CacheStoreClientLoadMetricsCollector> collector_;
    int                                                   device_id_{-1};
};

}  // namespace rtp_llm
