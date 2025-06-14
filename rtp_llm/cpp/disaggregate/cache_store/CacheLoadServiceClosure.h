#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/InitParams.h"
#include "rtp_llm/cpp/disaggregate/cache_store/metrics/CacheStoreMetricsCollector.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/arpc/arpc/ANetRPCController.h"

namespace rtp_llm {

class CacheLoadServiceClosure: public RPCClosure {
public:
    CacheLoadServiceClosure(const std::shared_ptr<MemoryUtil>&                           memory_util,
                            const std::shared_ptr<RequestBlockBuffer>                    request_block_buffer,
                            arpc::ANetRPCController*                                     controller,
                            CacheLoadRequest*                                            request,
                            CacheLoadResponse*                                           response,
                            CacheStoreLoadDoneCallback                                   callback,
                            const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector):
        memory_util_(memory_util),
        request_block_buffer_(request_block_buffer),
        controller_(controller),
        request_(request),
        response_(response),
        callback_(callback),
        collector_(collector),
        device_(rtp_llm::DeviceFactory::getDefaultDevice()) {}

    ~CacheLoadServiceClosure();

public:
    void Run() override;

public:
    CacheStoreErrorCode fromArpcErrorCode(arpc::ErrorCode ec);
    CacheStoreErrorCode fromResponseErrorCode(KvCacheStoreServiceErrorCode ec);

    void end(bool success, CacheStoreErrorCode ec);

protected:
    std::shared_ptr<MemoryUtil>                           memory_util_;
    std::shared_ptr<RequestBlockBuffer>                   request_block_buffer_;
    arpc::ANetRPCController*                              controller_{nullptr};
    CacheLoadRequest*                                     request_{nullptr};
    CacheLoadResponse*                                    response_{nullptr};
    CacheStoreLoadDoneCallback                            callback_{nullptr};
    std::shared_ptr<CacheStoreClientLoadMetricsCollector> collector_;
    rtp_llm::DeviceBase*                        device_{nullptr};
};

}  // namespace rtp_llm