#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreServiceImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"

namespace rtp_llm {

class TcpCacheStoreServiceImpl: public CacheStoreServiceImpl {
public:
    TcpCacheStoreServiceImpl(const std::shared_ptr<MemoryUtil>&              memory_util,
                             const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store,
                             const kmonitor::MetricsReporterPtr&             metrics_reporter,
                             const std::shared_ptr<arpc::TimerManager>&      timer_manager);
    virtual ~TcpCacheStoreServiceImpl() = default;

protected:
    void loadImpl(::google::protobuf::RpcController* controller,
                  const ::CacheLoadRequest*          request,
                  ::CacheLoadResponse*               response,
                  ::google::protobuf::Closure*       done) override;

    void loadTcpBlocks(const ::CacheLoadRequest*                                    request,
                       ::CacheLoadResponse*                                         response,
                       const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector,
                       ::google::protobuf::Closure*                                 done);
};

}  // namespace rtp_llm