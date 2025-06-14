#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store/metrics/CacheStoreMetricsReporter.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TimerManager.h"

namespace rtp_llm {

class CacheStoreServiceImpl: public KvCacheStoreService {
public:
    CacheStoreServiceImpl(const std::shared_ptr<MemoryUtil>&                memory_util,
                          const std::shared_ptr<RequestBlockBufferStore>&   request_block_buffer_store,
                          const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter,
                          const std::shared_ptr<arpc::TimerManager>&        timer_manager);
    virtual ~CacheStoreServiceImpl() = default;

    void load(::google::protobuf::RpcController* controller,
              const ::CacheLoadRequest*          request,
              ::CacheLoadResponse*               response,
              ::google::protobuf::Closure*       done) override;

protected:
    virtual void loadImpl(::google::protobuf::RpcController* controller,
                          const ::CacheLoadRequest*          request,
                          ::CacheLoadResponse*               response,
                          ::google::protobuf::Closure*       done) = 0;

protected:
    std::shared_ptr<MemoryUtil>                memory_util_;
    std::shared_ptr<RequestBlockBufferStore>   request_block_buffer_store_;
    std::shared_ptr<CacheStoreMetricsReporter> metrics_reporter_;
    std::shared_ptr<arpc::TimerManager>        timer_manager_;
};

class TcpCacheStoreServiceImpl: public CacheStoreServiceImpl {
public:
    TcpCacheStoreServiceImpl(const std::shared_ptr<MemoryUtil>&                memory_util,
                             const std::shared_ptr<RequestBlockBufferStore>&   request_block_buffer_store,
                             const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter,
                             const std::shared_ptr<arpc::TimerManager>&        timer_manager);
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