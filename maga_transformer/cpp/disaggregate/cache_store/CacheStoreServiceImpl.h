#pragma once

#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "maga_transformer/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "maga_transformer/cpp/disaggregate/cache_store/metrics/CacheStoreMetricsReporter.h"
#include "autil/Log.h"

namespace rtp_llm {

class CacheStoreServiceImpl: public KvCacheStoreService {
public:
    CacheStoreServiceImpl(const std::shared_ptr<MemoryUtil>&                memory_util,
                          const std::shared_ptr<RequestBlockBufferStore>&   request_block_buffer_store,
                          const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter);
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

private:
    AUTIL_LOG_DECLARE();
};

class TcpCacheStoreServiceImpl: public CacheStoreServiceImpl {
public:
    TcpCacheStoreServiceImpl(const std::shared_ptr<MemoryUtil>&                memory_util,
                             const std::shared_ptr<RequestBlockBufferStore>&   request_block_buffer_store,
                             const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter);
    virtual ~TcpCacheStoreServiceImpl() = default;

protected:
    void loadImpl(::google::protobuf::RpcController* controller,
                  const ::CacheLoadRequest*          request,
                  ::CacheLoadResponse*               response,
                  ::google::protobuf::Closure*       done) override;

    KvCacheStoreServiceErrorCode loadTcpBlocks(const ::CacheLoadRequest*                                    request,
                                               ::CacheLoadResponse*                                         response,
                                               const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector);

private:
    AUTIL_LOG_DECLARE();
};

}  // namespace rtp_llm