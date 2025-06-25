#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TimerManager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/LockedBlockBufferManager.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class CacheStoreServiceImpl: public KvCacheStoreService {
public:
    CacheStoreServiceImpl(const std::shared_ptr<MemoryUtil>&               memory_util,
                          const std::shared_ptr<RequestBlockBufferStore>&  request_block_buffer_store,
                          const kmonitor::MetricsReporterPtr&              metrics_reporter,
                          const std::shared_ptr<TimerManager>&             timer_manager,
                          const std::shared_ptr<LockedBlockBufferManager>& locked_block_buffer_manager);
    virtual ~CacheStoreServiceImpl() = default;

    void load(::google::protobuf::RpcController* controller,
              const ::CacheLoadRequest*          request,
              ::CacheLoadResponse*               response,
              ::google::protobuf::Closure*       done) override;

    void transfer(::google::protobuf::RpcController* controller,
                  const ::CacheTransferRequest*      request,
                  CacheTransferResponse*             response,
                  ::google::protobuf::Closure*       done) override;

    void blockRead(::google::protobuf::RpcController* controller,
                   const ::BlockReadRequest*          request,
                   BlockReadResponse*                 response,
                   ::google::protobuf::Closure*       done) override;

protected:
    virtual void loadImpl(::google::protobuf::RpcController* controller,
                          const ::CacheLoadRequest*          request,
                          ::CacheLoadResponse*               response,
                          ::google::protobuf::Closure*       done)                                      = 0;
    virtual void transferImpl(::google::protobuf::RpcController*                   controller,
                              const ::CacheTransferRequest*                        request,
                              CacheTransferResponse*                               response,
                              ::google::protobuf::Closure*                         done,
                              const std::vector<std::shared_ptr<BlockBuffer>>&     local_blocks,
                              const std::vector<std::shared_ptr<BlockBufferInfo>>& remote_blocks) = 0;
    virtual void blockReadImpl(::google::protobuf::RpcController* controller,
                               const ::BlockReadRequest*          request,
                               BlockReadResponse*                 response,
                               ::google::protobuf::Closure*       done)                                 = 0;

protected:
    std::shared_ptr<MemoryUtil>               memory_util_;
    std::shared_ptr<RequestBlockBufferStore>  request_block_buffer_store_;
    kmonitor::MetricsReporterPtr              metrics_reporter_;
    std::shared_ptr<TimerManager>             timer_manager_;
    std::shared_ptr<LockedBlockBufferManager> locked_block_buffer_manager_;
};

}  // namespace rtp_llm