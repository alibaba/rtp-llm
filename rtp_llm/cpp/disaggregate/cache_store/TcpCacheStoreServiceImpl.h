#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreServiceImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpClient.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"

namespace rtp_llm {

class TcpCacheStoreServiceImpl: public CacheStoreServiceImpl {
public:
    TcpCacheStoreServiceImpl(const std::shared_ptr<MemoryUtil>&               memory_util,
                             const std::shared_ptr<RequestBlockBufferStore>&  request_block_buffer_store,
                             const kmonitor::MetricsReporterPtr&              metrics_reporter,
                             const std::shared_ptr<TimerManager>&             timer_manager,
                             const std::shared_ptr<LockedBlockBufferManager>& locked_block_buffer_manager,
                             const std::shared_ptr<TcpClient>&                tcp_client);
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

    void transferImpl(::google::protobuf::RpcController*                   controller,
                      const ::CacheTransferRequest*                        request,
                      CacheTransferResponse*                               response,
                      ::google::protobuf::Closure*                         done,
                      const std::vector<std::shared_ptr<BlockBuffer>>&     local_blocks,
                      const std::vector<std::shared_ptr<BlockBufferInfo>>& remote_blocks) override;

    void blockReadImpl(::google::protobuf::RpcController* controller,
                       const ::BlockReadRequest*          request,
                       BlockReadResponse*                 response,
                       ::google::protobuf::Closure*       done) override;

private:
    std::shared_ptr<TcpClient> tcp_client_;
    rtp_llm::DeviceBase*       device_{nullptr};
};

}  // namespace rtp_llm