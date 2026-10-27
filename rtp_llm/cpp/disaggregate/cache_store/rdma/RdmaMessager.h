#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/Messager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpServer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClient.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaServer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaCacheStoreServiceImpl.h"

namespace rtp_llm {

class RdmaMessager: public Messager {
public:
    RdmaMessager(const std::shared_ptr<MemoryUtil>&              memory_util,
                 const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store,
                 const kmonitor::MetricsReporterPtr&             metrics_reporter):
        Messager(memory_util, request_block_buffer_store, metrics_reporter) {}
    virtual ~RdmaMessager() = default;

public:
    bool init(MessagerInitParams params) override;
    void load(const std::shared_ptr<LoadRequest>&                          request,
              const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector) override;

private:
    CacheLoadRequest* makeLoadRequest(const std::shared_ptr<RdmaConnection>& rdma_connection,
                                      const std::shared_ptr<LoadRequest>&    request);

    bool generateBlockInfo(::BlockBufferInfo*                  block_info,
                           const std::shared_ptr<BlockBuffer>& block_buffer,
                           uint32_t                            local_partition_count,
                           uint32_t                            local_partition_id) override;

private:
    std::shared_ptr<TcpServer>                 tcp_server_;
    std::shared_ptr<RdmaClient>                rdma_client_;
    std::shared_ptr<RdmaServer>                rdma_server_;
    std::shared_ptr<RdmaCacheStoreServiceImpl> service_;
};

}  // namespace rtp_llm