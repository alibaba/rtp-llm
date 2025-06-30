#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/Messager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpClient.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpServer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreServiceImpl.h"

namespace rtp_llm {

class TcpMessager: public Messager {
public:
    TcpMessager(const std::shared_ptr<MemoryUtil>&              memory_util,
                const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store,
                const kmonitor::MetricsReporterPtr&             metrics_reporter):
        Messager(memory_util, request_block_buffer_store, metrics_reporter) {}
    virtual ~TcpMessager() = default;

public:
    bool init(MessagerInitParams params) override;
    void load(const std::shared_ptr<LoadRequest>&                          request,
              const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector) override;

private:
    MessagerInitParams init_params_;

    std::shared_ptr<TcpClient>                tcp_client_;
    std::shared_ptr<TcpServer>                tcp_server_;
    std::shared_ptr<TcpCacheStoreServiceImpl> service_;
};

}  // namespace rtp_llm