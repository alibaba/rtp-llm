#pragma once

#include "maga_transformer/cpp/disaggregate/cache_store/InitParams.h"
#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "maga_transformer/cpp/disaggregate/cache_store/CacheStoreServiceImpl.h"
#include "maga_transformer/cpp/disaggregate/cache_store/metrics/CacheStoreMetricsReporter.h"
#include "aios/network/anet/transport.h"
#include "aios/network/arpc/arpc/ANetRPCServer.h"

namespace rtp_llm {

class MessagerServer {
public:
    MessagerServer(const std::shared_ptr<MemoryUtil>&                memory_util,
                             const std::shared_ptr<RequestBlockBufferStore>&   request_block_buffer_store,
                             const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter);
    ~MessagerServer();

public:
    virtual bool init(uint32_t listen_port, bool enable_metric);

private:
    bool initTcpServer(uint32_t listen_port, bool enable_metric);

private:
    std::shared_ptr<MemoryUtil>                memory_util_;
    std::shared_ptr<RequestBlockBufferStore>   request_block_buffer_store_;
    std::shared_ptr<CacheStoreMetricsReporter> metrics_reporter_;

    std::unique_ptr<anet::Transport>       rpc_server_transport_;
    std::shared_ptr<arpc::ANetRPCServer>   rpc_server_;
    AUTIL_LOG_DECLARE();

protected:
    std::unique_ptr<CacheStoreServiceImpl> rpc_service_;    
};

}  // namespace rtp_llm