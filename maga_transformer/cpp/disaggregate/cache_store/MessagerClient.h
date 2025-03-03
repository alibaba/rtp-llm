#pragma once

#include "maga_transformer/cpp/disaggregate/cache_store/CommonDefine.h"
#include "maga_transformer/cpp/disaggregate/cache_store/InitParams.h"
#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "maga_transformer/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "maga_transformer/cpp/disaggregate/cache_store/metrics/CacheStoreMetricsCollector.h"

#include "aios/network/anet/transport.h"
#include "aios/network/arpc/arpc/ANetRPCChannelManager.h"

namespace rtp_llm {

class MessagerClient {
public:
    MessagerClient(const std::shared_ptr<MemoryUtil>& memory_util);
    virtual ~MessagerClient();

public:
    virtual bool init(uint32_t connect_port, uint32_t rdma_connect_port, bool enable_metric);
    virtual void load(const std::string&                                           ip,
                      const std::shared_ptr<RequestBlockBuffer>&                   request_block_buffer,
                      CacheStoreLoadDoneCallback                                   callback,
                      uint32_t                                                     timeout_ms,
                      const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector,
                      int partition_count, int partition_id);

private:
    bool initTcpClient(bool enable_metric);
    void stopTcpClient();

    std::shared_ptr<arpc::RPCChannelBase> openChannel(const std::string& ip);
    CacheLoadRequest* makeLoadRequest(const std::shared_ptr<RequestBlockBuffer>& cache, uint32_t timeout_ms, int partition_count, int partition_id);

protected:
    std::shared_ptr<arpc::RPCChannelBase> getChannel(const std::string& ip);

protected:
    std::shared_ptr<MemoryUtil> memory_util_;

private:
    uint32_t                                                               connect_port_{0};
    std::unique_ptr<anet::Transport>                                       rpc_channel_transport_;
    std::shared_ptr<arpc::ANetRPCChannelManager>                           rpc_channel_manager_;
    std::mutex                                                             channel_map_mutex_;
    std::unordered_map<std::string, std::shared_ptr<arpc::RPCChannelBase>> channel_map_;
};

}  // namespace rtp_llm