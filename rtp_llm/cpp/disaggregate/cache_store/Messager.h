#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/MessagerRequest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TimerManager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"
#include "rtp_llm/cpp/disaggregate/cache_store/LockedBlockBufferManager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/InitParams.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpClient.h"

namespace rtp_llm {

class Messager {
public:
    Messager(const std::shared_ptr<MemoryUtil>&              memory_util,
             const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store,
             const kmonitor::MetricsReporterPtr&             metrics_reporter):
        memory_util_(memory_util),
        request_block_buffer_store_(request_block_buffer_store),
        metrics_reporter_(metrics_reporter),
        timer_manager_(new TimerManager),
        locked_block_buffer_manager_(new LockedBlockBufferManager) {}
    virtual ~Messager() = default;

public:
    virtual bool init(MessagerInitParams params) = 0;

    virtual void load(const std::shared_ptr<LoadRequest>&                          request,
                      const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector) = 0;
    virtual void transfer(const std::shared_ptr<TransferRequest>& request);

    const std::shared_ptr<LockedBlockBufferManager>& getLockedBlockBufferManager() const {
        return locked_block_buffer_manager_;
    }

protected:
    CacheLoadRequest*     makeLoadRequest(const std::shared_ptr<LoadRequest>& request);
    CacheTransferRequest* makeTransferRequest(const std::shared_ptr<TransferRequest>& request);

    virtual bool generateBlockInfo(::BlockBufferInfo*                  block_info,
                                   const std::shared_ptr<BlockBuffer>& block_buffer,
                                   uint32_t                            partition_count,
                                   uint32_t                            partition_id) = 0;

protected:
    std::shared_ptr<MemoryUtil>               memory_util_;
    std::shared_ptr<RequestBlockBufferStore>  request_block_buffer_store_;
    kmonitor::MetricsReporterPtr              metrics_reporter_;
    std::shared_ptr<TimerManager>             timer_manager_;
    std::shared_ptr<LockedBlockBufferManager> locked_block_buffer_manager_;
    MessagerInitParams                        init_params_;
    std::shared_ptr<TcpClient>                tcp_client_;
};
}  // namespace rtp_llm