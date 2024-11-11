#pragma once
#include "maga_transformer/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "maga_transformer/cpp/disaggregate/cache_store/metrics/CacheStoreMetricsCollector.h"
#include "maga_transformer/cpp/disaggregate/cache_store/TimerManager.h"
#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include <shared_mutex>

namespace rtp_llm {

class CacheStoreServiceImplContext: public std::enable_shared_from_this<CacheStoreServiceImplContext> {
public:
    CacheStoreServiceImplContext(const CacheLoadRequest*                                      request,
                                 CacheLoadResponse*                                           response,
                                 const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector,
                                 ::google::protobuf::Closure*                                 done,
                                 const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store);
    virtual ~CacheStoreServiceImplContext();

public:
    void loadBlockOnTcp(bool ok, const std::vector<std::shared_ptr<BlockBuffer>>& block);
    void setTimer(const std::shared_ptr<arpc::Timer>& timer) {
        timer_ = std::weak_ptr<arpc::Timer>(timer);
    }
    void runFailed(KvCacheStoreServiceErrorCode error_code);

protected:
    std::shared_ptr<BlockBufferInfo> getAndEraseUnLoadedBlock(const std::string& block_key);
    void                             stopTimer();
    void                             runSuccess(bool direct_write);

private:
    bool writeResponseBlock(const std::shared_ptr<BlockBuffer>&     block,
                            const std::shared_ptr<BlockBufferInfo>& peer_block);

protected:
    const CacheLoadRequest* request_;
    const int64_t           request_send_start_time_us_{0};
    const uint32_t          total_block_count_{0};
    const std::string       request_id_;
    const std::string       peer_ip_;

    std::mutex         response_mutex_;
    CacheLoadResponse* response_;

    std::shared_ptr<CacheStoreServerLoadMetricsCollector> collector_;

    std::atomic_bool             done_run_{false};
    ::google::protobuf::Closure* done_;

    std::weak_ptr<RequestBlockBufferStore> request_block_buffer_store_;

    std::weak_ptr<arpc::Timer> timer_;

    std::atomic_int write_cnt_{0};

    std::shared_mutex                                                 unloaded_blocks_mutex_;
    std::unordered_map<std::string, std::shared_ptr<BlockBufferInfo>> unloaded_blocks_;
};
}  // namespace rtp_llm