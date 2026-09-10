#pragma once
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TimerManager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include <shared_mutex>
#include <vector>

namespace rtp_llm {

class CacheStoreServiceImplContext: public std::enable_shared_from_this<CacheStoreServiceImplContext> {
public:
    CacheStoreServiceImplContext(const CacheLoadRequest*                                      request,
                                 CacheLoadResponse*                                           response,
                                 const std::shared_ptr<CacheStoreServerLoadMetricsCollector>& collector,
                                 ::google::protobuf::Closure*                                 done,
                                 const std::shared_ptr<RequestBlockBufferStore>& request_block_buffer_store);
    virtual ~CacheStoreServiceImplContext() = default;

public:
    void setTimer(const std::shared_ptr<Timer>& timer) {
        timer_ = std::weak_ptr<Timer>(timer);
    }
    void runFailed(KvCacheStoreServiceErrorCode error_code);

protected:
    // A half-open [offset, offset + len) byte range inside a locally stored block.
    struct SourceRange {
        uint32_t offset;
        uint32_t len;
    };

    // Byte ranges of the locally stored block that make up the slice the peer
    // asked for, concatenated in order. Returns false (and logs) when the peer's
    // declared length is not a consistent partition of the stored block, which is
    // the mismatch that used to surface as a bare CACHE_STORE_LOAD_SEND_REQUEST_FAILED.
    bool computeSourceRanges(const std::shared_ptr<BlockBuffer>&     block,
                             const std::shared_ptr<BlockBufferInfo>& peer_block,
                             std::vector<SourceRange>&               ranges) const;

    std::shared_ptr<BlockBufferInfo> getAndEraseUnLoadedBlock(const std::string& block_key);
    void                             stopTimer();
    void                             runSuccess(bool direct_write);

protected:
    const CacheLoadRequest* request_;
    const int64_t           request_send_start_time_us_{0};
    const uint32_t          total_block_count_{0};
    const std::string       request_id_;
    const std::string       peer_ip_;
    const int32_t           partition_count_{1};
    const int32_t           partition_id_{0};

    std::mutex         response_mutex_;
    CacheLoadResponse* response_;

    std::shared_ptr<CacheStoreServerLoadMetricsCollector> collector_;

    std::atomic_bool             done_run_{false};
    ::google::protobuf::Closure* done_;

    std::weak_ptr<RequestBlockBufferStore> request_block_buffer_store_;

    std::weak_ptr<Timer> timer_;

    std::atomic_int write_cnt_{0};

    std::shared_mutex                                                 unloaded_blocks_mutex_;
    std::unordered_map<std::string, std::shared_ptr<BlockBufferInfo>> unloaded_blocks_;
};
}  // namespace rtp_llm