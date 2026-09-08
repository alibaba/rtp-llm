#pragma once

#include <unordered_map>
#include <shared_mutex>
#include <atomic>
#include <mutex>
#include <functional>
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"

namespace rtp_llm {

/**
 * RequestBlockBufferStore 用于存储请求对应的block
 *
 */
class RequestBlockBufferStore {

public:
    RequestBlockBufferStore(const std::shared_ptr<MemoryUtil>& memory_util);
    ~RequestBlockBufferStore() = default;

public:
    void stop();
    bool setRequestBlockBuffer(const std::shared_ptr<RequestBlockBuffer>& layer_cache);
    bool setRequestBlockBufferWatchFunc(const std::string& requestid, RequestBlockBuffer::WatchFunc&& func);

    std::shared_ptr<BlockBuffer> getBlockBuffer(const std::string& requestid, const std::string& blockid) const;

    void delRequestBlockBuffer(const std::string& requestid);

    std::string debugInfoOnRequest(const std::string& requestid) const;
    void        debugInfo();

    bool                         regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>& buffers);
    std::shared_ptr<BlockBuffer> findUserBuffer(const std::string& buffer_key);

    // Test hook only: shrink the tombstone TTL so reclamation can be exercised
    // without sleeping for the production default (one hour).
    void setExpiredRequestCacheTtlUsForTest(int64_t ttl_us) {
        expired_request_cache_ttl_us_ = ttl_us;
    }

private:
    std::shared_ptr<RequestBlockBuffer> getRequestBlockBuffer(const std::string& requestid) const;
    std::shared_ptr<RequestBlockBuffer> getOrInsertRequestBlockBuffer(const std::string& requestid);
    bool                                isValidBlock(const std::shared_ptr<BlockBuffer>& block);
    // Stages every block into one pinned host allocation with a single batched copy.
    std::vector<std::shared_ptr<BlockBuffer>> makeValidBlocks(const std::vector<std::shared_ptr<BlockBuffer>>& blocks);

private:
    std::shared_ptr<MemoryUtil> memory_util_;

    mutable std::shared_mutex                                            request_cache_map_mutex_;
    std::unordered_map<std::string, std::shared_ptr<RequestBlockBuffer>> request_cache_map_;
    std::vector<std::pair<std::string, int64_t>>                         expired_request_caches_;
    // currentTimeUs() is microseconds: 1e6 us/s * 3600 s = one hour.
    int64_t expired_request_cache_ttl_us_ = 1000LL * 1000LL * 60LL * 60LL;

    std::shared_mutex                                             buffer_map_mutex_;
    std::unordered_map<std::string, std::shared_ptr<BlockBuffer>> buffer_map_;
};

}  // namespace rtp_llm