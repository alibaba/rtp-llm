#pragma once

#include <unordered_map>
#include <shared_mutex>
#include <atomic>
#include <mutex>
#include <functional>
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"

namespace rtp_llm {

/**
 * RequestBlockBufferStore 用于存储请求对应的block
 *
 */
class RequestBlockBufferStore {

public:
    RequestBlockBufferStore(const std::shared_ptr<MemoryUtil>& memory_util, rtp_llm::DeviceBase* device);
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

private:
    std::shared_ptr<RequestBlockBuffer> getRequestBlockBuffer(const std::string& requestid) const;
    std::shared_ptr<RequestBlockBuffer> getOrInsertRequestBlockBuffer(const std::string& requestid);
    bool                                isValidBlock(const std::shared_ptr<BlockBuffer>& block);
    std::shared_ptr<BlockBuffer>        makeValidBlock(const std::shared_ptr<BlockBuffer>& block);
    bool copyBlock(const std::shared_ptr<BlockBuffer>& dst, const std::shared_ptr<BlockBuffer>& src);

private:
    std::shared_ptr<MemoryUtil> memory_util_;
    rtp_llm::DeviceBase*        device_;

    mutable std::shared_mutex                                            request_cache_map_mutex_;
    std::unordered_map<std::string, std::shared_ptr<RequestBlockBuffer>> request_cache_map_;
    std::vector<std::pair<std::string, int64_t>>                         expired_request_caches_;

    std::shared_mutex                                             buffer_map_mutex_;
    std::unordered_map<std::string, std::shared_ptr<BlockBuffer>> buffer_map_;
};

}  // namespace rtp_llm