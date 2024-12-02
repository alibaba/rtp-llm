#pragma once

#include <unordered_map>
#include <shared_mutex>
#include <atomic>
#include <mutex>
#include <functional>
#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "src/fastertransformer/devices/DeviceBase.h"

namespace rtp_llm {

/**
 * RequestBlockBufferStore 用于存储请求对应的block
 *
 */
class RequestBlockBufferStore {

public:
    RequestBlockBufferStore(const std::shared_ptr<MemoryUtil>& memory_util, fastertransformer::DeviceBase* device);
    ~RequestBlockBufferStore() = default;

public:
    bool setRequestBlockBuffer(const std::shared_ptr<RequestBlockBuffer>& layer_cache);
    bool setRequestBlockBufferWatchFunc(const std::string& requestid, RequestBlockBuffer::WatchFunc&& func);

    std::shared_ptr<BlockBuffer> getBlockBuffer(const std::string& requestid, const std::string& blockid) const;

    void delRequestBlockBuffer(const std::string& requestid);

    std::string debugInfoOnRequest(const std::string& requestid) const;
    void        debugInfo();

private:
    std::shared_ptr<RequestBlockBuffer> getRequestBlockBuffer(const std::string& requestid) const;
    std::shared_ptr<RequestBlockBuffer> getOrInsertRequestBlockBuffer(const std::string& requestid);
    bool                                isValidBlock(const std::shared_ptr<BlockBuffer>& block);
    std::shared_ptr<BlockBuffer>        makeValidBlock(const std::shared_ptr<BlockBuffer>& block);
    bool copyBlock(const std::shared_ptr<BlockBuffer>& dst, const std::shared_ptr<BlockBuffer>& src);

private:
    std::shared_ptr<MemoryUtil> memory_util_;
    fastertransformer::DeviceBase*                 device_;

    mutable std::shared_mutex                                            request_cache_map_mutex_;
    std::unordered_map<std::string, std::shared_ptr<RequestBlockBuffer>> request_cache_map_;
};

}  // namespace rtp_llm