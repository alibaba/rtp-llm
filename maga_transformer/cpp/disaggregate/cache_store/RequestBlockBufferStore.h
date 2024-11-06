#pragma once

#include <unordered_map>
#include <shared_mutex>
#include <atomic>
#include <mutex>
#include <functional>
#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"

namespace rtp_llm {

typedef std::function<void(std::shared_ptr<BlockBuffer>)> StoreBlockBufferCallbackFunc;
/**
 * RequestBlockBufferStore 用于存储请求对应的block
 *
 */
class RequestBlockBufferStore {

public:
    RequestBlockBufferStore(const std::shared_ptr<MemoryUtil>& memory_util, void* stream);
    ~RequestBlockBufferStore() = default;

public:
    bool setRequestBlockBuffer(const std::shared_ptr<RequestBlockBuffer>& layer_cache);

    std::shared_ptr<BlockBuffer> getBlockBuffer(const std::string& requestid, const std::string& blockid) const;

    void delRequestBlockBuffer(const std::string& requestid);

    void debugInfo();

    void setStoreBlockBufferCallBack(const std::string& requestid, StoreBlockBufferCallbackFunc&& callback);
    void runStoreBlockBufferCallBack(const std::string& requestid, const std::shared_ptr<BlockBuffer>& block);

private:
    std::shared_ptr<RequestBlockBuffer> getRequestBlockBuffer(const std::string& requestid) const;
    std::shared_ptr<RequestBlockBuffer> getOrInsertRequestBlockBuffer(const std::string& requestid);
    bool                                isValidBlock(const std::shared_ptr<BlockBuffer>& block);
    std::shared_ptr<BlockBuffer>        makeValidBlock(const std::shared_ptr<BlockBuffer>& block);
    bool copyBlock(const std::shared_ptr<BlockBuffer>& dst, const std::shared_ptr<BlockBuffer>& src);

private:
    std::shared_ptr<MemoryUtil> memory_util_;
    void*                       stream_;

    struct RequestBlockBufferInfo{
        std::shared_ptr<RequestBlockBuffer>        block_buffer_;
        StoreBlockBufferCallbackFunc      callback_;
        RequestBlockBufferInfo(std::shared_ptr<RequestBlockBuffer> buffer, StoreBlockBufferCallbackFunc func) 
        : block_buffer_(buffer), callback_(func) {}
    };

    mutable std::shared_mutex                                            request_cache_map_mutex_;
    std::unordered_map<std::string, std::shared_ptr<RequestBlockBufferInfo>> request_cache_map_;
};

}  // namespace rtp_llm