#pragma once

#include <unordered_map>
#include <shared_mutex>
#include <atomic>
#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"

namespace rtp_llm {

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

private:
    std::shared_ptr<RequestBlockBuffer> getRequestBlockBuffer(const std::string& requestid) const;
    std::shared_ptr<RequestBlockBuffer> getOrInsertRequestBlockBuffer(const std::string& requestid);
    bool                                isValidBlock(const std::shared_ptr<BlockBuffer>& block);
    std::shared_ptr<BlockBuffer>        makeValidBlock(const std::shared_ptr<BlockBuffer>& block);
    bool copyBlock(const std::shared_ptr<BlockBuffer>& dst, const std::shared_ptr<BlockBuffer>& src);

private:
    std::shared_ptr<MemoryUtil> memory_util_;
    void*                       stream_;

    mutable std::shared_mutex                                            request_cache_map_mutex_;
    std::unordered_map<std::string, std::shared_ptr<RequestBlockBuffer>> request_cache_map_;

private:
    AUTIL_LOG_DECLARE();
};

}  // namespace rtp_llm