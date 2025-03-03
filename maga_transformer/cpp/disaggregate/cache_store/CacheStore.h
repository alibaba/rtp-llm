#pragma once
#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "maga_transformer/cpp/disaggregate/cache_store/CommonDefine.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "maga_transformer/cpp/disaggregate/cache_store/LoadContext.h"

#include <memory>

namespace rtp_llm {

class CacheStore: public std::enable_shared_from_this<CacheStore> {

public:
    CacheStore(){};
    virtual ~CacheStore(){};

    virtual void store(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                       CacheStoreStoreDoneCallback                callback) = 0;

    virtual void load(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                      CacheStoreLoadDoneCallback                 callback,
                      const std::string&                         ip         = "",
                      uint32_t                                   timeout_ms = 1000,
                      int partition_count = 1,
                      int partition_id = 0) = 0;

    virtual std::shared_ptr<LoadContext>
    loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
         const std::string&                                      ip,
         int64_t                                                 timeout_ms,
         LoadContext::CheckCancelFunc                            check_cancel_func,
         int partition_count = 1,
         int partition_id = 0) = 0;

    virtual std::shared_ptr<StoreContext>
    storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers, int64_t timeout_ms) = 0;

    virtual const std::shared_ptr<MemoryUtil>& getMemoryUtil() const = 0;

    virtual void debugInfo() = 0;
};

}  // namespace rtp_llm