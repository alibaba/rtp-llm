#pragma once
#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "maga_transformer/cpp/disaggregate/cache_store/CommonDefine.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"

#include <memory>

namespace rtp_llm {

class CacheStore {

public:
    CacheStore() {};
    virtual ~CacheStore() {};

    virtual void store(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, CacheStoreStoreDoneCallback callback) = 0;

    virtual void load(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
              CacheStoreLoadDoneCallback                 callback,
              const std::string&                         ip         = "",
              uint32_t                                   timeout_ms = 1000) = 0;

    virtual const std::shared_ptr<MemoryUtil>& getMemoryUtil() const = 0;

    virtual void debugInfo() = 0;
};

}