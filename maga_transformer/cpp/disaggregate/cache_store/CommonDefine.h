#pragma once

#include <string>
#include <functional>
#include <stdint.h>

namespace rtp_llm {

typedef std::function<void(bool)> CacheStoreStoreDoneCallback;
typedef std::function<void(bool)> CacheStoreLoadDoneCallback;
typedef std::function<void(bool)> WriteBlockDoneCallback;

const std::string kEnvRdmaMode = "CACHE_STORE_RDMA_MODE";

}  // namespace rtp_llm