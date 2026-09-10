#pragma once

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

inline bool isRetryableCacheStoreConnectError(ErrorCode error_code) {
    return error_code == ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED
           || error_code == ErrorCode::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED;
}

}  // namespace rtp_llm
