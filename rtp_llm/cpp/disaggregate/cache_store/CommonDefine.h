#pragma once

#include <string>
#include <functional>
#include <stdint.h>

namespace rtp_llm {

enum class CacheStoreErrorCode {
    None = 0,

    // common error
    InvalidParams        = 1,
    PushWorkerItemFailed = 2,

    // load failed error
    LoadConnectFailed     = 3,
    LoadSendRequestFailed = 4,
    CallPrefillTimeout    = 5,
    LoadRdmaConnectFailed = 6,
    LoadRdmaWriteFailed   = 7,
    LoadBufferTimeout     = 8,
    LoadErrorUnknown      = 9,

    // store failed
    StoreFailed = 10,
};

inline std::string CacheStoreErrorCodeToString(CacheStoreErrorCode code) {
    switch (code) {
        case CacheStoreErrorCode::None:
            return "None";
        case CacheStoreErrorCode::InvalidParams:
            return "InvalidParams";
        case CacheStoreErrorCode::PushWorkerItemFailed:
            return "PushWorkerItemFailed";
        case CacheStoreErrorCode::LoadConnectFailed:
            return "LoadConnectFailed";
        case CacheStoreErrorCode::LoadSendRequestFailed:
            return "LoadSendRequestFailed";
        case CacheStoreErrorCode::CallPrefillTimeout:
            return "CallPrefillTimeout";
        case CacheStoreErrorCode::LoadRdmaConnectFailed:
            return "LoadRdmaConnectFailed";
        case CacheStoreErrorCode::LoadRdmaWriteFailed:
            return "LoadRdmaWriteFailed";
        case CacheStoreErrorCode::LoadBufferTimeout:
            return "LoadBufferTimeout";
        case CacheStoreErrorCode::LoadErrorUnknown:
            return "LoadErrorUnknown";
        case CacheStoreErrorCode::StoreFailed:
            return "StoreFailed";
        default:
            return "Error: Unrecognized ErrorCode";
    }
}

typedef std::function<void(bool, CacheStoreErrorCode)> CacheStoreStoreDoneCallback;
typedef std::function<void(bool, CacheStoreErrorCode)> CacheStoreLoadDoneCallback;
typedef std::function<void(bool)>                      WriteBlockDoneCallback;
const std::string                                      kEnvRdmaMode             = "CACHE_STORE_RDMA_MODE";
const std::string                                      kEnvRdmaWriteBlockConcat = "CACHE_STORE_RDMA_WRITE_BLOCK_CONCAT";
const uint32_t                                         kTcpRdmaPortDiff         = 100;

}  // namespace rtp_llm