#pragma once

#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"

namespace rtp_llm {

inline ErrorCode transCacheStoreErrorCode(CacheStoreErrorCode error_code) {
    const static std::unordered_map<CacheStoreErrorCode, ErrorCode> error_code_map = {
        {CacheStoreErrorCode::None, ErrorCode::NONE_ERROR},
        {CacheStoreErrorCode::InvalidParams, ErrorCode::INVALID_PARAMS},
        {CacheStoreErrorCode::PushWorkerItemFailed, ErrorCode::CACHE_STORE_PUSH_ITEM_FAILED},
        {CacheStoreErrorCode::LoadConnectFailed, ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED},
        {CacheStoreErrorCode::LoadSendRequestFailed, ErrorCode::CACHE_STORE_LOAD_SEND_REQUEST_FAILED},
        {CacheStoreErrorCode::CallPrefillTimeout, ErrorCode::CACHE_STORE_CALL_PREFILL_TIMEOUT},
        {CacheStoreErrorCode::LoadRdmaConnectFailed, ErrorCode::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED},
        {CacheStoreErrorCode::LoadRdmaWriteFailed, ErrorCode::CACHE_STORE_LOAD_RDMA_WRITE_FAILED},
        {CacheStoreErrorCode::LoadBufferTimeout, ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT},
        {CacheStoreErrorCode::LoadErrorUnknown, ErrorCode::CACHE_STORE_LOAD_UNKNOWN_ERROR},
        {CacheStoreErrorCode::StoreFailed, ErrorCode::CACHE_STORE_STORE_FAILED},
    };
    auto it = error_code_map.find(error_code);
    if (it != error_code_map.end()) {
        return it->second;
    } else {
        return ErrorCode::UNKNOWN_ERROR;
    }
}

}  // namespace rtp_llm
