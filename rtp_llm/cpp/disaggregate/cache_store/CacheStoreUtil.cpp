#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreUtil.h"

namespace rtp_llm {

CacheStoreErrorCode CacheStoreUtil::fromArpcErrorCode(arpc::ErrorCode error_code) {
    switch (error_code) {
        case arpc::ARPC_ERROR_TIMEOUT:
            return CacheStoreErrorCode::CallPrefillTimeout;
        case arpc::ARPC_ERROR_CONNECTION_CLOSED:
        case arpc::ARPC_ERROR_METHOD_NOT_FOUND:
        case arpc::ARPC_ERROR_POST_PACKET:
            return CacheStoreErrorCode::LoadSendRequestFailed;
        case arpc::ARPC_ERROR_PUSH_WORKITEM:
        case arpc::ARPC_ERROR_QUEUE_FULL:
            return CacheStoreErrorCode::PushWorkerItemFailed;
        default:
            return CacheStoreErrorCode::LoadErrorUnknown;
    }
}
CacheStoreErrorCode CacheStoreUtil::fromKvCacheStoreErrorCode(KvCacheStoreServiceErrorCode error_code) {
    switch (error_code) {
        case KvCacheStoreServiceErrorCode::EC_SUCCESS:
            return CacheStoreErrorCode::None;
        case KvCacheStoreServiceErrorCode::EC_FAILED_INVALID_REQ:
            return CacheStoreErrorCode::LoadSendRequestFailed;
        case KvCacheStoreServiceErrorCode::EC_FAILED_RDMA_CONNECTION:
            return CacheStoreErrorCode::LoadRdmaConnectFailed;
        case KvCacheStoreServiceErrorCode::EC_FAILED_RDMA_WRITE:
            return CacheStoreErrorCode::LoadRdmaWriteFailed;
        case KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER:
            return CacheStoreErrorCode::LoadBufferTimeout;
        default:
            return CacheStoreErrorCode::LoadErrorUnknown;
    }
}
KvCacheStoreServiceErrorCode CacheStoreUtil::toKvCacheStoreErrorCode(CacheStoreErrorCode error_code) {
    switch (error_code) {
        case CacheStoreErrorCode::None:
            return KvCacheStoreServiceErrorCode::EC_SUCCESS;
        case CacheStoreErrorCode::LoadSendRequestFailed:
            return KvCacheStoreServiceErrorCode::EC_FAILED_INVALID_REQ;
        case CacheStoreErrorCode::LoadRdmaConnectFailed:
            return KvCacheStoreServiceErrorCode::EC_FAILED_RDMA_CONNECTION;
        case CacheStoreErrorCode::LoadRdmaWriteFailed:
            return KvCacheStoreServiceErrorCode::EC_FAILED_RDMA_WRITE;
        case CacheStoreErrorCode::LoadBufferTimeout:
            return KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER;
        default:
            return KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL;
    }
}

}  // namespace rtp_llm