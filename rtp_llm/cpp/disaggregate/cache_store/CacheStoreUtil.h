#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"

namespace rtp_llm {

class CacheStoreUtil {
public:
    static CacheStoreErrorCode          fromArpcErrorCode(arpc::ErrorCode error_code);
    static CacheStoreErrorCode          fromKvCacheStoreErrorCode(KvCacheStoreServiceErrorCode error_code);
    static KvCacheStoreServiceErrorCode toKvCacheStoreErrorCode(CacheStoreErrorCode error_code);
};

}  // namespace rtp_llm