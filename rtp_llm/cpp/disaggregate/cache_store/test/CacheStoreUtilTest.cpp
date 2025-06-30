#include "gtest/gtest.h"

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"

namespace rtp_llm {

class CacheStoreUtilTest: public CacheStoreTestBase {};

TEST_F(CacheStoreUtilTest, testFromArpcErrorCode) {
    ASSERT_EQ(CacheStoreErrorCode::CallPrefillTimeout, CacheStoreUtil::fromArpcErrorCode(arpc::ARPC_ERROR_TIMEOUT));

    ASSERT_EQ(CacheStoreErrorCode::LoadSendRequestFailed,
              CacheStoreUtil::fromArpcErrorCode(arpc::ARPC_ERROR_CONNECTION_CLOSED));
    ASSERT_EQ(CacheStoreErrorCode::LoadSendRequestFailed,
              CacheStoreUtil::fromArpcErrorCode(arpc::ARPC_ERROR_METHOD_NOT_FOUND));
    ASSERT_EQ(CacheStoreErrorCode::LoadSendRequestFailed,
              CacheStoreUtil::fromArpcErrorCode(arpc::ARPC_ERROR_POST_PACKET));

    ASSERT_EQ(CacheStoreErrorCode::PushWorkerItemFailed,
              CacheStoreUtil::fromArpcErrorCode(arpc::ARPC_ERROR_PUSH_WORKITEM));
    ASSERT_EQ(CacheStoreErrorCode::PushWorkerItemFailed,
              CacheStoreUtil::fromArpcErrorCode(arpc::ARPC_ERROR_QUEUE_FULL));

    ASSERT_EQ(CacheStoreErrorCode::LoadErrorUnknown, CacheStoreUtil::fromArpcErrorCode(arpc::ARPC_ERROR_APP_MIN));
}

TEST_F(CacheStoreUtilTest, testfromKvCacheStoreErrorCode) {
    ASSERT_EQ(CacheStoreErrorCode::None,
              CacheStoreUtil::fromKvCacheStoreErrorCode(KvCacheStoreServiceErrorCode::EC_SUCCESS));
    ASSERT_EQ(CacheStoreErrorCode::LoadSendRequestFailed,
              CacheStoreUtil::fromKvCacheStoreErrorCode(KvCacheStoreServiceErrorCode::EC_FAILED_INVALID_REQ));
    ASSERT_EQ(CacheStoreErrorCode::LoadRdmaConnectFailed,
              CacheStoreUtil::fromKvCacheStoreErrorCode(KvCacheStoreServiceErrorCode::EC_FAILED_RDMA_CONNECTION));
    ASSERT_EQ(CacheStoreErrorCode::LoadRdmaWriteFailed,
              CacheStoreUtil::fromKvCacheStoreErrorCode(KvCacheStoreServiceErrorCode::EC_FAILED_RDMA_WRITE));
    ASSERT_EQ(CacheStoreErrorCode::LoadBufferTimeout,
              CacheStoreUtil::fromKvCacheStoreErrorCode(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER));
    ASSERT_EQ(CacheStoreErrorCode::LoadErrorUnknown,
              CacheStoreUtil::fromKvCacheStoreErrorCode(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL));
}

}  // namespace rtp_llm