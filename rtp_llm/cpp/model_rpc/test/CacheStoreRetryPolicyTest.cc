#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/CacheStoreRetryPolicy.h"

namespace rtp_llm {

TEST(CacheStoreRetryPolicyTest, CacheStoreConnectErrorsAreRetryable) {
    EXPECT_TRUE(isRetryableCacheStoreConnectError(ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED));
    EXPECT_TRUE(isRetryableCacheStoreConnectError(ErrorCode::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED));
}

TEST(CacheStoreRetryPolicyTest, NonConnectErrorsAreNotRetryable) {
    EXPECT_FALSE(isRetryableCacheStoreConnectError(ErrorCode::NONE_ERROR));
    EXPECT_FALSE(isRetryableCacheStoreConnectError(ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT));
    EXPECT_FALSE(isRetryableCacheStoreConnectError(ErrorCode::CACHE_STORE_LOAD_RDMA_WRITE_FAILED));
    EXPECT_FALSE(isRetryableCacheStoreConnectError(ErrorCode::LOAD_CACHE_TIMEOUT));
}

}  // namespace rtp_llm
