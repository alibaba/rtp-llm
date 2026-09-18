#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/RpcTimeoutUtils.h"

namespace rtp_llm::test {

TEST(RpcTimeoutUtilsTest, UsesRequestTimeoutWhenPositive) {
    EXPECT_EQ(normalizeRpcTimeoutMs(1234, 5678), 1234);
}

TEST(RpcTimeoutUtilsTest, FallsBackToConfiguredMaxTimeoutWhenRequestIsZeroOrNegative) {
    EXPECT_EQ(normalizeRpcTimeoutMs(0, 5678), 5678);
    EXPECT_EQ(normalizeRpcTimeoutMs(-1, 5678), 5678);
}

TEST(RpcTimeoutUtilsTest, FallsBackToDefaultGrpcTimeoutWhenConfigIsNotPositive) {
    EXPECT_EQ(normalizeRpcTimeoutMs(0, 0), kDefaultRpcTimeoutMs);
    EXPECT_EQ(normalizeRpcTimeoutMs(-1, -1), kDefaultRpcTimeoutMs);
}

TEST(RpcTimeoutUtilsTest, ClampRpcTimeoutMsToInt32CapsLargeValues) {
    EXPECT_EQ(clampRpcTimeoutMsToInt32(1234), 1234);
    EXPECT_EQ(clampRpcTimeoutMsToInt32(static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 10),
              std::numeric_limits<int32_t>::max());
}

}  // namespace rtp_llm::test
