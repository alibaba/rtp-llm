#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/MockKVCMClient.h"

#include <gtest/gtest.h>

namespace kv_cache_manager {
namespace {
using ::testing::_;

template<class Client, class Mock>
auto checkExplicitReplicaCount(Client& sdk, Mock& mock, int) -> decltype(sdk.StartWrite("", {}, {}, {}, 1, 2), void()) {
    EXPECT_CALL(mock, OnStartWrite(_, _, _, _, 1, 2));
    (void)sdk.StartWrite("", {}, {}, {}, 1, 2);
}

// The older SDK has no explicit replica-count argument.
template<class Client, class Mock>
void checkExplicitReplicaCount(Client&, Mock&, long) {}

TEST(MockMetaClientSdkTest, PreservesSdkDefaultAndExplicitReplicaCount) {
    MockMetaClient mock;
    MetaClient&    sdk = mock;
    EXPECT_CALL(mock, OnStartWrite(_, _, _, _, 1, 0));
    (void)sdk.StartWrite("", {}, {}, {}, 1);
    checkExplicitReplicaCount(sdk, mock, 0);
}
}  // namespace
}  // namespace kv_cache_manager
