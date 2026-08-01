#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/connector/remote_connector/RemoteConnector.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/utils/Exception.h"

namespace rtp_llm::test {

TEST(RemoteConnectorStrictSingleGroupTest, RejectsHeterogeneousTopologyAtConstruction) {
    auto config = makeSimpleHybridMhaCacheConfig(/*layer_num=*/4,
                                                 /*block_num=*/8,
                                                 /*tokens_per_block=*/4,
                                                 DataType::TYPE_FP16,
                                                 /*group_layer_num=*/2,
                                                 /*local_head_num_kv=*/2,
                                                 /*size_per_head=*/8);

    EXPECT_THROW((void)RemoteConnector(config,
                                       KVCacheConfig{},
                                       RuntimeConfig{},
                                       ParallelismConfig{},
                                       SpeculativeExecutionConfig{},
                                       nullptr,
                                       0,
                                       nullptr),
                 RTPException);
}

TEST(RemoteConnectorStrictSingleGroupTest, AcceptsOneNativeGroup) {
    auto config = makeSimpleMhaCacheConfig(/*layer_num=*/2,
                                           /*block_num=*/8,
                                           /*tokens_per_block=*/4,
                                           DataType::TYPE_FP16,
                                           /*local_head_num_kv=*/2,
                                           /*size_per_head=*/8);

    EXPECT_NO_THROW((void)RemoteConnector(config,
                                          KVCacheConfig{},
                                          RuntimeConfig{},
                                          ParallelismConfig{},
                                          SpeculativeExecutionConfig{},
                                          nullptr,
                                          0,
                                          nullptr));
}

}  // namespace rtp_llm::test
