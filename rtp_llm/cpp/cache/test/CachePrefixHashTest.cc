#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {
namespace {
CompleteTokenIdsPtr tokens(std::initializer_list<int32_t> values) {
    auto input = std::make_shared<GenerateInput>();
    input->input_ids = torch::tensor(std::vector<int32_t>(values), torch::kInt32);
    auto result = std::make_shared<CompleteTokenIds>(1, 1, 32, 2);
    result->init(input);
    return result;
}
BatchKVCacheResourcePtr resource() {
    auto result = std::make_shared<BatchKVCacheResource>();
    result->resetBatchSize(1);
    return result;
}
TEST(CachePrefixHash, NextTokenDoesNotChangeCompletedPrefix) {
    auto a = resource(), b = resource();
    initCacheKeys(a, tokens({1,2,3,4,5}), 2, 37);
    initCacheKeys(b, tokens({1,2,3,4,90}), 2, 37);
    EXPECT_EQ(a->cacheKeys()[1], b->cacheKeys()[1]);
    int32_t prefix[]{1,2,3,4};
    EXPECT_EQ(a->cacheKeys()[1], hashInt64Array(37, prefix, prefix + 4));
}
TEST(CachePrefixHash, IncrementalCompletedPrefixMatchesFreshConsumer) {
    auto ids = tokens({1,2,3,4,5,6,7}); ids->setSeqLength(3);
    auto incremental = resource(), fresh = resource();
    initCacheKeys(incremental, ids, 2, 37);
    ids->setSeqLength(7);
    updateCacheKeys(incremental, ids, 2, 37);
    initCacheKeys(fresh, ids, 2, 37);
    ASSERT_EQ(incremental->cacheKeys().size(), 3u);
    for (size_t i = 0; i < 3; ++i) EXPECT_EQ(incremental->cacheKeys()[i], fresh->cacheKeys()[i]);
}
}
}
