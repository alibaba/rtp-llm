#include "gtest/gtest.h"
#include <utility>
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {
namespace {
CompleteTokenIdsPtr tokens(std::initializer_list<int32_t> values,
                           bool multimodal = false) {
    auto input = std::make_shared<GenerateInput>();
    input->input_ids = torch::tensor(std::vector<int32_t>(values), torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    if (multimodal) {
        auto identity = std::make_shared<V41RequestInputs>();
        V41ImageInput image;
        image.start = 1;
        image.types = torch::zeros({4}, torch::kInt32);
        image.n_vit_h = 2;
        image.n_vit_w = 2;
        image.content_sha256 = std::string(64, 'a');
        image.processor_identity = std::string(64, 'b');
        identity->images.push_back(std::move(image));
        input->v41_inputs = std::move(identity);
    }
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
TEST(CachePrefixHash, EosIncrementalKeysMatchOriginalFullRebuild) {
    for (const bool multimodal : {false, true}) {
        SCOPED_TRACE(multimodal);
        auto ids = tokens({1, 2, 3, 4, 5, 6, 7, 8}, multimodal);
        EXPECT_EQ(ids->imageCacheIdentity(0, 2).empty(), !multimodal);
        auto incremental = resource(), fresh = resource();
        ids->setSeqLength(3);
        initCacheKeys(incremental, ids, 2, 37);
        // Partial->aligned, aligned->partial, another EOS, shrink, grow again.
        for (const int length : {4, 5, 4, 5, 3, 7, 7, 3, 0, 8}) {
            SCOPED_TRACE(length);
            ids->setSeqLength(length);
            updateCacheKeys(incremental, ids, 2, 37, true);
            initCacheKeys(fresh, ids, 2, 37);
            EXPECT_EQ(incremental->cacheKeys(), fresh->cacheKeys());
            EXPECT_EQ(incremental->lastBlockAligned(), fresh->lastBlockAligned());
            // Allocator/default P update drops only the partial key. EOS can
            // restore it without changing the previously completed prefix.
            updateCacheKeys(incremental, ids, 2, 37);
            ASSERT_EQ(incremental->cacheKeys().size(), static_cast<size_t>(length / 2));
            EXPECT_TRUE(incremental->lastBlockAligned());
            updateCacheKeys(incremental, ids, 2, 37, true);
            EXPECT_EQ(incremental->cacheKeys(), fresh->cacheKeys());
        }
    }
}

}
}
