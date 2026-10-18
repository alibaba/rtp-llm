#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {
namespace {

constexpr int32_t kImageToken = 129264;

std::shared_ptr<GenerateInput> makeInput(std::vector<int32_t> first_hashes = {101, 102}) {
    auto input              = std::make_shared<GenerateInput>();
    input->input_ids        = torch::tensor({11, kImageToken, kImageToken, 14, kImageToken, 16}, torch::kInt32);
    input->text_tokens_mask = torch::tensor({1, 0, 0, 1, 0, 1}, torch::kInt32);
    input->mm_locs          = torch::tensor({1, 4}, torch::kInt32);
    auto spans              = std::make_shared<MultimodalCacheSpans>();
    spans->push_back(MultimodalCacheSpan{1, std::move(first_hashes)});
    spans->push_back(MultimodalCacheSpan{4, {103}});
    input->multimodal_cache_spans = spans;
    input->generate_config        = std::make_shared<GenerateConfig>();
    return input;
}

std::vector<int32_t> tensorValues(const torch::Tensor& tensor) {
    auto value = tensor.to(torch::kCPU).contiguous();
    return {value.data_ptr<int32_t>(), value.data_ptr<int32_t>() + value.numel()};
}

TEST(MultimodalCacheIdentityTest, CanonicalHistoryStaysCanonicalAndTextHasNoSidecar) {
    auto             input = makeInput();
    CompleteTokenIds tokens(1, 1, 32, 4);
    tokens.init(input);

    EXPECT_EQ(tokens.completeTokenIdsVec(0), (std::vector<int>{11, kImageToken, kImageToken, 14, kImageToken, 16}));
    EXPECT_TRUE(tokens.multimodalCacheIdentity(0, 1).empty());
    EXPECT_TRUE(tokens.multimodalCacheIdentity(3, 1).empty());
    EXPECT_FALSE(tokens.multimodalCacheIdentity(1, 2).empty());
    EXPECT_FALSE(tokens.multimodalCacheIdentity(4, 1).empty());
}

TEST(MultimodalCacheIdentityTest, SameCanonicalIdsWithDifferentRowsHaveDifferentCacheIdentity) {
    auto             first_input  = makeInput({101, 102});
    auto             second_input = makeInput({201, 102});
    CompleteTokenIds first(1, 1, 32, 4);
    CompleteTokenIds second(1, 1, 32, 4);
    first.init(first_input);
    second.init(second_input);

    EXPECT_EQ(first.completeTokenIdsVec(0), second.completeTokenIdsVec(0));
    EXPECT_NE(first.multimodalCacheIdentity(1, 2), second.multimodalCacheIdentity(1, 2));
    EXPECT_NE(first.multimodalCacheIdentity(1, 1), second.multimodalCacheIdentity(1, 1));
    // Cache blocks may end inside an image: an unchanged row slice keeps the
    // same identity even when another row in the same image changed.
    EXPECT_EQ(first.multimodalCacheIdentity(2, 1), second.multimodalCacheIdentity(2, 1));
    EXPECT_EQ(first.multimodalCacheIdentity(4, 1), second.multimodalCacheIdentity(4, 1));

    // CompleteTokenIds retains the sidecar it was initialized with; replacing
    // the request's pointer later must not mutate an in-flight cache key.
    const auto first_identity           = first.multimodalCacheIdentity(1, 2);
    first_input->multimodal_cache_spans = second_input->multimodal_cache_spans;
    EXPECT_EQ(first.multimodalCacheIdentity(1, 2), first_identity);
}

TEST(MultimodalCacheIdentityTest, CopyModesPreserveIdentityWithoutCoordinateShift) {
    auto             input = makeInput();
    CompleteTokenIds original(1, 1, 32, 4);
    original.init(input);

    CompleteTokenIds deep_copy(original);
    CompleteTokenIds shared_copy(original, true);

    EXPECT_EQ(deep_copy.seqLength(), original.seqLength());
    EXPECT_EQ(shared_copy.seqLength(), original.seqLength());
    EXPECT_EQ(deep_copy.multimodalCacheIdentity(1, 2), original.multimodalCacheIdentity(1, 2));
    EXPECT_EQ(shared_copy.multimodalCacheIdentity(1, 2), original.multimodalCacheIdentity(1, 2));
    EXPECT_NE(deep_copy.completeTokenIds().data_ptr<int32_t>(), original.completeTokenIds().data_ptr<int32_t>());
    EXPECT_EQ(shared_copy.completeTokenIds().data_ptr<int32_t>(), original.completeTokenIds().data_ptr<int32_t>());
}

TEST(MultimodalCacheIdentityTest, PrefixDeepCopiesAndShiftsSidecarExactlyOnce) {
    auto             input          = makeInput();
    auto             original_spans = input->multimodal_cache_spans;
    CompleteTokenIds before(1, 1, 32, 4);
    before.init(input);
    const auto identity = before.multimodalCacheIdentity(1, 2);

    input->updatePrefix({7, 8});
    ASSERT_TRUE(input->multimodal_cache_spans);
    EXPECT_NE(input->multimodal_cache_spans.get(), original_spans.get());
    ASSERT_EQ(original_spans->size(), 2u);
    EXPECT_EQ(original_spans->at(0).offset, 1);
    EXPECT_EQ(original_spans->at(1).offset, 4);
    EXPECT_EQ(input->multimodal_cache_spans->at(0).offset, 3);
    EXPECT_EQ(input->multimodal_cache_spans->at(1).offset, 6);
    EXPECT_EQ(input->multimodal_cache_spans->at(0).row_hashes, original_spans->at(0).row_hashes);
    EXPECT_EQ(tensorValues(input->input_ids),
              (std::vector<int32_t>{7, 8, 11, kImageToken, kImageToken, 14, kImageToken, 16}));

    CompleteTokenIds after(1, 1, 32, 4);
    after.init(input);
    EXPECT_EQ(after.multimodalCacheIdentity(3, 2), identity);
    EXPECT_TRUE(after.multimodalCacheIdentity(0, 3).empty());
}

TEST(MultimodalCacheIdentityTest, ReuseMayEndInsideImageButStillHonorsFreshTokenFloor) {
    auto             input = makeInput();
    CompleteTokenIds tokens(1, 1, 32, 4);
    tokens.init(input);

    EXPECT_TRUE(tokens.isValidReuseLength(2));
    EXPECT_TRUE(tokens.isValidReuseLength(5));
    EXPECT_FALSE(tokens.isValidReuseLength(5, 2));
    EXPECT_TRUE(tokens.isValidReuseLength(4, 2));
}

}  // namespace
}  // namespace rtp_llm
