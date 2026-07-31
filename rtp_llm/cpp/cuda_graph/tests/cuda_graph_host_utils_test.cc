#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_utils.h"

namespace rtp_llm {
namespace {

torch::Tensor makeRows(int64_t rows, int64_t cols, int32_t fill) {
    return torch::full({rows, cols}, fill, torch::dtype(torch::kInt32));
}

TEST(CudaGraphHostUtilsTest, ZeroTailClearsPaddedRowsOnly) {
    auto dst = makeRows(8, 4, 7);
    auto src = makeRows(3, 4, 5);

    copyStridedHost(src, dst);
    zeroStridedHostTail(src, dst, 6);

    EXPECT_TRUE(dst.slice(0, 0, 3).eq(5).all().item<bool>());
    EXPECT_TRUE(dst.slice(0, 3, 6).eq(0).all().item<bool>());
    EXPECT_TRUE(dst.slice(0, 6, 8).eq(7).all().item<bool>());
}

TEST(CudaGraphHostUtilsTest, ZeroTailDefaultLimitClearsToBufferEnd) {
    auto dst = makeRows(5, 3, 9);
    auto src = makeRows(2, 3, 1);

    copyStridedHost(src, dst);
    zeroStridedHostTail(src, dst);

    EXPECT_TRUE(dst.slice(0, 0, 2).eq(1).all().item<bool>());
    EXPECT_TRUE(dst.slice(0, 2, 5).eq(0).all().item<bool>());
}

TEST(CudaGraphHostUtilsTest, ZeroTailNoopWhenLimitEqualsSourceRows) {
    auto dst = makeRows(5, 3, 9);
    auto src = makeRows(2, 3, 1);

    zeroStridedHostTail(src, dst, src.size(0));

    EXPECT_TRUE(dst.eq(9).all().item<bool>());
}

TEST(CudaGraphHostUtilsTest, ZeroTailClampsLimitAndPreservesPrefix) {
    auto dst = makeRows(4, 2, 7);
    auto src = makeRows(1, 2, 5);

    zeroStridedHostTail(src, dst, 100);

    EXPECT_TRUE(dst.slice(0, 0, 1).eq(7).all().item<bool>());
    EXPECT_TRUE(dst.slice(0, 1, 4).eq(0).all().item<bool>());
}

TEST(CudaGraphHostUtilsTest, UndefinedSrcClearsWholeWindow) {
    auto dst = makeRows(4, 3, 7);
    zeroStridedHostTail(torch::Tensor(), dst, 2);

    EXPECT_TRUE(dst.slice(0, 0, 2).eq(0).all().item<bool>());
    EXPECT_TRUE(dst.slice(0, 2, 4).eq(7).all().item<bool>());
}

TEST(CudaGraphHostUtilsTest, OneDimensionalRange) {
    auto dst = torch::full({6}, 7, torch::dtype(torch::kInt32));
    auto src = torch::full({2}, 5, torch::dtype(torch::kInt32));

    copyStridedHost(src, dst);
    zeroStridedHostTail(src, dst, 4);

    EXPECT_TRUE(dst.slice(0, 0, 2).eq(5).all().item<bool>());
    EXPECT_TRUE(dst.slice(0, 2, 4).eq(0).all().item<bool>());
    EXPECT_TRUE(dst.slice(0, 4, 6).eq(7).all().item<bool>());
}

TEST(CudaGraphHostUtilsTest, MismatchedRankFailsLoud) {
    auto dst    = makeRows(4, 3, 7);
    auto src_1d = torch::full({3}, 5, torch::dtype(torch::kInt32));

    EXPECT_ANY_THROW(zeroStridedHostTail(src_1d, dst));
    EXPECT_ANY_THROW(copyStridedHost(src_1d, dst));
}

TEST(CudaGraphHostUtilsTest, RespectsRowStride) {
    auto backing = makeRows(4, 6, 7);
    auto dst     = backing.slice(1, 0, 4);
    auto src     = makeRows(1, 4, 5);

    copyStridedHost(src, dst);
    zeroStridedHostTail(src, dst, 3);

    EXPECT_TRUE(dst.slice(0, 0, 1).eq(5).all().item<bool>());
    EXPECT_TRUE(dst.slice(0, 1, 3).eq(0).all().item<bool>());
    EXPECT_TRUE(dst.slice(0, 3, 4).eq(7).all().item<bool>());
    EXPECT_TRUE(backing.slice(1, 4, 6).eq(7).all().item<bool>());
}

}  // namespace
}  // namespace rtp_llm
