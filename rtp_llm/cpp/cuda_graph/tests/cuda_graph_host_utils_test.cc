#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_utils.h"

namespace rtp_llm {
namespace {

torch::Tensor makeRows(int64_t rows, int64_t cols, int32_t fill) {
    return torch::full({rows, cols}, fill, torch::dtype(torch::kInt32));
}

// Batch shrink: src has fewer rows than the graph read window; the padded
// tail [src_rows, row_limit) must be zeroed, the copied rows left intact.
TEST(CudaGraphHostUtilsTest, ZeroTailClearsPaddedRowsOnly) {
    auto dst = makeRows(/*rows=*/8, /*cols=*/4, /*fill=*/7);  // stale entries everywhere
    auto src = makeRows(/*rows=*/3, /*cols=*/4, /*fill=*/5);

    copyStridedHost(src, dst);
    zeroStridedHostTail(src, dst, /*row_limit=*/6);

    for (int64_t r = 0; r < 3; ++r) {
        for (int64_t c = 0; c < 4; ++c) {
            EXPECT_EQ(dst[r][c].item<int32_t>(), 5) << "copied row " << r << " must be intact";
        }
    }
    for (int64_t r = 3; r < 6; ++r) {
        for (int64_t c = 0; c < 4; ++c) {
            EXPECT_EQ(dst[r][c].item<int32_t>(), 0) << "padded row " << r << " must be zeroed";
        }
    }
    // Rows past the graph read window are never read by this replay and must
    // not be touched (that is the point of the row_limit bound).
    for (int64_t r = 6; r < 8; ++r) {
        for (int64_t c = 0; c < 4; ++c) {
            EXPECT_EQ(dst[r][c].item<int32_t>(), 7) << "row " << r << " past row_limit must be untouched";
        }
    }
}

// Default row_limit (-1) zeroes through the full buffer height.
TEST(CudaGraphHostUtilsTest, ZeroTailDefaultLimitClearsToBufferEnd) {
    auto dst = makeRows(5, 3, 9);
    auto src = makeRows(2, 3, 1);

    copyStridedHost(src, dst);
    zeroStridedHostTail(src, dst);

    EXPECT_EQ(dst.slice(0, 0, 2).eq(1).all().item<bool>(), true);
    EXPECT_EQ(dst.slice(0, 2, 5).eq(0).all().item<bool>(), true);
}

// row_limit <= src rows: nothing to zero, nothing may be clobbered.
TEST(CudaGraphHostUtilsTest, ZeroTailNoopWhenWindowFull) {
    auto dst = makeRows(8, 4, 7);
    auto src = makeRows(3, 4, 5);

    copyStridedHost(src, dst);
    zeroStridedHostTail(src, dst, /*row_limit=*/3);

    EXPECT_EQ(dst.slice(0, 0, 3).eq(5).all().item<bool>(), true);
    EXPECT_EQ(dst.slice(0, 3, 8).eq(7).all().item<bool>(), true);
}

// row_limit larger than the buffer must clamp instead of overrunning.
TEST(CudaGraphHostUtilsTest, ZeroTailClampsLimitToBufferRows) {
    auto dst = makeRows(4, 2, 7);
    auto src = makeRows(1, 2, 5);

    zeroStridedHostTail(src, dst, /*row_limit=*/100);

    EXPECT_EQ(dst.slice(0, 0, 1).eq(7).all().item<bool>(), true);  // untouched (zero starts at src rows)
    EXPECT_EQ(dst.slice(0, 1, 4).eq(0).all().item<bool>(), true);
}

// Undefined src means no rows were refreshed: the whole window is stale
// padding and must be zeroed.
TEST(CudaGraphHostUtilsTest, ZeroTailUndefinedSrcClearsWholeWindow) {
    auto dst = makeRows(4, 3, 7);
    zeroStridedHostTail(torch::Tensor(), dst, /*row_limit=*/2);

    EXPECT_EQ(dst.slice(0, 0, 2).eq(0).all().item<bool>(), true);
    EXPECT_EQ(dst.slice(0, 2, 4).eq(7).all().item<bool>(), true);
}

// 1-D pair uses element ranges instead of rows.
TEST(CudaGraphHostUtilsTest, ZeroTailOneDimensional) {
    auto dst = torch::full({6}, 7, torch::dtype(torch::kInt32));
    auto src = torch::full({2}, 5, torch::dtype(torch::kInt32));

    copyStridedHost(src, dst);
    zeroStridedHostTail(src, dst, /*row_limit=*/4);

    EXPECT_EQ(dst.slice(0, 0, 2).eq(5).all().item<bool>(), true);
    EXPECT_EQ(dst.slice(0, 2, 4).eq(0).all().item<bool>(), true);
    EXPECT_EQ(dst.slice(0, 4, 6).eq(7).all().item<bool>(), true);
}

// A 1-D src refreshing a 2-D dst is a contract violation: without the check
// zeroStridedHostTail would treat from_row as 0 and wipe the copied data.
TEST(CudaGraphHostUtilsTest, MismatchedRankFailsLoud) {
    auto dst    = makeRows(4, 3, 7);
    auto src_1d = torch::full({3}, 5, torch::dtype(torch::kInt32));

    EXPECT_ANY_THROW(zeroStridedHostTail(src_1d, dst));
    EXPECT_ANY_THROW(copyStridedHost(src_1d, dst));
}

// Non-contiguous dst (wider stride than row width) must only zero row_bytes
// per row, not the stride gap.
TEST(CudaGraphHostUtilsTest, ZeroTailRespectsRowStride) {
    auto backing = makeRows(4, 6, 7);
    auto dst     = backing.slice(1, 0, 4);  // 4 cols, stride 6
    auto src     = makeRows(1, 4, 5);

    copyStridedHost(src, dst);
    zeroStridedHostTail(src, dst, /*row_limit=*/3);

    EXPECT_EQ(dst.slice(0, 0, 1).eq(5).all().item<bool>(), true);
    EXPECT_EQ(dst.slice(0, 1, 3).eq(0).all().item<bool>(), true);
    EXPECT_EQ(dst.slice(0, 3, 4).eq(7).all().item<bool>(), true);
    // Stride gap columns must be untouched.
    EXPECT_EQ(backing.slice(1, 4, 6).eq(7).all().item<bool>(), true);
}

}  // namespace
}  // namespace rtp_llm
