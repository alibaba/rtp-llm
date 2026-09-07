#include <gtest/gtest.h>

#include "rtp_llm/cpp/cuda_graph/strided_copy_validation.h"

namespace rtp_llm {
namespace {

TEST(StridedCopyValidationTest, AcceptsNoOpAndNarrowCopies) {
    std::string reason = "stale";
    EXPECT_TRUE(isStridedCopyCompatible(torch::Tensor(), torch::Tensor(), &reason));
    EXPECT_TRUE(reason.empty());
    EXPECT_TRUE(isStridedCopyCompatible(torch::empty({0, 2}), torch::Tensor(), &reason));
    EXPECT_TRUE(isStridedCopyCompatible(torch::zeros({2}), torch::zeros({3}), &reason));
    EXPECT_TRUE(isStridedCopyCompatible(torch::zeros({2, 2}), torch::zeros({3, 4}), &reason));
}

TEST(StridedCopyValidationTest, ReportsUndefinedDestinationDtypeAndDimensionErrors) {
    std::string reason;
    EXPECT_FALSE(isStridedCopyCompatible(torch::zeros({2}), torch::Tensor(), &reason));
    EXPECT_EQ(reason, "destination is undefined");

    EXPECT_FALSE(isStridedCopyCompatible(torch::zeros({2}, torch::kInt32), torch::zeros({2}, torch::kInt64), &reason));
    EXPECT_EQ(reason, "dtype mismatch");

    EXPECT_FALSE(isStridedCopyCompatible(torch::zeros({2}), torch::zeros({1, 2}), &reason));
    EXPECT_EQ(reason, "only matching 1D or 2D tensors are supported");
    EXPECT_FALSE(isStridedCopyCompatible(torch::zeros({1, 1, 1}), torch::zeros({1, 1, 1}), &reason));
    EXPECT_EQ(reason, "only matching 1D or 2D tensors are supported");
}

TEST(StridedCopyValidationTest, ReportsShapeAndStrideErrors) {
    std::string reason;
    EXPECT_FALSE(isStridedCopyCompatible(torch::zeros({4}), torch::zeros({3}), &reason));
    EXPECT_EQ(reason, "source length exceeds destination");
    EXPECT_FALSE(isStridedCopyCompatible(torch::zeros({3, 2}), torch::zeros({2, 2}), &reason));
    EXPECT_EQ(reason, "source rows or columns exceed destination");
    EXPECT_FALSE(isStridedCopyCompatible(torch::zeros({2, 3}), torch::zeros({2, 2}), &reason));
    EXPECT_EQ(reason, "source rows or columns exceed destination");

    const auto narrow_stride = torch::empty_strided({2, 2}, {1, 1}, torch::TensorOptions().dtype(torch::kInt32));
    EXPECT_FALSE(isStridedCopyCompatible(narrow_stride, torch::zeros({2, 2}, torch::kInt32), &reason));
    EXPECT_EQ(reason, "row bytes exceed source or destination stride");
    EXPECT_FALSE(isStridedCopyCompatible(torch::zeros({2, 2}, torch::kInt32), narrow_stride, &reason));
    EXPECT_EQ(reason, "row bytes exceed source or destination stride");
}

TEST(StridedCopyValidationTest, RejectsNonContiguousInnermostDimensions) {
    std::string reason;

    const auto stepped_vector = torch::arange(8).slice(0, 0, 8, 2);
    EXPECT_FALSE(isStridedCopyCompatible(stepped_vector, torch::zeros({4}, stepped_vector.options()), &reason));
    EXPECT_EQ(reason, "1D tensors must have unit stride");

    const auto stepped_vector_dst = torch::zeros({8}).slice(0, 0, 8, 2);
    EXPECT_FALSE(isStridedCopyCompatible(torch::zeros({4}), stepped_vector_dst, &reason));
    EXPECT_EQ(reason, "1D tensors must have unit stride");

    const auto stepped_table = torch::zeros({2, 6}).slice(1, 0, 6, 2);
    EXPECT_FALSE(isStridedCopyCompatible(stepped_table, torch::zeros({2, 3}), &reason));
    EXPECT_EQ(reason, "2D tensor rows must be contiguous");

    const auto stepped_table_dst = torch::zeros({2, 6}).slice(1, 0, 6, 2);
    EXPECT_FALSE(isStridedCopyCompatible(torch::zeros({2, 3}), stepped_table_dst, &reason));
    EXPECT_EQ(reason, "2D tensor rows must be contiguous");
}

TEST(StridedCopyValidationTest, AcceptsPaddedOuterStrideWithContiguousRows) {
    const auto options = torch::TensorOptions().dtype(torch::kInt32);
    const auto src     = torch::empty_strided({2, 3}, {5, 1}, options);
    const auto dst     = torch::empty_strided({3, 4}, {7, 1}, options);

    std::string reason;
    EXPECT_TRUE(isStridedCopyCompatible(src, dst, &reason));
    EXPECT_TRUE(reason.empty());
}

}  // namespace
}  // namespace rtp_llm
