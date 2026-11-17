#include "rtp_llm/models_py/bindings/cuda/kernels/glm5_packed_verify_prepare.h"
#include "rtp_llm/models_py/bindings/cuda/ops/tests/CudaTestUtils.h"

#include <ATen/cuda/CUDAContext.h>

namespace rtp_llm {

class Glm5PackedVerifyPrepareTest: public DeviceTestBase {};

TEST_F(Glm5PackedVerifyPrepareTest, MatchesSeparateMaterialization) {
    const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    for (auto token_dtype : {torch::kInt32, torch::kInt64}) {
        for (auto row_dtype : {torch::kInt32, torch::kInt64}) {
            for (bool optional : {false, true}) {
                auto dense = torch::arange(12, options.dtype(token_dtype)) + 100;
                auto caps = torch::tensor({0, 2, 1}, options);
                auto rows = torch::tensor({0, 4, 5, 6, 8, 9}, options.dtype(row_dtype));
                auto positions = optional ? torch::arange(12, options.dtype(torch::kInt64)) * 3 : torch::Tensor();
                auto types = optional ? torch::arange(12, options).remainder(2) : torch::Tensor();
                Glm5PackedVerifyInputs actual;
                ASSERT_TRUE(tryGlm5PackedVerifyPrepare(dense, caps, rows, positions, types, actual, stream));
                auto reference_rows = rows.to(torch::kInt64);
                EXPECT_TRUE(torch::equal(actual.tokens, dense.index_select(0, reference_rows)));
                EXPECT_TRUE(torch::equal(actual.lengths, caps + 1));
                EXPECT_TRUE(torch::equal(actual.lm_output_indexes, torch::arange(rows.numel(), caps.options())));
                EXPECT_EQ(actual.tokens.scalar_type(), dense.scalar_type());
                EXPECT_EQ(actual.lm_output_indexes.scalar_type(), caps.scalar_type());
                EXPECT_EQ(actual.position_ids.defined(), optional);
                EXPECT_EQ(actual.token_type_ids.defined(), optional);
                if (optional) {
                    EXPECT_TRUE(torch::equal(actual.position_ids, positions.index_select(0, reference_rows)));
                    EXPECT_TRUE(torch::equal(actual.token_type_ids, types.index_select(0, reference_rows)));
                }
            }
        }
    }
}

TEST_F(Glm5PackedVerifyPrepareTest, PreservesSuppliedRowOrderAndWideLengths) {
    const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64);
    auto dense = torch::arange(8, options) + 200;
    auto rows = torch::tensor({6, 1, 4, 0}, options);
    auto caps = torch::tensor({1, 1}, options);
    Glm5PackedVerifyInputs actual;
    ASSERT_TRUE(tryGlm5PackedVerifyPrepare(dense, caps, rows, {}, {}, actual,
                                         at::cuda::getCurrentCUDAStream().stream()));
    EXPECT_TRUE(torch::equal(actual.tokens, dense.index_select(0, rows)));
    EXPECT_TRUE(torch::equal(actual.lengths, caps + 1));
    EXPECT_EQ(actual.lengths.scalar_type(), torch::kInt64);
    EXPECT_EQ(actual.lm_output_indexes.scalar_type(), torch::kInt64);
}

TEST_F(Glm5PackedVerifyPrepareTest, EmptyRowsAndEmptyBatch) {
    const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto rows = torch::empty({0}, options.dtype(torch::kInt64));
    Glm5PackedVerifyInputs actual;
    ASSERT_TRUE(tryGlm5PackedVerifyPrepare(torch::arange(4, options), torch::tensor({0}, options), rows,
                                         {}, {}, actual, stream));
    EXPECT_EQ(actual.tokens.numel(), 0);
    EXPECT_EQ(actual.lm_output_indexes.numel(), 0);
    EXPECT_TRUE(torch::equal(actual.lengths, torch::ones({1}, options)));
    ASSERT_TRUE(tryGlm5PackedVerifyPrepare(torch::empty({0}, options), torch::empty({0}, options), rows,
                                         {}, {}, actual, stream));
    EXPECT_EQ(actual.lengths.numel(), 0);
}

TEST_F(Glm5PackedVerifyPrepareTest, UnsupportedMetadataLeavesOutputUntouched) {
    const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto dense = torch::arange(4, options);
    auto caps = torch::tensor({1}, options);
    auto rows = torch::tensor({0, 1}, options.dtype(torch::kInt64));
    Glm5PackedVerifyInputs actual;
    actual.tokens = dense;
    EXPECT_FALSE(tryGlm5PackedVerifyPrepare(dense.to(torch::kFloat32), caps, rows, {}, {}, actual, stream));
    EXPECT_FALSE(tryGlm5PackedVerifyPrepare(dense.cpu(), caps, rows, {}, {}, actual, stream));
    EXPECT_FALSE(tryGlm5PackedVerifyPrepare(dense.view({2, 2}), caps, rows, {}, {}, actual, stream));
    EXPECT_FALSE(tryGlm5PackedVerifyPrepare(dense.slice(0, 0, 4, 2), caps, rows, dense, {}, actual, stream));
    EXPECT_FALSE(tryGlm5PackedVerifyPrepare(dense, caps, rows, torch::empty({3}, options), {}, actual, stream));
    EXPECT_FALSE(tryGlm5PackedVerifyPrepare(dense, caps, torch::arange(5, rows.options()), {}, {}, actual, stream));
    EXPECT_EQ(actual.tokens.data_ptr(), dense.data_ptr());
    EXPECT_FALSE(actual.lengths.defined());
}

}  // namespace rtp_llm
