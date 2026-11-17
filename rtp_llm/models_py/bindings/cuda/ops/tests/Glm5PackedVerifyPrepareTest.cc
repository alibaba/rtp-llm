#include "rtp_llm/models_py/bindings/cuda/kernels/glm5_packed_verify_prepare.h"
#include "rtp_llm/models_py/bindings/cuda/ops/tests/CudaTestUtils.h"

#include <cstdlib>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>

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

TEST_F(Glm5PackedVerifyPrepareTest, CapsProduceOrderedRowsAndAllConsumers) {
    const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    for (int batch : {1, 3, 129, 257, 1024}) {
        auto caps_host = torch::empty({batch}, torch::kInt32);
        std::vector<int64_t> expected_rows;
        for (int request = 0; request < batch; ++request) {
            const int cap = request % 4;
            caps_host.data_ptr<int32_t>()[request] = cap;
            for (int position = 0; position <= cap; ++position) expected_rows.push_back(request * 4 + position);
        }
        auto caps = caps_host.to(torch::kCUDA);
        auto dense = torch::arange(batch * 4, options) + 11;
        auto positions = torch::arange(batch * 4, options.dtype(torch::kInt64)) * 7;
        auto types = dense.remainder(2);
        auto rows = torch::tensor(expected_rows, options.dtype(torch::kInt64));
        Glm5PackedVerifyInputs actual;
        ASSERT_TRUE(tryGlm5PackedVerifyPrepareFromCaps(dense, caps, rows.numel(), positions, types, actual,
                                                       at::cuda::getCurrentCUDAStream().stream()));
        EXPECT_TRUE(torch::equal(actual.rows, rows));
        EXPECT_TRUE(torch::equal(actual.tokens, dense.index_select(0, rows)));
        EXPECT_TRUE(torch::equal(actual.lengths, caps + 1));
        EXPECT_TRUE(torch::equal(actual.lm_output_indexes, torch::arange(rows.numel(), options)));
        EXPECT_TRUE(torch::equal(actual.position_ids, positions.index_select(0, rows)));
        EXPECT_TRUE(torch::equal(actual.token_type_ids, types.index_select(0, rows)));
    }
}

TEST_F(Glm5PackedVerifyPrepareTest, CapsGraphReplayConsumesChangedMetadata) {
    auto stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    auto dense = torch::arange(12, options);
    auto caps = torch::tensor({0, 2, 1}, options);
    auto positions = torch::arange(12, options.dtype(torch::kInt64));
    auto types = torch::zeros({12}, options);
    Glm5PackedVerifyInputs captured;
    ASSERT_TRUE(tryGlm5PackedVerifyPrepareFromCaps(dense, caps, 6, positions, types, captured, stream.stream()));
    ASSERT_EQ(cudaStreamSynchronize(stream.stream()), cudaSuccess);
    at::cuda::CUDAGraph graph;
    graph.capture_begin();
    ASSERT_TRUE(tryGlm5PackedVerifyPrepareFromCaps(dense, caps, 6, positions, types, captured, stream.stream()));
    graph.capture_end();
    for (int replay = 0; replay < 3; ++replay) {
        caps.copy_(torch::tensor({2, 1, 0}, options).roll(replay));
        dense.add_(100);
        positions.add_(17);
        types.fill_(replay + 1);
        graph.replay();
        auto dense_rows = torch::arange(12, options.dtype(torch::kInt64));
        auto rows = dense_rows.masked_select(dense_rows.remainder(4) <= caps.repeat_interleave(4));
        EXPECT_TRUE(torch::equal(captured.rows, rows));
        EXPECT_TRUE(torch::equal(captured.tokens, dense.index_select(0, rows)));
        EXPECT_TRUE(torch::equal(captured.lengths, caps + 1));
        EXPECT_TRUE(torch::equal(captured.position_ids, positions.index_select(0, rows)));
        EXPECT_TRUE(torch::equal(captured.token_type_ids, types.index_select(0, rows)));
    }
}

TEST_F(Glm5PackedVerifyPrepareTest, CapsUnsupportedMetadataPreservesFallbackOutput) {
    const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    auto dense = torch::arange(12, options);
    auto caps = torch::tensor({0, 2, 1}, options);
    Glm5PackedVerifyInputs output;
    output.tokens = dense;
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    EXPECT_FALSE(tryGlm5PackedVerifyPrepareFromCaps(dense, caps, 2, {}, {}, output, stream));
    EXPECT_FALSE(tryGlm5PackedVerifyPrepareFromCaps(dense, caps, 13, {}, {}, output, stream));
    EXPECT_FALSE(tryGlm5PackedVerifyPrepareFromCaps(dense.to(torch::kFloat32), caps, 6, {}, {}, output, stream));
    EXPECT_FALSE(tryGlm5PackedVerifyPrepareFromCaps(torch::zeros({4100}, options), torch::zeros({1025}, options),
                                                   1025, {}, {}, output, stream));
    EXPECT_EQ(output.tokens.data_ptr(), dense.data_ptr());
    EXPECT_FALSE(output.rows.defined());
}

TEST_F(Glm5PackedVerifyPrepareTest, PlannerTotalMismatchAssertsInIsolatedProcess) {
    const auto previous_style = ::testing::FLAGS_gtest_death_test_style;
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    // CUDA assertions poison their context. Re-exec the child and exit without
    // tensor teardown so the ordinary test process retains a usable context.
    ASSERT_EXIT({
        const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
        auto dense = torch::arange(12, options);
        auto caps = torch::tensor({0, 2, 1}, options);  // Six rows, not five.
        Glm5PackedVerifyInputs output;
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        if (!tryGlm5PackedVerifyPrepareFromCaps(dense, caps, 5, {}, {}, output, stream)) std::_Exit(2);
        std::_Exit(cudaStreamSynchronize(stream) == cudaErrorAssert ? 0 : 1);
    }, ::testing::ExitedWithCode(0), "");
    ::testing::FLAGS_gtest_death_test_style = previous_style;
}

}  // namespace rtp_llm
