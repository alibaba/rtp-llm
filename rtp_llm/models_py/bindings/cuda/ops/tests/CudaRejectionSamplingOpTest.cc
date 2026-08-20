#include "rtp_llm/cpp/testing/RejectionSamplingOpTest.hpp"

class CudaRejectionSamplingOpTest: public RejectionSamplingOpTest {};

TEST_F(CudaRejectionSamplingOpTest, referenceCases) {
    runReferenceCases();
}

TEST_F(CudaRejectionSamplingOpTest, zeroAndOneSpeculativeTokenCases) {
    runZeroAndOneSpeculativeTokenCases();
}

TEST_F(CudaRejectionSamplingOpTest, rejectsInvalidTensorMetadata) {
    runRejectsInvalidTensorMetadata();
}

TEST_F(CudaRejectionSamplingOpTest, acceptsImplicitPointMassWithoutDenseDraftProbabilities) {
    const auto float_cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto int_cuda   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto bool_cuda  = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);

    RejectionSamplingParams params{
        torch::Tensor(),
        torch::tensor({5}, int_cuda).reshape({1, 1}),
        torch::zeros({1, 2}, float_cuda),
        torch::full({1, 2, 16}, 1.0f / 16.0f, float_cuda),
        torch::tensor({7, 9}, int_cuda).reshape({2, 1}),
        torch::full({1, 2}, -1, int_cuda),
        torch::zeros({1}, int_cuda),
        torch::zeros({1}, bool_cuda),
        true,
    };

    EXPECT_NO_THROW(rejectionSampling(params));
    EXPECT_EQ(params.output_accepted_token_num_d.cpu()[0].item<int32_t>(), 1);
    EXPECT_EQ(params.output_token_ids_d.cpu()[0][0].item<int32_t>(), 7);
}
