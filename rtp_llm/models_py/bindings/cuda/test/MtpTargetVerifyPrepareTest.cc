#include <gtest/gtest.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>
#include <vector>

#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"

namespace rtp_llm {
namespace {

torch::Tensor toCudaI32(std::initializer_list<int32_t> values) {
    return torch::tensor(std::vector<int32_t>(values), torch::TensorOptions().dtype(torch::kInt32)).cuda();
}

std::vector<torch::Tensor> referenceAddressing(const torch::Tensor& request_block_table,
                                               const torch::Tensor& prefix_lengths,
                                               const torch::Tensor& input_lengths,
                                               int64_t              tokens_per_batch) {
    const auto batch_size           = request_block_table.size(0);
    auto       physical_block_table = request_block_table.repeat_interleave(tokens_per_batch, 0);
    auto       token_offsets        = torch::arange(tokens_per_batch, prefix_lengths.options()).repeat({batch_size});
    auto       positions            = prefix_lengths.repeat_interleave(tokens_per_batch) + token_offsets;
    auto       valid_token_mask     = input_lengths.repeat_interleave(tokens_per_batch).gt(0);
    auto       sequence_lengths     = torch::where(valid_token_mask, positions + 1, torch::zeros_like(positions));
    return {physical_block_table, positions, sequence_lengths, valid_token_mask};
}

void expectTensorVectorsEqual(const std::vector<torch::Tensor>& actual,
                              const std::vector<torch::Tensor>& expected,
                              int64_t                           expected_device) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_TRUE(actual[i].is_cuda());
        EXPECT_EQ(actual[i].get_device(), expected_device);
        EXPECT_EQ(actual[i].sizes(), expected[i].sizes());
        EXPECT_EQ(actual[i].scalar_type(), expected[i].scalar_type());
        EXPECT_TRUE(torch::equal(actual[i].cpu(), expected[i].cpu()));
    }
}

void expectMatchesReference(const torch::Tensor& request_block_table,
                            const torch::Tensor& prefix_lengths,
                            const torch::Tensor& input_lengths,
                            int64_t              tokens_per_batch) {
    auto actual =
        mtpMsaTargetVerifyAddressingPrepare(request_block_table, prefix_lengths, input_lengths, tokens_per_batch);
    auto expected = referenceAddressing(request_block_table, prefix_lengths, input_lengths, tokens_per_batch);
    expectTensorVectorsEqual(actual, expected, request_block_table.get_device());
}

TEST(MtpTargetVerifyPrepareTest, ExpandsAddressingAndMasksPaddedRequests) {
    auto request_block_table =
        torch::tensor({{11, 12, 13}, {21, 22, 23}}, torch::TensorOptions().dtype(torch::kInt32)).cuda();
    auto prefix_lengths = toCudaI32({80, 120});
    auto input_lengths  = toCudaI32({3, 0});

    auto outputs = mtpMsaTargetVerifyAddressingPrepare(request_block_table, prefix_lengths, input_lengths, 3);
    ASSERT_EQ(outputs.size(), 4);

    auto expected_block_table =
        torch::tensor({{11, 12, 13}, {11, 12, 13}, {11, 12, 13}, {21, 22, 23}, {21, 22, 23}, {21, 22, 23}},
                      torch::TensorOptions().dtype(torch::kInt32));
    auto expected_positions        = torch::tensor({80, 81, 82, 120, 121, 122}, torch::kInt32);
    auto expected_sequence_lengths = torch::tensor({81, 82, 83, 0, 0, 0}, torch::kInt32);
    auto expected_valid_mask       = torch::tensor({true, true, true, false, false, false}, torch::kBool);

    EXPECT_TRUE(torch::equal(outputs[0].cpu(), expected_block_table));
    EXPECT_TRUE(torch::equal(outputs[1].cpu(), expected_positions));
    EXPECT_TRUE(torch::equal(outputs[2].cpu(), expected_sequence_lengths));
    EXPECT_TRUE(torch::equal(outputs[3].cpu(), expected_valid_mask));
}

TEST(MtpTargetVerifyPrepareTest, MatchesReferenceAtProposalBoundaries) {
    torch::manual_seed(20260811);
    for (const int64_t tokens_per_batch : {1, 8}) {
        auto request_block_table =
            torch::randint(0, 10000, {3, 7}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        auto prefix_lengths =
            torch::randint(1, 1000, {3}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        auto input_lengths = toCudaI32({1, 0, 3});
        expectMatchesReference(request_block_table, prefix_lengths, input_lengths, tokens_per_batch);
    }
}

TEST(MtpTargetVerifyPrepareTest, MatchesReferenceForProductionWidth) {
    torch::manual_seed(20260812);
    auto request_block_table =
        torch::randint(0, 20000, {16, 641}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto prefix_lengths =
        torch::randint(1, 81920, {16}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto input_lengths = torch::ones({16}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    input_lengths.index_put_({3}, 0);
    input_lengths.index_put_({15}, 0);
    expectMatchesReference(request_block_table, prefix_lengths, input_lengths, 8);
}

TEST(MtpTargetVerifyPrepareTest, AllPaddedRowsKeepPositionsButMaskSequenceLengths) {
    auto request_block_table = torch::tensor({{11, 12}, {21, 22}}, torch::TensorOptions().dtype(torch::kInt32)).cuda();
    auto prefix_lengths      = toCudaI32({80, 120});
    auto input_lengths       = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto outputs = mtpMsaTargetVerifyAddressingPrepare(request_block_table, prefix_lengths, input_lengths, 8);

    expectMatchesReference(request_block_table, prefix_lengths, input_lengths, 8);
    EXPECT_TRUE(
        torch::equal(outputs[1].cpu(), torch::cat({torch::arange(80, 88), torch::arange(120, 128)}).to(torch::kInt32)));
    EXPECT_EQ(outputs[2].count_nonzero().item<int64_t>(), 0);
    EXPECT_EQ(outputs[3].count_nonzero().item<int64_t>(), 0);
}

TEST(MtpTargetVerifyPrepareTest, RunsOnCurrentNonDefaultStream) {
    const auto original_stream = at::cuda::getCurrentCUDAStream();
    const auto test_stream     = at::cuda::getStreamFromPool(/*isHighPriority=*/false);
    auto request_block_table   = torch::zeros({2, 4}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto prefix_lengths        = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto input_lengths         = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    {
        at::cuda::CUDAStreamGuard stream_guard(test_stream);
        request_block_table.copy_(torch::tensor({{1, 2, 3, 4}, {5, 6, 7, 8}}, torch::kInt32).cuda());
        prefix_lengths.copy_(toCudaI32({64, 128}));
        input_lengths.copy_(toCudaI32({1, 0}));
        expectMatchesReference(request_block_table, prefix_lengths, input_lengths, 3);
        test_stream.synchronize();
    }

    EXPECT_EQ(at::cuda::getCurrentCUDAStream(), original_stream);
}

TEST(MtpTargetVerifyPrepareTest, CapturesAndReplaysWithUpdatedInputs) {
    const auto test_stream   = at::cuda::getStreamFromPool(/*isHighPriority=*/false);
    auto request_block_table = torch::zeros({2, 4}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto prefix_lengths      = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto input_lengths       = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    at::cuda::CUDAStreamGuard stream_guard(test_stream);
    request_block_table.copy_(torch::tensor({{1, 2, 3, 4}, {5, 6, 7, 8}}, torch::kInt32).cuda());
    prefix_lengths.copy_(toCudaI32({64, 128}));
    input_lengths.copy_(toCudaI32({1, 0}));

    // Warm the allocator outside capture, then retain the captured output
    // tensors for the full graph lifetime.
    auto warmup_outputs = mtpMsaTargetVerifyAddressingPrepare(request_block_table, prefix_lengths, input_lengths, 3);
    test_stream.synchronize();
    warmup_outputs.clear();

    at::cuda::CUDAGraph        graph;
    std::vector<torch::Tensor> captured_outputs;
    graph.capture_begin();
    captured_outputs = mtpMsaTargetVerifyAddressingPrepare(request_block_table, prefix_lengths, input_lengths, 3);
    graph.capture_end();

    request_block_table.copy_(torch::tensor({{11, 12, 13, 14}, {21, 22, 23, 24}}, torch::kInt32).cuda());
    prefix_lengths.copy_(toCudaI32({80, 160}));
    input_lengths.copy_(toCudaI32({1, 1}));
    graph.replay();
    test_stream.synchronize();
    expectTensorVectorsEqual(captured_outputs,
                             referenceAddressing(request_block_table, prefix_lengths, input_lengths, 3),
                             request_block_table.get_device());

    request_block_table.copy_(torch::tensor({{31, 32, 33, 34}, {41, 42, 43, 44}}, torch::kInt32).cuda());
    prefix_lengths.copy_(toCudaI32({96, 192}));
    input_lengths.copy_(toCudaI32({0, 1}));
    graph.replay();
    test_stream.synchronize();
    expectTensorVectorsEqual(captured_outputs,
                             referenceAddressing(request_block_table, prefix_lengths, input_lengths, 3),
                             request_block_table.get_device());
}

}  // namespace
}  // namespace rtp_llm
