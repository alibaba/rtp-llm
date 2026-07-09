#include "rtp_llm/models_py/bindings/core/OpData.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <vector>

namespace rtp_llm {
GreedyOutput sampleGreedy(const GreedyParams& params);
}

using namespace rtp_llm;

namespace {

torch::Tensor deviceFloatTensor(std::vector<float> values, std::vector<int64_t> shape) {
    return torch::tensor(values, torch::kFloat32).reshape(shape).to(torch::kCUDA);
}

torch::Tensor deviceIntTensor(std::vector<int32_t> values, std::vector<int64_t> shape) {
    return torch::tensor(values, torch::kInt32).reshape(shape).to(torch::kCUDA);
}

torch::Tensor pinnedFloatTensor(std::vector<float> values) {
    return torch::tensor(values, torch::kFloat32).pin_memory();
}

torch::Tensor pinnedIntTensor(std::vector<int32_t> values) {
    return torch::tensor(values, torch::kInt32).pin_memory();
}

}  // namespace

TEST(SampleGreedyOrderTest, AppliesFrequencyPenaltyBeforeTemperatureForLowTempTopKOne) {
    constexpr size_t batch_size = 1;
    constexpr size_t vocab_size = 3;
    constexpr size_t step       = 4;

    auto logits = deviceFloatTensor({1.001f, 1.0f, 0.0f}, {batch_size, vocab_size});

    // Token 0 has only a tiny raw-logit lead, but it has already appeared four times.
    auto token_ids        = deviceIntTensor({0, 0, 0, 0, 0}, {batch_size, step + 1});
    auto input_lengths    = torch::tensor({1}, torch::kInt32);
    auto sequence_lengths = torch::tensor({4}, torch::kInt32);

    auto top_k       = pinnedIntTensor({1});
    auto top_p       = pinnedFloatTensor({1.0f});
    auto temperature = pinnedFloatTensor({1e-6f});

    auto repetition_penalty = pinnedFloatTensor({1.0f});
    auto presence_penalty   = pinnedFloatTensor({0.0f});
    auto frequency_penalty  = pinnedFloatTensor({0.01f});

    std::vector<at::Generator> generator(batch_size);
    GreedyParams params({logits,
                         input_lengths,
                         sequence_lengths,
                         token_ids,
                         step,
                         top_k,
                         top_p,
                         temperature,
                         repetition_penalty,
                         std::nullopt,
                         std::nullopt,
                         std::nullopt,
                         std::nullopt,
                         presence_penalty,
                         frequency_penalty,
                         std::nullopt,
                         generator});

    sampleGreedy(params);

    auto output_token_ids = token_ids.cpu().contiguous();
    EXPECT_EQ(output_token_ids.data_ptr<int32_t>()[step], 1);
}
