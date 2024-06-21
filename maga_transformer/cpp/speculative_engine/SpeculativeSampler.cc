#include "maga_transformer/cpp/speculative_engine/SpeculativeSampler.h"
#include <cmath>
#include <random>
#include <cassert>

using namespace std;
namespace ft = fastertransformer;

namespace rtp_llm {

SpeculativeSampler::SpeculativeSampler(const SamplerInitParams& params): device_(params.device) {}

SpeculativeSamplerOutput SpeculativeSampler::forward(const SpeculativeSamplerInput& inputs) {
    const auto&              draft_prob  = inputs.draft_prob;
    const auto&              target_prob = inputs.target_prob;
    const auto&              token_ids   = inputs.token_ids;
    size_t                   batch_size  = token_ids->shape()[0];
    size_t                   max_seq_len = token_ids->shape()[1];
    size_t                   vocab_size  = inputs.target_prob->shape()[2];
    size_t                   gen_num_per_circle     = inputs.gen_num_per_circle;
    SpeculativeSamplerOutput output;
    FT_CHECK(token_ids->dim() == 2);
    FT_CHECK(draft_prob->dim() == 2);
    FT_CHECK(target_prob->dim() == 2);
    FT_CHECK(token_ids->shape()[0] == 1);  // tmp not support batch
    auto sample_tokens_host =
        device_->allocateBuffer({token_ids->type(), token_ids->shape(), ft::AllocationType::HOST});
    auto draft_prob_host = device_->allocateBuffer({draft_prob->type(), draft_prob->shape(), ft::AllocationType::HOST});
    auto target_prob_host =
        device_->allocateBuffer({draft_prob->type(), draft_prob->shape(), ft::AllocationType::HOST});
    output.output_token_ids =
        device_->allocateBuffer({token_ids->type(), {batch_size, gen_num_per_circle}, ft::AllocationType::HOST});
    auto& output_token_len = output.output_token_len;
    output_token_len.resize(batch_size, 0);
    device_->copy({*draft_prob_host, *draft_prob});
    device_->copy({*target_prob_host, *target_prob});
    device_->copy({*sample_tokens_host, *inputs.token_ids});
    auto                             draft_prob_ptr   = (float(*)[gen_num_per_circle][vocab_size])draft_prob_host->data();
    auto                             target_prob_ptr  = (float(*)[gen_num_per_circle][vocab_size])target_prob_host->data();
    auto                             output_token_ptr = (int32_t(*)[gen_num_per_circle])output.output_token_ids->data();
    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (auto i = 0; i < batch_size; ++i) {
        uint accept_n = 0;
        for (auto j = 0; j < gen_num_per_circle; ++j) {
            double rand_num = dis(gen);
            uint   token_id = *(sample_tokens_host->dataWithOffset<int32_t>((i + 1) * max_seq_len - gen_num_per_circle + j));
            if (rand_num
                < std::abs(target_prob_ptr[i][j][token_id]) / (std::abs(draft_prob_ptr[i][j][token_id]) + 1e-7)) {
                output_token_len[i] += 1;
                output_token_ptr[i][j] = token_id;
                continue;
            } else {
                output_token_len[i] += 1;
                output_token_ptr[i][j] =
                    std::max_element(target_prob_ptr[i][j], target_prob_ptr[i][j] + vocab_size) - target_prob_ptr[i][j];
                break;
            }
        }
    }
    return output;
}

}  // namespace rtp_llm
