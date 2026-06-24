#pragma once

#include "rtp_llm/models_py/bindings/core/OpData.h"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <initializer_list>
#include <torch/torch.h>
#include <vector>

using namespace rtp_llm;

namespace rtp_llm {
void rejectionSampling(const RejectionSamplingParams& params);
}

class RejectionSamplingOpTest: public ::testing::Test {
protected:
    struct ReferenceOutput {
        std::vector<int32_t> token_ids;
        std::vector<int32_t> accepted_token_num;
    };

    void SetUp() override {
        torch::manual_seed(114514);
    }

    torch::Tensor cudaFloatTensor(const std::vector<int64_t>& shape, const std::vector<float>& data) {
        return cudaTensorFromVector<float>(shape, data, torch::kFloat32);
    }

    torch::Tensor cudaIntTensor(const std::vector<int64_t>& shape, const std::vector<int32_t>& data) {
        return cudaTensorFromVector<int32_t>(shape, data, torch::kInt32);
    }

    torch::Tensor cudaBoolTensor(const std::vector<bool>& data) {
        auto cpu = torch::empty({static_cast<int64_t>(data.size())}, torch::TensorOptions().dtype(torch::kBool));
        auto ptr = cpu.data_ptr<bool>();
        for (size_t i = 0; i < data.size(); ++i) {
            ptr[i] = data[i];
        }
        return cpu.to(torch::kCUDA);
    }

    void runReferenceCases() {
        constexpr int batch_size             = 4;
        constexpr int num_speculative_tokens = 3;
        constexpr int vocab_size             = 5;
        constexpr int target_token_stride    = 2;

        std::vector<float>   draft_probs(batch_size * num_speculative_tokens * vocab_size, 0.2f);
        std::vector<float>   target_probs(batch_size * (num_speculative_tokens + 1) * vocab_size, 0.2f);
        std::vector<int32_t> draft_token_ids{
            1,
            2,
            3,
            0,
            4,
            2,
            3,
            1,
            0,
            2,
            4,
            1,
        };
        std::vector<int32_t> target_token_ids(batch_size * (num_speculative_tokens + 1) * target_token_stride, -100);
        std::vector<float>   uniform_samples(batch_size * (num_speculative_tokens + 1), 0.5f);
        std::vector<bool>    do_sample{true, false, true, true};

        setTargetTokenIds(target_token_ids, batch_size, num_speculative_tokens, target_token_stride, 0, {1, 2, 3, 4});
        setTargetTokenIds(target_token_ids, batch_size, num_speculative_tokens, target_token_stride, 1, {0, 2, 3, 4});
        setTargetTokenIds(target_token_ids, batch_size, num_speculative_tokens, target_token_stride, 2, {4, 2, 0, 1});
        setTargetTokenIds(target_token_ids, batch_size, num_speculative_tokens, target_token_stride, 3, {0, 4, 3, 4});

        setProbRow(draft_probs, num_speculative_tokens, vocab_size, 2, 0, {0.05f, 0.05f, 0.05f, 0.20f, 0.65f});
        setProbRow(target_probs, num_speculative_tokens + 1, vocab_size, 2, 0, {0.05f, 0.05f, 0.05f, 0.80f, 0.05f});
        setProbRow(draft_probs, num_speculative_tokens, vocab_size, 2, 1, {0.10f, 0.70f, 0.10f, 0.05f, 0.05f});
        setProbRow(target_probs, num_speculative_tokens + 1, vocab_size, 2, 1, {0.05f, 0.10f, 0.60f, 0.20f, 0.05f});
        uniform_samples[2 * (num_speculative_tokens + 1) + 2] = 0.90f;

        setProbRow(draft_probs, num_speculative_tokens, vocab_size, 3, 0, {0.10f, 0.10f, 0.50f, 0.20f, 0.10f});
        setProbRow(target_probs, num_speculative_tokens + 1, vocab_size, 3, 0, {0.05f, 0.05f, 0.80f, 0.05f, 0.05f});
        setProbRow(draft_probs, num_speculative_tokens, vocab_size, 3, 2, {0.10f, 0.50f, 0.10f, 0.20f, 0.10f});
        setProbRow(target_probs, num_speculative_tokens + 1, vocab_size, 3, 2, {0.025f, 0.90f, 0.025f, 0.025f, 0.025f});
        setProbRow(target_probs, num_speculative_tokens + 1, vocab_size, 3, 3, {0.10f, 0.20f, 0.30f, 0.25f, 0.15f});
        uniform_samples[3 * (num_speculative_tokens + 1) + 0] = 0.20f;
        uniform_samples[3 * (num_speculative_tokens + 1) + 2] = 0.10f;
        uniform_samples[3 * (num_speculative_tokens + 1) + 3] = 0.55f;

        auto expected = referenceRejectionSampling(batch_size,
                                                   num_speculative_tokens,
                                                   vocab_size,
                                                   target_token_stride,
                                                   draft_probs,
                                                   draft_token_ids,
                                                   uniform_samples,
                                                   target_probs,
                                                   target_token_ids,
                                                   do_sample);

        RejectionSamplingParams params{
            cudaFloatTensor({batch_size, num_speculative_tokens, vocab_size}, draft_probs),
            cudaIntTensor({batch_size, num_speculative_tokens}, draft_token_ids),
            cudaFloatTensor({batch_size, num_speculative_tokens + 1}, uniform_samples),
            cudaFloatTensor({batch_size, num_speculative_tokens + 1, vocab_size}, target_probs),
            cudaIntTensor({batch_size * (num_speculative_tokens + 1), target_token_stride}, target_token_ids),
            torch::full({batch_size, num_speculative_tokens + 1},
                        -7,
                        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)),
            torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)),
            cudaBoolTensor(do_sample),
        };

        rejectionSampling(params);

        assertVectorEqual(getTensorValues<int32_t>(params.output_token_ids_d), expected.token_ids);
        assertVectorEqual(getTensorValues<int32_t>(params.output_accepted_token_num_d), expected.accepted_token_num);
    }

    void runZeroAndOneSpeculativeTokenCases() {
        {
            constexpr int batch_size             = 2;
            constexpr int num_speculative_tokens = 0;
            constexpr int vocab_size             = 4;
            constexpr int target_token_stride    = 2;

            std::vector<float>   draft_probs;
            std::vector<float>   target_probs(batch_size * (num_speculative_tokens + 1) * vocab_size, 0.25f);
            std::vector<int32_t> draft_token_ids;
            std::vector<float>   uniform_samples(batch_size * (num_speculative_tokens + 1), 0.0f);
            std::vector<int32_t> target_token_ids{10, 2, 11, 3};
            std::vector<bool>    do_sample{false, true};

            auto expected = referenceRejectionSampling(batch_size,
                                                       num_speculative_tokens,
                                                       vocab_size,
                                                       target_token_stride,
                                                       draft_probs,
                                                       draft_token_ids,
                                                       uniform_samples,
                                                       target_probs,
                                                       target_token_ids,
                                                       do_sample);

            RejectionSamplingParams params{
                cudaFloatTensor({batch_size, num_speculative_tokens, vocab_size}, draft_probs),
                cudaIntTensor({batch_size, num_speculative_tokens}, draft_token_ids),
                cudaFloatTensor({batch_size, num_speculative_tokens + 1}, uniform_samples),
                cudaFloatTensor({batch_size, num_speculative_tokens + 1, vocab_size}, target_probs),
                cudaIntTensor({batch_size * (num_speculative_tokens + 1), target_token_stride}, target_token_ids),
                torch::full({batch_size, num_speculative_tokens + 1},
                            -7,
                            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)),
                torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)),
                cudaBoolTensor(do_sample),
            };

            rejectionSampling(params);

            assertVectorEqual(getTensorValues<int32_t>(params.output_token_ids_d), expected.token_ids);
            assertVectorEqual(getTensorValues<int32_t>(params.output_accepted_token_num_d),
                              expected.accepted_token_num);
        }

        {
            constexpr int batch_size             = 2;
            constexpr int num_speculative_tokens = 1;
            constexpr int vocab_size             = 4;
            constexpr int target_token_stride    = 1;

            std::vector<float>   draft_probs(batch_size * num_speculative_tokens * vocab_size, 0.25f);
            std::vector<float>   target_probs(batch_size * (num_speculative_tokens + 1) * vocab_size, 0.25f);
            std::vector<int32_t> draft_token_ids{1, 0};
            std::vector<float>   uniform_samples(batch_size * (num_speculative_tokens + 1), 0.5f);
            std::vector<int32_t> target_token_ids{1, 3, 2, 0};
            std::vector<bool>    do_sample{true, false};

            auto expected = referenceRejectionSampling(batch_size,
                                                       num_speculative_tokens,
                                                       vocab_size,
                                                       target_token_stride,
                                                       draft_probs,
                                                       draft_token_ids,
                                                       uniform_samples,
                                                       target_probs,
                                                       target_token_ids,
                                                       do_sample);

            RejectionSamplingParams params{
                cudaFloatTensor({batch_size, num_speculative_tokens, vocab_size}, draft_probs),
                cudaIntTensor({batch_size, num_speculative_tokens}, draft_token_ids),
                cudaFloatTensor({batch_size, num_speculative_tokens + 1}, uniform_samples),
                cudaFloatTensor({batch_size, num_speculative_tokens + 1, vocab_size}, target_probs),
                cudaIntTensor({batch_size * (num_speculative_tokens + 1), target_token_stride}, target_token_ids),
                torch::full({batch_size, num_speculative_tokens + 1},
                            -7,
                            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)),
                torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)),
                cudaBoolTensor(do_sample),
            };

            rejectionSampling(params);

            assertVectorEqual(getTensorValues<int32_t>(params.output_token_ids_d), expected.token_ids);
            assertVectorEqual(getTensorValues<int32_t>(params.output_accepted_token_num_d),
                              expected.accepted_token_num);
        }
    }

    void runRejectsInvalidTensorMetadata() {
        constexpr int batch_size             = 2;
        constexpr int num_speculative_tokens = 1;
        constexpr int vocab_size             = 4;
        constexpr int target_token_stride    = 1;

        auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto int_options   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        auto bool_options  = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);

        auto draft_probs =
            torch::full({batch_size, num_speculative_tokens, vocab_size}, 0.25f, float_options).contiguous();
        auto draft_token_ids = torch::zeros({batch_size, num_speculative_tokens}, int_options);
        auto uniform_samples = torch::zeros({batch_size, num_speculative_tokens + 1}, float_options);
        auto target_probs =
            torch::full({batch_size, num_speculative_tokens + 1, vocab_size}, 0.25f, float_options).contiguous();
        auto target_token_ids =
            torch::zeros({batch_size * (num_speculative_tokens + 1), target_token_stride}, int_options);
        auto output_token_ids          = torch::zeros({batch_size, num_speculative_tokens + 1}, int_options);
        auto output_accepted_token_num = torch::zeros({batch_size}, int_options);
        auto do_sample                 = torch::ones({batch_size}, bool_options);

        auto makeParams = [&]() {
            return RejectionSamplingParams{draft_probs,
                                           draft_token_ids,
                                           uniform_samples,
                                           target_probs,
                                           target_token_ids,
                                           output_token_ids,
                                           output_accepted_token_num,
                                           do_sample};
        };

        auto bad_uniform = torch::zeros({num_speculative_tokens + 1, batch_size}, float_options).transpose(0, 1);
        auto params      = makeParams();
        params.uniform_samples_d = bad_uniform;
        EXPECT_ANY_THROW(rejectionSampling(params));

        params = makeParams();
        params.target_token_ids_d =
            torch::zeros({batch_size * (num_speculative_tokens + 1) - 1, target_token_stride}, int_options);
        EXPECT_ANY_THROW(rejectionSampling(params));

        params                    = makeParams();
        params.output_token_ids_d = torch::zeros({batch_size, num_speculative_tokens}, int_options);
        EXPECT_ANY_THROW(rejectionSampling(params));

        params             = makeParams();
        params.do_sample_d = torch::ones({batch_size + 1}, bool_options);
        EXPECT_ANY_THROW(rejectionSampling(params));

        params                    = makeParams();
        params.target_token_ids_d = target_token_ids.cpu();
        EXPECT_ANY_THROW(rejectionSampling(params));

        params             = makeParams();
        params.do_sample_d = torch::ones({batch_size}, int_options);
        EXPECT_ANY_THROW(rejectionSampling(params));
    }

private:
    template<typename T>
    torch::Tensor
    cudaTensorFromVector(const std::vector<int64_t>& shape, const std::vector<T>& data, c10::ScalarType dtype) {
        auto cpu = torch::empty(shape, torch::TensorOptions().dtype(dtype));
        if (!data.empty()) {
            std::memcpy(cpu.data_ptr<T>(), data.data(), data.size() * sizeof(T));
        }
        return cpu.to(torch::kCUDA);
    }

    template<typename T>
    std::vector<T> getTensorValues(const torch::Tensor& tensor) {
        auto           cpu_tensor = tensor.cpu().contiguous();
        std::vector<T> values(cpu_tensor.numel());
        std::memcpy(values.data(), cpu_tensor.data_ptr<T>(), sizeof(T) * cpu_tensor.numel());
        return values;
    }

    template<typename T>
    void assertVectorEqual(const std::vector<T>& actual, const std::vector<T>& expected) {
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            ASSERT_EQ(actual[i], expected[i]) << "vectors differ at index " << i;
        }
    }

    void setProbRow(std::vector<float>&          probs,
                    int                          rows_per_batch,
                    int                          vocab_size,
                    int                          batch_idx,
                    int                          row_idx,
                    std::initializer_list<float> values) {
        ASSERT_EQ(static_cast<int>(values.size()), vocab_size);
        auto offset = (batch_idx * rows_per_batch + row_idx) * vocab_size;
        std::copy(values.begin(), values.end(), probs.begin() + offset);
    }

    void setTargetTokenIds(std::vector<int32_t>&          target_token_ids,
                           int                            batch_size,
                           int                            num_speculative_tokens,
                           int                            target_token_stride,
                           int                            batch_idx,
                           std::initializer_list<int32_t> ids) {
        (void)batch_size;
        ASSERT_EQ(static_cast<int>(ids.size()), num_speculative_tokens + 1);
        int row = 0;
        for (auto id : ids) {
            target_token_ids[((batch_idx * (num_speculative_tokens + 1) + row) * target_token_stride)
                             + target_token_stride - 1] = id;
            ++row;
        }
    }

    ReferenceOutput referenceRejectionSampling(int                         batch_size,
                                               int                         num_speculative_tokens,
                                               int                         vocab_size,
                                               int                         target_token_stride,
                                               const std::vector<float>&   draft_probs,
                                               const std::vector<int32_t>& draft_token_ids,
                                               const std::vector<float>&   uniform_samples,
                                               const std::vector<float>&   target_probs,
                                               const std::vector<int32_t>& target_token_ids,
                                               const std::vector<bool>&    do_sample) {
        ReferenceOutput output;
        output.token_ids.assign(batch_size * (num_speculative_tokens + 1), -1);
        output.accepted_token_num.assign(batch_size, 0);

        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            bool all_same_token = true;
            int  pos            = num_speculative_tokens;
            for (int i = 0; i < num_speculative_tokens; ++i) {
                auto draft_id = draft_token_ids[batch_idx * num_speculative_tokens + i];
                auto target_id =
                    targetTokenId(target_token_ids, num_speculative_tokens, target_token_stride, batch_idx, i);
                auto q = target_probs[((batch_idx * (num_speculative_tokens + 1) + i) * vocab_size) + draft_id];
                auto p = draft_probs[((batch_idx * num_speculative_tokens + i) * vocab_size) + draft_id];
                auto u = uniform_samples[batch_idx * (num_speculative_tokens + 1) + i];

                bool same_token = target_id == draft_id;
                if (same_token || (do_sample[batch_idx] && u * p < q)) {
                    output.token_ids[batch_idx * (num_speculative_tokens + 1) + i] = draft_id;
                    all_same_token                                                 = all_same_token && same_token;
                } else {
                    pos = i;
                    break;
                }
            }

            output.accepted_token_num[batch_idx] = pos + 1;

            if (all_same_token) {
                output.token_ids[batch_idx * (num_speculative_tokens + 1) + pos] =
                    targetTokenId(target_token_ids, num_speculative_tokens, target_token_stride, batch_idx, pos);
                continue;
            }

            output.token_ids[batch_idx * (num_speculative_tokens + 1) + pos] = sampleFromReluDiff(
                batch_idx, pos, num_speculative_tokens, vocab_size, draft_probs, uniform_samples, target_probs);
        }

        return output;
    }

    int32_t targetTokenId(const std::vector<int32_t>& target_token_ids,
                          int                         num_speculative_tokens,
                          int                         target_token_stride,
                          int                         batch_idx,
                          int                         row_idx) {
        return target_token_ids[((batch_idx * (num_speculative_tokens + 1) + row_idx) * target_token_stride)
                                + target_token_stride - 1];
    }

    int32_t sampleFromReluDiff(int                       batch_idx,
                               int                       pos,
                               int                       num_speculative_tokens,
                               int                       vocab_size,
                               const std::vector<float>& draft_probs,
                               const std::vector<float>& uniform_samples,
                               const std::vector<float>& target_probs) {
        std::vector<float> relu_q_minus_p(vocab_size, 0.0f);
        float              sum = 0.0f;
        for (int token_id = 0; token_id < vocab_size; ++token_id) {
            auto q = target_probs[((batch_idx * (num_speculative_tokens + 1) + pos) * vocab_size) + token_id];
            auto p = pos == num_speculative_tokens ?
                         0.0f :
                         draft_probs[((batch_idx * num_speculative_tokens + pos) * vocab_size) + token_id];
            relu_q_minus_p[token_id] = std::max(q - p, 0.0f);
            sum += relu_q_minus_p[token_id];
        }

        auto uniform_idx = batch_idx * (num_speculative_tokens + 1) + std::min(pos + 1, num_speculative_tokens);
        auto threshold   = uniform_samples[uniform_idx] * sum;
        auto aggregate   = 0.0f;
        for (int token_id = 0; token_id < vocab_size; ++token_id) {
            aggregate += relu_q_minus_p[token_id];
            if (relu_q_minus_p[token_id] > 0.0f && aggregate > threshold) {
                return token_id;
            }
        }
        return vocab_size - 1;
    }
};
