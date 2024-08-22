#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/cpu_allocator.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"

#include <cstring>
#include <random>
#include <vector>
#include <set>
#include <cfloat>
#include <omp.h>
#include <functional>

#define TOP_K_MAX 1024

namespace fastertransformer {

float random_one_data(std::mt19937& generator) {
    static std::uniform_real_distribution<float> dist(0.0, 1.0);
    float rand = dist(generator);
    return rand;
}

void temperaturePenalty(float *           logits,
                        const float* temperatures,
                        const int    batch_size,
                        const int    vocab_size) {
    for (int i = 0; i < batch_size; i++) {
        float inv_temperatures = 1.0f / (temperatures[i] + 1e-6f);
        for (int j = 0; j < vocab_size; j++) {
            logits[i * vocab_size + j] *= inv_temperatures;
        }
    }
}

void setup_topk_runtime_args(int    batch_size,
                             uint   top_k,
                             uint*  top_ks,
                             int    top_ks_size,
                             float  top_p,
                             float* top_ps,
                             int    top_ps_size,
                             bool*  skip_decode)
{
    for (int i = 0; i < batch_size; i++) {
        uint  k = top_ks_size > 1 ? top_ks[i] : top_k;
        float p = top_ps_size > 1 ? top_ps[i] : top_p;
        if (k == 0 && p == 0.0f) {
            k = 1;
        }
        if (k > 0 && p == 0.0f) {
            p = 1.0f;
        }
        // Clip k value. A topk sampling kernel supports up to TOP_K_MAX=64.
        top_ks[i] = k > TOP_K_MAX ? TOP_K_MAX : k;
        if (k > TOP_K_MAX) {
            printf("[WARNING] topk (%d) is larger than max supported number (%d) for token %d"
                   " clip to max supported number %d. \n",
                   k,
                   TOP_K_MAX,
                   i,
                   top_ks[i]);
        }
        // Clip p value if it is out of range. range = [0.0, 1.0].
        top_ps[i] = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
        if (p < 0.0f || p > 1.0f) {
            printf("[WARNING] topp (%f) is out of range ([0.0, 1.0f]) for token %d"
                   " clip to closest number %f.\n",
                   p,
                   i,
                   top_ps[i]);
        }
        skip_decode[i] = k == 0;
    }
}

void repetitionPenalty(float *           logits,
                       const float* penalties,
                       const int*   output_ids,
                       const int    batch_size,
                       const int    vocab_size,
                       const int*   input_lengths,
                       const int    max_input_length,
                       const int    step) {
    for (int i = 0; i < batch_size; i++) {
        const int input_length = input_lengths != nullptr ? input_lengths[i] : max_input_length;
        std::set<int> preIdSet;
        for (int j = 0; j < step && j < input_length; j++) {
            if (j >= input_lengths[i]) {
                continue;
            }
            preIdSet.insert(output_ids[i * (step + 1) + j]);
        }
        for (auto id : preIdSet) {
            float logit = logits[i * vocab_size + id];
            logits[i * vocab_size + id] = logit < 0.0f ? logit * penalties[i] : logit / penalties[i];
        }
    }
}

void minLengthPenaltyNew(float*         logits,
                         const int* min_lengths,
                         const int* end_ids,
                         const int* sequence_lengths,
                         const int* input_lengths,
                         const int batch_size,
                         const int vocab_size) {
    for (int i = 0; i < batch_size; i++) {
        if (sequence_lengths[i] - input_lengths[i] < min_lengths[i]) {
            logits[i * vocab_size + end_ids[i]] = -FLT_MAX;;
        }
    }
}

void topKKernel(float* output,
                int* output_indices,
                const float* logits,
                int batch_size,
                int vocab_size,
                int max_top_k,
                const uint32_t* topk) {
    parallel_for(batch_size, [&](int b) {
        std::vector<std::pair<float, int> > data_vec(vocab_size);
        parallel_for(vocab_size, [&](int i) {
            data_vec[i] = std::make_pair(logits[b * vocab_size + i], i);
        });
        std::partial_sort(data_vec.begin(), data_vec.begin() + topk[b], data_vec.end(),
                          [](std::pair<float, int> lhs, std::pair<float, int> rhs) {
                              if (lhs.first > rhs.first) return true;
                              if (lhs.first < rhs.first) return false;
                              return lhs.second < rhs.second;
                          });
        parallel_for(topk[b], [&](int i) {
            output[b * max_top_k + i] = data_vec[i].first;
            output_indices[b * max_top_k + i] = data_vec[i].second;
        });
    });
}

void topk_sampling(int    batch_size,
                   const int*  topk_tmp_id_buf,
                   float*         topk_tmp_val_buf,
                   std::vector<std::mt19937>& generator_lists,
                   int*           ids,
                   int*           sequence_length,
                   bool*          finished,
                   float*         cum_log_probs,
                   float*         output_log_probs,
                   int*           token_id_for_index_prob,
                   const int      max_top_k,
                   const uint32_t*     top_ks,
                   const float    top_p,
                   const float*   top_ps,
                   const int*     end_ids,
                   const int      vocab_size,
                   const bool*    skip_decode,
                   int step)
{
    parallel_for(batch_size, [&](int batch_id) {
        if (skip_decode != nullptr && skip_decode[batch_id]) {
            return;
        }

        if (finished != nullptr && finished[batch_id] == true) {
            ids[batch_id * step + step - 1] = end_ids[batch_id];
            return;
        }

        const int k = (top_ks != nullptr) ? top_ks[batch_id] : max_top_k;
        const float prob_threshold = (top_ps != nullptr) ? top_ps[batch_id] : top_p;

        float s_sum = 0.0f;
        float s_max = topk_tmp_val_buf[batch_id * max_top_k];
        for (int i = 0; i < k; i++) {
            topk_tmp_val_buf[batch_id * max_top_k + i] = std::exp(topk_tmp_val_buf[batch_id * max_top_k + i] - s_max);
            s_sum += topk_tmp_val_buf[batch_id * max_top_k + i];
        }

        float rand_num = random_one_data(generator_lists[batch_id]) * prob_threshold * s_sum;
        for (int i = 0; i < k; i++) {
            float exp_logit = topk_tmp_val_buf[batch_id * max_top_k + i];
            rand_num = rand_num - exp_logit;
            if (rand_num <= 0.0f || i == k - 1) {
                ids[batch_id * step + step - 1] = topk_tmp_id_buf[batch_id * max_top_k + i];
                if (cum_log_probs != nullptr || output_log_probs != nullptr) {
                    float log_prob = logf(exp_logit) - logf(s_sum);
                    if (cum_log_probs != nullptr) {
                        cum_log_probs[batch_id] += log_prob;
                    }
                    if (output_log_probs != nullptr) {
                        output_log_probs[batch_id] = log_prob;
                    }
                }
                break;
            }
        }
        if (sequence_length != nullptr && finished != nullptr) {
            sequence_length[batch_id] = finished[batch_id] ? sequence_length[batch_id] : sequence_length[batch_id] +
                                                                                         1;
            finished[batch_id] = ids[batch_id] == end_ids[batch_id] ? true : false;
        }
    });
}

void ArmCpuDevice::sampleGreedy(const GreedyParams& params) {
    const auto& logits = params.logits;
    const auto batch_size = logits.shape()[0];
    RUNTIME_ASSERT_OP_ARG(batch_size < init_params_.max_batch_size,
                          "batch_size exceeded device limit %ld: %ld",
                          init_params_.max_batch_size, batch_size);
    const auto vocab_size_padded = logits.shape()[1];
    const auto step = params.step;
    RUNTIME_ASSERT_OP_ARG(batch_size == params.token_ids.shape()[0],
                          "logits.shape[0] should equal to token_ids.shape[0], but %ld vs %ld",
                          batch_size, params.token_ids.shape()[0]);
    RUNTIME_ASSERT_OP_ARG((step == params.token_ids.shape()[1] - 1),
                          "step should equal to token_ids.shape[1] - 1, but %ld vs %ld",
                          step, params.token_ids.shape()[1] - 1);
    auto& tokens = params.token_ids;
    // auto transposed_tokens = transpose({*device_tokens});

    // 1. prepare buffers
    auto& top_k = params.top_k;
    auto& top_p = params.top_p;
    auto& temperature = params.temperature;
    auto& random_seed = params.random_seed;
    FT_CHECK(top_k.size() == batch_size);
    FT_CHECK(top_p.size() == batch_size);
    FT_CHECK(temperature.size() == batch_size);

    auto default_top_k = top_k.data<uint32_t>()[0];
    auto default_top_p = top_p.data<float>()[0];
    
    if (default_top_k == 0) {
        default_top_k = 1;
    }

    auto max_top_k = *std::max_element(top_k.data<uint32_t>(), top_k.dataWithOffset<uint32_t>(top_k.size()));
    if (max_top_k == 0) {
        // for safety. TopKSamplingLayer handles a case of top_k=0 and top_p=0 as
        // a greedy decode, i.e. top_k=1, although such case has max_top_k=0.
        max_top_k = 1;
    }
    auto max_top_p = *std::max_element(top_p.data<float>(), top_p.dataWithOffset<float>(top_p.size()));
    FT_LOG_DEBUG("max_top_k: %d, max_top_p: %f", max_top_k, max_top_p);

    auto skip_top_k_decode_buf = allocateBuffer({DataType::TYPE_BOOL, {batch_size}});
    auto topk_tmp_val_buf = allocateBuffer({DataType::TYPE_FP32, {batch_size * max_top_k}});
    auto topk_tmp_id_buf = allocateBuffer({DataType::TYPE_INT32, {batch_size * max_top_k}});

    // std::mt19937 generator(seed);
    std::vector<std::mt19937> generator_lists(batch_size);
    unsigned one_seed = std::random_device{}();
    for (size_t i = 0; i < batch_size; i++) {
        generator_lists[i] = std::mt19937(one_seed + i);
    }
    if (random_seed) {
        auto& seeds = random_seed.value().get();
        uint64_t* seedsPtr = seeds.data<uint64_t>();
        if (seeds.size() == 1) {
            for (int i = 0; i < batch_size; i++) {
                generator_lists[i] = std::mt19937(seedsPtr[0]);
            }
        } else {
            RUNTIME_ASSERT_OP_ARG((seeds.size() == batch_size),
                                  "random_seed.size() should equal to batch_size, but %ld vs %ld",
                                  seeds.size(), batch_size);
            for (int i = 0; i < batch_size; i++) {
                generator_lists[i] = std::mt19937(seedsPtr[i]);
            }
        }
    }

    // 3.2. compute logits penalty
    if (std::any_of(temperature.data<float>(),
                    temperature.data<float>() + batch_size,
                    [&](auto t) { return t != 1.0f; }))
    {
        temperaturePenalty(
                logits.data<float>(),
                temperature.data<float>(),
                batch_size,
                vocab_size_padded
        );
    }

    const auto decoder_batch_size = params.sequence_lengths.shape()[0];
    if (decoder_batch_size) {
        if (step > 1 && params.repetition_penalty && decoder_batch_size) {
            auto& repetition_penalty = params.repetition_penalty->get();
            if (std::any_of(repetition_penalty.data<float>(),
                            repetition_penalty.data<float>() + batch_size,
                            [&](auto t) { return t != 1.0f; }))
            {
//                const auto repetition_penalty_type = RepetitionPenaltyType::Multiplicative;
                repetitionPenalty(logits.data<float>(),
                                  repetition_penalty.data<float>(),
                                  tokens.data<int32_t>(),
                                  batch_size,
                                  vocab_size_padded,
                                  params.sequence_lengths.data<int32_t>(),
                                  step + 1,
                                  step);
            }
        }
        if (params.min_lengths.has_value())
            if (params.min_lengths && params.eos_ids) {
                minLengthPenaltyNew(logits.data<float>(),
                                    params.min_lengths.value().get().data<int32_t>(),
                                    params.eos_ids.value().get().data<int32_t>(),
                                    params.sequence_lengths.data<int32_t>(),
                                    params.input_lengths.data<int32_t>(),
                                    batch_size,
                                    vocab_size_padded);
            }
    }

    // 4. run sampling
    // 4.1 run top_k
    setup_topk_runtime_args(batch_size,
                            default_top_k,
                            top_k.data<uint32_t>(),
                            batch_size,
                            default_top_p,
                            top_p.data<float>(),
                            batch_size,
                            skip_top_k_decode_buf->data<bool>());

    float* topk_logs = topk_tmp_val_buf->data<float>();
    int* topk_logs_indices = topk_tmp_id_buf->data<int>();

    topKKernel(topk_logs,
               topk_logs_indices,
               logits.data<float>(), batch_size,
               vocab_size_padded,
               max_top_k,
               top_k.data<uint32_t>());


    topk_sampling(batch_size, topk_logs_indices, topk_logs, generator_lists,
                  tokens.data<int32_t>(),
                  nullptr, // sequence_length
                  nullptr, // finished
                  nullptr, //          cum_log_probs,
                  nullptr, //         output_log_probs,
                  nullptr, //            token_id_for_index_prob,
                  max_top_k,
                  top_k.data<uint32_t>(),
                  1.0f,
                  top_p.data<float>(),
                  params.eos_ids.value().get().data<int32_t>(),
                  vocab_size_padded,
                  skip_top_k_decode_buf->data<bool>(),
                  step + 1);

    return;
}
}; // namespace fastertransformer
