#include "rtp_llm/cpp/devices/cpu_impl/CpuDevice.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/core/cpu_allocator.h"
#include <immintrin.h>
#include <random>
#include <cfloat>

using namespace std;

namespace rtp_llm {

static inline __m512 vexp(const __m512& _x) {
    __m512 p16f_1      = _mm512_set1_ps(1.0f);
    __m512 p16f_half   = _mm512_set1_ps(0.5f);
    __m512 p16f_127    = _mm512_set1_ps(127.f);
    __m512 p16f_exp_hi = _mm512_set1_ps(88.3762626647950f);
    __m512 p16f_exp_lo = _mm512_set1_ps(-88.3762626647949f);

    __m512 p16f_cephes_LOG2EF = _mm512_set1_ps(1.44269504088896341f);

    __m512 p16f_cephes_exp_p0 = _mm512_set1_ps(1.9875691500E-4f);
    __m512 p16f_cephes_exp_p1 = _mm512_set1_ps(1.3981999507E-3f);
    __m512 p16f_cephes_exp_p2 = _mm512_set1_ps(8.3334519073E-3f);
    __m512 p16f_cephes_exp_p3 = _mm512_set1_ps(4.1665795894E-2f);
    __m512 p16f_cephes_exp_p4 = _mm512_set1_ps(1.6666665459E-1f);
    __m512 p16f_cephes_exp_p5 = _mm512_set1_ps(5.0000001201E-1f);

    // Clamp x.
    __m512 x = _mm512_max_ps(_mm512_min_ps(_x, p16f_exp_hi), p16f_exp_lo);

    // Express exp(x) as exp(m*ln(2) + r), start by extracting
    // m = floor(x/ln(2) + 0.5).
    __m512 m = _mm512_floor_ps(_mm512_fmadd_ps(x, p16f_cephes_LOG2EF, p16f_half));

    // Get r = x - m*ln(2). If no FMA instructions are available, m*ln(2) is
    // subtracted out in two parts, m*C1+m*C2 = m*ln(2), to avoid accumulating
    // truncation errors. Note that we don't use the "pmadd" function here to
    // ensure that a precision-preserving FMA instruction is used.
    __m512 p16f_nln2 = _mm512_set1_ps(-0.6931471805599453f);
    __m512 r         = _mm512_fmadd_ps(m, p16f_nln2, x);

    __m512 r2 = _mm512_mul_ps(r, r);

    // TODO(gonnet): Split into odd/even polynomials and try to exploit
    //               instruction-level parallelism.
    __m512 y = p16f_cephes_exp_p0;
    y        = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p1);
    y        = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p2);
    y        = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p3);
    y        = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p4);
    y        = _mm512_fmadd_ps(y, r, p16f_cephes_exp_p5);
    y        = _mm512_fmadd_ps(y, r2, r);
    y        = _mm512_add_ps(y, p16f_1);

    // Build emm0 = 2^m.
    __m512i emm0 = _mm512_cvttps_epi32(_mm512_add_ps(m, p16f_127));
    emm0         = _mm512_slli_epi32(emm0, 23);

    // Return 2^m * exp(r).
    return _mm512_max_ps(_mm512_mul_ps(y, _mm512_castsi512_ps(emm0)), _x);
}

void computeSoftMax(float* data, int size) {
    int       vecs     = (size + 15) / 16;                                    // how many avx512 vectors
    __mmask16 tailMask = (size % 16 == 0 ? 0xffff : (1 << (size % 16)) - 1);  // mask of last vector

    __m512 vsum = _mm512_set1_ps(0);

    // maxVal is used to avoid exp(x) = inf
    float  maxVal = std::numeric_limits<float>::lowest();
    __m512 vmax   = _mm512_set1_ps(maxVal);

    int i = 0;
    for (i = 0; i < vecs; ++i) {
        __mmask16 k  = (i == vecs - 1 ? tailMask : 0xffff);
        __m512    vx = _mm512_maskz_loadu_ps(k, data + i * 16);
        vmax         = _mm512_mask_max_ps(vmax, k, vmax, vx);
    }

    maxVal = _mm512_reduce_max_ps(vmax);
    vmax   = _mm512_set1_ps(maxVal);

    // Compute vexp(vx - vmax) and sum it
    for (i = 0; i < vecs; ++i) {
        __mmask16 k  = (i == vecs - 1 ? tailMask : 0xffff);
        __m512    vx = _mm512_maskz_loadu_ps(k, data + i * 16);
        vx           = vexp(vx - vmax);
        _mm512_mask_storeu_ps(data + i * 16, k, vx);
        vsum = _mm512_mask_add_ps(vsum, k, vsum, vx);
    }

    float  sum   = _mm512_reduce_add_ps(vsum);
    __m512 vrsum = _mm512_set1_ps(1.0f / sum);

    // Compute exp/sum(exp) and store
    for (i = 0; i < vecs; ++i) {
        __mmask16 k  = (i == vecs - 1 ? tailMask : 0xffff);
        __m512    vx = _mm512_maskz_loadu_ps(k, data + i * 16);
        vx           = vx * vrsum;
        _mm512_mask_storeu_ps(data + i * 16, k, vx);
    }
}

void repetitionPenalty(float*       logits,
                       const float* penalties,
                       const int*   output_ids,
                       const int    batch_size,
                       const int    vocab_size,
                       const int*   input_lengths,
                       const int    max_input_length,
                       const int    step) {
    for (int i = 0; i < batch_size; i++) {
        const int     input_length = input_lengths != nullptr ? input_lengths[i] : max_input_length;
        std::set<int> preIdSet;
        for (int j = 0; j < step && j < input_length; j++) {
            if (j >= input_lengths[i]) {
                continue;
            }
            preIdSet.insert(output_ids[i * (step + 1) + j]);
        }
        for (auto id : preIdSet) {
            float logit                 = logits[i * vocab_size + id];
            logits[i * vocab_size + id] = logit < 0.0f ? logit * penalties[i] : logit / penalties[i];
        }
    }
}

void setup_topk(int    batch_size,
                uint   top_k,
                uint*  top_ks,
                int    top_ks_size,
                float  top_p,
                float* top_ps,
                int    top_ps_size,
                bool*  skip_decode) {
    for (int i = 0; i < batch_size; ++i) {
        uint  k = (top_ks_size > 1) ? top_ks[i] : top_k;
        float p = (top_ps_size > 1) ? top_ps[i] : top_p;

        if (k == 0 && p == 0.0f) {
            k = 1;
        }
        if (k > 0 && p == 0.0f) {
            p = 1.0f;
        }

        // Clip p value if it is out of range. range = [0.0, 1.0].
        top_ps[i] = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
        if (p < 0.0f || p > 1.0f) {
            std::cout << "[WARNING] topp (" << p << ") is out of range ([0.0, 1.0f]) for token " << i
                      << " clip to closest number " << top_ps[i] << ".\n";
        }

        skip_decode[i] = (k == 0);
    }
}

void setup_topp(int    batch_size,
                uint   top_k,
                uint*  top_ks,
                int    top_ks_size,
                float  top_p,
                float* top_ps,
                int    top_ps_size,
                bool*  skip_decode) {
    for (int i = 0; i < batch_size; ++i) {
        uint  k = top_ks_size > 1 ? top_ks[i] : top_k;
        float p = top_ps_size > 1 ? top_ps[i] : top_p;
        if (k == 0 && p == 0.0f) {
            k = 1;
        }
        top_ks[i] = k;

        // Clip p value if it is out of range. range = [0.0, 1.0].
        top_ps[i] = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
        if (p < 0.0f || p > 1.0f) {
            printf("[WARNING] topp (%f) is out of range ([0.0, 1.0f]) for token %d"
                   " clip to closest number %f.\n",
                   p,
                   i,
                   top_ps[i]);
        }
        skip_decode[i] = k > 0;
    }
}

void applyTemperaturePenalty(
    float* logits, const float* temperatures, int batch_size, int vocab_size, int vocab_size_padd) {
    // Prepare inverse temperatures with padding
    std::vector<float> inv_temperatures(batch_size);
    const float        epsilon = 1e-6f;

    // Calculating inverse temperatures
    int i = 0;
    // Process full vectors when possible
    for (; i <= batch_size - 16; i += 16) {
        __m512 temperatures_vec = _mm512_loadu_ps(&temperatures[i]);

        // Add a small epsilon to avoid division by zero
        __m512 epsilon_vec    = _mm512_set1_ps(epsilon);
        __m512 adjusted_temps = _mm512_add_ps(temperatures_vec, epsilon_vec);
        __m512 inv_temp_vec   = _mm512_div_ps(_mm512_set1_ps(1.0f), adjusted_temps);

        _mm512_storeu_ps(&inv_temperatures[i], inv_temp_vec);
    }

    // Handle remaining temperatures using masking if batch_size is not a multiple of 16
    if (i < batch_size) {
        // Load the remaining temperatures
        __m512 temperatures_vec = _mm512_loadu_ps(&temperatures[i]);
        __m512 epsilon_vec      = _mm512_set1_ps(epsilon);
        __m512 adjusted_temps   = _mm512_add_ps(temperatures_vec, epsilon_vec);

        // Calculate the inverse
        __m512 inv_temp_vec = _mm512_div_ps(_mm512_set1_ps(1.0f), adjusted_temps);

        // Determine how many elements to store
        int       remainder = batch_size - i;
        __mmask16 mask      = (1 << remainder) - 1;  // Create a mask for remaining elements

        // Store results back to inv_temperatures using the mask
        _mm512_mask_storeu_ps(&inv_temperatures[i], mask, inv_temp_vec);
    }

    // Apply temperature penalty
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        float inv_temp = inv_temperatures[batch_idx];  // Cache inverse temperature for current batch

        for (int vocab_idx = 0; vocab_idx < vocab_size_padd; ++vocab_idx) {
            int   index = batch_idx * vocab_size_padd + vocab_idx;
            float logit = (vocab_idx < vocab_size) ? logits[index] : -FLT_MAX;

            // Apply penalty only for valid vocab indices
            if (vocab_idx < vocab_size) {
                logit *= inv_temp;
            }
            logits[index] = logit;
        }
    }
}

void batchTopKSampling(const float* log_probs,
                       int*         ids,
                       int          step,
                       float*       cum_log_probs,
                       float*       output_log_probs,
                       const int    max_top_k,
                       uint32_t*    top_ks,
                       const int    vocab_size_padded,
                       const int    batch_size,
                       bool*        skip_decode,
                       float*       rand_nums) {
    const int vocab_size = vocab_size_padded;

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        if (skip_decode != nullptr && skip_decode[batch_id]) {
            continue;
        }

        const int k = (top_ks != nullptr) ? top_ks[batch_id] : max_top_k;

        // Store (log_prob, index) pairs
        std::vector<std::pair<float, int>> log_prob_index_pairs;
        for (int i = 0; i < vocab_size; ++i) {
            log_prob_index_pairs.emplace_back(log_probs[batch_id * vocab_size + i], i);
        }

        // Sort by log_probs in descending order
        std::sort(log_prob_index_pairs.begin(), log_prob_index_pairs.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first > rhs.first;
        });

        // Keep top k values
        std::vector<int>   topk_ids(k);
        std::vector<float> topk_vals(k);

        for (int i = 0; i < k; ++i) {
            topk_ids[i]  = log_prob_index_pairs[i].second;
            topk_vals[i] = log_prob_index_pairs[i].first;
        }

        // Compute probabilities
        float max_val = *std::max_element(topk_vals.begin(), topk_vals.end());
        float sum_exp = 0.0;
        for (const auto& val : topk_vals) {
            sum_exp += std::exp(val - max_val);
        }

        float rand_num = rand_nums[batch_id] * sum_exp;

        // Select token based on sampling
        for (int i = 0; i < k; ++i) {
            float exp_logit = std::exp(topk_vals[i] - max_val);
            rand_num -= exp_logit;
            if (rand_num <= 0.0f || i == k - 1) {
                ids[batch_id * (step + 1) + step] = topk_ids[i];
                if (cum_log_probs != nullptr || output_log_probs != nullptr) {
                    float log_prob = logf(exp_logit) - logf(sum_exp);
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
    }
}

void batchTopPSampling(int*         output_ids,
                       float*       cum_log_probs,
                       float*       output_log_probs,
                       const float* log_probs,
                       int          step,
                       const int    batch_size,
                       const size_t vocab_size_padded,
                       const float  max_top_p,
                       const float* top_ps,
                       const bool*  skip_decode,
                       float*       rand_nums) {
    const int vocab_size = vocab_size_padded;
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        if (skip_decode != nullptr && skip_decode[batch_id]) {
            continue;
        }

        float p_threshold = (top_ps != nullptr) ? top_ps[batch_id] : max_top_p;

        // Store (log_prob, index) pairs
        std::vector<std::pair<float, int>> log_prob_index_pairs;
        for (int i = 0; i < vocab_size; ++i) {
            log_prob_index_pairs.emplace_back(log_probs[batch_id * vocab_size + i], i);
        }

        // Sort by log_probs in descending order
        std::sort(log_prob_index_pairs.begin(), log_prob_index_pairs.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first > rhs.first;
        });

        float rand_num = rand_nums[batch_id] * p_threshold;

        // Sampling process
        std::vector<float> cumulative_probs(vocab_size);
        float              total_prob = 0.0f;

        // Calculate cumulative probabilities
        for (int i = 0; i < vocab_size; ++i) {
            float prob = log_prob_index_pairs[i].first;
            total_prob += prob;
            cumulative_probs[i] = total_prob;
        }

        // Normalize cumulative probabilities to be between 0 and 1
        for (float& cp : cumulative_probs) {
            cp /= total_prob;
        }

        int selected_index = -1;

        for (int i = 0; i < vocab_size; ++i) {
            if (cumulative_probs[i] >= rand_num) {
                selected_index = i;
                break;
            }
        }

        if (selected_index != -1) {
            output_ids[batch_id * (step + 1) + step] = log_prob_index_pairs[selected_index].second;
            if (cum_log_probs != nullptr || output_log_probs != nullptr) {
                float lprob = logf(log_prob_index_pairs[selected_index].first);
                if (cum_log_probs != nullptr) {
                    cum_log_probs[batch_id] += lprob;
                }
                if (output_log_probs != nullptr) {
                    output_log_probs[batch_id] = lprob;
                }
            }
        }
    }
}

GreedyOutput CpuDevice::sampleGreedy(const GreedyParams& params) {
    const auto& logits            = params.logits;
    const auto  batch_size        = logits.shape()[0];
    const auto  vocab_size_padded = logits.shape()[1];
    const auto  step              = params.step;

    auto& token_ids = params.token_ids;
    RUNTIME_ASSERT_OP_ARG(batch_size == params.token_ids.shape()[0],
                          "logits.shape[0] should equal to token_ids.shape[0], but %ld vs %ld",
                          batch_size,
                          params.token_ids.shape()[0]);
    RUNTIME_ASSERT_OP_ARG((step == params.token_ids.shape()[1] - 1),
                          "step should equal to token_ids.shape[1] - 1, but %ld vs %ld",
                          step,
                          params.token_ids.shape()[1] - 1);

    // 1. prepare
    auto& top_k        = params.top_k;
    auto& top_p        = params.top_p;
    auto& temperature  = params.temperature;
    auto& cum_log_prob = params.cum_log_probs;

    auto default_top_k = top_k.data<uint32_t>()[0];
    auto default_top_p = top_p.data<float>()[0];

    auto max_top_k = *std::max_element(top_k.data<uint32_t>(), top_k.dataWithOffset<uint32_t>(top_k.size()));
    if (max_top_k == 0) {
        max_top_k = 1;
    }
    auto max_top_p = *std::max_element(top_p.data<float>(), top_p.dataWithOffset<float>(top_p.size()));

    bool* skip_top_k_decode = static_cast<bool*>(aligned_alloc(64, batch_size * sizeof(bool)));
    bool* skip_top_p_decode = static_cast<bool*>(aligned_alloc(64, batch_size * sizeof(bool)));

    uint32_t* runtime_top_k = static_cast<uint32_t*>(aligned_alloc(64, batch_size * sizeof(uint32_t)));
    std::memcpy(runtime_top_k, top_k.data(), batch_size * sizeof(uint32_t));

    float* runtime_top_p = static_cast<float*>(aligned_alloc(64, batch_size * sizeof(float)));
    std::memcpy(runtime_top_p, top_p.data(), batch_size * sizeof(float));

    auto cum_log_probs = cum_log_prob.has_value() ? params.cum_log_probs.value().get().data<float>() : nullptr;
    auto output_log_probs =
        params.output_log_probs.has_value() ? params.output_log_probs.value().get().data<float>() : nullptr;

    // 3.1 setup random seeds
    float* rand_nums = static_cast<float*>(aligned_alloc(64, batch_size * sizeof(float)));

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::default_random_engine            generator;

    for (int i = 0; i < batch_size; i++) {
        if (params.generator[i].defined()) {
            generator.seed(params.generator[i].current_seed());
        } else {
            generator.seed(std::random_device{}());
        }
        rand_nums[i] = distribution(generator);
    }

    // 3.2. compute logits penalty
    if (std::any_of(
            temperature.data<float>(), temperature.data<float>() + batch_size, [&](auto t) { return t != 1.0f; })) {
        applyTemperaturePenalty(
            logits.data<float>(), temperature.data<float>(), batch_size, vocab_size_padded, vocab_size_padded);
    }

    const auto decoder_batch_size = params.sequence_lengths.shape()[0];

    if (decoder_batch_size) {
        if (step > 1 && params.repetition_penalty && decoder_batch_size) {
            auto& repetition_penalty = params.repetition_penalty->get();
            if (std::any_of(repetition_penalty.data<float>(),
                            repetition_penalty.data<float>() + batch_size,
                            [&](auto t) { return t != 1.0f; })) {
                repetitionPenalty(logits.data<float>(),
                                  repetition_penalty.data<float>(),
                                  token_ids.data<int32_t>(),
                                  batch_size,
                                  vocab_size_padded,
                                  params.sequence_lengths.data<int32_t>(),
                                  step + 1,
                                  step);
            }
        }
    }

    // 4. run sampling
    // 4.1 run top_k
    setup_topk(batch_size,
               default_top_k,
               runtime_top_k,
               batch_size,
               default_top_p,
               runtime_top_p,
               batch_size,
               skip_top_k_decode);

    if (std::any_of(skip_top_k_decode, skip_top_k_decode + batch_size, [](auto s) { return !s; })) {
        batchTopKSampling(logits.data<float>(),
                          token_ids.data<int>(),
                          step,
                          cum_log_probs,
                          output_log_probs,
                          max_top_k,
                          runtime_top_k,  // top_ks,
                          vocab_size_padded,
                          batch_size,
                          skip_top_k_decode,
                          rand_nums);
    }

    // 4.2. run top_p
    setup_topp(batch_size,
               default_top_k,
               runtime_top_k,
               batch_size,
               default_top_p,
               runtime_top_p,
               batch_size,
               skip_top_p_decode);

    for (int i = 0; i < batch_size; ++i) {
        computeSoftMax(logits.data<float>() + i * vocab_size_padded, vocab_size_padded);
    }

    batchTopPSampling(token_ids.data<int>(),
                      cum_log_probs,
                      output_log_probs,
                      logits.data<float>(),
                      step,
                      batch_size,
                      vocab_size_padded,
                      max_top_p,
                      runtime_top_p,
                      skip_top_p_decode,
                      rand_nums);
    return GreedyOutput{};
}

}  // namespace rtp_llm
