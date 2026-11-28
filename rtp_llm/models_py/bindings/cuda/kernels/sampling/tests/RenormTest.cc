#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

using namespace rtp_llm;

namespace {

struct RefRenormResult {
    std::vector<float> probs;
    float              cutoff       = 0.0f;
    bool               has_margin   = false;  // True when sum at boundary differs from eff_p by > 5e-5.
    double             boundary_gap = 0.0;
};

RefRenormResult referenceTopKTopPRenormRow(const float* row, int vocab, int top_k, float top_p, bool apply_top_k) {
    RefRenormResult res;
    res.probs.assign(vocab, 0.0f);

    std::vector<float> pos_vals;
    pos_vals.reserve(vocab);
    double total_sum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        const float v = row[i];
        if (v > 0.0f) {
            pos_vals.push_back(v);
            total_sum += static_cast<double>(v);
        }
    }
    if (pos_vals.empty()) {
        res.has_margin = true;
        return res;
    }

    std::sort(pos_vals.begin(), pos_vals.end(), std::greater<float>());
    const int    pos_cnt = static_cast<int>(pos_vals.size());
    const double eff_p =
        (std::fabs(total_sum - 1.0) <= 1e-5) ? static_cast<double>(top_p) : static_cast<double>(top_p) * total_sum;
    const bool no_top_k = !apply_top_k || top_k <= 0 || top_k >= pos_cnt;
    const bool no_top_p = (top_p >= 1.0f) || (eff_p >= total_sum);

    double kept_sum = 0.0;
    if (no_top_k && no_top_p) {
        res.cutoff     = 0.0f;
        res.has_margin = true;
        kept_sum       = total_sum;
    } else {
        float t_k = 0.0f;
        if (!no_top_k) {
            t_k = pos_vals[top_k - 1];
        }
        float  t_p         = 0.0f;
        double running_sum = 0.0;
        double min_gap     = std::numeric_limits<double>::infinity();
        if (!no_top_p) {
            if (eff_p <= 0.0) {
                t_p     = pos_vals[0];
                min_gap = 1.0;
            } else {
                t_p = pos_vals.back();
                for (int i = 0; i < pos_cnt; ++i) {
                    running_sum += static_cast<double>(pos_vals[i]);
                    min_gap = std::min(min_gap, std::fabs(running_sum - eff_p));
                    if (running_sum >= eff_p) {
                        t_p = pos_vals[i];
                        break;
                    }
                }
            }
        } else {
            min_gap = 1.0;
        }
        res.cutoff       = std::max(t_k, t_p);
        res.has_margin   = (min_gap > 5e-5);
        res.boundary_gap = min_gap;
        for (float v : pos_vals) {
            if (v >= res.cutoff) {
                kept_sum += static_cast<double>(v);
            }
        }
    }

    const double denom = (kept_sum <= 1e-8) ? 1.0 : kept_sum;
    for (int i = 0; i < vocab; ++i) {
        if (row[i] > 0.0f && (res.cutoff == 0.0f || row[i] >= res.cutoff)) {
            res.probs[i] = static_cast<float>(static_cast<double>(row[i]) / denom);
        }
    }
    return res;
}

std::vector<float> toCpuFloatVec(const torch::Tensor& t) {
    auto cpu = t.to(torch::kCPU).contiguous();
    return std::vector<float>(cpu.data_ptr<float>(), cpu.data_ptr<float>() + cpu.numel());
}

}  // namespace

TEST(RenormTest, testFusedTopKTopPRenormEdgeCasesAndExactParity) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto stream  = reinterpret_cast<int64_t>(at::cuda::getCurrentCUDAStream().stream());

    // 1. Exact tie inclusion at both t_k and t_p boundaries, sum(top_k) < top_p, sum(top_k) >= top_p,
    //    top_k=1, top_p=0, top_p=1, and all-zero row.
    {
        constexpr int batch     = 9;
        constexpr int vocab     = 16;
        auto          probs_cpu = torch::zeros({batch, vocab}, torch::kFloat32);
        auto          p         = probs_cpu.accessor<float, 2>();

        // Row 0: tie at t_k=2 -> [0.4, 0.2, 0.2, 0.2] must all survive (sum=1.0).
        p[0][0] = 0.4f;
        p[0][1] = 0.2f;
        p[0][2] = 0.2f;
        p[0][3] = 0.2f;
        // Row 1: tie at t_p=0.5 -> [0.3, 0.2, 0.2, 0.2, 0.1] keeps 0.3 and all three 0.2s.
        p[1][0] = 0.3f;
        p[1][1] = 0.2f;
        p[1][2] = 0.2f;
        p[1][3] = 0.2f;
        p[1][4] = 0.1f;
        // Row 2: sum(top_k) < top_p -> top-3 sum=0.60 < top_p=0.95 -> governed by t_k.
        p[2][0] = 0.3f;
        p[2][1] = 0.2f;
        p[2][2] = 0.1f;
        for (int j = 3; j < vocab; ++j) {
            p[2][j] = 0.4f / (vocab - 3);
        }
        // Row 3: sum(top_k) >= top_p -> top-5 sum=0.95 >= top_p=0.5 -> governed by t_p.
        p[3][0] = 0.4f;
        p[3][1] = 0.25f;
        p[3][2] = 0.15f;
        p[3][3] = 0.10f;
        p[3][4] = 0.05f;
        p[3][5] = 0.05f;
        // Row 4: top_k = 1 -> single peak + tie if any.
        p[4][5] = 0.7f;
        p[4][6] = 0.3f;
        // Row 5: top_p = 0.0 -> max elements only (with ties).
        p[5][2] = 0.35f;
        p[5][9] = 0.35f;
        p[5][1] = 0.30f;
        // Row 6: top_p = 1.0, top_k = 0 -> keep all positives cleanly.
        for (int j = 0; j < vocab; ++j) {
            p[6][j] = (j % 2 == 0) ? 0.125f : 0.0f;
        }
        // Row 7: all zeros -> remain all zeros, no NaN/Inf.
        // Row 8: single positive element, huge vocab padding zeros.
        p[8][15] = 1.0f;

        auto top_k = torch::tensor({2, 100, 3, 5, 1, 100, 0, 5, 1}, torch::kInt32).to(torch::kCUDA);
        auto top_p =
            torch::tensor({1.0f, 0.5f, 0.95f, 0.5f, 0.9f, 0.0f, 1.0f, 0.9f, 0.5f}, torch::kFloat32).to(torch::kCUDA);

        auto probs_d      = probs_cpu.to(torch::kCUDA);
        auto out_of_place = torch::empty_like(probs_d);
        rtp_llm::top_k_top_p_renorm_probs(probs_d, out_of_place, top_k, 0, top_p, 1.0, stream);

        // Also test in-place (probs == renorm_probs).
        auto in_place = probs_d.clone();
        rtp_llm::top_k_top_p_renorm_probs(in_place, in_place, top_k, 0, top_p, 1.0, stream);
        check_cuda_error();

        ASSERT_TRUE(torch::equal(out_of_place, in_place));
        auto out_vec = toCpuFloatVec(out_of_place);
        auto in_vec  = toCpuFloatVec(probs_cpu);
        auto k_vec   = top_k.cpu();
        auto p_vec   = top_p.cpu();
        auto k_acc   = k_vec.accessor<int32_t, 1>();
        auto p_acc   = p_vec.accessor<float, 1>();
        for (int r = 0; r < batch; ++r) {
            auto ref = referenceTopKTopPRenormRow(in_vec.data() + r * vocab, vocab, k_acc[r], p_acc[r], true);
            for (int c = 0; c < vocab; ++c) {
                ASSERT_NEAR(out_vec[r * vocab + c], ref.probs[c], 1e-6f) << "row=" << r << " col=" << c;
                if (ref.probs[c] == 0.0f) {
                    EXPECT_EQ(out_vec[r * vocab + c], 0.0f) << "row=" << r << " col=" << c;
                }
            }
        }
    }

    // 2. Non-div4 vocabularies (257, 4093, 129283), unaligned pointer offset, and mixed batch values.
    torch::manual_seed(20261128);
    for (int vocab : {257, 4093, 129283}) {
        for (int offset : {0, 1}) {
            constexpr int batch = 16;
            auto          logits =
                torch::randn({batch, vocab}, options) * 3.0f;  // moderately peaked so gaps around threshold are clear
            auto probs_aligned = torch::softmax(logits, -1);
            auto storage       = torch::empty({batch * vocab + offset}, options);
            auto probs_view    = storage.slice(0, offset).reshape({batch, vocab});
            probs_view.copy_(probs_aligned);
            auto out_storage = torch::empty({batch * vocab + offset}, options);
            auto out_view    = out_storage.slice(0, offset).reshape({batch, vocab});

            auto top_k = torch::tensor({-1, 0, 1, 5, 20, 50, 200, 5000, 1, 20, 0, 10, 30, 100, -1, 25}, torch::kInt32)
                             .to(torch::kCUDA);
            auto top_p = torch::tensor({0.0f,
                                        0.5f,
                                        0.9f,
                                        0.95f,
                                        0.95f,
                                        1.0f,
                                        0.8f,
                                        0.95f,
                                        0.2f,
                                        0.1f,
                                        1.0f,
                                        0.6f,
                                        0.95f,
                                        0.3f,
                                        0.95f,
                                        0.7f},
                                       torch::kFloat32)
                             .to(torch::kCUDA);

            rtp_llm::top_k_top_p_renorm_probs(probs_view, out_view, top_k, 0, top_p, 1.0, stream);
            check_cuda_error();

            auto out_cpu = toCpuFloatVec(out_view);
            auto in_cpu  = toCpuFloatVec(probs_view);
            auto k_cpu   = top_k.cpu();
            auto p_cpu   = top_p.cpu();
            auto k_acc   = k_cpu.accessor<int32_t, 1>();
            auto p_acc   = p_cpu.accessor<float, 1>();

            for (int r = 0; r < batch; ++r) {
                auto   ref     = referenceTopKTopPRenormRow(in_cpu.data() + r * vocab, vocab, k_acc[r], p_acc[r], true);
                double row_sum = 0.0;
                for (int c = 0; c < vocab; ++c) {
                    row_sum += out_cpu[r * vocab + c];
                }
                ASSERT_NEAR(row_sum, 1.0, 2e-5) << "vocab=" << vocab << " row=" << r;
                if (ref.has_margin) {
                    for (int c = 0; c < vocab; ++c) {
                        ASSERT_NEAR(out_cpu[r * vocab + c], ref.probs[c], 1e-5f)
                            << "vocab=" << vocab << " offset=" << offset << " row=" << r << " col=" << c;
                        if (ref.probs[c] == 0.0f) {
                            ASSERT_EQ(out_cpu[r * vocab + c], 0.0f)
                                << "vocab=" << vocab << " row=" << r << " col=" << c;
                        }
                    }
                }
            }
        }
    }
}

TEST(RenormTest, testOverflowRoundsThreePassesAndSubnormals) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto stream  = reinterpret_cast<int64_t>(at::cuda::getCurrentCUDAStream().stream());

    constexpr int vocab = 129280;
    // Uniform distribution -> all 129,280 entries collide into the SAME Round 1, Round 2, and Round 3 bin (> 4096)!
    {
        auto probs = torch::full({6, vocab}, 1.0f / static_cast<float>(vocab), options);
        auto out   = torch::empty_like(probs);
        auto top_k = torch::tensor({0, 50, 6000, 0, 1, 129280}, torch::kInt32).to(torch::kCUDA);
        auto top_p = torch::tensor({0.5f, 0.5f, 1.0f, 0.0f, 0.9f, 1.0f}, torch::kFloat32).to(torch::kCUDA);
        rtp_llm::top_k_top_p_renorm_probs(probs, out, top_k, 0, top_p, 1.0, stream);
        check_cuda_error();

        auto row_sums = out.sum(-1).cpu();
        for (int r = 0; r < 6; ++r) {
            // Because all positive entries are bit-identical, any tie-inclusive cutoff retains all vocab entries!
            EXPECT_NEAR(row_sums[r].item<float>(), 1.0f, 2e-5f) << "row=" << r;
            EXPECT_EQ((out[r] > 0.0f).sum().item<int64_t>(), vocab) << "row=" << r;
        }
    }

    // Tiered distribution inside the SAME Round 1 & Round 2 bucket (bits[30..9] identical, differing only in
    // low mantissa bits[8..0]), targeting b_target < 511 in Round 3 (`scanBuckets512` non-top-bin check).
    {
        constexpr int  batch     = 4;
        auto           probs_cpu = torch::zeros({batch, vocab}, torch::kFloat32);
        auto           p_acc     = probs_cpu.accessor<float, 2>();
        const uint32_t hi_bits   = 0x38000000u;  // 2^-15 (~3.0517578e-5f)
        // Put 5000 elems at mantissa_low=400, 5000 elems at mantissa_low=200, 5000 elems at mantissa_low=50.
        // Sum of 15,000 elements is ~0.45776; normalize so sum == 1.0 (keeping high bits still shared after norm).
        auto bits_to_float = [](uint32_t b) {
            float f;
            std::memcpy(&f, &b, sizeof(f));
            return f;
        };
        const float v_hi  = bits_to_float(hi_bits | 400u);
        const float v_mid = bits_to_float(hi_bits | 200u);
        const float v_lo  = bits_to_float(hi_bits | 50u);
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < 5000; ++j) {
                p_acc[b][j]         = v_hi;
                p_acc[b][5000 + j]  = v_mid;
                p_acc[b][10000 + j] = v_lo;
            }
        }
        auto probs_d = probs_cpu.to(torch::kCUDA);
        auto out_d   = torch::empty_like(probs_d);
        // Row sums are ~0.45776; raw_p * row_sum scaling: top_p=0.50 covers v_hi (1/3) and stops inside v_mid (2/3)!
        // top_k=7000 also stops inside v_mid! v_lo must be dropped, and row sum of kept MUST equal 1.0!
        auto top_k = torch::tensor({7000, 0, 12000, 6000}, torch::kInt32).to(torch::kCUDA);
        auto top_p = torch::tensor({1.0f, 0.50f, 0.50f, 0.80f}, torch::kFloat32).to(torch::kCUDA);
        rtp_llm::top_k_top_p_renorm_probs(probs_d, out_d, top_k, 0, top_p, 1.0, stream);
        check_cuda_error();

        auto out_cpu = toCpuFloatVec(out_d);
        auto in_cpu  = toCpuFloatVec(probs_cpu);
        auto k_cpu   = top_k.cpu();
        auto p_cpu   = top_p.cpu();
        for (int b = 0; b < batch; ++b) {
            auto ref = referenceTopKTopPRenormRow(
                in_cpu.data() + b * vocab, vocab, k_cpu[b].item<int32_t>(), p_cpu[b].item<float>(), true);
            double sum = 0.0;
            for (int c = 0; c < vocab; ++c) {
                sum += out_cpu[b * vocab + c];
                ASSERT_NEAR(out_cpu[b * vocab + c], ref.probs[c], 1e-6f) << "b=" << b << " c=" << c;
            }
            ASSERT_NEAR(sum, 1.0, 2e-5) << "b=" << b;
        }
    }

    // Subnormal values (1e-42) cleanly preserved/normalized without NaN or infinite search.
    {
        auto probs_sub = torch::full({2, 256}, 1e-42f, options);
        auto out_sub   = torch::empty_like(probs_sub);
        rtp_llm::top_k_top_p_renorm_probs(probs_sub, out_sub, std::nullopt, 64, std::nullopt, 0.9, stream);
        check_cuda_error();
        ASSERT_FALSE(torch::isnan(out_sub).any().item<bool>());
        ASSERT_FALSE(torch::isinf(out_sub).any().item<bool>());
    }
}

TEST(RenormTest, testTopPRenormMatchesReferenceLargeVocabAndUnnormalizedInputs) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto stream  = reinterpret_cast<int64_t>(at::cuda::getCurrentCUDAStream().stream());
    torch::manual_seed(20261130);

    constexpr int batch  = 80;
    constexpr int vocab  = 129280;
    auto          logits = torch::randn({batch, vocab}, options) * 2.5f;
    auto          probs  = torch::softmax(logits, -1);
    auto          out    = torch::empty_like(probs);

    auto top_p_cpu = torch::empty({batch}, torch::kFloat32);
    for (int b = 0; b < batch; ++b) {
        if (b % 10 == 0) {
            top_p_cpu[b] = 1.0f;
        } else if (b % 10 == 1) {
            top_p_cpu[b] = 0.0f;
        } else if (b % 2 == 0) {
            top_p_cpu[b] = 0.95f;
        } else {
            top_p_cpu[b] = 0.5f;
        }
    }
    auto top_p = top_p_cpu.to(torch::kCUDA);
    rtp_llm::top_p_renorm_probs(probs, out, top_p, 1.0, stream);
    check_cuda_error();

    auto probs_cpu = toCpuFloatVec(probs);
    auto out_cpu   = toCpuFloatVec(out);
    auto p_acc     = top_p_cpu.accessor<float, 1>();
    for (int b = 0; b < batch; ++b) {
        auto   ref     = referenceTopKTopPRenormRow(probs_cpu.data() + b * vocab, vocab, 0, p_acc[b], false);
        double row_sum = 0.0;
        for (int c = 0; c < vocab; ++c) {
            row_sum += out_cpu[b * vocab + c];
        }
        ASSERT_NEAR(row_sum, 1.0, 3e-5) << "row=" << b;
        if (ref.has_margin) {
            for (int c = 0; c < vocab; ++c) {
                ASSERT_NEAR(out_cpu[b * vocab + c], ref.probs[c], 1e-5f) << "row=" << b << " col=" << c;
            }
        }
    }

    // Also verify unnormalized input row sums (3.5 and 0.02).
    auto unnorm = probs.slice(0, 0, 4).clone();
    unnorm[0].mul_(3.5f);
    unnorm[1].mul_(0.02f);
    auto out_unnorm = torch::empty_like(unnorm);
    rtp_llm::top_p_renorm_probs(unnorm, out_unnorm, std::nullopt, 0.85, stream);
    check_cuda_error();
    auto sums = out_unnorm.sum(-1).cpu();
    for (int b = 0; b < 4; ++b) {
        ASSERT_NEAR(sums[b].item<float>(), 1.0f, 3e-5f);
    }
}

TEST(RenormTest, testStatisticalDistributionParityWithFixedSeeds) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto stream  = reinterpret_cast<int64_t>(at::cuda::getCurrentCUDAStream().stream());

    constexpr int   vocab      = 64;
    constexpr int   num_trials = 24000;
    constexpr int   top_k      = 12;
    constexpr float top_p      = 0.85f;

    torch::manual_seed(42);
    auto base_logits = torch::randn({vocab}, options) * 1.4f;
    auto base_prob   = torch::softmax(base_logits, -1).unsqueeze(0).contiguous();
    auto expected_p  = torch::empty_like(base_prob);
    rtp_llm::top_k_top_p_renorm_probs(base_prob, expected_p, std::nullopt, top_k, std::nullopt, top_p, stream);
    check_cuda_error();

    auto base_cpu     = toCpuFloatVec(base_prob);
    auto expected_cpu = toCpuFloatVec(expected_p);
    auto ref          = referenceTopKTopPRenormRow(base_cpu.data(), vocab, top_k, top_p, true);
    ASSERT_TRUE(ref.has_margin);
    for (int c = 0; c < vocab; ++c) {
        ASSERT_NEAR(expected_cpu[c], ref.probs[c], 1e-6f);
    }

    auto probs_batch = base_prob.expand({num_trials, vocab}).contiguous();
    auto out_tokens  = torch::empty({num_trials}, options.dtype(torch::kInt32));
    auto valid       = torch::empty({num_trials}, options.dtype(torch::kBool));
    auto seeds = torch::full({num_trials}, 1234567LL, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto offsets =
        torch::arange(0, num_trials * 32LL, 32LL, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    rtp_llm::top_k_top_p_sampling_from_probs(probs_batch,
                                             out_tokens,
                                             valid,
                                             std::nullopt,
                                             std::nullopt,
                                             top_k,
                                             std::nullopt,
                                             top_p,
                                             true,
                                             seeds,
                                             0,
                                             offsets,
                                             0,
                                             stream);
    check_cuda_error();
    ASSERT_TRUE(valid.cpu().all().item<bool>());

    auto             sampled_cpu = out_tokens.cpu();
    auto             s_acc       = sampled_cpu.accessor<int32_t, 1>();
    std::vector<int> counts(vocab, 0);
    for (int i = 0; i < num_trials; ++i) {
        const int tok = s_acc[i];
        ASSERT_GE(tok, 0);
        ASSERT_LT(tok, vocab);
        ASSERT_GT(ref.probs[tok], 0.0f) << "sampled excluded token " << tok;
        counts[tok]++;
    }

    double chi2 = 0.0;
    int    df   = -1;
    double tvd  = 0.0;
    for (int c = 0; c < vocab; ++c) {
        const double empirical = static_cast<double>(counts[c]) / num_trials;
        tvd += 0.5 * std::fabs(empirical - ref.probs[c]);
        if (ref.probs[c] * num_trials >= 5.0) {
            const double exp_cnt = ref.probs[c] * num_trials;
            const double diff    = counts[c] - exp_cnt;
            chi2 += (diff * diff) / exp_cnt;
            df++;
        }
    }
    EXPECT_LT(tvd, 0.015);
    if (df > 0) {
        // Mean chi2 = df, std = sqrt(2*df); 5-sigma guard catches biased truncation immediately.
        EXPECT_LT(chi2, df + 5.0 * std::sqrt(2.0 * df));
    }
}

TEST(RenormTest, testMultiCTAMultiStreamAndTopKOnlyParity) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::manual_seed(20261201);

    auto s1 = at::cuda::getStreamFromPool();
    auto s2 = at::cuda::getStreamFromPool();

    for (int vocab : {32768, 129280, 129283}) {
        for (int batch : {1, 24, 80, 96}) {
            if (vocab != 129280 && batch != 24) {
                continue;
            }
            // Moderate spread (0.5f) exercises Round 2 Multi-CTA overflow buckets on large vocabs.
            auto logits = torch::randn({batch, vocab}, options) * 0.5f;
            auto probs  = torch::softmax(logits, -1);

            auto top_k_cpu = torch::empty({batch}, torch::kInt32);
            auto top_p_cpu = torch::empty({batch}, torch::kFloat32);
            for (int r = 0; r < batch; ++r) {
                top_k_cpu[r] = (r % 5 == 0) ? 0 : ((r % 3 == 0) ? 1 : (10 + (r % 40)));
                top_p_cpu[r] = (r % 4 == 0) ? 1.0f : ((r % 4 == 1) ? 0.95f : 0.7f);
            }
            auto top_k_d = top_k_cpu.to(torch::kCUDA);
            auto top_p_d = top_p_cpu.to(torch::kCUDA);

            auto out_kp_1   = torch::empty_like(probs);
            auto out_kp_2   = torch::empty_like(probs);
            auto out_topk_1 = torch::empty_like(probs);
            auto out_topk_2 = torch::empty_like(probs);

            {
                c10::cuda::CUDAStreamGuard g(s1);
                const int64_t              st1 = reinterpret_cast<int64_t>(s1.stream());
                rtp_llm::top_k_top_p_renorm_probs(probs, out_kp_1, top_k_d, 0, top_p_d, 1.0, st1);
                rtp_llm::top_k_renorm_probs(probs, out_topk_1, top_k_d, 0, st1);
            }
            {
                c10::cuda::CUDAStreamGuard g(s2);
                const int64_t              st2 = reinterpret_cast<int64_t>(s2.stream());
                rtp_llm::top_k_top_p_renorm_probs(probs, out_kp_2, top_k_d, 0, top_p_d, 1.0, st2);
                rtp_llm::top_k_renorm_probs(probs, out_topk_2, top_k_d, 0, st2);
            }
            s1.synchronize();
            s2.synchronize();
            check_cuda_error();

            // Multi-CTA determinism across independent streams and repeated runs.
            ASSERT_TRUE(torch::equal(out_kp_1, out_kp_2)) << "vocab=" << vocab << " batch=" << batch;
            ASSERT_TRUE(torch::equal(out_topk_1, out_topk_2)) << "vocab=" << vocab << " batch=" << batch;

            if (vocab == 129280 && batch == 80) {
                auto probs_cpu    = toCpuFloatVec(probs);
                auto kp_cpu       = toCpuFloatVec(out_kp_1);
                auto topk_cpu_out = toCpuFloatVec(out_topk_1);
                auto k_acc        = top_k_cpu.accessor<int32_t, 1>();
                auto p_acc        = top_p_cpu.accessor<float, 1>();
                for (int r = 0; r < batch; ++r) {
                    auto ref_kp =
                        referenceTopKTopPRenormRow(probs_cpu.data() + r * vocab, vocab, k_acc[r], p_acc[r], true);
                    auto ref_k  = referenceTopKTopPRenormRow(probs_cpu.data() + r * vocab, vocab, k_acc[r], 1.0f, true);
                    double s_kp = 0.0, s_k = 0.0;
                    for (int c = 0; c < vocab; ++c) {
                        s_kp += kp_cpu[r * vocab + c];
                        s_k += topk_cpu_out[r * vocab + c];
                        if (ref_kp.has_margin) {
                            ASSERT_NEAR(kp_cpu[r * vocab + c], ref_kp.probs[c], 1e-5f);
                        }
                        if (ref_k.has_margin) {
                            ASSERT_NEAR(topk_cpu_out[r * vocab + c], ref_k.probs[c], 1e-5f);
                        }
                    }
                    ASSERT_NEAR(s_kp, 1.0, 3e-5);
                    ASSERT_NEAR(s_k, 1.0, 3e-5);
                }
            }
        }
    }
}

TEST(RenormTest, testLargeVocabSamplingDistributionParityWithQuadSearch) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto stream  = reinterpret_cast<int64_t>(at::cuda::getCurrentCUDAStream().stream());

    constexpr int   vocab      = 4096;
    constexpr int   num_trials = 40960;
    constexpr int   top_k      = 8;
    constexpr float top_p      = 0.85f;

    torch::manual_seed(20261202);
    auto base_logits = torch::randn({vocab}, options) * 3.5f;
    auto base_prob   = torch::softmax(base_logits, -1).unsqueeze(0).contiguous();
    auto base_cpu    = toCpuFloatVec(base_prob);

    auto probs_batch = base_prob.expand({num_trials, vocab}).contiguous();
    auto out_tokens  = torch::empty({num_trials}, options.dtype(torch::kInt32));
    auto valid       = torch::empty({num_trials}, options.dtype(torch::kBool));
    auto seeds = torch::full({num_trials}, 9876543LL, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto offsets =
        torch::arange(0, num_trials * 32LL, 32LL, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    for (bool joint : {false, true}) {
        auto ref = referenceTopKTopPRenormRow(base_cpu.data(), vocab, joint ? top_k : 0, top_p, joint);
        ASSERT_TRUE(ref.has_margin);

        if (joint) {
            rtp_llm::top_k_top_p_sampling_from_probs(probs_batch,
                                                     out_tokens,
                                                     valid,
                                                     std::nullopt,
                                                     std::nullopt,
                                                     top_k,
                                                     std::nullopt,
                                                     top_p,
                                                     true,
                                                     seeds,
                                                     0,
                                                     offsets,
                                                     0,
                                                     stream);
        } else {
            rtp_llm::top_p_sampling_from_probs(
                probs_batch, out_tokens, valid, std::nullopt, std::nullopt, top_p, true, seeds, 0, offsets, 0, stream);
        }
        check_cuda_error();
        ASSERT_TRUE(valid.cpu().all().item<bool>());

        auto             sampled_cpu = out_tokens.cpu();
        auto             s_acc       = sampled_cpu.accessor<int32_t, 1>();
        std::vector<int> counts(vocab, 0);
        for (int i = 0; i < num_trials; ++i) {
            const int tok = s_acc[i];
            ASSERT_GE(tok, 0);
            ASSERT_LT(tok, vocab);
            ASSERT_GT(ref.probs[tok], 0.0f) << "joint=" << joint << " sampled excluded token " << tok;
            counts[tok]++;
        }

        double chi2 = 0.0;
        int    df   = -1;
        double tvd  = 0.0;
        for (int c = 0; c < vocab; ++c) {
            const double empirical = static_cast<double>(counts[c]) / num_trials;
            tvd += 0.5 * std::fabs(empirical - ref.probs[c]);
            if (ref.probs[c] * num_trials >= 5.0) {
                const double exp_cnt = ref.probs[c] * num_trials;
                const double diff    = counts[c] - exp_cnt;
                chi2 += (diff * diff) / exp_cnt;
                df++;
            }
        }
        EXPECT_LT(tvd, 0.015) << "joint=" << joint;
        if (df > 0) {
            EXPECT_LT(chi2, df + 5.0 * std::sqrt(2.0 * df)) << "joint=" << joint;
        }
    }
}

TEST(RenormTest, testMultiCTASamplingAndFusedRenormParity) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto i32_opt = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto i64_opt = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto b_opt   = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
    auto stream  = reinterpret_cast<int64_t>(at::cuda::getCurrentCUDAStream().stream());

    torch::manual_seed(20261203);

    // 1. Bit-exact parity between fused renorm_and_sample and separate sampling_from_probs + renorm_probs,
    //    plus sampling_from_probs vs top_p_sampling(1.0), invalid parameter handling, and all-zero rows.
    for (int vocab : {32768, 129280, 129283}) {
        constexpr int batch  = 32;
        auto          logits = torch::randn({batch, vocab}, options) * 1.5f;
        auto          probs  = torch::softmax(logits, -1);
        probs[3].zero_();  // all-zero row -> valid=false, renorm=0

        auto top_k_cpu = torch::empty({batch}, torch::kInt32);
        auto top_p_cpu = torch::empty({batch}, torch::kFloat32);
        for (int r = 0; r < batch; ++r) {
            top_k_cpu[r] = (r == 5) ? 0 : ((r % 4 == 0) ? 1 : (10 + r * 2));
            top_p_cpu[r] = (r == 7) ? -0.5f : ((r % 3 == 0) ? 1.0f : 0.9f);
        }
        auto top_k_d = top_k_cpu.to(torch::kCUDA);
        auto top_p_d = top_p_cpu.to(torch::kCUDA);
        auto seeds   = torch::full({batch}, 424242LL, i64_opt);
        auto offsets = torch::arange(0, batch * 32LL, 32LL, i64_opt);

        // 1a. sampling_from_probs vs top_p_sampling(1.0) vs top_p_renorm_and_sample(1.0)
        auto out_pure = torch::empty({batch}, i32_opt);
        auto val_pure = torch::empty({batch}, b_opt);
        auto out_tp1  = torch::empty({batch}, i32_opt);
        auto val_tp1  = torch::empty({batch}, b_opt);
        auto out_rns1 = torch::empty({batch}, i32_opt);
        auto val_rns1 = torch::empty({batch}, b_opt);
        auto rn_tp1   = torch::empty_like(probs);
        auto rn_rns1  = torch::empty_like(probs);

        rtp_llm::sampling_from_probs(probs, out_pure, val_pure, std::nullopt, true, seeds, 0, offsets, 0, stream);
        rtp_llm::top_p_sampling_from_probs(
            probs, out_tp1, val_tp1, std::nullopt, std::nullopt, 1.0, true, seeds, 0, offsets, 0, stream);
        rtp_llm::top_p_renorm_probs(probs, rn_tp1, std::nullopt, 1.0, stream);
        rtp_llm::top_p_renorm_and_sample_probs(
            probs, rn_rns1, out_rns1, val_rns1, std::nullopt, std::nullopt, 1.0, true, seeds, 0, offsets, 0, stream);
        check_cuda_error();

        ASSERT_TRUE(torch::equal(out_pure, out_tp1));
        ASSERT_TRUE(torch::equal(val_pure, val_tp1));
        ASSERT_TRUE(torch::equal(out_pure, out_rns1));
        ASSERT_TRUE(torch::equal(val_pure, val_rns1));
        ASSERT_TRUE(torch::equal(rn_tp1, rn_rns1));
        ASSERT_FALSE(val_pure[3].cpu().item<bool>());

        // 1b. Joint top_k_top_p_renorm_and_sample_probs vs separate sampling + renorm (valid rows + invalid rows)
        auto out_kp_sep = torch::empty({batch}, i32_opt);
        auto val_kp_sep = torch::empty({batch}, b_opt);
        auto rn_kp_sep  = torch::empty_like(probs);
        auto out_kp_fus = torch::empty({batch}, i32_opt);
        auto val_kp_fus = torch::empty({batch}, b_opt);
        auto rn_kp_fus  = torch::empty_like(probs);

        rtp_llm::top_k_top_p_sampling_from_probs(
            probs, out_kp_sep, val_kp_sep, std::nullopt, top_k_d, 0, top_p_d, 1.0, true, seeds, 0, offsets, 0, stream);
        rtp_llm::top_k_top_p_renorm_probs(probs, rn_kp_sep, top_k_d, 0, top_p_d, 1.0, stream);
        rtp_llm::top_k_top_p_renorm_and_sample_probs(probs,
                                                     rn_kp_fus,
                                                     out_kp_fus,
                                                     val_kp_fus,
                                                     std::nullopt,
                                                     top_k_d,
                                                     0,
                                                     top_p_d,
                                                     1.0,
                                                     true,
                                                     seeds,
                                                     0,
                                                     offsets,
                                                     0,
                                                     stream);
        check_cuda_error();

        ASSERT_TRUE(torch::equal(out_kp_sep, out_kp_fus));
        ASSERT_TRUE(torch::equal(val_kp_sep, val_kp_fus));
        ASSERT_FALSE(val_kp_fus[3].cpu().item<bool>());
        ASSERT_FALSE(val_kp_fus[5].cpu().item<bool>());
        ASSERT_FALSE(val_kp_fus[7].cpu().item<bool>());
        auto val_cpu = val_kp_fus.cpu();
        for (int r = 0; r < batch; ++r) {
            if (val_cpu[r].item<bool>()) {
                ASSERT_TRUE(torch::equal(rn_kp_fus[r], rn_kp_sep[r])) << "vocab=" << vocab << " row=" << r;
            } else {
                ASSERT_EQ(rn_kp_fus[r].count_nonzero().item<int64_t>(), 0);
            }
        }

        // 1c. top_k_renorm_and_sample_probs vs separate sampling + renorm
        auto out_k_sep = torch::empty({batch}, i32_opt);
        auto val_k_sep = torch::empty({batch}, b_opt);
        auto rn_k_sep  = torch::empty_like(probs);
        auto out_k_fus = torch::empty({batch}, i32_opt);
        auto val_k_fus = torch::empty({batch}, b_opt);
        auto rn_k_fus  = torch::empty_like(probs);
        rtp_llm::top_k_sampling_from_probs(
            probs, out_k_sep, val_k_sep, std::nullopt, top_k_d, 0, true, seeds, 0, offsets, 0, stream);
        rtp_llm::top_k_renorm_probs(probs, rn_k_sep, top_k_d, 0, stream);
        rtp_llm::top_k_renorm_and_sample_probs(
            probs, rn_k_fus, out_k_fus, val_k_fus, std::nullopt, top_k_d, 0, true, seeds, 0, offsets, 0, stream);
        check_cuda_error();

        ASSERT_TRUE(torch::equal(out_k_sep, out_k_fus));
        ASSERT_TRUE(torch::equal(val_k_sep, val_k_fus));
        for (int r = 0; r < batch; ++r) {
            if (r != 3 && r != 5) {
                ASSERT_TRUE(torch::equal(rn_k_fus[r], rn_k_sep[r])) << "vocab=" << vocab << " row=" << r;
            }
        }
    }

    // 2. Statistical distribution parity (TVD < 0.015 & 5-sigma chi2) on Multi-CTA path (V=32768, 129280).
    for (int vocab : {32768, 129280}) {
        constexpr int   batch_step = 64;
        constexpr int   rounds     = 400;
        constexpr int   num_trials = batch_step * rounds;
        constexpr int   top_k      = 24;
        constexpr float top_p      = 0.85f;

        // Place ~36 boosted active tokens strided across [0, vocab) well above the -12 noise floor
        // so cutoff gaps far exceed 5e-5 while still exercising Multi-CTA warp tiles across chunks.
        auto      base_logits = torch::randn({vocab}, options) * 0.8f - 12.0f;
        const int stride      = vocab / 36;
        auto      hot_indices = torch::arange(0, 36 * stride, stride, torch::kLong).to(torch::kCUDA);
        auto      hot_boost   = torch::linspace(18.0f, 13.0f, 36, options);
        base_logits.index_add_(0, hot_indices, hot_boost);
        auto base_prob = torch::softmax(base_logits, -1).unsqueeze(0).contiguous();
        auto base_cpu  = toCpuFloatVec(base_prob);
        auto ref       = referenceTopKTopPRenormRow(base_cpu.data(), vocab, top_k, top_p, true);
        ASSERT_TRUE(ref.has_margin) << "vocab=" << vocab << " boundary_gap=" << ref.boundary_gap;

        auto probs_chunk = base_prob.expand({batch_step, vocab}).contiguous();
        auto rn_chunk    = torch::empty_like(probs_chunk);
        auto out_tokens  = torch::empty({batch_step}, i32_opt);
        auto valid       = torch::empty({batch_step}, b_opt);
        auto seeds       = torch::full({batch_step}, 556677LL, i64_opt);

        std::vector<int> counts(vocab, 0);
        for (int r = 0; r < rounds; ++r) {
            auto offsets = torch::arange(r * batch_step * 32LL, (r + 1) * batch_step * 32LL, 32LL, i64_opt);
            rtp_llm::top_k_top_p_renorm_and_sample_probs(probs_chunk,
                                                         rn_chunk,
                                                         out_tokens,
                                                         valid,
                                                         std::nullopt,
                                                         std::nullopt,
                                                         top_k,
                                                         std::nullopt,
                                                         top_p,
                                                         true,
                                                         seeds,
                                                         0,
                                                         offsets,
                                                         0,
                                                         stream);
            ASSERT_TRUE(valid.cpu().all().item<bool>());
            auto tok_cpu = out_tokens.cpu();
            auto acc     = tok_cpu.accessor<int32_t, 1>();
            for (int i = 0; i < batch_step; ++i) {
                const int tok = acc[i];
                ASSERT_GE(tok, 0);
                ASSERT_LT(tok, vocab);
                ASSERT_GT(ref.probs[tok], 0.0f) << "vocab=" << vocab << " sampled excluded token " << tok;
                counts[tok]++;
            }
        }
        check_cuda_error();

        double chi2 = 0.0;
        int    df   = -1;
        double tvd  = 0.0;
        for (int c = 0; c < vocab; ++c) {
            const double empirical = static_cast<double>(counts[c]) / num_trials;
            tvd += 0.5 * std::fabs(empirical - ref.probs[c]);
            if (ref.probs[c] * num_trials >= 5.0) {
                const double exp_cnt = ref.probs[c] * num_trials;
                const double diff    = counts[c] - exp_cnt;
                chi2 += (diff * diff) / exp_cnt;
                df++;
            }
        }
        EXPECT_LT(tvd, 0.015) << "vocab=" << vocab;
        if (df > 0) {
            EXPECT_LT(chi2, df + 5.0 * std::sqrt(2.0 * df)) << "vocab=" << vocab;
        }
    }
}
