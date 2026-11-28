#include "rtp_llm/cpp/models/SamplingRows.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/common/kernels/dspark_packed_sampling.h"
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>

namespace rtp_llm {
namespace {
constexpr int              B = 4, G = 3, D = B * (G + 1), V = 1024, H = 16;
std::vector<at::Generator> generators() {
    std::vector<at::Generator> result;
    auto                       shared = at::cuda::detail::createCUDAGenerator();
    shared.set_current_seed(789);
    for (int b = 0; b < B; ++b) {
        auto gen = b >= 2 ? shared : at::cuda::detail::createCUDAGenerator();
        if (b < 2)
            gen.set_current_seed(123 + b);
        for (int i = 0; i <= G; ++i)
            result.push_back(gen);
    }
    return result;
}
SamplerInputs makeInput(int mode, const std::vector<at::Generator>& gens, const torch::Tensor& logits) {
    SamplerInputs s;
    s.batch_size = s.batch_size_out = D;
    s.step                          = H - 1;
    s.vocab_size                    = V;
    s.logits                        = logits;
    s.generator                     = gens;
    s.top_k                         = torch::zeros({D}, torch::kInt32);
    s.top_p                         = torch::full({D}, .9f);
    s.temperature                   = torch::full({D}, .7f);
    s.input_lengths                 = torch::full({D}, 4, torch::kInt32);
    s.sequence_lengths              = torch::empty({D}, torch::kInt32);
    s.num_beams_in                  = torch::ones({D}, torch::kInt64);
    s.num_beams_out                 = torch::ones({D}, torch::kInt64);
    s.do_sample                     = torch::ones({D}, torch::kBool);
    s.skip_logits_probs_restore     = true;
    s.return_logprobs               = mode != 0;
    s.force_topk                    = mode == 1 || mode == 2 || mode == 3 ? 128 : 0;
    s.probability_output = mode == 12 ? SamplerProbabilityOutput::DENSE : SamplerProbabilityOutput::DISTRIBUTION;
    if (mode >= 13) {
        s.probability_output = SamplerProbabilityOutput::NONE;
        s.return_logprobs    = mode == 15;
    }
    if (mode == 0 || mode == 11 || mode == 13)
        s.top_k.fill_(1);
    if (mode == 12)
        s.force_topk = 128;
    if (mode == 4)
        s.top_p.fill_(1);
    if (mode == 5)
        s.top_p.fill_(0);  // normalized unrestricted sentinel
    if (mode == 6 || mode == 7) {
        for (int r = 0; r < D; ++r)
            s.top_k.data_ptr<int32_t>()[r] = mode == 6 ? 8 + r / (G + 1) : (r < 4 ? 0 : 32);
    }
    if (mode == 8)
        s.cum_log_probs = torch::arange(D, torch::kFloat32) * .01f;
    if (mode == 9)
        s.return_original_all_probs = true;
    if (mode == 10) {
        s.do_sample.narrow(0, 4, 4).fill_(false);
        s.top_k.narrow(0, 4, 4).fill_(1);
    }
    if (mode == 2) {
        s.repetition_penalty = torch::full({D}, 1.2f);
        s.presence_penalty   = torch::full({D}, .2f);
        s.frequency_penalty  = torch::full({D}, .1f);
    }
    if (mode == 3)
        s.no_repeat_ngram_size = torch::full({D}, 2, torch::kInt32);
    for (int r = 0; r < D; ++r)
        s.sequence_lengths.data_ptr<int32_t>()[r] = 4 + r / (G + 1) + r % (G + 1);
    return s;
}
}  // namespace

TEST(MtpPackedSamplingTest, MatchesDenseSamplingAndRngAcrossThreeRounds) {
    const auto gpu = torch::TensorOptions().device(torch::kCUDA);
    auto draft     = torch::tensor({{7, 8, 9}, {10, 7, 12}, {13, 14, 7}, {7, 16, 17}}, torch::kInt32).to(torch::kCUDA);
    auto dense_history = torch::zeros({D, H}, torch::kInt32);
    for (int r = 0; r < D; ++r) {
        const int b = r / (G + 1), base = 4 + b;
        dense_history[r].narrow(0, 0, base).fill_(7);
        dense_history[r].narrow(0, base, G).copy_(draft[b].cpu());
    }
    for (int mode = 0; mode <= 15; ++mode) {
        SCOPED_TRACE(mode);
        auto    dg = generators(), pg = generators();
        Sampler ds({D}), ps({D});
        for (int round = 0; round < 3; ++round) {
            SCOPED_TRACE(round);
            std::vector<int64_t> map_values;
            for (int b = 0; b < B; ++b)
                for (int j = 0; j <= (b + round) % 4; ++j)
                    map_values.push_back(b * (G + 1) + j);
            auto map        = torch::tensor(map_values, torch::kInt64).to(torch::kCUDA);
            auto logits     = torch::randn({D, V}, gpu);
            auto dense      = makeInput(mode, dg, logits.clone());
            dense.token_ids = dense_history.to(torch::kCUDA);
            auto packed     = makeInput(mode, pg, logits.index_select(0, map));
            auto inverse    = torch::full({D}, -1, gpu.dtype(torch::kInt32));
            inverse.index_copy_(0, map, torch::arange(map.numel(), inverse.options()));
            packed.row_mapping  = SamplingRowMapping{map, inverse};
            packed.token_ids    = mode == 2 || mode == 3 ? dense_history.to(torch::kCUDA) : torch::Tensor();
            const auto expected = ds.forward(dense);
            const auto actual   = ps.forward(packed);
            EXPECT_EQ(packed.batch_size, D);
            EXPECT_TRUE(packed.top_k.is_cpu());
            EXPECT_EQ(packed.sequence_lengths.numel(), D);
            EXPECT_EQ(actual.all_probs.defined(), expected.all_probs.defined());
            auto selected_tokens = actual.token_ids.index_select(0, map).select(1, 0);
            EXPECT_TRUE(torch::equal(selected_tokens, expected.token_ids.index_select(0, map).select(1, H - 1)));
            EXPECT_EQ(actual.token_ids_are_point_mass, expected.token_ids_are_point_mass);
            if (actual.all_probs.defined() && !actual.token_ids_are_point_mass) {
                EXPECT_TRUE(torch::allclose(
                    actual.all_probs.index_select(0, map), expected.all_probs.index_select(0, map), 1e-6, 1e-7));
                auto missing = inverse.lt(0).nonzero().squeeze(1);
                EXPECT_EQ(actual.all_probs.index_select(0, missing).count_nonzero().item<int64_t>(), 0);
            }
            if (actual.all_probs_indices.defined()) {
                EXPECT_TRUE(torch::equal(actual.all_probs_indices.index_select(0, map),
                                         expected.all_probs_indices.index_select(0, map)));
            }
            if (actual.sampling_probs.defined()) {
                EXPECT_TRUE(torch::allclose(actual.sampling_probs.index_select(0, map),
                                            expected.sampling_probs.index_select(0, map),
                                            1e-6,
                                            1e-7));
            }
            if (actual.cum_log_probs.defined()) {
                EXPECT_TRUE(torch::allclose(actual.cum_log_probs.index_select(0, map),
                                            expected.cum_log_probs.to(torch::kCUDA).index_select(0, map),
                                            1e-6,
                                            1e-7));
            }
            EXPECT_TRUE(torch::equal(actual.success.index_select(0, map), expected.success.index_select(0, map)));
            for (int r = 0; r < D; ++r)
                EXPECT_EQ(dg[r].get_offset(), pg[r].get_offset());
        }
    }
}

TEST(MtpPackedSamplingTest, DefaultGeneratorReservationMatchesDense) {
    auto       gen    = at::cuda::detail::getDefaultCUDAGenerator();
    const auto saved  = gen.get_state();
    auto       logits = torch::randn({D, V}, torch::TensorOptions().device(torch::kCUDA));
    auto       map    = torch::tensor({0, 1, 4, 8, 9, 10, 12}, torch::kInt64).to(torch::kCUDA);
    Sampler    sampler({D});
    for (int round = 0; round < 3; ++round) {
        const auto before   = gen.get_state();
        auto       dense    = makeInput(1, std::vector<at::Generator>(D), logits.clone());
        dense.token_ids     = torch::empty({D, H}, logits.options().dtype(torch::kInt32));
        auto       expected = sampler.forward(dense);
        const auto after    = gen.get_state();
        gen.set_state(before);
        auto packed  = makeInput(1, std::vector<at::Generator>(D), logits.index_select(0, map));
        auto inverse = torch::full({D}, -1, logits.options().dtype(torch::kInt32));
        inverse.index_copy_(0, map, torch::arange(map.numel(), inverse.options()));
        packed.row_mapping = SamplingRowMapping{map, inverse};
        auto actual        = sampler.forward(packed);
        EXPECT_TRUE(torch::equal(gen.get_state(), after));
        EXPECT_TRUE(torch::equal(actual.token_ids.index_select(0, map).select(1, 0),
                                 expected.token_ids.index_select(0, map).select(1, H - 1)));
    }
    gen.set_state(saved);
}

TEST(MtpPackedSamplingTest, MaskRemapsDenseRowsAndSkipsOmittedRows) {
    auto gpu     = torch::TensorOptions().device(torch::kCUDA);
    auto inverse = torch::tensor({0, -1, 1, -1, 2, -1}, torch::kInt32).to(torch::kCUDA);
    auto mask    = torch::tensor({1, 2, 4, 8, 16, 32}, torch::kInt32).reshape({6, 1}).to(torch::kCUDA);
    auto dense   = torch::ones({6, 32}, gpu);
    auto packed  = torch::ones({3, 32}, gpu);
    runtimeApplyPackedMaskLogits(dense, mask, 32);
    runtimeApplyPackedMaskLogits(packed, mask, {}, 32, inverse);
    auto map = torch::tensor({0, 2, 4}, torch::kInt64).to(torch::kCUDA);
    EXPECT_TRUE(torch::equal(packed, dense.index_select(0, map)));
    auto indices = torch::tensor({5, 2, 0}, torch::kInt32).to(torch::kCUDA);
    mask         = mask.narrow(0, 0, 3);
    dense.fill_(1);
    packed.fill_(1);
    runtimeApplyPackedMaskLogits(dense, mask, indices, 32);
    runtimeApplyPackedMaskLogits(packed, mask, indices, 32, inverse);
    EXPECT_TRUE(torch::equal(packed, dense.index_select(0, map)));
}

TEST(MtpPackedSamplingTest, ScatterHandlesUnalignedWidthsAndDistinctDistributions) {
    auto          gpu = torch::TensorOptions().device(torch::kCUDA);
    SamplerOutput packed;
    packed.token_ids         = torch::tensor({{1, 2, 3}, {4, 5, 6}}, torch::kInt32).to(torch::kCUDA);
    packed.all_probs         = torch::rand({2, 7}, gpu);
    packed.all_probs_indices = torch::arange(14, gpu.dtype(torch::kInt32)).reshape({2, 7});
    packed.sampling_probs    = torch::rand({2, 9}, gpu);
    packed.cum_log_probs     = torch::tensor({.1f, .2f}).to(torch::kCUDA);
    packed.success           = torch::tensor({true, false}, torch::kBool).to(torch::kCUDA);
    auto inverse             = torch::tensor({0, -1, 1}, torch::kInt32).to(torch::kCUDA);
    auto actual              = scatterSamplingRows(packed, inverse);
    auto map                 = torch::tensor({0, 2}, torch::kInt64).to(torch::kCUDA);
    EXPECT_TRUE(torch::equal(actual.all_probs.index_select(0, map), packed.all_probs));
    EXPECT_TRUE(torch::equal(actual.sampling_probs.index_select(0, map), packed.sampling_probs));
    EXPECT_TRUE(torch::equal(actual.all_probs_indices.index_select(0, map), packed.all_probs_indices));
    EXPECT_TRUE(torch::equal(actual.cum_log_probs.index_select(0, map), packed.cum_log_probs));
    EXPECT_TRUE(torch::equal(actual.success.index_select(0, map), packed.success));
    EXPECT_EQ(actual.all_probs[1].count_nonzero().item<int64_t>(), 0);
    EXPECT_EQ(actual.sampling_probs[1].count_nonzero().item<int64_t>(), 0);
    EXPECT_TRUE(actual.all_probs_indices[1].eq(-1).all().item<bool>());
    EXPECT_EQ(actual.token_ids[1][0].item<int32_t>(), 0);
    EXPECT_FALSE(actual.success[1].item<bool>());
}

// Opt-in microbenchmark: production kernels, preallocated buffers, CUDA graphs
// to remove CPU launch gaps. Enable explicitly with --gtest_also_run_disabled_tests.
TEST(MtpPackedSamplingTest, DISABLED_KernelTiming) {
    const auto                gpu    = torch::TensorOptions().device(torch::kCUDA);
    const auto                stream = at::cuda::getStreamFromPool();
    at::cuda::CUDAStreamGuard guard(stream);
    auto                      time_us = [&](const std::function<void()>& fn) {
        for (int i = 0; i < 10; ++i)
            fn();
        stream.synchronize();
        at::cuda::CUDAGraph graph(/*keep_graph=*/true);
        graph.capture_begin();
        for (int i = 0; i < 100; ++i)
            fn();
        graph.capture_end();
        size_t node_count = 0;
        cudaGraphGetNodes(graph.raw_cuda_graph(), nullptr, &node_count);
        std::vector<cudaGraphNode_t> nodes(node_count);
        cudaGraphGetNodes(graph.raw_cuda_graph(), nodes.data(), &node_count);
        int kernels = 0;
        for (auto node : nodes) {
            cudaGraphNodeType type;
            cudaGraphNodeGetType(node, &type);
            kernels += type == cudaGraphNodeTypeKernel;
        }
        EXPECT_EQ(kernels, static_cast<int>(node_count));
        EXPECT_TRUE(kernels == 100 || kernels == 200);
        graph.instantiate();
        for (int i = 0; i < 10; ++i)
            graph.replay();
        stream.synchronize();
        std::vector<float> times;
        cudaEvent_t        begin, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        for (int i = 0; i < 15; ++i) {
            cudaEventRecord(begin, stream);
            graph.replay();
            cudaEventRecord(end, stream);
            cudaEventSynchronize(end);
            float ms;
            cudaEventElapsedTime(&ms, begin, end);
            times.push_back(ms * 10);
        }
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
        std::sort(times.begin(), times.end());
        return times[times.size() / 2];
    };
    const auto* props = at::cuda::getDeviceProperties(0);
    std::printf("PACKED_BENCH_GPU name=%s capability=%d.%d sms=%d\n",
                props->name,
                props->major,
                props->minor,
                props->multiProcessorCount);
    for (int b : {64, 128})
        for (int length : {0, 2, 4, 5}) {
            const int            gamma = 5, d = b * 6, q = b * (length + 1), vocab = 129280;
            std::vector<int64_t> m;
            for (int r = 0; r < b; ++r)
                for (int j = 0; j <= length; ++j)
                    m.push_back(r * 6 + j);
            auto map     = torch::tensor(m, torch::kInt64).to(torch::kCUDA);
            auto inverse = torch::full({d}, -1, gpu.dtype(torch::kInt32));
            inverse.index_copy_(0, map, torch::arange(q, inverse.options()));
            const auto make_params = [&](int count) {
                GreedyParams params;
                auto         floats          = torch::ones({6, count}, gpu);
                auto         ints            = torch::ones({5, count}, gpu.dtype(torch::kInt32));
                params.top_p                 = floats[0];
                params.temperature           = floats[1];
                params.repetition_penalty    = floats[2];
                params.presence_penalty      = floats[3];
                params.frequency_penalty     = floats[4];
                params.inverse_temperature   = floats[5];
                params.top_k                 = ints[0];
                params.no_repeat_ngram_size  = ints[1];
                params.input_lengths         = ints[2];
                params.sequence_lengths      = ints[3];
                params.sequence_last_indexes = ints[4];
                params.seed                  = torch::ones({count}, gpu.dtype(torch::kInt64));
                params.offset                = torch::ones({count}, gpu.dtype(torch::kInt64));
                params.do_sample             = torch::ones({count}, gpu.dtype(torch::kBool));
                return params;
            };
            auto src = make_params(d), dst = make_params(q);
            auto tokens = torch::empty({q, 1}, gpu.dtype(torch::kInt32));
            auto gather = [&] { invokeSamplingGather(src, dst, map, {}, tokens, stream); };
            gather();
            auto       logits = torch::randn({q, vocab}, gpu), restored = torch::empty({d, vocab}, gpu);
            const auto zero_us   = time_us([&] { restored.zero_(); });
            const auto copy_us   = time_us([&] { restored.index_copy_(0, map, logits); });
            const auto old_us    = time_us([&] {
                restored.zero_();
                restored.index_copy_(0, map, logits);
            });
            const auto gather_us = time_us(gather);
            for (int width : {0, 1024, vocab}) {
                SamplingResult packed{torch::ones({q, 1}, gpu.dtype(torch::kInt32)),
                                      width ? torch::rand({q, width}, gpu) : torch::Tensor(),
                                      width == 1024 ? torch::zeros({q, width}, gpu.dtype(torch::kInt32)) :
                                                      torch::Tensor(),
                                      {},
                                      {},
                                      torch::ones({q}, gpu.dtype(torch::kBool))};
                SamplingResult dense{torch::empty({d, 1}, gpu.dtype(torch::kInt32)),
                                     width ? torch::empty({d, width}, gpu) : torch::Tensor(),
                                     width == 1024 ? torch::empty({d, width}, gpu.dtype(torch::kInt32)) :
                                                     torch::Tensor(),
                                     {},
                                     {},
                                     torch::empty({d}, gpu.dtype(torch::kBool))};
                auto           scatter    = [&] { invokeSamplingScatter(packed, dense, inverse, stream); };
                const auto     scatter_us = time_us(scatter);
                const auto     new_us     = time_us([&] {
                    gather();
                    scatter();
                });
                std::printf(
                    "PACKED_BENCH bs=%d gamma=%d valid=%d D=%d Q=%d V=%d K=%d zero_us=%.3f copy_us=%.3f old_us=%.3f gather_us=%.3f scatter_us=%.3f new_us=%.3f\n",
                    b,
                    gamma,
                    length,
                    d,
                    q,
                    vocab,
                    width,
                    zero_us,
                    copy_us,
                    old_us,
                    gather_us,
                    scatter_us,
                    new_us);
            }
        }
}
}  // namespace rtp_llm
