#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <limits>
#include <vector>
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

using namespace rtp_llm;

namespace {

template<typename T>
void runPenaltyParityCase(int           batch_size,
                          int           vocab_size,
                          int           max_input_length,
                          int           step,
                          int           token_stride,
                          bool          all_active,
                          torch::Tensor ws_sparse,
                          cudaStream_t  stream) {
    const auto scalar_type = std::is_same_v<T, half> ? torch::kFloat16 : torch::kFloat32;
    auto       f32_opts    = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto       i32_opts    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    auto logits_ref    = torch::randn({batch_size, vocab_size}, f32_opts).to(scalar_type);
    auto logits_sparse = logits_ref.clone();

    auto rep  = torch::ones({batch_size}, f32_opts);
    auto pres = torch::zeros({batch_size}, f32_opts);
    auto freq = torch::zeros({batch_size}, f32_opts);
    for (int b = 0; b < batch_size; ++b) {
        if (all_active || (b % 3 != 0)) {
            rep[b]  = (b % 2 == 0) ? 1.25f : 1.0f;
            pres[b] = (b % 4 == 1) ? 0.4f : 0.0f;
            freq[b] = (b % 5 == 0) ? 0.15f : 0.05f;
        }
    }

    auto input_lengths = torch::empty({batch_size}, i32_opts);
    for (int b = 0; b < batch_size; ++b) {
        input_lengths[b] = (b % 2 == 0) ? std::max(1, max_input_length - (b % 13) - 1) : max_input_length;
    }

    // Build strided row-major tokens [batch_size, token_stride], keeping columns >= step filled with
    // sentinel valid IDs that MUST NOT be penalized.
    auto token_ids = torch::randint(0, vocab_size, {batch_size, token_stride}, i32_opts);
    if (token_stride > step) {
        token_ids.slice(1, step, token_stride).fill_(vocab_size > 3 ? 3 : 0);
    }
    // Force duplicates, negative IDs, out-of-range IDs, and valid tokens inside padding region.
    for (int b = 0; b < batch_size; ++b) {
        if (step >= 6) {
            token_ids[b][0] = 0;
            token_ids[b][1] = 0;
            token_ids[b][2] = 0;
            token_ids[b][3] = -1;
            token_ids[b][4] = vocab_size;
            token_ids[b][5] = vocab_size + 77;
        }
    }

    auto transposed_tokens = token_ids.slice(1, 0, step).transpose(0, 1).contiguous();
    auto ws_ref            = torch::zeros({batch_size, vocab_size}, i32_opts);

    invokeBatchApplyRepetitionPenalty<T>(static_cast<T*>(logits_ref.data_ptr()),
                                         ws_ref.data_ptr<int32_t>(),
                                         rep.data_ptr<float>(),
                                         pres.data_ptr<float>(),
                                         freq.data_ptr<float>(),
                                         transposed_tokens.data_ptr<int32_t>(),
                                         batch_size,
                                         batch_size,
                                         vocab_size,
                                         input_lengths.data_ptr<int32_t>(),
                                         max_input_length,
                                         step,
                                         stream);

    invokeBatchApplyRepetitionPenaltyRowMajorSparse<T>(static_cast<T*>(logits_sparse.data_ptr()),
                                                       ws_sparse.data_ptr<int32_t>(),
                                                       rep.data_ptr<float>(),
                                                       pres.data_ptr<float>(),
                                                       freq.data_ptr<float>(),
                                                       token_ids.data_ptr<int32_t>(),
                                                       batch_size,
                                                       token_stride,
                                                       vocab_size,
                                                       input_lengths.data_ptr<int32_t>(),
                                                       max_input_length,
                                                       step,
                                                       stream);
    check_cuda_error();

    const auto view_type = std::is_same_v<T, half> ? torch::kInt16 : torch::kInt32;
    ASSERT_TRUE(torch::equal(logits_sparse.view(view_type), logits_ref.view(view_type)))
        << "batch=" << batch_size << " vocab=" << vocab_size << " step=" << step;
    ASSERT_EQ(ws_sparse.count_nonzero().item<int64_t>(), 0)
        << "workspace leaked nonzero elements! batch=" << batch_size << " vocab=" << vocab_size;
}

}  // namespace

TEST(SparseRepetitionPenaltyTest, testMatchesLegacyAcrossDtypesAndLayouts) {
    torch::manual_seed(20261128);
    auto stream   = at::cuda::getCurrentCUDAStream().stream();
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    for (int batch : {1, 16, 80}) {
        for (int vocab : {257, 4093, 129280}) {
            auto ws_f32 = torch::zeros({batch, vocab}, i32_opts);
            runPenaltyParityCase<float>(batch, vocab, 48, 64, 72, false, ws_f32, stream);

            auto ws_f16 = torch::zeros({batch, vocab}, i32_opts);
            runPenaltyParityCase<half>(batch, vocab, 48, 64, 72, false, ws_f16, stream);
        }
    }

    // Null optional pointers (e.g., frequency_penalty = nullptr, input_lengths = nullptr)
    constexpr int batch = 8, vocab = 1024, step = 32, max_in = 16;
    auto          f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto          logits_a = torch::randn({batch, vocab}, f32_opts);
    auto          logits_b = logits_a.clone();
    auto          rep      = torch::full({batch}, 1.3f, f32_opts);
    auto          tokens   = torch::randint(0, vocab, {batch, step}, i32_opts);
    auto          tokens_t = tokens.transpose(0, 1).contiguous();
    auto          ws_ref   = torch::zeros({batch, vocab}, i32_opts);
    auto          ws_sp    = torch::zeros({batch, vocab}, i32_opts);

    invokeBatchApplyRepetitionPenalty<float>(logits_a.data_ptr<float>(),
                                             ws_ref.data_ptr<int32_t>(),
                                             rep.data_ptr<float>(),
                                             nullptr,
                                             nullptr,
                                             tokens_t.data_ptr<int32_t>(),
                                             batch,
                                             batch,
                                             vocab,
                                             nullptr,
                                             max_in,
                                             step,
                                             stream);
    invokeBatchApplyRepetitionPenaltyRowMajorSparse<float>(logits_b.data_ptr<float>(),
                                                           ws_sp.data_ptr<int32_t>(),
                                                           rep.data_ptr<float>(),
                                                           nullptr,
                                                           nullptr,
                                                           tokens.data_ptr<int32_t>(),
                                                           batch,
                                                           step,
                                                           vocab,
                                                           nullptr,
                                                           max_in,
                                                           step,
                                                           stream);
    check_cuda_error();
    ASSERT_TRUE(torch::equal(logits_b.view(torch::kInt32), logits_a.view(torch::kInt32)));
    ASSERT_EQ(ws_sp.count_nonzero().item<int64_t>(), 0);
}

TEST(SparseRepetitionPenaltyTest, testMultiStepAndMultiStreamZeroLeakReuse) {
    torch::manual_seed(20261129);
    auto stream0  = c10::cuda::getStreamFromPool();
    auto stream1  = c10::cuda::getStreamFromPool();
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    constexpr int max_batch = 96;
    constexpr int vocab     = 129280;
    auto          ws0       = torch::zeros({static_cast<int64_t>(max_batch) * vocab}, i32_opts);
    auto          ws1       = torch::zeros({static_cast<int64_t>(max_batch) * vocab}, i32_opts);

    const std::vector<int> batches = {80, 1, 96, 24, 80, 48, 80, 32};
    const std::vector<int> steps   = {1, 256, 512, 64, 256, 128, 300, 16};
    for (size_t iter = 0; iter < batches.size(); ++iter) {
        const int  batch       = batches[iter];
        const int  step        = steps[iter];
        const int  max_in      = std::max(1, step / 2);
        const bool active_only = (iter % 2 == 1);

        {
            c10::cuda::CUDAStreamGuard g0(stream0);
            runPenaltyParityCase<float>(batch, vocab, max_in, step, step + 4, active_only, ws0, stream0.stream());
        }
        {
            c10::cuda::CUDAStreamGuard g1(stream1);
            runPenaltyParityCase<float>(batch, vocab, max_in, step, step + 8, !active_only, ws1, stream1.stream());
        }
    }
    stream0.synchronize();
    stream1.synchronize();
    ASSERT_EQ(ws0.count_nonzero().item<int64_t>(), 0);
    ASSERT_EQ(ws1.count_nonzero().item<int64_t>(), 0);
}

TEST(SparseRepetitionPenaltyTest, testScalarPenaltyAndUniformLengthParity) {
    torch::manual_seed(20261201);
    auto stream   = at::cuda::getCurrentCUDAStream().stream();
    auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    constexpr int batch = 80, vocab = 129280, step = 129, max_input_length = 129, uniform_in_len = 120;
    auto          logits_ref = torch::randn({batch, vocab}, f32_opts);
    auto          logits_sp  = logits_ref.clone();

    constexpr float s_rep  = 1.15f;
    constexpr float s_pres = 0.1f;
    constexpr float s_freq = 0.05f;
    auto            rep    = torch::full({batch}, s_rep, f32_opts);
    auto            pres   = torch::full({batch}, s_pres, f32_opts);
    auto            freq   = torch::full({batch}, s_freq, f32_opts);
    auto            in_len = torch::full({batch}, uniform_in_len, i32_opts);

    auto token_ids = torch::randint(0, vocab, {batch, step + 4}, i32_opts);
    auto tokens_t  = token_ids.slice(1, 0, step).transpose(0, 1).contiguous();
    auto ws_ref    = torch::zeros({batch, vocab}, i32_opts);
    auto ws_sparse = torch::zeros({batch, vocab}, i32_opts);

    invokeBatchApplyRepetitionPenalty<float>(logits_ref.data_ptr<float>(),
                                             ws_ref.data_ptr<int32_t>(),
                                             rep.data_ptr<float>(),
                                             pres.data_ptr<float>(),
                                             freq.data_ptr<float>(),
                                             tokens_t.data_ptr<int32_t>(),
                                             batch,
                                             batch,
                                             vocab,
                                             in_len.data_ptr<int32_t>(),
                                             max_input_length,
                                             step,
                                             stream);
    invokeBatchApplyRepetitionPenaltyRowMajorSparse<float>(logits_sp.data_ptr<float>(),
                                                           ws_sparse.data_ptr<int32_t>(),
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           s_rep,
                                                           s_pres,
                                                           s_freq,
                                                           token_ids.data_ptr<int32_t>(),
                                                           batch,
                                                           step + 4,
                                                           vocab,
                                                           nullptr,
                                                           uniform_in_len,
                                                           max_input_length,
                                                           step,
                                                           stream);
    check_cuda_error();
    ASSERT_TRUE(torch::equal(logits_sp.view(torch::kInt32), logits_ref.view(torch::kInt32)));
    ASSERT_EQ(ws_sparse.count_nonzero().item<int64_t>(), 0);
}
