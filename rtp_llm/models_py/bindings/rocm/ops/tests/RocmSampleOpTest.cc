#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <torch/torch.h>

#include <vector>

using namespace std;
using namespace rtp_llm;

// Regression coverage for the ROCm seeded-sampling path in CudaSampleOp.cc.
//
// When a batch carries a request-level random_seed (a defined per-row generator) and does NOT take
// the top_k==1 fast path, the ROCm branch samples each row independently. Sampling a [1, vocab] row
// yields a [1] token; it must be written into the length-1 slice samples_t[i:i+1], not the 0-D
// scalar samples_t[i] (which fails the shape check and crashes). These tests drive that path with
// top_k / top_p / top_k+top_p and assert three things beyond "does not crash": every output row is
// written back (output column pre-filled with an illegal sentinel), the token is in-vocab, and the
// result is reproducible under a fixed per-row random_seed (proving the seed is actually honored).
class RocmSampleOpTest: public DeviceTestBase {
protected:
    torch::TensorOptions cpu_int_   = torch::TensorOptions(torch::kInt).device(torch::kCPU);
    torch::TensorOptions cpu_float_ = torch::TensorOptions(torch::kFloat).device(torch::kCPU);

    static constexpr int kSentinel = -1;  // illegal token id: proves the op wrote each output row

    // Run seeded sampling once against a fixed logits tensor and return the sampled token ids on CPU.
    // The output column is pre-filled with kSentinel so the caller can prove every row was written to
    // the correct column. Per-row generators are (re)seeded deterministically so two runs with the
    // same logits must yield identical tokens iff the request-level seed is actually honored.
    torch::Tensor runOnce(const torch::Tensor&        logits_cpu,
                          const std::vector<int32_t>& top_k,
                          const std::vector<float>&   top_p) {
        const int64_t batch_size = static_cast<int64_t>(top_k.size());
        const size_t  step       = 1;

        // Fresh clone each run: the ROCm path filters logits in place.
        auto logits           = logits_cpu.clone().to(torch::kCUDA);
        auto input_lengths    = torch::full({batch_size}, -1, cpu_int_).to(torch::kCUDA);
        auto sequence_lengths = torch::full({batch_size}, static_cast<int>(step), cpu_int_).to(torch::kCUDA);
        // Column 0 = valid prior token (0); output column (step) = sentinel to detect write-back.
        auto token_ids = torch::zeros({batch_size, static_cast<int64_t>(step + 1)}, cpu_int_);
        token_ids.select(1, static_cast<int64_t>(step)).fill_(kSentinel);
        token_ids = token_ids.to(torch::kCUDA);

        // top_k / top_p / temperature are read via host pointers in the ROCm branch — keep on CPU.
        auto top_k_t = torch::tensor(top_k, cpu_int_);
        auto top_p_t = torch::tensor(top_p, cpu_float_);
        auto temp_t  = torch::full({batch_size}, 1.0f, cpu_float_);

        // Defined per-row generators == request-level random_seed. Reset to the SAME per-row seeds
        // on every run so results are reproducible only if the op consumes these generators. Same
        // construction the production path uses (GenerateStream.cc); the header hipifies for ROCm.
        std::vector<at::Generator> generator(batch_size);
        for (int64_t i = 0; i < batch_size; ++i) {
            auto gen = torch::make_generator<torch::CUDAGeneratorImpl>();
            gen.set_current_seed(static_cast<uint64_t>(i + 1));
            generator[i] = gen;
        }

        GreedyParams params({
            logits,
            input_lengths,
            sequence_lengths,
            token_ids,
            step,
            top_k_t,
            top_p_t,
            temp_t,
            std::nullopt,  // repetition_penalty
            std::nullopt,  // no_repeat_ngram_size
            std::nullopt,  // cum_log_probs
            std::nullopt,  // output_log_probs
            false,         // return_original_all_probs
            std::nullopt,  // output_all_probs
            std::nullopt,  // presence_penalty
            std::nullopt,  // frequency_penalty
            std::nullopt,  // do_sample
            generator,
        });

        // ROCm sampleGreedy returns an empty GreedyOutput{} by contract (success is not populated on
        // this path), so we validate via the written token ids. The .to(CPU) copy also forces a
        // device sync that surfaces any kernel / shape-mismatch crash on this seeded path.
        execSampleGreedy(params);
        return token_ids.to(torch::kCPU);
    }

    void runSeededSamplingTest(const std::vector<int32_t>& top_k, const std::vector<float>& top_p) {
        const int64_t batch_size = static_cast<int64_t>(top_k.size());
        const int64_t vocab      = 16;
        const size_t  step       = 1;

        // Fixed logits (deterministic seed) so two runs with the same per-row seeds must match.
        auto cpu_gen    = at::detail::createCPUGenerator(20240701ULL);
        auto logits_cpu = torch::randn({batch_size, vocab}, cpu_gen, cpu_float_);

        auto ids1 = runOnce(logits_cpu, top_k, top_p);
        auto ids2 = runOnce(logits_cpu, top_k, top_p);
        auto a1   = ids1.accessor<int, 2>();
        auto a2   = ids2.accessor<int, 2>();

        for (int64_t b = 0; b < batch_size; ++b) {
            int tok = a1[b][step];
            // Write-back: the sentinel must have been overwritten in this row's output column.
            ASSERT_NE(tok, kSentinel) << "row " << b << " output token not written back";
            ASSERT_GE(tok, 0);
            ASSERT_LT(tok, static_cast<int>(vocab));
            // Seed semantics: identical logits + identical per-row seeds => identical token.
            ASSERT_EQ(tok, a2[b][step]) << "row " << b << " not reproducible under fixed random_seed";
        }
    }
};

TEST_F(RocmSampleOpTest, seededTopKSampling) {
    runSeededSamplingTest(/*top_k=*/{4, 3, 2, 5}, /*top_p=*/{1.0f, 1.0f, 1.0f, 1.0f});
}

TEST_F(RocmSampleOpTest, seededTopPSampling) {
    runSeededSamplingTest(/*top_k=*/{0, 0, 0, 0}, /*top_p=*/{0.9f, 0.8f, 0.95f, 0.7f});
}

TEST_F(RocmSampleOpTest, seededTopKTopPSampling) {
    runSeededSamplingTest(/*top_k=*/{5, 4, 3, 2}, /*top_p=*/{0.9f, 0.8f, 0.95f, 0.7f});
}
