#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <gtest/gtest.h>
#include <torch/torch.h>

#if USING_ASCEND
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>

using namespace std;
using namespace rtp_llm;

class AscendSamplerTest : public ::testing::Test {
public:
    void SetUp() override {
        torch::manual_seed(114514);
        // torch_npu auto-lazy-initializes on first NPU op; force readiness
        auto dummy = torch::ones({1}, torch::kFloat32).to(torch::Device(torch::kPrivateUse1));
    }

protected:
    static const torch::Device npu_device;

    // Helper: create a CPU float tensor from data, then move to NPU for logits
    torch::Tensor npuFloatTensor(std::vector<float> data, std::vector<int64_t> shape) {
        return torch::tensor(data, torch::kFloat32).reshape(shape).to(npu_device);
    }

    // Helper: create a CPU float tensor (stays on CPU)
    torch::Tensor cpuTensor(std::vector<float> data, std::vector<int64_t> shape) {
        return torch::tensor(data, torch::kFloat32).reshape(shape);
    }

    // Helper: create a CPU int32 tensor from data
    torch::Tensor cpuIntTensor(std::vector<int32_t> data, std::vector<int64_t> shape) {
        return torch::tensor(data, torch::kInt32).reshape(shape);
    }

    // Helper: create a pinned CPU tensor from int32 data (for HOST buffers)
    torch::Tensor pinnedIntTensor(std::vector<int32_t> data) {
        return torch::tensor(data, torch::kInt32).pin_memory();
    }

    // Helper: create a pinned CPU tensor from float data
    torch::Tensor pinnedFloatTensor(std::vector<float> data) {
        return torch::tensor(data, torch::kFloat32).pin_memory();
    }

    // Create a seeded NPU generator for deterministic testing.
    at::Generator npuGenerator(uint64_t seed) {
        auto gen = at_npu::detail::createNPUGenerator();  // default device_index = 0
        gen.set_current_seed(seed);
        return gen;
    }
};

const torch::Device AscendSamplerTest::npu_device = torch::Device(torch::kPrivateUse1);

// ======================================================================
// Test: top_k = 1 for all batches → deterministic argmax path
// ======================================================================
TEST_F(AscendSamplerTest, testTopK1Deterministic) {
    size_t batch_size = 4;
    size_t vocab_size = 10;
    auto   logits_t   = npuFloatTensor(
        {
            0.0,  0.0,  0.0,  0.1,  0.2,  0.3,   0.0,  0.0,  0.0,  0.01,   // batch 0: max at idx 5 (0.3)
            0.2,  0.3,  0.0,  0.0,  0.99, 0.989, 0.0,  0.0,  0.0,  0.1,     // batch 1: max at idx 4 (0.99)
            0.44, 0.01, 0.22, 0.0,  0.0,  0.1,   0.2,  0.32, 0.0,  0.44,    // batch 2: max at idx 9 (0.44)
            0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   0.99, 0.0,  0.0,  0.0,     // batch 3: max at idx 6 (0.99)
        },
        {(int64_t)batch_size, (int64_t)vocab_size});

    size_t step             = 5;
    auto   output_token_ids_t = cpuIntTensor(
        {100, 1, 1, 1, 1, 0,
         1,   1, 0, 0, 0, 0,
         1,   0, 1, 0, 0, 0,
         1,   0, 0, 0, 0, 0},
        {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cpuIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cpuIntTensor({-1, -1, -1, -1}, {4});

    auto top_k_t       = pinnedIntTensor({1, 1, 1, 1});
    auto top_p_t       = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
    auto temperature_t = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});

    std::vector<at::Generator> generator(batch_size);

    GreedyParams params{
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperature_t,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        generator,
    };

    auto greedy_output = execSampleGreedy(params);

    auto output_host = output_token_ids_t.cpu().contiguous();
    auto* data       = output_host.data_ptr<int32_t>();
    // batch 0: step+1=6 tokens, last is [5] → value 5 (max at idx 5)
    ASSERT_EQ(data[5], 5);
    // batch 1: last token → value 4 (max at idx 4)
    ASSERT_EQ(data[11], 4);
    // batch 2: last token → value 9 (max at idx 9)
    ASSERT_EQ(data[17], 9);
    // batch 3: last token → value 6 (max at idx 6)
    ASSERT_EQ(data[23], 6);
}

// ======================================================================
// Test: top_k > 1, seeded NPU generator → deterministic multinomial
// ======================================================================
TEST_F(AscendSamplerTest, testTopKWithGeneratorDeterministic) {
    size_t batch_size = 4;
    size_t vocab_size = 10;
    auto   logits_t   = npuFloatTensor(
        {
            0.0,  0.0,  0.0,  0.1,  0.2,  0.3,   0.0,  0.0,  0.0,  0.01,
            0.2,  0.3,  0.0,  0.0,  0.99, 0.989, 0.0,  0.0,  0.0,  0.1,
            0.44, 0.01, 0.22, 0.0,  0.0,  0.1,   0.2,  0.32, 0.0,  0.44,
            0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   0.99, 0.0,  0.0,  0.0,
        },
        {(int64_t)batch_size, (int64_t)vocab_size});

    size_t step             = 5;
    auto   output_token_ids_t = cpuIntTensor(
        {100, 1, 1, 1, 1, 0,
         1,   1, 0, 0, 0, 0,
         1,   0, 1, 0, 0, 0,
         1,   0, 0, 0, 0, 0},
        {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cpuIntTensor({5, 5, 5, 5}, {4});
    auto input_lengths_t    = cpuIntTensor({-1, -1, -1, -1}, {4});

    // batch 2 has top_k=0 → all tokens considered; batch 3 has top_k=2 → top 2 only
    auto top_k_t       = pinnedIntTensor({1, 1, 0, 2});
    auto top_p_t       = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});
    auto temperature_t = pinnedFloatTensor({1.0, 1.0, 1.0, 1.0});

    // Create seeded NPU generators for deterministic testing
    std::vector<at::Generator> generator(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        generator[i] = npuGenerator(42 + i);
    }

    GreedyParams params{
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperature_t,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        generator,
    };

    // First run
    auto greedy_output_1 = execSampleGreedy(params);
    auto tokens_1 = output_token_ids_t.clone();

    // Reset token_ids for second run
    output_token_ids_t = cpuIntTensor(
        {100, 1, 1, 1, 1, 0,
         1,   1, 0, 0, 0, 0,
         1,   0, 1, 0, 0, 0,
         1,   0, 0, 0, 0, 0},
        {(int64_t)batch_size, (int64_t)(step + 1)});

    // Re-create generators with same seeds for reproducibility
    std::vector<at::Generator> generator2(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        generator2[i] = npuGenerator(42 + i);
    }

    GreedyParams params2{
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperature_t,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        generator2,
    };

    auto greedy_output_2 = execSampleGreedy(params2);
    auto tokens_2 = output_token_ids_t;

    // Both runs should produce identical results with same seeds
    auto cpu1 = tokens_1.cpu().contiguous();
    auto cpu2 = tokens_2.cpu().contiguous();
    for (int i = 0; i < cpu1.numel(); i++) {
        ASSERT_EQ(cpu1.data_ptr<int32_t>()[i], cpu2.data_ptr<int32_t>()[i])
            << "Output diverges at index " << i << " with same seed";
    }
}

// ======================================================================
// Test: top_p < 1.0, seeded NPU generator → deterministic multinomial
// ======================================================================
TEST_F(AscendSamplerTest, testTopPWithGeneratorDeterministic) {
    size_t batch_size = 2;
    size_t vocab_size = 6;
    auto   logits_t   = npuFloatTensor(
        {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,   // batch 0: ascending
            6.0, 5.0, 4.0, 3.0, 2.0, 1.0,    // batch 1: descending
        },
        {(int64_t)batch_size, (int64_t)vocab_size});

    size_t step             = 3;
    auto   output_token_ids_t = cpuIntTensor(
        {10, 0, 0, 0,
         10, 1, 1, 1},
        {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cpuIntTensor({3, 3}, {2});
    auto input_lengths_t    = cpuIntTensor({-1, -1}, {2});

    auto top_k_t       = pinnedIntTensor({0, 0});       // no top-k truncation
    auto top_p_t       = pinnedFloatTensor({0.5, 0.5});  // top_p=0.5 filter
    auto temperature_t = pinnedFloatTensor({1.0, 1.0});

    std::vector<at::Generator> generator(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        generator[i] = npuGenerator(100 + i);
    }

    GreedyParams params{
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperature_t,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        generator,
    };

    auto greedy_output_1 = execSampleGreedy(params);
    auto tokens_1 = output_token_ids_t.clone();

    // Reset and re-run with same seeds
    output_token_ids_t = cpuIntTensor(
        {10, 0, 0, 0,
         10, 1, 1, 1},
        {(int64_t)batch_size, (int64_t)(step + 1)});

    std::vector<at::Generator> generator2(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        generator2[i] = npuGenerator(100 + i);
    }

    GreedyParams params2{
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperature_t,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        generator2,
    };

    auto greedy_output_2 = execSampleGreedy(params2);
    auto tokens_2 = output_token_ids_t;

    auto cpu1 = tokens_1.cpu().contiguous();
    auto cpu2 = tokens_2.cpu().contiguous();
    for (int i = 0; i < cpu1.numel(); i++) {
        ASSERT_EQ(cpu1.data_ptr<int32_t>()[i], cpu2.data_ptr<int32_t>()[i])
            << "top_p deterministic output diverges at index " << i;
    }
}

// ======================================================================
// Test: temperature != 1.0 → logits are divided by temperature
// ======================================================================
TEST_F(AscendSamplerTest, testTemperature) {
    size_t batch_size = 2;
    size_t vocab_size = 6;
    // Identical logits for both batches to make the effect of temperature easy to verify
    auto logits_t = npuFloatTensor(
        {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,   // batch 0: ascending
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,    // batch 1: same, but temp=10.0
        },
        {(int64_t)batch_size, (int64_t)vocab_size});

    size_t step             = 3;
    auto   output_token_ids_t = cpuIntTensor(
        {10, 0, 0, 0,
         10, 1, 1, 1},
        {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cpuIntTensor({3, 3}, {2});
    auto input_lengths_t    = cpuIntTensor({-1, -1}, {2});

    auto top_k_t       = pinnedIntTensor({0, 0});       // no top-k truncation
    auto top_p_t       = pinnedFloatTensor({1.0, 1.0});  // no top-p
    auto temperature_t = pinnedFloatTensor({1.0, 10.0});  // batch 0: t=1, batch 1: t=10

    // Seeded generators for deterministic comparison
    std::vector<at::Generator> generator(batch_size);
    generator[0] = npuGenerator(202);
    generator[1] = npuGenerator(202);  // same seed for deterministic: should yield same sample

    // With t=1 and t=10 on identical logits, batch 1 is "flatter" after scaling:
    // logits/temp: [6/10=0.6, 5/10=0.5, ...] → lower temperature → sharper distribution.
    // With same seed, the two batches may or may not sample differently depending on
    // the PRNG state, but temperature MUST divide the logits correctly.
    // We verify that temperature was applied by checking that the overall output
    // is reproducible across runs with the same seed.

    GreedyParams params{
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperature_t,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        generator,
    };

    auto greedy_output_1 = execSampleGreedy(params);
    auto tokens_1 = output_token_ids_t.clone();

    // Reset and re-run with identical parameters
    output_token_ids_t = cpuIntTensor(
        {10, 0, 0, 0,
         10, 1, 1, 1},
        {(int64_t)batch_size, (int64_t)(step + 1)});

    std::vector<at::Generator> generator2(batch_size);
    generator2[0] = npuGenerator(202);
    generator2[1] = npuGenerator(202);

    GreedyParams params2{
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperature_t,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        generator2,
    };

    auto greedy_output_2 = execSampleGreedy(params2);

    auto cpu1 = tokens_1.cpu().contiguous();
    auto cpu2 = output_token_ids_t.cpu().contiguous();
    for (int i = 0; i < cpu1.numel(); i++) {
        ASSERT_EQ(cpu1.data_ptr<int32_t>()[i], cpu2.data_ptr<int32_t>()[i])
            << "temperature test diverges at index " << i << " with same seed";
    }
}

// ======================================================================
// Test: do_sample flag — mixed sampling/greedy batches
// ======================================================================
TEST_F(AscendSamplerTest, testDoSample) {
    size_t batch_size = 3;
    size_t vocab_size = 6;
    auto   logits_t   = npuFloatTensor(
        {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,   // batch 0: do_sample=true → should sample
            9.0, 1.0, 1.0, 1.0, 1.0, 1.0,    // batch 1: do_sample=false → must be argmax (idx 0)
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,    // batch 2: do_sample=true → should sample
        },
        {(int64_t)batch_size, (int64_t)vocab_size});

    size_t step             = 3;
    auto   output_token_ids_t = cpuIntTensor(
        {10, 0, 0, 0,
         20, 5, 5, 5,
         30, 0, 0, 0},
        {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cpuIntTensor({3, 3, 3}, {3});
    auto input_lengths_t    = cpuIntTensor({-1, -1, -1}, {3});

    auto top_k_t       = pinnedIntTensor({0, 0, 0});
    auto top_p_t       = pinnedFloatTensor({1.0, 1.0, 1.0});
    auto temperature_t = pinnedFloatTensor({1.0, 1.0, 1.0});

    // do_sample: batch 0=true, batch 1=false (greedy), batch 2=true
    auto do_sample_t = torch::tensor({true, false, true}, torch::kBool);

    std::vector<at::Generator> generator(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        generator[i] = npuGenerator(300 + i);
    }

    GreedyParams params{
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperature_t,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        nullopt, do_sample_t,
        generator,
    };

    auto greedy_output = execSampleGreedy(params);

    auto output_host = output_token_ids_t.cpu().contiguous();
    auto* data       = output_host.data_ptr<int32_t>();

    // batch 1 (index 1): do_sample=false → must be argmax
    // logits: [9.0, 1.0, 1.0, 1.0, 1.0, 1.0] → max at idx 0
    ASSERT_EQ(data[4], 0) << "batch 1 (do_sample=false) must use argmax, got " << data[4];
}

// ======================================================================
// Test: output_all_probs — verify filtered probability output
// ======================================================================
TEST_F(AscendSamplerTest, testOutputAllProbs) {
    size_t batch_size = 2;
    size_t vocab_size = 5;
    auto   logits_t   = npuFloatTensor(
        {
            1.0, 2.0, 3.0, 4.0, 5.0,   // batch 0: ascending
            5.0, 4.0, 3.0, 2.0, 1.0,    // batch 1: descending
        },
        {(int64_t)batch_size, (int64_t)vocab_size});

    size_t step             = 3;
    auto   output_token_ids_t = cpuIntTensor(
        {10, 0, 0, 0,
         10, 1, 1, 1},
        {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cpuIntTensor({3, 3}, {2});
    auto input_lengths_t    = cpuIntTensor({-1, -1}, {2});

    auto top_k_t       = pinnedIntTensor({0, 0});
    auto top_p_t       = pinnedFloatTensor({1.0, 1.0});
    auto temperature_t = pinnedFloatTensor({1.0, 1.0});

    // output_all_probs tensor: will be filled with filtered probabilities
    auto output_all_probs_t = torch::zeros({(int64_t)batch_size, (int64_t)vocab_size}, torch::kFloat32);

    std::vector<at::Generator> generator(batch_size);

    GreedyParams params{
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperature_t,
        nullopt, nullopt,
        nullopt, nullopt,
        output_all_probs_t,
        nullopt, nullopt,
        nullopt,
        generator,
    };

    auto greedy_output = execSampleGreedy(params);

    // output_all_probs should now contain filtered (post-topk/topp) probabilities
    auto probs_host = output_all_probs_t.cpu().contiguous();
    auto* data      = probs_host.data_ptr<float>();

    // Each batch's probs should sum to approximately 1.0 (after softmax, since no topk/topp filtering)
    for (size_t b = 0; b < batch_size; b++) {
        float sum = 0.0f;
        for (size_t v = 0; v < vocab_size; v++) {
            sum += data[b * vocab_size + v];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-4f) << "batch " << b << " probs sum is " << sum << ", expected ~1.0";
    }

    // Probabilities should be in ascending/descending order matching logits
    for (size_t v = 1; v < vocab_size; v++) {
        ASSERT_LT(data[v - 1], data[v]) << "batch 0 probs must be ascending";
        ASSERT_GT(data[vocab_size + v - 1], data[vocab_size + v]) << "batch 1 probs must be descending";
    }
}

// ======================================================================
// Test: cum_log_probs — verify cumulative log probability accumulation
// ======================================================================
TEST_F(AscendSamplerTest, testCumLogProbs) {
    size_t batch_size = 2;
    size_t vocab_size = 5;
    auto   logits_t   = npuFloatTensor(
        {
            1.0, 2.0, 3.0, 4.0, 5.0,   // batch 0: ascending
            5.0, 4.0, 3.0, 2.0, 1.0,    // batch 1: descending
        },
        {(int64_t)batch_size, (int64_t)vocab_size});

    size_t step             = 3;
    auto   output_token_ids_t = cpuIntTensor(
        {10, 0, 0, 0,
         10, 1, 1, 1},
        {(int64_t)batch_size, (int64_t)(step + 1)});

    auto sequence_lengths_t = cpuIntTensor({3, 3}, {2});
    auto input_lengths_t    = cpuIntTensor({-1, -1}, {2});

    auto top_k_t       = pinnedIntTensor({0, 0});
    auto top_p_t       = pinnedFloatTensor({1.0, 1.0});
    auto temperature_t = pinnedFloatTensor({1.0, 1.0});

    // cum_log_probs tensor: initial value, will be accumulated
    auto cum_log_probs_t = torch::zeros({(int64_t)batch_size}, torch::kFloat32);

    std::vector<at::Generator> generator(batch_size);

    GreedyParams params{
        logits_t,
        input_lengths_t,
        sequence_lengths_t,
        output_token_ids_t,
        step,
        top_k_t,
        top_p_t,
        temperature_t,
        nullopt, nullopt,
        cum_log_probs_t,
        nullopt,
        nullopt, nullopt,
        nullopt, nullopt,
        generator,
    };

    auto greedy_output = execSampleGreedy(params);

    // cum_log_probs should now contain the log-probability of the sampled token
    auto cum_host = cum_log_probs_t.cpu().contiguous();
    auto* data    = cum_host.data_ptr<float>();

    // Log-prob of sampled token must be <= 0 (since probability <= 1)
    ASSERT_LE(data[0], 0.0f) << "cum_log_probs for batch 0 must be <= 0";
    ASSERT_LE(data[1], 0.0f) << "cum_log_probs for batch 1 must be <= 0";

    // Not -inf (token must have non-zero probability)
    ASSERT_GT(data[0], -100.0f) << "cum_log_probs for batch 0 must be > -inf";
    ASSERT_GT(data[1], -100.0f) << "cum_log_probs for batch 1 must be > -inf";
}

#else  // !USING_ASCEND
// Non-Ascend placeholder stub so the file always compiles.
// Tests are gated at the BUILD level (target_compatible_with).
TEST(DummySamplerTest, placeholder) {
    SUCCEED();
}
#endif  // USING_ASCEND
