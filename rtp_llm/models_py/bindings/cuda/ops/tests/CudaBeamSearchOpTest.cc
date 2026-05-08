#include "rtp_llm/cpp/testing/BeamSearchOpTest.hpp"
#include <ATen/cuda/CUDAGeneratorImpl.h>
using namespace std;
using namespace rtp_llm;

namespace rtp_llm {
BeamSearchOutput sampleBeamSearch(const BeamSearchParams& params);
}

class CudaBeamSearchOpTest: public BeamSearchOpTest {
protected:
    struct StochasticTestInput {
        torch::Tensor logits;
        torch::Tensor token_ids;
        torch::Tensor input_lengths;
        torch::Tensor sequence_lengths;
        torch::Tensor cum_log_probs;
    };

    StochasticTestInput prepareStochasticInput(int batch_size, int beam_width, int vocab_size, int max_seq_len) {
        auto logits = torch::randn({batch_size, beam_width, vocab_size}, float_options).to(torch::kCUDA);
        auto token_ids =
            torch::randint(0, vocab_size, {batch_size, beam_width, max_seq_len}, int_options).to(torch::kCUDA);
        auto input_lengths    = torch::full({batch_size, beam_width}, 3, int_options).to(torch::kCUDA);
        auto sequence_lengths = torch::full({batch_size, beam_width}, 5, int_options).to(torch::kCUDA);
        auto cum_log_probs    = torch::randn({batch_size, beam_width}, float_options).to(torch::kCUDA);
        return {logits, token_ids, input_lengths, sequence_lengths, cum_log_probs};
    }

    BeamSearchOutput runBeamSearch(const StochasticTestInput& input,
                                   torch::Tensor              temperature,
                                   int                        beam_width,
                                   std::vector<at::Generator> generators = {}) {
        BeamSearchParams params{input.logits,
                                temperature,
                                input.token_ids,
                                input.input_lengths,
                                input.sequence_lengths,
                                input.cum_log_probs,
                                0,
                                std::move(generators)};
        return sampleBeamSearch(params);
    }

    BeamSearchOutput runBeamSearch(const StochasticTestInput& input,
                                   torch::Tensor              temperature,
                                   int                        beam_width,
                                   at::Generator              generator) {
        std::vector<at::Generator> gens(input.logits.size(0), generator);
        return runBeamSearch(input, temperature, beam_width, std::move(gens));
    }

    at::Generator makeSeededGenerator(uint64_t seed) {
        auto gen = torch::make_generator<at::CUDAGeneratorImpl>();
        gen.set_current_seed(seed);
        return gen;
    }
};

TEST_F(CudaBeamSearchOpTest, simpleTest) {
    std::vector<int> batch_sizes = {1, 2, 32};
    std::vector<int> beam_widths = {1, 4, 64, 500, 2500};
    std::vector<int> max_seq_len = {10, 100};
    const int        vocab_size  = 7000;

    for (auto batch_size : batch_sizes) {
        for (auto beam_width : beam_widths) {
            for (auto seq_len : max_seq_len) {
                std::cout << "batch_size: " << batch_size << ", beam_width: " << beam_width
                          << ", vocab_size: " << vocab_size << ", seq_len: " << seq_len << std::endl;
                simpleTest(batch_size, beam_width, vocab_size, seq_len);
            }
        }
    }
}

TEST_F(CudaBeamSearchOpTest, variableBeamWidthTest) {
    std::vector<int> batch_sizes = {1, 2};
    std::vector<int> beam_widths = {1, 5, 70, 500};
    std::vector<int> max_seq_len = {10, 100};
    const int        vocab_size  = 7000;

    for (auto batch_size : batch_sizes) {
        for (auto beam_width_in : beam_widths) {
            for (auto beam_width_out : beam_widths) {
                if (beam_width_in == beam_width_out)
                    continue;
                for (auto seq_len : max_seq_len) {
                    std::cout << "batch_size: " << batch_size << ", beam_width_in: " << beam_width_in
                              << ", beam_width_out: " << beam_width_out << ", vocab_size: " << vocab_size
                              << ", seq_len: " << seq_len << std::endl;
                    variableBeamWidthTest(batch_size, beam_width_in, beam_width_out, vocab_size, seq_len);
                }
            }
        }
    }
}

TEST_F(CudaBeamSearchOpTest, stochasticBeamSearch_Temperature1_ValidOutput) {
    const int batch_size = 1, beam_width = 4, vocab_size = 7000, max_seq_len = 10;
    auto      input       = prepareStochasticInput(batch_size, beam_width, vocab_size, max_seq_len);
    auto      temperature = torch::ones({batch_size, beam_width}, float_options).to(torch::kCUDA);

    auto result = runBeamSearch(input, temperature, beam_width);

    ASSERT_EQ(result.token_ids.size(1), beam_width);
    auto clp = result.cum_log_probs.cpu();
    ASSERT_FALSE(torch::any(torch::isnan(clp)).item<bool>());
    ASSERT_FALSE(torch::any(torch::isinf(clp)).item<bool>());
}

TEST_F(CudaBeamSearchOpTest, stochasticBeamSearch_EmptyTemperature_ValidOutput) {
    const int batch_size = 1, beam_width = 4, vocab_size = 7000, max_seq_len = 10;
    auto      input = prepareStochasticInput(batch_size, beam_width, vocab_size, max_seq_len);

    auto result = runBeamSearch(input, torch::Tensor(), beam_width);

    ASSERT_EQ(result.token_ids.size(1), beam_width);
    auto clp = result.cum_log_probs.cpu();
    ASSERT_FALSE(torch::any(torch::isnan(clp)).item<bool>());
    ASSERT_FALSE(torch::any(torch::isinf(clp)).item<bool>());
}

TEST_F(CudaBeamSearchOpTest, stochasticBeamSearch_Temperature0_ValidOutput) {
    const int batch_size = 1, beam_width = 4, vocab_size = 7000, max_seq_len = 10;
    auto      input       = prepareStochasticInput(batch_size, beam_width, vocab_size, max_seq_len);
    auto      temperature = torch::zeros({batch_size, beam_width}, float_options).to(torch::kCUDA);

    auto result = runBeamSearch(input, temperature, beam_width);

    ASSERT_EQ(result.token_ids.size(1), beam_width);
    auto clp = result.cum_log_probs.cpu();
    ASSERT_FALSE(torch::any(torch::isnan(clp)).item<bool>());
    ASSERT_FALSE(torch::any(torch::isinf(clp)).item<bool>());
}

TEST_F(CudaBeamSearchOpTest, stochasticBeamSearch_OutputShapesAndValidity) {
    const int batch_size = 2, beam_width = 4, vocab_size = 7000, max_seq_len = 10;
    auto      input       = prepareStochasticInput(batch_size, beam_width, vocab_size, max_seq_len);
    auto      temperature = torch::full({batch_size, beam_width}, 0.5f, float_options).to(torch::kCUDA);

    auto result = runBeamSearch(input, temperature, beam_width);

    ASSERT_EQ(result.token_ids.size(0), batch_size);
    ASSERT_EQ(result.token_ids.size(1), beam_width);
    ASSERT_EQ(result.token_ids.size(2), max_seq_len);
    ASSERT_EQ(result.cum_log_probs.size(0), batch_size);
    ASSERT_EQ(result.cum_log_probs.size(1), beam_width);
    ASSERT_EQ(result.beam_indices.size(0), batch_size);
    ASSERT_EQ(result.beam_indices.size(1), beam_width);

    auto cum_log_cpu = result.cum_log_probs.cpu();
    ASSERT_FALSE(torch::any(torch::isnan(cum_log_cpu)).item<bool>()) << "cum_log_probs contains NaN";
    ASSERT_FALSE(torch::any(torch::isinf(cum_log_cpu)).item<bool>()) << "cum_log_probs contains Inf";
}

TEST_F(CudaBeamSearchOpTest, stochasticBeamSearch_DoesNotModifyInputLogits) {
    const int batch_size = 1, beam_width = 4, vocab_size = 7000, max_seq_len = 10;
    auto      input         = prepareStochasticInput(batch_size, beam_width, vocab_size, max_seq_len);
    auto      logits_backup = input.logits.clone();
    auto      temperature   = torch::full({batch_size, beam_width}, 0.5f, float_options).to(torch::kCUDA);

    runBeamSearch(input, temperature, beam_width);

    ASSERT_TRUE(torch::equal(input.logits, logits_backup))
        << "sampleBeamSearch should not modify the input logits tensor";
}

TEST_F(CudaBeamSearchOpTest, stochasticBeamSearch_HighTemperature_ValidOutput) {
    const int batch_size = 2, beam_width = 4, vocab_size = 7000, max_seq_len = 10;
    auto      input       = prepareStochasticInput(batch_size, beam_width, vocab_size, max_seq_len);
    auto      temperature = torch::full({batch_size, beam_width}, 5.0f, float_options).to(torch::kCUDA);

    for (int i = 0; i < 10; ++i) {
        auto result = runBeamSearch(input, temperature, beam_width);
        auto clp    = result.cum_log_probs.cpu();
        ASSERT_FALSE(torch::any(torch::isnan(clp)).item<bool>()) << "cum_log_probs contains NaN on iteration " << i;
        ASSERT_FALSE(torch::any(torch::isinf(clp)).item<bool>()) << "cum_log_probs contains Inf on iteration " << i;
        ASSERT_EQ(result.token_ids.size(0), batch_size);
        ASSERT_EQ(result.token_ids.size(1), beam_width);
        ASSERT_EQ(result.token_ids.size(2), max_seq_len);
    }
}

TEST_F(CudaBeamSearchOpTest, stochasticBeamSearch_MultipleTemperatureValues) {
    const int batch_size = 1, beam_width = 4, vocab_size = 7000, max_seq_len = 10;
    auto      input = prepareStochasticInput(batch_size, beam_width, vocab_size, max_seq_len);

    for (float temp : {0.1f, 0.3f, 0.5f, 0.8f, 1.5f, 3.0f}) {
        auto temperature = torch::full({batch_size, beam_width}, temp, float_options).to(torch::kCUDA);
        auto result      = runBeamSearch(input, temperature, beam_width);

        auto cum_log_cpu = result.cum_log_probs.cpu();
        ASSERT_FALSE(torch::any(torch::isnan(cum_log_cpu)).item<bool>())
            << "NaN in cum_log_probs at temperature=" << temp;
        ASSERT_FALSE(torch::any(torch::isinf(cum_log_cpu)).item<bool>())
            << "Inf in cum_log_probs at temperature=" << temp;
        ASSERT_EQ(result.token_ids.size(1), beam_width);
    }
}

TEST_F(CudaBeamSearchOpTest, stochasticBeamSearch_PerBatchTemperature) {
    const int batch_size = 3, beam_width = 4, vocab_size = 7000, max_seq_len = 10;
    auto      input = prepareStochasticInput(batch_size, beam_width, vocab_size, max_seq_len);

    auto temperature = torch::zeros({batch_size, beam_width}, float_options);
    temperature[0].fill_(0.5f);
    temperature[1].fill_(1.0f);
    temperature[2].fill_(0.0f);

    for (int i = 0; i < 5; ++i) {
        auto result = runBeamSearch(input, temperature.to(torch::kCUDA), beam_width);

        auto clp = result.cum_log_probs.cpu();
        ASSERT_FALSE(torch::any(torch::isnan(clp)).item<bool>()) << "NaN on iteration " << i;
        ASSERT_FALSE(torch::any(torch::isinf(clp)).item<bool>()) << "Inf on iteration " << i;
        ASSERT_EQ(result.token_ids.size(0), batch_size);
        ASSERT_EQ(result.token_ids.size(1), beam_width);
        ASSERT_EQ(result.token_ids.size(2), max_seq_len);
    }
}

TEST_F(CudaBeamSearchOpTest, stochasticBeamSearch_FixedSeed_Deterministic) {
    const int      batch_size = 2, beam_width = 4, vocab_size = 7000, max_seq_len = 10;
    const uint64_t seed        = 42;
    auto           input       = prepareStochasticInput(batch_size, beam_width, vocab_size, max_seq_len);
    auto           temperature = torch::full({batch_size, beam_width}, 0.5f, float_options);

    auto gen1    = makeSeededGenerator(seed);
    auto result1 = runBeamSearch(input, temperature, beam_width, gen1);

    auto gen2    = makeSeededGenerator(seed);
    auto result2 = runBeamSearch(input, temperature, beam_width, gen2);

    ASSERT_TRUE(torch::equal(result1.token_ids.cpu(), result2.token_ids.cpu()))
        << "Same seed should produce identical token_ids";
    ASSERT_TRUE(torch::allclose(result1.cum_log_probs.cpu(), result2.cum_log_probs.cpu()))
        << "Same seed should produce identical cum_log_probs";
    ASSERT_TRUE(torch::equal(result1.beam_indices.cpu(), result2.beam_indices.cpu()))
        << "Same seed should produce identical beam_indices";
}

TEST_F(CudaBeamSearchOpTest, stochasticBeamSearch_DifferentSeed_DifferentOutput) {
    const int batch_size = 1, beam_width = 4, vocab_size = 7000, max_seq_len = 10;
    auto      input       = prepareStochasticInput(batch_size, beam_width, vocab_size, max_seq_len);
    auto      temperature = torch::full({batch_size, beam_width}, 0.5f, float_options).to(torch::kCUDA);

    auto gen1    = makeSeededGenerator(42);
    auto result1 = runBeamSearch(input, temperature, beam_width, gen1);

    auto gen2    = makeSeededGenerator(12345);
    auto result2 = runBeamSearch(input, temperature, beam_width, gen2);

    bool token_ids_differ  = !torch::equal(result1.token_ids.cpu(), result2.token_ids.cpu());
    bool cum_log_differ    = !torch::allclose(result1.cum_log_probs.cpu(), result2.cum_log_probs.cpu());
    bool beam_indices_diff = !torch::equal(result1.beam_indices.cpu(), result2.beam_indices.cpu());
    ASSERT_TRUE(token_ids_differ || cum_log_differ || beam_indices_diff)
        << "Different seeds should produce different outputs (with high probability)";
}

TEST_F(CudaBeamSearchOpTest, stochasticBeamSearch_NoGenerator_NonDeterministic) {
    const int batch_size = 1, beam_width = 4, vocab_size = 7000, max_seq_len = 10;
    auto      input       = prepareStochasticInput(batch_size, beam_width, vocab_size, max_seq_len);
    auto      temperature = torch::full({batch_size, beam_width}, 0.5f, float_options).to(torch::kCUDA);

    bool found_difference = false;
    auto result1          = runBeamSearch(input, temperature, beam_width);
    for (int i = 0; i < 10 && !found_difference; ++i) {
        auto result2 = runBeamSearch(input, temperature, beam_width);
        if (!torch::equal(result1.cum_log_probs.cpu(), result2.cum_log_probs.cpu())) {
            found_difference = true;
        }
    }
    ASSERT_TRUE(found_difference)
        << "Without generator, stochastic beam search should produce different results across runs";
}
