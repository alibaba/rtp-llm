#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/test/ModelTestUtil.h"
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

using namespace std;
using namespace rtp_llm;

// TODO: make this test device-independent
class SamplerTest: public DeviceTestBase {
public:
    void SetUp() override {
        DeviceTestBase::SetUp();
        SamplerInitParams params{};
        sampler_.reset(new Sampler(params));
    }

protected:
    std::unique_ptr<Sampler> sampler_;
};

namespace {
SamplerInputs makeGreedySamplerInput(int64_t rows) {
    SamplerInputs inputs;
    inputs.batch_size = inputs.batch_size_out = rows;
    inputs.vocab_size                         = 8;
    inputs.step                               = 1;
    inputs.logits                             = torch::zeros({rows, 8}, torch::TensorOptions().device(torch::kCUDA));
    inputs.logits.select(1, 3).fill_(10);
    inputs.token_ids     = torch::zeros({rows, 2}, torch::kInt32);
    inputs.input_lengths = inputs.sequence_lengths = torch::ones({rows}, torch::kInt32);
    inputs.num_beams_in = inputs.num_beams_out = torch::ones({rows}, torch::kInt64);
    inputs.top_k                               = torch::ones({rows}, torch::kInt32);
    inputs.top_p                               = torch::zeros({rows}, torch::kFloat32);
    inputs.temperature                         = torch::ones({rows}, torch::kFloat32);
    inputs.generator.resize(rows);
    return inputs;
}
}  // namespace

TEST_F(SamplerTest, testDynamicBatchCapacityThroughForward) {
    Sampler sampler(SamplerInitParams{1, false});
    for (const int64_t rows : {2, 5, 1, 5}) {
        auto inputs = makeGreedySamplerInput(rows);
        auto output = sampler.forward(inputs);
        EXPECT_TRUE(torch::equal(output.token_ids.select(1, 1).cpu(), torch::full({rows}, 3, torch::kInt32)));
        EXPECT_TRUE(output.success.all().item<bool>());
    }
}

TEST_F(SamplerTest, testFixedBatchCapacityThroughForward) {
    Sampler sampler(SamplerInitParams{1, true});
    auto    oversized = makeGreedySamplerInput(2);
    EXPECT_THROW(sampler.forward(oversized), rtp_llm::RTPException);
    // Rejecting an oversized batch must not prevent a subsequent valid call.
    auto valid = makeGreedySamplerInput(1);
    EXPECT_EQ(sampler.forward(valid).token_ids[0][1].item<int32_t>(), 3);
}

TEST_F(SamplerTest, testTokenBufferBindingWithOptionalHistory) {
    for (int scenario = 0; scenario < 5; ++scenario) {
        SCOPED_TRACE(scenario);
        auto inputs             = makeGreedySamplerInput(2);
        inputs.step             = 3;
        inputs.sequence_lengths = torch::full({2}, 3, torch::kInt32);
        inputs.logits.select(1, 2).fill_(5);
        inputs.token_ids = scenario == 0 ? torch::Tensor() : torch::full({2, 4}, 3, torch::kInt32);
        if (scenario == 2 || scenario == 4) {
            inputs.token_ids = inputs.token_ids.to(inputs.logits.device());
        }
        if (scenario >= 3) {
            // The repeated bigram bans token 3, so sampling must choose token 2.
            inputs.no_repeat_ngram_size = torch::full({2}, 2, torch::kInt32);
        }
        auto output = sampler_->forward(inputs);
        EXPECT_EQ(output.token_ids.size(0), 2);
        EXPECT_EQ(output.token_ids.size(1), 4);
        EXPECT_EQ(output.token_ids.device(), inputs.logits.device());
        EXPECT_TRUE(
            torch::equal(output.token_ids.select(1, 3).cpu(), torch::full({2}, scenario >= 3 ? 2 : 3, torch::kInt32)));
        EXPECT_TRUE(output.success.all().item<bool>());
        if (scenario != 0) {
            EXPECT_TRUE(torch::equal(output.token_ids.narrow(1, 0, 3).cpu(), torch::full({2, 3}, 3, torch::kInt32)));
        }
    }
}

#if USING_CUDA
TEST_F(SamplerTest, testMappedSamplingRejectsBeamAndContextRows) {
    for (int scenario = 0; scenario < 3; ++scenario) {
        SCOPED_TRACE(scenario);
        auto inputs        = makeGreedySamplerInput(2);
        inputs.row_mapping = SamplingRowMapping{torch::tensor({0}, torch::kInt64).to(torch::kCUDA),
                                                torch::tensor({0, -1}, torch::kInt32).to(torch::kCUDA)};
        inputs.logits      = inputs.logits.narrow(0, 0, 1);
        if (scenario == 0) {
            inputs.num_beams_in = torch::full({2}, 2, torch::kInt64);
        } else if (scenario == 1) {
            inputs.num_beams_out = torch::full({2}, 2, torch::kInt64);
        } else {
            inputs.sequence_lengths = inputs.sequence_lengths.narrow(0, 0, 1);
        }
        try {
            sampler_->forward(inputs);
            FAIL() << "row mapping must reject beam and context rows";
        } catch (const c10::Error& error) {
            EXPECT_NE(std::string(error.what()).find("row selection requires single-beam decoder rows"),
                      std::string::npos);
        }
        inputs.num_beams_in = inputs.num_beams_out = torch::ones({2}, torch::kInt64);
        inputs.sequence_lengths                    = torch::ones({2}, torch::kInt32);
        auto output                                = sampler_->forward(inputs);
        EXPECT_TRUE(torch::equal(output.token_ids.cpu(), torch::tensor({{3}, {0}}, torch::kInt32)));
    }
}

TEST_F(SamplerTest, testMappedSamplingRejectsLogitsProcessorState) {
    auto inputs                        = makeGreedySamplerInput(2);
    inputs.row_mapping                 = SamplingRowMapping{torch::tensor({0}, torch::kInt64).to(torch::kCUDA),
                                            torch::tensor({0, -1}, torch::kInt32).to(torch::kCUDA)};
    inputs.logits                      = inputs.logits.narrow(0, 0, 1);
    inputs.logits_processor_states_ptr = std::make_shared<LogitsProcessorStates>();
    try {
        sampler_->forward(inputs);
        FAIL() << "mapped logits must be processed before sampling";
    } catch (const c10::Error& error) {
        EXPECT_NE(std::string(error.what()).find("mapped sampling requires logits processors to run before"),
                  std::string::npos);
    }
    inputs.logits_processor_states_ptr.reset();
    auto output = sampler_->forward(inputs);
    EXPECT_TRUE(torch::equal(output.token_ids.cpu(), torch::tensor({{3}, {0}}, torch::kInt32)));
}

TEST_F(SamplerTest, testProbabilityOutputsAreOwnedByForward) {
    for (int scenario = 0; scenario < 9; ++scenario) {
        SCOPED_TRACE(scenario);
        auto inputs                      = makeGreedySamplerInput(2);
        inputs.probability_output        = scenario == 0                  ? SamplerProbabilityOutput::NONE :
                                           scenario == 1 || scenario == 6 ? SamplerProbabilityOutput::DENSE :
                                                                            SamplerProbabilityOutput::DISTRIBUTION;
        inputs.skip_logits_probs_restore = true;
        inputs.return_original_all_probs = scenario == 3;
        inputs.return_logprobs           = scenario == 4;
        if (scenario >= 5) {
            inputs.vocab_size = 1024;
            inputs.logits     = torch::full({2, 1024}, -100.f, inputs.logits.options());
            inputs.logits.select(1, 3).fill_(10.f);
            inputs.logits.select(1, 5).fill_(9.f);
            inputs.top_k.fill_(scenario == 7 ? 2 : 0);
            inputs.top_p.fill_(.95f);
            inputs.force_topk = 128;
        }
        if (scenario == 8) {
            inputs.cum_log_probs = torch::zeros({2}, inputs.logits.options());
        }
        const auto original = torch::softmax(inputs.logits, -1);
        auto       output   = sampler_->forward(inputs);
        EXPECT_EQ(output.all_probs.defined(), scenario != 0);
        EXPECT_EQ(output.token_ids_are_point_mass, scenario == 2);
        EXPECT_EQ(output.all_probs_indices.defined(), scenario == 5);
        if (scenario != 0 && scenario != 2) {
            EXPECT_EQ(output.all_probs.size(1), scenario == 5 ? 128 : inputs.vocab_size);
            EXPECT_TRUE(torch::allclose(output.all_probs.sum(-1), torch::ones({2}, original.options()), 1e-5, 1e-6));
            if (scenario == 3) {
                EXPECT_TRUE(torch::allclose(output.all_probs, original, 1e-5, 1e-6));
            } else {
                auto dense = output.all_probs_indices.defined() ?
                                 torch::zeros_like(original).scatter_(
                                     1, output.all_probs_indices.to(torch::kLong), output.all_probs) :
                                 output.all_probs;
                auto expected =
                    scenario >= 5 ? original : torch::zeros_like(original).scatter_(1, original.argmax(-1, true), 1.f);
                EXPECT_TRUE(torch::allclose(dense, expected, 1e-5, 1e-6));
            }
        }
        EXPECT_TRUE(inputs.top_k.is_cpu());
        EXPECT_TRUE(inputs.top_p.is_cpu());
    }
}

TEST_F(SamplerTest, testUnchangedBeamGroupsReuseHistoryStorage) {
    for (const auto& beams : std::vector<std::vector<int64_t>>{{1, 1, 1, 1}, {2, 2, 2, 2}, {1, 2, 2, 1}}) {
        SCOPED_TRACE(::testing::PrintToString(beams));
        auto inputs         = makeGreedySamplerInput(4);
        inputs.token_ids    = torch::tensor({{1, -1}, {2, -1}, {3, -1}, {4, -1}}, torch::kInt32).to(torch::kCUDA);
        inputs.num_beams_in = inputs.num_beams_out = torch::tensor(beams, torch::kInt64);
        inputs.cum_log_probs                       = torch::zeros({4}, inputs.logits.options());
        inputs.return_logprobs                     = true;
        auto output                                = sampler_->forward(inputs);
        EXPECT_EQ(output.token_ids.data_ptr(), inputs.token_ids.data_ptr());
        EXPECT_TRUE(output.token_ids.select(1, 1).eq(3).all().item<bool>());
        // Each beam has one distinct winning candidate, so all four prefixes survive.
        auto prefixes = std::get<0>(output.token_ids.select(1, 0).sort()).cpu();
        EXPECT_TRUE(torch::equal(prefixes, torch::tensor({1, 2, 3, 4}, torch::kInt32)));
        EXPECT_TRUE(output.success.all().item<bool>());
        EXPECT_EQ(output.sampling_probs.sizes().vec(), (std::vector<int64_t>{4, 8}));
        EXPECT_TRUE(
            torch::allclose(output.sampling_probs.sum(-1), torch::ones({4}, inputs.logits.options()), 1e-5, 1e-6));
        EXPECT_TRUE(output.cum_log_probs.isfinite().all().item<bool>());
    }
}

TEST_F(SamplerTest, testForwardPreservesCpuMetadataAndRngOrderAcrossGroups) {
    auto inputs         = makeGreedySamplerInput(6);
    inputs.vocab_size   = 16;
    inputs.step         = 9;
    inputs.logits       = torch::arange(16, inputs.logits.options()).repeat({6, 1});
    inputs.token_ids    = torch::zeros({6, 10}, torch::kInt32);
    inputs.num_beams_in = inputs.num_beams_out = torch::tensor({1L, 1L, 2L, 2L, 1L, 1L}, torch::kInt64);
    inputs.top_k                               = torch::tensor({1, 1, 1, 1, 0, 4}, torch::kInt32);
    inputs.top_p                               = torch::tensor({0.f, 1.f, 1.f, 1.f, 0.f, .7f});
    inputs.temperature                         = torch::tensor({1.f, 1.f, 1.f, 1.f, .5f, 1.f});
    inputs.input_lengths                       = torch::full({6}, 3, torch::kInt32);
    inputs.sequence_lengths                    = torch::tensor({5, 6, 7, 8, 9}, torch::kInt32);
    auto generator                             = at::cuda::detail::createCUDAGenerator();
    generator.set_current_seed(123);
    inputs.generator = std::vector<at::Generator>(6, generator);
    auto original_k = inputs.top_k.clone(), original_p = inputs.top_p.clone();
    for (int round = 0; round < 2; ++round) {
        inputs.logits = torch::arange(16, inputs.logits.options()).repeat({6, 1});
        auto output   = sampler_->forward(inputs);
        EXPECT_EQ(output.token_ids.size(0), 6);
        EXPECT_TRUE(output.token_ids.narrow(0, 0, 2).select(1, 9).eq(15).all().item<bool>());
        EXPECT_EQ(generator.get_offset(), 64 * (round + 1));
        EXPECT_TRUE(torch::equal(inputs.top_k, original_k));
        EXPECT_TRUE(torch::equal(inputs.top_p, original_p));
        EXPECT_EQ(inputs.sequence_lengths.numel(), 5);
    }
}

#endif

TEST_F(SamplerTest, testMixedDynamicBeamTransitionsUseIndependentOutput) {
    constexpr size_t batch_size     = 4;
    constexpr size_t batch_size_out = 4;
    constexpr size_t vocab_size     = 16;
    constexpr size_t step           = 2;
    // Sampler inputs always carry token_ids of width step + 1; the greedy kernel
    // writes the newly sampled token into the last column.
    constexpr size_t max_seq_len = step + 1;

    auto logits = torch::full({(int64_t)batch_size, (int64_t)vocab_size}, -100.0f, torch::kFloat32);
    logits[0][5].fill_(10.0f);
    logits[1][6].fill_(10.0f);
    logits[2][7].fill_(10.0f);
    logits[3][8].fill_(10.0f);
    logits = logits.to(torch::kCUDA);

    auto token_ids = torch::tensor({1, 1, -1, 2, 2, -1, 3, 3, -1, 4, 4, -1}, torch::kInt32)
                         .reshape({(int64_t)batch_size, (int64_t)max_seq_len});
    auto input_lengths    = torch::ones({(int64_t)batch_size}, torch::kInt32);
    auto sequence_lengths = torch::full({(int64_t)batch_size}, (int64_t)step, torch::kInt32);
    auto num_beams_in     = torch::tensor({1L, 2L, 2L, 1L}, torch::kLong);
    auto num_beams_out    = torch::tensor({2L, 1L, 1L, 1L}, torch::kLong);
    auto top_k            = torch::ones({(int64_t)batch_size}, torch::kInt32).pin_memory();
    auto top_p            = torch::zeros({(int64_t)batch_size}, torch::kFloat32).pin_memory();
    auto temperature      = torch::ones({(int64_t)batch_size}, torch::kFloat32).pin_memory();
    auto cum_log_probs    = torch::tensor({0.0f, 0.0f, -100.0f, 0.0f}, torch::kFloat32).to(torch::kCUDA);

    std::vector<at::Generator> generators(batch_size);
    SamplerInputs              inputs{logits,
                         token_ids,
                         input_lengths,
                         sequence_lengths,
                         std::make_shared<LogitsProcessorStates>(),
                         vocab_size,
                         step,
                         batch_size,
                         batch_size_out,
                         num_beams_in,
                         num_beams_out,
                         top_k,
                         top_p,
                         temperature,
                         torch::Tensor(),
                         torch::Tensor(),
                         torch::Tensor(),
                         torch::Tensor(),
                         torch::Tensor(),
                         torch::Tensor(),
                         false,
                         false,  // skip_logits_probs_restore
                         0,      // force_topk
                         SamplerProbabilityOutput::NONE,
                         cum_log_probs,
                         false,  // return_logprobs
                         generators};

    auto device_history_inputs          = inputs;
    device_history_inputs.token_ids     = token_ids.to(torch::kCUDA);
    device_history_inputs.logits        = inputs.logits.clone();
    device_history_inputs.cum_log_probs = inputs.cum_log_probs.clone();
    auto output                         = sampler_->forward(inputs).token_ids.cpu().contiguous();

    ASSERT_EQ(output.size(0), (int64_t)batch_size_out);
    // The 2->1 group must read its original first parent, not the preceding 1->2 output row.
    EXPECT_EQ(output[2][0].item<int32_t>(), 2);
    EXPECT_EQ(output[2][1].item<int32_t>(), 2);
    // A 1->1 subgroup still needs its history copied when another group requires separate output storage.
    EXPECT_EQ(output[3][0].item<int32_t>(), 4);
    EXPECT_EQ(output[3][1].item<int32_t>(), 4);
    EXPECT_EQ(output[3][step].item<int32_t>(), 8);

    // MTP supplies history already on the consumer stream. Skipping its
    // redundant upload must preserve mixed-beam independent-output semantics.
    auto device_result = sampler_->forward(device_history_inputs);
    EXPECT_NE(device_result.token_ids.data_ptr(), device_history_inputs.token_ids.data_ptr());
    EXPECT_TRUE(torch::equal(output, device_result.token_ids.cpu().contiguous()));
}

TEST_F(SamplerTest, testGeneralSampling) {
    size_t batch_size = 5;
    size_t vocab_size = 8;

    // logits must be on CUDA (GPU kernel operates on them directly)
    auto logits = torch::tensor(
                      {
                          0.1f, 0.1f, 0.2f,  0.1f,  0.3f,  0.1f,  0.1f,  0.1f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f, 6.0f,
                          7.0f, 8.0f, 0.01f, 0.02f, 0.04f, 0.08f, 0.16f, 0.32f, 0.64f, 0.01f, 0.1f,  0.2f,  0.3f, 0.4f,
                          0.5f, 0.6f, 0.7f,  0.8f,  0.99f, 0.98f, 0.97f, 0.96f, 0.95f, 0.94f, 0.93f, 0.92f,
                      },
                      torch::kFloat32)
                      .reshape({(int64_t)batch_size, (int64_t)vocab_size})
                      .to(torch::kCUDA);

    int32_t step = 3;  // also max_input_length - 1

    // token_ids cloned to GPU inside sampleGreedy — regular CPU memory is fine
    auto output_token_ids = torch::tensor({1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 2, 0, 3, 3, 0, 0, 8, 2, 3, 0}, torch::kInt32)
                                .reshape({(int64_t)batch_size, step + 1});

    auto input_lengths    = torch::tensor({1, 1, 1, 1, 1}, torch::kInt32);
    auto sequence_lengths = torch::tensor({1, 2, 3, 2, 3}, torch::kInt32);

    // num_beams accessed from host code only — regular CPU
    auto num_beams_in  = torch::tensor({1L, 1L, 1L, 1L, 1L}, torch::kLong);
    auto num_beams_out = torch::tensor({1L, 1L, 1L, 1L, 1L}, torch::kLong);

    // cum_log_probs on CUDA (kernel writes to it)
    auto cum_log_probs = torch::tensor({-1.0f, -1.0f, -1.0f, -1.0f, -1.0f}, torch::kFloat32).to(torch::kCUDA);

    // These are read from CPU (std::any_of) AND from GPU (flashinfer kernels).
    // Must use pin_memory() for dual CPU+GPU access (like original cudaMallocHost).
    auto repetition_penalty = torch::tensor({1.0f, 1.0f, 1.0f, 10000.0f, 1.0f}, torch::kFloat32).pin_memory();
    auto presence_penalty   = torch::tensor({0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, torch::kFloat32).pin_memory();
    auto frequency_penalty  = torch::tensor({0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, torch::kFloat32).pin_memory();

    auto top_k       = torch::tensor({1, 4, 0, 0, 8}, torch::kInt32).pin_memory();
    auto top_p       = torch::tensor({0.0f, 0.0f, 0.001f, 0.99f, 0.9f}, torch::kFloat32).pin_memory();
    auto temperature = torch::tensor({0.1f, 0.001f, 0.2f, 1.0f, 100.0f}, torch::kFloat32).pin_memory();

    LogitsProcessorStatesPtr state_ptr = std::make_shared<LogitsProcessorStates>();

    std::vector<at::Generator> generator;
    generator.resize(batch_size);

    SamplerInputs inputs{
        logits.clone(),
        output_token_ids.clone(),
        input_lengths,
        sequence_lengths,
        state_ptr,
        size_t(vocab_size),
        size_t(step),
        batch_size,
        batch_size,
        num_beams_in,
        num_beams_out,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        torch::Tensor(),  // no_repeat_ngram_size
        torch::Tensor(),  // do_sample
        torch::Tensor(),  // finished_mask
        false,            // return_original_all_probs
        false,            // skip_logits_probs_restore
        0,                // force_topk
        SamplerProbabilityOutput::NONE,
        cum_log_probs.clone(),
        false,  // return_logprobs
        generator,
    };

    auto outputs = sampler_->forward(inputs);
    std::cout << "output_token_ids: " << outputs.token_ids.cpu() << std::endl;
    std::cout << "cum_log_probs: " << outputs.cum_log_probs.cpu() << std::endl;

    auto token_ids_cpu         = outputs.token_ids.cpu().contiguous();
    auto output_token_ids_host = std::vector<int32_t>(token_ids_cpu.data_ptr<int32_t>(),
                                                      token_ids_cpu.data_ptr<int32_t>() + token_ids_cpu.numel());
    auto cum_log_probs_cpu     = outputs.cum_log_probs.cpu().contiguous();
    auto cum_log_probs_host    = std::vector<float>(cum_log_probs_cpu.data_ptr<float>(),
                                                 cum_log_probs_cpu.data_ptr<float>() + cum_log_probs_cpu.numel());

    ASSERT_EQ(output_token_ids_host[3], 4);
    ASSERT_EQ(output_token_ids_host[7], 7);
    ASSERT_EQ(output_token_ids_host[11], 6);

    vector<vector<int32_t>> token_bins(batch_size, vector<int32_t>(vocab_size, 0));
    for (size_t i = 0; i < 10000; i++) {

        inputs.token_ids     = output_token_ids.clone();
        inputs.cum_log_probs = cum_log_probs.clone();
        outputs              = sampler_->forward(inputs);

        token_ids_cpu         = outputs.token_ids.cpu().contiguous();
        output_token_ids_host = std::vector<int32_t>(token_ids_cpu.data_ptr<int32_t>(),
                                                     token_ids_cpu.data_ptr<int32_t>() + token_ids_cpu.numel());
        for (size_t j = 0; j < batch_size; j++) {
            token_bins[j][output_token_ids_host[j * (step + 1) + step]]++;
        }
    }

    for (size_t j = 0; j < batch_size; j++) {
        printf("batch %ld: ", j);
        for (size_t i = 0; i < vocab_size; i++) {
            printf("%d ", token_bins[j][i]);
        }
        printf("\n");
    }

    for (size_t i = 0; i < batch_size; i++) {
        // batch 0: top_k = 1 and top_p = 0, expected deterministic output
        if (i == 0) {
            for (size_t token = 0; token < vocab_size; token++) {
                if (token == 4) {
                    ASSERT_EQ(token_bins[i][token], 10000);
                } else {
                    ASSERT_EQ(token_bins[i][token], 0);
                }
            }
        }

        // batch 1: top_k > 1 but temperature = 0.001, expected deterministic output
        if (i == 1) {
            for (size_t token = 0; token < vocab_size; token++) {
                if (token == 7) {
                    ASSERT_EQ(token_bins[i][token], 10000);
                } else {
                    ASSERT_EQ(token_bins[i][token], 0);
                }
            }
        }
        // batch 2: top_k = 0 and top_p = 0.001, expected deterministic output
        if (i == 2) {
            for (size_t token = 0; token < vocab_size; token++) {
                if (token == 6) {
                    ASSERT_EQ(token_bins[i][token], 10000);
                } else {
                    ASSERT_EQ(token_bins[i][token], 0);
                }
            }
        }
        // batch 3: top_p = 0.99 thus expected evenly distributed output
        // except token 3, which should be punished by repetition_penalty
        if (i == 3) {
            ASSERT_EQ(std::min_element(token_bins[3].begin(), token_bins[3].end()), token_bins[3].begin() + 3);
            for (size_t token = 0; token < vocab_size; token++) {
                ASSERT_GT(token_bins[i][token], 0);
            }
        }
    }

    auto oracle_logits        = logits[1].repeat({(int64_t)batch_size, 1});
    inputs.repetition_penalty = inputs.presence_penalty = inputs.frequency_penalty = torch::Tensor();
    inputs.temperature = torch::ones({(int64_t)batch_size}, torch::kFloat32).pin_memory();
    inputs.top_k.fill_(1);
    inputs.top_p.fill_(1.0f);
    inputs.logits             = oracle_logits.clone();
    inputs.probability_output = SamplerProbabilityOutput::DENSE;
    inputs.cum_log_probs      = torch::Tensor();
    outputs                   = sampler_->forward(inputs);
    auto point_mass           = torch::zeros_like(oracle_logits).scatter_(1, oracle_logits.argmax(-1, true), 1.0);
    EXPECT_TRUE(torch::allclose(outputs.all_probs, point_mass));

    // The original top token probability exceeds 0.5, so top-p keeps only token 7.
    inputs.top_k.fill_(2);
    inputs.top_p.fill_(0.5f);
    inputs.token_ids     = output_token_ids.clone();
    inputs.logits        = oracle_logits.clone();
    inputs.vocab_size    = 4;
    inputs.cum_log_probs = cum_log_probs.clone();
    outputs              = sampler_->forward(inputs);
    EXPECT_TRUE(
        torch::equal(outputs.token_ids.select(1, step).cpu(), torch::full({(int64_t)batch_size}, 7, torch::kInt32)));
    EXPECT_TRUE(torch::equal(outputs.all_probs, point_mass.slice(1, 0, 4)));
    EXPECT_TRUE(torch::allclose(outputs.cum_log_probs, cum_log_probs));
}
