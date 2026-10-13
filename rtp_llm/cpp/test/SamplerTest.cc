#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/test/ModelTestUtil.h"

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

TEST_F(SamplerTest, testDynamicGreedySamplingBuffersGrow) {
    Sampler sampler(SamplerInitParams{1, false});

    sampler.ensureGreedySamplingBuffers(2);

    ASSERT_EQ(sampler.max_batch_size_, 2);
    for (const auto& slot : sampler.greedy_sampling_buffer_slots_) {
        ASSERT_EQ(slot.buffers.max_batch_size, 2);
        ASSERT_GE(slot.buffers.seed_host.numel(), 2);
        ASSERT_GE(slot.buffers.offset_host.numel(), 2);
        ASSERT_GE(slot.buffers.output_ids_ptrs_host.numel(), 2);
    }

    GreedySamplingBuffers* used_slots[Sampler::kGreedySamplingBufferSlots] = {};
    for (size_t i = 0; i < Sampler::kGreedySamplingBufferSlots; ++i) {
        auto& buffers = sampler.nextGreedySamplingBuffers(2);
        used_slots[i] = &buffers;
        ASSERT_EQ(&sampler.greedy_sampling_buffer_slots_[i].buffers, &buffers);
        sampler.markGreedySamplingBufferReady();
        ASSERT_TRUE(sampler.greedy_sampling_buffer_slots_[i].ready_event);
    }

    auto& reused_buffers = sampler.nextGreedySamplingBuffers(2);
    ASSERT_EQ(used_slots[0], &reused_buffers);
    ASSERT_FALSE(sampler.greedy_sampling_buffer_slots_[0].ready_event);
    sampler.markGreedySamplingBufferReady();
}

TEST_F(SamplerTest, testFixedGreedySamplingBuffersRejectGrow) {
    Sampler sampler(SamplerInitParams{1, true});

    EXPECT_THROW(sampler.ensureGreedySamplingBuffers(2), rtp_llm::RTPException);
}

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
                         cum_log_probs,
                         torch::Tensor(),
                         generators};

    auto output = sampler_->forward(inputs).token_ids.cpu().contiguous();

    ASSERT_EQ(output.size(0), (int64_t)batch_size_out);
    // The 2->1 group must read its original first parent, not the preceding 1->2 output row.
    EXPECT_EQ(output[2][0].item<int32_t>(), 2);
    EXPECT_EQ(output[2][1].item<int32_t>(), 2);
    // A 1->1 subgroup still needs its history copied when another group requires separate output storage.
    EXPECT_EQ(output[3][0].item<int32_t>(), 4);
    EXPECT_EQ(output[3][1].item<int32_t>(), 4);
    EXPECT_EQ(output[3][step].item<int32_t>(), 8);
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
        cum_log_probs.clone(),
        torch::Tensor(),  // all_probs
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
    inputs.logits        = oracle_logits.clone();
    inputs.all_probs     = torch::empty_like(oracle_logits);
    inputs.cum_log_probs = torch::Tensor();
    outputs              = sampler_->forward(inputs);
    auto point_mass      = torch::zeros_like(oracle_logits).scatter_(1, oracle_logits.argmax(-1, true), 1.0);
    EXPECT_TRUE(torch::allclose(outputs.all_probs, point_mass));

    // The original top token probability exceeds 0.5, so top-p keeps only token 7.
    inputs.top_k.fill_(2);
    inputs.top_p.fill_(0.5f);
    inputs.token_ids     = output_token_ids.clone();
    inputs.logits        = oracle_logits.clone();
    inputs.all_probs     = torch::empty({(int64_t)batch_size, 4}, logits.options());
    inputs.cum_log_probs = cum_log_probs.clone();
    outputs              = sampler_->forward(inputs);
    EXPECT_TRUE(
        torch::equal(outputs.token_ids.select(1, step).cpu(), torch::full({(int64_t)batch_size}, 7, torch::kInt32)));
    EXPECT_TRUE(torch::equal(outputs.all_probs, point_mass.slice(1, 0, 4)));
    EXPECT_TRUE(torch::allclose(outputs.cum_log_probs, cum_log_probs));
}
