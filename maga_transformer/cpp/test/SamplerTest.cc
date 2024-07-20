#include "src/fastertransformer/devices/testing/TestBase.h"
// #include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "maga_transformer/cpp/test/ModelTestUtil.h"

using namespace std;
using namespace rtp_llm;
using namespace fastertransformer;
// TODO: make this test device-independent
class SamplerTest : public DeviceTestBase {
public:

    void SetUp() override {
        DeviceTestBase::SetUp();
        const auto eos_id = 1;
        sampler_.reset(new Sampler({device_, eos_id, 256}));
    }

protected:
    std::unique_ptr<Sampler> sampler_;
};

TEST_F(SamplerTest, testGeneralSampling) {
    size_t batch_size = 5;
    size_t vocab_size = 8;
    BufferPtr logits = createBuffer<float>({batch_size, vocab_size}, {
        0.1, 0.1, 0.2, 0.1, 0.3, 0.1, 0.1, 0.1,
        1, 2, 3, 4, 5, 6, 7, 8,
        0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.01,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92,
    });
    BufferPtr original_logits_host = device_->clone({*logits, AllocationType::HOST});

    int32_t step = 3; // also max_input_length - 1
    // BufferPtr finished = createBuffer<bool>({1}, {0});
    BufferPtr output_token_ids = createBuffer<int32_t>({batch_size, (uint)step + 1}, {
        1, 0, 0, 0,
        1, 1, 0, 0,
        1, 1, 2, 0,
        3, 3, 0, 0,
        8, 2, 3, 0,
    }, AllocationType::HOST);

    BufferPtr input_lengths = createBuffer<int32_t>({batch_size}, {1, 1, 1, 1, 1}, AllocationType::HOST);
    BufferPtr sequence_lengths = createBuffer<int32_t>({batch_size}, {1, 2, 3, 2, 3}, AllocationType::HOST);
    BufferPtr num_beams = createBuffer<uint64_t>({batch_size}, {1, 1, 1, 1, 1}, AllocationType::HOST);

    BufferPtr cum_log_probs = createBuffer<float>({batch_size}, {-1, -1, -1, -1, -1});
    BufferPtr rand_seed = createBuffer<uint64_t>({batch_size}, {0, 0, 0, 0, 0}, AllocationType::HOST);
    BufferPtr repetition_penalty = createBuffer<float>({batch_size}, {1.0f, 1.0f, 1.0f, 10000.0f, 1.0f}, AllocationType::HOST);
    BufferPtr min_length = createBuffer<int32_t>({batch_size}, {0, 0, 0, 0, 10}, AllocationType::HOST);

    auto kv_blocks = createBuffer<int64_t>({batch_size, 1}, {0, 0, 0, 0, 0}, AllocationType::HOST);

    auto top_k = createBuffer<uint32_t>({batch_size}, {1, 4, 0, 0, 8}, AllocationType::HOST);
    auto top_p = createBuffer<float>({batch_size}, {0.0, 0.0, 0.001, 0.99, 0.9}, AllocationType::HOST);
    auto temperature = createBuffer<float>({batch_size}, {0.1, 0.001, 0.2, 1.0, 100.0f}, AllocationType::HOST);

    SamplerInputs inputs {
        move(logits),
        device_->clone({*output_token_ids, AllocationType::HOST}),
        move(input_lengths),
        move(sequence_lengths),
        size_t(step),
        batch_size,
        move(num_beams),
        move(top_k),
        move(top_p),
        move(temperature),
        rand_seed,
        repetition_penalty,
        min_length,
        move(kv_blocks),
        device_->clone({*cum_log_probs}),
    };

    auto outputs = sampler_->forward(inputs);
    printBuffer<int32_t>(*outputs.token_ids, "output_token_ids");
    printBuffer<float>(*outputs.cum_log_probs, "cum_log_probs");

    auto output_token_ids_host = getBufferValues<int32_t>(*outputs.token_ids);
    auto cum_log_probs_host = getBufferValues<float>(*outputs.cum_log_probs);

    ASSERT_EQ(output_token_ids_host[3], 4);
    ASSERT_EQ(output_token_ids_host[7], 7);
    ASSERT_EQ(output_token_ids_host[11], 6);

    vector<vector<int32_t>> token_bins(batch_size, vector<int32_t>(vocab_size, 0));
    for (size_t i = 0; i < 10000; i++) {
        for (size_t j = 0; j < batch_size; j++) {
            rand_seed->data<uint64_t>()[j] = i * 1000 + j;
        }

        inputs.token_ids = device_->clone({*output_token_ids, AllocationType::HOST});
        inputs.cum_log_probs = device_->clone({*cum_log_probs});
        outputs = sampler_->forward(inputs);

        // printf("i=%d  ", i);
        // printBuffer<int32_t>(*outputs.token_ids, "output_token_ids");
        output_token_ids_host = getBufferValues<int32_t>(*outputs.token_ids);
        for (size_t j = 0; j < batch_size; j++) {
            token_bins[j][output_token_ids_host[j * (step + 1) + step]]++;
        }
    }

    for (size_t j = 0; j < batch_size; j++) {
        printf("batch %d: ", j);
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
        // batch 4: temperature = 100.0, expected uniform distribution
        // except token 1, which is eos token and should be punished by min_length
        if (i == 4) {
            for (size_t token = 0; token < vocab_size; token++) {
                if (token == 1) {
                    ASSERT_EQ(token_bins[4][token], 0);
                } else {
                    ASSERT_GT(token_bins[4][token], 0);
                }
            }
        }
    }
}
