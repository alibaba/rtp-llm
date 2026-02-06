#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include <torch/torch.h>
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"

using namespace std;
using namespace rtp_llm;

class CudaSamplerTest: public DeviceTestBase {
public:
protected:
};

TEST_F(CudaSamplerTest, testTopK) {
    size_t    batch_size   = 4;
    BufferPtr logits       = createBuffer<float>({batch_size, 10},
                                                 {
                                               0,     0,     0,       0.1, 0.2, 0.3,   0, 0,      0,    0.01,
                                               0.987, 0.887, 0.99999, 0.1, 0.2, 0.3,   0, 0,      0.99, 0.989,
                                               0.221, 0,     0,       0.1, 0.2, 0.321, 0, 0.4432, 0.44, 0.01,
                                               0.221, 0,     0,       0.1, 0.2, 0.321, 0, 0.4432, 0.44, 0.01,
                                           });
    size_t    step         = 5;  // also max_input_length
    BufferPtr eos_token_id = createBuffer<int32_t>({1}, {2});
    // BufferPtr finished = createBuffer<bool>({1}, {0});
    BufferPtr output_token_ids =
        createBuffer<int32_t>({batch_size, step + 1},
                              {
                                  100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                              });

    // TODO: test lengths
    BufferPtr input_lengths    = createBuffer<int32_t>({4}, {5, 5, 5, 5});
    BufferPtr sequence_lengths = createBuffer<int32_t>({4}, {-1, -1, -1, -1});
    BufferPtr cum_log_probs    = createBuffer<float>({4}, {-1.0, -2.0, -3.0, -3.0});
    // BufferPtr rand_seed = createBuffer<uint64_t>({4}, {1, 2, 3, 123}, AllocationType::HOST);
    // ROCm uses a different random function vs CUDA;
    // Here define the appropriate random seed to generate pseudo-random numbers which is close with CUDA
    BufferPtr rand_seed = createBuffer<uint64_t>({4}, {1, 80, 90, 123}, AllocationType::HOST);

    auto top_k      = createBuffer<uint32_t>({4}, {1, 1, 2, 2}, AllocationType::HOST);
    auto top_p      = createBuffer<float>({4}, {0.0, 0.0, 0.0, 0.0}, AllocationType::HOST);
    auto temperture = createBuffer<float>({4}, {1.0, 1.0, 10.0, 10.0}, AllocationType::HOST);

    GreedyParams params({*logits,
                         *sequence_lengths,
                         *input_lengths,
                         *output_token_ids,
                         step,
                         *top_k,
                         *top_p,
                         *temperture,
                         *rand_seed,
                         nullopt,
                         nullopt,
                         nullopt,
                         false,
                         nullopt,
                         *cum_log_probs,
                         nullopt,
                         nullopt});
    device_->sampleGreedy(params);
    check_cuda_error();

    // printBuffer<int32_t>(*output_token_ids, "output_token_ids");
    // printBuffer<float>(*cum_log_probs, "cum_log_probs");
    auto output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
    auto cum_log_probs_host    = getBufferValues<float>(*cum_log_probs);
    ASSERT_EQ(output_token_ids_host[5], 5);
    ASSERT_EQ(output_token_ids_host[11], 2);
    ASSERT_EQ(output_token_ids_host[17], 8);
    ASSERT_EQ(output_token_ids_host[23], 7);
    ASSERT_NEAR(cum_log_probs_host[2], -3.693, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[3], -3.693, 1e-3);
}

TEST_F(CudaSamplerTest, testTopP) {
    size_t    batch_size   = 4;
    BufferPtr logits       = createBuffer<float>({batch_size, 10},
                                                 {
                                               0,     0,     0,       0.1, 0.2, 0.3,   0, 0,      0,    0.01,
                                               0.987, 0.887, 0.99999, 0.1, 0.2, 0.3,   0, 0,      0.99, 0.989,
                                               0.221, 0,     0,       0.1, 0.2, 0.321, 0, 0.4432, 0.44, 0.01,
                                               0.221, 0,     0,       0.1, 0.2, 0.321, 0, 0.4432, 0.44, 0.01,
                                           });
    size_t    step         = 5;  // also max_input_length
    BufferPtr eos_token_id = createBuffer<int32_t>({1}, {2});
    // BufferPtr finished = createBuffer<bool>({1}, {0});
    BufferPtr output_token_ids =
        createBuffer<int32_t>({batch_size, step + 1},
                              {
                                  100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                              });

    BufferPtr input_lengths    = createBuffer<int32_t>({4}, {5, 5, 5, 5});
    BufferPtr sequence_lengths = createBuffer<int32_t>({4}, {-1, -1, -1, -1});
    BufferPtr cum_log_probs    = createBuffer<float>({4}, {-1.0, -2.0, -3.0, -3.0});
    // BufferPtr rand_seed = createBuffer<uint64_t>({4}, {1, 2, 3, 123}, AllocationType::HOST);
    // ROCm uses a different random function vs CUDA;
    // Here define the appropriate random seed to generate pseudo-random numbers which is close with CUDA
    BufferPtr rand_seed = createBuffer<uint64_t>({4}, {1, 80, 90, 123}, AllocationType::HOST);

    auto top_k      = createBuffer<uint32_t>({4}, {0, 0, 0, 0}, AllocationType::HOST);
    auto top_p      = createBuffer<float>({4}, {0.01, 0.7, 0.001, 0.9}, AllocationType::HOST);
    auto temperture = createBuffer<float>({4}, {0.01, 0.5, 0.9, 0.9}, AllocationType::HOST);

    GreedyParams params({*logits,
                         *sequence_lengths,
                         *input_lengths,
                         *output_token_ids,
                         step,
                         *top_k,
                         *top_p,
                         *temperture,
                         *rand_seed,
                         nullopt,
                         nullopt,
                         nullopt,
                         false,
                         nullopt,
                         *cum_log_probs,
                         nullopt,
                         nullopt});
    device_->sampleGreedy(params);
    check_cuda_error();

    // printBuffer<int32_t>(*output_token_ids, "output_token_ids");
    // printBuffer<float>(*cum_log_probs, "cum_log_probs");
    auto output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
    auto cum_log_probs_host    = getBufferValues<float>(*cum_log_probs);

    ASSERT_EQ(output_token_ids_host[5], 5);
    ASSERT_EQ(output_token_ids_host[11], 8);
    ASSERT_EQ(output_token_ids_host[17], 7);
    ASSERT_EQ(output_token_ids_host[23], 0);
    ASSERT_NEAR(cum_log_probs_host[0], -1.0, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[1], -3.745, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[2], -5.02131, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[3], -5.2682, 1e-3);

    for (int i = 0; i < 100; i++) {
        device_->sampleGreedy(params);
        // printBuffer<int32_t>(*output_token_ids, "output_token_ids");
    }
}

TEST_F(CudaSamplerTest, testRandom) {
    size_t    batch_size   = 1;
    size_t    vocab_size   = 10;
    BufferPtr logits       = createBuffer<float>({batch_size, vocab_size},
                                                 {
                                               0.987,
                                               0.887,
                                               0.99999,
                                               0.1,
                                               0.2,
                                               0.3,
                                               0,
                                               0,
                                               0.99,
                                               0.989,
                                           });
    size_t    step         = 5;  // also max_input_length
    BufferPtr eos_token_id = createBuffer<int32_t>({1}, {2});
    // BufferPtr finished = createBuffer<bool>({1}, {0});
    BufferPtr output_token_ids = createBuffer<int32_t>({batch_size, step + 1},
                                                       {
                                                           1,
                                                           1,
                                                           0,
                                                           0,
                                                           0,
                                                           0,
                                                       });

    BufferPtr input_lengths    = createBuffer<int32_t>({1}, {5});
    BufferPtr sequence_lengths = createBuffer<int32_t>({1}, {-1});
    BufferPtr cum_log_probs    = createBuffer<float>({1}, {-1.0});
    BufferPtr rand_seed        = createBuffer<uint64_t>({1}, {1}, AllocationType::HOST);

    auto top_k      = createBuffer<uint32_t>({1}, {0}, AllocationType::HOST);
    auto top_p      = createBuffer<float>({1}, {0.5f}, AllocationType::HOST);
    auto temperture = createBuffer<float>({1}, {0.2}, AllocationType::HOST);

    GreedyParams params({*logits,
                         *sequence_lengths,
                         *input_lengths,
                         *output_token_ids,
                         step,
                         *top_k,
                         *top_p,
                         *temperture,
                         *rand_seed,
                         nullopt,
                         nullopt,
                         nullopt,
                         false,
                         nullopt,
                         *cum_log_probs,
                         nullopt,
                         nullopt});
    device_->sampleGreedy(params);
    check_cuda_error();

    // printBuffer<int32_t>(*output_token_ids, "output_token_ids");
    auto output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);

    std::vector<size_t> counts(vocab_size, 0);
    for (int i = 0; i < 10000; i++) {
        rand_seed->data<uint64_t>()[0] = i * 100;
        device_->sampleGreedy(params);
        // printBuffer<int32_t>(*output_token_ids, "output_token_ids");
        output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
        counts[output_token_ids_host[5]]++;
    }
    for (int i = 0; i < vocab_size; i++) {
        // cout << i << ": " << counts[i] << endl;
    }
    EXPECT_GE(counts[0], 1000);
    EXPECT_GE(counts[1], 1000);
    EXPECT_GE(counts[2], 1000);
    EXPECT_GE(counts[8], 1000);
    EXPECT_GE(counts[9], 1000);

    top_k->data<uint32_t>()[0] = 4;
    top_p->data<float>()[0]    = 0.0f;
    counts                     = std::vector<size_t>(vocab_size, 0);
    for (int i = 0; i < 10000; i++) {
        rand_seed->data<uint64_t>()[0] += i * 100;
        device_->sampleGreedy(params);
        // printBuffer<int32_t>(*output_token_ids, "output_token_ids");
        output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
        counts[output_token_ids_host[5]]++;
    }
    std::unordered_set<size_t> expected = {1, 2, 8, 9};
    for (int i = 0; i < vocab_size; i++) {
        // cout << i << ": " << counts[i] << endl;
        if (expected.find(i) != expected.end()) {
            EXPECT_GE(counts[i], 1000);
        } else {
            EXPECT_EQ(counts[i], 0);
        }
    }
}
