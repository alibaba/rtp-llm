#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

using namespace std;
using namespace fastertransformer;

class CudaSamplerTest: public CudaDeviceTestBase {
public:
};

TEST_F(CudaSamplerTest, testTopK) {
    size_t batch_size = 4;
    auto logits = createBuffer<float>({batch_size, 10}, {
        0, 0, 0, 0.1, 0.2, 0.3, 0, 0, 0, 0.01,
        0.987, 0.887, 0.99999, 0.1, 0.2, 0.3, 0, 0, 0.99, 0.989,
        0.221, 0, 0, 0.1, 0.2, 0.321, 0, 0.4432, 0.44, 0.01,
        0.221, 0, 0, 0.1, 0.2, 0.321, 0, 0.4432, 0.44, 0.01,
    });
    int32_t step = 5; // also max_input_length
    auto eos_token_id = createBuffer<int32_t>({1}, {2});
    // auto finished = createBuffer<bool>({1}, {0});
    auto output_token_ids = createBuffer<int32_t>({(uint)step + 1, batch_size}, {
        100, 1, 1, 1,
        1, 1, 0, 0,
        1, 0, 1, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        0, 0, 0, 0,
    });

    auto input_lengths = createBuffer<int32_t>({4}, {5, 5, 5});
    auto sequence_lengths = createBuffer<int32_t>({1}, {-1});
    auto cum_log_probs = createBuffer<float>({4}, {-1.0, -2.0, -3.0, -3.0});

    auto top_k = createBuffer<int32_t>({4}, {1, 1, 2, 2}, AllocationType::HOST);
    auto top_p = createBuffer<float>({4}, {0.0, 0.0, 0.0, 0.0}, AllocationType::HOST);
    auto temperture = createBuffer<float>({4}, {1.0, 1.0, 10.0, 10.0}, AllocationType::HOST);
    auto repetition_penalty = createBuffer<float>({4}, {0.0, 0.0, 0.0, 0.0}, AllocationType::HOST);
    auto length_penalty = createBuffer<float>({4}, {0.0, 0.0, 0.0, 0.0}, AllocationType::HOST);
    auto rand_seed = createBuffer<int64_t>({4}, {1, 2, 3, 123}, AllocationType::HOST);

    GreedyParams params({
        *logits, *input_lengths, *output_token_ids,
        *top_k, *top_p, *temperture, *rand_seed,
        *repetition_penalty, *length_penalty,
        *cum_log_probs, nullopt
    });
    device_->sampleGreedy(params);
    sync_check_cuda_error();

    printBuffer<int32_t>(*output_token_ids, "output_token_ids");
    printBuffer<float>(*cum_log_probs, "cum_log_probs");
    auto output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
    auto cum_log_probs_host = getBufferValues<float>(*cum_log_probs);
    ASSERT_EQ(output_token_ids_host[20], 5);
    ASSERT_EQ(output_token_ids_host[21], 2);
    ASSERT_EQ(output_token_ids_host[22], 8);
    ASSERT_EQ(output_token_ids_host[23], 7);
    ASSERT_NEAR(cum_log_probs_host[2], -3.69475, 1e-4);
    ASSERT_NEAR(cum_log_probs_host[3], -3.69155, 1e-4);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
