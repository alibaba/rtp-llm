#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

using namespace std;
using namespace fastertransformer;

class CudaSamplerTest: public CudaDeviceTestBase {
public:

    void SetUp() override {
        CudaDeviceTestBase::SetUp();
    }
    void TearDown() override {
        CudaDeviceTestBase::TearDown();
    }
};

TEST_F(CudaSamplerTest, testTopK) {
    auto logits = createBuffer<float>({1, 10}, {0, 0, 0, 0.1, 0.2, 0.3, 0, 0, 0, 0.01});
    int32_t step = 5; // also max_input_length
    auto eos_token_id = createBuffer<int32_t>({1}, {2});
    // auto finished = createBuffer<bool>({1}, {0});
    auto output_token_ids = createBuffer<int32_t>({6, 1}, {100, 1, 1, 1, 1, 0});
    auto output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);

    auto input_lengths = createBuffer<int32_t>({1}, {5});
    auto sequence_lengths = createBuffer<int32_t>({1}, {-1});
    auto cum_log_probs = createBuffer<float>({1}, {-1.0});

    auto top_k = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    auto top_p = createBuffer<float>({1}, {0.0}, AllocationType::HOST);
    auto temperture = createBuffer<float>({1}, {1.0}, AllocationType::HOST);
    auto repetition_penalty = createBuffer<float>({1}, {0.0}, AllocationType::HOST);
    auto length_penalty = createBuffer<float>({1}, {0.0}, AllocationType::HOST);
    auto rand_seed = createBuffer<int64_t>({1}, {0}, AllocationType::HOST);

    GreedyParams params({
        *logits, *input_lengths, *output_token_ids,
        *top_k, *top_p, *temperture, *rand_seed,
        *repetition_penalty, *length_penalty,
        *cum_log_probs, nullopt
    });
    device_->sampleGreedy(params);
    sync_check_cuda_error();
    output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
    for (int i = 0; i < 6; i++) {
        cout << output_token_ids_host[i] << " ";
    }
    cout << "ids." << endl;
    ASSERT_EQ(output_token_ids_host[5], 5);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
