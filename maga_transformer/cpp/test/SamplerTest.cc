#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "maga_transformer/cpp/test/ModelTestUtil.h"

using namespace std;
using namespace rtp_llm;

// TODO: make this test device-independent
class SamplerTest : public DeviceTestBase {
public:

    void SetUp() override {
        DeviceTestBase::SetUp();
        sampler_.reset(new Sampler({device_}));
    }

protected:
    std::unique_ptr<Sampler> sampler_;
};

TEST_F(SamplerTest, testSimple) {
    size_t batch_size = 5;
    BufferPtr logits = createBuffer<float>({batch_size, 8}, {
        0.1, 0.1, 0.2, 0.1, 0.3, 0.1, 0.1, 0.1,
        1, 2, 3, 4, 5, 6, 7, 8,
        0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.01,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92,
    });

    int32_t step = 3; // also max_input_length
    // BufferPtr finished = createBuffer<bool>({1}, {0});
    BufferPtr output_token_ids = createBuffer<int32_t>({batch_size, (uint)step + 1}, {
        1, 0, 0, 0,
        1, 1, 0, 0,
        1, 1, 2, 0,
        2, 3, 0, 0,
        1, 2, 3, 0,
    }, AllocationType::HOST);

    BufferPtr input_lengths = createBuffer<int32_t>({batch_size}, {1, 2, 3, 2, 3}, AllocationType::HOST);
    BufferPtr sequence_lengths = createBuffer<int32_t>({batch_size}, {1, 2, -1, 2, -1}, AllocationType::HOST);
    BufferPtr num_beams = createBuffer<uint64_t>({batch_size}, {1, 1, 1, 1, 1}, AllocationType::HOST);

    BufferPtr cum_log_probs = createBuffer<float>({batch_size}, {-1, -1, -1, -1, -1});
    BufferPtr rand_seed = createBuffer<int64_t>({batch_size}, {0, 0, 0, 0, 1}, AllocationType::HOST);

    auto kv_blocks = createBuffer<int64_t>({batch_size, 1}, {0, 0, 0, 0, 0}, AllocationType::HOST);

    auto top_k = createBuffer<int32_t>({batch_size}, {1, 4, 0, 0, 10}, AllocationType::HOST);
    auto top_p = createBuffer<float>({batch_size}, {0.0, 0.0, 0.001, 0.99, 0.7}, AllocationType::HOST);
    auto temperature = createBuffer<float>({batch_size}, {0.01, 0.01, 0.9, 0.7, 10}, AllocationType::HOST);

    SamplerInputs inputs {
        move(logits),
        move(output_token_ids),
        move(input_lengths),
        move(sequence_lengths),
        size_t(step),
        batch_size,
        move(num_beams),
        move(top_k),
        move(top_p),
        move(temperature),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        move(kv_blocks),
        move(cum_log_probs)
    };

    auto outputs = sampler_->forward(inputs);
    printBuffer<int32_t>(*outputs.token_ids, "output_token_ids");
    printBuffer<float>(*outputs.cum_log_probs, "cum_log_probs");

    auto output_token_ids_host = getBufferValues<int32_t>(*outputs.token_ids);
    auto cum_log_probs_host = getBufferValues<float>(*outputs.cum_log_probs);
    ASSERT_EQ(output_token_ids_host[3], 4);
    ASSERT_EQ(output_token_ids_host[7], 7);
    ASSERT_EQ(output_token_ids_host[11], 6);
    ASSERT_EQ(output_token_ids_host[15], 3);
    ASSERT_EQ(output_token_ids_host[19], 4);

    ASSERT_NEAR(cum_log_probs_host[0], -1.0, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[1], -1.0, 1e-3);

}

