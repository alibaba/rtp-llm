#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaTestUtils.h"
#include "rtp_llm/cpp/kernels/banRepeatNgram.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
using namespace std;
using namespace rtp_llm;

class CudaSamplerTest: public DeviceTestBase {
public:
protected:
};

TEST_F(CudaSamplerTest, testFlashinferKernelTopK1) {
    DeviceInitParams device_init_params;
    device_init_params.sampler_config.enable_flashinfer_sample_kernel = true;
    device_                                                           = new CudaDevice(device_init_params);
    device_->init();

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
    BufferPtr output_token_ids =
        createBuffer<int32_t>({batch_size, step + 1},
                              {
                                  100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                              });

    BufferPtr sequence_lengths = createBuffer<int32_t>({4}, {5, 5, 5, 5});
    BufferPtr input_lengths    = createBuffer<int32_t>({4}, {-1, -1, -1, -1});

    auto top_k      = createBuffer<uint32_t>({4}, {1, 1, 1, 1}, AllocationType::HOST);
    auto top_p      = createBuffer<float>({4}, {1.0, 1.0, 1.0, 1.0}, AllocationType::HOST);
    auto temperture = createBuffer<float>({4}, {1.0, 10.0, 1.0, 10.0}, AllocationType::HOST);

    GreedyParams params({
        *logits,
        *input_lengths,
        *sequence_lengths,
        *output_token_ids,
        step,
        *top_k,
        *top_p,
        *temperture,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
    });
    auto         greedy_output = device_->sampleGreedy(params);
    check_cuda_error();
    auto output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
    ASSERT_EQ(output_token_ids_host[5], 5);
    ASSERT_EQ(output_token_ids_host[11], 2);
    ASSERT_EQ(output_token_ids_host[17], 7);
    ASSERT_EQ(output_token_ids_host[23], 7);
}

TEST_F(CudaSamplerTest, testFlashinferKernelTopK) {
    DeviceInitParams device_init_params;
    device_init_params.sampler_config.enable_flashinfer_sample_kernel = true;
    device_                                                           = new CudaDevice(device_init_params);
    device_->init();

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
    BufferPtr output_token_ids =
        createBuffer<int32_t>({batch_size, step + 1},
                              {
                                  100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                              });

    BufferPtr sequence_lengths = createBuffer<int32_t>({4}, {5, 5, 5, 5});
    BufferPtr input_lengths    = createBuffer<int32_t>({4}, {-1, -1, -1, -1});

    auto top_k      = createBuffer<uint32_t>({4}, {1, 1, 0, 2}, AllocationType::HOST);
    auto top_p      = createBuffer<float>({4}, {1.0, 1.0, 1.0, 1.0}, AllocationType::HOST);
    auto temperture = createBuffer<float>({4}, {1.0, 10.0, 1.0, 10.0}, AllocationType::HOST);

    GreedyParams params({
        *logits,
        *input_lengths,
        *sequence_lengths,
        *output_token_ids,
        step,
        *top_k,
        *top_p,
        *temperture,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
    });
    auto         greedy_output = device_->sampleGreedy(params);
    check_cuda_error();
    ASSERT_TRUE(greedy_output.success != nullptr);
    ASSERT_EQ(greedy_output.success->size(), 4);
    // printbuffer<int32_t>(*output_token_ids, "output_token_ids");
    auto         success_buffer = device_->clone({*greedy_output.success, AllocationType::HOST});
    auto         success        = success_buffer->data<bool>();
    vector<bool> expect_success{true, true, true, true};
    for (int i = 0; i < expect_success.size(); ++i) {
        ASSERT_EQ(success[i], (bool)expect_success[i]);
    }
    auto output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
    ASSERT_EQ(output_token_ids_host[5], 5);
    ASSERT_EQ(output_token_ids_host[11], 2);
    ASSERT_EQ(output_token_ids_host[17], 0);
    ASSERT_EQ(output_token_ids_host[23], 8);
}

TEST_F(CudaSamplerTest, testFlashinferKernelTopP) {
    DeviceInitParams device_init_params;
    device_init_params.sampler_config.enable_flashinfer_sample_kernel = true;
    device_                                                           = new CudaDevice(device_init_params);
    device_->init();
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
    BufferPtr output_token_ids =
        createBuffer<int32_t>({batch_size, step + 1},
                              {
                                  100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                              });

    // TODO: test lengths
    BufferPtr sequence_lengths = createBuffer<int32_t>({4}, {5, 5, 5, 5});
    BufferPtr input_lengths    = createBuffer<int32_t>({4}, {-1, -1, -1, -1});

    auto top_k      = createBuffer<uint32_t>({4}, {0, 0, 0, 0}, AllocationType::HOST);
    auto top_p      = createBuffer<float>({4}, {0.1, 0.1, 0.6, 0.8}, AllocationType::HOST);
    auto temperture = createBuffer<float>({4}, {1.0, 10.0, 1.0, 10.0}, AllocationType::HOST);

    GreedyParams params({
        *logits,
        *input_lengths,
        *sequence_lengths,
        *output_token_ids,
        step,
        *top_k,
        *top_p,
        *temperture,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
    });
    auto         greedy_output = device_->sampleGreedy(params);
    check_cuda_error();
    ASSERT_TRUE(greedy_output.success != nullptr);
    ASSERT_EQ(greedy_output.success->size(), 4);

    // printBuffer<int32_t>(*output_token_ids, "output_token_ids");
    auto         success_buffer = device_->clone({*greedy_output.success, AllocationType::HOST});
    auto         success        = success_buffer->data<bool>();
    vector<bool> expect_success{true, true, true, true};
    for (int i = 0; i < expect_success.size(); ++i) {
        ASSERT_EQ(success[i], expect_success[i]);
    }

    auto output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
    ASSERT_EQ(output_token_ids_host[5], 5);
    ASSERT_EQ(output_token_ids_host[11], 2);
    ASSERT_EQ(output_token_ids_host[17], 0);
    ASSERT_EQ(output_token_ids_host[23], 1);
}

TEST_F(CudaSamplerTest, testFlashinferKernelTopKTopP) {
    DeviceInitParams device_init_params;
    device_init_params.sampler_config.enable_flashinfer_sample_kernel = true;
    device_                                                           = new CudaDevice(device_init_params);
    device_->init();

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
    BufferPtr output_token_ids =
        createBuffer<int32_t>({batch_size, step + 1},
                              {
                                  100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                              });

    // TODO: test lengths
    BufferPtr sequence_lengths = createBuffer<int32_t>({4}, {5, 5, 5, 5});
    BufferPtr input_lengths    = createBuffer<int32_t>({4}, {-1, -1, -1, -1});
    BufferPtr cum_log_probs    = createBuffer<float>({4}, {-1.0, -2.0, -3.0, -3.0});

    auto top_k      = createBuffer<uint32_t>({4}, {1, 0, 0, 2}, AllocationType::HOST);
    auto top_p      = createBuffer<float>({4}, {0.2, 0.2, 0.6, 0.6}, AllocationType::HOST);
    auto temperture = createBuffer<float>({4}, {1.0, 10.0, 1.0, 10.0}, AllocationType::HOST);

    GreedyParams params({
        *logits,
        *input_lengths,
        *sequence_lengths,
        *output_token_ids,
        step,
        *top_k,
        *top_p,
        *temperture,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
    });
    auto         greedy_output = device_->sampleGreedy(params);
    check_cuda_error();
    ASSERT_TRUE(greedy_output.success != nullptr);
    ASSERT_EQ(greedy_output.success->size(), 4);

    // printBuffer<int32_t>(*output_token_ids, "output_token_ids");
    auto         success_buffer = device_->clone({*greedy_output.success, AllocationType::HOST});
    auto         success        = success_buffer->data<bool>();
    vector<bool> expect_success{true, true, true, true};
    for (int i = 0; i < expect_success.size(); ++i) {
        ASSERT_EQ(success[i], expect_success[i]);
    }
    auto output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
    ASSERT_EQ(output_token_ids_host[5], 5);
    ASSERT_EQ(output_token_ids_host[11], 2);
    ASSERT_EQ(output_token_ids_host[17], 0);
    ASSERT_EQ(output_token_ids_host[23], 8);
}

TEST_F(CudaSamplerTest, testFlashinferKernelFailed) {
    DeviceInitParams device_init_params;
    device_init_params.sampler_config.enable_flashinfer_sample_kernel = true;
    device_                                                           = new CudaDevice(device_init_params);
    device_->init();

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
    BufferPtr output_token_ids =
        createBuffer<int32_t>({batch_size, step + 1},
                              {
                                  100, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                              });

    // TODO: test lengths
    BufferPtr sequence_lengths = createBuffer<int32_t>({4}, {5, 5, 5, 5});
    BufferPtr input_lengths    = createBuffer<int32_t>({4}, {-1, -1, -1, -1});
    BufferPtr cum_log_probs    = createBuffer<float>({4}, {-1.0, -2.0, -3.0, -3.0});

    auto top_k      = createBuffer<uint32_t>({4}, {1, 2, 2, 2}, AllocationType::HOST);
    auto top_p      = createBuffer<float>({4}, {-1.0, -1.0, 0.6, 0.2}, AllocationType::HOST);
    auto temperture = createBuffer<float>({4}, {1.0, 10.0, 1.0, 10.0}, AllocationType::HOST);

    GreedyParams params({
        *logits,
        *input_lengths,
        *sequence_lengths,
        *output_token_ids,
        step,
        *top_k,
        *top_p,
        *temperture,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
    });
    auto         greedy_output = device_->sampleGreedy(params);
    check_cuda_error();

    ASSERT_TRUE(greedy_output.success != nullptr);
    ASSERT_EQ(greedy_output.success->size(), 4);

    // printBuffer<int32_t>(*output_token_ids, "output_token_ids");
    auto         success_buffer = device_->clone({*greedy_output.success, AllocationType::HOST});
    auto         success        = success_buffer->data<bool>();
    vector<bool> expect_success{false, false, true, true};
    for (int i = 0; i < expect_success.size(); ++i) {
        ASSERT_EQ(success[i], expect_success[i]);
    }
    auto output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
    ASSERT_EQ(output_token_ids_host[17], 7);
    ASSERT_EQ(output_token_ids_host[23], 8);
}

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
    BufferPtr sequence_lengths = createBuffer<int32_t>({4}, {5, 5, 5, 5});
    BufferPtr input_lengths    = createBuffer<int32_t>({4}, {-1, -1, -1, -1});
    BufferPtr cum_log_probs    = createBuffer<float>({4}, {-1.0, -2.0, -3.0, -3.0});
    BufferPtr rand_seed        = createBuffer<uint64_t>({4}, {1, 2, 3, 123}, AllocationType::HOST);

    auto top_k      = createBuffer<uint32_t>({4}, {1, 1, 2, 2}, AllocationType::HOST);
    auto top_p      = createBuffer<float>({4}, {0.0, 0.0, 0.0, 0.6}, AllocationType::HOST);
    auto temperture = createBuffer<float>({4}, {1.0, 1.0, 10.0, 10.0}, AllocationType::HOST);

    BufferPtr output_all_probs =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_FP32, {4, 10}, rtp_llm::AllocationType::DEVICE});
    device_->bufMemset(*output_all_probs, 0);

    GreedyParams params({
        *logits,
        *input_lengths,
        *sequence_lengths,
        *output_token_ids,
        step,
        *top_k,
        *top_p,
        *temperture,
        *rand_seed,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        *cum_log_probs,
        nullopt,
        *output_all_probs,
    });
    device_->sampleGreedy(params);
    check_cuda_error();

    printBuffer<int32_t>(*output_token_ids, "output_token_ids");
    printBuffer<float>(*cum_log_probs, "cum_log_probs");
    auto output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
    auto cum_log_probs_host    = getBufferValues<float>(*cum_log_probs);
    ASSERT_EQ(output_token_ids_host[5], 5);
    ASSERT_EQ(output_token_ids_host[11], 2);
    ASSERT_EQ(output_token_ids_host[17], 8);
    ASSERT_EQ(output_token_ids_host[23], 7);
    ASSERT_NEAR(cum_log_probs_host[2], -3.693, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[3], -3.693, 1e-3);

    printBuffer<float>(*output_all_probs, "output_all_probs");

    auto output_all_probs_host = getBufferValues<float>(*output_all_probs);
    ASSERT_VECTOR_NEAR(
        output_all_probs_host,
        std::vector<float>({0, 0, 0, 0, 0, 1, 0, 0,       0,       0, 0, 0, 1, 0, 0, 0, 0, 0,        0,        0,
                            0, 0, 0, 0, 0, 0, 0, 0.50008, 0.49992, 0, 0, 0, 0, 0, 0, 0, 0, 0.833467, 0.166533, 0}),
        1e-3);
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
    BufferPtr rand_seed        = createBuffer<uint64_t>({4}, {1, 2, 3, 123}, AllocationType::HOST);

    auto top_k      = createBuffer<uint32_t>({4}, {0, 0, 0, 0}, AllocationType::HOST);
    auto top_p      = createBuffer<float>({4}, {0.01, 0.7, 0.001, 0.9}, AllocationType::HOST);
    auto temperture = createBuffer<float>({4}, {0.01, 0.5, 0.9, 0.9}, AllocationType::HOST);

    BufferPtr output_all_probs =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_FP32, {4, 10}, rtp_llm::AllocationType::DEVICE});
    device_->bufMemset(*output_all_probs, 0);

    GreedyParams params({
        *logits,
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
        nullopt,
        *cum_log_probs,
        nullopt,
        *output_all_probs,
    });
    device_->sampleGreedy(params);
    check_cuda_error();

    printBuffer<int32_t>(*output_token_ids, "output_token_ids");
    printBuffer<float>(*cum_log_probs, "cum_log_probs");
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

    printBuffer<float>(*output_all_probs, "output_all_probs");
    auto output_all_probs_host = getBufferValues<float>(*output_all_probs);
    ASSERT_VECTOR_NEAR(
        output_all_probs_host,
        std::vector<float>({0,         0,        0,        0,        0,        0.999999, 0,        0,
                            0,         0,        0.247309, 0,        0.254418, 0,        0,        0,
                            0,         0,        0.249385, 0.248887, 0,        0,        0,        0,
                            0,         0,        0,        1,        0,        0,        0.114998, 0.0899594,
                            0.0688079, 0.100531, 0.112346, 0.128512, 0,        0.147202, 0.146679, 0.0909646}),
        1e-3);
}

TEST_F(CudaSamplerTest, testTopKTopPBatch) {
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
    BufferPtr sequence_lengths = createBuffer<int32_t>({4}, {5, 5, 5, 5});
    BufferPtr input_lengths    = createBuffer<int32_t>({4}, {-1, -1, -1, -1});
    BufferPtr cum_log_probs    = createBuffer<float>({4}, {-1.0, -2.0, -3.0, -3.0});
    BufferPtr rand_seed        = createBuffer<uint64_t>({4}, {1, 2, 3, 123}, AllocationType::HOST);

    auto top_k      = createBuffer<uint32_t>({4}, {0, 0, 2, 2}, AllocationType::HOST);
    auto top_p      = createBuffer<float>({4}, {0.01, 0.7, 0.0, 0.6}, AllocationType::HOST);
    auto temperture = createBuffer<float>({4}, {0.01, 0.5, 10.0, 10.0}, AllocationType::HOST);

    BufferPtr output_all_probs =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_FP32, {4, 10}, rtp_llm::AllocationType::DEVICE});
    device_->bufMemset(*output_all_probs, 0);

    GreedyParams params({
        *logits,
        *input_lengths,
        *sequence_lengths,
        *output_token_ids,
        step,
        *top_k,
        *top_p,
        *temperture,
        *rand_seed,
        nullopt,
        nullopt,
        nullopt,
        nullopt,
        *cum_log_probs,
        nullopt,
        *output_all_probs,
    });
    device_->sampleGreedy(params);
    check_cuda_error();

    printBuffer<int32_t>(*output_token_ids, "output_token_ids");
    printBuffer<float>(*cum_log_probs, "cum_log_probs");
    auto output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
    auto cum_log_probs_host    = getBufferValues<float>(*cum_log_probs);
    ASSERT_EQ(output_token_ids_host[5], 5);
    ASSERT_EQ(output_token_ids_host[11], 8);
    ASSERT_EQ(output_token_ids_host[17], 8);
    ASSERT_EQ(output_token_ids_host[23], 7);
    ASSERT_NEAR(cum_log_probs_host[0], -1.0, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[1], -3.745, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[2], -3.693, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[3], -3.693, 1e-3);

    printBuffer<float>(*output_all_probs, "output_all_probs");

    auto output_all_probs_host = getBufferValues<float>(*output_all_probs);
    ASSERT_VECTOR_NEAR(
        output_all_probs_host,
        std::vector<float>({0,       0, 0, 0, 0,        0.999999, 0, 0, 0, 0,        0.247309, 0, 0.254418, 0,
                            0,       0, 0, 0, 0.249385, 0.248887, 0, 0, 0, 0,        0,        0, 0,        0.50008,
                            0.49992, 0, 0, 0, 0,        0,        0, 0, 0, 0.833467, 0.166533, 0}),
        1e-3);
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
    BufferPtr output_token_ids = createBuffer<int32_t>({batch_size, step + 1}, {1, 2, 8, 1, 2, 0});

    // NOTE(wangyin): The lengths are substrated by 1 here, as it needs to add 1 in the kernel.
    // TODO(wangyin): fix this when new fmha is available.
    BufferPtr sequence_lengths = createBuffer<int32_t>({1}, {4});
    BufferPtr input_lengths    = createBuffer<int32_t>({1}, {-1});
    BufferPtr cum_log_probs    = createBuffer<float>({1}, {-1.0});
    BufferPtr rand_seed        = createBuffer<uint64_t>({1}, {1}, AllocationType::HOST);

    auto top_k                = createBuffer<uint32_t>({1}, {0}, AllocationType::HOST);
    auto top_p                = createBuffer<float>({1}, {0.5f}, AllocationType::HOST);
    auto temperture           = createBuffer<float>({1}, {0.2}, AllocationType::HOST);
    auto no_repeat_ngram_size = createBuffer<int32_t>({1}, {3}, AllocationType::HOST);

    auto         logits_input = device_->clone({*logits});
    GreedyParams params({*logits_input,
                         *input_lengths,
                         *sequence_lengths,
                         *output_token_ids,
                         step,
                         *top_k,
                         *top_p,
                         *temperture,
                         *rand_seed,
                         nullopt,
                         nullopt,
                         nullopt,
                         *no_repeat_ngram_size,
                         *cum_log_probs,
                         nullopt,
                         nullopt});
    device_->sampleGreedy(params);
    check_cuda_error();

    printBuffer<int32_t>(*output_token_ids, "output_token_ids");
    auto output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);

    std::vector<size_t> counts(vocab_size, 0);
    for (int i = 0; i < 10000; i++) {
        rand_seed->data<uint64_t>()[0] = i * 100;
        device_->copy({*logits_input, *logits});
        device_->sampleGreedy(params);
        output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
        counts[output_token_ids_host[5]]++;
    }
    // for (int i = 0; i < vocab_size; i++) {
    //     printf("counts[%d] = %ld\n", i, counts[i]);
    // }
    std::unordered_set<size_t> expected = {2, 9};
    for (int i = 0; i < vocab_size; i++) {
        if (expected.find(i) != expected.end()) {
            EXPECT_GE(counts[i], 1000);
        } else {
            EXPECT_EQ(counts[i], 0);
        }
    }

    top_k->data<uint32_t>()[0] = 4;
    top_p->data<float>()[0]    = 0.0f;
    counts                     = std::vector<size_t>(vocab_size, 0);
    for (int i = 0; i < 10000; i++) {
        rand_seed->data<uint64_t>()[0] += i * 100;
        device_->copy({*logits_input, *logits});
        device_->sampleGreedy(params);
        output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
        counts[output_token_ids_host[5]]++;
    }

    // for (int i = 0; i < vocab_size; i++) {
    //     printf("counts[%d] = %ld\n", i, counts[i]);
    // }
    expected = {0, 1, 2, 9};
    for (int i = 0; i < vocab_size; i++) {
        if (expected.find(i) != expected.end()) {
            EXPECT_GE(counts[i], 1000);
        } else {
            EXPECT_EQ(counts[i], 0);
        }
    }
}

TEST_F(CudaSamplerTest, testBanRepeatNGram) {
    const auto no_repeat_ngram_size_buf = createBuffer<int32_t>({4}, {2, 3, 2, 3});
    const auto vocab_size               = 10;

    const auto batch_size = 4;
    const auto beam_width = 1;

    BufferPtr logits = createBuffer<float>({batch_size, vocab_size},
                                           {
                                               0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                               0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1,
                                               0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1,
                                           });

    size_t    step             = 8;  // also max_input_length
    BufferPtr output_token_ids = createBuffer<int32_t>(
        {batch_size, step + 1},
        {
            0, 2, 3, 4, 5, 0, 0, 2, 0, 1, 2, 3, 3, 3, 1, 2, 3, 0, 1, 2, 1, 2, 1, 2, 1, 2, 0, 9, 8, 6, 9, 8, 0, 0, 0, 0,
        });

    // NOTE(wangyin): The lengths are substrated by 1 here, as it needs to add 1 in the kernel.
    // TODO(wangyin): fix this when new fmha is available.
    BufferPtr sequence_lengths     = createBuffer<int32_t>({4}, {7, 7, 7, 4});
    BufferPtr no_repeat_ngram_size = createBuffer<int32_t>({4}, {3, 4, 5, 2});

    const auto cuda_device = dynamic_cast<CudaDevice*>(device_);
    const auto stream      = cuda_device->getStream();

    check_cuda_error();

    std::vector<uint64_t> output_ids_ptrs(batch_size);
    for (int i = 0; i < batch_size; i++) {
        output_ids_ptrs[i] = (uint64_t)(output_token_ids->data<int32_t>() + i * (step + 1));
        // printf("output_ids_ptrs[%d] = %p\n", i, (void*)output_ids_ptrs[i]);
    }
    auto output_ids_ptrs_device = createBuffer<uint64_t>({batch_size}, output_ids_ptrs);

    tensorrt_llm::kernels::invokeBanRepeatNgram(logits->data<float>(),
                                                (int32_t const**)(output_ids_ptrs_device->data()),
                                                nullptr,  // finished_buf
                                                nullptr,  // parent_ids_buf
                                                nullptr,  // batch_slot
                                                sequence_lengths->data<int32_t>(),
                                                batch_size,
                                                beam_width,
                                                step,
                                                no_repeat_ngram_size_buf->data<int32_t>(),
                                                vocab_size,
                                                step,
                                                stream);
    check_cuda_error();

    std::vector<int32_t> expcted_ban_token_ids = {3, 3, 1, 6};
    const auto           logits_tensor         = bufferToTensor(*logits, device_);
    for (int i = 0; i < batch_size; i++) {
        auto ban_id = expcted_ban_token_ids[i];
        for (int j = 0; j < vocab_size; j++) {
            if (j == ban_id) {
                EXPECT_EQ(logits_tensor[i][j].item<float>(), -INFINITY);
            } else {
                EXPECT_GT(logits_tensor[i][j].item<float>(), 0.0f);
            }
        }
    }
}

TEST_F(CudaSamplerTest, testPenalty) {
    size_t    batch_size = 4;
    BufferPtr logits     = createBuffer<float>(
        {batch_size, 10},
        {
            0.01, 0.88, 0.92, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.01, 0.88, 0.92, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
            0.01, 0.88, 0.92, 0.1, 0.2, 0.3, 0.4, 0.1, 0.1, 0.1, 0.01, 0.88, 0.92, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        });
    size_t    step         = 5;  // also max_input_length
    BufferPtr eos_token_id = createBuffer<int32_t>({1}, {2});
    // BufferPtr finished = createBuffer<bool>({1}, {0});
    BufferPtr output_token_ids =
        createBuffer<int32_t>({batch_size, step + 1},
                              {
                                  2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0,
                              });

    // TODO: test lengths
    BufferPtr sequence_lengths = createBuffer<int32_t>({4}, {5, 5, 5, 5});
    BufferPtr input_lengths    = createBuffer<int32_t>({4}, {-1, -1, -1, -1});
    BufferPtr cum_log_probs    = createBuffer<float>({4}, {-1.0, -2.0, -3.0, -3.0});
    BufferPtr rand_seed        = createBuffer<uint64_t>({4}, {1, 2, 3, 123}, AllocationType::HOST);

    auto top_k              = createBuffer<uint32_t>({4}, {0, 0, 0, 0}, AllocationType::HOST);
    auto top_p              = createBuffer<float>({4}, {1.0, 1.0, 1.0, 1.0}, AllocationType::HOST);
    auto temperture         = createBuffer<float>({4}, {1.0, 1.0, 1.0, 1.0}, AllocationType::HOST);
    auto repetition_penalty = createBuffer<float>({4}, {2.4, 1.0, 1.0, 1.2}, AllocationType::HOST);
    auto presence_penalty   = createBuffer<float>({4}, {0, 0.6, 0, 0.3}, AllocationType::HOST);
    auto frequency_penalty  = createBuffer<float>({4}, {0, 0, 0.2, 0.1}, AllocationType::HOST);

    BufferPtr output_all_probs =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_FP32, {4, 10}, rtp_llm::AllocationType::DEVICE});
    device_->bufMemset(*output_all_probs, 0);

    GreedyParams params({*logits,
                         *input_lengths,
                         *sequence_lengths,
                         *output_token_ids,
                         step,
                         *top_k,
                         *top_p,
                         *temperture,
                         *rand_seed,
                         *repetition_penalty,
                         nullopt,
                         nullopt,
                         nullopt,
                         *cum_log_probs,
                         nullopt,
                         *output_all_probs,
                         *presence_penalty,
                         *frequency_penalty});
    device_->sampleGreedy(params);
    check_cuda_error();

    printBuffer<int32_t>(*output_token_ids, "output_token_ids");
    printBuffer<float>(*cum_log_probs, "cum_log_probs");
    auto output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
    auto cum_log_probs_host    = getBufferValues<float>(*cum_log_probs);
    ASSERT_EQ(output_token_ids_host[5], 1);
    ASSERT_EQ(output_token_ids_host[11], 7);
    ASSERT_EQ(output_token_ids_host[17], 9);
    ASSERT_EQ(output_token_ids_host[23], 6);
    ASSERT_NEAR(cum_log_probs_host[0], -3.3125, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[1], -4.16467, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[2], -5.42469, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[3], -5.24668, 1e-3);

    printBuffer<float>(*output_all_probs, "output_all_probs");

    auto output_all_probs_host = getBufferValues<float>(*output_all_probs);
    ASSERT_VECTOR_NEAR(
        output_all_probs_host,
        std::vector<float>({0.0693098, 0.0990131, 0.100677,  0.075837,  0.0838128, 0.0926275, 0.102369,  0.113135,
                            0.125034,  0.138184,  0.0703223, 0.0921197, 0.0958792, 0.0769448, 0.0850372, 0.0939806,
                            0.103865,  0.114788,  0.126861,  0.140203,  0.080888,  0.12942,   0.110285,  0.0885056,
                            0.0978138, 0.108101,  0.11947,   0.0885056, 0.0885056, 0.0885056, 0.0715989, 0.0895156,
                            0.0837425, 0.0783417, 0.0865809, 0.0956867, 0.10575,   0.116872,  0.129164,  0.142748}),
        1e-3);
}

TEST_F(CudaSamplerTest, testDoSample) {
    size_t    batch_size = 4;
    BufferPtr logits     = createBuffer<float>(
        {batch_size, 10},
        {
            0.01, 0.8, 0.98, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.01, 0.8, 0.98, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
            0.01, 0.8, 0.98, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.01, 0.8, 0.98, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        });
    size_t    step         = 5;  // also max_input_length
    BufferPtr eos_token_id = createBuffer<int32_t>({1}, {2});
    // BufferPtr finished = createBuffer<bool>({1}, {0});
    BufferPtr output_token_ids =
        createBuffer<int32_t>({batch_size, step + 1},
                              {
                                  2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 0,
                              });

    // TODO: test lengths
    BufferPtr sequence_lengths = createBuffer<int32_t>({4}, {5, 5, 5, 5});
    BufferPtr input_lengths    = createBuffer<int32_t>({4}, {-1, -1, -1, -1});
    BufferPtr cum_log_probs    = createBuffer<float>({4}, {-1.0, -2.0, -3.0, -3.0});
    BufferPtr rand_seed        = createBuffer<uint64_t>({4}, {1, 2, 3, 4}, AllocationType::HOST);

    auto top_k                            = createBuffer<uint32_t>({4}, {2, 2, 2, 2}, AllocationType::HOST);
    auto top_p                            = createBuffer<float>({4}, {1.0, 1.0, 1.0, 1.0}, AllocationType::HOST);
    auto temperture                       = createBuffer<float>({4}, {2.0, 2.0, 4.0, 4.0}, AllocationType::HOST);
    auto do_sample                        = createBuffer({4}, rtp_llm::DataType::TYPE_BOOL, AllocationType::HOST);
    *(do_sample->dataWithOffset<bool>(0)) = false;
    *(do_sample->dataWithOffset<bool>(1)) = true;
    *(do_sample->dataWithOffset<bool>(2)) = false;
    *(do_sample->dataWithOffset<bool>(3)) = true;

    BufferPtr output_all_probs =
        device_->allocateBuffer({rtp_llm::DataType::TYPE_FP32, {4, 10}, rtp_llm::AllocationType::DEVICE});
    device_->bufMemset(*output_all_probs, 0);

    GreedyParams params({*logits,
                         *input_lengths,
                         *sequence_lengths,
                         *output_token_ids,
                         step,
                         *top_k,
                         *top_p,
                         *temperture,
                         *rand_seed,
                         nullopt,
                         nullopt,
                         nullopt,
                         nullopt,
                         *cum_log_probs,
                         nullopt,
                         *output_all_probs,
                         nullopt,
                         nullopt,
                         *do_sample});
    device_->sampleGreedy(params);
    check_cuda_error();

    printBuffer<int32_t>(*output_token_ids, "output_token_ids");
    printBuffer<float>(*cum_log_probs, "cum_log_probs");

    auto output_token_ids_host = getBufferValues<int32_t>(*output_token_ids);
    auto cum_log_probs_host    = getBufferValues<float>(*cum_log_probs);
    ASSERT_EQ(output_token_ids_host[5], 1);
    ASSERT_EQ(output_token_ids_host[11], 2);
    ASSERT_EQ(output_token_ids_host[17], 1);
    ASSERT_EQ(output_token_ids_host[23], 2);
    ASSERT_NEAR(cum_log_probs_host[0], -1.78719, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[1], -2.64916, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[2], -3.78719, 1e-3);
    ASSERT_NEAR(cum_log_probs_host[3], -3.6709, 1e-3);

    printBuffer<float>(*output_all_probs, "output_all_probs");

    auto output_all_probs_host = getBufferValues<float>(*output_all_probs);
    ASSERT_VECTOR_NEAR(
        output_all_probs_host,
        std::vector<float>({0, 0.455121, 0.544879, 0, 0, 0, 0, 0, 0, 0, 0, 0.477515, 0.522485, 0, 0, 0, 0, 0, 0, 0,
                            0, 0.455121, 0.544879, 0, 0, 0, 0, 0, 0, 0, 0, 0.488752, 0.511248, 0, 0, 0, 0, 0, 0, 0}),
        1e-3);
}
