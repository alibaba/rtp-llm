
#include "maga_transformer/cpp/devices/cuda_impl/CudaDevice.h"
#include "maga_transformer/cpp/devices/cuda_impl/tests/CudaTestUtils.h"
#include "maga_transformer/cpp/devices/base_tests/AttentionOpTest.hpp"

using namespace rtp_llm;

TEST_F(AttentionOpTest, SelfAttentionOpTest) {
    // batch size > 8 may exceed cache manager buffer size.
    setenv("ENABLE_MULTI_BLOCK_MODE", "OFF", 1);
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();
    ASSERT_FALSE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
    std::vector<size_t> batch = {2, 4, 8};
    std::vector<size_t> seq   = {1};
    std::vector<size_t> kv_seq = {0, 1, 2, 4, 8};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            for (auto kv_seq_len: kv_seq) {
                size_t num_heads = 64;
                size_t num_key_value_heads = num_heads;
                size_t head_dim = 64;
                selfAttentionOpTest(batch_size,
                                    seq_len,
                                    kv_seq_len,
                                    num_heads,
                                    num_key_value_heads,
                                    head_dim);
            }
        }
    }
}

TEST_F(AttentionOpTest, MultiBlockSelfAttentionOpTest) {
    // batch size > 8 may exceed cache manager buffer size.
    setenv("ENABLE_MULTI_BLOCK_MODE", "ON", 1);
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
    std::vector<size_t> batch = {2, 4, 8};
    std::vector<size_t> seq   = {1};
    std::vector<size_t> kv_seq = {0, 1, 2, 4, 8};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            for (auto kv_seq_len: kv_seq) {
                size_t num_heads = 64;
                size_t num_key_value_heads = num_heads;
                size_t head_dim = 64;
                selfAttentionOpTest(batch_size,
                                    seq_len,
                                    kv_seq_len,
                                    num_heads,
                                    num_key_value_heads,
                                    head_dim);
            }
        }
    }
}

TEST_F(AttentionOpTest, ContextAttentionOpTest) {
    setenv("ENABLE_TRT_FMHA", "OFF", 1);
    setenv("ENABLE_TRTV1_FMHA", "OFF", 1);
    setenv("ENABLE_OPENSOURCE_FMHA", "OFF", 1);
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_open_source_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
    std::vector<size_t> batch = {1, 2, 4, 8};
    std::vector<size_t> seq   = {1, 10, 20, 30};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads = 64;
            size_t num_key_value_heads = num_heads;
            size_t head_dim = 64;
            contextAttentionOpTest(batch_size,
                                   seq_len,
                                   num_heads,
                                   num_key_value_heads,
                                   head_dim);
        }
    }
}

TEST_F(AttentionOpTest, ContextAttentionOpMultiGroupTest) {
    setenv("ENABLE_TRT_FMHA", "OFF", 1);
    setenv("ENABLE_TRTV1_FMHA", "OFF", 1);
    setenv("ENABLE_OPENSOURCE_FMHA", "OFF", 1);
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_open_source_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
    std::vector<size_t> batch = {1, 2, 4, 8};
    std::vector<size_t> seq   = {1, 10, 20, 30};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads = 64;
            size_t num_key_value_heads = 4;
            size_t head_dim = 64;
            contextAttentionOpTest(batch_size,
                                   seq_len,
                                   num_heads,
                                   num_key_value_heads,
                                   head_dim);
        }
    }
}

TEST_F(AttentionOpTest, OpenSourceFMHAContextAttentionOpTest) {
    setenv("ENABLE_TRT_FMHA", "OFF", 1);
    setenv("ENABLE_TRTV1_FMHA", "OFF", 1);
    setenv("ENABLE_OPENSOURCE_FMHA", "ON", 1);
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_open_source_fmha);

    std::vector<size_t> batch = {1, 2, 4, 8};
    std::vector<size_t> seq   = {1, 10, 20, 30};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads = 64;
            size_t num_key_value_heads = num_heads;
            size_t head_dim = 64;
            contextAttentionOpTest(batch_size,
                                   seq_len,
                                   num_heads,
                                   num_key_value_heads,
                                   head_dim);
        }
    }
}


TEST_F(AttentionOpTest, TrtV2ContextAttentionOpTest) {
    setenv("ENABLE_TRT_FMHA", "ON", 1);
    setenv("ENABLE_OPENSOURCE_FMHA", "OFF", 1);
    setenv("ENABLE_TRTV1_FMHA", "OFF", 1);
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_open_source_fmha);

    std::vector<size_t> batch = {1, 2, 4, 8};
    std::vector<size_t> seq   = {1, 10, 20, 30};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads = 64;
            size_t num_key_value_heads = num_heads;
            size_t head_dim = 64;
            contextAttentionOpTest(batch_size,
                                   seq_len,
                                   num_heads,
                                   num_key_value_heads,
                                   head_dim);
        }
    }
}

TEST_F(AttentionOpTest, TrtV1ContextAttentionOpTest) {
    setenv("ENABLE_TRT_FMHA", "OFF", 1);
    setenv("ENABLE_OPENSOURCE_FMHA", "OFF", 1);
    setenv("ENABLE_TRTV1_FMHA", "ON", 1);
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_open_source_fmha);

    std::vector<size_t> batch = {1, 2, 4, 8};
    std::vector<size_t> seq   = {1, 10, 20, 30};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads = 64;
            size_t num_key_value_heads = num_heads;
            size_t head_dim = 64;
            contextAttentionOpTest(batch_size,
                                   seq_len,
                                   num_heads,
                                   num_key_value_heads,
                                   head_dim);
        }
    }
}

TEST_F(AttentionOpTest, LongSeqMultiBlockSelfAttentionOpTest) {
    setenv("ENABLE_MULTI_BLOCK_MODE", "ON", 1);
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
    std::vector<size_t> batch = {4};
    std::vector<size_t> seq   = {1};
    std::vector<size_t> kv_seq = {16000};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            for (auto kv_seq_len: kv_seq) {
                size_t num_heads = 64;
                size_t num_key_value_heads = num_heads;
                size_t head_dim = 64;
                selfAttentionOpTest(batch_size,
                                    seq_len,
                                    kv_seq_len,
                                    num_heads,
                                    num_key_value_heads,
                                    head_dim);
            }
        }
    }
}

TEST_F(AttentionOpTest, LongSeqSelfAttentionOpTest) {
    setenv("ENABLE_MULTI_BLOCK_MODE", "OFF", 1);
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();
    ASSERT_FALSE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
    std::vector<size_t> batch = {4};
    std::vector<size_t> seq   = {1};
    std::vector<size_t> kv_seq = {16000};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            for (auto kv_seq_len: kv_seq) {
                size_t num_heads = 64;
                size_t num_key_value_heads = num_heads;
                size_t head_dim = 64;
                selfAttentionOpTest(batch_size,
                                    seq_len,
                                    kv_seq_len,
                                    num_heads,
                                    num_key_value_heads,
                                    head_dim);
            }
        }
    }
}

#ifdef USING_CUDA12
TEST_F(AttentionOpTest, XqaAttentionOpTest) {
    setenv("ENABLE_TRT_FMHA", "OFF", 1);
    setenv("ENABLE_TRTV1_FMHA", "OFF", 1);
    setenv("ENABLE_OPENSOURCE_FMHA", "OFF", 1);
    setenv("DISABLE_FLASH_INFER", "1", 1);
    setenv("ENABLE_XQA", "ON", 1);
    setenv("ENABLE_MULTI_BLOCK_MODE", "OFF", 1);
    device_ = new CudaDevice(DeviceInitParams());
    device_->init();
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_xqa);
    ASSERT_FALSE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
    std::vector<size_t> batch = {2};
    std::vector<size_t> seq   = {1};
    std::vector<size_t> kv_seq = {1, TOKENS_PER_PAGE - 1, TOKENS_PER_PAGE, TOKENS_PER_PAGE + 1};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            for (auto kv_seq_len: kv_seq) {
                size_t num_key_value_heads = 4;
                size_t num_heads = num_key_value_heads * HEAD_GRP_SIZE;
                size_t head_dim = HEAD_ELEMS;
                xqaAttentionOpTest(batch_size,
                                   seq_len,
                                   kv_seq_len,
                                   num_heads,
                                   num_key_value_heads,
                                   head_dim);
            }
        }
    }
}
#endif
