
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaTestUtils.h"
#include "rtp_llm/cpp/devices/base_tests/AttentionOpTest.hpp"
#include "rtp_llm/cpp/config/ConfigModules.h"
using namespace rtp_llm;

TEST_F(AttentionOpTest, SelfAttentionOpTest) {
    // batch size > 8 may exceed cache manager buffer size.
    DeviceInitParams device_init_params;
    device_init_params.fmha_config.enable_trt_fmha              = false;
    device_init_params.fmha_config.enable_trtv1_fmha            = false;
    device_init_params.fmha_config.enable_open_source_fmha      = false;
    device_init_params.fmha_config.disable_flash_infer          = true;
    device_init_params.fmha_config.enable_xqa                   = false;
    device_init_params.hw_kernel_config.enable_multi_block_mode = false;
    device_                                                     = new CudaDevice(device_init_params);
    device_->init();
    ASSERT_FALSE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
    std::vector<size_t> batch  = {2, 4, 8};
    std::vector<size_t> seq    = {1};
    std::vector<size_t> kv_seq = {0, 1, 2, 4, 8};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            for (auto kv_seq_len : kv_seq) {
                size_t num_heads           = 64;
                size_t num_key_value_heads = num_heads;
                size_t head_dim            = 64;
                selfAttentionOpTest(batch_size, seq_len, kv_seq_len, num_heads, num_key_value_heads, head_dim);
            }
        }
    }
}

TEST_F(AttentionOpTest, MultiBlockSelfAttentionOpTest) {
    // batch size > 8 may exceed cache manager buffer size.
    DeviceInitParams device_init_params;
    device_init_params.fmha_config.enable_trt_fmha              = false;
    device_init_params.fmha_config.enable_trtv1_fmha            = false;
    device_init_params.fmha_config.enable_open_source_fmha      = false;
    device_init_params.fmha_config.disable_flash_infer          = true;
    device_init_params.fmha_config.enable_xqa                   = false;
    device_init_params.hw_kernel_config.enable_multi_block_mode = true;
    device_                                                     = new CudaDevice(device_init_params);
    device_->init();
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
    std::vector<size_t> batch  = {2, 4, 8};
    std::vector<size_t> seq    = {1};
    std::vector<size_t> kv_seq = {0, 1, 2, 4, 8};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            for (auto kv_seq_len : kv_seq) {
                size_t num_heads           = 64;
                size_t num_key_value_heads = num_heads;
                size_t head_dim            = 64;
                selfAttentionOpTest(batch_size, seq_len, kv_seq_len, num_heads, num_key_value_heads, head_dim);
            }
        }
    }
}

TEST_F(AttentionOpTest, ContextAttentionOpTest) {
    auto device_init_params                                = DeviceInitParams();
    device_init_params.fmha_config.enable_trt_fmha         = false;
    device_init_params.fmha_config.enable_trtv1_fmha       = false;
    device_init_params.fmha_config.enable_open_source_fmha = false;
    device_init_params.fmha_config.disable_flash_infer     = true;
    device_init_params.fmha_config.enable_xqa              = false;
    device_                                                = new CudaDevice(device_init_params);
    device_->init();
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_open_source_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
    std::vector<size_t> batch = {1, 2, 4, 8};
    std::vector<size_t> seq   = {1, 10, 20, 30};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads           = 64;
            size_t num_key_value_heads = num_heads;
            size_t head_dim            = 64;
            contextAttentionOpTest(batch_size, seq_len, num_heads, num_key_value_heads, head_dim);
        }
    }
}

TEST_F(AttentionOpTest, ContextAttentionOpMultiGroupTest) {
    auto device_init_params                                = DeviceInitParams();
    device_init_params.fmha_config.enable_trt_fmha         = false;
    device_init_params.fmha_config.enable_trtv1_fmha       = false;
    device_init_params.fmha_config.enable_open_source_fmha = false;
    device_init_params.fmha_config.disable_flash_infer     = true;
    device_init_params.fmha_config.enable_xqa              = false;
    device_                                                = new CudaDevice(device_init_params);
    device_->init();
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_open_source_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
    std::vector<size_t> batch = {1, 2, 4, 8};
    std::vector<size_t> seq   = {1, 10, 20, 30};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads           = 64;
            size_t num_key_value_heads = 4;
            size_t head_dim            = 64;
            contextAttentionOpTest(batch_size, seq_len, num_heads, num_key_value_heads, head_dim);
        }
    }
}

TEST_F(AttentionOpTest, OpenSourceFMHAContextAttentionOpTest) {
    auto device_init_params                                = DeviceInitParams();
    device_init_params.fmha_config.enable_trt_fmha         = false;
    device_init_params.fmha_config.enable_trtv1_fmha       = false;
    device_init_params.fmha_config.enable_open_source_fmha = true;
    device_init_params.fmha_config.disable_flash_infer     = true;
    device_init_params.fmha_config.enable_xqa              = false;
    device_                                                = new CudaDevice(device_init_params);
    device_->init();
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_open_source_fmha);

    std::vector<size_t> batch = {1, 2, 4, 8};
    std::vector<size_t> seq   = {1, 10, 20, 30};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads           = 64;
            size_t num_key_value_heads = num_heads;
            size_t head_dim            = 64;
            contextAttentionOpTest(batch_size, seq_len, num_heads, num_key_value_heads, head_dim);
        }
    }
}

TEST_F(AttentionOpTest, TrtV2ContextAttentionOpTest) {
    auto device_init_params                                = DeviceInitParams();
    device_init_params.fmha_config.enable_trt_fmha         = true;
    device_init_params.fmha_config.enable_trtv1_fmha       = false;
    device_init_params.fmha_config.enable_open_source_fmha = false;
    device_init_params.fmha_config.disable_flash_infer     = true;
    device_init_params.fmha_config.enable_xqa              = false;
    device_                                                = new CudaDevice(device_init_params);
    device_->init();
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_open_source_fmha);

    std::vector<size_t> batch = {1, 2, 4, 8};
    std::vector<size_t> seq   = {1, 10, 20, 30};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads           = 64;
            size_t num_key_value_heads = num_heads;
            size_t head_dim            = 64;
            contextAttentionOpTest(batch_size, seq_len, num_heads, num_key_value_heads, head_dim);
        }
    }
}

TEST_F(AttentionOpTest, TrtV1ContextAttentionOpTest) {
    auto device_init_params                                = DeviceInitParams();
    device_init_params.fmha_config.enable_trt_fmha         = false;
    device_init_params.fmha_config.enable_trtv1_fmha       = true;
    device_init_params.fmha_config.enable_open_source_fmha = false;
    device_init_params.fmha_config.disable_flash_infer     = true;
    device_init_params.fmha_config.enable_xqa              = false;
    device_                                                = new CudaDevice(device_init_params);
    device_->init();
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_trtv1_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_trtv2_fmha);
    ASSERT_TRUE(!static_cast<CudaDevice*>(device_)->use_open_source_fmha);

    std::vector<size_t> batch = {1, 2, 4, 8};
    std::vector<size_t> seq   = {1, 10, 20, 30};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads           = 64;
            size_t num_key_value_heads = num_heads;
            size_t head_dim            = 64;
            contextAttentionOpTest(batch_size, seq_len, num_heads, num_key_value_heads, head_dim);
        }
    }
}

TEST_F(AttentionOpTest, LongSeqMultiBlockSelfAttentionOpTest) {
    DeviceInitParams device_init_params;
    device_init_params.fmha_config.enable_trt_fmha              = false;
    device_init_params.fmha_config.enable_trtv1_fmha            = false;
    device_init_params.fmha_config.enable_open_source_fmha      = false;
    device_init_params.fmha_config.disable_flash_infer          = true;
    device_init_params.fmha_config.enable_xqa                   = false;
    device_init_params.hw_kernel_config.enable_multi_block_mode = true;
    device_                                                     = new CudaDevice(device_init_params);
    device_->init();
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
    std::vector<size_t> batch  = {4};
    std::vector<size_t> seq    = {1};
    std::vector<size_t> kv_seq = {16000};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            for (auto kv_seq_len : kv_seq) {
                size_t num_heads           = 64;
                size_t num_key_value_heads = num_heads;
                size_t head_dim            = 64;
                selfAttentionOpTest(batch_size, seq_len, kv_seq_len, num_heads, num_key_value_heads, head_dim);
            }
        }
    }
}

TEST_F(AttentionOpTest, LongSeqSelfAttentionOpTest) {
    DeviceInitParams device_init_params;
    device_init_params.fmha_config.enable_trt_fmha              = false;
    device_init_params.fmha_config.enable_trtv1_fmha            = false;
    device_init_params.fmha_config.enable_open_source_fmha      = false;
    device_init_params.fmha_config.disable_flash_infer          = true;
    device_init_params.fmha_config.enable_xqa                   = false;
    device_init_params.hw_kernel_config.enable_multi_block_mode = false;
    device_                                                     = new CudaDevice(device_init_params);
    device_->init();
    ASSERT_FALSE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
    std::vector<size_t> batch  = {4};
    std::vector<size_t> seq    = {1};
    std::vector<size_t> kv_seq = {16000};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            for (auto kv_seq_len : kv_seq) {
                size_t num_heads           = 64;
                size_t num_key_value_heads = num_heads;
                size_t head_dim            = 64;
                selfAttentionOpTest(batch_size, seq_len, kv_seq_len, num_heads, num_key_value_heads, head_dim);
            }
        }
    }
}

#ifdef USING_CUDA12
TEST_F(AttentionOpTest, XqaAttentionOpTest) {
    auto device_init_params                                     = DeviceInitParams();
    device_init_params.fmha_config.enable_trt_fmha              = false;
    device_init_params.fmha_config.enable_trtv1_fmha            = false;
    device_init_params.fmha_config.enable_open_source_fmha      = false;
    device_init_params.fmha_config.disable_flash_infer          = true;
    device_init_params.fmha_config.enable_xqa                   = true;
    device_init_params.hw_kernel_config.enable_multi_block_mode = false;
    device_                                                     = new CudaDevice(device_init_params);
    device_->init();
    ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_xqa);
    ASSERT_FALSE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
    size_t              batch_size      = 3;
    std::vector<size_t> head_dim        = {64, 128, 256};
    size_t              seq_q           = 1;
    size_t              seq_kv          = 129;
    size_t              head_q          = 64;
    std::vector<size_t> head_kv         = {4, 8, 16, 32, 64};
    std::vector<size_t> page_size       = {16, 32, 64, 128};
    std::vector<bool>   is_kv_cache_fp8 = {true, false};
    for (auto hd : head_dim) {
        for (auto hkv : head_kv) {
            for (auto ps : page_size) {
                for (auto is_kv_fp8 : is_kv_cache_fp8) {
                    xqaAttentionOpTest(batch_size, seq_q, seq_kv, head_q, hkv, hd, ps, is_kv_fp8);
                }
            }
        }
    }
}

TEST_F(AttentionOpTest, FlashinferContextAttentionOpTest) {
    DeviceInitParams device_init_params;
    device_init_params.fmha_config.enable_trt_fmha              = false;
    device_init_params.fmha_config.enable_trtv1_fmha            = false;
    device_init_params.fmha_config.enable_open_source_fmha      = false;
    device_init_params.fmha_config.disable_flash_infer          = false;
    device_init_params.fmha_config.enable_xqa                   = false;
    device_init_params.hw_kernel_config.enable_multi_block_mode = false;
    device_                                                     = new CudaDevice(device_init_params);
    device_->init();
    std::vector<size_t> batch  = {3};
    std::vector<size_t> seq    = {1};
    std::vector<size_t> kv_seq = {2049};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            for (auto kv_seq_len : kv_seq) {
                size_t num_heads           = 64;
                size_t num_key_value_heads = 4;
                size_t head_dim            = 128;
                flashinferPrefillOpTest(batch_size, seq_len, kv_seq_len, num_heads, num_key_value_heads, head_dim);
            }
        }
    }
}

// TEST_F(AttentionOpTest, XqaContextAttentionOpTest) {
//     DeviceInitParams device_init_params;
//     device_init_params.fmha_config.enable_trt_fmha = true;
//     device_init_params.fmha_config.enable_trtv1_fmha = false;
//     device_init_params.fmha_config.enable_open_source_fmha = false;
//     device_init_params.fmha_config.disable_flash_infer = false;
//     device_init_params.fmha_config.enable_xqa = true;
//     device_init_params.hw_kernel_config.enable_multi_block_mode = false;
//     device_ = new CudaDevice(device_init_params);
//     device_->init();
//     ASSERT_TRUE(static_cast<CudaDevice*>(device_)->use_xqa);
//     ASSERT_FALSE(static_cast<CudaDevice*>(device_)->use_multi_block_mode);
//     std::vector<size_t> batch = {3};
//     std::vector<size_t> seq   = {1};
//     std::vector<size_t> kv_seq = {2049};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             for (auto kv_seq_len: kv_seq) {
//                 size_t num_heads = 64;
//                 size_t num_key_value_heads = 4;
//                 size_t head_dim = 128;
//                 xqaPrefillOpTest(batch_size, seq_len, kv_seq_len, num_heads, num_key_value_heads, head_dim);
//             }
//         }
//     }
// }
#endif
