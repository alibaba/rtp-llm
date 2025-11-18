// #define private public
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"

#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include <torch/torch.h>
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/devices/base_tests/AttentionOpTest.hpp"

template<>
struct c10::CppTypeToScalarType<half>: std::integral_constant<c10::ScalarType, c10::ScalarType::Half> {};
template<>
struct c10::CppTypeToScalarType<__nv_bfloat16>: std::integral_constant<c10::ScalarType, c10::ScalarType::BFloat16> {};

using namespace std;
using namespace rtp_llm;

TEST_F(AttentionOpTest, SelfAttentionOpTest) {
    // batch size > 8 may exceed cache manager buffer size.
    printf("Runing SelfAttentionOpTest\n");
    DeviceInitParams device_init_params;
    device_init_params.hw_kernel_config.enable_multi_block_mode = false;
    device_                                                     = new ROCmDevice(device_init_params);
    device_->init();
    // ASSERT_FALSE(static_cast<ROCmDevice*>(device_)->use_multi_block_mode);
    std::vector<size_t> batch  = {128};
    std::vector<size_t> seq    = {1};
    std::vector<size_t> kv_seq = {4096};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            for (auto kv_seq_len : kv_seq) {
                size_t num_heads           = 8;
                size_t num_key_value_heads = 1;
                size_t head_dim            = 128;
                selfAttentionOpTest(batch_size, seq_len, kv_seq_len, num_heads, num_key_value_heads, head_dim);
            }
        }
    }
}

TEST_F(AttentionOpTest, AiterPageAttentionOpTest) {
    autil::EnvUtil::setEnv("USE_AITER_PA", "1");
    device_ = new ROCmDevice(DeviceInitParams());
    device_->init();
    std::vector<size_t> batch  = {128};
    std::vector<size_t> seq    = {1};
    std::vector<size_t> kv_seq = {4096};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            for (auto kv_seq_len : kv_seq) {
                size_t num_key_value_heads = 4;
                size_t num_heads           = 64;
                size_t head_dim            = 128;
                aiterPageAttentionOpTest(batch_size, seq_len, kv_seq_len, num_heads, num_key_value_heads, head_dim);
            }
        }
    }
}

/*
TEST_F(AttentionOpTest, MultiBlockSelfAttentionOpTest) {
    // batch size > 8 may exceed cache manager buffer size.
    DeviceInitParams device_init_params;
    device_init_params.hw_kernel_config.enable_multi_block_mode = true;
    device_ = new ROCmDevice(device_init_params);
    device_->init();
    // ASSERT_TRUE(static_cast<ROCmDevice*>(device_)->use_multi_block_mode);
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
*/
TEST_F(AttentionOpTest, ContextAttentionOpTest) {
    printf("Runing ContextAttentionOpTest\n");
    auto device_init_params                                = DeviceInitParams();
    device_init_params.fmha_config.enable_trt_fmha         = false;
    device_init_params.fmha_config.enable_trtv1_fmha       = false;
    device_init_params.fmha_config.enable_open_source_fmha = false;
    device_                                                = new ROCmDevice(device_init_params);
    device_->init();
    // ASSERT_TRUE(!static_cast<ROCmDevice*>(device_)->use_trtv2_fmha);
    // ASSERT_TRUE(!static_cast<ROCmDevice*>(device_)->use_openSource_fmha);
    // ASSERT_TRUE(!static_cast<ROCmDevice*>(device_)->use_trtv1_fmha);
    std::vector<size_t> batch = {1};
    std::vector<size_t> seq   = {4000};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            size_t num_heads           = 8;
            size_t num_key_value_heads = 1;
            size_t head_dim            = 128;
            size_t dim                 = head_dim;
            contextAttentionOpTest(batch_size, seq_len, num_heads, num_key_value_heads, head_dim);
        }
    }
}

/*
TEST_F(AttentionOpTest, AiterPageAttentionOpTest) {
    device_ = new ROCmDevice(DeviceInitParams());
    device_->init();
    std::vector<size_t> batch = {128};
    std::vector<size_t> seq   = {1};
    std::vector<size_t> kv_seq = {4096};
    for (auto batch_size : batch) {
        for (auto seq_len : seq) {
            for (auto kv_seq_len: kv_seq) {
                size_t num_key_value_heads = 1;
                size_t num_heads = 8;
                size_t head_dim = 128;
                aiterPageAttentionOpTest(batch_size,
                                         seq_len,
                                         kv_seq_len,
                                         num_heads,
                                         num_key_value_heads,
                                         head_dim);
            }
        }
    }
}
*/

// TEST_F(AttentionOpTest, OpenSourceFMHAContextAttentionOpTest) {
//     setenv("ENABLE_TRT_FMHA", "OFF", 1);
//     setenv("ENABLE_TRTV1_FMHA", "OFF", 1);
//     setenv("ENABLE_OPENSOURCE_FMHA", "ON", 1);
//     device_ = new ROCmDevice(DeviceInitParams());
//     device_->init();
//     ASSERT_TRUE(!static_cast<ROCmDevice*>(device_)->use_trtv2_fmha);
//     ASSERT_TRUE(!static_cast<ROCmDevice*>(device_)->use_trtv1_fmha);
//     ASSERT_TRUE(static_cast<ROCmDevice*>(device_)->use_openSource_fmha);

//     std::vector<size_t> batch = {1, 2, 4, 8};
//     std::vector<size_t> seq   = {1, 10, 20, 30};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             size_t num_heads = 64;
//             size_t num_key_value_heads = num_heads;
//             size_t head_dim = 64;
//             size_t dim = head_dim;
//             contextAttentionOpTest(batch_size,
//                                    seq_len,
//                                    num_heads,
//                                    num_key_value_heads,
//                                    head_dim);
//         }
//     }
// }

// TEST_F(AttentionOpTest, TrtV2ContextAttentionOpTest) {
//     setenv("ENABLE_TRT_FMHA", "ON", 1);
//     setenv("ENABLE_OPENSOURCE_FMHA", "OFF", 1);
//     setenv("ENABLE_TRTV1_FMHA", "OFF", 1);
//     device_ = new ROCmDevice(DeviceInitParams());
//     device_->init();
//     ASSERT_TRUE(static_cast<ROCmDevice*>(device_)->use_trtv2_fmha);
//     ASSERT_TRUE(!static_cast<ROCmDevice*>(device_)->use_trtv1_fmha);
//     ASSERT_TRUE(!static_cast<ROCmDevice*>(device_)->use_openSource_fmha);

//     std::vector<size_t> batch = {1, 2, 4, 8};
//     std::vector<size_t> seq   = {1, 10, 20, 30};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             size_t num_heads = 64;
//             size_t num_key_value_heads = num_heads;
//             size_t head_dim = 64;
//             size_t dim = head_dim;
//             contextAttentionOpTest(batch_size,
//                                    seq_len,
//                                    num_heads,
//                                    num_key_value_heads,
//                                    head_dim);
//         }
//     }
// }

// TEST_F(AttentionOpTest, TrtV1ContextAttentionOpTest) {
//     setenv("ENABLE_TRT_FMHA", "OFF", 1);
//     setenv("ENABLE_OPENSOURCE_FMHA", "OFF", 1);
//     setenv("ENABLE_TRTV1_FMHA", "ON", 1);
//     device_ = new ROCmDevice(DeviceInitParams());
//     device_->init();
//     ASSERT_TRUE(static_cast<ROCmDevice*>(device_)->use_trtv1_fmha);
//     ASSERT_TRUE(!static_cast<ROCmDevice*>(device_)->use_trtv2_fmha);
//     ASSERT_TRUE(!static_cast<ROCmDevice*>(device_)->use_openSource_fmha);

//     std::vector<size_t> batch = {1, 2, 4, 8};
//     std::vector<size_t> seq   = {1, 10, 20, 30};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             size_t num_heads = 64;
//             size_t num_key_value_heads = num_heads;
//             size_t head_dim = 64;
//             size_t dim = head_dim;
//             contextAttentionOpTest(batch_size,
//                                    seq_len,
//                                    num_heads,
//                                    num_key_value_heads,
//                                    head_dim);
//         }
//     }
// }

// TEST_F(AttentionOpTest, LongSeqMultiBlockSelfAttentionOpTest) {
//     setenv("ENABLE_MULTI_BLOCK_MODE", "ON", 1);
//     device_ = new ROCmDevice(DeviceInitParams());
//     device_->init();
//     ASSERT_TRUE(static_cast<ROCmDevice*>(device_)->use_multi_block_mode);
//     std::vector<size_t> batch = {4};
//     std::vector<size_t> seq   = {1};
//     std::vector<size_t> kv_seq = {16000};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             for (auto kv_seq_len: kv_seq) {
//                 size_t num_heads = 64;
//                 size_t num_key_value_heads = num_heads;
//                 size_t head_dim = 64;
//                 selfAttentionOpTest(batch_size,
//                                     seq_len,
//                                     kv_seq_len,
//                                     num_heads,
//                                     num_key_value_heads,
//                                     head_dim);
//             }
//         }
//     }
// }

// TEST_F(AttentionOpTest, LongSeqSelfAttentionOpTest) {
//     setenv("ENABLE_MULTI_BLOCK_MODE", "OFF", 1);
//     device_ = new ROCmDevice(DeviceInitParams());
//     device_->init();
//     ASSERT_FALSE(static_cast<ROCmDevice*>(device_)->use_multi_block_mode);
//     std::vector<size_t> batch = {4};
//     std::vector<size_t> seq   = {1};
//     std::vector<size_t> kv_seq = {16000};
//     for (auto batch_size : batch) {
//         for (auto seq_len : seq) {
//             for (auto kv_seq_len: kv_seq) {
//                 size_t num_heads = 64;
//                 size_t num_key_value_heads = num_heads;
//                 size_t head_dim = 64;
//                 selfAttentionOpTest(batch_size,
//                                     seq_len,
//                                     kv_seq_len,
//                                     num_heads,
//                                     num_key_value_heads,
//                                     head_dim);
//             }
//         }
//     }
// }
