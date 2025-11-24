
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaTestUtils.h"
#include "rtp_llm/cpp/devices/base_tests/UnfusedAttentionTest.hpp"

using namespace rtp_llm;

#ifdef USING_CUDA12
TEST_F(UnfusedAttentionTest, AddFusedQKVBiasTransposeTest) {
    auto device_init_params = DeviceInitParams();
    device_                 = new CudaDevice(device_init_params);
    device_->init();
    std::vector<size_t> batch_size = {1, 3};
    std::vector<size_t> seq        = {1, 65, 129};
    size_t              head_q     = 64;
    std::vector<size_t> head_kv    = {4, 8};
    std::vector<size_t> head_dim   = {128};
    std::vector<size_t> page_size  = {16, 64};
    for (auto bs : batch_size) {
        for (auto s : seq) {
            for (auto hkv : head_kv) {
                for (auto hd : head_dim) {
                    for (auto ps : page_size) {
                        addFusedQKVBiasTransposeTest(bs, s, head_q, hkv, hd, ps);
                    }
                }
            }
        }
    }
}

TEST_F(UnfusedAttentionTest, AddFusedQKVBiasTransposeBenchmark) {
    auto device_init_params = DeviceInitParams();
    device_                 = new CudaDevice(device_init_params);
    device_->init();
    std::vector<size_t> batch_size = {1, 2, 4, 8, 16};
    std::vector<size_t> seq_q      = {2048, 4096, 8192};
    size_t              head_q     = 64;
    std::vector<size_t> head_kv    = {4, 8};
    std::vector<size_t> head_dim   = {128};
    std::vector<size_t> page_size  = {64};
    for (auto bs : batch_size) {
        for (auto sq : seq_q) {
            for (auto hkv : head_kv) {
                for (auto hd : head_dim) {
                    for (auto ps : page_size) {
                        addFusedQKVBiasTransposeTest(bs, sq, head_q, hkv, hd, ps, true);
                    }
                }
            }
        }
    }
}

TEST_F(UnfusedAttentionTest, DecodeAddFusedQKVBiasTransposeTest) {
    auto device_init_params = DeviceInitParams();
    device_                 = new CudaDevice(device_init_params);
    device_->init();
    std::vector<size_t> batch_size = {1};
    size_t              seq_q      = 1;
    std::vector<size_t> seq_kv     = {1};
    size_t              head_q     = 64;
    std::vector<size_t> head_kv    = {4};
    size_t              head_dim   = 128;
    std::vector<size_t> page_size  = {16};
    for (auto bs : batch_size) {
        for (auto skv : seq_kv) {
            for (auto hkv : head_kv) {
                for (auto ps : page_size) {
                    decodeAddFusedQKVBiasTransposeTest(bs, seq_q, skv, head_q, hkv, head_dim, ps);
                }
            }
        }
    }
}

// TEST_F(UnfusedAttentionTest, DecodeAddFusedQKVBiasTransposeBenchmark) {
//     auto device_init_params = DeviceInitParams();
//     device_                 = new CudaDevice(device_init_params);
//     device_->init();
//     std::vector<size_t> batch_size = {1, 2, 4, 8, 16, 32, 64, 128};
//     size_t              seq_q      = 1;
//     std::vector<size_t> seq_kv     = {2048, 4096, 8192};
//     size_t              head_q     = 64;
//     std::vector<size_t> head_kv    = {4, 8};
//     std::vector<size_t> head_dim   = {128};
//     std::vector<size_t> page_size  = {64};
//     for (auto bs : batch_size) {
//         for (auto skv : seq_kv) {
//             for (auto hkv : head_kv) {
//                 for (auto hd : head_dim) {
//                     for (auto ps : page_size) {
//                         decodeAddFusedQKVBiasTransposeTest(bs, seq_q, skv, head_q, hkv, hd, ps, true);
//                     }
//                 }
//             }
//         }
//     }
// }
#endif
