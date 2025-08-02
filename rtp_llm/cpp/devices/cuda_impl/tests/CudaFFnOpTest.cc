#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/tests/CudaTestUtils.h"
#include "rtp_llm/cpp/devices/base_tests/FfnLayerTest.hpp"

using namespace rtp_llm;

class CudaFfnLayerTest: public FfnLayerTest {};

TEST_F(CudaFfnLayerTest, Gate_Fp16_FfnOpTest) {
    FfnOpTest(4, 16, 16, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(4, 32, 32, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(4, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(4, 2048, 4096, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(128, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(1000, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(1, 2, 4096, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(1000, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP16);
}

TEST_F(CudaFfnLayerTest, NoGate_Fp16_FfnOpTest) {
    FfnOpTest(4, 16, 16, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(4, 32, 32, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(4, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(4, 2048, 4096, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(128, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(1000, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(1, 2, 4096, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(1000, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
}

TEST_F(CudaFfnLayerTest, GateFp16LoraTest) {
    FfnLayerLoraTest({1}, {8}, {8}, {8}, 64, 64, DataType::TYPE_FP16, DataType::TYPE_FP16, ActivationType::Swiglu);
    FfnLayerLoraTest(
        {1000}, {8}, {128}, {64}, 64, 64, DataType::TYPE_FP16, DataType::TYPE_FP16, ActivationType::Swiglu);
    FfnLayerLoraTest({1000, 1, 10, 2},
                     {8, 8, 16, 128},
                     {128, 8, 16, 64},
                     {64, 8, 16, 64},
                     64,
                     64,
                     DataType::TYPE_FP16,
                     DataType::TYPE_FP16,
                     ActivationType::Swiglu);
}

TEST_F(CudaFfnLayerTest, NoGateFp16LoraOpTest) {
    FfnLayerLoraTest({1}, {8}, {8}, {8}, 64, 64, DataType::TYPE_FP16, DataType::TYPE_FP16, ActivationType::Silu);
    FfnLayerLoraTest({1000}, {8}, {128}, {64}, 64, 64, DataType::TYPE_FP16, DataType::TYPE_FP16, ActivationType::Silu);
    FfnLayerLoraTest({1000, 1, 10, 2},
                     {8, 8, 16, 128},
                     {128, 8, 16, 64},
                     {64, 8, 16, 64},
                     64,
                     64,
                     DataType::TYPE_FP16,
                     DataType::TYPE_FP16,
                     ActivationType::Silu);
}

class CudaMoEGateSelectTest: public MoEGateSelectTest {};

TEST_F(CudaMoEGateSelectTest, Basic) {
    constexpr int bench_round = 0;
    Bencher       bencher     = [](const char* case_name, const std::function<void()>& func) {
        if constexpr (bench_round > 0) {
            cudaEvent_t start, end;
            cudaEventCreate(&start);
            cudaEventCreate(&end);
            cudaEventRecord(start);
            for (int ii = 0; ii < bench_round; ++ii) {
                func();
            }
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            float duration;
            cudaEventElapsedTime(&duration, start, end);
            std::cout << "[" << case_name << "] bench_round: " << bench_round << ", total: " << duration
                      << ", avg: " << duration / bench_round << std::endl;
            cudaEventDestroy(start);
            cudaEventDestroy(end);
        }
    };

    for (auto token_num : {1, 10, 37, 128, 130}) {
        for (auto expert_num : {16, 32, 64}) {
            for (auto topk : {1, 2, 4, 8}) {
                for (auto group_num : {4, 8, 16}) {
                    for (auto group_topk : {1, 2, 4, 6, 8}) {
                        if (group_topk > group_num)
                            continue;
                        if (expert_num / group_num < 2)
                            continue;
                        if (topk > expert_num / group_num * group_topk)
                            continue;

                        for (auto has_bias : {false, true}) {
                            for (auto scoring_func : {0, 1}) {
                                for (auto has_moe_norm : {false, true}) {
                                    MoEGateSelTest(token_num,
                                                   7168,
                                                   expert_num,
                                                   topk,
                                                   has_bias,
                                                   group_num,
                                                   group_topk,
                                                   scoring_func,
                                                   has_moe_norm,
                                                   &bencher);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
