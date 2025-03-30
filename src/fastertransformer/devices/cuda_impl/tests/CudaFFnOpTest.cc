#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/base_tests/FfnLayerTest.hpp"

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
    FfnOpTest(4, 16, 16, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(4, 32, 32, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(4, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(4, 2048, 4096, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(128, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(1000, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(1, 2, 4096, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(1000, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
}

TEST_F(CudaFfnLayerTest, GateFp16LoraTest) {
    FfnLayerLoraTest({1}, {8}, {8}, {8}, 64, 64, DataType::TYPE_FP16, DataType::TYPE_FP16, ActivationType::Swiglu);
    FfnLayerLoraTest({1000}, {8}, {128}, {64}, 64, 64, DataType::TYPE_FP16, DataType::TYPE_FP16, ActivationType::Swiglu);
    FfnLayerLoraTest({1000, 1, 10, 2}, {8, 8, 16, 128}, {128, 8, 16, 64}, {64, 8, 16, 64}, 64, 64, DataType::TYPE_FP16, DataType::TYPE_FP16, ActivationType::Swiglu);
}

TEST_F(CudaFfnLayerTest, NoGateFp16LoraOpTest) {
    FfnLayerLoraTest({1}, {8}, {8}, {8}, 64, 64, DataType::TYPE_FP16, DataType::TYPE_FP16, ActivationType::Silu);
    FfnLayerLoraTest({1000}, {8}, {128}, {64}, 64, 64, DataType::TYPE_FP16, DataType::TYPE_FP16, ActivationType::Silu);
    FfnLayerLoraTest({1000, 1, 10, 2}, {8, 8, 16, 128}, {128, 8, 16, 64}, {64, 8, 16, 64}, 64, 64, DataType::TYPE_FP16, DataType::TYPE_FP16, ActivationType::Silu);
}

TEST_F(MoELayerTest, GateFp8MoEOpTest) {
    MoEOpTest(10, 128, 256, 8, 6, ActivationType::Silu, DataType::TYPE_FP8_E4M3, 2e-1, 1e2);
    MoEOpTest(4, 2048, 1408, 64, 8, ActivationType::Swiglu, DataType::TYPE_FP8_E4M3, 3e-1, 2e4);
}

