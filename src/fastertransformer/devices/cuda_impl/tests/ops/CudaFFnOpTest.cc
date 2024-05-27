#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/base_tests/FfnLayerTest.hpp"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"


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
