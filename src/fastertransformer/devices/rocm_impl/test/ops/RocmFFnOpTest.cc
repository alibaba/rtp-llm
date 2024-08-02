#include "src/fastertransformer/devices/rocm_impl/RocmTestUtils.h"
#include "src/fastertransformer/devices/base_tests/FfnLayerTest.hpp"
#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/cuda/Dispatch.h"


class RocmFfnLayerTest: public FfnLayerTest {};

TEST_F(RocmFfnLayerTest, Gate_Fp16_FfnOpTest) {
    FfnOpTest(4, 16, 16, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(4, 32, 32, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(4, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(4, 2048, 4096, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(128, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(1000, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(1, 2, 4096, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(1000, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP16);
}

TEST_F(RocmFfnLayerTest, NoGate_Fp16_FfnOpTest) {
    FfnOpTest(4, 16, 16, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(4, 32, 32, ActivationType::Swiglu, DataType::TYPE_FP16);
    FfnOpTest(4, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(4, 2048, 4096, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(128, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(1000, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(1, 2, 4096, ActivationType::Silu, DataType::TYPE_FP16);
    FfnOpTest(1000, 2048, 128, ActivationType::Silu, DataType::TYPE_FP16);
}

TEST_F(MoELayerTest, Gate_Fp16_MoEOpTest) {
    MoEOpTest(4, 3584, 2560, 64, 8, ActivationType::Silu, DataType::TYPE_FP16);
    MoEOpTest(10, 448, 320, 8, 2, ActivationType::Swiglu, DataType::TYPE_FP16);  // FUll divide 8
    MoEOpTest(10, 448, 320, 8, 2, ActivationType::Silu, DataType::TYPE_FP16);  // FUll divide 8
}
