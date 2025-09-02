#include "maga_transformer/cpp/devices/base_tests/FfnLayerTest.hpp"
#include "maga_transformer/cpp/devices/arm_impl/ArmDevice.h"


class ArmFfnLayerTest: public FfnLayerTest {};

TEST_F(ArmFfnLayerTest, Gate_Fp16_FfnOpTest) {
    FfnOpTest(4, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP32);
    // FfnOpTest(4, 2048, 4096, ActivationType::Swiglu, DataType::TYPE_FP32);
    // FfnOpTest(128, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP32);
    // FfnOpTest(1000, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP32);
    // FfnOpTest(1, 2, 4096, ActivationType::Swiglu, DataType::TYPE_FP32);
    // FfnOpTest(1000, 2048, 128, ActivationType::Swiglu, DataType::TYPE_FP32);
}

// TEST_F(ArmFfnLayerTest, NoGate_Fp16_FfnOpTest) {
//     FfnOpTest(4, 2048, 128, ActivationType::Geglu, DataType::TYPE_FP32);
//     FfnOpTest(4, 2048, 4096, ActivationType::Geglu, DataType::TYPE_FP32);
//     FfnOpTest(128, 2048, 128, ActivationType::Geglu, DataType::TYPE_FP32);
//     FfnOpTest(1000, 2048, 128, ActivationType::Geglu, DataType::TYPE_FP32);
//     FfnOpTest(1, 2, 4096, ActivationType::Geglu, DataType::TYPE_FP32);
//     FfnOpTest(1000, 2048, 128, ActivationType::Geglu, DataType::TYPE_FP32);
// }

TEST_F(MoELayerTest, Gate_Fp16_MoEOpTest) {
    // MoEOpTest(4, 3584, 2560, 64, 8, ActivationType::Silu, DataType::TYPE_FP16);
    MoEOpTest(10, 448, 320, 8, 2, ActivationType::Swiglu, DataType::TYPE_FP32);  // FUll divide 8
    // MoEOpTest(10, 448, 320, 8, 2, ActivationType::Silu, DataType::TYPE_FP16);  // FUll divide 8
}
