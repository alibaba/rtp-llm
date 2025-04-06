#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/base_tests/FfnLayerTest.hpp"

TEST_F(MoELayerTest, GateFp8MoEOpTest) {
    MoEOpTest(10, 512, 128, 128, 6, ActivationType::Silu, DataType::TYPE_FP8_E4M3, 2e-1, 1e2);
}

