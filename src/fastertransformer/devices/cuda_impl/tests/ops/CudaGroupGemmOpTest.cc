#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/base_tests/GroupGemmOpTest.hpp"

using namespace std;
using namespace fastertransformer;

class CudaGroupGemmOpTest: public GroupGemmOpTest {};


TEST_F(CudaGroupGemmOpTest, GroupGemmOpTest) {
    groupGemmOpTest({{64, 64}}, {{64, 64}}, DataType::TYPE_FP16);
    groupGemmOpTest({{64, 64}, {64, 64}}, {{64, 64}, {64, 64}}, DataType::TYPE_FP16);
    groupGemmOpTest({{1, 8}, {100, 64}}, {{8, 2048}, {64, 2048}}, DataType::TYPE_FP16);
    groupGemmOpTest({{1, 8}, {100, 64}, {3, 128}}, {{8, 2048}, {64, 2048}, {128, 128}}, DataType::TYPE_FP16);

    groupGemmOpTest({{64, 64}}, {{64, 64}}, DataType::TYPE_FP16, true);
    groupGemmOpTest({{64, 64}, {64, 64}}, {{64, 64}, {64, 64}}, DataType::TYPE_FP16, true);
    groupGemmOpTest({{1, 8}, {100, 64}}, {{8, 2048}, {64, 2048}}, DataType::TYPE_FP16, true);
    groupGemmOpTest({{1, 8}, {100, 64}, {3, 128}}, {{8, 2048}, {64, 2048}, {128, 128}}, DataType::TYPE_FP16, true);
}
