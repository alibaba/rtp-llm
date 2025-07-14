#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/base_tests/GroupGemmOpTest.hpp"

using namespace std;
using namespace rtp_llm;

class CudaGroupGemmOpTest: public GroupGemmOpTest {};

TEST_F(CudaGroupGemmOpTest, GroupGemmOpTest) {
    std::vector<DataType> dtypes = {DataType::TYPE_FP16, DataType::TYPE_BF16, DataType::TYPE_FP32};
    for (auto dtype : dtypes) {
        double atol = (dtype == DataType::TYPE_FP16) ? 0 : 1e-02;
        double rtol = (dtype == DataType::TYPE_FP16) ? 0 : 1e-02;
        groupGemmOpTest({{64, 64}}, {{64, 64}}, dtype, atol, rtol);
        groupGemmOpTest({{64, 64}}, {{64, 64}}, dtype, atol, rtol);
        groupGemmOpTest({{64, 64}, {64, 64}}, {{64, 64}, {64, 64}}, dtype, atol, rtol);
        groupGemmOpTest({{1, 8}, {100, 64}}, {{8, 2048}, {64, 2048}}, dtype, atol, rtol);
        groupGemmOpTest({{1, 8}, {100, 64}, {3, 128}}, {{8, 2048}, {64, 2048}, {128, 128}}, dtype, atol, rtol);

        groupGemmOpTest({{64, 64}}, {{64, 64}}, dtype, atol, rtol);
        groupGemmOpTest({{64, 64}, {64, 64}}, {{64, 64}, {64, 64}}, dtype, atol, rtol);
        groupGemmOpTest({{1, 8}, {100, 64}}, {{8, 2048}, {64, 2048}}, dtype, atol, rtol);
        groupGemmOpTest({{1, 8}, {100, 64}, {3, 128}}, {{8, 2048}, {64, 2048}, {128, 128}}, dtype, atol, rtol);
    }
}
