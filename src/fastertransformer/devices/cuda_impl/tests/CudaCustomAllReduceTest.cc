#include <gtest/gtest.h>

#define private public
#include "src/fastertransformer/devices/cuda_impl/tests/CudaTestUtils.h"
#include "src/fastertransformer/devices/base_tests/CustomAllReduceTest.hpp"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
using namespace std;
using namespace fastertransformer;

class CudaCustomAllReduceTest: public CustomAllReduceTest {};

TEST_F(CudaCustomAllReduceTest, base) {
    if (getenv("SKIP_DISTRIBUTED_TEST")) {
        FT_LOG_INFO("CustomAllReduce test skipped");
        return;
    }

    testForWorldSizeMultiProcess(2, false);
    testForWorldSizeMultiProcess(4, false);
}


TEST_F(CudaCustomAllReduceTest, benchmark) {
    if (getenv("BENCHMARK_AR_TEST")) {
        FT_LOG_INFO("CustomAllReduce benchmark");
        testForWorldSizeMultiProcess(2, true);
        testForWorldSizeMultiProcess(4, true);
        return;
    }
}
