#include <gtest/gtest.h>
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "trt_plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h"
#include "rtp_llm/cpp/core/torch_utils/TypeConvert.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include <nvtx3/nvToolsExt.h>
#include <cuda_fp8.h>

using namespace rtp_llm;

class CudaQuantizeTest: public DeviceTestBase {
public:
    void RunCudaQuantizeFp8() {
        // TODO: quantize() has not been migrated to a free function yet.
        // Once an execQuantize() or similar is available, update this test.
        GTEST_SKIP() << "quantize() free function not yet available";
    }
};

TEST_F(CudaQuantizeTest, Test1) {
    RunCudaQuantizeFp8();
}
