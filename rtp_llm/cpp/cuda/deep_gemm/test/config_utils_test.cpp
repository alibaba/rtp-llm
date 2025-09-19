#include <gtest/gtest.h>
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"

#include "rtp_llm/cpp/cuda/deep_gemm/ConfigUtils.h"

using namespace rtp_llm;

class ConfigUtilsTest: public DeviceTestBase {
public:
};

TEST_F(ConfigUtilsTest, SimpleTest) {
    const int    m          = 120;
    const int    n          = 6144;
    const int    k          = 8192;
    const int    num_groups = 5;
    const int    num_sms    = 132;
    DeepGemmType gemm_type  = DeepGemmType::GroupedMasked;
    const int    expected_m = 120;

    auto config = getBestConfig(m, n, k, num_groups, num_sms, gemm_type, expected_m);
    std::cout << "block m: " << config.block_m << ", block n: " << config.block_n
              << ", num stages: " << config.num_stages << ", smem size: " << config.smem_size
              << ", swap ab: " << (config.swap_ab ? "true" : "false") << std::endl;
}
