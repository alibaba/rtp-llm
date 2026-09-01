#include "gtest/gtest.h"

#include "rtp_llm/models_py/bindings/core/DeviceData.h"

namespace rtp_llm {

TEST(DeviceDataTest, LmHeadReplicationMatchesWeightLayout) {
    ParallelismConfig    parallelism_config;
    DeviceResourceConfig device_resource_config;

    EXPECT_TRUE(buildExecProperties(parallelism_config, device_resource_config).lm_head_is_replicated);

    parallelism_config.ep_size = 4;
    EXPECT_TRUE(buildExecProperties(parallelism_config, device_resource_config).lm_head_is_replicated);

    parallelism_config.ep_size = 1;
    parallelism_config.tp_size = 4;
    EXPECT_FALSE(buildExecProperties(parallelism_config, device_resource_config).lm_head_is_replicated);

    parallelism_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    EXPECT_TRUE(buildExecProperties(parallelism_config, device_resource_config).lm_head_is_replicated);

    parallelism_config.ep_size = 4;
    EXPECT_FALSE(buildExecProperties(parallelism_config, device_resource_config).lm_head_is_replicated);

    parallelism_config.ep_size = 1;
    parallelism_config.dp_size = 2;
    EXPECT_FALSE(buildExecProperties(parallelism_config, device_resource_config).lm_head_is_replicated);
}

}  // namespace rtp_llm
