#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "aios/autil/autil/EnvUtil.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace ::testing;
namespace rtp_llm {

TEST(ParallelInfoTest, Constructor) {
    int          tp_size          = 1;
    int          pp_size          = 2;
    int          ep_size          = 1;
    int          dp_size          = 1;
    int          world_size       = 2;
    int          world_rank       = 0;
    int          local_world_size = 1;
    ParallelInfo parallel_info(tp_size, pp_size, ep_size, dp_size, world_size, world_rank, local_world_size);
    EXPECT_EQ(parallel_info.getTpSize(), tp_size);
    EXPECT_EQ(parallel_info.getPpSize(), pp_size);
    EXPECT_EQ(parallel_info.getEpSize(), ep_size);
    EXPECT_EQ(parallel_info.getDpSize(), dp_size);
    EXPECT_EQ(parallel_info.getWorldSize(), world_size);
    EXPECT_EQ(parallel_info.getWorldRank(), world_rank);
    EXPECT_EQ(parallel_info.getLocalWorldSize(), local_world_size);
    EXPECT_EQ(parallel_info.getLocalRank(), world_rank % local_world_size);
    EXPECT_EQ(parallel_info.isMaster(), world_rank == 0);
    EXPECT_EQ(parallel_info.isWorker(), !parallel_info.isMaster());
}

TEST(ParallelInfoTest, GlobalParallelInfo) {
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    {
        autil::EnvGuard tp_size_env("TP_SIZE", "2");
        autil::EnvGuard pp_size_env("PP_SIZE", "2");
        autil::EnvGuard ep_size_env("EP_SIZE", "2");
        autil::EnvGuard dp_size_env("DP_SIZE", "2");
        autil::EnvGuard world_size_env("WORLD_SIZE", "2");
        autil::EnvGuard world_rank_env("WORLD_RANK", "2");
        autil::EnvGuard local_world_size_env("LOCAL_WORLD_SIZE", "2");
        parallel_info.reload();

        EXPECT_EQ(parallel_info.tp_size_, 2);
        EXPECT_EQ(parallel_info.pp_size_, 2);
        EXPECT_EQ(parallel_info.ep_size_, 2);
        EXPECT_EQ(parallel_info.dp_size_, 2);
        EXPECT_EQ(parallel_info.world_size_, 2);
        EXPECT_EQ(parallel_info.world_rank_, 2);
        EXPECT_EQ(parallel_info.local_world_size_, 2);
    }
    // 测试完毕后需要改回来, 否则其他地方使用 globalParallelInfo 会出错
    parallel_info.reload();
}

}  // namespace rtp_llm
