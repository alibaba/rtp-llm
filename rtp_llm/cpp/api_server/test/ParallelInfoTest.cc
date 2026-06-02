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

}  // namespace rtp_llm
