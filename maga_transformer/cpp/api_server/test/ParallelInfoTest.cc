#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "maga_transformer/cpp/api_server/ParallelInfo.h"

using namespace ::testing;
namespace rtp_llm {

TEST(ParallelInfoTest, Constructor) {
    int          world_size       = 2;
    int          world_rank       = 0;
    int          local_world_size = 1;
    ParallelInfo parallel_info(world_size, world_rank, local_world_size);
    EXPECT_EQ(parallel_info.getWorldSize(), world_size);
    EXPECT_EQ(parallel_info.getWorldRank(), world_rank);
    EXPECT_EQ(parallel_info.getLocalWorldSize(), local_world_size);
    EXPECT_EQ(parallel_info.getLocalRank(), world_rank % local_world_size);
    EXPECT_EQ(parallel_info.isMaster(), world_rank == 0);
    EXPECT_EQ(parallel_info.isWorker(), !parallel_info.isMaster());
}

TEST(ParallelInfoTest, GlobalParallelInfo) {
    autil::EnvGuard word_size_env("WORLD_SIZE", "2");
    autil::EnvGuard word_rank_env("WORLD_RANK", "2");
    autil::EnvGuard local_word_size_env("LOCAL_WORLD_SIZE", "2");
    auto&           parallel_info = ParallelInfo::globalParallelInfo();
    parallel_info.reload();
    EXPECT_EQ(parallel_info.world_size_, 2);
    EXPECT_EQ(parallel_info.world_rank_, 2);
    EXPECT_EQ(parallel_info.local_world_size_, 2);
}

}  // namespace rtp_llm