#include "rtp_llm/cpp/model_rpc/MlaCacheTpTransfer.h"

#include <gtest/gtest.h>

namespace rtp_llm {

TEST(MlaCacheTpTransferTest, ReconstructsTokenMajor576ByteExactly) {
    constexpr int kTp     = 8;
    constexpr int kTokens = 3;
    K3MlaCacheTpLayout layout(kTp);
    std::vector<torch::Tensor> shards;
    for (int rank = 0; rank < kTp; ++rank) {
        auto shard = torch::empty({kTokens, layout.localWidth()}, torch::dtype(torch::kBFloat16));
        shard.narrow(1, 0, layout.localLatent()).fill_(rank + 1);
        shard.narrow(1, layout.localLatent(), layout.localSuffix()).fill_(100 + rank);
        shards.push_back(shard);
    }
    auto destination = torch::zeros({kTokens, K3MlaCacheTpLayout::kFullWidth}, torch::dtype(torch::kBFloat16));

    layout.reconstruct(shards, destination);

    for (int rank = 0; rank < kTp; ++rank) {
        EXPECT_TRUE(torch::all(destination.narrow(1, rank * layout.localLatent(), layout.localLatent()) == rank + 1)
                        .item<bool>());
        EXPECT_TRUE(torch::all(destination.narrow(1,
                                                  K3MlaCacheTpLayout::kFullLatent + rank * layout.localSuffix(),
                                                  layout.localSuffix())
                                   == 100 + rank)
                        .item<bool>());
    }
}

TEST(MlaCacheTpTransferTest, KeepsKdaRankToRankAndMlaOwnerOnly) {
    std::vector<std::string> peers{"p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"};
    auto owner_plan = makeK3MlaCacheTpPeerPlan(peers, 3, 3);
    ASSERT_EQ(owner_plan.kda_peer_addrs, std::vector<std::string>({"p3"}));
    ASSERT_EQ(owner_plan.mla_peer_addrs, peers);

    auto non_owner_plan = makeK3MlaCacheTpPeerPlan(peers, 4, 3);
    ASSERT_EQ(non_owner_plan.kda_peer_addrs, std::vector<std::string>({"p4"}));
    ASSERT_TRUE(non_owner_plan.mla_peer_addrs.empty());
}

TEST(MlaCacheTpTransferTest, RejectsMissingShardAndInvalidPlacement) {
    K3MlaCacheTpLayout layout(8);
    auto destination = torch::zeros({1, K3MlaCacheTpLayout::kFullWidth}, torch::dtype(torch::kBFloat16));
    std::vector<torch::Tensor> missing(7,
                                       torch::zeros({1, layout.localWidth()}, torch::dtype(torch::kBFloat16)));
    EXPECT_THROW(layout.reconstruct(missing, destination), std::invalid_argument);
    EXPECT_THROW(makeK3MlaCacheTpPeerPlan({"p0", "p1"}, 0, 2), std::invalid_argument);
}

}  // namespace rtp_llm
