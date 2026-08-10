#include "rtp_llm/cpp/model_rpc/PrefillPeerSelector.h"

#include <set>

#include "gtest/gtest.h"

namespace rtp_llm {

TEST(PrefillPeerSelectorTest, SameInputIsStable) {
    const auto first = selectPrefillPeerIndex(123456789, 7, 0, 16);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(first, selectPrefillPeerIndex(123456789, 7, 0, 16));
    }
}

TEST(PrefillPeerSelectorTest, DecodeDpRanksCoverAllPrefillPeers) {
    std::set<size_t> selected_peers;
    for (int64_t dp_rank = 0; dp_rank < 16; ++dp_rank) {
        selected_peers.insert(selectPrefillPeerIndex(42, dp_rank, 0, 16));
    }

    EXPECT_EQ(selected_peers.size(), 16);
    for (size_t peer_index = 0; peer_index < 16; ++peer_index) {
        EXPECT_EQ(selected_peers.count(peer_index), 1);
    }
}

TEST(PrefillPeerSelectorTest, LocalWorkersRotateFromSameRequestAndDpRank) {
    const auto first = selectPrefillPeerIndex(42, 3, 0, 16);
    for (size_t worker_index = 0; worker_index < 16; ++worker_index) {
        EXPECT_EQ(selectPrefillPeerIndex(42, 3, worker_index, 16), (first + worker_index) % 16);
    }
}

TEST(PrefillPeerSelectorTest, RejectsInvalidTopology) {
    EXPECT_THROW(selectPrefillPeerIndex(42, 0, 0, 0), std::invalid_argument);
    EXPECT_THROW(selectPrefillPeerIndex(42, -1, 0, 16), std::invalid_argument);
}

TEST(PrefillPeerSelectorTest, MlaCp16DecodeDp16BalancesAcrossReplicatedPrefillPeers) {
    std::set<size_t> selected_peers;
    for (int64_t dp_rank = 0; dp_rank < 16; ++dp_rank) {
        selected_peers.insert(selectMlaPrefillPeerIndex(
            /*request_id=*/42,
            dp_rank,
            /*worker_index=*/0,
            /*worker_count=*/1,
            /*peer_count=*/16));
    }

    EXPECT_EQ(selected_peers.size(), 16);
    for (size_t peer_index = 0; peer_index < 16; ++peer_index) {
        EXPECT_EQ(selected_peers.count(peer_index), 1);
    }
}

TEST(PrefillPeerSelectorTest, MlaPreservesTopologyPeerGroups) {
    EXPECT_EQ(selectMlaPrefillPeerIndex(42, 0, 0, 4, 2), 0);
    EXPECT_EQ(selectMlaPrefillPeerIndex(42, 0, 1, 4, 2), 0);
    EXPECT_EQ(selectMlaPrefillPeerIndex(42, 0, 2, 4, 2), 1);
    EXPECT_EQ(selectMlaPrefillPeerIndex(42, 0, 3, 4, 2), 1);

    for (int64_t request_id = 0; request_id < 32; ++request_id) {
        EXPECT_LT(selectMlaPrefillPeerIndex(request_id, 0, 0, 2, 4), 2);
        EXPECT_GE(selectMlaPrefillPeerIndex(request_id, 0, 1, 2, 4), 2);
    }
}

TEST(PrefillPeerSelectorTest, MlaRejectsInvalidTopology) {
    EXPECT_THROW(selectMlaPrefillPeerIndex(42, 0, 0, 0, 16), std::invalid_argument);
    EXPECT_THROW(selectMlaPrefillPeerIndex(42, 0, 1, 1, 16), std::invalid_argument);
    EXPECT_THROW(selectMlaPrefillPeerIndex(42, 0, 0, 3, 5), std::invalid_argument);
}

}  // namespace rtp_llm
