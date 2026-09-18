#include "rtp_llm/cpp/normal_engine/EngramTokenHistory.h"
#include "gtest/gtest.h"

namespace rtp_llm {

TEST(EngramTokenHistoryTest, PrefixAndChunkMatchFullPrompt) {
    const std::vector<int> tokens{10, 20, 30, 40, 50};
    int32_t                full[20];
    fillEngramTokenWindows(tokens, 0, tokens.size(), full);
    EXPECT_EQ(full[0], 10);
    EXPECT_EQ(full[1], -1);
    EXPECT_EQ(full[2], -1);
    EXPECT_EQ(full[3], -1);
    int32_t chunk[8];
    fillEngramTokenWindows(tokens, 3, 2, chunk);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(chunk[i], full[12 + i]);
    }
    EXPECT_THROW(fillEngramTokenWindows(tokens, 4, 2, chunk), std::invalid_argument);
}

TEST(EngramTokenHistoryTest, VerifyUsesCandidatesAndRollbackReplacesThem) {
    const int32_t anchor[] = {50, 40, 30, 20};
    const int32_t verify[] = {50, 60, 70, 80, 90, 100};
    int32_t       windows[24];
    extendEngramVerifyWindows(anchor, verify, 6, windows);
    for (int row = 0; row < 6; ++row) {
        for (int lag = 0; lag < 4; ++lag) {
            const int source = row - lag;
            EXPECT_EQ(windows[row * 4 + lag], source >= 0 ? verify[source] : anchor[-source]);
        }
    }
    const int32_t retry[] = {50, 999};
    extendEngramVerifyWindows(anchor, retry, 2, windows);
    EXPECT_EQ(windows[4], 999);
    EXPECT_EQ(windows[5], 50);
    EXPECT_EQ(windows[6], 40);
    EXPECT_EQ(windows[7], 30);
}

}  // namespace rtp_llm
