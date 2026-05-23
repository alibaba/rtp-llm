#include "rtp_llm/cpp/cache/connector/memory/DiskSpillProtocol.h"

#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

TEST(DiskSpillProtocolTest, PackUnpackBitmapRoundtrip) {
    std::vector<bool> bits(17);
    bits[0]  = true;
    bits[3]  = true;
    bits[7]  = true;
    bits[8]  = true;
    bits[16] = true;
    const auto packed = packBitmap(bits);
    EXPECT_EQ(packed.size(), (17u + 7) / 8);
    const auto roundtrip = unpackBitmap(packed, 17);
    EXPECT_EQ(roundtrip, bits);
}

TEST(DiskSpillProtocolTest, UnpackRejectsWrongSize) {
    // packed is 2 bytes but slot_count=24 needs 3 bytes -> reject
    std::string packed(2, '\0');
    packed[0] = 0x05;
    packed[1] = 0xFF;
    EXPECT_TRUE(unpackBitmap(packed, 24).empty()) << "expects 3 bytes, given 2";
    // packed is 3 bytes but slot_count=8 needs 1 byte -> reject
    std::string oversized(3, '\0');
    EXPECT_TRUE(unpackBitmap(oversized, 8).empty()) << "expects 1 byte, given 3";
}

TEST(DiskSpillProtocolTest, OpSequenceFirstAcceptedThenStrictlyIncreases) {
    OpSequenceTracker t;
    EXPECT_EQ(t.checkReceived(100), OpSequenceTracker::CheckResult::OK);
    EXPECT_EQ(t.checkReceived(101), OpSequenceTracker::CheckResult::OK);
    EXPECT_EQ(t.checkReceived(102), OpSequenceTracker::CheckResult::OK);
    EXPECT_EQ(t.checkReceived(102), OpSequenceTracker::CheckResult::DUPLICATE);
    EXPECT_EQ(t.checkReceived(101), OpSequenceTracker::CheckResult::OUT_OF_ORDER);
    EXPECT_EQ(t.checkReceived(104), OpSequenceTracker::CheckResult::SKIPPED);
}

TEST(DiskSpillProtocolTest, OpSequenceMasterSideMonotonicNext) {
    OpSequenceTracker t;
    EXPECT_EQ(t.next(), 1u);
    EXPECT_EQ(t.next(), 2u);
    EXPECT_EQ(t.next(), 3u);
}

TEST(DiskSpillProtocolTest, OpSequenceReset) {
    OpSequenceTracker t;
    EXPECT_EQ(t.checkReceived(50), OpSequenceTracker::CheckResult::OK);
    t.reset();
    EXPECT_EQ(t.checkReceived(1), OpSequenceTracker::CheckResult::OK);
    EXPECT_EQ(t.checkReceived(2), OpSequenceTracker::CheckResult::OK);
}

}  // namespace
}  // namespace rtp_llm
