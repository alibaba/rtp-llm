#include "aios/navi_sched/collector/PPCacheResidentState.h"
#include "unittest/unittest.h"
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <set>

using namespace navi_sched;
class PPCacheResidentStateTest : public TESTBASE {};
namespace {
std::string hash(uint64_t n)
{
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(64) << n;
    return out.str();
}
std::string identity(uint32_t group, const std::vector<std::string> &hashes)
{
    std::string result = "{\"group_id\":" + std::to_string(group) + ",\"block_hashes\":[";
    for (size_t i = 0; i < hashes.size(); ++i)
        result += (i ? ",\"" : "\"") + hashes[i] + "\"";
    return result + "]}";
}
}

TEST_F(PPCacheResidentStateTest, testAllFullGroupsAndSeparateTail)
{
    PPCacheResidentState state(
        {PPCacheGroupRole::FULL, PPCacheGroupRole::FULL, PPCacheGroupRole::TAIL},
        {1, 1, 1});
    EXPECT_TRUE(state.resident({"a", identity(0, {hash(8)})}).addedFull.empty());
    auto full = state.resident({"b", identity(1, {hash(8)})});
    EXPECT_EQ(std::vector<uint64_t>({8}), full.addedFull);
    EXPECT_TRUE(full.addedTail.empty());
    auto tail = state.resident({"t", identity(2, {hash(9)})});
    EXPECT_TRUE(tail.addedFull.empty());
    EXPECT_EQ(std::vector<uint64_t>({9}), tail.addedTail);
    EXPECT_EQ(std::vector<uint64_t>({8}), state.removed("a").removedFull);
    EXPECT_EQ(std::vector<uint64_t>({9}), state.removed("t").removedTail);
}

TEST_F(PPCacheResidentStateTest, testOverlappingObjectsAndRepeatedNotification)
{
    PPCacheResidentState state({PPCacheGroupRole::FULL}, {1});
    const auto value = identity(0, {hash(8)});
    EXPECT_EQ(std::vector<uint64_t>({8}), state.resident({"a", value}).addedFull);
    EXPECT_TRUE(state.resident({"a", value}).addedFull.empty());
    EXPECT_TRUE(state.resident({"b", value}).addedFull.empty());
    EXPECT_TRUE(state.removed("a").removedFull.empty());
    EXPECT_TRUE(state.removed("a").removedFull.empty());
    EXPECT_EQ(std::vector<uint64_t>({8}), state.removed("b").removedFull);
}

TEST_F(PPCacheResidentStateTest, testScaledFullExpandsAndScaledTailUsesLastHash)
{
    PPCacheResidentState state({PPCacheGroupRole::FULL, PPCacheGroupRole::TAIL}, {2, 2});
    const auto combined = hash(8) + hash(9);
    const auto full = state.resident({"a", identity(0, {combined})});
    EXPECT_EQ(std::set<uint64_t>({8, 9}),
              std::set<uint64_t>(full.addedFull.begin(), full.addedFull.end()));
    const auto tail = state.resident({"t", identity(1, {combined})});
    EXPECT_EQ(std::vector<uint64_t>({9}), tail.addedTail);
    EXPECT_EQ(std::vector<uint64_t>({8}),
              state.resident({"single", identity(1, {hash(8)})}).addedTail);
}

TEST_F(PPCacheResidentStateTest, testMalformedReplacementDoesNotRemoveOldState)
{
    PPCacheResidentState state({PPCacheGroupRole::FULL}, {1});
    ASSERT_EQ(std::vector<uint64_t>({8}),
              state.resident({"a", identity(0, {hash(8)})}).addedFull);
    EXPECT_THROW(state.resident({"a", identity(0, {"01"})}), std::invalid_argument);
    EXPECT_THROW(state.replace({{"a", identity(0, {hash(9)})}, {"b", "bad-json"}}),
                 std::invalid_argument);
    EXPECT_EQ(std::vector<uint64_t>({8}), state.removed("a").removedFull);
}

TEST_F(PPCacheResidentStateTest, testSnapshotReplacesAndEmptyMetadataRetracts)
{
    PPCacheResidentState state({PPCacheGroupRole::FULL}, {1});
    state.resident({"old", identity(0, {hash(8)})});
    const auto snapshot =
        state.replace({{"a", identity(0, {hash(9)})}, {"unrelated", ""}});
    ASSERT_TRUE(snapshot.fullSnapshot);
    EXPECT_EQ(std::vector<uint64_t>({9}), snapshot.addedFull);
    EXPECT_TRUE(state.removed("old").removedFull.empty());
    EXPECT_EQ(std::vector<uint64_t>({9}), state.resident({"a", ""}).removedFull);
    EXPECT_TRUE(state.replace({}).addedFull.empty());
}

TEST_F(PPCacheResidentStateTest, testObjectIdentityChangeAndInvalidPolicies)
{
    PPCacheResidentState state({PPCacheGroupRole::FULL}, {1});
    state.resident({"a", identity(0, {hash(8)})});
    const auto delta = state.resident({"a", identity(0, {hash(9)})});
    EXPECT_EQ(std::vector<uint64_t>({8}), delta.removedFull);
    EXPECT_EQ(std::vector<uint64_t>({9}), delta.addedFull);
    EXPECT_THROW(
        state.replace({{"a", identity(0, {hash(8)})}, {"a", identity(0, {hash(9)})}}),
        std::invalid_argument);
    EXPECT_THROW(state.resident({"", identity(0, {hash(8)})}), std::invalid_argument);
    EXPECT_THROW(PPCacheResidentState({}, {}), std::invalid_argument);
    EXPECT_THROW(PPCacheResidentState({PPCacheGroupRole::FULL}, {0}),
                 std::invalid_argument);
}
