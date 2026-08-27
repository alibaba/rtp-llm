#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"
#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/FullPrefixInvariantScanner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::makeStructuralDevicePool;
using block_tree_cache_test::prepareGroupSetsForTest;

std::vector<GroupSetPtr> makeFullGroupSet() {
    std::vector<GroupSetPtr> group_sets;
    group_sets.push_back(
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{makeStructuralDevicePool(0)}, nullptr, nullptr));
    prepareGroupSetsForTest(group_sets);
    return group_sets;
}

// FULL, SWA and LINEAR in one tree so a single walk proves both cross-GroupSet
// independence and non-FULL exclusion.
std::vector<GroupSetPtr> makeMixedGroupSets() {
    std::vector<GroupSetPtr> group_sets;
    group_sets.push_back(
        std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{makeStructuralDevicePool(0)}, nullptr, nullptr));
    group_sets.push_back(std::make_shared<SWAGroupSet>(
        128, 64, std::vector<DeviceBlockPoolPtr>{makeStructuralDevicePool(1)}, nullptr, nullptr));
    group_sets.push_back(std::make_shared<LinearGroupSet>(
        std::vector<DeviceBlockPoolPtr>{makeStructuralDevicePool(2)}, nullptr, nullptr));
    prepareGroupSetsForTest(group_sets);
    return group_sets;
}

GroupSetResource deviceRes() {
    GroupSetResource resource;
    resource.device_blocks = {7};
    return resource;
}

GroupSetResource loadingDeviceRes() {
    GroupSetResource resource = deviceRes();
    resource.transfer_state   = GroupSetTransferState::LOADING;
    return resource;
}

GroupSetResource hostRes() {
    GroupSetResource resource;
    resource.host_block = 7;
    return resource;
}

GroupSetResource diskRes() {
    GroupSetResource resource;
    resource.disk_block = 7;
    return resource;
}

GroupSetResource emptyRes() {
    return GroupSetResource{};
}

// Builds node chains through the public BlockTree API, then overwrites the installed
// resources with synthetic tiers. The originals are restored on destruction so
// ~BlockTree unreferences exactly the blocks insertNode took ownership of, which lets the
// tests fabricate Host/Disk tiers without real lower-tier pools.
class SyntheticTree {
public:
    explicit SyntheticTree(std::vector<GroupSetPtr> group_sets):
        group_set_count_(group_sets.size()), tree_(std::move(group_sets)) {}

    ~SyntheticTree() {
        for (const auto& entry : pristine_) {
            entry.first->group_set_resources = entry.second;
        }
    }

    // Each entry of `path` holds the per-group-set resources of one node, shallowest first.
    std::vector<TreeNode*> addPath(const std::vector<std::vector<GroupSetResource>>& path) {
        CacheKeysType                              keys;
        std::vector<std::vector<GroupSetResource>> seed(path.size());
        for (size_t i = 0; i < path.size(); ++i) {
            keys.push_back(next_key_++);
            seed[i].resize(group_set_count_);
            for (size_t group_set_id = 0; group_set_id < group_set_count_; ++group_set_id) {
                seed[i][group_set_id].device_blocks = {next_block_++};
            }
        }
        BlockTreeInsertResult result = tree_.insertNode(keys, seed, /*collect_path=*/false);
        RTP_LLM_CHECK(result.inserted_nodes.size() == path.size());
        for (size_t i = 0; i < path.size(); ++i) {
            TreeNode* node = result.inserted_nodes[i];
            pristine_.emplace_back(node, node->group_set_resources);
            node->group_set_resources = path[i];
        }
        return result.inserted_nodes;
    }

    std::vector<TreeNode*> addSingleGroupPath(const std::vector<GroupSetResource>& resources) {
        std::vector<std::vector<GroupSetResource>> path;
        path.reserve(resources.size());
        for (const GroupSetResource& resource : resources) {
            path.push_back({resource});
        }
        return addPath(path);
    }

    std::vector<FullViolationDetail> detectAll() const {
        std::vector<FullViolationDetail> details;
        tree_.visitNodeRangeLocked(
            0, tree_.size(), tree_.size(), [&](const TreeNode& node) { detectNodeViolations(tree_, node, details); });
        return details;
    }

    BlockTree& tree() {
        return tree_;
    }

private:
    size_t                                                           group_set_count_;
    BlockTree                                                        tree_;
    CacheKeyType                                                     next_key_{100};
    BlockIdxType                                                     next_block_{1};
    std::vector<std::pair<TreeNode*, std::vector<GroupSetResource>>> pristine_;
};

template<typename Predicate>
bool waitFor(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return predicate();
}

// ---------------------------------------------------------------------------
// 1. FULL path and stability matrix
// ---------------------------------------------------------------------------

struct ExpectedViolation {
    FullViolationType type;
    int               parent_index;  // index into the path
    int               current_index;
    bool              stable{true};
};

struct PathCase {
    std::string                    name;
    std::vector<GroupSetResource>  path;
    std::vector<ExpectedViolation> expected;
};

TEST(FullPrefixDetectorTest, PathAndStabilityMatrix) {
    const std::vector<PathCase> cases{
        // Legal shapes: descending tiers, a lower-tier-only path, a stable hole and the
        // root sentinel edge (root(EMPTY) -> first data node is not a gap).
        {"D-D-H-K", {deviceRes(), deviceRes(), hostRes(), diskRes()}, {}},
        {"H-K", {hostRes(), diskRes()}, {}},
        {"D-D-E-E", {deviceRes(), deviceRes(), emptyRes(), emptyRes()}, {}},
        // lower tier followed by Device.
        {"D-H-D", {deviceRes(), hostRes(), deviceRes()}, {{FullViolationType::LOWER_TO_DEVICE, 1, 2}}},
        // Data after a stable hole.
        {"D-E-D", {deviceRes(), emptyRes(), deviceRes()}, {{FullViolationType::GAP_TO_DATA, 1, 2}}},
        // A busy (LOADING) endpoint makes the edge transient rather than stable.
        {"D-H-Dloading",
         {deviceRes(), hostRes(), loadingDeviceRes()},
         {{FullViolationType::LOWER_TO_DEVICE, 1, 2, /*stable=*/false}}},
    };

    for (const PathCase& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        SyntheticTree tree(makeFullGroupSet());
        const auto    nodes   = tree.addSingleGroupPath(test_case.path);
        const auto    details = tree.detectAll();

        ASSERT_EQ(details.size(), test_case.expected.size());
        for (size_t i = 0; i < test_case.expected.size(); ++i) {
            const ExpectedViolation& expected = test_case.expected[i];
            EXPECT_EQ(details[i].type, expected.type) << "violation " << i;
            EXPECT_EQ(details[i].group_set_id, 0u);
            EXPECT_EQ(details[i].stable, expected.stable) << "violation " << i;
            EXPECT_EQ(details[i].parent.cache_key, nodes[expected.parent_index]->cache_key) << "violation " << i;
            EXPECT_EQ(details[i].current.cache_key, nodes[expected.current_index]->cache_key) << "violation " << i;
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Only FULL GroupSets are judged on the prefix invariant
// ---------------------------------------------------------------------------

TEST(FullPrefixDetectorTest, NonFullGroupSetsAreNotReported) {
    SyntheticTree tree(makeMixedGroupSets());
    // Same "suspicious" shape on every GroupSet: FULL walks D-H-D, SWA walks D-H-D,
    // LINEAR walks D-E-D. Only the FULL group may report a path violation.
    tree.addPath({
        {deviceRes(), deviceRes(), deviceRes()},
        {hostRes(), hostRes(), emptyRes()},
        {deviceRes(), deviceRes(), deviceRes()},
    });

    const auto details = tree.detectAll();
    ASSERT_EQ(details.size(), 1u);
    EXPECT_EQ(details[0].group_set_id, 0u);
    EXPECT_EQ(details[0].type, FullViolationType::LOWER_TO_DEVICE);
}

// ---------------------------------------------------------------------------
// 3. Invalid resource: locatable reason, no derived path violation
// ---------------------------------------------------------------------------

TEST(FullPrefixDetectorTest, InvalidResourceReportsReasonInsteadOfPathViolation) {
    // A resource living in two serving tiers at once. Its top tier is HOST, which would
    // otherwise make the DEVICE child look like lower_to_device; the resource itself is
    // inconsistent, so only that is reported.
    GroupSetResource multi_tier;
    multi_tier.host_block = 2;
    multi_tier.disk_block  = 3;

    SyntheticTree tree(makeFullGroupSet());
    const auto    nodes   = tree.addSingleGroupPath({deviceRes(), multi_tier, deviceRes()});
    const auto    details = tree.detectAll();

    ASSERT_EQ(details.size(), 1u);
    EXPECT_EQ(details[0].type, FullViolationType::INVALID_RESOURCE);
    EXPECT_EQ(details[0].reason, InvalidResourceReason::MULTI_TIER);
    EXPECT_EQ(details[0].current.cache_key, nodes[1]->cache_key);
}

// ---------------------------------------------------------------------------
// 4. Background thread: detects, logs, stops, leaves the data plane intact
// ---------------------------------------------------------------------------

TEST(FullPrefixScannerTest, BackgroundScanLogsViolationWithoutAffectingTreeOperations) {
    test::TestLogCapture log_capture("full_prefix_scan");

    SyntheticTree          tree(makeFullGroupSet());
    const auto             nodes = tree.addSingleGroupPath({deviceRes(), hostRes(), deviceRes()});
    const CacheKeysType    violation_path{nodes[0]->cache_key, nodes[1]->cache_key, nodes[2]->cache_key};
    const GroupSetResource pristine_resource = nodes[2]->group_set_resources[0];

    std::mutex                 cache_mutex;
    FullPrefixInvariantScanner scanner(tree.tree(), cache_mutex, /*interval_ms=*/1000);
    ASSERT_TRUE(scanner.start());

    const std::string expected_key = "key=" + std::to_string(nodes[2]->cache_key);
    ASSERT_TRUE(waitFor([&] {
        const std::string content = log_capture.content();
        return content.find("event=block_tree_full_violation") != std::string::npos
               && content.find("type=lower_to_device") != std::string::npos
               && content.find("group_set_id=0") != std::string::npos
               && content.find(expected_key) != std::string::npos;
    })) << log_capture.content();

    // Normal tree operations still work while the scanner is alive.
    size_t size_after_insert = 0;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        tree.addSingleGroupPath({deviceRes(), deviceRes()});
        size_after_insert = tree.tree().size();
        EXPECT_EQ(tree.tree().findNode(violation_path).size(), violation_path.size());
    }

    scanner.stop();

    // The scanner observed the anomaly without touching it.
    EXPECT_EQ(tree.tree().size(), size_after_insert);
    EXPECT_EQ(nodes[2]->group_set_resources[0].device_blocks, pristine_resource.device_blocks);
    EXPECT_EQ(nodes[2]->group_set_resources[0].transfer_state, pristine_resource.transfer_state);
}

}  // namespace
}  // namespace rtp_llm
