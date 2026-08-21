#include <gtest/gtest.h>

#include <algorithm>
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
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::makeStructuralDevicePool;
using block_tree_cache_test::prepareGroupSetsForTest;

std::vector<GroupSetPtr> makeFullGroupSets(size_t count) {
    std::vector<GroupSetPtr> group_sets;
    group_sets.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        group_sets.push_back(std::make_shared<FullGroupSet>(
            std::vector<DeviceBlockPoolPtr>{makeStructuralDevicePool(i)}, nullptr, nullptr));
    }
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
    resource.disk_slot = 7;
    return resource;
}

GroupSetResource emptyRes() {
    return GroupSetResource{};
}

std::vector<GroupSetResource> cleanPath(size_t length) {
    return std::vector<GroupSetResource>(length, deviceRes());
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
        std::this_thread::yield();
    }
    return predicate();
}

FullPrefixScanOptions testOptions(size_t nodes_per_round, size_t max_details = kFullPrefixScanMaxDetailsPerCycle) {
    FullPrefixScanOptions options;
    options.interval_ms           = 0;
    options.nodes_per_round       = nodes_per_round;
    options.max_details_per_cycle = max_details;
    return options;
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
        // Legal shapes: descending tiers, lower-tier siblings and a stable hole.
        {"D-D-H-K", {deviceRes(), deviceRes(), hostRes(), diskRes()}, {}},
        {"H-K", {hostRes(), diskRes()}, {}},
        {"D-D-E-E", {deviceRes(), deviceRes(), emptyRes(), emptyRes()}, {}},
        // root -> first data node is not a gap.
        {"root-D", {deviceRes()}, {}},
        // lower tier followed by Device.
        {"H-D", {hostRes(), deviceRes()}, {{FullViolationType::LOWER_TO_DEVICE, 0, 1}}},
        {"D-H-D", {deviceRes(), hostRes(), deviceRes()}, {{FullViolationType::LOWER_TO_DEVICE, 1, 2}}},
        // Only the offending K->D edge is reported; H->K in front is legal.
        {"D-H-K-D", {deviceRes(), hostRes(), diskRes(), deviceRes()}, {{FullViolationType::LOWER_TO_DEVICE, 2, 3}}},
        // Data after a stable hole.
        {"D-E-D", {deviceRes(), emptyRes(), deviceRes()}, {{FullViolationType::GAP_TO_DATA, 1, 2}}},
        // Two independent local violations on one path.
        {"D-E-H-D",
         {deviceRes(), emptyRes(), hostRes(), deviceRes()},
         {{FullViolationType::GAP_TO_DATA, 1, 2}, {FullViolationType::LOWER_TO_DEVICE, 2, 3}}},
        // A busy (LOADING) endpoint makes the edge transient rather than stable.
        {"D-H-Dloading",
         {deviceRes(), hostRes(), loadingDeviceRes()},
         {{FullViolationType::LOWER_TO_DEVICE, 1, 2, /*stable=*/false}}},
    };

    for (const PathCase& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        SyntheticTree tree(makeFullGroupSets(1));
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
// 2. GroupSet routing: independence + non-FULL exclusion
// ---------------------------------------------------------------------------

TEST(FullPrefixDetectorTest, GroupSetRouting) {
    SyntheticTree tree(makeMixedGroupSets());
    // Same "suspicious" shape on every GroupSet: FULL walks D-H-D, SWA walks D-H-D,
    // LINEAR walks D-E-D. Only the FULL group may report a path violation, and groups are
    // judged independently, so exactly one violation on group 0 is expected.
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
// 3. Invalid resources + parent-edge suppression
// ---------------------------------------------------------------------------

struct InvalidResourceCase {
    std::string           name;
    GroupSetResource      resource;
    InvalidResourceReason reason;
    bool                  stable;
};

std::vector<InvalidResourceCase> invalidResourceCases() {
    GroupSetResource multi_tier;
    multi_tier.device_blocks = {1};
    multi_tier.host_block    = 2;

    GroupSetResource partial_device;
    partial_device.device_blocks = {1, NULL_BLOCK_IDX};

    GroupSetResource idle_detached;
    idle_detached.device_blocks     = {1};
    idle_detached.transfer_detached = true;

    GroupSetResource busy_empty;
    busy_empty.transfer_state = GroupSetTransferState::LOADING;

    return {
        {"multi_tier", multi_tier, InvalidResourceReason::MULTI_TIER, true},
        {"partial_device", partial_device, InvalidResourceReason::PARTIAL_DEVICE, true},
        {"idle_detached", idle_detached, InvalidResourceReason::IDLE_DETACHED, true},
        {"busy_empty", busy_empty, InvalidResourceReason::BUSY_EMPTY, false},
    };
}

TEST(FullPrefixDetectorTest, InvalidResourceMatrixAndEdgeSuppression) {
    for (const InvalidResourceCase& test_case : invalidResourceCases()) {
        SCOPED_TRACE(test_case.name);
        SyntheticTree tree(makeFullGroupSets(1));
        const auto    nodes   = tree.addSingleGroupPath({deviceRes(), test_case.resource});
        const auto    details = tree.detectAll();
        ASSERT_EQ(details.size(), 1u);
        EXPECT_EQ(details[0].type, FullViolationType::INVALID_RESOURCE);
        EXPECT_EQ(details[0].reason, test_case.reason);
        EXPECT_EQ(details[0].stable, test_case.stable);
        EXPECT_EQ(details[0].current.cache_key, nodes[1]->cache_key);
    }

    // Valid steady-state resources produce no reason.
    for (const GroupSetResource& resource : {deviceRes(), hostRes(), diskRes(), emptyRes()}) {
        EXPECT_EQ(invalidResourceReason(resource), InvalidResourceReason::NONE);
    }

    // An invalid parent suppresses the child edge. The HOST+DISK parent has top tier HOST,
    // which would otherwise flag its DEVICE child as lower_to_device; because the parent is
    // itself invalid, only the parent's invalid_resource is reported.
    GroupSetResource host_disk_parent;
    host_disk_parent.host_block = 2;
    host_disk_parent.disk_slot  = 3;
    SyntheticTree tree(makeFullGroupSets(1));
    tree.addSingleGroupPath({deviceRes(), host_disk_parent, deviceRes()});
    const auto details = tree.detectAll();
    ASSERT_EQ(details.size(), 1u);
    EXPECT_EQ(details[0].type, FullViolationType::INVALID_RESOURCE);
    EXPECT_EQ(details[0].reason, InvalidResourceReason::MULTI_TIER);
}

// ---------------------------------------------------------------------------
// 4. Log formatting
// ---------------------------------------------------------------------------

TEST(FullPrefixDetectorTest, FormatsViolationDetails) {
    FullViolationDetail edge;
    edge.group_set_id = 0;
    edge.type         = FullViolationType::LOWER_TO_DEVICE;
    edge.stable       = true;
    edge.parent       = NodeBrief{303, Tier::DISK, GroupSetTransferState::IDLE, 0};
    edge.current      = NodeBrief{404, Tier::DEVICE, GroupSetTransferState::IDLE, 0};
    EXPECT_EQ(formatViolationDetail(edge, 0, 0),
              "event=block_tree_full_violation group_set_id=0 type=lower_to_device status=stable "
              "edge=[DISK(key=303) -> DEVICE(key=404)] world_rank=0 local_rank=0");

    FullViolationDetail gap;
    gap.group_set_id = 1;
    gap.type         = FullViolationType::GAP_TO_DATA;
    gap.stable       = false;
    gap.parent       = NodeBrief{202, Tier::NONE, GroupSetTransferState::IDLE, 0};
    gap.current      = NodeBrief{303, Tier::HOST, GroupSetTransferState::LOADING, 0};
    EXPECT_EQ(formatViolationDetail(gap, 2, 3),
              "event=block_tree_full_violation group_set_id=1 type=gap_to_data status=transient "
              "edge=[EMPTY(key=202) -> HOST(key=303,state=LOADING)] world_rank=2 local_rank=3");

    GroupSetResource multi_tier;
    multi_tier.device_blocks = {1};
    multi_tier.host_block    = 2;
    TreeNode node;
    node.cache_key = 202;
    FullViolationDetail resource;
    resource.group_set_id = 0;
    resource.type         = FullViolationType::INVALID_RESOURCE;
    resource.reason       = InvalidResourceReason::MULTI_TIER;
    resource.stable       = true;
    resource.current      = makeNodeBrief(node, multi_tier);
    EXPECT_EQ(formatViolationDetail(resource, 0, 0),
              "event=block_tree_resource_violation group_set_id=0 type=invalid_resource reason=multi_tier "
              "status=stable key=202 tiers=[DEVICE,HOST] transfer_state=IDLE world_rank=0 local_rank=0");
}

// ---------------------------------------------------------------------------
// 5. Bounded batches, cycle pinning and deferral
// ---------------------------------------------------------------------------

TEST(FullPrefixScannerTest, BoundedBatchesFindViolationAndPinCycleEnd) {
    SyntheticTree tree(makeFullGroupSets(1));
    // Violation edge H->D at node index 2; a trailing clean node keeps it off the last
    // slot so the count survives the batch that scans it (a completing batch resets stats).
    tree.addSingleGroupPath({deviceRes(), hostRes(), deviceRes(), deviceRes()});

    std::mutex                 cache_mutex;
    FullPrefixInvariantScanner scanner(tree.tree(), cache_mutex, testOptions(/*nodes_per_round=*/1));

    // One node per round: the violation is only reached on the third batch.
    scanner.runOneBatch();
    scanner.runOneBatch();
    EXPECT_EQ(scanner.stats().stable_violations, 0u);

    scanner.runOneBatch();
    const auto mid_cycle = scanner.stats();
    EXPECT_TRUE(mid_cycle.cycle_active);
    EXPECT_EQ(mid_cycle.stable_violations, 1u);
    EXPECT_EQ(mid_cycle.details_logged, 1u);
    EXPECT_EQ(mid_cycle.cycles_completed, 0u);

    // A node added mid-cycle is pinned out of the current cycle (cycle_end fixed at start).
    tree.addSingleGroupPath({deviceRes(), hostRes(), deviceRes(), deviceRes()});
    scanner.runOneBatch();  // scans node 3, reaches the pinned end, completes cycle 0
    ASSERT_EQ(scanner.stats().cycles_completed, 1u);

    // Cycle 1 rescans everything, so it must observe both the original and the deferred
    // violation simultaneously before the cycle completes.
    size_t max_stable = 0;
    for (int i = 0; i < 20 && scanner.stats().cycles_completed < 2; ++i) {
        scanner.runOneBatch();
        max_stable = std::max(max_stable, scanner.stats().stable_violations);
    }
    EXPECT_EQ(scanner.stats().cycles_completed, 2u);
    EXPECT_GE(max_stable, 2u);
}

// ---------------------------------------------------------------------------
// 6. Detail cap
// ---------------------------------------------------------------------------

TEST(FullPrefixScannerTest, DetailCapSuppressesButCountsAll) {
    SyntheticTree tree(makeFullGroupSets(1));
    tree.addSingleGroupPath({deviceRes(), hostRes(), deviceRes()});
    tree.addSingleGroupPath({deviceRes(), hostRes(), deviceRes()});
    tree.addSingleGroupPath({deviceRes(), hostRes(), deviceRes()});
    tree.addSingleGroupPath(cleanPath(1));

    std::mutex                 cache_mutex;
    FullPrefixInvariantScanner scanner(tree.tree(), cache_mutex, testOptions(/*nodes_per_round=*/9, /*max_details=*/1));

    scanner.runOneBatch();
    const auto stats = scanner.stats();
    EXPECT_EQ(stats.stable_violations, 3u);
    EXPECT_EQ(stats.details_logged, 1u);
    EXPECT_EQ(stats.details_suppressed, 2u);
}

// ---------------------------------------------------------------------------
// 7. Background thread: waits for the mutex, detects, leaves data-plane intact
// ---------------------------------------------------------------------------

TEST(FullPrefixScannerTest, BackgroundScanFindsViolationWithoutAffectingTreeOperations) {
    SyntheticTree tree(makeFullGroupSets(1));
    // Violation at node index 2; trailing clean nodes keep the cycle multi-batch so the
    // violation count is observable mid-cycle instead of only at completion.
    tree.addSingleGroupPath({deviceRes(), hostRes(), deviceRes(), deviceRes(), deviceRes(), deviceRes()});

    std::mutex cache_mutex;
    auto       options  = testOptions(/*nodes_per_round=*/1);
    options.interval_ms = 1;
    FullPrefixInvariantScanner scanner(tree.tree(), cache_mutex, options);

    size_t size_with_insert = 0;
    {
        std::unique_lock<std::mutex> lock(cache_mutex);
        ASSERT_TRUE(scanner.start());
        // The first batch fires immediately, then blocks acquiring the cache mutex we hold.
        ASSERT_TRUE(waitFor([&] { return scanner.stats().batches_started >= 1; }));
        // Proof it waits rather than scanning without the lock.
        EXPECT_EQ(scanner.stats().cycles_completed, 0u);

        // A normal tree mutation proceeds while the scanner is alive and blocked.
        tree.addSingleGroupPath(cleanPath(2));
        size_with_insert = tree.tree().size();
    }

    // Once the mutex is released, the scanner runs and reports the violation.
    EXPECT_TRUE(waitFor([&] { return scanner.stats().stable_violations >= 1; }));

    scanner.stop();
    // The concurrent insert took effect and the scanner never mutated the tree.
    EXPECT_EQ(tree.tree().size(), size_with_insert);
}

}  // namespace
}  // namespace rtp_llm
