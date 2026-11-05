#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/events/KVCMEventReport.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventQueue.h"

#include <gtest/gtest.h>
#include <rapidjson/document.h>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace rtp_llm {
namespace {

// Independent, deliberately small model of the receiver's index contract. It
// consumes the actual wire JSON, never KVCacheSnapshot or sender state helpers.
// It models successful serial requests, not server fencing/liveness/concurrency.
class IndexReceiver {
public:
    using IndexKey = std::tuple<std::string, std::string, std::string>;
    using Index = std::map<IndexKey, std::string>;

    void receive(const std::string& wire) {
        rapidjson::Document body;
        body.Parse(wire.c_str());
        if (body.HasParseError() || !body.IsObject()) {
            throw std::runtime_error("invalid report JSON");
        }
        EXPECT_STREQ(body["storage_type"].GetString(), "ST_EVENT_REPORT_L1P5");
        for (const auto& event : body["events"].GetArray()) {
            const std::string type = event["event_type"].GetString();
            if (type == "EVENT_BLOCK_ADD") {
                add(index, event["block_add"]);
            } else if (type == "EVENT_BLOCK_DELETE") {
                const auto& item = event["block_delete"];
                for (const auto& name : item["spec_names"].GetArray()) {
                    index.erase({item["block_key"].GetString(), item["medium"].GetString(), name.GetString()});
                }
            } else if (type == "EVENT_BLOCK_SNAPSHOT") {
                Index replacement;
                std::set<std::pair<std::string, std::string>> seen;
                for (const auto& item : event["block_snapshot"]["blocks"].GetArray()) {
                    EXPECT_TRUE(seen.emplace(item["block_key"].GetString(), item["medium"].GetString()).second);
                    add(replacement, item);
                }
                index = std::move(replacement);
            } else {
                throw std::runtime_error("unexpected event type");
            }
        }
    }

    bool contains(const std::string& key, const std::string& medium) const {
        return index.count({key, medium, "F"}) != 0;
    }
    Index index;

private:
    static void add(Index& target, const rapidjson::Value& item) {
        const std::string key = item["block_key"].GetString();
        const std::string medium = item["medium"].GetString();
        EXPECT_TRUE(medium == "hbm" || medium == "mem" || medium == "disk");
        for (const auto& spec : item["specs"].GetArray()) {
            EXPECT_STREQ(spec["name"].GetString(), "F");
            target[{key, medium, spec["name"].GetString()}] = spec["uri"].GetString();
        }
    }
};

// Only the publisher's worker scheduling/HTTP boundary is replaced. Admission,
// coalescing and serialization below are the production implementations.
class QueuedPublisher final: public KVCacheEventPublisher {
public:
    explicit QueuedPublisher(size_t capacity = 64): queue(capacity) {}
    bool start() noexcept override { return true; }
    void stop() noexcept override { queue.stop(); }
    PublisherStatus status() const noexcept override { return {}; }
    PublishResult tryPublish(KVCacheEvent event) noexcept override {
        return detail::toPublishResult(queue.tryPush(event));
    }
    void flush(const KVCacheEventPublisherContext& context, IndexReceiver& receiver) {
        if (queue.size() == 0) {
            return;
        }
        const auto batch = queue.waitPop(64, std::chrono::milliseconds(1));
        receiver.receive(detail::buildMutationReport(context, "mock-delta", detail::coalesceMutations(batch)));
    }
    detail::KVCacheEventQueue queue;
};

class BlockTreeReportEventTest: public ::testing::Test {
protected:
    void SetUp() override {
        context.instance_id = "mock-instance";
        context.host_ip_port = "127.0.0.1:19000";
        context.specs = {{"F", "event_report://127.0.0.1:19000/hbm"}};
        context.enabled_mediums = {KVCacheEventMedium::HBM, KVCacheEventMedium::HOST, KVCacheEventMedium::DISK};
        group = std::make_shared<GroupSet>(std::vector<std::string>{"hca_kv", "csa_kv", "indexer_kv"});
        tree = std::make_unique<BlockTree>(std::vector<GroupSetPtr>{group});
        publisher = std::make_shared<QueuedPublisher>();
        attach();
    }

    void attach() { tree->setEventPublisher(publisher, {"hca_kv"}, context.enabled_mediums); }

    TreeNode* insert(int64_t key, Tier tier) {
        GroupSetResource resource;
        resource.setBlocks(tier, tier == Tier::DEVICE ? BlockIdList{10, 11, 12} : BlockIdList{20});
        const auto result = tree->insertNode({key}, {{resource}}, true, false);
        EXPECT_EQ(result.path.size(), 1u);
        return result.path.at(0);
    }

    void flush() { publisher->flush(context, receiver); }
    void snapshot() {
        receiver.receive(detail::buildSnapshotReport(context, "mock-snapshot", tree->logicalCacheSnapshot()));
    }

    // Inject a transport outcome; the production tree decides what may publish.
    // This does not execute BlockTreeLoader/BlockTreeEvictor scheduling hooks.
    void state(TreeNode* node, GroupSetTransferState value, bool detached = false) {
        node->group_set_resources[0].transfer_state = value;
        node->group_set_resources[0].transfer_detached = detached;
        tree->refreshPublishedState(node);
    }
    void complete(TreeNode* node, Tier source, Tier target) {
        auto& resource = node->group_set_resources[0];
        resource.evictFromTier(source);
        resource.setBlocks(target, target == Tier::DEVICE ? BlockIdList{30, 31, 32} : BlockIdList{40});
        state(node, GroupSetTransferState::IDLE);
    }

    KVCacheEventPublisherContext context;
    GroupSetPtr group;
    std::unique_ptr<BlockTree> tree;
    std::shared_ptr<QueuedPublisher> publisher;
    IndexReceiver receiver;
};

TEST_F(BlockTreeReportEventTest, HostAndDiskInsertionAndRemovalReachReceiver) {
    for (const auto tier : {Tier::HOST, Tier::DISK}) {
        const std::string medium = tier == Tier::HOST ? "mem" : "disk";
        SCOPED_TRACE(medium);
        auto* node = insert(100, tier);
        flush();
        ASSERT_EQ(receiver.index.size(), 1u);
        EXPECT_TRUE(receiver.contains("100", medium));
        EXPECT_EQ(receiver.index.at({"100", medium, "F"}), "event_report://127.0.0.1:19000/" + medium);
        node->group_set_resources[0].evictFromTier(tier);
        tree->removeNodeAndEmptyAncestors(node);
        flush();
        EXPECT_TRUE(receiver.index.empty());
    }
}

TEST_F(BlockTreeReportEventTest, DemotionAndPromotionPublishOnlyCompletedTargets) {
    auto* node = insert(100, Tier::DEVICE);
    flush();
    ASSERT_TRUE(receiver.contains("100", "hbm"));
    state(node, GroupSetTransferState::DEMOTING);
    flush();
    EXPECT_TRUE(receiver.index.empty());
    complete(node, Tier::DEVICE, Tier::HOST);
    flush();
    ASSERT_TRUE(receiver.contains("100", "mem"));
    EXPECT_FALSE(receiver.contains("100", "hbm"));
    state(node, GroupSetTransferState::LOAD_PENDING);
    flush();
    EXPECT_TRUE(receiver.index.empty());
    state(node, GroupSetTransferState::LOADING);
    flush();
    EXPECT_TRUE(receiver.index.empty());
    complete(node, Tier::HOST, Tier::DEVICE);
    flush();
    EXPECT_TRUE(receiver.contains("100", "hbm"));
    EXPECT_FALSE(receiver.contains("100", "mem"));
}

TEST_F(BlockTreeReportEventTest, FailedOrCancelledTransfersRestoreSourceButDetachedSourcesStayHidden) {
    auto* node = insert(100, Tier::HOST);
    flush();
    for (const auto busy : {GroupSetTransferState::LOAD_PENDING, GroupSetTransferState::LOADING,
                           GroupSetTransferState::DEMOTING}) {
        state(node, busy);
        flush();
        EXPECT_TRUE(receiver.index.empty());
        state(node, GroupSetTransferState::IDLE);
        flush();
        EXPECT_TRUE(receiver.contains("100", "mem"));
    }
    state(node, GroupSetTransferState::LOADING, true);
    flush();
    state(node, GroupSetTransferState::IDLE, true);
    snapshot();
    EXPECT_TRUE(receiver.index.empty());
}

TEST_F(BlockTreeReportEventTest, IncompleteDeviceGroupIsNeverAdvertised) {
    auto* node = insert(100, Tier::HOST);
    flush();
    auto& resource = node->group_set_resources[0];
    resource.evictFromTier(Tier::HOST);
    resource.device_blocks = {10, 11};
    state(node, GroupSetTransferState::IDLE);
    flush();
    EXPECT_TRUE(receiver.index.empty());
    resource.device_blocks = {10, NULL_BLOCK_IDX, 12};
    state(node, GroupSetTransferState::IDLE);
    snapshot();
    EXPECT_TRUE(receiver.index.empty());
    resource.device_blocks = {10, 11, 12};
    state(node, GroupSetTransferState::IDLE);
    flush();
    EXPECT_TRUE(receiver.contains("100", "hbm"));
}

TEST_F(BlockTreeReportEventTest, RetieringInOneBatchDeletesOldMediumWithoutLosingNewMedium) {
    auto* node = insert(100, Tier::DEVICE);
    flush();
    state(node, GroupSetTransferState::DEMOTING);
    complete(node, Tier::DEVICE, Tier::HOST);
    flush();
    ASSERT_EQ(receiver.index.size(), 1u);
    EXPECT_TRUE(receiver.contains("100", "mem"));
    // The real insertion path replaces a lower-tier FULL value with freshly
    // computed DEVICE blocks and must automatically update both media.
    insert(100, Tier::DEVICE);
    flush();
    ASSERT_EQ(receiver.index.size(), 1u);
    EXPECT_TRUE(receiver.contains("100", "hbm"));
    EXPECT_FALSE(receiver.contains("100", "mem"));
}

TEST_F(BlockTreeReportEventTest, SnapshotRepairsDroppedAddsAndDeletesAcrossAllMedia) {
    publisher = std::make_shared<QueuedPublisher>(1);
    attach();
    insert(100, Tier::DEVICE);
    auto* host = insert(200, Tier::HOST);
    insert(300, Tier::DISK);
    flush();
    ASSERT_EQ(receiver.index.size(), 1u);
    snapshot();
    ASSERT_EQ(receiver.index.size(), 3u);
    EXPECT_TRUE(receiver.contains("100", "hbm"));
    EXPECT_TRUE(receiver.contains("200", "mem"));
    EXPECT_TRUE(receiver.contains("300", "disk"));
    host->group_set_resources[0].evictFromTier(Tier::HOST);
    tree->removeNodeAndEmptyAncestors(host);
    publisher->queue.discardPending();  // Simulate a lost incremental batch.
    snapshot();
    ASSERT_EQ(receiver.index.size(), 2u);
    EXPECT_FALSE(receiver.contains("200", "mem"));
    EXPECT_TRUE(receiver.contains("100", "hbm"));
    EXPECT_TRUE(receiver.contains("300", "disk"));
}

TEST_F(BlockTreeReportEventTest, HbmHostSelectionExcludesDiskFromDeltasAndSnapshots) {
    context.enabled_mediums = {KVCacheEventMedium::HBM, KVCacheEventMedium::HOST};
    attach();
    insert(100, Tier::DEVICE);
    insert(200, Tier::HOST);
    insert(300, Tier::DISK);
    flush();
    EXPECT_EQ(receiver.index.size(), 2u);
    EXPECT_FALSE(receiver.contains("300", "disk"));
    snapshot();
    EXPECT_EQ(receiver.index.size(), 2u);
    EXPECT_TRUE(receiver.contains("100", "hbm"));
    EXPECT_TRUE(receiver.contains("200", "mem"));
    EXPECT_FALSE(receiver.contains("300", "disk"));
}

TEST_F(BlockTreeReportEventTest, StartupSnapshotIncludesExistingHostAndEmptySnapshotClearsIt) {
    tree->setEventPublisher(nullptr, {});
    auto* node = insert(100, Tier::HOST);
    attach();
    publisher->queue.discardPending();
    snapshot();
    ASSERT_TRUE(receiver.contains("100", "mem"));
    node->group_set_resources[0].evictFromTier(Tier::HOST);
    tree->removeNodeAndEmptyAncestors(node);
    snapshot();
    EXPECT_TRUE(receiver.index.empty());
}

}  // namespace
}  // namespace rtp_llm
