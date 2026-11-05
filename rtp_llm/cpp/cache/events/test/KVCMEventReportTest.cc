#include "rtp_llm/cpp/cache/events/KVCMEventReport.h"

#include <gtest/gtest.h>
#include <rapidjson/document.h>

namespace rtp_llm {
namespace {

KVCacheEventPublisherContext multiTierContext() {
    KVCacheEventPublisherContext context;
    context.instance_id     = "instance";
    context.instance_group  = "group";
    context.host_ip_port    = "127.0.0.1:8000";
    context.specs           = {{"F", "event_report://127.0.0.1:8000/hbm"}, {"L", "event_report://127.0.0.1:8000/hbm"}};
    context.enabled_mediums = {KVCacheEventMedium::HBM, KVCacheEventMedium::HOST, KVCacheEventMedium::DISK};
    return context;
}

KVCacheEvent mutation(KVCacheEventType type, KVCacheEventMedium medium, int64_t key = 42, size_t group = 0) {
    KVCacheEvent event{type, key, group};
    event.medium = medium;
    return event;
}

TEST(KVCMEventReportTest, MutationsKeepMediumAndLocationTogether) {
    const auto                      context = multiTierContext();
    const std::vector<KVCacheEvent> events{mutation(KVCacheEventType::BLOCK_ADD, KVCacheEventMedium::HBM),
                                           mutation(KVCacheEventType::BLOCK_ADD, KVCacheEventMedium::HOST),
                                           mutation(KVCacheEventType::BLOCK_ADD, KVCacheEventMedium::DISK),
                                           mutation(KVCacheEventType::BLOCK_DELETE, KVCacheEventMedium::HOST),
                                           mutation(KVCacheEventType::BLOCK_DELETE, KVCacheEventMedium::DISK)};
    rapidjson::Document             body;
    body.Parse(detail::buildMutationReport(context, "trace", events).c_str());
    ASSERT_FALSE(body.HasParseError());
    EXPECT_STREQ(body["storage_type"].GetString(), "ST_EVENT_REPORT_L1P5");
    const auto& wire = body["events"];
    ASSERT_EQ(wire.Size(), 5u);
    const char* mediums[] = {"hbm", "mem", "disk"};
    const char* uris[]    = {
        "event_report://127.0.0.1:8000/hbm", "event_report://127.0.0.1:8000/mem", "event_report://127.0.0.1:8000/disk"};
    for (rapidjson::SizeType i = 0; i < 3; ++i) {
        const auto& add = wire[i]["block_add"];
        EXPECT_STREQ(add["medium"].GetString(), mediums[i]);
        EXPECT_STREQ(add["block_key"].GetString(), "42");
        EXPECT_STREQ(add["specs"][0]["name"].GetString(), "F");
        EXPECT_STREQ(add["specs"][0]["uri"].GetString(), uris[i]);
    }
    EXPECT_STREQ(wire[3]["block_delete"]["medium"].GetString(), "mem");
    EXPECT_STREQ(wire[4]["block_delete"]["medium"].GetString(), "disk");
    EXPECT_STREQ(wire[4]["block_delete"]["spec_names"][0].GetString(), "F");
}

TEST(KVCMEventReportTest, CoalescingKeepsDifferentMediaAndComponentsIndependent) {
    const auto events =
        detail::coalesceMutations({mutation(KVCacheEventType::BLOCK_ADD, KVCacheEventMedium::HBM),
                                   mutation(KVCacheEventType::BLOCK_ADD, KVCacheEventMedium::HOST),
                                   mutation(KVCacheEventType::BLOCK_DELETE, KVCacheEventMedium::HBM),
                                   mutation(KVCacheEventType::BLOCK_ADD, KVCacheEventMedium::DISK),
                                   mutation(KVCacheEventType::BLOCK_DELETE, KVCacheEventMedium::HOST),
                                   mutation(KVCacheEventType::BLOCK_ADD, KVCacheEventMedium::HOST),
                                   mutation(KVCacheEventType::BLOCK_ADD, KVCacheEventMedium::HOST, 42, 1)});
    ASSERT_EQ(events.size(), 4u);
    EXPECT_EQ(events[0].medium, KVCacheEventMedium::HBM);
    EXPECT_EQ(events[0].type, KVCacheEventType::BLOCK_DELETE);
    EXPECT_EQ(events[1].medium, KVCacheEventMedium::HOST);
    EXPECT_EQ(events[1].type, KVCacheEventType::BLOCK_ADD);
    EXPECT_EQ(events[2].medium, KVCacheEventMedium::DISK);
    EXPECT_EQ(events[3].group_index, 1u);
}

TEST(KVCMEventReportTest, SnapshotIsOneAuthoritativeSetAcrossAllMedia) {
    KVCacheSnapshot snapshot;
    snapshot.block_groups      = {{42, {0}}};
    snapshot.host_block_groups = {{42, {0, 1}}};
    snapshot.disk_block_groups = {{-1, {1}}};
    rapidjson::Document body;
    body.Parse(detail::buildSnapshotReport(multiTierContext(), "trace", snapshot).c_str());
    ASSERT_FALSE(body.HasParseError());
    ASSERT_EQ(body["events"].Size(), 1u);
    const auto& event = body["events"][0];
    EXPECT_STREQ(event["event_type"].GetString(), "EVENT_BLOCK_SNAPSHOT");
    EXPECT_FALSE(event["block_snapshot"].HasMember("medium"));
    const auto& blocks = event["block_snapshot"]["blocks"];
    ASSERT_EQ(blocks.Size(), 3u);
    EXPECT_STREQ(blocks[0]["medium"].GetString(), "hbm");
    EXPECT_STREQ(blocks[1]["medium"].GetString(), "mem");
    ASSERT_EQ(blocks[1]["specs"].Size(), 2u);
    EXPECT_STREQ(blocks[1]["specs"][1]["uri"].GetString(), "event_report://127.0.0.1:8000/mem");
    EXPECT_STREQ(blocks[2]["medium"].GetString(), "disk");
    EXPECT_STREQ(blocks[2]["block_key"].GetString(), "-1");
    EXPECT_STREQ(blocks[2]["specs"][0]["uri"].GetString(), "event_report://127.0.0.1:8000/disk");
}

TEST(KVCMEventReportTest, EmptySnapshotStillClearsEveryMedium) {
    rapidjson::Document body;
    body.Parse(detail::buildSnapshotReport(multiTierContext(), "trace", {}).c_str());
    ASSERT_FALSE(body.HasParseError());
    ASSERT_EQ(body["events"].Size(), 1u);
    EXPECT_TRUE(body["events"][0]["block_snapshot"]["blocks"].Empty());
}

TEST(KVCMEventReportTest, LegacyHbmProvidersKeepTheirDefaultMediumAndUri) {
    KVCacheEventPublisherContext context;
    context.specs = {{"legacy", "event_report://legacy-host/hbm"}};
    const KVCacheEvent event{KVCacheEventType::BLOCK_ADD, 42, 0, 17};
    EXPECT_EQ(event.sequence, 17u);
    EXPECT_EQ(event.medium, KVCacheEventMedium::HBM);
    rapidjson::Document body;
    body.Parse(detail::buildMutationReport(context, "trace", {event}).c_str());
    const auto& add = body["events"][0]["block_add"];
    EXPECT_STREQ(add["medium"].GetString(), "hbm");
    EXPECT_STREQ(add["specs"][0]["uri"].GetString(), "event_report://legacy-host/hbm");
    const KVCacheSnapshot snapshot{7, {{42, {0}}}};
    body.Parse(detail::buildSnapshotReport(context, "trace", snapshot).c_str());
    const auto& blocks = body["events"][0]["block_snapshot"]["blocks"];
    ASSERT_EQ(blocks.Size(), 1u);
    EXPECT_STREQ(blocks[0]["medium"].GetString(), "hbm");
    body.Parse(detail::buildControlReport(context, "trace", detail::ControlEventType::NODE_REGISTER).c_str());
    const auto& mediums = body["events"][0]["node_register"]["mediums"];
    ASSERT_EQ(mediums.Size(), 1u);
    EXPECT_STREQ(mediums[0].GetString(), "hbm");
}

TEST(KVCMEventReportTest, RegistrationDeclaresEnabledTiersAndControlOnlyDeclaresNone) {
    auto                context = multiTierContext();
    rapidjson::Document body;
    body.Parse(detail::buildControlReport(context, "trace", detail::ControlEventType::NODE_REGISTER).c_str());
    const auto& mediums = body["events"][0]["node_register"]["mediums"];
    ASSERT_EQ(mediums.Size(), 3u);
    EXPECT_STREQ(mediums[0].GetString(), "hbm");
    EXPECT_STREQ(mediums[1].GetString(), "mem");
    EXPECT_STREQ(mediums[2].GetString(), "disk");
    context.enabled_mediums = {KVCacheEventMedium::DISK};
    body.Parse(detail::buildControlReport(context, "trace", detail::ControlEventType::NODE_REGISTER).c_str());
    ASSERT_EQ(body["events"][0]["node_register"]["mediums"].Size(), 1u);
    EXPECT_STREQ(body["events"][0]["node_register"]["mediums"][0].GetString(), "disk");
    body.Parse(detail::buildControlReport(context, "trace", detail::ControlEventType::NODE_REGISTER, false).c_str());
    EXPECT_TRUE(body["events"][0]["node_register"]["mediums"].Empty());
}

}  // namespace
}  // namespace rtp_llm
