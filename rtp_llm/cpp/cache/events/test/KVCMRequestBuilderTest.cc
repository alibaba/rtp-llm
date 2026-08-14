#include "rtp_llm/cpp/cache/events/KVCMRequestBuilder.h"

#include <atomic>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace rtp_llm::detail {
namespace {

KVCacheEventPublisherContext makeContext() {
    KVCacheEventPublisherContext context;
    context.instance_group    = "group-a";
    context.instance_id       = "instance-1";
    context.host_ip_port      = "127.0.0.1:9000";
    context.model_name        = "model-a";
    context.dtype             = "bf16";
    context.spec_name         = "kv_cache";
    context.location_uri      = "rtp-llm://127.0.0.1:9000/hbm?size=4096";
    context.block_size_tokens = 16;
    context.spec_size_bytes   = 4096;
    context.tp_size           = 2;
    context.dp_size           = 4;
    context.pp_size           = 1;
    context.dp_rank           = 3;
    context.use_mla           = true;
    return context;
}

TEST(KVCMRequestBuilderTest, BuildsRegistrationSchemaExactly) {
    EXPECT_EQ(
        R"({"trace_id":"trace-1","instance_group":"group-a","instance_id":"instance-1","block_size":16,"model_deployment":{"model_name":"model-a","dtype":"bf16","use_mla":true,"tp_size":2,"dp_size":4,"pp_size":1},"location_spec_infos":[{"name":"kv_cache","size":4096}],"location_spec_groups":[{"name":"default","spec_names":["kv_cache"]}]})",
        buildRegisterInstanceRequest(makeContext(), "trace-1", 4096));
}

TEST(KVCMRequestBuilderTest, CoalescesToFinalTransitionInFirstSeenKeyOrder) {
    const std::vector<KVCacheEvent> events = {
        {KVCacheEventType::BLOCK_ADD, 10, 1},
        {KVCacheEventType::BLOCK_DELETE, 20, 2},
        {KVCacheEventType::BLOCK_DELETE, 10, 3},
        {KVCacheEventType::BLOCK_ADD, 20, 4},
    };

    const auto coalesced = coalesceMutations(events);

    ASSERT_EQ(2u, coalesced.size());
    EXPECT_EQ(KVCacheEventType::BLOCK_DELETE, coalesced[0].type);
    EXPECT_EQ(10, coalesced[0].block_key);
    EXPECT_EQ(3u, coalesced[0].sequence);
    EXPECT_EQ(KVCacheEventType::BLOCK_ADD, coalesced[1].type);
    EXPECT_EQ(20, coalesced[1].block_key);
    EXPECT_EQ(4u, coalesced[1].sequence);

    const auto report = buildMutationReport(makeContext(), "trace-2", coalesced, 4096);
    EXPECT_NE(std::string::npos, report.find("EVENT_BLOCK_DELETE"));
    EXPECT_NE(std::string::npos, report.find("\"block_key\":\"10\""));
    EXPECT_NE(std::string::npos, report.find("EVENT_BLOCK_ADD"));
    EXPECT_NE(std::string::npos, report.find("\"block_key\":\"20\""));
}

TEST(KVCMRequestBuilderTest, BuildsEveryControlEventSchema) {
    const auto context = makeContext();
    EXPECT_NE(std::string::npos,
              buildControlReport(context, "down", ControlEventType::HOST_DOWN, 4096).find("EVENT_HOST_DOWN"));
    EXPECT_NE(
        std::string::npos,
        buildControlReport(context, "register", ControlEventType::NODE_REGISTER, 4096).find(R"("mediums":["hbm"])"));
    const auto heartbeat = buildControlReport(context, "heartbeat", ControlEventType::HEARTBEAT, 4096);
    EXPECT_NE(std::string::npos, heartbeat.find("EVENT_HEARTBEAT"));
    EXPECT_NE(std::string::npos, heartbeat.find(R"("dp_rank":"3")"));
}

TEST(KVCMRequestBuilderTest, EnforcesByteLimitAndSnapshotCancellation) {
    const auto context = makeContext();
    const auto request = buildSnapshotReport(context, "snapshot", KVCacheSnapshot{{10, 20}}, 4096);
    ASSERT_FALSE(request.empty());
    EXPECT_THROW(buildSnapshotReport(context, "snapshot", KVCacheSnapshot{{10, 20}}, request.size() - 1),
                 JsonPayloadLimitExceeded);

    std::atomic<bool> cancelled{true};
    EXPECT_THROW(buildSnapshotReport(context, "snapshot", KVCacheSnapshot{{10}}, 4096, &cancelled),
                 SnapshotBuildCancelled);
}

}  // namespace
}  // namespace rtp_llm::detail
