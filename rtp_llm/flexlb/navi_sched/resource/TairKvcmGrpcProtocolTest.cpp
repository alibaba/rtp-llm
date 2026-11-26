#include "aios/navi_sched/scheduler/TairKvcmGrpcProtocol.h"
#include "autil/TimeUtility.h"
#include "unittest/unittest.h"
#include <limits>

namespace navi_sched {
namespace proto = kv_cache_manager::proto::meta;
class TairKvcmGrpcProtocolTest : public TESTBASE {};
namespace {
TairKvcmQueryInput input()
{
    TairKvcmQueryInput result;
    auto config = std::make_shared<NaviSchedTairKvcmConfig>();
    config->instanceId = "registered_native_instance";
    result.policy = config;
    result.traceId = "grpc-contract-test";
    result.objects = {{std::numeric_limits<int64_t>::min(), "full_pp0"},
                      {-1, "mamba_pp0"}, {19, "mamba_pp1"}};
    return result;
}
proto::GetCacheLocationsByBackendResponse response()
{
    proto::GetCacheLocationsByBackendResponse result;
    result.mutable_header()->mutable_status()->set_code(proto::OK);
    for (const auto &object : input().objects) {
        auto location = result.add_key_locations()->add_locations();
        location->set_type(proto::ST_TAIRMEMPOOL);
        auto spec = location->add_location_specs();
        spec->set_name(object.locationSpecName);
        spec->set_uri("pace://isolated/key");
    }
    return result;
}
}

TEST_F(TairKvcmGrpcProtocolTest, leaderDiscoveryOmitsInstanceIdAndUsesRpcPort)
{
    proto::GetClusterInfoRequest request;
    ASSERT_TRUE(request.ParseFromString(makeTairKvcmGrpcLeaderRequest("trace")));
    EXPECT_EQ("trace", request.trace_id());
    EXPECT_TRUE(request.instance_id().empty());
    proto::GetClusterInfoResponse reply;
    reply.mutable_header()->mutable_status()->set_code(proto::OK);
    auto leader = reply.mutable_leader_endpoint();
    leader->set_host("127.0.0.2");
    leader->set_meta_rpc_port(6381);
    leader->set_meta_http_port(6382);
    TairKvcmGrpcEndpoint endpoint;
    ASSERT_TRUE(readTairKvcmGrpcLeader(reply.SerializeAsString(), endpoint));
    EXPECT_EQ("127.0.0.2:6381", endpoint.target());
    leader->set_meta_rpc_port(0);
    EXPECT_FALSE(readTairKvcmGrpcLeader(reply.SerializeAsString(), endpoint));
    EXPECT_FALSE(endpoint.valid());
    leader->set_meta_rpc_port(6381);
    reply.mutable_header()->mutable_status()->set_code(proto::INTERNAL_ERROR);
    EXPECT_FALSE(readTairKvcmGrpcLeader(reply.SerializeAsString(), endpoint));
}

TEST_F(TairKvcmGrpcProtocolTest, exactV6dRemoteOnlyQueryWithSignedKeysAndMasks)
{
    const auto query = input();
    std::string payload;
    ASSERT_TRUE(makeTairKvcmGrpcQueryRequest(query, payload, 4096));
    proto::GetCacheLocationsByBackendRequest request;
    ASSERT_TRUE(request.ParseFromString(payload));
    EXPECT_EQ(query.traceId, request.trace_id());
    EXPECT_EQ(query.policy->instanceId, request.instance_id());
    EXPECT_EQ(proto::QT_BATCH_GET, request.query_type());
    ASSERT_EQ(3, request.block_keys_size());
    EXPECT_EQ(std::numeric_limits<int64_t>::min(), request.block_keys(0));
    EXPECT_EQ(-1, request.block_keys(1));
    ASSERT_EQ(3, request.location_spec_names_size());
    ASSERT_EQ(3, request.block_mask().bool_masks().values_size());
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(query.objects[i].locationSpecName, request.location_spec_names(i));
        EXPECT_FALSE(request.block_mask().bool_masks().values(i));
    }
    ASSERT_EQ(2, request.backend_selectors_size());
    EXPECT_EQ(proto::ST_TAIRMEMPOOL, request.backend_selectors(0).backend_type());
    EXPECT_EQ(proto::ST_NFS, request.backend_selectors(1).backend_type());
    for (const auto &selector : request.backend_selectors())
        EXPECT_EQ(proto::LSS_WEIGHTED_RANDOM, selector.strategy());
    EXPECT_TRUE(request.token_ids().empty());
    EXPECT_EQ(0, request.sw_size());
}

TEST_F(TairKvcmGrpcProtocolTest, malformedOrOversizedRequestIsNotSerialized)
{
    auto query = input();
    std::string payload;
    EXPECT_FALSE(makeTairKvcmGrpcQueryRequest(query, payload, 1));
    EXPECT_TRUE(payload.empty());
    query.objects[1].locationSpecName.clear();
    EXPECT_FALSE(makeTairKvcmGrpcQueryRequest(query, payload, 4096));
    query.policy.reset();
    EXPECT_FALSE(makeTairKvcmGrpcQueryRequest(query, payload, 4096));
}

TEST_F(TairKvcmGrpcProtocolTest, onlyExactRemoteSpecWithNonemptyUriCounts)
{
    auto reply = response();
    reply.mutable_key_locations(0)->mutable_locations(0)->set_type(proto::ST_EVENT_REPORT_L2);
    reply.mutable_key_locations(1)->mutable_locations(0)->mutable_location_specs(0)->set_uri("");
    auto parsed = readTairKvcmGrpcQueryResponse(input(), reply.SerializeAsString());
    EXPECT_EQ(TairKvcmQueryStatus::OK, parsed.status);
    EXPECT_EQ((std::vector<uint8_t>{0, 0, 1}), parsed.remoteHits);
    reply.mutable_key_locations(2)->mutable_locations(0)->mutable_location_specs(0)->set_name("other_spec");
    parsed = readTairKvcmGrpcQueryResponse(input(), reply.SerializeAsString());
    EXPECT_EQ((std::vector<uint8_t>{0, 0, 0}), parsed.remoteHits);
}

TEST_F(TairKvcmGrpcProtocolTest, malformedOrBusinessErrorCannotPublishPartialHits)
{
    auto reply = response();
    reply.mutable_key_locations()->RemoveLast();
    auto parsed = readTairKvcmGrpcQueryResponse(input(), reply.SerializeAsString());
    EXPECT_EQ(TairKvcmQueryStatus::INVALID_RESPONSE, parsed.status);
    EXPECT_EQ((std::vector<uint8_t>{0, 0, 0}), parsed.remoteHits);
    reply = response();
    reply.mutable_header()->mutable_status()->set_code(proto::SERVER_NOT_LEADER);
    parsed = readTairKvcmGrpcQueryResponse(input(), reply.SerializeAsString());
    EXPECT_EQ(TairKvcmQueryStatus::BUSINESS_ERROR, parsed.status);
    EXPECT_EQ((std::vector<uint8_t>{0, 0, 0}), parsed.remoteHits);
    parsed = readTairKvcmGrpcQueryResponse(input(), std::string(1, '\xff'));
    EXPECT_EQ(TairKvcmQueryStatus::INVALID_RESPONSE, parsed.status);
    EXPECT_EQ((std::vector<uint8_t>{0, 0, 0}), parsed.remoteHits);
}

TEST_F(TairKvcmGrpcProtocolTest, representativeProtobufPreparationCost)
{
    auto query = input();
    query.objects.clear();
    for (size_t count : {size_t(4096), size_t(16384), size_t(32768)}) {
        query.objects.resize(count);
        for (size_t i = 0; i < count; ++i)
            query.objects[i] = {static_cast<int64_t>(i / 32), "group0_pp" + std::to_string(i % 32)};
        std::string payload;
        const auto begin = autil::TimeUtility::monotonicTimeUs();
        ASSERT_TRUE(makeTairKvcmGrpcQueryRequest(query, payload, 4 * 1024 * 1024));
        const auto elapsed = autil::TimeUtility::monotonicTimeUs() - begin;
        std::cout << "[tair-grpc-protobuf-cost] objects=" << count << " serialize_us=" << elapsed
                  << " bytes=" << payload.size() << std::endl;
    }
}
} // namespace navi_sched
