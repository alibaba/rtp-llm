#include "aios/navi_sched/scheduler/TairKvcmGrpcProtocol.h"
#include "aios/navi_sched/test/TairKvcmGrpcTestServer.h"
#include "aios/network/gig/multi_call/interface/GrpcRequest.h"
#include "aios/network/gig/multi_call/grpc/client/GrpcUnaryStatusReceiver.h"
#include "aios/network/gig/multi_call/interface/GrpcResponse.h"
#include "aios/network/gig/multi_call/interface/QuerySession.h"
#include "aios/network/gig/multi_call/interface/RequestGenerator.h"
#include "aios/network/gig/multi_call/service/GrpcConnectionPool.h"
#include "aios/network/gig/multi_call/service/GrpcStreamConnectionPool.h"
#include "aios/network/gig/multi_call/service/SearchServiceManager.h"
#include "aios/network/gig/multi_call/stream/GigStreamClosure.h"
#include "unittest/unittest.h"
#include <condition_variable>

namespace navi_sched {
namespace {
namespace wire = kv_cache_manager::proto::meta;
class StatusResponse final : public multi_call::GrpcResponse, public multi_call::GrpcUnaryStatusReceiver {
public:
    void onGrpcStatus(const grpc::Status &value, bool ok) override {
        status = value; eventOk = ok; statusReceived = true;
    }
    grpc::Status status;
    bool eventOk = false;
    bool statusReceived = false;
};
class StatusRequest final : public multi_call::GrpcRequest {
public:
    using GrpcRequest::GrpcRequest;
    multi_call::ResponsePtr newResponse() override { return std::make_shared<StatusResponse>(); }
};
class FixedGrpcGenerator : public multi_call::RequestGenerator {
public:
    FixedGrpcGenerator(const std::string &host, int port, const std::string &method,
                       google::protobuf::Message *message, uint32_t timeoutMs)
        : multi_call::RequestGenerator("kvcm_unary_loopback") {
        multi_call::Spec spec; spec.ip = host; spec.ports[multi_call::MC_PROTOCOL_GRPC] = port;
        setSpec(spec); setDisableRetry(true); setDisableProbe(true);
        request = std::make_shared<StatusRequest>(method, getProtobufArena());
        request->setMessage(message); request->setTimeout(timeoutMs);
    }
    void generate(multi_call::PartIdTy, multi_call::PartRequestMap &output) override { output.emplace(0, request); }
    multi_call::GrpcRequestPtr request;
};
struct Completion {
    std::mutex mutex;
    std::condition_variable condition;
    multi_call::ReplyPtr reply;
    bool done = false;
    size_t calls = 0;
    bool wait() {
        std::unique_lock<std::mutex> lock(mutex);
        return condition.wait_for(lock, std::chrono::seconds(3), [&] { return done; });
    }
};
class Closure : public multi_call::Closure {
public:
    explicit Closure(std::shared_ptr<Completion> completion) : completion(std::move(completion)) {}
    void Run() override {
        std::unique_ptr<Closure> destroy(this);
        std::lock_guard<std::mutex> lock(completion->mutex);
        completion->done = true; ++completion->calls; completion->condition.notify_all();
    }
    std::shared_ptr<Completion> completion;
};
multi_call::SearchServicePtr service(size_t threads)
{
    auto value = std::make_shared<multi_call::SearchService>();
    multi_call::MultiCallConfig config; config.connectConf["grpc"].threadNum = threads;
    config.miscConf.disableMetricReport = true;
    if (!value->init(config)) throw std::runtime_error("GIG grpc init failed");
    return value;
}
std::shared_ptr<Completion> send(const multi_call::SearchServicePtr &service,
    const test::KvcmGrpcMock &server, const std::string &method,
    google::protobuf::Message *message, uint32_t timeoutMs = 1000)
{
    auto completion = std::make_shared<Completion>();
    multi_call::SearchParam param;
    param.generatorVec.push_back(std::make_shared<FixedGrpcGenerator>(server.bindIp, server.port, method, message, timeoutMs));
    param.closure = new Closure(completion);
    service->search(param, completion->reply);
    return completion;
}
multi_call::GrpcResponsePtr response(const std::shared_ptr<Completion> &completion)
{
    if (!completion->reply) return {};
    size_t lack = 0; const auto responses = completion->reply->getResponses(lack);
    if (lack || responses.size() != 1) return {};
    return std::dynamic_pointer_cast<multi_call::GrpcResponse>(responses[0]);
}
class ProbeStreamHandler : public multi_call::GigStreamHandlerBase {
public:
    ProbeStreamHandler() : GigStreamHandlerBase(0, multi_call::GigStreamBasePtr()) {
        atomic_inc(&_streamRpcInfoController._closureCount);
    }
    bool wait() {
        std::unique_lock<std::mutex> lock(mutex);
        return condition.wait_for(lock, std::chrono::seconds(3), [&] { return callbacks == 1; });
    }
    std::mutex mutex; std::condition_variable condition; size_t callbacks = 0;
protected:
    bool init() override { return false; }
    void initCallback(bool ok) override {
        std::lock_guard<std::mutex> lock(mutex); if (ok) ++callbacks; condition.notify_all();
    }
    multi_call::GigStreamClosureBase *doSend(bool, grpc::ByteBuffer *, multi_call::GigStreamClosureBase *, bool &ok) override { ok = false; return nullptr; }
    multi_call::GigStreamClosureBase *receiveNext() override { return nullptr; }
    bool ignoreReceiveError() override { return false; }
    multi_call::GigStreamClosureBase *finish() override { return nullptr; }
    void clean() override {}
};
}
class TairKvcmGigGrpcTest : public TESTBASE {};

TEST_F(TairKvcmGigGrpcTest, existingGigUnaryCarriesKvcmLeaderProtoWithoutChangingStream)
{
    test::KvcmGrpcMock server([](const auto &method, const auto &payload) {
        wire::GetClusterInfoRequest request;
        if (method != TAIR_KVCM_GRPC_GET_CLUSTER_INFO || !request.ParseFromString(payload) || !request.instance_id().empty())
            return test::KvcmGrpcMock::Reply{{}, grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "wrong contract")};
        wire::GetClusterInfoResponse reply;
        reply.mutable_header()->mutable_status()->set_code(wire::OK);
        reply.mutable_leader_endpoint()->set_host("127.0.0.2"); reply.mutable_leader_endpoint()->set_meta_rpc_port(16381);
        return test::KvcmGrpcMock::Reply{reply.SerializeAsString(), grpc::Status::OK};
    });
    auto client = service(4); auto request = new wire::GetClusterInfoRequest(); request->set_trace_id("gig-kvcm-loopback");
    auto completion = send(client, server, TAIR_KVCM_GRPC_GET_CLUSTER_INFO, request);
    ASSERT_TRUE(completion->wait()); EXPECT_EQ(1u, completion->calls);
    auto reply = response(completion); ASSERT_TRUE(reply); ASSERT_FALSE(reply->isFailed());
    auto parsed = static_cast<wire::GetClusterInfoResponse *>(reply->getMessage(new wire::GetClusterInfoResponse()));
    ASSERT_TRUE(parsed); EXPECT_EQ(wire::OK, parsed->header().status().code());
    EXPECT_EQ("127.0.0.2", parsed->leader_endpoint().host()); EXPECT_EQ(16381, parsed->leader_endpoint().meta_rpc_port());
    ASSERT_EQ(1u, server.requests.size());
}

TEST_F(TairKvcmGigGrpcTest, existingGigUnaryCarriesRemoteOnlyLookup)
{
    test::KvcmGrpcMock server([](const auto &, const auto &payload) {
        wire::GetCacheLocationsByBackendRequest request;
        if (!request.ParseFromString(payload) || request.instance_id() != "registered_native" ||
            request.query_type() != wire::QT_BATCH_GET || request.backend_selectors_size() != 2)
            return test::KvcmGrpcMock::Reply{{}, grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "wrong contract")};
        wire::GetCacheLocationsByBackendResponse reply; reply.mutable_header()->mutable_status()->set_code(wire::OK);
        for (const auto &specName : request.location_spec_names()) {
            auto location = reply.add_key_locations()->add_locations(); location->set_type(wire::ST_TAIRMEMPOOL);
            auto spec = location->add_location_specs(); spec->set_name(specName); spec->set_uri("pace://loopback/object");
        }
        return test::KvcmGrpcMock::Reply{reply.SerializeAsString(), grpc::Status::OK};
    });
    auto client = service(4); auto request = new wire::GetCacheLocationsByBackendRequest();
    TairKvcmQueryInput input; auto policy = std::make_shared<NaviSchedTairKvcmConfig>(); policy->instanceId = "registered_native";
    input.policy = policy; input.objects = {{-1, "full_pp0"}, {18, "mamba_pp0"}};
    std::string payload; ASSERT_TRUE(makeTairKvcmGrpcQueryRequest(input, payload, 4096)); ASSERT_TRUE(request->ParseFromString(payload));
    auto completion = send(client, server, TAIR_KVCM_GRPC_GET_REMOTE_LOCATIONS, request);
    ASSERT_TRUE(completion->wait()); auto reply = response(completion); ASSERT_TRUE(reply); ASSERT_FALSE(reply->isFailed());
    auto parsed = static_cast<wire::GetCacheLocationsByBackendResponse *>(reply->getMessage(new wire::GetCacheLocationsByBackendResponse()));
    ASSERT_TRUE(parsed); auto hits = readTairKvcmGrpcQueryResponse(input, parsed->SerializeAsString());
    EXPECT_EQ(TairKvcmQueryStatus::OK, hits.status); EXPECT_EQ((std::vector<uint8_t>{1, 1}), hits.remoteHits);
    EXPECT_EQ(1u, completion->calls);
}

TEST_F(TairKvcmGigGrpcTest, unaryTransportErrorAndDeadlineCompleteExactlyOnce)
{
    for (bool deadline : {false, true}) {
        test::KvcmGrpcMock server([deadline](const auto &, const auto &) {
            return test::KvcmGrpcMock::Reply{{}, deadline ? grpc::Status::OK :
                grpc::Status(grpc::StatusCode::UNAVAILABLE, "synthetic"), deadline};
        });
        auto client = service(2); auto request = new wire::GetClusterInfoRequest();
        auto completion = send(client, server, TAIR_KVCM_GRPC_GET_CLUSTER_INFO, request, 50);
        ASSERT_TRUE(completion->wait()); EXPECT_EQ(1u, completion->calls);
        auto reply = response(completion); ASSERT_TRUE(reply); EXPECT_TRUE(reply->isFailed());
        auto actual = std::dynamic_pointer_cast<StatusResponse>(reply); ASSERT_TRUE(actual);
        EXPECT_TRUE(actual->statusReceived);
        EXPECT_EQ(deadline ? grpc::StatusCode::DEADLINE_EXCEEDED : grpc::StatusCode::UNAVAILABLE,
                  actual->status.error_code());
        ASSERT_EQ(1u, server.requests.size()); server.release();
    }
}
TEST_F(TairKvcmGigGrpcTest, unaryInitLimitRejectsOversizedReplyAndCannotChangeAfterConnection)
{
    auto client = service(4);
    auto pool = std::dynamic_pointer_cast<multi_call::GrpcConnectionPool>(
        client->getSearchServiceManager()->getConnectionManager()->getConnectionPool(multi_call::MC_PROTOCOL_GRPC));
    ASSERT_TRUE(pool);
    EXPECT_FALSE(pool->configureMaxMessageBytes(0));
    ASSERT_TRUE(pool->configureMaxMessageBytes(128));
    EXPECT_FALSE(pool->configureMaxMessageBytes(256)); // Fixed even if GIG later retires a cached connection.
    test::KvcmGrpcMock server([](const auto &, const auto &) {
        wire::GetClusterInfoResponse response;
        response.mutable_header()->mutable_status()->set_message(std::string(1024, 'x'));
        return test::KvcmGrpcMock::Reply{response.SerializeAsString(), grpc::Status::OK};
    });
    auto completion = send(client, server, TAIR_KVCM_GRPC_GET_CLUSTER_INFO, new wire::GetClusterInfoRequest());
    ASSERT_TRUE(completion->wait()); EXPECT_EQ(1u, completion->calls);
    auto actual = std::dynamic_pointer_cast<StatusResponse>(response(completion)); ASSERT_TRUE(actual);
    EXPECT_TRUE(actual->statusReceived); EXPECT_TRUE(actual->isFailed());
    EXPECT_EQ(grpc::StatusCode::RESOURCE_EXHAUSTED, actual->status.error_code());
    EXPECT_TRUE(pool->configureMaxMessageBytes(128));
    EXPECT_FALSE(pool->configureMaxMessageBytes(256));
    ASSERT_EQ(1u, server.requests.size());
}

TEST_F(TairKvcmGigGrpcTest, unaryAndStreamPoolsDispatchTheirOwnClosureTypesConcurrently)
{
    multi_call::MultiCallConfig config;
    config.connectConf["grpc"].threadNum = 4;
    config.connectConf["grpc_stream"].threadNum = 2;
    config.miscConf.disableMetricReport = true;
    auto client = std::make_shared<multi_call::SearchService>(); ASSERT_TRUE(client->init(config));
    const auto manager = client->getSearchServiceManager()->getConnectionManager();
    auto unary = std::dynamic_pointer_cast<multi_call::GrpcConnectionPool>(manager->getConnectionPool(multi_call::MC_PROTOCOL_GRPC));
    auto stream = std::dynamic_pointer_cast<multi_call::GrpcStreamConnectionPool>(manager->getConnectionPool(multi_call::MC_PROTOCOL_GRPC_STREAM));
    ASSERT_TRUE(unary); ASSERT_TRUE(stream);
    EXPECT_FALSE(unary->_worker->_isStream.load()); EXPECT_TRUE(stream->_worker->_isStream.load());
    EXPECT_EQ(4u, unary->_worker->_completionQueues.size()); EXPECT_EQ(2u, stream->_worker->_completionQueues.size());
    EXPECT_NE(unary->_worker, stream->_worker);
    auto handler = std::make_shared<ProbeStreamHandler>();
    auto closure = new multi_call::GigStreamInitClosure(handler);
    auto queue = stream->_worker->getCompletionQueue(0);
    grpc::Alarm alarm(queue->cq.get(), gpr_now(GPR_CLOCK_MONOTONIC), closure);
    test::KvcmGrpcMock server([](const auto &, const auto &) {
        wire::GetClusterInfoResponse reply; reply.mutable_header()->mutable_status()->set_code(wire::OK);
        return test::KvcmGrpcMock::Reply{reply.SerializeAsString(), grpc::Status::OK};
    });
    auto request = new wire::GetClusterInfoRequest();
    auto completion = send(client, server, TAIR_KVCM_GRPC_GET_CLUSTER_INFO, request);
    ASSERT_TRUE(handler->wait()); ASSERT_TRUE(completion->wait());
    EXPECT_EQ(1u, handler->callbacks); EXPECT_EQ(1u, completion->calls);
    auto reply = response(completion); ASSERT_TRUE(reply); EXPECT_FALSE(reply->isFailed());
}

} // namespace navi_sched
