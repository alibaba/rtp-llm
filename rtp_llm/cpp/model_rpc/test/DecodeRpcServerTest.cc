#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"

namespace rtp_llm {

namespace {

DecodeRpcServer::LoadKVCacheContext makeLoadContext(const std::string&               request_key,
                                                    const std::vector<std::string>&  peer_addrs,
                                                    const std::vector<CacheKeyType>& cache_keys,
                                                    const GroupBlockIds&             block_ids_by_group,
                                                    int32_t                          prefill_cp_size,
                                                    int64_t                          reuse_block_size = 0) {
    return {/*request_id=*/42,
            request_key,
            peer_addrs,
            cache_keys,
            block_ids_by_group,
            reuse_block_size,
            /*timeout_ms=*/1000,
            /*partition_count=*/1,
            /*partition_id=*/0,
            /*server_context=*/nullptr,
            prefill_cp_size};
}

class TestDecodeRpcService final: public RpcService::Service {
public:
    TestDecodeRpcService() {
        server_.meta_ = meta_;
    }

    grpc::Status RemoteGenerate(grpc::ServerContext*                                            context,
                                grpc::ServerReaderWriter<GenerateOutputsPB, GenerateRequestPB>* stream) override {
        return server_.RemoteGenerate(context, stream);
    }

    EngineScheduleInfo scheduleInfo() const {
        return meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    }

private:
    std::shared_ptr<RpcServerRuntimeMeta> meta_ = std::make_shared<RpcServerRuntimeMeta>();
    DecodeRpcServer                       server_;
};

GroupBase makeRpcGroup(std::string tag, std::vector<int> layer_ids) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->tag                = tag;
    spec->seq_size_per_block = 8;

    GroupBase group;
    group.tag                       = std::move(tag);
    group.spec                      = std::move(spec);
    group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.layer_ids                 = std::move(layer_ids);
    group.block_num                 = 8;
    group.seq_size_per_block        = 8;
    group.kernel_seq_size_per_block = 8;
    return group;
}

}  // namespace

TEST(DecodeRpcServerTest, invalidQueryUsesInvalidArgumentWithDetails) {
    TestDecodeRpcService service;
    int                  listen_port = 0;
    grpc::ServerBuilder  builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &listen_port);
    builder.RegisterService(&service);
    auto grpc_server = builder.BuildAndStart();
    ASSERT_NE(grpc_server, nullptr);
    ASSERT_NE(listen_port, 0);

    auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(listen_port), grpc::InsecureChannelCredentials());
    auto stub    = RpcService::NewStub(channel);
    grpc::ClientContext context;
    auto                stream = stub->RemoteGenerate(&context);
    GenerateRequestPB   request;
    request.set_stage(RemoteStage::ALLOCATE);
    request.set_client_id("test");
    request.set_request_id(1);
    request.mutable_input()->add_token_ids(0);
    request.mutable_input()->mutable_generate_config()->set_max_new_tokens(-1);

    EXPECT_TRUE(stream->Write(request));
    stream->WritesDone();
    GenerateOutputsPB response;
    EXPECT_FALSE(stream->Read(&response));
    const auto status = stream->Finish();
    grpc_server->Shutdown();
    grpc_server->Wait();

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB error_details;
    ASSERT_TRUE(error_details.ParseFromString(status.error_details()));
    EXPECT_EQ(error_details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
    EXPECT_NE(error_details.error_message().find("max_new_tokens"), std::string::npos);

    const auto schedule_info = service.scheduleInfo();
    ASSERT_EQ(schedule_info.finished_task_info_list.size(), 1);
    EXPECT_EQ(schedule_info.finished_task_info_list.front().request_id, 1);
    EXPECT_EQ(schedule_info.finished_task_info_list.front().error_code,
              static_cast<int64_t>(ErrorCode::INVALID_PARAMS));
}

TEST(DecodeRpcServerTest, zeroRequestIdWithInvalidStageReportsExactlyOnce) {
    TestDecodeRpcService service;
    int                  listen_port = 0;
    grpc::ServerBuilder  builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &listen_port);
    builder.RegisterService(&service);
    auto grpc_server = builder.BuildAndStart();
    ASSERT_NE(grpc_server, nullptr);
    ASSERT_NE(listen_port, 0);

    auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(listen_port), grpc::InsecureChannelCredentials());
    auto stub    = RpcService::NewStub(channel);
    grpc::ClientContext context;
    auto                stream = stub->RemoteGenerate(&context);
    GenerateRequestPB   request;
    request.set_stage(RemoteStage::LOAD);
    request.set_client_id("test");
    request.set_request_id(0);

    EXPECT_TRUE(stream->Write(request));
    stream->WritesDone();
    GenerateOutputsPB response;
    EXPECT_FALSE(stream->Read(&response));
    const auto status = stream->Finish();
    grpc_server->Shutdown();
    grpc_server->Wait();

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB error_details;
    ASSERT_TRUE(error_details.ParseFromString(status.error_details()));
    EXPECT_EQ(error_details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));

    const auto schedule_info = service.scheduleInfo();
    ASSERT_EQ(schedule_info.finished_task_info_list.size(), 1);
    EXPECT_EQ(schedule_info.finished_task_info_list.front().request_id, 0);
    EXPECT_EQ(schedule_info.finished_task_info_list.front().error_code,
              static_cast<int64_t>(ErrorCode::INVALID_PARAMS));
}

TEST(DecodeRpcServerTest, reportEarlyFinishTaskToleratesMissingRuntimeMeta) {
    DecodeRpcServer              server;
    DecodeRpcContext             rpc_context{nullptr};
    grpc::ServerContext          server_context;
    kmonitor::MetricsReporterPtr metrics_reporter;
    DecodeGenerateContext        decode_context(rpc_context, 0, &server_context, metrics_reporter, nullptr);
    decode_context.request_id         = 1;
    decode_context.request_id_present = true;
    decode_context.error_info         = ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid parameters");
    decode_context.error_status       = grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "invalid parameters");

    EXPECT_NO_THROW(server.reportEarlyFinishTask(decode_context));
    EXPECT_FALSE(decode_context.early_finish_reported);
}

TEST(DecodeRpcServerTest, reportEarlyFinishTaskReportsSerializedErrorExactlyOnce) {
    DecodeRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();

    DecodeRpcContext             rpc_context{nullptr};
    grpc::ServerContext          server_context;
    kmonitor::MetricsReporterPtr metrics_reporter;
    DecodeGenerateContext        decode_context(rpc_context, 0, &server_context, metrics_reporter, server.meta_);
    decode_context.request_id         = 42;
    decode_context.request_id_present = true;
    decode_context.request_key        = "test_request_id_42";

    ErrorDetailsPB error_details;
    error_details.set_error_code(static_cast<int>(ErrorCode::LOAD_CACHE_TIMEOUT));
    error_details.set_error_message("load cache timeout");
    std::string serialized_error_details;
    ASSERT_TRUE(error_details.SerializeToString(&serialized_error_details));
    decode_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, "load failed", serialized_error_details);

    server.reportEarlyFinishTask(decode_context);
    server.reportEarlyFinishTask(decode_context);

    const auto schedule_info = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(schedule_info.finished_task_info_list.size(), 1);
    EXPECT_EQ(schedule_info.finished_task_info_list.front().request_id, 42);
    EXPECT_EQ(schedule_info.finished_task_info_list.front().error_code,
              static_cast<int64_t>(ErrorCode::LOAD_CACHE_TIMEOUT));
    EXPECT_EQ(schedule_info.finished_task_info_list.front().error_message, "load cache timeout");
}

TEST(DecodeRpcServerTest, reportEarlyFinishTaskMapsBareGrpcErrorsExactlyOnce) {
    DecodeRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();

    struct TestCase {
        int64_t          request_id;
        grpc::StatusCode grpc_code;
        const char*      message;
        ErrorCode        expected_error_code;
    };
    const TestCase test_cases[] = {
        {0, grpc::StatusCode::INTERNAL, "read failed", ErrorCode::EXECUTION_EXCEPTION},
        {1, grpc::StatusCode::INTERNAL, "write failed", ErrorCode::EXECUTION_EXCEPTION},
        {2, grpc::StatusCode::CANCELLED, "cancelled", ErrorCode::CANCELLED},
        {3, grpc::StatusCode::DEADLINE_EXCEEDED, "timeout", ErrorCode::DEADLINE_EXCEEDED},
    };

    DecodeRpcContext             rpc_context{nullptr};
    grpc::ServerContext          server_context;
    kmonitor::MetricsReporterPtr metrics_reporter;
    {
        DecodeGenerateContext missing_id_context(rpc_context, 0, &server_context, metrics_reporter, server.meta_);
        missing_id_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, "read failed before first frame");
        server.reportEarlyFinishTask(missing_id_context);
    }

    for (const auto& test_case : test_cases) {
        DecodeGenerateContext decode_context(rpc_context, 0, &server_context, metrics_reporter, server.meta_);
        decode_context.request_id         = test_case.request_id;
        decode_context.request_id_present = true;
        decode_context.request_key        = "test_request_id_" + std::to_string(test_case.request_id);
        decode_context.error_status       = grpc::Status(test_case.grpc_code, test_case.message);

        server.reportEarlyFinishTask(decode_context);
        server.reportEarlyFinishTask(decode_context);
    }

    const auto schedule_info = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(schedule_info.finished_task_info_list.size(), std::size(test_cases));
    for (size_t i = 0; i < std::size(test_cases); ++i) {
        EXPECT_EQ(schedule_info.finished_task_info_list[i].request_id, test_cases[i].request_id);
        EXPECT_EQ(schedule_info.finished_task_info_list[i].error_code,
                  static_cast<int64_t>(test_cases[i].expected_error_code));
        EXPECT_EQ(schedule_info.finished_task_info_list[i].error_message, test_cases[i].message);
    }
}

TEST(ModelRpcProtoTest, GroupedCacheFieldsPreserveLegacyNumbers) {
    const auto* broadcast = BroadcastLoadRequestPB::descriptor();
    ASSERT_NE(broadcast, nullptr);
    EXPECT_TRUE(broadcast->IsReservedNumber(5));
    EXPECT_TRUE(broadcast->IsReservedNumber(12));
    EXPECT_EQ(broadcast->FindFieldByName("block_num")->number(), 6);
    EXPECT_EQ(broadcast->FindFieldByName("reuse_block_size")->number(), 7);
    EXPECT_EQ(broadcast->FindFieldByName("timeout_ms")->number(), 8);
    EXPECT_EQ(broadcast->FindFieldByName("dp_rank")->number(), 9);
    EXPECT_EQ(broadcast->FindFieldByName("partition_count")->number(), 10);
    EXPECT_EQ(broadcast->FindFieldByName("partition_id")->number(), 11);
    EXPECT_EQ(broadcast->FindFieldByName("prefill_cp_size")->number(), 13);
    EXPECT_EQ(broadcast->FindFieldByName("tagged_group_block_ids")->number(), 14);

    const auto* remote = RemoteOperationRequestPB::descriptor();
    ASSERT_NE(remote, nullptr);
    EXPECT_TRUE(remote->IsReservedNumber(3));
    EXPECT_EQ(remote->FindFieldByName("group_ids"), nullptr);
    EXPECT_EQ(remote->FindFieldByName("block_ids")->number(), 4);
    EXPECT_EQ(remote->FindFieldByName("uris")->number(), 5);
    EXPECT_EQ(remote->FindFieldByName("group_tags")->number(), 6);
}

TEST(DecodeRpcServerTest, CPShardedLoadRequestReadsFromEveryPrefillPeer) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1"};

    const std::string               request_key = "request";
    const std::vector<std::string>  peer_addrs  = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType> cache_keys  = {101, 102};
    const GroupBlockIds             block_ids_by_group;
    const auto                      load_context =
        makeLoadContext(request_key, peer_addrs, cache_keys, block_ids_by_group, /*cp_size=*/2, /*reuse=*/3);

    const auto request = server.constructRemoteLoadRequest(load_context, /*index=*/0, peer_addrs);

    EXPECT_EQ(request.prefill_cp_size(), 2);
    EXPECT_EQ(request.partition_count(), 1);
    EXPECT_EQ(request.partition_id(), 0);
    EXPECT_EQ(request.reuse_block_size(), 3);
    ASSERT_EQ(request.peer_addrs_size(), 2);
    EXPECT_EQ(request.peer_addrs(0), "prefill-0");
    EXPECT_EQ(request.peer_addrs(1), "prefill-1");
    ASSERT_EQ(request.cache_keys_size(), 2);
    EXPECT_EQ(request.cache_keys(0), 101);
    EXPECT_EQ(request.cache_keys(1), 102);
}

TEST(DecodeRpcServerTest, CPShardedMlaLoadRequestReadsFromEveryPrefillPeer) {
    DecodeRpcServer server;
    server.resource_.workers = {"decode-0", "decode-1"};

    const std::string               request_key = "request";
    const std::vector<std::string>  peer_addrs  = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType> cache_keys  = {101};
    const GroupBlockIds             block_ids_by_group;
    const auto                      load_context =
        makeLoadContext(request_key, peer_addrs, cache_keys, block_ids_by_group, /*cp_size=*/2, /*reuse=*/3);

    const auto request = server.constructRemoteLoadRequestForMla(load_context, /*index=*/1, peer_addrs);

    EXPECT_EQ(request.prefill_cp_size(), 2);
    EXPECT_EQ(request.partition_count(), 1);
    EXPECT_EQ(request.partition_id(), 0);
    EXPECT_EQ(request.reuse_block_size(), 3);
    ASSERT_EQ(request.peer_addrs_size(), 2);
    EXPECT_EQ(request.peer_addrs(0), "prefill-0");
    EXPECT_EQ(request.peer_addrs(1), "prefill-1");
}

TEST(DecodeRpcServerTest, TaggedBlockRowsResolveByLocalTagOrder) {
    auto                   topology = CacheTopology::create({makeRpcGroup("linear", {0}), makeRpcGroup("full", {1})},
                                                            {{0, {"linear"}}, {1, {"full"}}});
    BroadcastLoadRequestPB request;
    auto*                  full = request.add_tagged_group_block_ids();
    full->set_tag("full");
    full->add_block_ids(10);
    auto* linear = request.add_tagged_group_block_ids();
    linear->set_tag("linear");
    linear->add_block_ids(20);

    const auto blocks = DecodeRpcServer::decodeGroupBlockIds(request, *topology);
    EXPECT_EQ(blocks[topology->groupIdForTag("full")]->blocks(), (BlockIndicesType{10}));
    EXPECT_EQ(blocks[topology->groupIdForTag("linear")]->blocks(), (BlockIndicesType{20}));

    auto reordered = CacheTopology::create({makeRpcGroup("full", {1}), makeRpcGroup("linear", {0})},
                                           {{0, {"linear"}}, {1, {"full"}}});
    EXPECT_NE(topology->groupIdForTag("full"), reordered->groupIdForTag("full"));
    EXPECT_EQ(DecodeRpcServer::makeTaggedRequestKey(42, 1, topology->group("full").tag),
              DecodeRpcServer::makeTaggedRequestKey(42, 1, reordered->group("full").tag));
}

TEST(DecodeRpcServerTest, EmptyTaggedBlockRowsAreRejected) {
    auto                   topology = CacheTopology::create({makeRpcGroup("full", {0})}, {{0, {"full"}}});
    BroadcastLoadRequestPB request;
    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockIds(request, *topology));
}

TEST(DecodeRpcServerTest, TaggedBlockRowsRejectTopologyMismatch) {
    auto topology =
        CacheTopology::create({makeRpcGroup("full", {0}), makeRpcGroup("linear", {0})}, {{0, {"full", "linear"}}});
    BroadcastLoadRequestPB missing_tag;
    auto*                  row = missing_tag.add_tagged_group_block_ids();
    row->set_tag("full");
    row->add_block_ids(1);

    EXPECT_ANY_THROW(DecodeRpcServer::decodeGroupBlockIds(missing_tag, *topology));
}

TEST(DecodeRpcServerTest, MtpCacheKeyUsesSharedBaseModelIdForEverySlot) {
    constexpr size_t mtp_base_model_id = 17;

    for (size_t mtp_model_id = 0; mtp_model_id < 2; ++mtp_model_id) {
        EXPECT_EQ(DecodeRpcServer::makeMTPModuleCacheKey(mtp_base_model_id, "101", /*layer_id=*/0),
                  "model_id_17_token_id_str_101_layer_id_0")
            << "mtp_model_id=" << mtp_model_id;
    }
}

TEST(DecodeRpcServerTest, MtpLoadPlanContainsOnlyModule0) {
    auto module0          = std::make_unique<EngineInitParams>();
    module0->model_id     = 17;
    auto module1          = std::make_unique<EngineInitParams>();
    module1->model_id     = 23;
    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(std::move(module0));
    mtp_model_params->push_back(std::move(module1));
    ProposeModelEngineInitParams propose_params(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));

    const auto plan = DecodeRpcServer::makeMTPModuleLoadPlan(&propose_params);

    ASSERT_EQ(plan.size(), 1);
    EXPECT_EQ(plan[0].module_index, 0);
    EXPECT_EQ(plan[0].engine_init_params, propose_params.mtp_model_params_->at(0).get());
    EXPECT_EQ(plan[0].cache_model_id, 17);
}

TEST(DecodeRpcServerTest, MtpLoadPlanRejectsMissingModule0) {
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(nullptr).empty());

    ProposeModelEngineInitParams missing_params;
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&missing_params).empty());

    auto                         empty_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    ProposeModelEngineInitParams no_modules(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(empty_params));
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&no_modules).empty());

    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(nullptr);
    mtp_model_params->push_back(std::make_unique<EngineInitParams>());
    ProposeModelEngineInitParams null_module0(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&null_module0).empty());
}

TEST(DecodeRpcServerTest, MtpLoadPlanIgnoresInactiveModules) {
    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(std::make_unique<EngineInitParams>());
    mtp_model_params->push_back(nullptr);
    ProposeModelEngineInitParams propose_params(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));

    const auto plan = DecodeRpcServer::makeMTPModuleLoadPlan(&propose_params);

    ASSERT_EQ(plan.size(), 1);
    EXPECT_EQ(plan[0].engine_init_params, propose_params.mtp_model_params_->at(0).get());
}

TEST(DecodeRpcServerTest, ReadFailureLogContainsPeerErrorAndEveryBlockKey) {
    test::TestLogCapture log_capture("read_cache_failure");
    DecodeRpcServer::logReadFailures(/*request_id=*/42,
                                     "127.0.0.1:1:2",
                                     ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED,
                                     "connect failed",
                                     {"blocks={kv_key_0,kv_key_1}"});

    const auto log_content = log_capture.content();
    EXPECT_NE(log_content.find("PD_CACHE_KEY_READ_FAILED"), std::string::npos);
    EXPECT_NE(log_content.find("127.0.0.1:1:2"), std::string::npos);
    EXPECT_NE(log_content.find("kv_key_0"), std::string::npos);
    EXPECT_NE(log_content.find("kv_key_1"), std::string::npos);
}

TEST(DecodeRpcServerTest, ReadTimeoutLogsKeysAndCancellationIsSilent) {
    test::TestLogCapture log_capture("read_cache_timeout_cancel");
    DecodeRpcServer::logReadFailures(
        /*request_id=*/43, "peer", ErrorCode::LOAD_CACHE_TIMEOUT, "timeout", {"blocks={timeout_key}"});
    DecodeRpcServer::logReadFailures(
        /*request_id=*/44, "peer", ErrorCode::CANCELLED, "cancelled", {"blocks={cancelled_key}"});

    const auto log_content = log_capture.content();
    EXPECT_NE(log_content.find("timeout_key"), std::string::npos);
    EXPECT_EQ(log_content.find("cancelled_key"), std::string::npos);
}

}  // namespace rtp_llm
