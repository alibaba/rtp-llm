#include <gtest/gtest.h>

#include <chrono>

#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm {

namespace {

DecodeRpcServer::LoadKVCacheContext makeLoadContext(const std::string&                 request_key,
                                                    const std::vector<std::string>&    peer_addrs,
                                                    const std::vector<CacheKeyType>&   cache_keys,
                                                    const GroupBlockIds&               block_ids_by_group,
                                                    int32_t                            prefill_cp_size,
                                                    int64_t                            reuse_block_size  = 0,
                                                    const std::vector<StagePeerGroup>& stage_peer_groups = {}) {
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
            prefill_cp_size,
            stage_peer_groups};
}

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
    EXPECT_EQ(broadcast->FindFieldByName("stage_peer_groups")->number(), 15);

    const auto* remote = RemoteOperationRequestPB::descriptor();
    ASSERT_NE(remote, nullptr);
    EXPECT_TRUE(remote->IsReservedNumber(3));
    EXPECT_EQ(remote->FindFieldByName("group_ids"), nullptr);
    EXPECT_EQ(remote->FindFieldByName("block_ids")->number(), 4);
    EXPECT_EQ(remote->FindFieldByName("uris")->number(), 5);
    EXPECT_EQ(remote->FindFieldByName("group_tags")->number(), 6);
}

TEST(ModelRpcProtoTest, GenerateRequestCarriesPpTopologyFields) {
    const auto* request = GenerateRequestPB::descriptor();
    ASSERT_NE(request, nullptr);
    EXPECT_EQ(request->FindFieldByName("peer_addrs")->number(), 7);
    ASSERT_NE(request->FindFieldByName("stage_peer_groups"), nullptr);
    EXPECT_EQ(request->FindFieldByName("stage_peer_groups")->number(), 12);

    const auto* group = StagePeerGroupPB::descriptor();
    ASSERT_NE(group, nullptr);
    EXPECT_EQ(group->FindFieldByName("layer_begin")->number(), 1);
    EXPECT_EQ(group->FindFieldByName("layer_count")->number(), 2);
    EXPECT_EQ(group->FindFieldByName("peer_addrs")->number(), 3);
}

TEST(DecodeRpcServerTest, PpLoadRequestCarriesStagePeerGroups) {
    DecodeRpcServer server;
    server.resource_.workers                            = {"decode-0", "decode-1", "decode-2", "decode-3"};
    server.maga_init_params_.parallelism_config.tp_size = 2;

    const std::string                 request_key = "request";
    const std::vector<std::string>    peer_addrs  = {"prefill-0", "prefill-1", "prefill-2", "prefill-3"};
    const std::vector<CacheKeyType>   cache_keys  = {101};
    const GroupBlockIds               block_ids_by_group;
    const std::vector<StagePeerGroup> groups = {{{0, 2}, {"prefill-0", "prefill-1"}, false},
                                                {{2, 2}, {"prefill-2", "prefill-3"}, true}};
    const auto                        load_context =
        makeLoadContext(request_key, peer_addrs, cache_keys, block_ids_by_group, /*cp_size=*/1, /*reuse=*/0, groups);

    const auto request = server.constructRemoteLoadRequest(load_context, /*index=*/3, peer_addrs);

    // PP routing ignores the flat fields and ships the stage groups instead.
    EXPECT_EQ(request.partition_count(), 1);
    EXPECT_EQ(request.partition_id(), 0);
    EXPECT_EQ(request.peer_addrs_size(), 0);
    ASSERT_EQ(request.stage_peer_groups_size(), 2);
    EXPECT_EQ(request.stage_peer_groups(0).layer_begin(), 0);
    EXPECT_EQ(request.stage_peer_groups(0).layer_count(), 2);
    ASSERT_EQ(request.stage_peer_groups(0).peer_addrs_size(), 2);
    EXPECT_EQ(request.stage_peer_groups(0).peer_addrs(1), "prefill-1");
    EXPECT_EQ(request.stage_peer_groups(1).layer_begin(), 2);
    EXPECT_EQ(request.stage_peer_groups(1).peer_addrs(0), "prefill-2");
    EXPECT_FALSE(request.stage_peer_groups(0).is_last_stage());
    EXPECT_TRUE(request.stage_peer_groups(1).is_last_stage());
}

TEST(DecodeRpcServerTest, PpMlaLoadRequestCarriesStagePeerGroups) {
    DecodeRpcServer server;
    server.resource_.workers                            = {"decode-0", "decode-1"};
    server.maga_init_params_.parallelism_config.tp_size = 1;

    const std::string                 request_key = "request";
    const std::vector<std::string>    peer_addrs  = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType>   cache_keys  = {101};
    const GroupBlockIds               block_ids_by_group;
    const std::vector<StagePeerGroup> groups = {{{0, 2}, {"prefill-0"}}, {{2, 2}, {"prefill-1"}}};
    const auto                        load_context =
        makeLoadContext(request_key, peer_addrs, cache_keys, block_ids_by_group, /*cp_size=*/1, /*reuse=*/0, groups);

    const auto request = server.constructRemoteLoadRequestForMla(load_context, /*index=*/0, peer_addrs);

    EXPECT_EQ(request.peer_addrs_size(), 0);
    ASSERT_EQ(request.stage_peer_groups_size(), 2);
    EXPECT_EQ(request.stage_peer_groups(1).peer_addrs(0), "prefill-1");
}

TEST(DecodeRpcServerTest, FlatLoadRequestMapsDecodeLaneToPrefillPeer) {
    DecodeRpcServer server;
    server.resource_.workers                            = {"decode-0", "decode-1", "decode-2", "decode-3"};
    server.maga_init_params_.parallelism_config.tp_size = 2;

    const std::string               request_key = "request";
    const std::vector<std::string>  peer_addrs  = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType> cache_keys  = {101};
    const GroupBlockIds             block_ids_by_group;
    const auto load_context = makeLoadContext(request_key, peer_addrs, cache_keys, block_ids_by_group, /*cp_size=*/1);

    // decode stage 1 lane 1 reads its whole block from the same-lane peer.
    const auto request = server.constructRemoteLoadRequest(load_context, /*index=*/3, peer_addrs);
    ASSERT_EQ(request.peer_addrs_size(), 1);
    EXPECT_EQ(request.peer_addrs(0), "prefill-1");
    EXPECT_EQ(request.partition_count(), 1);
    EXPECT_EQ(request.partition_id(), 0);
    EXPECT_EQ(request.stage_peer_groups_size(), 0);
}

TEST(DecodeRpcServerTest, FlatLoadRequestSlicesSourceForFinerDecodeTp) {
    DecodeRpcServer server;
    server.resource_.workers                            = {"decode-0", "decode-1", "decode-2", "decode-3"};
    server.maga_init_params_.parallelism_config.tp_size = 2;

    const std::string               request_key = "request";
    const std::vector<std::string>  peer_addrs  = {"prefill-0"};
    const std::vector<CacheKeyType> cache_keys  = {101};
    const GroupBlockIds             block_ids_by_group;
    const auto load_context = makeLoadContext(request_key, peer_addrs, cache_keys, block_ids_by_group, /*cp_size=*/1);

    // decode stage 1 lane 1 reads slice 1 of 2 from the single prefill peer.
    const auto request = server.constructRemoteLoadRequest(load_context, /*index=*/3, peer_addrs);
    ASSERT_EQ(request.peer_addrs_size(), 1);
    EXPECT_EQ(request.peer_addrs(0), "prefill-0");
    EXPECT_EQ(request.partition_count(), 2);
    EXPECT_EQ(request.partition_id(), 1);
}

TEST(DecodeRpcServerTest, MlaFlatLoadRequestMapsDecodeLaneToPrefillPeer) {
    DecodeRpcServer server;
    server.resource_.workers                            = {"decode-0", "decode-1", "decode-2", "decode-3"};
    server.maga_init_params_.parallelism_config.tp_size = 2;

    const std::string               request_key = "request";
    const std::vector<std::string>  peer_addrs  = {"prefill-0", "prefill-1"};
    const std::vector<CacheKeyType> cache_keys  = {101};
    const GroupBlockIds             block_ids_by_group;
    const auto load_context = makeLoadContext(request_key, peer_addrs, cache_keys, block_ids_by_group, /*cp_size=*/1);

    const auto request = server.constructRemoteLoadRequestForMla(load_context, /*index=*/2, peer_addrs);
    ASSERT_EQ(request.peer_addrs_size(), 1);
    EXPECT_EQ(request.peer_addrs(0), "prefill-0");
    EXPECT_EQ(request.partition_count(), 1);
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

class HandoffProcessor: public BaseLogitsProcessor {
public:
    explicit HandoffProcessor(GenerateStream& stream): stream_(stream) {}

    std::optional<ErrorInfo> process(const SamplerInputs&, size_t, size_t) override {
        return std::nullopt;
    }
    void updateMultiSeqStatus(const std::vector<int>&) override {}
    std::optional<ErrorInfo> updateStatus(const torch::Tensor& tokens, int32_t count) override {
        ++update_calls;
        had_sp_buffer_at_update = stream_.getSPOutputBuffer() != nullptr;
        committed_tokens.insert(committed_tokens.end(), tokens.data_ptr<int32_t>(), tokens.data_ptr<int32_t>() + count);
        if (fail_update) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "handoff processor update failed");
        }
        return std::nullopt;
    }
    std::optional<int64_t> committedOutputLen() const override {
        return committed_tokens.size();
    }

    int              update_calls = 0;
    bool             fail_update  = false;
    bool             had_sp_buffer_at_update = false;
    std::vector<int> committed_tokens;

private:
    GenerateStream& stream_;
};

class HandoffEngine: public EngineBase {
public:
    HandoffEngine(): EngineBase(EngineInitParams()) {}
    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }
    void enqueue(std::shared_ptr<GenerateStream>& stream) override {
        ++enqueue_count;
        /** Drive the scheduler transition, then complete the request without a model step. */
        stream->moveToNext();
        stream->reportEvent(StreamEvents::GenerateDone);
        stream->moveToNext();
    }
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("not used by DecodeRpcBootstrapTest");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }

    bool isDSpark() override {
        return is_dspark;
    }

    int  enqueue_count = 0;
    bool is_dspark     = false;
};

class HandoffRpcService: public RpcService::Service {
public:
    explicit HandoffRpcService(DecodeRpcServer& server): server_(server) {}

    grpc::Status RemoteGenerate(grpc::ServerContext* context, ServerStream* writer) override {
        DecodeRpcContext rpc_context{writer};
        kmonitor::MetricsReporterPtr reporter;
        DecodeGenerateContext decode_context(rpc_context, 5000, context, reporter, server_.meta_);
        decode_context.request_id = stream->streamId();
        decode_context.request_key = std::to_string(stream->streamId());
        decode_context.time_info = {};
        decode_context.setStream(stream);
        try {
            server_.localGenerate(decode_context);
        } catch (const std::exception& error) {
            decode_context.error_status = grpc::Status(grpc::StatusCode::INTERNAL, error.what());
        }
        return decode_context.error_status;
    }

    GenerateStreamPtr stream;

private:
    DecodeRpcServer& server_;
};

class DecodeRpcBootstrapTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        engine_ = std::make_shared<HandoffEngine>();
        rpc_.engine_ = engine_;
        rpc_.meta_ = std::make_shared<RpcServerRuntimeMeta>();
        rpc_.maga_init_params_.sp_config.type = SP_TYPE_MTP;
        rpc_.maga_init_params_.sp_config.gen_num_per_cycle = 3;
        rpc_.maga_init_params_.parallelism_config.pp_size = 2;
        cache_manager_ = std::make_shared<KVCacheManager>(makeMhaCacheConfig(1, 16, 1, 8, 4, DataType::TYPE_FP16));
        ASSERT_TRUE(cache_manager_->init());
        service_ = std::make_unique<HandoffRpcService>(rpc_);
        grpc::ServerBuilder builder;
        int port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(service_.get());
        server_ = builder.BuildAndStart();
        ASSERT_NE(server_, nullptr);
        ASSERT_NE(port, 0);
        stub_ = RpcService::NewStub(grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                                                       grpc::InsecureChannelCredentials()));
    }

    void TearDown() override {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
        }
        DeviceTestBase::TearDown();
    }

    std::shared_ptr<NormalGenerateStream> makeStream() {
        ModelConfig model;
        model.max_seq_len = 64;
        model.vocab_size = 128;
        model.input_vocab_size = 128;
        model.num_layers = 1;
        model.attn_config.tokens_per_block = 4;
        model.special_tokens.eos_token_id = -1;
        auto input = std::make_shared<GenerateInput>();
        input->request_id = 42;
        input->begin_time_us = currentTimeUs();
        input->input_ids = torch::tensor({1, 2, 3}, torch::kInt32);
        input->generate_config = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens = 16;
        input->generate_config->is_streaming = true;
        ResourceContext resources;
        resources.role_type = RoleType::DECODE;
        resources.cache_manager = cache_manager_;
        auto stream = std::make_shared<NormalGenerateStream>(input, model, RuntimeConfig{}, resources, nullptr);
        processor_ = std::make_shared<HandoffProcessor>(*stream);
        stream->sampling_state_.logits_processors = {processor_};
        EXPECT_TRUE(stream->stream_cache_resource_->initKVBlock().ok());
        EXPECT_GT(stream->curBlocksNum(), 0);
        stream->reportEvent(StreamEvents::LoadInitiated);
        stream->reportEvent(StreamEvents::CanRun);
        EXPECT_EQ(stream->getStatus(), StreamState::WAITING);
        return stream;
    }

    GenerateRequestPB makeRequest(bool remote_payload = true) {
        GenerateRequestPB request;
        request.set_stage(RemoteStage::GENERATE);
        request.set_first_generate_token_id(7);
        request.add_position_ids(2);
        request.add_position_ids(12);
        if (remote_payload) {
            const bool pp_mtp = rpc_.maga_init_params_.parallelism_config.pp_size > 1
                                && rpc_.maga_init_params_.sp_config.type != SP_TYPE_NONE && !engine_->isDSpark();
            const auto tokens = pp_mtp ? std::vector<int>{7, 8} : std::vector<int>{7, 8, 9, 10};
            for (int token : tokens) {
                request.add_propose_token_ids(token);
            }
            QueryConverter::transTensorPB(request.mutable_propose_probs(), torch::full({1, 1, 128}, 1.0f / 128));
            QueryConverter::transTensorPB(request.mutable_propose_hidden(), torch::full({1, 8}, 0.5f));
        }
        return request;
    }

    grpc::Status runHandoff(const GenerateStreamPtr& stream, const GenerateRequestPB& request) {
        service_->stream = stream;
        engine_->enqueue_count = 0;
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        auto call = stub_->RemoteGenerate(&context);
        EXPECT_TRUE(call->Write(request));
        EXPECT_TRUE(call->WritesDone());
        GenerateOutputsPB response;
        int output_count = 0;
        while (call->Read(&response)) {
            ++output_count;
        }
        const auto status = call->Finish();
        EXPECT_EQ(output_count, 0) << "D must not emit the anchor a second time";
        EXPECT_TRUE(rpc_.meta_->getEngineScheduleInfo(0).running_task_info_list.empty());
        EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
        EXPECT_TRUE(stream->stream_cache_resource_->isResourceReleased());
        EXPECT_EQ(cache_manager_->freeBlocksNum(), 15);
        return status;
    }

    DecodeRpcServer rpc_;
    std::shared_ptr<HandoffEngine> engine_;
    std::shared_ptr<KVCacheManager> cache_manager_;
    std::shared_ptr<HandoffProcessor> processor_;
    std::unique_ptr<HandoffRpcService> service_;
    std::unique_ptr<grpc::Server> server_;
    std::unique_ptr<RpcService::Stub> stub_;
};

TEST_F(DecodeRpcBootstrapTest, PpMtpAndEaglePreserveD1AndPadCandidates) {
    for (auto type : {SP_TYPE_MTP, SP_TYPE_EAGLE}) {
        SCOPED_TRACE(type);
        rpc_.maga_init_params_.sp_config.type = type;
        for (int64_t count : {1, 3, 4}) {
            rpc_.maga_init_params_.sp_config.gen_num_per_cycle = count;
            for (bool tensor_payload : {false, true}) {
                SCOPED_TRACE("K=" + std::to_string(count) + ", tensor_payload=" + std::to_string(tensor_payload));
                auto stream = makeStream();
                auto request = makeRequest(false);
                request.add_propose_token_ids(7);
                request.add_propose_token_ids(8);
                if (tensor_payload) {
                    /** Invalid tensor encodings must never reach QueryConverter in the PP MTP branch. */
                    request.mutable_propose_hidden();
                    request.mutable_propose_probs();
                }
                ASSERT_TRUE(runHandoff(stream, request).ok());
                EXPECT_EQ(engine_->enqueue_count, 1);
                EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2, 3, 7}));
                EXPECT_EQ(stream->last_output_pos_, 4);
                EXPECT_EQ(processor_->update_calls, 1);
                EXPECT_FALSE(processor_->had_sp_buffer_at_update);
                EXPECT_EQ(processor_->committed_tokens, (std::vector<int>{7}));
                EXPECT_EQ(stream->reuseLength(), 3);
                EXPECT_EQ(stream->getMtpTokenIndex(), 3);
                EXPECT_FALSE(stream->isContextStream());
                const auto buffer = stream->getSPOutputBuffer();
                ASSERT_NE(buffer, nullptr);
                EXPECT_EQ(buffer->propose_step, count);
                EXPECT_EQ(buffer->tokens.sizes().vec(), (std::vector<int64_t>{1, count + 1}));
                EXPECT_EQ(buffer->tokens.scalar_type(), torch::kInt32);
                EXPECT_TRUE(buffer->tokens.is_pinned());
                std::vector<int> expected_tokens(count + 1, 0);
                expected_tokens[0] = 7;
                expected_tokens[1] = 8;
                EXPECT_TRUE(torch::equal(buffer->tokens,
                                         torch::tensor(expected_tokens, torch::kInt32).reshape({1, count + 1})));
                EXPECT_FALSE(buffer->hidden_states.defined());
                EXPECT_FALSE(buffer->all_probs.defined());
                EXPECT_EQ(stream->getProposeToken(), expected_tokens);
                EXPECT_EQ(stream->getMtpAsyncDeviceState().next_real_seq_len, -1);
                EXPECT_TRUE(torch::equal(stream->getContextPositionIds(), torch::tensor({2, 12}, torch::kInt32)));
            }
        }
    }
}

TEST_F(DecodeRpcBootstrapTest, NonPipelineHandoffRestoresRemoteDraftState) {
    rpc_.maga_init_params_.parallelism_config.pp_size = 1;
    for (auto type : {SP_TYPE_MTP, SP_TYPE_EAGLE}) {
        rpc_.maga_init_params_.sp_config.type = type;
        for (const std::string flag : {"0", "1"}) {
            autil::EnvGuard stream_async("RTP_LLM_STREAM_ASYNC", flag);
            autil::EnvGuard mtp_async("RTP_LLM_MTP_ASYNC_DEVICE_STATE", flag);
            auto stream = makeStream();
            auto request = makeRequest();
            ASSERT_TRUE(runHandoff(stream, request).ok());
            EXPECT_EQ(engine_->enqueue_count, 1);
            EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2, 3, 7}));
            EXPECT_EQ(processor_->update_calls, 1);
            EXPECT_FALSE(processor_->had_sp_buffer_at_update);
            EXPECT_EQ(processor_->committed_tokens, (std::vector<int>{7}));
            EXPECT_EQ(stream->reuseLength(), 3);
            EXPECT_EQ(stream->getMtpTokenIndex(), 3);
            EXPECT_TRUE(torch::equal(stream->getContextPositionIds(), torch::tensor({2, 12}, torch::kInt32)));
            const auto buffer = stream->getSPOutputBuffer();
            ASSERT_NE(buffer, nullptr);
            EXPECT_TRUE(torch::equal(buffer->tokens, torch::tensor({{7, 8, 9, 10}}, torch::kInt32)));
            EXPECT_TRUE(buffer->hidden_states.is_cuda());
            EXPECT_TRUE(buffer->all_probs.is_cuda());
            EXPECT_TRUE(torch::equal(buffer->hidden_states.cpu(), QueryConverter::transTensor(request.propose_hidden())));
            EXPECT_TRUE(torch::equal(buffer->all_probs.cpu(), QueryConverter::transTensor(request.propose_probs())));
            const auto& state = stream->getMtpAsyncDeviceState();
            EXPECT_EQ(state.next_real_seq_len, flag == "1" ? 4 : -1);
            EXPECT_EQ(state.next_seq_len_gpu.defined(), flag == "1");
        }
    }
}

TEST_F(DecodeRpcBootstrapTest, NormalHandoffUsesTheSameAnchorCommit) {
    rpc_.maga_init_params_.sp_config.type = SP_TYPE_NONE;
    for (int pp_size : {1, 2}) {
        rpc_.maga_init_params_.parallelism_config.pp_size = pp_size;
        auto stream = makeStream();
        ASSERT_TRUE(runHandoff(stream, makeRequest(false)).ok());
        EXPECT_EQ(engine_->enqueue_count, 1);
        EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2, 3, 7}));
        EXPECT_EQ(stream->last_output_pos_, 4);
        EXPECT_EQ(processor_->update_calls, 1);
        EXPECT_EQ(processor_->committed_tokens, (std::vector<int>{7}));
        EXPECT_EQ(stream->getSPOutputBuffer(), nullptr);
    }
}

TEST_F(DecodeRpcBootstrapTest, NonPipelineDSparkKeepsItsProposalContract) {
    rpc_.maga_init_params_.sp_config.type = SP_TYPE_DSPARK;
    rpc_.maga_init_params_.parallelism_config.pp_size = 1;
    engine_->is_dspark = true;
    autil::EnvGuard stream_async("RTP_LLM_STREAM_ASYNC", "1");
    autil::EnvGuard mtp_async("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "1");
    auto stream = makeStream();
    ASSERT_TRUE(runHandoff(stream, makeRequest(false)).ok());
    EXPECT_EQ(engine_->enqueue_count, 1);
    EXPECT_EQ(processor_->update_calls, 1);
    EXPECT_EQ(stream->getMtpAsyncDeviceState().next_real_seq_len, -1);
    EXPECT_EQ(stream->getSPOutputBuffer(), nullptr);
    EXPECT_TRUE(stream->getProposeToken().empty());
}

TEST_F(DecodeRpcBootstrapTest, AsyncFlagsNeverPublishUnsupportedPpState) {
    for (const std::string flag : {"0", "1"}) {
        autil::EnvGuard stream_async("RTP_LLM_STREAM_ASYNC", flag);
        autil::EnvGuard mtp_async("RTP_LLM_MTP_ASYNC_DEVICE_STATE", flag);
        auto stream = makeStream();
        ASSERT_TRUE(runHandoff(stream, makeRequest()).ok());
        EXPECT_EQ(stream->getMtpAsyncDeviceState().epoch, 0);
        EXPECT_EQ(stream->getMtpAsyncDeviceState().next_real_seq_len, -1);
        EXPECT_FALSE(stream->getMtpAsyncDeviceState().propose_tokens_gpu.defined());
        EXPECT_FALSE(stream->getMtpAsyncDeviceState().last_hidden_states_gpu.defined());
    }
}

TEST_F(DecodeRpcBootstrapTest, AnchorErrorsFollowTheCommonStreamErrorPath) {
    for (int pp_size : {1, 2}) {
        rpc_.maga_init_params_.parallelism_config.pp_size = pp_size;
        for (int anchor : {-1, 128}) {
            auto stream = makeStream();
            auto request = makeRequest();
            request.set_first_generate_token_id(anchor);
            EXPECT_FALSE(runHandoff(stream, request).ok());
            EXPECT_EQ(stream->statusInfo().code(), ErrorCode::OUT_OF_VOCAB_RANGE);
            EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{1, 2, 3}));
            EXPECT_EQ(processor_->update_calls, 0);
            EXPECT_EQ(engine_->enqueue_count, 1);
        }
        auto stream = makeStream();
        processor_->fail_update = true;
        EXPECT_FALSE(runHandoff(stream, makeRequest()).ok());
        EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
        EXPECT_EQ(processor_->update_calls, 1);
        EXPECT_EQ(engine_->enqueue_count, 1);
    }
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
